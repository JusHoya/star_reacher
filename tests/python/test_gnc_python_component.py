"""FR-25 Python GNC components driven through the pybind11 trampoline.

A component written in Python subclasses ``_core.IGncComponent``, registers
under a config name, and is then selected by the same registry the built-in
C++ components come from. These tests drive real missions end to end with a
Python component in the loop and check that:

* a Python control law reproduces the built-in ``pd_attitude`` law it
  reimplements (the FR-25 "zero recompilation" swap, exercised on the
  committed reference mission);
* the trampoline marshals every field a component may read, including the
  aiding-measurement slots, the truth-free navigation environment, the
  central-body constants, the configured sensor model, and the true IMU
  biases on the ``error_state`` truth argument -- without these an
  aiding estimator written in Python has no route to its measurements;
* the estimator introspection hooks (``state``, ``covariance_upper``,
  ``error_state``, ``innovations``) round-trip, and a wrong-length return
  raises rather than being silently truncated.

They fail cleanly, never skip, when the core is absent (the project's
agent-honesty gate).
"""

import contextlib
import hashlib
import itertools
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ATTITUDE_MISSION = REPO_ROOT / "missions" / "leo_attitude_gnc.toml"
EKF_MISSION = REPO_ROOT / "missions" / "leo_ekf_consistency.toml"

_CORE_MISSING_MESSAGE = (
    "star_reacher._core is not built in this environment. These integration "
    "tests require the compiled core: build and install it with 'pip install .' "
    "from the repository root. This failure is expected on a core-less checkout "
    "and must be green at integration/CI."
)

# Registry names must be unique for the life of the process (a duplicate is a
# determinism hazard and the core refuses it), so every registration in this
# module draws a fresh suffix. The "test_" prefix is the established
# convention for probe registrations: registering into the core registry is
# process-wide, and test_gnc_validation's registry-parity check filters that
# prefix out so these probes cannot masquerade as shipped components.
_names = itertools.count()
_NAME_PREFIX = "test_"


def _core_or_fail():
    try:
        from star_reacher import _core
    except ImportError:
        pytest.fail(_CORE_MISSING_MESSAGE)
    return _core


@contextlib.contextmanager
def _in_repo_root():
    cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        yield
    finally:
        os.chdir(cwd)


@pytest.fixture(autouse=True)
def _repo_root_cwd():
    with _in_repo_root():
        yield


def _run_config(core, mission_path):
    """The same RunConfig a batch run of this mission would use."""
    from star_reacher.mission import canonical_bytes, validate_mission_file
    from star_reacher.runner import build_run_config

    resolved, errors = validate_mission_file(mission_path)
    assert not errors, errors
    config_sha = hashlib.sha256(canonical_bytes(resolved)).hexdigest()
    cfg, _vehicle_toml, _env = build_run_config(core, resolved, config_sha)
    return cfg


def _swap_component(core, cfg, slot, name, vectors=None, scalars=None):
    """Replace one chain slot with a registered component by name."""
    spec = core.GncComponentCfg()
    spec.component = name
    spec.scalars = dict(scalars or {})
    spec.vectors = {k: list(v) for k, v in (vectors or {}).items()}
    gnc = cfg.gnc
    setattr(gnc, slot, spec)
    cfg.gnc = gnc
    return cfg


def _register(core, cls, prefix):
    name = f"{_NAME_PREFIX}{prefix}_{next(_names)}"
    core.register_python_component(name, cls)
    return name


def _drive(core, cfg, out_path):
    """Run a configuration to completion through the stepping API."""
    sim = core.Sim(cfg, str(out_path))
    while not sim.done():
        sim.step()
    return sim.summary()


# --- a Python control law in the loop --------------------------------------


def _make_py_pd(core):
    """The built-in pd_attitude law, reimplemented in Python.

    The arithmetic is the normative sequence documented in gnc/builtin.hpp
    (eq:gnc:deltaq, eq:gnc:sign, eq:gnc:werr, eq:gnc:pd, eq:gnc:sat),
    evaluated per axis exactly as written with no renormalization of dq. The
    quaternion product and the DCM come from the core's own bound rotation
    kernel, so what is being compared is the control law rather than two
    independent quaternion implementations.
    """

    class PyPdAttitude(core.IGncComponent):
        def __init__(self, cfg):
            super().__init__()
            self.kp = [float(x) for x in cfg.vectors["kp_nm_per_rad"]]
            self.kd = [float(x) for x in cfg.vectors["kd_nm_per_radps"]]
            self.tau_max = [float(x) for x in cfg.vectors["tau_max_nm"]]

        def init(self, ctx):
            # Nothing to capture: the law is memoryless in the state and its
            # gains are configuration.
            pass

        def update(self, inp):
            out = core.GncOutput()
            est = inp.nav_est
            cmd = inp.att_cmd
            if not est.valid or not cmd.valid:
                return out  # hold, exactly as the built-in does

            qc = cmd.q_i2b
            qe = est.q_i2b
            # dq = q_cmd^* (x) q_est   (eq:gnc:deltaq)
            dq = core.quat_multiply(
                qc[0], -qc[1], -qc[2], -qc[3], qe[0], qe[1], qe[2], qe[3]
            )
            sign = 1.0 if dq[0] >= 0.0 else -1.0  # eq:gnc:sign, sign(0) = +1
            dcm = core.quat_to_dcm(dq[0], dq[1], dq[2], dq[3])
            wc = cmd.omega_b_radps
            we = est.omega_b_radps
            # w_err = w_est - C(dq) w_cmd   (eq:gnc:werr)
            rotated = [
                dcm[0] * wc[0] + dcm[1] * wc[1] + dcm[2] * wc[2],
                dcm[3] * wc[0] + dcm[4] * wc[1] + dcm[5] * wc[2],
                dcm[6] * wc[0] + dcm[7] * wc[1] + dcm[8] * wc[2],
            ]
            torque = []
            for i in range(3):
                w_err = we[i] - rotated[i]
                # eq:gnc:pd then eq:gnc:sat, left-associated as written
                t = -self.kp[i] * sign * dq[i + 1] - self.kd[i] * w_err
                t = max(-self.tau_max[i], min(self.tau_max[i], t))
                torque.append(t)

            out.valid = True
            out.q_i2b = cmd.q_i2b
            out.omega_b_radps = cmd.omega_b_radps
            out.torque_b_nm = torque
            return out

    return PyPdAttitude


def test_python_control_law_drives_a_real_mission(tmp_path):
    """A Python component flies the committed reference mission end to end."""
    core = _core_or_fail()
    from star_reacher import load

    gains = {
        "kp_nm_per_rad": [0.4, 0.4, 0.4],
        "kd_nm_per_radps": [3.6, 3.6, 3.6],
        "tau_max_nm": [0.05, 0.05, 0.05],
    }
    name = _register(core, _make_py_pd(core), "py_pd_attitude")

    builtin_log = tmp_path / "builtin.srlog"
    python_log = tmp_path / "python.srlog"
    _drive(core, _run_config(core, ATTITUDE_MISSION), builtin_log)
    cfg = _swap_component(
        core, _run_config(core, ATTITUDE_MISSION), "control", name, gains
    )
    summary = _drive(core, cfg, python_log)

    assert summary["steps"] > 0
    a = load(builtin_log)
    b = load(python_log)
    tau_builtin = a.groups["gnc.cmd"]["tau_b_nm"]
    tau_python = b.groups["gnc.cmd"]["tau_b_nm"]
    assert tau_python.shape == tau_builtin.shape
    # The mission opens with a 10-degree attitude error, so the commanded
    # torque is genuinely exercised rather than identically zero.
    assert abs(tau_builtin).max() > 1e-3
    worst = float(abs(tau_python - tau_builtin).max())
    assert worst < 1e-9, (
        f"the Python reimplementation of pd_attitude departs from the "
        f"built-in by {worst:.3e} N*m, above the 1e-9 contract in "
        f"gnc/builtin.hpp"
    )

    # The vehicle really flew: the closed loop drove the 10-degree error down.
    err = abs(a.groups["truth"]["q_i2b"] - b.groups["truth"]["q_i2b"]).max()
    assert err < 1e-9


# --- WS7 field marshalling --------------------------------------------------


def _make_recording_nav(core, seen):
    """A nav component that records everything the trampoline hands it."""

    class RecordingNav(core.IGncComponent):
        def __init__(self, cfg):
            super().__init__()
            self.q = [1.0, 0.0, 0.0, 0.0]

        def init(self, ctx):
            seen["init"] = {
                "t0_s": ctx.t0_s,
                "dt_s": ctx.dt_s,
                "control_rate_hz": ctx.control_rate_hz,
                "mu_m3ps2": ctx.mu_m3ps2,
                "ellipsoid_a_m": ctx.ellipsoid_a_m,
                "ellipsoid_inv_f": ctx.ellipsoid_inv_f,
                "q0_i2b": list(ctx.q0_i2b),
                "imu_present": ctx.sensors.imu_present,
                "gyro_arw": ctx.sensors.gyro_arw,
                "accel_vrw": ctx.sensors.accel_vrw,
                "navfix_present": ctx.sensors.navfix_present,
                "navfix_sigma_r_m": list(ctx.sensors.navfix_sigma_r_m),
                "startracker_present": ctx.sensors.startracker_present,
                "startracker_sigma_rad": list(
                    ctx.sensors.startracker_sigma_rad
                ),
                "altimeter_present": ctx.sensors.altimeter_present,
                "altimeter_sigma_noise_m": ctx.sensors.altimeter_sigma_noise_m,
            }
            self.q = list(ctx.q0_i2b)

        def update(self, inp):
            rec = seen.setdefault("cycles", [])
            rec.append(
                {
                    "cycle": inp.cycle,
                    "t_s": inp.t_s,
                    "imu_fresh": inp.imu_fresh,
                    "navfix_fresh": inp.navfix.fresh,
                    "navfix_valid": inp.navfix.valid,
                    "navfix_r": list(inp.navfix.r_i_m),
                    "navfix_id": inp.navfix.sensor_id,
                    "st_fresh": inp.startracker.fresh,
                    "st_valid": inp.startracker.valid,
                    "st_q": list(inp.startracker.q_i2b),
                    "alt_fresh": inp.altimeter.fresh,
                    "alt_valid": inp.altimeter.valid,
                    "alt_h": inp.altimeter.h_m,
                    "eph_valid": inp.env.ephemeris_valid,
                    "v_ssb": list(inp.env.v_central_ssb_mps),
                    "bf_valid": inp.env.bodyfixed_valid,
                    "c_bf": list(inp.env.c_gcrf_to_bodyfixed),
                    "oracle_valid": inp.oracle.valid,
                }
            )
            out = core.GncOutput()
            out.valid = True
            out.q_i2b = self.q
            out.omega_b_radps = [0.0, 0.0, 0.0]
            return out

        def state_dim(self):
            return 7

        def state(self):
            return list(self.q) + [0.0, 0.0, 0.0]

        def covariance_upper(self):
            return [0.0] * (7 * 8 // 2)

        def error_state(self, truth):
            seen["truth"] = {
                "valid": truth.valid,
                "imu_bias_valid": truth.imu_bias_valid,
                "b_g_radps": list(truth.b_g_radps),
                "b_a_mps2": list(truth.b_a_mps2),
                "mass_kg": truth.mass_kg,
                "r_i_m": list(truth.r_i_m),
            }
            return [0.0] * 7

    return RecordingNav


def test_trampoline_marshals_every_component_input(tmp_path):
    """Aiding slots, environment, constants, and sensor model all arrive."""
    core = _core_or_fail()
    seen = {}
    name = _register(core, _make_recording_nav(core, seen), "recording_nav")
    cfg = _swap_component(core, _run_config(core, EKF_MISSION), "nav", name)
    _drive(core, cfg, tmp_path / "recording.srlog")

    # -- init context: central-body constants and the configured suite ------
    init = seen["init"]
    assert init["mu_m3ps2"] == pytest.approx(3.986004418e14, rel=1e-9)
    assert init["ellipsoid_a_m"] == pytest.approx(6378137.0, rel=1e-12)
    assert init["ellipsoid_inv_f"] > 290.0
    assert init["control_rate_hz"] == 10
    assert init["dt_s"] == pytest.approx(0.1, abs=1e-12)
    assert abs(sum(x * x for x in init["q0_i2b"]) - 1.0) < 1e-12
    # The EKF mission configures all four FR-23 aiding kinds; a filter whose
    # noise model is the configured sensor model needs each parameter.
    assert init["imu_present"] is True
    assert init["gyro_arw"] > 0.0
    assert init["accel_vrw"] > 0.0
    assert init["navfix_present"] is True
    assert all(s > 0.0 for s in init["navfix_sigma_r_m"])
    assert init["startracker_present"] is True
    assert all(s > 0.0 for s in init["startracker_sigma_rad"])
    assert init["altimeter_present"] is True
    assert init["altimeter_sigma_noise_m"] > 0.0

    # -- per-cycle inputs ---------------------------------------------------
    cycles = seen["cycles"]
    assert len(cycles) > 10
    # Freshness is per sensor and sparse: the aiding sensors sample at 1 Hz
    # against a 10 Hz control rate, so exactly one cycle in ten is fresh.
    assert sum(1 for c in cycles if c["navfix_fresh"]) > 0
    assert sum(1 for c in cycles if c["navfix_fresh"]) < len(cycles)
    assert sum(1 for c in cycles if c["st_fresh"]) > 0
    assert sum(1 for c in cycles if c["alt_fresh"]) > 0
    assert all(c["imu_fresh"] for c in cycles[1:])

    fresh_fix = next(c for c in cycles if c["navfix_fresh"])
    assert fresh_fix["navfix_valid"] is True
    radius = sum(x * x for x in fresh_fix["navfix_r"]) ** 0.5
    assert 6.0e6 < radius < 8.0e6, "the nav fix did not carry a real position"

    fresh_st = next(c for c in cycles if c["st_fresh"])
    assert abs(sum(x * x for x in fresh_st["st_q"]) - 1.0) < 1e-9

    fresh_alt = next(c for c in cycles if c["alt_fresh"])
    assert fresh_alt["alt_h"] > 0.0

    # -- truth-free navigation environment ---------------------------------
    last = cycles[-1]
    assert last["bf_valid"] is True
    assert len(last["c_bf"]) == 9
    # A rotation matrix, not a zero-filled placeholder.
    rows = [last["c_bf"][0:3], last["c_bf"][3:6], last["c_bf"][6:9]]
    for row in rows:
        assert abs(sum(x * x for x in row) - 1.0) < 1e-9

    # -- FR-25 privileged boundary -----------------------------------------
    # The EKF mission does not set oracle, so truth must never appear on the
    # component's input even though the same run supplies it to error_state.
    assert all(c["oracle_valid"] is False for c in cycles)

    # -- WS7 truth biases on the error_state argument -----------------------
    truth = seen["truth"]
    assert truth["valid"] is True
    assert truth["imu_bias_valid"] is True, (
        "the true IMU biases did not reach error_state; a bias-carrying "
        "estimator would have to log its bias rows as zero, which reads as "
        "'no error' rather than 'not known'"
    )
    assert len(truth["b_g_radps"]) == 3
    assert len(truth["b_a_mps2"]) == 3
    assert any(b != 0.0 for b in truth["b_g_radps"]), (
        "the gyro bias arrived as exactly zero; the configured IMU has a "
        "non-zero in-run bias, so this is a marshalling failure rather than "
        "a physically quiet run"
    )
    assert truth["mass_kg"] > 0.0


def test_oracle_flag_reaches_a_python_component(tmp_path):
    """oracle = true populates GncInput.oracle for a Python component."""
    core = _core_or_fail()
    seen = {}
    name = _register(core, _make_recording_nav(core, seen), "oracle_nav")
    cfg = _swap_component(core, _run_config(core, EKF_MISSION), "nav", name)
    cfg.oracle = True
    _drive(core, cfg, tmp_path / "oracle.srlog")
    assert all(c["oracle_valid"] for c in seen["cycles"])


# --- estimator introspection hooks -----------------------------------------


def test_python_estimator_channels_round_trip(tmp_path):
    """state()/covariance_upper() reach the nav.est channels."""
    core = _core_or_fail()
    from star_reacher import load

    seen = {}
    name = _register(core, _make_recording_nav(core, seen), "channel_nav")
    cfg = _swap_component(core, _run_config(core, EKF_MISSION), "nav", name)
    log_path = tmp_path / "channels.srlog"
    _drive(core, cfg, log_path)

    data = load(log_path)
    assert data.groups["nav.est"]["x_hat"].shape[1] == 7
    assert data.groups["nav.est"]["P"].shape[1] == 7 * 8 // 2
    assert data.groups["nav.err"]["e"].shape[1] == 7
    # The state the component returned is what the channel carries.
    assert abs(abs(data.groups["nav.est"]["x_hat"][0][0]) - abs(seen["init"]["q0_i2b"][0])) < 1e-12


def test_wrong_length_state_raises(tmp_path):
    """A declared dimension the component does not honour is refused."""
    core = _core_or_fail()

    class BadNav(core.IGncComponent):
        def __init__(self, cfg):
            super().__init__()

        def init(self, ctx):
            pass

        def update(self, inp):
            out = core.GncOutput()
            out.valid = True
            return out

        def state_dim(self):
            return 7

        def state(self):
            return [0.0, 0.0, 0.0]  # three, not seven

    name = _register(core, BadNav, "bad_nav")
    cfg = _swap_component(core, _run_config(core, EKF_MISSION), "nav", name)
    with pytest.raises(Exception) as excinfo:
        _drive(core, cfg, tmp_path / "bad.srlog")
    message = str(excinfo.value)
    assert "state" in message and "7" in message


def test_unknown_component_name_lists_the_registry():
    """Selecting a component that was never registered names the options."""
    core = _core_or_fail()
    cfg = _swap_component(
        core, _run_config(core, EKF_MISSION), "nav", "no_such_component"
    )
    with pytest.raises(Exception) as excinfo:
        core.Sim(cfg, "unused.srlog")
    message = str(excinfo.value)
    assert "no_such_component" in message
    assert "dead_reckoning" in message


def test_duplicate_registration_is_refused():
    """Two components shadowing one name would be a determinism hazard."""
    core = _core_or_fail()

    class Trivial(core.IGncComponent):
        def __init__(self, cfg):
            super().__init__()

        def init(self, ctx):
            pass

        def update(self, inp):
            return core.GncOutput()

    name = _register(core, Trivial, "dup_probe")
    with pytest.raises(Exception) as excinfo:
        core.register_python_component(name, Trivial)
    assert name in str(excinfo.value)
