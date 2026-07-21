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
  central-body constants, and the configured sensor model -- without these
  an aiding estimator written in Python has no route to its measurements;
* the estimator introspection hooks (``state``, ``covariance_upper``,
  ``error_layout``, ``innovations``) round-trip, and a wrong-length return
  raises rather than being silently truncated;
* the FR-24 privileged-truth boundary holds against a component built to
  break it, and the sanctioned ``oracle = true`` debug path still works.

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
            # Attitude followed by a gyro-bias estimate held at exactly zero,
            # which makes the bias rows of nav.err equal the true in-run bias
            # and so directly readable in the log.
            return list(self.q) + [0.0, 0.0, 0.0]

        def covariance_upper(self):
            return [0.0] * (7 * 8 // 2)

        def error_layout(self):
            # The FR-24 replacement for error_state(truth): a description of
            # the state vector, carrying no information about the world. The
            # loop differences against it to write nav.err.
            return [
                core.ErrorBlock(
                    core.ErrorQuantity.ATTITUDE,
                    core.ErrorForm.QUAT_ERROR_LOCAL,
                    0,
                ),
                core.ErrorBlock(
                    core.ErrorQuantity.GYRO_BIAS,
                    core.ErrorForm.DIFFERENCE,
                    4,
                ),
            ]

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
    # The EKF mission does not set oracle, so truth appears nowhere on the
    # component's input, and no other call hands it truth either.
    assert all(c["oracle_valid"] is False for c in cycles)

    # -- true IMU biases reach nav.err through the declared layout ----------
    # The component declares a GYRO_BIAS block over three state slots it
    # holds at exactly zero, so those rows of nav.err are the true in-run
    # bias itself. Reading them out of the log is what shows the bias truth
    # reached the channel: without it a bias-carrying estimator's rows would
    # log as zero, which reads as "no error" rather than "not known" -- and
    # the component obtained none of this, the loop did the differencing.
    from star_reacher import load

    e = load(tmp_path / "recording.srlog").groups["nav.err"]["e"]
    assert e.shape[1] == 7
    bias_rows = e[:, 4:7]
    assert bias_rows.any(), (
        "every gyro-bias row of nav.err is exactly zero; the configured IMU "
        "has a non-zero in-run bias, so this is a broken error-layout path "
        "rather than a physically quiet run"
    )


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


# --- FR-24/FR-25 privileged-truth boundary, probed adversarially -----------
#
# FR-24 says truth() is "privileged; never visible to GNC plugins" and FR-25
# says "truth never appears in GncInput". Reading the loop is not enough to
# establish that: the interesting question is what a component actively
# hunting for truth can reach. The tests below answer it by driving a
# component that scrapes every value it is handed or can reach, and comparing
# the harvest against the true state read from the privileged accessor on the
# very same cycles.
#
# This replaces an earlier pair of tests that pinned a real leak. The
# interface used to expose error_state(truth, e), which the loop called for
# every estimator on every cycle regardless of the oracle flag, handing over
# the true state so the component could compute nav.err. That method no
# longer exists: a component declares its state layout through
# error_layout() and the loop computes nav.err itself, so there is no
# truth-bearing argument left to retain (gnc/component.hpp).


def _make_truth_hunter(core, caught):
    """A component that scrapes every value any route offers it.

    Every override records what it was given, and ``probe`` walks an object
    two levels deep harvesting every float it can see. A truth-bearing field
    added to any of these structures in future therefore shows up as a
    harvested value rather than as a silent widening of the boundary.
    """

    def probe(obj, sink, depth=2):
        for name in dir(obj):
            if name.startswith("_"):
                continue
            try:
                value = getattr(obj, name)
            except Exception:  # noqa: BLE001 - an unreadable attribute is a miss
                continue
            if callable(value) or isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                sink.add(float(value))
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, (int, float)) and not isinstance(
                        item, bool
                    ):
                        sink.add(float(item))
            elif depth > 0:
                probe(value, sink, depth - 1)

    class TruthHunter(core.IGncComponent):
        def __init__(self, cfg):
            super().__init__()
            self.q = [0.0, 0.7071067811865476, 0.7071067811865476, 0.0]
            # Route: the factory argument, which is config the mission states.
            caught["cfg_attrs"] = sorted(
                a for a in dir(cfg) if not a.startswith("_")
            )
            probe(cfg, caught["floats"])

        def init(self, ctx):
            # Route: the one-time init context.
            caught["init_attrs"] = sorted(
                a for a in dir(ctx) if not a.startswith("_")
            )
            probe(ctx, caught["floats"])
            # Route: anything reachable from self, including whatever the base
            # class and the trampoline put there.
            caught["self_attrs"] = sorted(
                a for a in dir(self) if not a.startswith("_")
            )
            probe(self, caught["floats"])
            # Route: the garbage collector. A determined plugin author would
            # ask what holds a reference to it and try to walk back to the
            # driver that owns the truth.
            import gc

            for referrer in gc.get_referrers(self):
                probe(referrer, caught["floats"], depth=1)

        def update(self, inp):
            # Route: the per-cycle input, every member of it.
            caught.setdefault("input_attrs", sorted(
                a for a in dir(inp) if not a.startswith("_")
            ))
            caught.setdefault("oracle_valid", set()).add(inp.oracle.valid)
            caught.setdefault("oracle_r", []).append(list(inp.oracle.r_i_m))
            probe(inp, caught["floats"])
            out = core.GncOutput()
            out.valid = True
            out.q_i2b = self.q
            return out

        # Route: every remaining virtual, each recording every argument it
        # receives. The *args is deliberate - if any of these ever grows a
        # parameter, the value lands in the harvest instead of being missed by
        # a fixed signature.
        def state_dim(self, *args):
            caught.setdefault("argc", {})["state_dim"] = len(args)
            for a in args:
                probe(a, caught["floats"])
            return 4

        def cov_dim(self, *args):
            caught.setdefault("argc", {})["cov_dim"] = len(args)
            for a in args:
                probe(a, caught["floats"])
            return 4

        def innov_max_dim(self, *args):
            caught.setdefault("argc", {})["innov_max_dim"] = len(args)
            for a in args:
                probe(a, caught["floats"])
            # Non-zero so the loop actually calls innovations() and that
            # route is probed too; the component reports no updates.
            return 3

        def state(self, *args):
            caught.setdefault("argc", {})["state"] = len(args)
            for a in args:
                probe(a, caught["floats"])
            return self.q

        def covariance_upper(self, *args):
            caught.setdefault("argc", {})["covariance_upper"] = len(args)
            for a in args:
                probe(a, caught["floats"])
            return [0.0] * 10

        def innovations(self, *args):
            caught.setdefault("argc", {})["innovations"] = len(args)
            for a in args:
                probe(a, caught["floats"])
            return []

        def error_layout(self, *args):
            # The descriptor call: it must be handed nothing at all.
            caught.setdefault("argc", {})["error_layout"] = len(args)
            for a in args:
                probe(a, caught["floats"])
            return [
                core.ErrorBlock(
                    core.ErrorQuantity.ATTITUDE,
                    core.ErrorForm.QUAT_ERROR_LOCAL,
                    0,
                )
            ]

        def error_state(self, *args):
            # The removed route. Defining it must have no effect whatsoever;
            # if the core ever calls it again, this records the fact.
            caught["error_state_called"] = True
            for a in args:
                probe(a, caught["floats"])
            return [0.0] * 4

    return TruthHunter


def test_truth_is_unreachable_from_a_python_component(tmp_path):
    """With oracle false, no route a component can take yields truth.

    The component above scrapes every float reachable through the factory
    argument, the init context, itself, its referrers, the per-cycle input,
    and every argument of every virtual it can override. This test drives it
    one cycle at a time and reads the true state through ``Sim.truth()`` --
    the privileged FR-24 accessor, available to the driver and not to the
    component -- on the same cycles. Nothing the component harvested may
    equal any of those true values.

    Exact equality is the right comparison: the aiding measurements are
    deliberately noisy views of truth, so a component that had obtained a
    true value would match it to the bit while a measurement never does.
    """
    core = _core_or_fail()
    caught = {"floats": set()}
    name = _register(core, _make_truth_hunter(core, caught), "truth_hunter")
    cfg = _swap_component(core, _run_config(core, ATTITUDE_MISSION), "nav", name)
    assert cfg.oracle is False, "the reference mission must not set oracle"

    initial = set()
    evolved = set()
    first = True
    sim = core.Sim(cfg, str(tmp_path / "hunt.srlog"))
    while not sim.done():
        sim.step()
        state = sim.truth()
        assert state["valid"], "the privileged accessor must report a state"
        # The FIRST cycle's true state is separated out because a component
        # legitimately knows it: the mission file states the initial attitude
        # and the vehicle's mass, and init() is given q0 by contract. What a
        # leak would deliver is the state as it EVOLVES, which no
        # configuration a component can read contains.
        sink = initial if first else evolved
        for key in ("r_i_m", "v_i_mps", "q_i2b", "omega_b_radps",
                    "b_g_radps", "b_a_mps2"):
            for v in state[key]:
                sink.add(float(v))
        sink.add(float(state["mass_kg"]))
        first = False
    sim.summary()

    # Also drop the trivially shared constants, which carry no information
    # about the world.
    evolved -= initial
    evolved -= {0.0, 1.0, -1.0}
    assert len(evolved) > 1000, (
        "the truth harvest is too small for this comparison to mean anything"
    )
    leaked = sorted(evolved & caught["floats"])
    assert not leaked, (
        f"a component obtained {len(leaked)} evolving true value(s) with "
        f"oracle = false; first few: {leaked[:5]}"
    )

    # The removed route was not silently revived.
    assert "error_state_called" not in caught, (
        "the core called error_state() on a component; that method was "
        "removed precisely because it handed truth across the boundary"
    )
    assert not hasattr(core.IGncComponent, "error_state"), (
        "IGncComponent still exposes error_state; a plugin must have no "
        "truth-bearing virtual to override"
    )

    # Every virtual the loop calls is called with no arguments at all, so
    # there is no argument through which truth could arrive.
    assert caught["argc"] == {
        "state_dim": 0,
        "cov_dim": 0,
        "innov_max_dim": 0,
        "state": 0,
        "covariance_upper": 0,
        "innovations": 0,
        "error_layout": 0,
    }, caught["argc"]

    # GncInput.oracle is the only truth-shaped member, and it is empty.
    assert caught["oracle_valid"] == {False}
    assert all(r == [0.0, 0.0, 0.0] for r in caught["oracle_r"])
    # Pinning the member lists makes a future truth-bearing field a test
    # failure rather than a silent widening of the boundary. The aiding slots
    # carry noisy measurements and env carries ephemeris and frame context,
    # both of which a real onboard navigator computes for itself.
    assert caught["input_attrs"] == [
        "altimeter", "att_cmd", "cycle", "dt_s", "env", "imu", "imu_fresh",
        "nav_est", "navfix", "oracle", "prev_applied", "startracker", "t_s",
    ]
    assert "truth" not in caught["init_attrs"]
    assert "oracle" not in caught["init_attrs"]
    assert "truth" not in caught["cfg_attrs"]
    # Nothing on the component itself is a truth channel either.
    assert not [
        a for a in caught["self_attrs"] if "truth" in a or "oracle" in a
    ]


def test_nav_err_still_reaches_a_python_estimator(tmp_path):
    """The declared layout is what buys back nav.err for a plugin.

    Closing the leak by refusing to compute nav.err for plugins at all would
    have passed the test above trivially. This one holds the channel to its
    purpose: a Python estimator that declares a layout gets a real error
    channel, computed by the loop in the convention the component declared.
    """
    core = _core_or_fail()
    from star_reacher import load

    caught = {"floats": set()}
    name = _register(core, _make_truth_hunter(core, caught), "layout_nav")
    cfg = _swap_component(core, _run_config(core, ATTITUDE_MISSION), "nav", name)
    log_path = tmp_path / "layout.srlog"
    _drive(core, cfg, log_path)

    e = load(log_path).groups["nav.err"]["e"]
    assert e.shape[1] == 4, "the declared 4-slot attitude block is the channel"
    # Canonicalized to the +w hemisphere on every record, and a real error:
    # the component holds a fixed attitude while the vehicle rotates.
    assert (e[:, 0] >= 0.0).all()
    assert abs(e[:, 1:]).max() > 1e-6


def test_a_component_declaring_no_layout_gets_no_nav_err(tmp_path):
    """No layout means no channel -- not a crash, and not a channel of zeros.

    A zero-filled nav.err would be indistinguishable from a perfect estimate,
    so the run declares the channel only when the component said how to read
    its state vector.
    """
    core = _core_or_fail()
    from star_reacher import load

    class SilentNav(core.IGncComponent):
        def __init__(self, cfg):
            super().__init__()
            self.q = [1.0, 0.0, 0.0, 0.0]

        def init(self, ctx):
            self.q = list(ctx.q0_i2b)

        def update(self, inp):
            out = core.GncOutput()
            out.valid = True
            out.q_i2b = self.q
            return out

        def state_dim(self):
            return 4

        def state(self):
            return self.q

        def covariance_upper(self):
            return [0.0] * 10

    name = _register(core, SilentNav, "silent_nav")
    cfg = _swap_component(core, _run_config(core, ATTITUDE_MISSION), "nav", name)
    log_path = tmp_path / "silent.srlog"
    _drive(core, cfg, log_path)

    data = load(log_path)
    assert "nav.est" in data.groups, "the estimate channel is unaffected"
    assert "nav.err" not in data.groups, (
        "a component that declared no error layout still got a nav.err "
        "channel; zeros there would read as a perfect estimate"
    )


def test_a_partial_layout_is_refused(tmp_path):
    """A layout with a hole is an error, not a zero-filled channel."""
    core = _core_or_fail()

    class PartialNav(core.IGncComponent):
        def __init__(self, cfg):
            super().__init__()
            self.q = [1.0, 0.0, 0.0, 0.0]

        def init(self, ctx):
            self.q = list(ctx.q0_i2b)

        def update(self, inp):
            out = core.GncOutput()
            out.valid = True
            out.q_i2b = self.q
            return out

        def state_dim(self):
            return 7  # attitude plus three slots the layout never covers

        def state(self):
            return self.q + [0.0, 0.0, 0.0]

        def covariance_upper(self):
            return [0.0] * (7 * 8 // 2)

        def error_layout(self):
            return [
                core.ErrorBlock(
                    core.ErrorQuantity.ATTITUDE,
                    core.ErrorForm.QUAT_ERROR_LOCAL,
                    0,
                )
            ]

    name = _register(core, PartialNav, "partial_nav")
    cfg = _swap_component(core, _run_config(core, ATTITUDE_MISSION), "nav", name)
    with pytest.raises(Exception) as excinfo:
        core.Sim(cfg, str(tmp_path / "partial.srlog"))
    message = str(excinfo.value)
    assert "cover 4 slots" in message and "state_dim() == 7" in message, message


def test_a_non_quaternion_led_layout_at_n_equals_m_plus_one_is_refused(tmp_path):
    """KNOWN-ISSUE-P6-5's surviving half, refused at the plugin boundary.

    ``star consistency`` collapses slots 0..3 as a scalar-first error
    quaternion whenever ``n == m + 1`` (``n >= 4``), and the SRLOG header
    carries no layout for it to check that against. This component declares
    the shape that reaches the collapse with no quaternion there: a 3-slot
    velocity block first, a 4-slot attitude block second, ``n == 7`` against
    ``m == 6``. Constructing the run must refuse it rather than produce a log
    whose NEES would be positive, order-unity, and wrong.
    """
    core = _core_or_fail()

    def _make(cov_dim_value):
        class VelocityLedNav(core.IGncComponent):
            def __init__(self, cfg):
                super().__init__()
                self.q = [1.0, 0.0, 0.0, 0.0]

            def init(self, ctx):
                self.q = list(ctx.q0_i2b)

            def update(self, inp):
                out = core.GncOutput()
                out.valid = True
                out.q_i2b = self.q
                return out

            def state_dim(self):
                return 7

            def cov_dim(self):
                return cov_dim_value

            def state(self):
                return [0.0, 0.0, 0.0] + self.q

            def covariance_upper(self):
                n = cov_dim_value
                return [0.0] * (n * (n + 1) // 2)

            def error_layout(self):
                return [
                    core.ErrorBlock(
                        core.ErrorQuantity.VELOCITY, core.ErrorForm.DIFFERENCE, 0
                    ),
                    core.ErrorBlock(
                        core.ErrorQuantity.ATTITUDE,
                        core.ErrorForm.QUAT_ERROR_LOCAL,
                        3,
                    ),
                ]

        return VelocityLedNav

    name = _register(core, _make(6), "velocity_led_nav")
    cfg = _swap_component(core, _run_config(core, ATTITUDE_MISSION), "nav", name)
    with pytest.raises(Exception) as excinfo:
        core.Sim(cfg, str(tmp_path / "velocity_led.srlog"))
    message = str(excinfo.value)
    # The rejection must name the problem, not merely refuse: the block that
    # holds offset 0, both dimensions, and the consumer that would misread it.
    assert "velocity" in message, message
    assert "state_dim() == 7" in message and "cov_dim() == 6" in message, message
    assert "star consistency" in message, message

    # NOT over-rejection, demonstrated on the same layout: declaring the
    # honest 7-dimensional covariance takes the component out of the
    # collapse's path, and the identical block list is then accepted. The two
    # runs differ in cov_dim() alone, so the refusal above is attributable to
    # the n == m + 1 pairing and to nothing else about the layout.
    ok_name = _register(core, _make(7), "velocity_led_nav_square")
    ok_cfg = _swap_component(
        core, _run_config(core, ATTITUDE_MISSION), "nav", ok_name
    )
    core.Sim(ok_cfg, str(tmp_path / "velocity_led_ok.srlog"))


def test_oracle_true_still_injects_truth_and_stamps_the_header(tmp_path):
    """The sanctioned debug path is intact.

    oracle = true is the one route by which a component may see the true
    state, and a run that used it must be identifiable from the log header
    alone. Both halves are checked against the same truth-hunting component
    the boundary test uses, so what is confirmed here is that the structural
    change closed the unsanctioned route without closing this one.
    """
    core = _core_or_fail()
    from star_reacher import load

    caught = {"floats": set()}
    name = _register(core, _make_truth_hunter(core, caught), "oracle_hunter")
    cfg = _swap_component(core, _run_config(core, ATTITUDE_MISSION), "nav", name)
    cfg.oracle = True

    log_path = tmp_path / "oracle.srlog"
    truth_r = []
    sim = core.Sim(cfg, str(log_path))
    while not sim.done():
        sim.step()
        truth_r.append(list(sim.truth()["r_i_m"]))
    sim.summary()

    assert caught["oracle_valid"] == {True}, "oracle = true must populate it"
    # The injected value is the real state, not a placeholder.
    seen_r = [r for r in caught["oracle_r"] if r != [0.0, 0.0, 0.0]]
    assert seen_r, "oracle.r_i_m stayed zero under oracle = true"
    radius = sum(x * x for x in seen_r[-1]) ** 0.5
    assert 6.0e6 < radius < 8.0e6
    assert seen_r[-1] in truth_r, (
        "the injected position does not match the privileged accessor's"
    )

    header = load(log_path).header
    assert header["oracle"] is True, (
        "an oracle run must be identifiable from the header alone (FR-25); "
        "test_gnc_missions.test_oracle_flag_stamped_in_header covers the "
        "built-in chain, this covers a plugin component"
    )


def test_stack_walking_reaches_truth_under_a_python_stepping_driver(tmp_path):
    """A KNOWN-OPEN route, pinned rather than wished away.

    The boundary above governs what the component *interface* hands over. It
    does not, and cannot, govern what arbitrary Python in the same process
    can reach by other means, and one such means is concrete enough to be
    worth recording: when the run is driven from Python through ``Sim``, the
    driver's own ``Sim`` handle is a local on the interpreter stack. A
    component that walks the stack finds it and may call ``Sim.truth()`` --
    the privileged FR-24 accessor -- getting the full evolving true state.

    This is a different animal from the ``error_state(truth)`` channel this
    module's other tests replaced. That one was unconditional: the loop
    pushed truth to every estimator on every cycle of every run, batch or
    stepped, with no action by the plugin author. This one requires a Python
    stepping driver and a plugin that deliberately introspects frames. It is
    the "a plugin is a program, not data" limit already stated in
    docs/gnc_plugins.md sections 6 and 7, and closing it would take
    sandboxing that this project does not attempt.

    The batch path is checked alongside it: under ``star run`` the loop is
    owned by the core, no ``Sim`` is on the Python stack, and the route
    finds nothing. The test asserts today's behaviour on both paths, so a
    future change in either direction is visible rather than silent.
    """
    core = _core_or_fail()

    reached = []

    class FrameHunter(core.IGncComponent):
        def __init__(self, cfg):
            super().__init__()
            self.q = [1.0, 0.0, 0.0, 0.0]

        def init(self, ctx):
            self.q = list(ctx.q0_i2b)

        def update(self, inp):
            import sys

            depth = 0
            try:
                while True:
                    frame = sys._getframe(depth)
                    for value in list(frame.f_locals.values()):
                        if isinstance(value, core.Sim):
                            reached.append(list(value.truth()["r_i_m"]))
                            raise StopIteration
                    depth += 1
            except (ValueError, StopIteration):
                pass
            out = core.GncOutput()
            out.valid = True
            out.q_i2b = self.q
            return out

        def state_dim(self):
            return 4

        def state(self):
            return self.q

        def covariance_upper(self):
            return [0.0] * 10

    name = _register(core, FrameHunter, "frame_hunter")
    cfg = _swap_component(core, _run_config(core, ATTITUDE_MISSION), "nav", name)
    assert cfg.oracle is False

    truth_r = []
    sim = core.Sim(cfg, str(tmp_path / "frames.srlog"))
    while not sim.done():
        sim.step()
        truth_r.append(list(sim.truth()["r_i_m"]))
    sim.summary()

    assert reached, (
        "the stack-walk route found no Sim handle. If the stepping API was "
        "changed so a component can no longer reach the driver, that is an "
        "improvement -- delete this test and say so in docs/gnc_plugins.md "
        "section 7, which currently documents the route as open."
    )
    # It is the real, evolving state, not a stale or placeholder value.
    assert len({tuple(r) for r in reached}) > 100
    assert reached[-1] in truth_r


# --- the plugin boundary as a memory-safety perimeter ----------------------
#
# Everything a Python component returns crosses into fixed-size C++ buffers
# that were sized ONCE, at GNC activation, from the dimensions the component
# declared then. Nothing resizes them afterwards. A Python method, though, is
# re-evaluated on every call, so a component can report one dimension at
# construction and a different one later - and each of the copies below was
# bounded by a value the component controls rather than by the destination.
# These are the reachable-from-pure-Python heap overruns of the Phase 6
# review (findings 2 and 3); each must now be a named exception.


def test_a_growing_state_dim_is_refused_rather_than_overrunning(tmp_path):
    """An estimator whose declared state dimension changes mid-run raises.

    The shape the review found: copy_fixed() compared the length Python
    returned against state_dim(), which it re-queried from Python on the
    same call, so both moved together and the check passed while the
    destination stayed the six doubles it was allocated at construction.
    An augmented or adaptive filter - exactly what FR-25 exists to allow -
    is the realistic way to write this by accident.
    """
    core = _core_or_fail()

    class GrowingNav(core.IGncComponent):
        def __init__(self, cfg):
            super().__init__()
            self.n = 7
            self.cycles = 0

        def init(self, ctx):
            pass

        def update(self, inp):
            self.cycles += 1
            if self.cycles > 3:
                self.n = 12  # the state augments mid-run
            out = core.GncOutput()
            out.valid = True
            return out

        def state_dim(self):
            return self.n

        def cov_dim(self):
            return 7  # pinned, so only state_dim diverges

        def state(self):
            return [0.0] * self.n

        def covariance_upper(self):
            return [0.0] * (7 * 8 // 2)

    name = _register(core, GrowingNav, "growing_nav")
    cfg = _swap_component(core, _run_config(core, EKF_MISSION), "nav", name)
    with pytest.raises(Exception) as excinfo:
        _drive(core, cfg, tmp_path / "growing.srlog")
    message = str(excinfo.value)
    # Names the method, both dimensions, and why they must agree.
    assert "state_dim" in message
    assert "12" in message and "7" in message
    assert "fixed for the lifetime of a run" in message


def test_a_negative_declared_dimension_is_refused(tmp_path):
    """A negative dimension would make the length check vacuously true."""
    core = _core_or_fail()

    class NegativeNav(core.IGncComponent):
        def __init__(self, cfg):
            super().__init__()

        def init(self, ctx):
            pass

        def update(self, inp):
            out = core.GncOutput()
            out.valid = True
            return out

        def state_dim(self):
            return -1

    name = _register(core, NegativeNav, "negative_nav")
    cfg = _swap_component(core, _run_config(core, EKF_MISSION), "nav", name)
    with pytest.raises(Exception) as excinfo:
        core.Sim(cfg, str(tmp_path / "negative.srlog"))
    assert "must not be negative" in str(excinfo.value)


def _innovating_nav(core, y_len, s_len, declared_max):
    """A minimal estimator returning one innovation of a stated shape."""

    class InnovNav(core.IGncComponent):
        def __init__(self, cfg):
            super().__init__()
            self.samples = []

        def init(self, ctx):
            pass

        def update(self, inp):
            s = core.InnovationSample()
            s.sensor_id = 0
            s.y = [1.0] * y_len
            s.s_upper = [1.0] * s_len
            self.samples = [s]
            out = core.GncOutput()
            out.valid = True
            return out

        def state_dim(self):
            return 4

        def cov_dim(self):
            return 4

        def innov_max_dim(self):
            return declared_max

        def state(self):
            return [1.0, 0.0, 0.0, 0.0]

        def covariance_upper(self):
            return [0.0] * 10

        def innovations(self):
            return self.samples

    return InnovNav


def test_an_innovation_wider_than_declared_is_refused(tmp_path):
    """y longer than innov_max_dim() was a write past the end of the buffer.

    The review's reproduction: declare 1, return 6, and the copy runs 40
    bytes past a one-element vector on the first aiding update. The bound is
    now the destination buffer's own capacity, fixed at activation.
    """
    core = _core_or_fail()
    cls = _innovating_nav(core, y_len=6, s_len=21, declared_max=1)
    name = _register(core, cls, "wide_innov")
    cfg = _swap_component(core, _run_config(core, EKF_MISSION), "nav", name)
    with pytest.raises(Exception) as excinfo:
        _drive(core, cfg, tmp_path / "wide.srlog")
    message = str(excinfo.value)
    assert "innov_max_dim" in message
    assert "6" in message and "1" in message


def test_a_short_innovation_covariance_is_refused(tmp_path):
    """A short s_upper was an out-of-bounds READ of uninitialised heap.

    The mirror of the case above, and the more dangerous one: it wrote
    whatever the heap happened to hold into the nav.innov channel, where it
    would have been read as a covariance.
    """
    core = _core_or_fail()
    # y of 6 needs 21 packed entries; supply 10.
    cls = _innovating_nav(core, y_len=6, s_len=10, declared_max=6)
    name = _register(core, cls, "short_cov")
    cfg = _swap_component(core, _run_config(core, EKF_MISSION), "nav", name)
    with pytest.raises(Exception) as excinfo:
        _drive(core, cfg, tmp_path / "short.srlog")
    message = str(excinfo.value)
    assert "packed upper triangle" in message
    assert "21" in message and "10" in message


def test_a_well_formed_innovation_still_logs(tmp_path):
    """The bound checks must not refuse the legitimate case.

    A gate that rejects everything is as useless as one that accepts
    everything, so the same probe is driven with a consistent declaration
    and must produce a readable nav.innov channel.
    """
    core = _core_or_fail()
    from star_reacher import load

    cls = _innovating_nav(core, y_len=3, s_len=6, declared_max=6)
    name = _register(core, cls, "good_innov")
    cfg = _swap_component(core, _run_config(core, EKF_MISSION), "nav", name)
    log_path = tmp_path / "good.srlog"
    _drive(core, cfg, log_path)

    innov = load(log_path).groups["nav.innov"]
    assert len(innov) > 0
    assert innov["y"].shape[1] == 6  # zero-padded to the declared maximum
    assert set(int(m) for m in innov["m"]) == {3}
    # The padding is structural: the 3-wide block sits in the leading corner.
    assert (innov["y"][:, 3:] == 0.0).all()
