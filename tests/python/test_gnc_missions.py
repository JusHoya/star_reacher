"""Phase 6 GNC system tests that REQUIRE the compiled core.

These drive the real run_vehicle GNC path via the CLI runner on the
committed reference mission (missions/leo_attitude_gnc.toml) and on tmp
variants of it, then audit the v1.2 log content: determinism, group
presence and shapes, the ideal IMU's bit-exact increments, dead-reckoning
tracking of the torque-driven truth, the PD law's Python-reimplementation
contract (< 1e-9 N*m, exit criterion 2), the latency_cycles application
shift observed through gnc.cmd (exit criterion 8), oracle header stamping,
and the open-loop/closed-loop pitch-command equality on the ascent mission.

The criterion-2 contract is checked on a purpose-built scenario rather than
on the reference mission's attitude hold, for the reason set out at
``_PD_SCENARIO_RATIONALE`` below: the attitude hold satisfies the tolerance
while leaving three of the law's five equations multiplied by zero. The
other half of the criterion -- the same Python controller against the
independent mpmath goldens -- is ``tests/python/test_gnc_pd_golden.py``.

They fail cleanly, never skip, when the core is absent (the project's
agent-honesty gate).
"""

import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MISSION = REPO_ROOT / "missions" / "leo_attitude_gnc.toml"

sys.path.insert(0, str(REPO_ROOT / "tests" / "refs"))
import pd_attitude as pd  # noqa: E402
import quaternions as qref  # noqa: E402

_CORE_MISSING_MESSAGE = (
    "star_reacher._core is not built in this environment. These integration "
    "tests require the compiled core: build and install it with 'pip install .' "
    "from the repository root. This failure is expected on a core-less checkout "
    "and must be green at integration/CI."
)


def _core_or_fail():
    try:
        from star_reacher import _core
    except ImportError:
        pytest.fail(_CORE_MISSING_MESSAGE)
    return _core


@pytest.fixture(scope="module")
def reference_run(tmp_path_factory, monkeypatch_module=None):
    """One run of the committed GNC mission plus its loaded log."""
    _core_or_fail()
    from star_reacher import load
    from star_reacher.runner import run_mission

    import os

    cwd = os.getcwd()
    os.chdir(REPO_ROOT)  # the mission's vehicle path is repo-relative
    try:
        out = tmp_path_factory.mktemp("gnc_reference")
        result = run_mission(MISSION, out / "run")
        return result, load(result.srlog_path)
    finally:
        os.chdir(cwd)


def _run_variant(tmp_path, name, replacements):
    """Run a textual variant of the reference mission from the repo root."""
    from star_reacher.runner import run_mission

    import os

    text = MISSION.read_text(encoding="utf-8")
    for old, new in replacements:
        assert old in text, f"variant anchor not found: {old!r}"
        text = text.replace(old, new)
    mission = tmp_path / f"{name}.toml"
    mission.write_text(text, encoding="utf-8")
    cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        return run_mission(mission, tmp_path / name)
    finally:
        os.chdir(cwd)


def test_gnc_mission_reruns_bit_identical(tmp_path):
    _core_or_fail()
    r1 = _run_variant(tmp_path, "a", [])
    r2 = _run_variant(tmp_path, "b", [])
    # FR-21 determinism at the whole-file level with the GNC chain, the
    # ideal IMU, and the torque-driven attitude dynamics in the loop.
    assert r1.srlog_sha256 == r2.srlog_sha256


def test_v12_groups_present_and_shaped(reference_run):
    _, run = reference_run
    header = run.header
    assert header["format"] == {"name": "SRLOG", "major": 1, "minor": 2}
    # The "gnc" header object makes cycle rate, latency, and the sensor
    # identity table readable from the header alone (format doc section 3).
    assert header["gnc"] == {
        "cycle_rate_hz": 10,
        "latency_cycles": 0,
        "sensors": ["imu"],
    }
    assert header["oracle"] is False

    # 60 s at 10 Hz: cycles 0..600. Sensors emit from their first sample
    # instant (k >= 1); nav.*/gnc.cmd emit every cycle from activation.
    truth = run.groups["truth"]
    assert len(truth) == 601
    imu = run.groups["sensors.imu"]
    assert len(imu) == 600
    assert imu["dtheta_b_rad"].shape == (600, 3)
    assert imu["dv_b_mps"].shape == (600, 3)
    assert imu["t_s"][0] == pytest.approx(0.1)
    est = run.groups["nav.est"]
    assert est["x_hat"].shape == (601, 7)
    assert est["P"].shape == (601, 28)
    # Dead reckoning carries no covariance: P is identically zero by
    # contract (format doc section 3.2).
    assert np.all(est["P"] == 0.0)
    err = run.groups["nav.err"]
    assert err["e"].shape == (601, 7)
    # nav.err and nav.est share rate and record count by construction.
    assert np.array_equal(err["t_s"], est["t_s"])
    cmd = run.groups["gnc.cmd"]
    assert len(cmd) == 601
    assert cmd["tau_b_nm"].shape == (601, 3)
    assert cmd["q_cmd_i2b"].shape == (601, 4)
    # latency_cycles = 0: every applied command is a fresh chain output.
    assert np.all(cmd["valid"] == 1)
    # No aiding estimator in this phase: nav.innov is not declared.
    assert "nav.innov" not in run.groups
    # The estimator state's quaternion block stays unit-norm.
    qn = np.linalg.norm(est["x_hat"][:, :4], axis=1)
    assert np.max(np.abs(qn - 1.0)) < 1e-12


def test_ideal_imu_increments_bit_exact(reference_run):
    _, run = reference_run
    truth = run.groups["truth"]
    imu = run.groups["sensors.imu"]
    w = truth["w_b_radps"]
    # truth logs every control cycle here and the IMU samples every cycle,
    # so the increment emitted at t_k is the trapezoidal quadrature of
    # cycle k-1 (eq:imu:quadrature): dtheta_k = (w_(k-1) + w_k) * (0.5 dt),
    # exactly the two floating-point operations the core performs (the
    # half-step factor 0.5 * 0.1 is an exact power-of-two scaling) - bit
    # equality, not tolerance (sensors/imu.hpp). The logged truth
    # rate at t_k IS the cycle k-1 attitude-integration endpoint, which is
    # what makes the reconstruction exact.
    expected = (w[:-1] + w[1:]) * (0.5 * 0.1)
    assert np.array_equal(imu["dtheta_b_rad"], expected)
    # No thrust, no aero, no SRP, no drag on this free-flyer, and the
    # gravitational terms cancel exactly (an accelerometer in free fall
    # reads zero, eq:imu:specificforce): dv is exactly zero.
    assert np.all(imu["dv_b_mps"] == 0.0)


def test_dead_reckoning_tracks_torque_driven_truth(reference_run):
    _, run = reference_run
    err = run.groups["nav.err"]["e"]
    # The estimate composes cycle-start held rates while the truth
    # integrates through each cycle: per-cycle first-order-hold error is
    # bounded by |omega_dot| dt^2 / 2 with omega_dot <= tau_max/I_min
    # ~ 5.6e-3 rad/s^2, accumulating through the ~25 s transient to well
    # under 5e-3 (quaternion components / rad/s).
    assert np.max(np.abs(err[:, :4])) < 5e-3
    assert np.max(np.abs(err[:, 4:])) < 5e-3
    # The slew actually happened and settled: the final truth attitude is
    # the commanded attitude to sub-milliradian level.
    truth = run.groups["truth"]
    cmd = run.groups["gnc.cmd"]
    q_end = truth["q_i2b"][-1]
    q_cmd = cmd["q_cmd_i2b"][-1]
    align = abs(float(np.dot(q_end, q_cmd)))  # |cos(theta/2)|
    assert align > 0.99999


# --- Exit criterion 2: the non-degenerate PD scenario ---------------------

_PD_SCENARIO_RATIONALE = """
The committed reference mission holds a fixed attitude, and on it three of the
PD law's five equations are multiplied by zero: attitude_hold commands zero
body rate, so eq:gnc:werr's rotation of w_cmd has nothing to rotate; the
tracking error stays well inside a half turn, so eq:gnc:sign never takes its
short-path branch; and the 10-degree transient never reaches tau_max, so
eq:gnc:sat never clamps. A Python reference that dropped any of the three
reproduced the logged torques exactly, which means no tolerance could have
separated it.

This scenario is built so each of those three has a measurable effect, driving
the FR-24 external-guidance seam so the commanded attitude and rate are the
driver's to choose while the compiled built-in pd_attitude component still
computes every torque:

* a constant non-zero commanded body rate about _PD_RATE_AXIS, so eq:gnc:werr
  carries a real vector;
* an initial commanded attitude offset of _PD_OFFSET_RAD about a DIFFERENT
  axis, _PD_OFFSET_AXIS, so the error quaternion is not parallel to the
  commanded rate. Parallel axes would leave C(dq) w_cmd == w_cmd and make both
  the rotation and its transpose invisible -- the trap a single-axis slew falls
  into;
* the commanded quaternion expressed ANTIPODALLY for the first half of the run.
  The attitude commanded is physically identical either way, so a controller
  that honours eq:gnc:sign produces the same torque from both representations
  while one that omits the branch reverses it. That is the defect the equation
  exists to prevent, expressed as a scenario rather than as an assertion;
* a 60-degree offset against 0.4 N*m/rad gains and a 0.05 N*m authority, so the
  opening transient saturates every axis and eq:gnc:sat clamps for a
  substantial run of cycles.

test_pd_scenario_exercises_every_equation asserts each of these properties
holds, so the gate cannot quietly return to being degenerate. It measures
eq:gnc:werr on C(dq) w_cmd itself rather than on the distance of C(dq) from
the identity or from symmetric, because the parallel-axis case leaves both of
those proxies large while the term they stand in for is inert.
"""

_PD_CYCLE_S = 0.1
_PD_OFFSET_AXIS = np.array([1.0, 2.0, -2.0]) / 3.0
_PD_OFFSET_RAD = np.deg2rad(60.0)
_PD_RATE_AXIS = np.array([2.0, -1.0, -2.0]) / 3.0
_PD_RATE_RADPS = 0.02
# The representation flip lands mid-run so both branches cover a comparable
# number of cycles, with the negative branch holding the saturated transient.
_PD_SIGN_FLIP_CYCLE = 300


def _pd_external_guidance_mission(tmp_path):
    """The reference mission with the guidance slot driven externally.

    Written into tmp_path and run from the repository root, exactly as the
    other mission-variant tests do, so the relative vehicle path resolves and
    the repository tree is untouched.
    """
    text = MISSION.read_text(encoding="utf-8")
    head, sep, tail = text.partition("[gnc.guidance]")
    assert sep, "reference mission no longer has a [gnc.guidance] table"
    # Everything from [gnc.guidance] to the next top-level table is the
    # attitude-hold block; the external slot takes no parameters, so the block
    # is replaced wholesale rather than patched (a leftover q_cmd would be an
    # unknown key for the external component and the validator would reject it).
    lines = tail.split("\n")
    rest = ""
    for index, line in enumerate(lines):
        if line.startswith("[") and index > 0:
            rest = "\n".join(lines[index:])
            break
    out = tmp_path / "pd_nondegenerate.toml"
    out.write_text(
        head + '[gnc.guidance]\ncomponent = "external"\n\n' + rest, encoding="utf-8"
    )
    return out


def _pd_command(cycle, q_start):
    """The commanded attitude and body rate at one control cycle.

    A pure function of the cycle index, so the run is reproducible without the
    driver reading any observation back out of the simulation.
    """
    q = qref.quat_mul(
        q_start, qref.quat_from_rotation_vector(_PD_OFFSET_RAD * _PD_OFFSET_AXIS)
    )
    q = qref.quat_mul(
        q,
        qref.quat_from_rotation_vector(
            _PD_RATE_RADPS * cycle * _PD_CYCLE_S * _PD_RATE_AXIS
        ),
    )
    if cycle < _PD_SIGN_FLIP_CYCLE:
        q = -q  # the antipodal representation of the same attitude
    return q, _PD_RATE_RADPS * _PD_RATE_AXIS


@pytest.fixture(scope="module")
def pd_scenario(tmp_path_factory):
    """Drive the non-degenerate criterion-2 scenario; return its log and gains."""
    _core_or_fail()
    import os

    from star_reacher import load
    from star_reacher.sim import Sim

    tmp = tmp_path_factory.mktemp("pd_nondegenerate")
    mission_path = _pd_external_guidance_mission(tmp)
    config = tomllib.loads(mission_path.read_text(encoding="utf-8"))
    q_start = np.array(config["gnc"]["nav"]["q0"])

    cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        sim = Sim(mission_path, tmp / "run", force=True)
        sim.reset()
        cycle = 0
        while not sim.done():
            q_cmd, w_cmd = _pd_command(cycle, q_start)
            sim.step(
                {
                    "q_i2b": list(q_cmd),
                    "omega_b_radps": list(w_cmd),
                    "valid": True,
                }
            )
            cycle += 1
    finally:
        os.chdir(cwd)
    return load(tmp / "run" / "run.srlog"), config


def _pd_scenario_arrays(pd_scenario):
    """The controller's logged inputs and output, as the reference reads them."""
    run, config = pd_scenario
    est = run.groups["nav.est"]["x_hat"]
    cmd = run.groups["gnc.cmd"]
    control = config["gnc"]["control"]
    return {
        "q_est": est[:, :4],
        "w_est": est[:, 4:],
        "q_cmd": cmd["q_cmd_i2b"],
        "w_cmd": cmd["w_cmd_b_radps"],
        "logged_tau": cmd["tau_b_nm"],
        "kp": np.array(control["kp_nm_per_rad"]),
        "kd": np.array(control["kd_nm_per_radps"]),
        "tau_max": np.array(control["tau_max_nm"]),
    }


def test_pd_law_python_reimplementation_contract(pd_scenario):
    """Phase 6 exit criterion 2, mission side: the Python controller of
    ``tests/refs/pd_attitude`` reproduces the compiled built-in's logged
    applied torques to < 1e-9 N*m across the whole run (latency 0: the applied
    command IS the cycle's chain output).

    The same function is evaluated against the independent mpmath goldens by
    ``tests/python/test_gnc_pd_golden.py``; the two together are the
    criterion's conjunction. The scenario is the one _PD_SCENARIO_RATIONALE
    describes, chosen so every equation the law names is exercised.
    """
    a = _pd_scenario_arrays(pd_scenario)
    tau = pd.pd_torque(
        a["q_cmd"], a["q_est"], a["w_cmd"], a["w_est"], a["kp"], a["kd"], a["tau_max"]
    )
    worst = float(np.max(np.abs(tau - a["logged_tau"])))
    assert worst < 1e-9, (
        f"worst Python-versus-core commanded-torque residual {worst:.6e} N*m "
        f"exceeds the exit-criterion-2 gate of 1e-9 N*m"
    )


def test_pd_scenario_exercises_every_equation(pd_scenario):
    """The criterion-2 scenario is not degenerate on any of the five equations.

    Without this the gate above could pass while three of the equations it
    claims to cover were multiplied by zero, which is precisely how the
    previous fixture failed. Each assertion names the equation it protects;
    the thresholds sit well inside the measured values so a legitimate model
    change does not trip them, while any return to a degenerate fixture does.
    """
    a = _pd_scenario_arrays(pd_scenario)
    dq = pd.error_quaternion(a["q_cmd"], a["q_est"])
    dq0 = dq[:, 0]
    cycles = len(dq0)

    # eq:gnc:werr -- a commanded rate to rotate at all.
    rate_scale = float(np.abs(a["w_cmd"]).max())
    assert rate_scale > 1e-3

    # eq:gnc:werr -- and an error rotation that actually changes the term the
    # torque consumes, C(dq) w_cmd, in both value and handedness.
    #
    # This is measured on that term directly rather than on how far C(dq)
    # sits from the identity or from symmetric. Those two proxies are not
    # equivalent to it, and a mutation survives them: with the error axis
    # PARALLEL to w_cmd, C(dq) w_cmd == w_cmd, so the term is inert and
    # transposing C(dq) changes nothing, yet |C - I| and |C - C^T| stay large
    # because C(dq) is still a large rotation. Measured on this scenario the
    # rotation moves the commanded rate by 125 % of its own peak and its
    # transpose differs by 173 %; on a variant built with _PD_OFFSET_AXIS set
    # equal to _PD_RATE_AXIS the same two quantities collapse to 5.6 % and
    # 13.1 % while the proxies barely move, reading 0.70 and 1.19 against
    # this scenario's 0.71 and 1.24. The 50 % gate below separates the two by
    # better than a factor of two either way; a 0.5 gate on the proxies does
    # not separate them at all.
    c = pd.error_dcm(dq)
    rotated = np.einsum("kij,kj->ki", c, a["w_cmd"])
    transposed = np.einsum("kji,kj->ki", c, a["w_cmd"])
    moved = float(np.abs(rotated - a["w_cmd"]).max())
    handed = float(np.abs(rotated - transposed).max())
    assert moved > 0.5 * rate_scale, (
        f"C(dq) moves the commanded rate by only {moved:.3e} rad/s against a "
        f"commanded rate of {rate_scale:.3e} rad/s; eq:gnc:werr's rotation "
        f"would be invisible to the criterion-2 residual"
    )
    assert handed > 0.5 * rate_scale, (
        f"C(dq) w_cmd differs from its transpose by only {handed:.3e} rad/s "
        f"against a commanded rate of {rate_scale:.3e} rad/s; eq:gnc:werr's "
        f"handedness would be invisible to the criterion-2 residual"
    )

    # eq:gnc:sign -- both branches, each over a substantial run of cycles.
    assert int(np.sum(dq0 < 0.0)) > cycles // 8
    assert int(np.sum(dq0 >= 0.0)) > cycles // 8

    # eq:gnc:sat -- the clamp actually catches, on a real fraction of cycles.
    unclamped = pd.pd_torque(
        a["q_cmd"], a["q_est"], a["w_cmd"], a["w_est"], a["kp"], a["kd"], None
    )
    clamped = np.any(np.abs(unclamped) > a["tau_max"], axis=1)
    assert int(clamped.sum()) > cycles // 20

    # eq:gnc:pd -- every axis carries torque, so no per-axis gain path is
    # multiplied by zero.
    assert np.all(np.abs(a["logged_tau"]).max(axis=0) > 1e-3)


def test_latency_two_cycles_shifts_application(tmp_path):
    _core_or_fail()
    from star_reacher import load

    r0 = _run_variant(tmp_path, "lat0", [])
    r2 = _run_variant(
        tmp_path, "lat2", [("latency_cycles = 0", "latency_cycles = 2")]
    )
    cmd0 = load(r0.srlog_path).groups["gnc.cmd"]
    cmd2 = load(r2.srlog_path).groups["gnc.cmd"]
    header2 = load(r2.srlog_path).header
    assert header2["gnc"]["latency_cycles"] == 2

    # Exit criterion 8: the first applied (valid) command moves from t = 0
    # to t = 2 cycles exactly; the pre-fill holds apply zero torque.
    assert cmd0["valid"][0] == 1
    assert float(cmd0["t_s"][np.argmax(cmd0["valid"] == 1)]) == 0.0
    assert list(cmd2["valid"][:2]) == [0, 0]
    assert np.all(cmd2["tau_b_nm"][:2] == 0.0)
    first_valid = int(np.argmax(cmd2["valid"] == 1))
    assert first_valid == 2
    assert float(cmd2["t_s"][2]) == float(cmd0["t_s"][0]) + 2 * 0.1
    # Both runs compute the cycle-0 command from the same initial state, so
    # the shifted application is bit-identical.
    assert np.array_equal(cmd2["tau_b_nm"][2], cmd0["tau_b_nm"][0])
    assert np.array_equal(cmd2["q_cmd_i2b"][2], cmd0["q_cmd_i2b"][0])


def test_oracle_flag_stamped_in_header(tmp_path):
    _core_or_fail()
    from star_reacher import load

    r = _run_variant(
        tmp_path, "oracle", [("latency_cycles = 0", "latency_cycles = 0\noracle = true")]
    )
    header = load(r.srlog_path).header
    # An oracle run is identifiable from the log header alone (FR-25 /
    # Phase 6 exit criterion 5); the behavioral gating (truth enters
    # GncInput iff this flag is set) is asserted in the C++ suite with a
    # probe component.
    assert header["oracle"] is True


def test_pitch_program_guidance_equals_openloop_command(tmp_path):
    """The closed-loop ascent's commanded attitude equals the open-loop
    pitch-program attitude bit-for-bit at every shared cycle time: both
    paths call the same pitch-table machinery with the same pad basis
    (gnc/builtin.hpp contract), so gnc.cmd.q_cmd_i2b of the GNC ascent must
    equal the kinematic truth q_i2b of the open-loop ascent."""
    _core_or_fail()
    from star_reacher import load
    from star_reacher.runner import run_mission

    import os

    ascent = REPO_ROOT / "missions" / "ascent_leo.toml"
    text = ascent.read_text(encoding="utf-8")
    doc = tomllib.loads(text)
    pitch = next(e for e in doc["sequence"] if e["name"] == "pitch")

    # Remove the open-loop pitch [[sequence]] entry (attitude authority
    # moves to the GNC chain), keep every propulsion/staging/terminate
    # entry, and append the [gnc]/[sensors] tables with the same pitch
    # table. Gains are coarse: the equality under test is about the
    # COMMANDED attitude, which is trajectory-independent (frozen pad
    # basis + absolute-time table).
    segments = text.split("[[sequence]]")
    kept = [segments[0]] + [s for s in segments[1:] if 'name = "pitch"' not in s]
    gnc_text = "[[sequence]]".join(kept)
    gnc_text += (
        "\n[gnc]\n"
        "control_rate_hz = 10\n"
        "latency_cycles = 0\n"
        "[gnc.nav]\n"
        'component = "dead_reckoning"\n'
        # An arbitrary (wrong) initial estimate is fine here: the equality
        # under test concerns the guidance COMMAND, which never reads the
        # nav estimate.
        "q0 = [1.0, 0.0, 0.0, 0.0]\n"
        "[gnc.guidance]\n"
        'component = "pitch_program"\n'
        f"azimuth_deg = {pitch['azimuth_deg']}\n"
        f"pitch_t_s = {pitch['pitch_t_s']}\n"
        f"pitch_deg = {pitch['pitch_deg']}\n"
        "[gnc.control]\n"
        'component = "pd_attitude"\n'
        # Gains sized for discrete-loop stability across BOTH stack
        # configurations (kd*dt/I < ~0.3 down to the stage-2 dry inertia
        # diag(60, 400, 400) kg*m^2); tracking quality is irrelevant here -
        # the equality under test concerns the commanded attitude only.
        "kp_nm_per_rad = [50.0, 300.0, 300.0]\n"
        "kd_nm_per_radps = [150.0, 1200.0, 1200.0]\n"
        "tau_max_nm = [2000.0, 20000.0, 20000.0]\n"
        "[sensors.imu]\n"
        "sample_rate_hz = 10\n"
    )
    mission = tmp_path / "ascent_gnc.toml"
    mission.write_text(gnc_text, encoding="utf-8")

    cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        open_loop = run_mission(ascent, tmp_path / "open")
        closed_loop = run_mission(mission, tmp_path / "closed")
    finally:
        os.chdir(cwd)

    ol = load(open_loop.srlog_path)
    cl = load(closed_loop.srlog_path)
    truth_t = ol.groups["truth"]["t_s"]
    truth_q = ol.groups["truth"]["q_i2b"]
    cmd = cl.groups["gnc.cmd"]

    # Open loop: attitude IS the pitch command from the pitch event (fires
    # at release, t = 2.0 s). Closed loop: gnc.cmd starts at release. The
    # commanded attitude is a pure function of time and the frozen pad
    # basis, so equality holds bit-for-bit on every shared timestamp even
    # though the two trajectories diverge dynamically.
    t0 = 2.0
    common = sorted(
        set(np.round(truth_t / 0.1).astype(int))
        & set(np.round(cmd["t_s"] / 0.1).astype(int))
    )
    common = [k for k in common if k * 0.1 >= t0]
    assert len(common) > 1000  # several hundred seconds of overlap
    ol_index = {int(round(t / 0.1)): i for i, t in enumerate(truth_t)}
    cl_index = {int(round(t / 0.1)): i for i, t in enumerate(cmd["t_s"])}
    q_ol = truth_q[[ol_index[k] for k in common]]
    q_cl = cmd["q_cmd_i2b"][[cl_index[k] for k in common]]
    assert np.array_equal(q_ol, q_cl)


# --- FR-23 full sensor suite in the loop --------------------------------

# All six canonical sensor kinds appended to the reference mission, with
# representative error coefficients. Sample rates divide the 10 Hz control
# rate exactly; the IMU's must equal it (ch:sensors-imu assumption 1).
_FULL_SENSOR_SUITE = """
[sensors.startracker]
sample_rate_hz = 5
boresight_b = [0.0, 0.0, 1.0]
sigma_rad = [1.0e-5, 1.0e-5, 5.0e-5]
sun_exclusion_rad = 0.5236
central_body_exclusion_rad = 0.4363
slew_limit_radps = 0.05

[sensors.sunsensor]
sample_rate_hz = 5
boresight_b = [1.0, 0.0, 0.0]
fov_half_angle_rad = 1.0472
sigma_rad = 2.0e-3

[sensors.navfix]
sample_rate_hz = 1
sigma_r_m = [5.0, 5.0, 9.0]
sigma_v_mps = [0.05, 0.05, 0.09]

[sensors.altimeter]
sample_rate_hz = 2
sigma_bias_m = 3.0
sigma_noise_m = 0.5
h_min_m = 0.0
h_max_m = 1.0e6

[sensors.camera]
sample_rate_hz = 1
fx_px = 800.0
fy_px = 600.0
cx_px = 511.5
cy_px = 383.5
width_px = 1024.0
height_px = 768.0
r_cam_b_m = [0.5, -0.25, 0.125]
q_b2c = [0.9659258262890683, 0.0, 0.25881904510252074, 0.0]
landmarks_fixed_m = [6378137.0, 0.0, 0.0, 0.0, 6378137.0, 0.0]
"""


@pytest.fixture(scope="module")
def full_sensor_run(tmp_path_factory):
    """One run of the reference mission with every FR-23 sensor enabled."""
    _core_or_fail()
    from star_reacher import load

    tmp = tmp_path_factory.mktemp("gnc_full_sensors")
    result = _run_variant(tmp, "full", [("[sensors.imu]",
                                         _FULL_SENSOR_SUITE.lstrip()
                                         + "\n[sensors.imu]")])
    return result, load(result.srlog_path)


def test_full_sensor_suite_reruns_bit_identical(tmp_path):
    """Exit criterion 1: every sensor bit-identical across two seeded runs."""
    _core_or_fail()
    from star_reacher import load

    variant = [("[sensors.imu]",
                _FULL_SENSOR_SUITE.lstrip() + "\n[sensors.imu]")]
    r1 = _run_variant(tmp_path, "full_a", variant)
    r2 = _run_variant(tmp_path, "full_b", variant)
    # Whole-file identity covers every sensor channel at once.
    assert r1.srlog_sha256 == r2.srlog_sha256

    # Per-group identity as well, so a future change that made two groups
    # differ in compensating ways could not hide behind the file hash.
    a = load(r1.srlog_path)
    b = load(r2.srlog_path)
    for kind in ("imu", "startracker", "sunsensor", "navfix", "altimeter",
                 "camera"):
        group = f"sensors.{kind}"
        assert group in a.groups, f"{group} missing from the log"
        for channel in a.groups[group].dtype.names:
            assert np.array_equal(a.groups[group][channel],
                                  b.groups[group][channel]), (
                f"{group}.{channel} differs across two seeded runs")


def test_full_sensor_suite_groups_and_rates(full_sensor_run):
    """Every declared group is present, decimated on the cycle grid."""
    result, run = full_sensor_run
    header = run.header
    # Declared in canonical kind order regardless of TOML key order.
    assert header["gnc"]["sensors"] == [
        "imu", "startracker", "sunsensor", "navfix", "altimeter", "camera",
    ]
    # 60 s run at a 10 Hz control rate: a sensor at R Hz emits 60*R records,
    # starting at its first sample instant (k >= 1).
    for kind, rate in (("imu", 10), ("startracker", 5), ("sunsensor", 5),
                       ("navfix", 1), ("altimeter", 2), ("camera", 1)):
        group = run.groups[f"sensors.{kind}"]
        assert len(group) == 60 * rate, f"{kind} record count"
        # First sample lands one sensor period after activation.
        assert group["t_s"][0] == pytest.approx(1.0 / rate)
        # Timestamps sit exactly on the sensor's grid.
        steps = np.round(group["t_s"] * rate).astype(int)
        assert np.array_equal(steps, np.arange(1, 60 * rate + 1))


def test_camera_pose_channels_are_bit_exact_truth(full_sensor_run):
    """Exit criterion 7: camera pose equals the truth channels bit-exactly.

    The hook copies the same doubles the truth writer receives rather than
    recomputing them (ch:camera implementation note 2), so this is an
    array-equality assertion, not a tolerance.
    """
    _, run = full_sensor_run
    cam = run.groups["sensors.camera"]
    truth = run.groups["truth"]
    # Match on the shared timestamps: the camera samples at 1 Hz, truth at
    # 10 Hz, and both grids are exact multiples of the 0.1 s cycle.
    truth_index = {int(round(t / 0.1)): i for i, t in enumerate(truth["t_s"])}
    rows = [truth_index[int(round(t / 0.1))] for t in cam["t_s"]]
    assert len(rows) == len(cam["t_s"])
    assert np.array_equal(cam["r_m"], truth["r_m"][rows])
    assert np.array_equal(cam["q_i2b"], truth["q_i2b"][rows])


def test_camera_landmark_pixels_are_finite_and_shaped(full_sensor_run):
    """The declared landmark count fixes the record's pixel-pair width."""
    _, run = full_sensor_run
    cam = run.groups["sensors.camera"]
    # Two landmarks were configured, so px_uv carries 2*2 = 4 doubles.
    assert cam["px_uv"].shape == (60, 4)
    assert np.all(np.isfinite(cam["px_uv"]))


def test_startracker_quaternions_are_unit(full_sensor_run):
    """eq:optical:stmodel emits a normalized quaternion every sample."""
    _, run = full_sensor_run
    q = run.groups["sensors.startracker"]["q_meas_i2b"]
    norms = np.linalg.norm(q, axis=1)
    assert np.max(np.abs(norms - 1.0)) < 1e-12
    # The sign is deliberately NOT canonicalized (consumers own the double
    # cover), so this asserts only unit norm.


def test_navfix_residuals_match_configured_sigmas(full_sensor_run):
    """Exit criterion 6, in the loop: fix residuals track their sigmas.

    The chi-square gate itself is a core unit test at fixed truth; here the
    residual against the moving logged truth is checked to be
    sigma-consistent, which is what catches a fix wired to the wrong truth
    row or the wrong axis order.
    """
    _, run = full_sensor_run
    fix = run.groups["sensors.navfix"]
    truth = run.groups["truth"]
    truth_index = {int(round(t / 0.1)): i for i, t in enumerate(truth["t_s"])}
    rows = [truth_index[int(round(t / 0.1))] for t in fix["t_s"]]
    sigma_r = np.array([5.0, 5.0, 9.0])
    resid = fix["r_meas_m"] - truth["r_m"][rows]
    # 60 samples per axis: the sample standard deviation is itself noisy, so
    # this is a loose consistency band, not a distributional gate.
    ratio = resid.std(axis=0) / sigma_r
    assert np.all(ratio > 0.5), ratio
    assert np.all(ratio < 1.8), ratio
    # A residual wired to the wrong truth row would be orbit-scale, not
    # metre-scale: the fix follows a 7000 km trajectory.
    assert np.max(np.abs(resid)) < 100.0
