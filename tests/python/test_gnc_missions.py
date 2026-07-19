"""Phase 6 GNC system tests that REQUIRE the compiled core.

These drive the real run_vehicle GNC path via the CLI runner on the
committed reference mission (missions/leo_attitude_gnc.toml) and on tmp
variants of it, then audit the v1.2 log content: determinism, group
presence and shapes, the ideal IMU's bit-exact increments, dead-reckoning
tracking of the torque-driven truth, the PD law's Python-reimplementation
contract (< 1e-9 N*m, exit criterion 2), the latency_cycles application
shift observed through gnc.cmd (exit criterion 8), oracle header stamping,
and the open-loop/closed-loop pitch-command equality on the ascent mission.

They fail cleanly, never skip, when the core is absent (the project's
agent-honesty gate).
"""

import tomllib
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MISSION = REPO_ROOT / "missions" / "leo_attitude_gnc.toml"

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
    # equality, not tolerance (sensors/imu_ideal.hpp). The logged truth
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


def test_pd_law_python_reimplementation_contract(reference_run):
    """Phase 6 exit criterion 2, arithmetic side: an independent Python
    reimplementation of the documented PD law reproduces the logged applied
    torques to < 1e-9 N*m across the whole run (latency 0: the applied
    command IS the cycle's chain output)."""
    _, run = reference_run
    est = run.groups["nav.est"]["x_hat"]
    cmd = run.groups["gnc.cmd"]

    mission = tomllib.loads(MISSION.read_text(encoding="utf-8"))
    kp = np.array(mission["gnc"]["control"]["kp_nm_per_rad"])
    kd = np.array(mission["gnc"]["control"]["kd_nm_per_radps"])
    tau_max = np.array(mission["gnc"]["control"]["tau_max_nm"])

    q_est = est[:, :4]
    w_est = est[:, 4:]
    q_cmd = cmd["q_cmd_i2b"]
    w_cmd = cmd["w_cmd_b_radps"]

    # Hamilton product conj(q_cmd) (x) q_est, scalar-first (D-7): the
    # conjugate keeps the scalar and negates the vector part
    # (eq:gnc:deltaq; no renormalization of dq).
    pw = q_cmd[:, 0]
    px = -q_cmd[:, 1]
    py = -q_cmd[:, 2]
    pz = -q_cmd[:, 3]
    qw, qx, qy, qz = q_est[:, 0], q_est[:, 1], q_est[:, 2], q_est[:, 3]
    dq0 = pw * qw - px * qx - py * qy - pz * qz
    dqx = pw * qx + px * qw + py * qz - pz * qy
    dqy = pw * qy - px * qz + py * qw + pz * qx
    dqz = pw * qz + px * qy - py * qx + pz * qw
    s = np.where(dq0 >= 0.0, 1.0, -1.0)  # sign(0) = +1 (eq:gnc:sign)
    dq_vec = np.stack([dqx, dqy, dqz], axis=1)
    # eq:gnc:werr: resolve the commanded rate into the estimated body frame
    # through the error DCM C(dq) (quaternion-to-DCM, eq:notation:quat2dcm;
    # dq is cmd-to-body). Built row by row from the dq components.
    ww, xx, yy, zz = dq0 * dq0, dqx * dqx, dqy * dqy, dqz * dqz
    c = np.empty((len(dq0), 3, 3))
    c[:, 0, 0] = ww + xx - yy - zz
    c[:, 0, 1] = 2.0 * (dqx * dqy + dq0 * dqz)
    c[:, 0, 2] = 2.0 * (dqx * dqz - dq0 * dqy)
    c[:, 1, 0] = 2.0 * (dqx * dqy - dq0 * dqz)
    c[:, 1, 1] = ww - xx + yy - zz
    c[:, 1, 2] = 2.0 * (dqy * dqz + dq0 * dqx)
    c[:, 2, 0] = 2.0 * (dqx * dqz + dq0 * dqy)
    c[:, 2, 1] = 2.0 * (dqy * dqz - dq0 * dqx)
    c[:, 2, 2] = ww - xx - yy + zz
    w_cmd_b = np.einsum("kij,kj->ki", c, w_cmd)
    tau = -kp * s[:, None] * dq_vec - kd * (w_est - w_cmd_b)  # eq:gnc:pd
    tau = np.clip(tau, -tau_max, tau_max)  # eq:gnc:sat

    assert np.max(np.abs(tau - cmd["tau_b_nm"])) < 1e-9


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
