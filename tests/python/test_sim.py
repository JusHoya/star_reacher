"""FR-24 stepping API tests, including Phase 6 exit criterion 4.

Criterion 4 has two clauses and both are asserted here on the committed
reference GNC mission:

1. step-wise (``Sim.step``) and batch runs of the same scenario produce
   identical log hashes -- ``test_stepped_run_hashes_identically_to_batch``;
2. ``observe()`` twice without ``step()`` returns identical dictionaries --
   ``test_observe_is_idempotent_without_step``.

The C++ half of clause 1 is already pinned by the doctest case
``gnc_cycle_batch_wrapper_matches_stepping`` (cpp/tests/test_gnc_cycle.cpp),
which drives ``VehicleCycle`` directly. These tests extend it to the Python
``Sim`` path a user actually holds, so the two clauses are proved on the
surface FR-24 specifies rather than only on the core beneath it.

They fail cleanly, never skip, when the core is absent (the project's
agent-honesty gate).
"""

import contextlib
import hashlib
import os
import tomllib
from pathlib import Path

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


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextlib.contextmanager
def _in_repo_root():
    """Resolve the mission's relative vehicle path from the repository root.

    The same convention the Phase 6 mission tests use: a mission file names
    its vehicle relatively, and that path is resolved against the working
    directory.
    """
    cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        yield
    finally:
        os.chdir(cwd)


@pytest.fixture(autouse=True)
def _repo_root_cwd():
    """Every test in this module drives a mission with a relative vehicle."""
    with _in_repo_root():
        yield


def _batch_run(outdir):
    """Run the reference mission through the ordinary batch entry point."""
    from star_reacher.runner import run_mission

    run_mission(MISSION, outdir=str(outdir), force=True)
    return outdir / "run.srlog"


def _stepped_run(outdir):
    """Run the same mission one control period at a time through Sim."""
    from star_reacher.sim import Sim

    sim = Sim(MISSION, outdir, force=True)
    sim.reset()
    steps = 0
    while not sim.done():
        sim.step()
        steps += 1
    return outdir / "run.srlog", steps, sim.summary()


# --- exit criterion 4, clause 1 -------------------------------------------


def test_stepped_run_hashes_identically_to_batch(tmp_path):
    """Exit criterion 4: stepped and batch logs are byte-identical."""
    _core_or_fail()
    batch_log = _batch_run(tmp_path / "batch")
    stepped_log, steps, summary = _stepped_run(tmp_path / "stepped")

    batch_hash = _sha256(batch_log)
    stepped_hash = _sha256(stepped_log)
    assert stepped_hash == batch_hash, (
        f"stepped and batch logs differ:\n"
        f"  batch   {batch_log} sha256={batch_hash}\n"
        f"  stepped {stepped_log} sha256={stepped_hash}\n"
        f"run_vehicle() is a loop over the same cycle core the stepping API "
        f"drives, so a difference means the two paths no longer share it."
    )
    # A non-vacuous comparison: the run really advanced, and really wrote.
    assert steps > 0
    assert summary["steps"] > 0
    assert batch_log.stat().st_size > 0


def test_stepped_summary_matches_batch(tmp_path):
    """The stepped run reports the same tallies as the batch run."""
    _core_or_fail()
    from star_reacher.runner import run_mission

    result = run_mission(MISSION, outdir=str(tmp_path / "batch"), force=True)
    _, _, stepped = _stepped_run(tmp_path / "stepped")
    assert stepped["steps"] == result.summary["steps"]
    assert stepped["truth_records"] == result.summary["truth_records"]
    assert stepped["event_records"] == result.summary["event_records"]


# --- exit criterion 4, clause 2 -------------------------------------------


def test_observe_is_idempotent_without_step(tmp_path):
    """Exit criterion 4: observe() twice without step() is identical."""
    _core_or_fail()
    from star_reacher.sim import Sim

    sim = Sim(MISSION, tmp_path / "obs", force=True)
    sim.reset()

    # Before the first step, mid-run, and after several steps: idempotence
    # must hold everywhere, not only on a quiescent initial state.
    checkpoints = [0, 1, 5, 17]
    taken = 0
    for target in checkpoints:
        while taken < target:
            sim.step()
            taken += 1
        first = sim.observe()
        second = sim.observe()
        third = sim.observe()
        assert first == second == third, (
            f"observe() is not idempotent after {taken} step(s); a read "
            f"mutated state or returned a view.\nfirst={first}\n"
            f"second={second}"
        )


def test_observe_returns_copies_not_views(tmp_path):
    """Mutating a returned observation cannot reach into the core."""
    _core_or_fail()
    from star_reacher.sim import Sim

    sim = Sim(MISSION, tmp_path / "views", force=True)
    sim.reset()
    sim.step()
    obs = sim.observe()
    pristine = sim.observe()

    obs["t_s"] = -12345.0
    obs["applied"]["torque_b_nm"][0] = 99.0
    obs["imu"]["dtheta_b_rad"][2] = -7.0
    if obs["nav_x_hat"]:
        obs["nav_x_hat"][0] = 1.0e30

    assert sim.observe() == pristine, (
        "mutating the dict returned by observe() changed what the core "
        "reports; the observation is aliasing core memory"
    )


def test_truth_is_idempotent_and_absent_from_observation(tmp_path):
    """truth() is the separate privileged accessor FR-24 specifies."""
    _core_or_fail()
    from star_reacher.sim import Sim

    sim = Sim(MISSION, tmp_path / "truth", force=True)
    sim.reset()
    sim.step()

    assert sim.truth() == sim.truth()
    obs = sim.observe()
    # The observation must not carry the true state under any spelling.
    assert "truth" not in obs
    assert "r_i_m" not in obs
    assert "v_i_mps" not in obs
    truth = sim.truth()
    assert truth["valid"] is True
    assert len(truth["r_i_m"]) == 3
    # The reference mission flies a ~7000 km LEO state; a truth read that
    # returned zeros would pass a shape check but not this one.
    radius = sum(x * x for x in truth["r_i_m"]) ** 0.5
    assert 6.0e6 < radius < 8.0e6


# --- observation content ---------------------------------------------------


def test_observation_describes_one_instant(tmp_path):
    """Every observation field refers to the cycle just processed."""
    _core_or_fail()
    from star_reacher.sim import Sim

    with MISSION.open("rb") as fh:
        mission = tomllib.load(fh)
    dt = mission["integrator"]["dt_s"]

    sim = Sim(MISSION, tmp_path / "instant", force=True)
    obs0, info = sim.reset()
    # The construction-time snapshot: cycle 0 exists but has not been run.
    assert obs0["cycle"] == 0
    assert obs0["t_s"] == 0.0
    assert obs0["processed"] is False
    assert obs0["applied"]["valid"] is False
    assert info["srlog_path"].endswith("run.srlog")
    assert len(info["config_sha256"]) == 64

    # The nth step() processes cycle n-1, and the observation describes that
    # cycle; time() is the next cycle still to be processed.
    for n in range(1, 6):
        obs = sim.step()
        assert obs["processed"] is True
        assert obs["cycle"] == n - 1
        assert obs["t_s"] == pytest.approx((n - 1) * dt, abs=1e-12)
        assert sim.time() == pytest.approx(n * dt, abs=1e-12)
        assert obs["gnc_active"] is True
        assert obs["applied"]["valid"] is True


def test_observation_carries_estimator_state(tmp_path):
    """A run whose nav component declares a state exposes it in observe()."""
    _core_or_fail()
    from star_reacher.sim import Sim

    sim = Sim(MISSION, tmp_path / "est", force=True)
    sim.reset()
    sim.step()
    obs = sim.observe()
    # dead_reckoning declares n = 7 (quaternion + body rate) with a
    # zero covariance, per gnc/builtin.hpp.
    assert len(obs["nav_x_hat"]) == 7
    assert len(obs["nav_p_upper"]) == 7 * 8 // 2
    assert obs["nav_est"]["valid"] is True
    quat = obs["nav_est"]["q_i2b"]
    assert len(quat) == 4
    assert sum(x * x for x in quat) == pytest.approx(1.0, abs=1e-9)


# --- lifecycle and command surface ----------------------------------------


def test_step_after_done_raises(tmp_path):
    """The log is complete and closed; stepping again is an error."""
    _core_or_fail()
    from star_reacher.sim import Sim

    sim = Sim(MISSION, tmp_path / "done", force=True)
    sim.reset()
    while not sim.done():
        sim.step()
    with pytest.raises(Exception) as excinfo:
        sim.step()
    assert "ended" in str(excinfo.value) or "closed" in str(excinfo.value)


def test_step_before_reset_raises(tmp_path):
    """Stepping an unopened run names the reason."""
    _core_or_fail()
    from star_reacher.sim import Sim, SimError

    sim = Sim(MISSION, tmp_path / "noreset", force=True)
    with pytest.raises(SimError, match="reset"):
        sim.step()


def test_commands_without_external_component_raise(tmp_path):
    """Commanding a self-flying mission fails loudly rather than silently."""
    _core_or_fail()
    from star_reacher.sim import Sim

    sim = Sim(MISSION, tmp_path / "nocmd", force=True)
    sim.reset()
    assert sim._active().has_external_command() is False
    with pytest.raises(Exception) as excinfo:
        sim.step({"torque_b_nm": [0.0, 0.0, 0.01]})
    assert "external" in str(excinfo.value)


def test_unknown_command_key_raises(tmp_path):
    """FR-24: unknown command keys raise rather than being ignored."""
    _core_or_fail()
    from star_reacher.sim import Sim

    mission_path = _external_control_mission(tmp_path)
    sim = Sim(mission_path, tmp_path / "unknown", force=True)
    sim.reset()
    assert sim._active().has_external_command() is True
    with pytest.raises(Exception) as excinfo:
        sim.step({"thrust": 1.0})
    message = str(excinfo.value)
    assert "thrust" in message and "torque_b_nm" in message


def test_reset_rejects_unknown_override(tmp_path):
    """An override the core would ignore is refused at the boundary."""
    _core_or_fail()
    from star_reacher.sim import Sim, SimError

    sim = Sim(MISSION, tmp_path / "override", force=True)
    with pytest.raises(SimError, match="unknown reset override"):
        sim.reset(overrides={"gravity": "off"})


def test_reset_seed_changes_the_config_hash(tmp_path):
    """A reseeded run is a different, individually reproducible scenario."""
    _core_or_fail()
    from star_reacher.sim import Sim

    sim = Sim(MISSION, tmp_path / "seed", force=True)
    _, info_a = sim.reset()
    _, info_b = sim.reset(seed=int(info_a["seed"]) + 1)
    assert info_b["seed"] == info_a["seed"] + 1
    assert info_b["config_sha256"] != info_a["config_sha256"]


# --- external command path (FR-24 step(commands), D-5 zero-order hold) ----


def _external_control_mission(tmp_path):
    """The reference mission with an externally commanded control slot.

    Written into tmp_path and driven with the working directory at the
    repository root, exactly as the Phase 6 mission-variant tests do, so the
    mission's relative vehicle path still resolves and the repository tree is
    left untouched.
    """
    text = MISSION.read_text(encoding="utf-8")
    head, sep, tail = text.partition("[gnc.control]")
    assert sep, "reference mission no longer has a [gnc.control] table"
    # Everything from [gnc.control] to the next top-level table is the PD
    # block; replace it wholesale with the parameter-free external slot.
    rest = _tail_after_table(tail)
    out = tmp_path / "external_control.toml"
    out.write_text(
        head + '[gnc.control]\ncomponent = "external"\n\n' + rest,
        encoding="utf-8",
    )
    return out


def _tail_after_table(tail):
    """The remainder of the file starting at the next top-level table."""
    lines = tail.split("\n")
    for index, line in enumerate(lines):
        if line.startswith("[") and index > 0:
            return "\n".join(lines[index:])
    return ""


def test_external_command_applies_and_holds(tmp_path):
    """A commanded torque is applied, then held when no command is given."""
    _core_or_fail()
    from star_reacher.sim import Sim

    mission_path = _external_control_mission(tmp_path)
    sim = Sim(mission_path, tmp_path / "ext", force=True)
    sim.reset()

    commanded = [0.001, -0.002, 0.003]
    obs = sim.step({"torque_b_nm": commanded})
    applied = obs["applied"]
    assert applied["valid"] is True
    assert applied["torque_b_nm"] == pytest.approx(commanded, abs=0.0)

    # Zero-order hold (D-5): a step with no command re-applies the last.
    held = sim.step()
    assert held["applied"]["torque_b_nm"] == pytest.approx(commanded, abs=0.0)

    # A partial command replaces only the supplied field.
    partial = sim.step({"torque_b_nm": [0.0, 0.0, 0.0]})
    assert partial["applied"]["torque_b_nm"] == pytest.approx(
        [0.0, 0.0, 0.0], abs=0.0
    )


def test_external_command_changes_the_trajectory(tmp_path):
    """The command reaches the dynamics, not just the log.

    Without this the applied-torque assertions above would pass on a build
    that logged the command and threw it away.
    """
    _core_or_fail()
    from star_reacher.sim import Sim

    mission_path = _external_control_mission(tmp_path)
    quiet = Sim(mission_path, tmp_path / "quiet", force=True)
    quiet.reset()
    for _ in range(30):
        quiet.step({"torque_b_nm": [0.0, 0.0, 0.0]})
    quiet_rate = quiet.truth()["omega_b_radps"]

    pushed = Sim(mission_path, tmp_path / "pushed", force=True)
    pushed.reset()
    for _ in range(30):
        pushed.step({"torque_b_nm": [0.0, 0.0, 0.02]})
    pushed_rate = pushed.truth()["omega_b_radps"]

    assert quiet_rate[2] == pytest.approx(0.0, abs=1e-12)
    # tau_z = 0.02 N*m on I_zz = 11 kg*m^2 for 3.0 s gives ~5.5e-3 rad/s.
    assert pushed_rate[2] > 1e-3
