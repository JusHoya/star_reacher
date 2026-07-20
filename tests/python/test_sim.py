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
    """An override the core would ignore is refused at the boundary.

    The refusal is what makes a Monte Carlo manifest's per-run ``overrides``
    entry trustworthy (FR-27): a sweep that silently dropped a dimension
    would report results for a scenario it never ran.
    """
    _core_or_fail()
    from star_reacher.sim import Sim, SimError

    sim = Sim(MISSION, tmp_path / "override", force=True)
    with pytest.raises(SimError, match="no such key"):
        sim.reset(overrides={"gravity": 1.0})


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


# --- FR-24 reset(overrides): the resolved-mission override surface ---------


def test_reset_overrides_accept_dotted_paths(tmp_path):
    """A dotted path reaches any numeric leaf of the resolved mission.

    FR-24 names ``overrides`` without enumerating them, and FR-27 records a
    per-run ``overrides`` entry in a Monte Carlo manifest, so the surface has
    to be general enough to express a sweep dimension rather than the two
    keys a first implementation happened to need.
    """
    _core_or_fail()
    from star_reacher.sim import Sim

    sim = Sim(MISSION, tmp_path / "dotted", force=True)
    _, base = sim.reset()
    _, changed = sim.reset(
        overrides={
            "mission.duration_s": 5.0,
            "gnc.control.kp_nm_per_rad": [0.5, 0.5, 0.5],
        }
    )
    assert changed["duration_s"] == 5.0
    # The override is applied before hashing, so the overridden run is a
    # distinct, individually reproducible scenario rather than an unrecorded
    # deviation from the one the hash names.
    assert changed["config_sha256"] != base["config_sha256"]


def test_reset_override_aliases_still_work(tmp_path):
    """The two bare shorthands predate dotted paths and remain accepted."""
    _core_or_fail()
    from star_reacher.sim import Sim

    sim = Sim(MISSION, tmp_path / "alias", force=True)
    _, info = sim.reset(overrides={"duration_s": 3.0, "latency_cycles": 2})
    assert info["duration_s"] == 3.0


def test_reset_override_preserves_integer_leaves(tmp_path):
    """An integer leaf stays an integer, so the config bytes stay canonical.

    Letting ``latency_cycles`` become ``2.0`` would change the canonical
    config JSON -- and therefore config_sha256 -- without changing the run.
    """
    _core_or_fail()
    from star_reacher.sim import Sim

    sim = Sim(MISSION, tmp_path / "intleaf", force=True)
    _, from_int = sim.reset(overrides={"gnc.latency_cycles": 2})
    _, from_float = sim.reset(overrides={"gnc.latency_cycles": 2.0})
    assert from_int["config_sha256"] == from_float["config_sha256"]


@pytest.mark.parametrize(
    "overrides, fragment",
    [
        # A path the mission does not already set: the validator never saw it.
        ({"mission.no_such_key": 1.0}, "no such key"),
        ({"nope.deeper": 1.0}, "has no 'nope'"),
        # Structure is not overridable: a component name selects code whose
        # consequences the mission validator checked and this path cannot.
        ({"gnc.control.component": "external"}, "only numbers"),
        ({"mission.name": 3.0}, "only numbers"),
        # Kind and length must match the leaf being replaced.
        ({"mission.duration_s": "long"}, "expected a number"),
        ({"gnc.control.kp_nm_per_rad": [1.0, 2.0]}, "array of 3"),
        ({"gnc.control.kp_nm_per_rad": 1.0}, "array of 3"),
    ],
)
def test_reset_override_refusals(tmp_path, overrides, fragment):
    """Every refusal names what was wrong with the override, not just that."""
    _core_or_fail()
    from star_reacher.sim import Sim, SimError

    sim = Sim(MISSION, tmp_path / "refuse", force=True)
    with pytest.raises(SimError, match=fragment):
        sim.reset(overrides=overrides)


def test_out_of_range_override_fails_loudly_in_the_core(tmp_path):
    """Range is not rechecked here; the core's own checks are the backstop.

    This pins the honest boundary of the override surface: an override can
    produce a configuration the mission validator never saw, and the run then
    fails with a named reason rather than propagating something absurd.
    """
    _core_or_fail()
    from star_reacher.sim import Sim

    sim = Sim(MISSION, tmp_path / "range", force=True)
    with pytest.raises(Exception) as excinfo:
        sim.reset(overrides={"gnc.control_rate_hz": 0})
    assert "control_rate_hz" in str(excinfo.value), str(excinfo.value)


# --- FR-24 "missing keys hold and are logged" ------------------------------


def test_held_command_keys_are_logged_every_cycle(tmp_path):
    """FR-24: a key omitted from step(commands) holds AND appears in the log.

    The hold half is asserted by test_external_command_applies_and_holds on
    the observation. This asserts the second half on the artifact that
    outlives the run: gnc.cmd carries the command as applied on every control
    cycle, so a held field is written out again rather than being absent from
    the record for the cycles nobody commanded. Without that, a log reader
    could not tell a held command from a gap.
    """
    _core_or_fail()
    from star_reacher import load
    from star_reacher.sim import Sim

    mission_path = _external_control_mission(tmp_path)
    sim = Sim(mission_path, tmp_path / "heldlog", force=True)
    sim.reset()

    first = [0.001, -0.002, 0.003]
    second = [0.004, 0.0, 0.0]
    sim.step({"torque_b_nm": first})  # cycle 0: commanded
    sim.step()  # cycle 1: nothing supplied at all
    sim.step({"torque_b_nm": second})  # cycle 2: commanded
    sim.step({"valid": True})  # cycle 3: torque_b_nm omitted
    while not sim.done():
        sim.step()

    tau = load(tmp_path / "heldlog" / "run.srlog").groups["gnc.cmd"]["tau_b_nm"]
    assert tau[0] == pytest.approx(first, abs=0.0)
    assert tau[1] == pytest.approx(first, abs=0.0), "the full hold was not logged"
    assert tau[2] == pytest.approx(second, abs=0.0)
    assert tau[3] == pytest.approx(second, abs=0.0), "the partial hold was not logged"


# --- lifetime: the log handle, and the episode loop ------------------------


def test_close_releases_the_log_of_an_abandoned_run(tmp_path):
    """A run stopped part way must be able to release its log on request.

    Windows refuses to unlink or reopen a file another handle holds, so an
    abandoned Sim whose C++ object is still alive - pinned by a traceback, by
    a reference cycle, or simply by refcount timing - turns an unrelated
    teardown into PermissionError. That is a real failure already seen in
    this phase, and the remedy is that the file's lifetime is something a
    driver can state rather than something it has to infer.
    """
    _core_or_fail()
    from star_reacher.sim import Sim

    outdir = tmp_path / "abandoned"
    sim = Sim(MISSION, outdir, force=True)
    with _in_repo_root():
        sim.reset()
        sim.step()
        sim.step()
    log = outdir / "run.srlog"
    assert log.exists() and log.stat().st_size > 0

    sim.close()
    # The handle is gone: on Windows this is what an open handle would refuse.
    log.unlink()
    assert not log.exists()
    sim.close()  # idempotent


def test_sim_is_a_context_manager(tmp_path):
    """__exit__ closes even when an exception escapes the body."""
    _core_or_fail()
    from star_reacher.sim import Sim, SimError

    outdir = tmp_path / "ctx"
    with pytest.raises(RuntimeError, match="driver gave up"):
        with Sim(MISSION, outdir, force=True) as sim:
            with _in_repo_root():
                sim.reset()
                sim.step()
            raise RuntimeError("driver gave up")

    (outdir / "run.srlog").unlink()  # would raise on Windows if still open

    # And the closed Sim says so rather than reporting it was never reset.
    with pytest.raises(SimError, match="was closed"):
        sim.step()


def test_step_after_close_raises_in_the_core(tmp_path):
    """The core refuses to write to a released log rather than failing later.

    Checked against the core object directly, because the Python wrapper
    drops its reference on close() and would report the wrapper's own
    message; the guarantee being asserted is the core's.
    """
    core = _core_or_fail()
    from star_reacher.mission import canonical_bytes, validate_mission_file
    from star_reacher.runner import build_run_config

    with _in_repo_root():
        resolved, errors = validate_mission_file(MISSION)
        assert not errors
        cfg, _, _ = build_run_config(
            core, resolved, hashlib.sha256(canonical_bytes(resolved)).hexdigest()
        )
        sim = core.Sim(cfg, str(tmp_path / "closed.srlog"))
        sim.step()
        sim.close()
        sim.close()  # idempotent
        with pytest.raises(Exception, match="after close"):
            sim.step()


def test_reset_twice_starts_a_new_run_at_the_default_force(tmp_path):
    """The FR-24 episode loop works without force=True.

    ``for ep in range(N): sim.reset()`` is the documented usage, and the
    docstring promises a second reset starts a new run over the same path.
    The force guard exists to protect an output this Sim did NOT write; once
    it has opened that path itself, overwriting its own previous run is the
    documented behaviour rather than an accident.
    """
    _core_or_fail()
    from star_reacher.sim import Sim

    outdir = tmp_path / "episodes"
    sim = Sim(MISSION, outdir, force=False)
    with _in_repo_root():
        hashes = []
        for episode in range(3):
            _obs, info = sim.reset(seed=1000 + episode)
            sim.step()
            hashes.append(info["config_sha256"])
        sim.close()
    # Each episode really was a different run, not the same one re-reported.
    assert len(set(hashes)) == 3


def test_reset_still_refuses_an_output_this_sim_did_not_write(tmp_path):
    """The force guard is not disarmed by the episode-loop fix."""
    _core_or_fail()
    from star_reacher.runner import RunnerError
    from star_reacher.sim import Sim

    outdir = tmp_path / "occupied"
    outdir.mkdir()
    (outdir / "run.srlog").write_bytes(b"someone else's run")

    sim = Sim(MISSION, outdir, force=False)
    with _in_repo_root():
        with pytest.raises(RunnerError, match="already exists"):
            sim.reset()


@pytest.mark.parametrize(
    "overrides, seed, fragment",
    [
        ({"gnc.latency_cycles": 2.7}, None, "would be truncated"),
        ({"gnc.control_rate_hz": 10.5}, None, "would be truncated"),
        (None, 3.9, "would be truncated"),
    ],
)
def test_a_fractional_value_for_an_integer_leaf_is_refused(
    tmp_path, overrides, seed, fragment
):
    """Truncating is worse than refusing: the run is reproducible and wrong.

    ``latency_cycles = 2.7`` silently became 2, and config_sha256 recorded
    2, so the run was perfectly reproducible and was not the run the driver
    asked for. The docstring already promised the leaf's kind is preserved;
    it is now enforced instead of approximated by int().
    """
    _core_or_fail()
    from star_reacher.sim import Sim, SimError

    sim = Sim(MISSION, tmp_path / "fractional", force=True)
    with _in_repo_root():
        with pytest.raises(SimError, match=fragment):
            sim.reset(seed=seed, overrides=overrides)


def test_an_integral_float_is_still_accepted_for_an_integer_leaf(tmp_path):
    """The refusal must not reject 2.0, which is the integer 2."""
    _core_or_fail()
    from star_reacher.sim import Sim

    sim = Sim(MISSION, tmp_path / "integral", force=True)
    with _in_repo_root():
        _, from_int = sim.reset(overrides={"gnc.latency_cycles": 2})
        _, from_float = sim.reset(overrides={"gnc.latency_cycles": 2.0})
    assert from_int["config_sha256"] == from_float["config_sha256"]


# --- criterion 4 on a MULTI-SENSOR mission ---------------------------------
#
# The criterion-4 fixtures were single-sensor throughout the phase, and a
# single sensor is a degenerate case: the canonical FR-23 order and the
# alphabetical order a sort_keys round trip produces are then the same list,
# so batch and stepped agreed no matter how either built its sensor
# configuration. The two paths genuinely disagreed on the mission below,
# whose four kinds order canonically as (imu, startracker, navfix,
# altimeter) and alphabetically as (altimeter, imu, navfix, startracker).


EKF_MISSION = REPO_ROOT / "missions" / "leo_ekf_consistency.toml"


def test_the_two_entry_points_are_fed_the_same_order(tmp_path):
    """The builder imposes the canonical order on both of its inputs.

    Asserted at the source rather than only through the hash, because this
    is the property that has to hold: `star run` hands the builder the
    validator's canonical dict and `Sim.reset` hands it one round-tripped
    through canonical_bytes, and the builder must not inherit either.
    """
    import json

    from star_reacher.mission import (
        canonical_bytes,
        canonical_sensor_items,
        validate_mission_file,
    )

    with _in_repo_root():
        resolved, errors = validate_mission_file(EKF_MISSION)
    assert not errors
    round_tripped = json.loads(canonical_bytes(resolved).decode("utf-8"))

    # The premise: the two inputs really are ordered differently.
    assert list(resolved["sensors"]) != list(round_tripped["sensors"])
    canonical = ["imu", "startracker", "navfix", "altimeter"]
    assert [k for k, _ in canonical_sensor_items(resolved["sensors"])] == canonical
    assert [
        k for k, _ in canonical_sensor_items(round_tripped["sensors"])
    ] == canonical


def test_multi_sensor_stepped_run_hashes_identically_to_batch(tmp_path):
    """Criterion 4 on four sensors and a filter that folds their updates."""
    _core_or_fail()
    from star_reacher.runner import run_mission
    from star_reacher.sim import Sim
    from star_reacher.srlog import load

    with _in_repo_root():
        batch = run_mission(EKF_MISSION, outdir=str(tmp_path / "batch"), force=True)
        with Sim(EKF_MISSION, tmp_path / "stepped", force=True) as sim:
            sim.reset()
            stepped_summary = sim.run_to_completion()

    stepped_log = tmp_path / "stepped" / "run.srlog"
    batch_hash = _sha256(batch.srlog_path)
    stepped_hash = _sha256(stepped_log)
    assert stepped_hash == batch_hash, (
        f"stepped and batch logs differ on a multi-sensor mission:\n"
        f"  batch   sha256={batch_hash}\n"
        f"  stepped sha256={stepped_hash}"
    )
    assert stepped_summary["steps"] == batch.summary["steps"]

    # Non-vacuous: the comparison covers a sensor set whose canonical order
    # is not its alphabetical order, and a filter that actually aided.
    declared = load(batch.srlog_path).header["gnc"]["sensors"]
    assert declared == ["imu", "startracker", "navfix", "altimeter"]
    assert declared != sorted(declared)
    assert len(load(batch.srlog_path).groups["nav.innov"]) > 0
