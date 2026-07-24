"""FR-28 / Phase 7 criterion-3: the SpaceEnv Gymnasium adapter.

Two properties are gated here, both by running real code rather than by
inspection:

* ``gymnasium.utils.env_checker.check_env`` passes on a :class:`SpaceEnv` built
  over ``missions/leo_attitude_rl.toml`` -- the env survives Gymnasium's full
  reset/step/space battery.

* **Gym-side seeding is exactly core seeding.** ``env.reset(seed=S)`` seeds the
  core PRNG identically to ``Sim.reset(seed=S)``, so a SpaceEnv episode driven
  by a fixed command sequence is byte-identical to a bare :class:`Sim` episode
  of the same mission+seed stepped with the same commands. This is proved by
  hashing the two ``run.srlog`` files. A plain zero-order hold episode is in
  turn byte-identical to a batch ``star run`` of the mission, tying the whole
  chain back to the batch path.

The tests require both the compiled core and gymnasium. On a core-less checkout
they fail (never skip) with an actionable message, matching the project's
agent-honesty gate; gymnasium's absence is the RL extra not being installed and
skips, because the adapter is genuinely optional (``pip install .[rl]``).
"""

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import star_reacher

REPO_ROOT = Path(__file__).resolve().parents[2]
RL_MISSION = REPO_ROOT / "missions" / "leo_attitude_rl.toml"

_CORE_MISSING_MESSAGE = (
    "star_reacher._core is not built in this environment. These integration "
    "tests require the compiled core: build and install it with 'pip install .' "
    "from the repository root. This failure is expected on a core-less checkout "
    "and must be green at integration/CI."
)

# gymnasium is the [rl] extra. Its absence is not a failure -- the adapter is
# optional and the rest of the package must import without it -- so these tests
# skip when it is missing rather than failing the suite.
gymnasium = pytest.importorskip("gymnasium", reason="the [rl] extra is not installed")
from gymnasium.utils.env_checker import check_env  # noqa: E402

from star_reacher.gym import SpaceEnv, make_attitude_env  # noqa: E402


def _core_or_fail():
    try:
        from star_reacher import _core  # noqa: F401
    except ImportError:
        pytest.fail(_CORE_MISSING_MESSAGE)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _run_cli(*args):
    env = os.environ.copy()
    pkg_parent = Path(star_reacher.__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(pkg_parent) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "star_reacher", *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )


def _env_episode(outdir, seed, action=None):
    """Run a SpaceEnv episode under a fixed action to completion; return the log."""
    env = make_attitude_env(str(RL_MISSION), str(outdir))
    if action is None:
        # A fixed deterministic zero-torque policy: the cleanest driver for a
        # seeding-identity proof, since the episode depends only on the seed.
        action = np.zeros(env.action_space.shape, dtype=np.float32)
    _, info = env.reset(seed=seed)
    terminated = truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(action)
    env.close()
    return info["srlog_path"]


def _sim_episode(outdir, seed, commands):
    """Run a bare Sim episode with a fixed command dict per step; return the log."""
    from star_reacher.sim import Sim

    sim = Sim(str(RL_MISSION), str(outdir), force=True)
    _, info = sim.reset(seed=seed)
    while not sim.done():
        sim.step(commands)
    sim.close()
    return info["srlog_path"]


# --- check_env -------------------------------------------------------------


def test_check_env_passes(tmp_path):
    """The FR-28 headline: check_env passes on a SpaceEnv over the RL mission.

    Run with skip_render_check=False so the full battery, including the render
    check, is exercised: this env declares no render modes and render_mode=None,
    so the render check is a legitimate no-op pass rather than something skipped.
    """
    _core_or_fail()
    env = make_attitude_env(str(RL_MISSION), str(tmp_path / "check"))
    try:
        # Raises AssertionError on any API violation; returning is the pass.
        check_env(env, skip_render_check=False)
    finally:
        env.close()


def test_spaceenv_hardcodes_no_semantics(tmp_path):
    """SpaceEnv takes the obs/act/reward specs from the caller, not from itself.

    Constructing one directly with arbitrary spaces and callables proves the
    class carries no built-in notion of the observation, action, or reward --
    the FR-28 requirement that those be user-supplied.
    """
    _core_or_fail()
    from gymnasium import spaces

    box1 = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    box3 = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    seen = {}

    def custom_reward(obs, truth, info):
        # Record that the reward path was handed the privileged truth dict.
        seen["saw_truth"] = "q_i2b" in truth
        return 0.0

    env = SpaceEnv(
        str(RL_MISSION),
        str(tmp_path / "generic"),
        observation_space=box1,
        action_space=box3,
        # A degenerate projection: the point is that the class accepts it.
        observation=lambda obs: np.array([obs["t_s"]], dtype=np.float32),
        action=lambda a: {"torque_b_nm": [0.0, 0.0, 0.0], "valid": True},
        reward=custom_reward,
    )
    try:
        obs, info = env.reset(seed=1)
        assert obs.shape == (1,)
        obs, reward, term, trunc, info = env.step(np.zeros(3, dtype=np.float32))
        # The reward callable was handed the privileged truth dict.
        assert seen["saw_truth"] is True
        assert isinstance(reward, float)
    finally:
        env.close()


# --- seeding identity ------------------------------------------------------


def test_env_seeding_is_core_seeding(tmp_path):
    """env.reset(seed=S) seeds the core exactly as Sim.reset(seed=S).

    The SpaceEnv episode under a fixed zero-torque action is byte-identical to a
    bare Sim episode of the same mission+seed stepped with the SAME command
    (torque zero, valid=True). Equal sha256 over run.srlog is the proof that the
    seed reached the core unchanged -- had it been hashed, sub-seeded, or routed
    through gym's RNG, the two logs would diverge.
    """
    _core_or_fail()
    seed = 20260601
    held = {"torque_b_nm": [0.0, 0.0, 0.0], "valid": True}
    env_log = _env_episode(tmp_path / "env", seed)
    sim_log = _sim_episode(tmp_path / "sim", seed, held)
    assert _sha256(env_log) == _sha256(sim_log), (
        "the SpaceEnv-seeded episode diverged from the Sim-seeded episode for "
        "the same seed and commands; Gym-side seeding is not core seeding"
    )


def test_env_reset_is_repeatable(tmp_path):
    """Two identically seeded episodes are bit-identical (determinism)."""
    _core_or_fail()
    seed = 20260601
    # Same directory reused across episodes: reset() overwrites its own log, the
    # documented FR-24 episode loop, so both hashes describe the last episode.
    first = _sha256(_env_episode(tmp_path / "rep", seed))
    second = _sha256(_env_episode(tmp_path / "rep", seed))
    assert first == second


def test_different_seeds_diverge(tmp_path):
    """Different seeds produce different logs -- the seed is actually consumed."""
    _core_or_fail()
    a = _sha256(_env_episode(tmp_path / "a", 20260601))
    b = _sha256(_env_episode(tmp_path / "b", 424242))
    assert a != b, "changing the seed did not change the run"


def test_hold_episode_matches_batch_star_run(tmp_path):
    """A pure-hold Sim episode is byte-identical to a batch `star run`.

    This bridges the seeding-identity chain back to the batch path: a Sim
    stepped with no commands (pure zero-order hold, the initial-hold semantics
    an uncommanded external slot uses) equals `star run` of the same mission,
    which carries the mission's own seed. Distinct from the SpaceEnv comparison
    only in the logged command-validity flag: an env's zero-torque action is a
    live command (valid=True), whereas an uncommanded hold logs valid=False, so
    the two differ solely in that flag while both apply zero torque.
    """
    _core_or_fail()
    seed = 20260601  # equal to the mission's own [run] seed, which `star run` uses
    hold_log = _sim_episode(tmp_path / "hold", seed, None)  # None => pure ZOH hold

    run_out = tmp_path / "batch"
    result = _run_cli("run", str(RL_MISSION), "-o", str(run_out), "--force")
    assert result.returncode == 0, result.stderr
    assert _sha256(hold_log) == _sha256(run_out / "run.srlog"), (
        "the stepped pure-hold episode diverged from the batch star run"
    )
