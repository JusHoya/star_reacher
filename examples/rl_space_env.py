"""Drive the FR-28 SpaceEnv with a trivial policy -- no learning framework.

The RL adapter is a plain ``gymnasium.Env``, so it needs nothing more than
gymnasium and the built star_reacher core to run. This example builds the shipped
attitude-control environment over ``missions/leo_attitude_rl.toml`` and steps it
to completion under a fixed zero-torque policy, printing the per-episode return.
Swap ``policy`` for a learned one (or wrap the env in your RL library of choice)
without changing anything about the environment.

Run it from the repository root::

    pip install .[rl]
    python examples/rl_space_env.py

The trivial policy commands zero torque, so the attitude drifts on its opening
10-degree error and the return is negative -- exactly the signal a controller
would learn to improve.
"""

from pathlib import Path

import numpy as np

from star_reacher.gym import make_attitude_env

REPO_ROOT = Path(__file__).resolve().parents[1]
MISSION = REPO_ROOT / "missions" / "leo_attitude_rl.toml"


def policy(observation):
    """A fixed do-nothing policy: command zero torque every cycle.

    ``observation`` is the 6-vector [attitude error (3), body rate (3)] the
    default env exposes; a learned policy would map it to a torque. Zero torque
    is the honest baseline the reward is measured against.
    """
    return np.zeros(3, dtype=np.float32)


def main():
    outdir = REPO_ROOT / "out" / "rl_example"
    seed = 20260601
    env = make_attitude_env(str(MISSION), str(outdir))
    try:
        observation, info = env.reset(seed=seed)
        total_reward = 0.0
        steps = 0
        terminated = truncated = False
        while not (terminated or truncated):
            action = policy(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
        print(f"mission: {info['srlog_path']}")
        print(f"seed: {seed}  steps: {steps}  return: {total_reward:.4f}")
        print("log written -- inspect with: star plot", outdir / "run.srlog")
    finally:
        # Release the log handle so the directory can be reopened or removed.
        env.close()


if __name__ == "__main__":
    main()
