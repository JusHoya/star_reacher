"""A ready-made attitude-control env for ``missions/leo_attitude_rl.toml``.

FR-28 requires that :class:`~star_reacher.gym.space_env.SpaceEnv` hard-code no
observation, action, or reward semantics. This module is where the *concrete*
semantics for the shipped RL mission live, so that ``check_env`` and the seeding
test can build an env in one line without those choices leaking into the generic
class. It is an example of how to specialise ``SpaceEnv``, not a part of it: a
different problem supplies a different set of these four callables.

The default problem: attitude regulation
-----------------------------------------

The reference mission opens with a 10-degree body-Z attitude error and an
``external`` control slot. The natural task is to command reaction-wheel torque
that drives the estimated attitude onto the commanded attitude and holds it.

* **Observation** (6-vector, from the non-privileged ``Sim.observe()``): the
  attitude-error rotation vector between the navigation estimate and the
  guidance command, followed by the estimated body rate. Both come from
  ``nav_est`` and ``att_cmd`` -- channels an on-board agent would genuinely
  have. No truth enters the observation.

* **Action** (3-vector): a per-axis body torque, clipped to the smallsat
  reaction-wheel authority, mapped to ``{"torque_b_nm": [...], "valid": True}``.

* **Reward**: the negative of the true attitude-error angle plus a small rate
  penalty, using ``truth()`` -- legitimate because a reward is computed outside
  the agent's sight. A perfectly held attitude scores near zero; a large error
  scores negative.
"""

from __future__ import annotations

import math

import numpy as np

from star_reacher.gym.space_env import SpaceEnv

__all__ = ["make_attitude_env"]

# Reaction-wheel torque authority of the smallsat bus (vehicles/smallsat.toml:
# max_torque_Nm = 0.02 per wheel). The action box matches the hardware so a
# learned policy cannot command a torque the vehicle could not produce.
_TAU_MAX_NM = 0.02


def _quat_conjugate(q):
    w, x, y, z = q
    return (w, -x, -y, -z)


def _quat_multiply(a, b):
    # Hamilton product, scalar-first (D-7). Kept small and local rather than
    # reaching into the compiled core: the default env is a plain-Python example
    # and importing the core here would couple this file to a built adapter.
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _error_rotation_vector(q_est, q_cmd):
    """Rotation vector [rad] taking the commanded attitude to the estimate.

    The error quaternion is ``dq = q_cmd^* (x) q_est``; its vector part scaled
    by the half-angle relation gives a rotation vector whose norm is the
    attitude-error angle, a well-behaved 3-vector setpoint for a controller.
    """
    dq = _quat_multiply(_quat_conjugate(q_cmd), q_est)
    w = dq[0]
    # Shortest-arc convention: flip so the scalar part is non-negative, keeping
    # the error angle in [0, pi] rather than reporting its 2*pi complement.
    if w < 0.0:
        dq = tuple(-c for c in dq)
        w = dq[0]
    v = np.array(dq[1:], dtype=np.float64)
    vnorm = float(np.linalg.norm(v))
    if vnorm < 1e-12:
        # At zero error the axis is undefined; the rotation vector is zero.
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.atan2(vnorm, max(-1.0, min(1.0, w)))
    return (angle / vnorm) * v


def _observation(obs_dict):
    """Attitude error (3) then estimated body rate (3), all from nav_est/att_cmd."""
    est = obs_dict["nav_est"]
    cmd = obs_dict["att_cmd"]
    err = _error_rotation_vector(est["q_i2b"], cmd["q_i2b"])
    omega = np.array(est["omega_b_radps"], dtype=np.float64)
    return np.concatenate([err, omega]).astype(np.float32)


def _action(a):
    """A 3-vector action to a torque command, clipped to wheel authority."""
    tau = np.clip(np.asarray(a, dtype=np.float64), -_TAU_MAX_NM, _TAU_MAX_NM)
    # valid=True marks this a live command; without it the core logs a hold.
    return {"torque_b_nm": tau.tolist(), "valid": True}


def _reward(obs_dict, truth_dict, info):
    """Negative true attitude-error angle with a small body-rate penalty.

    Uses truth (privileged) rather than the estimate: reward shaping is the env
    owner's, computed outside the agent's sight, so it may read the true state
    the observation withholds. The command attitude comes from the observation's
    att_cmd channel -- guidance output, not truth.
    """
    err = _error_rotation_vector(truth_dict["q_i2b"], obs_dict["att_cmd"]["q_i2b"])
    angle = float(np.linalg.norm(err))
    rate = float(np.linalg.norm(truth_dict["omega_b_radps"]))
    return -angle - 0.1 * rate


def make_attitude_env(mission_path, outdir, *, max_cycles=None, sim_kwargs=None):
    """Build a :class:`SpaceEnv` with the attitude-control defaults above.

    A one-line constructor for the shipped mission, used by the example and the
    tests. The observation/action/reward semantics are this function's, not
    ``SpaceEnv``'s -- swapping this factory changes the problem without touching
    the generic class.
    """
    # gymnasium is an optional extra; importing it here (not at module top)
    # keeps `import star_reacher.gym.defaults` from hard-requiring it before the
    # env is actually built. In practice SpaceEnv's import already pulled it in.
    from gymnasium import spaces

    # Bounds chosen wide enough to contain any physically reachable state over
    # the short episode (error angle <= pi, body rate well under 1 rad/s here),
    # so check_env's containment assertions hold without clipping observations.
    high_obs = np.array([np.pi, np.pi, np.pi, 10.0, 10.0, 10.0], dtype=np.float32)
    observation_space = spaces.Box(low=-high_obs, high=high_obs, dtype=np.float32)
    action_space = spaces.Box(
        low=-_TAU_MAX_NM, high=_TAU_MAX_NM, shape=(3,), dtype=np.float32
    )
    return SpaceEnv(
        mission_path,
        outdir,
        observation_space=observation_space,
        action_space=action_space,
        observation=_observation,
        action=_action,
        reward=_reward,
        max_cycles=max_cycles,
        sim_kwargs=sim_kwargs,
    )
