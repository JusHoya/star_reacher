"""FR-28 Gymnasium adapter: a :class:`Sim` behind a ``gymnasium.Env``.

``SpaceEnv`` wraps the FR-24 stepping API (:class:`star_reacher.sim.Sim`) in the
Gymnasium interface so a reinforcement-learning agent can drive a mission one
control period at a time. Two properties shape the whole design:

* **The core has zero Gym knowledge.** ``gymnasium`` is imported only inside
  this subpackage; ``sim.py`` and the compiled core never learn it exists. This
  file only *imports* :class:`Sim` -- it adds no capability the core lacks, it
  reshapes what the core already offers into the Gymnasium 5-tuple.

* **SpaceEnv hard-codes no observation, action, or reward semantics.** What a
  cycle *means* to an agent -- which of the many ``Sim.observe()`` channels
  form the observation vector, how an action vector becomes a ``Sim.step``
  command dict, and what makes one cycle better than another -- is entirely
  the env owner's, supplied as spaces and callables to the constructor. The
  class contributes the plumbing (episode lifecycle, the truth boundary, the
  5-tuple, seeding identity) and nothing about the problem being learned. The
  shipped default for the reference mission lives in
  :mod:`star_reacher.gym.defaults`, deliberately outside this class.

Seeding identity (the FR-28 / Phase 7 criterion-3 clause)
---------------------------------------------------------

``env.reset(seed=S)`` must seed the core exactly as ``star run --seed S`` does,
so that an agent's episode is reproducible against the batch path. The core PRNG
is seeded from the mission's master seed; :meth:`SpaceEnv.reset` therefore
passes the integer seed *unchanged* to ``Sim.reset(seed=S)``. It does not derive
a sub-seed, hash it, or route it through ``gymnasium``'s own RNG -- doing any of
those would make the Gym-side seed differ from the core seed and break the
identity the criterion tests by byte-comparing the two logs.

The truth boundary
------------------

``Sim.truth()`` is privileged: an agent must never see it inside its
observation. This adapter keeps that honest by construction -- the ``observation``
callable is handed only ``Sim.observe()``, never truth, while the ``reward``
callable is handed truth explicitly (rewards are the env owner's, computed
outside the agent's sight). The two paths never cross inside this class.
"""

from __future__ import annotations

# Imported here, not at package top level, so that importing star_reacher does
# not require gymnasium: the RL adapter is an optional extra (`.[rl]`), and the
# core, CLI, and log reader must import without it (FR-31, D-12).
import gymnasium as gym

from star_reacher.sim import Sim

__all__ = ["SpaceEnv"]


class SpaceEnv(gym.Env):
    """A Gymnasium environment driving a :class:`Sim` one control period per step.

    Parameters
    ----------
    mission_path, outdir : path-like
        Passed straight to :class:`Sim`. ``outdir`` receives ``run.srlog`` as a
        batch ``star run`` would write it; each :meth:`reset` overwrites the
        previous episode's log in that directory (the FR-24 episode loop).
    observation_space, action_space : gymnasium.spaces.Box
        The agent-facing spaces. The env owner guarantees that ``observation``
        returns points inside ``observation_space`` and that ``action`` accepts
        any point of ``action_space``.
    observation : callable(obs_dict) -> np.ndarray
        Projects a ``Sim.observe()`` dict into ``observation_space``. Handed the
        non-privileged observation only -- never truth.
    action : callable(np.ndarray) -> dict
        Maps an action into a ``Sim.step`` commands dict (e.g.
        ``{"torque_b_nm": [...], "valid": True}``). Its output must satisfy the
        ``Sim.step`` contract for the mission (an ``external`` control or
        guidance slot for torque/attitude commands).
    reward : callable(obs_dict, truth_dict, info) -> float
        The env owner's reward. Handed the observation, the privileged truth,
        and the constructor's ``info`` so a reward may legitimately use truth
        while the agent's observation cannot.
    max_cycles : int, optional
        Truncation horizon. ``None`` (default) lets the mission's own duration
        end the episode via ``Sim.done()`` (terminated), which is the natural
        fit for a fixed-duration mission. A finite value truncates at that many
        completed steps even if the mission would run longer.
    sim_kwargs : dict, optional
        Extra keyword arguments forwarded to :class:`Sim` (e.g. ``strict`` or
        ``gnc_plugins``). ``force=True`` is set by default so an env can reopen
        its own output directory across constructions in a test session.

    Notes
    -----
    ``render_mode`` is accepted for Gymnasium API completeness; this env has no
    frame to render, so :meth:`render` returns ``None``. The mission's own
    quicklook and 3D playback come from the recorded ``run.srlog`` via
    ``star plot`` / ``star view``, not from a live render loop.
    """

    # No pixel/human rendering: the artifact is the recorded log, replayed by
    # the existing viewer tools. An empty list is the honest declaration.
    metadata = {"render_modes": []}

    def __init__(
        self,
        mission_path,
        outdir,
        *,
        observation_space,
        action_space,
        observation,
        action,
        reward,
        max_cycles=None,
        render_mode=None,
        sim_kwargs=None,
    ):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self._observation = observation
        self._action = action
        self._reward = reward
        self._max_cycles = None if max_cycles is None else int(max_cycles)
        # render_mode must be a member of metadata["render_modes"] or None, or
        # check_env's render assertions fail; None is the only honest value here.
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"render_mode {render_mode!r} is not one of "
                f"{self.metadata['render_modes']}"
            )
        self.render_mode = render_mode

        kwargs = dict(sim_kwargs or {})
        # An env instance is reset repeatedly, and every reset reopens the same
        # output path; force=True lets it own that path rather than refusing on
        # the second episode. The env owner may still override it.
        kwargs.setdefault("force", True)
        self._sim = Sim(mission_path, outdir, **kwargs)
        # Set by reset(); read in the info dicts and by the reward callable.
        self._info = None
        # Steps completed in the current episode, for max_cycles truncation.
        self._steps = 0

    # -- Gymnasium API -----------------------------------------------------

    def reset(self, *, seed=None, options=None):
        """Open a fresh episode and return ``(observation, info)``.

        Follows the Gymnasium contract: ``super().reset(seed=seed)`` first (it
        seeds ``self.np_random``, which check_env verifies is set), then
        ``Sim.reset(seed=<the same integer>)``. The seed reaches the core
        unchanged -- see the module docstring on seeding identity.

        ``options`` may carry ``{"overrides": {...}}``, forwarded to
        ``Sim.reset(overrides=...)``; any other option key is ignored, as the
        Gymnasium API allows.
        """
        # Seeds gym.Env.np_random. It does NOT feed the core PRNG -- the core is
        # seeded solely by the integer passed to Sim.reset below, so Gym-side
        # seeding is exactly core seeding.
        super().reset(seed=seed)
        overrides = None
        if options is not None:
            overrides = options.get("overrides")
        obs_dict, info = self._sim.reset(seed=seed, overrides=overrides)
        self._info = info
        self._steps = 0
        # A reset-time convenience: an agent that intends to command must know
        # the mission wired the external slot. Sim.reset already surfaces it.
        return self._observation(obs_dict), dict(info)

    def step(self, action):
        """Advance one control period and return the Gymnasium 5-tuple.

        ``(observation, reward, terminated, truncated, info)``. ``terminated``
        is ``Sim.done()`` (the mission reached its natural end); ``truncated``
        is set when ``max_cycles`` is reached first. The two are mutually
        exclusive here: a truncation is reported only while the mission has not
        yet terminated, as Gymnasium expects.
        """
        commands = self._action(action)
        obs_dict = self._sim.step(commands)
        self._steps += 1

        terminated = bool(self._sim.done())
        truncated = bool(
            self._max_cycles is not None
            and self._steps >= self._max_cycles
            and not terminated
        )
        # Truth reaches the reward, never the observation: the env owner's
        # reward may use the privileged state the agent is not allowed to see.
        reward = float(self._reward(obs_dict, self._sim.truth(), self._info))
        return (
            self._observation(obs_dict),
            reward,
            terminated,
            truncated,
            dict(self._info),
        )

    def render(self):
        """No live frame -- see the class Notes. Returns ``None``."""
        return None

    def close(self):
        """Release the ``Sim`` log handle.

        Closing matters on Windows: an open ``run.srlog`` handle blocks a later
        reopen or unlink of the directory (see ``Sim.close``). check_env
        constructs and closes envs, so this must actually release the file.
        """
        if self._sim is not None:
            self._sim.close()

    # -- introspection -----------------------------------------------------

    @property
    def sim(self):
        """The wrapped :class:`Sim`, for drivers that need the raw stepping API.

        Exposed read-only so a test or an advanced driver can reach
        ``Sim.summary()`` or ``Sim.truth()`` without SpaceEnv having to mirror
        every method; the RL loop itself needs none of them.
        """
        return self._sim
