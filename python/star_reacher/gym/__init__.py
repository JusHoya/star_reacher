"""FR-28 reinforcement-learning adapter: ``star_reacher.gym.SpaceEnv``.

A pure-Python Gymnasium wrapper over the FR-24 stepping API. The compiled core
has zero Gym knowledge; ``gymnasium`` is imported only inside this subpackage,
so it is an optional extra (``pip install star_reacher[rl]``) and importing
``star_reacher`` itself never requires it.

:class:`SpaceEnv` is generic -- it hard-codes no observation, action, or reward
semantics; the env owner supplies those. :func:`make_attitude_env` is the shipped
default for ``missions/leo_attitude_rl.toml``, an *example* specialisation that
lives outside the generic class.
"""

from star_reacher.gym.defaults import make_attitude_env
from star_reacher.gym.space_env import SpaceEnv

__all__ = ["SpaceEnv", "make_attitude_env"]
