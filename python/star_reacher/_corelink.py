"""Lazy gateway to the compiled core.

The import lives behind a function so that log reading, validation, and
export never pull in the native module (FR-31): only the code paths that
actually propagate a trajectory or draw core RNG streams call this, and they
fail with one consistent, actionable message when the core is absent.
"""

from __future__ import annotations


class CoreMissingError(Exception):
    """The compiled star_reacher._core module is not available."""


_CORE_HELP = (
    "the compiled core 'star_reacher._core' is not available in this environment. "
    "Build and install it with 'pip install .' from the repository root "
    "(requires CMake >= 3.26 and a C++17 compiler). Reading SRLOG files, "
    "validating missions, and exporting CSV work without the core; propagation "
    "and RNG streams require it."
)


def import_core():
    """Return the star_reacher._core module, or raise CoreMissingError."""
    try:
        from star_reacher import _core
    except ImportError as exc:
        raise CoreMissingError(_CORE_HELP) from exc
    return _core
