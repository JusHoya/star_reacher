"""star_reacher Python frontend: mission validation, SRLOG reading, CLI.

The compiled core (``star_reacher._core``) is deliberately not imported at
module level: reading SRLOG logs, validating missions, and exporting CSV must
work on machines without a compiler toolchain (FR-31, D-12). Commands that
propagate a trajectory import the core lazily and fail with an actionable
message when it is absent.
"""

from star_reacher.srlog import (
    Run,
    SrlogCorruptError,
    SrlogError,
    SrlogVersionError,
    load,
)

# Kept in sync manually with pyproject.toml [project] version and the CMake
# project() VERSION; single-source versioning is deferred until a release
# process earns it (Phase 1 contract).
__version__ = "0.4.0"

__all__ = [
    "Run",
    "SrlogCorruptError",
    "SrlogError",
    "SrlogVersionError",
    "__version__",
    "load",
]
