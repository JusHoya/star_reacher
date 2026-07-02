"""Make the source-tree package importable when no wheel is installed.

The path is inserted only when ``star_reacher`` is not already importable:
in CI the installed wheel carries the compiled ``_core`` and must not be
shadowed by the pure-Python source tree, while on a core-less development
checkout the source tree under ``python/`` is the only copy available.
"""

import importlib.util
import sys
from pathlib import Path

if importlib.util.find_spec("star_reacher") is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
