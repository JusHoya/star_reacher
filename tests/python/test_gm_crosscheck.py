"""Anti-drift gate for the Python GM duplicates (FR-31 vs single-home rule).

``star_reacher.derived`` must work without the compiled core, so it carries
its own copies of the central-body GM values whose single home is
``cpp/include/star/constants.hpp``. Duplicated constants can drift silently;
this test pins every Python entry bit-exactly to the core's ``gm()`` binding
so any edit to either side that is not mirrored fails CI.

Requires the compiled core and FAILS (never skips) when it is missing, per
the test_integration_core.py convention: a skip would let drifted constants
masquerade as verified.
"""

import pytest

from star_reacher import derived

_CORE_MISSING_MESSAGE = (
    "star_reacher._core is not built in this environment. The GM cross-check "
    "requires the compiled core: build and install it with 'pip install .' "
    "from the repository root. This failure is expected on a core-less "
    "checkout and must be green at integration/CI."
)


def _core_or_fail():
    try:
        from star_reacher import _core
    except ImportError:
        pytest.fail(_CORE_MISSING_MESSAGE)
    return _core


def test_gm_table_covers_exactly_the_supported_central_bodies():
    # The core's central-body vocabulary (cpp/include/star/run.hpp): any
    # body a log header can name must have a Python GM, and nothing more.
    assert set(derived.GM_M3_PER_S2) == {"earth", "moon", "mars", "sun"}


def test_gm_values_match_core_bit_exactly():
    core = _core_or_fail()
    for body, python_value in derived.GM_M3_PER_S2.items():
        core_value = core.gm(body)
        # Bit equality, not closeness: these are copies of one constant,
        # and any representable difference is drift.
        assert python_value == core_value, (
            f"derived.GM_M3_PER_S2[{body!r}] = {python_value!r} != "
            f"core.gm({body!r}) = {core_value!r}; the Python copy in "
            f"python/star_reacher/derived.py must mirror "
            f"cpp/include/star/constants.hpp exactly"
        )
