"""FR-27 per-run seed derivation: SplitMix64(master_seed)[index].

The Monte Carlo layer derives run ``i``'s master seed as element ``i`` of a
SplitMix64 stream seeded with the sweep master seed. Three implementations must
agree bit-for-bit, and this asserts it:

* the committed golden vectors (``tests/golden/mc/splitmix64_stream.toml``);
* the pure-Python mirror ``star_reacher.mc._splitmix64_stream`` (the build-free
  fallback); and
* the compiled core's ``splitmix64_stream`` binding (the D-9 source of truth).

The mirror and goldens are asserted unconditionally, core-less; the "== core
binding" clause is skipped only on a checkout whose compiled core predates the
binding (the binding is added in the same phase as this test, so a rebuilt core
carries it). The seed_0 case shares its first output with the published
SplitMix64 anchor 0xE220A8397B1DCDAF, tying this to the reference.
"""

import tomllib
from pathlib import Path

import pytest

from star_reacher.mc import _splitmix64_stream, splitmix64_stream

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN = REPO_ROOT / "tests" / "golden" / "mc" / "splitmix64_stream.toml"


def _load_golden_cases():
    with GOLDEN.open("rb") as fh:
        doc = tomllib.load(fh)
    cases = []
    for case in doc["case"]:
        seed = int(case["seed"], 16)
        values = [int(v, 16) for v in case["values"]]
        cases.append((case["name"], seed, values))
    return cases


_GOLDEN_CASES = _load_golden_cases()


def _core_binding_or_none():
    """The core's splitmix64_stream, or None if the core/binding is absent."""
    try:
        from star_reacher._corelink import import_core

        return getattr(import_core(), "splitmix64_stream", None)
    except Exception:
        return None


@pytest.mark.parametrize("name, seed, values", _GOLDEN_CASES)
def test_mirror_matches_golden(name, seed, values):
    """The pure-Python mirror reproduces the committed golden vectors."""
    assert _splitmix64_stream(seed, len(values)) == values


@pytest.mark.parametrize("name, seed, values", _GOLDEN_CASES)
def test_public_helper_matches_golden(name, seed, values):
    """splitmix64_stream (binding-or-fallback) reproduces the goldens too."""
    assert splitmix64_stream(seed, len(values)) == values


@pytest.mark.parametrize("name, seed, values", _GOLDEN_CASES)
def test_core_binding_matches_golden_and_mirror(name, seed, values):
    """The compiled binding equals the mirror and the goldens bit-for-bit.

    Skipped only when the core does not carry the binding yet (a core-less
    checkout, or one built before the binding landed). This is the assertion
    that keeps the mirror honest: it is a stand-in for the core, and here it is
    measured against the real thing.
    """
    binding = _core_binding_or_none()
    if binding is None:
        pytest.skip(
            "compiled core has no splitmix64_stream binding; rebuild the core "
            "('pip install .') to exercise the D-9 source of truth. The mirror "
            "and golden assertions above still ran."
        )
    from_binding = list(binding(seed, len(values)))
    assert from_binding == values
    assert from_binding == _splitmix64_stream(seed, len(values))


def test_seed_0_matches_published_anchor():
    """seed_0's first output is the published SplitMix64 anchor for seed 0."""
    assert _splitmix64_stream(0, 1)[0] == 0xE220A8397B1DCDAF


def test_stream_indexing_is_prefix_stable():
    """splitmix64_stream(seed, n) is a prefix of splitmix64_stream(seed, n+k).

    The per-run seed for index i is element i of the stream, so a longer sweep
    must not renumber a shorter one's seeds: the first n elements are the same
    regardless of how many are requested.
    """
    short = _splitmix64_stream(12345, 8)
    long = _splitmix64_stream(12345, 64)
    assert long[:8] == short


def test_every_output_is_a_u64():
    """A per-run seed is a full-width unsigned 64-bit integer."""
    for value in _splitmix64_stream(20260723, 256):
        assert isinstance(value, int)
        assert 0 <= value <= 2**64 - 1
