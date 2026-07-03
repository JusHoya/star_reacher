"""Ephemeris validation against committed JPL reference vectors (FR-4, D-8).

Phase 2 exit criterion 2 evidence, executed offline: the compiled core's
``Ephemeris`` evaluator runs over the committed excerpt
``tests/golden/ephemeris/excerpt_de440s.sreph`` and is compared against

- geometric ICRF state vectors fetched from the JPL Horizons API (committed
  with transcripts under ``tests/golden/ephemeris/horizons/``), gated at
  < 1 m for the Sun, EMB, Venus, Mars, and Jupiter barycenters and the Earth;
- jplephem's independent evaluation of the checksummed DE440 kernel for the
  lunar segments (``moon_de440_jplephem.toml``), gated at < 1 mm, because
  Horizons serves the Moon from DE441, whose lunar orbit differs from DE440
  by roughly 2-5 m across 2020-2060 by design (Park et al. 2021, AJ 161:105,
  Section 6); Horizons remains a 10 m DE441-envelope bound for the Moon;
- jplephem's evaluation of the DE440 lunar principal-axis PCK for the
  libration angles (``librations_jplephem.toml``).

Like the other integration tests, these FAIL (never skip) without the
compiled core: a green suite must mean the evaluator surface works.
"""

import math
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = REPO_ROOT / "tests" / "golden" / "ephemeris"
EXCERPT = GOLDEN_DIR / "excerpt_de440s.sreph"

_CORE_MISSING_MESSAGE = (
    "star_reacher._core is not built in this environment. These integration tests "
    "require the compiled core: build and install it with 'pip install .' from the "
    "repository root (CMake >= 3.26 and a C++17 compiler required). This failure is "
    "expected on a core-less checkout and must be green at integration/CI."
)

# Tolerances for jplephem-referenced goldens: identical DE440 coefficients
# evaluated by an independent implementation with a different summation
# order. Measured worst cases (generation log 2026-07-02): 8.6e-8 m position,
# 9.1e-13 rad angle, 8.5e-22 rad/s rate; bounds carry two-plus orders of
# margin while still failing decisively on any wrong record, coefficient,
# unit, or center (all of which displace results by >= whole meters/arcsec).
MOON_DE440_TOL_M = 1e-3
LIBRATION_TOL_RAD = 1e-10
LIBRATION_TOL_RADPS = 1e-19


def _core_or_fail():
    try:
        from star_reacher import _core
    except ImportError:
        pytest.fail(_CORE_MISSING_MESSAGE)
    return _core


def _load_ephemeris():
    core = _core_or_fail()
    return core.Ephemeris.load(str(EXCERPT))


def _toml_cases(name: str) -> list[dict]:
    with open(GOLDEN_DIR / name, "rb") as fh:
        return tomllib.load(fh)["case"]


def test_repacked_positions_match_horizons():
    """Repack vs Horizons geometric ICRF positions at every committed epoch.

    Gate per case: ``gate_m`` from the golden file - 1 m (exit criterion 2)
    for the six DE440==DE441 quantities, 10 m DE441-envelope for the two
    lunar quantities (see module docstring; the authoritative sub-meter lunar
    gate is test_moon_matches_de440_jplephem_reference).
    """
    eph = _load_ephemeris()
    cases = _toml_cases("horizons_vectors.toml")
    per_quantity: dict[str, int] = {}
    worst: dict[str, float] = {}
    for case in cases:
        quantity = case["quantity"]
        t = float(case["tdb_s"])
        composer = case["composer"]
        if composer == "moon_minus_earth":
            r_m, _v = eph.moon_geocentric(t)
        else:
            assert composer == f"state:{quantity}", composer
            r_m, _v = eph.state(quantity, t)
        r_ref = [float(c) * 1000.0 for c in case["r_km"]]
        err = math.dist(r_m, r_ref)
        gate = float(case["gate_m"])
        assert err < gate, (
            f"{quantity} at {case['epoch_iso']}: |r_repack - r_horizons| = {err:.6f} m "
            f">= gate {gate} m"
        )
        per_quantity[quantity] = per_quantity.get(quantity, 0) + 1
        worst[quantity] = max(worst.get(quantity, 0.0), err)
    # The exit criterion demands >= 20 epochs spanning 2020-2060 per quantity
    # (span endpoints and record boundaries are baked in by generate.py).
    assert set(per_quantity) == {
        "sun",
        "emb",
        "venus_bary",
        "mars_bary",
        "jupiter_bary",
        "earth",
        "moon",
        "moon_geocentric",
    }
    for quantity, count in per_quantity.items():
        assert count >= 20, f"{quantity}: only {count} epochs"
    # The six DE440==DE441 quantities must clear the hard 1 m gate.
    for quantity in ("sun", "emb", "venus_bary", "mars_bary", "jupiter_bary", "earth"):
        assert worst[quantity] < 1.0


def test_moon_matches_de440_jplephem_reference():
    """Lunar segments vs jplephem's independent DE440 evaluation, < 1 mm.

    This is the authoritative DE440 fidelity gate for the Moon (Horizons
    serves DE441 for lunar states; see module docstring).
    """
    eph = _load_ephemeris()
    cases = _toml_cases("moon_de440_jplephem.toml")
    assert len(cases) >= 20
    for case in cases:
        t = float(case["tdb_s"])
        r_emb, v_emb = eph.state("moon", t)
        r_geo, v_geo = eph.moon_geocentric(t)
        ref_emb_r = [float.fromhex(h) for h in case["moon_emb_r_m"]]
        ref_emb_v = [float.fromhex(h) for h in case["moon_emb_v_mps"]]
        ref_geo_r = [float.fromhex(h) for h in case["moon_geo_r_m"]]
        ref_geo_v = [float.fromhex(h) for h in case["moon_geo_v_mps"]]
        assert math.dist(r_emb, ref_emb_r) < MOON_DE440_TOL_M, case["epoch_iso"]
        assert math.dist(r_geo, ref_geo_r) < MOON_DE440_TOL_M, case["epoch_iso"]
        # Velocity tolerance scaled by a typical position/velocity magnitude
        # ratio is unnecessary: measured velocity differences are < 2e-11
        # m/s, so the position bound over one second bounds them trivially.
        assert math.dist(v_emb, ref_emb_v) < MOON_DE440_TOL_M, case["epoch_iso"]
        assert math.dist(v_geo, ref_geo_v) < MOON_DE440_TOL_M, case["epoch_iso"]


def test_lunar_librations_match_jplephem():
    """Libration angles/rates vs jplephem's PCK evaluation at 21 epochs."""
    eph = _load_ephemeris()
    cases = _toml_cases("librations_jplephem.toml")
    assert len(cases) >= 20
    for case in cases:
        t = float(case["tdb_s"])
        angles, rates = eph.lunar_librations(t)
        ref_angles = [float.fromhex(h) for h in case["angles_rad"]]
        ref_rates = [float.fromhex(h) for h in case["rates_radps"]]
        for got, ref in zip(angles, ref_angles):
            assert abs(got - ref) < LIBRATION_TOL_RAD, case["epoch_iso"]
        for got, ref in zip(rates, ref_rates):
            assert abs(got - ref) < LIBRATION_TOL_RADPS, case["epoch_iso"]


def test_ephemeris_error_paths_via_bindings():
    """The C++ error contract crosses the binding as typed Python errors."""
    eph = _load_ephemeris()
    with pytest.raises(ValueError, match="phobos"):
        eph.state("phobos", 631108800.0)
    # std::out_of_range crosses pybind11 as IndexError.
    with pytest.raises(IndexError, match="extrapolate"):
        eph.state("sun", 6.0e8)  # 2019-01, before every stored record
    with pytest.raises(IndexError, match="extrapolate"):
        eph.lunar_librations(1.9e9)  # 2060-03, after every stored record
    with pytest.raises(RuntimeError):
        type(eph).load(str(GOLDEN_DIR / "manifest.toml"))  # not an SREPH file
