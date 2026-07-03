"""Python-side evidence for the Phase 2 integrator/event exit criteria.

These tests drive the compiled core's test-support entry points -- thin
wrappers over the same star/testsupport/acceptance.hpp drivers the doctest
suite asserts on -- against the committed goldens in
``tests/golden/integrators/`` (provenance in that directory's manifest).
The pytest layer exists so the acceptance numbers (convergence slopes,
invariant drift, apsis-event errors) are reproducible from Python with the
installed wheel, not only from the C++ test binary.

Like the other core-backed tests, these FAIL (never skip) when the compiled
core is absent: a green suite must mean the whole contract holds.
"""

from __future__ import annotations

import math
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = REPO_ROOT / "tests" / "golden" / "integrators"

_CORE_MISSING_MESSAGE = (
    "star_reacher._core is not built in this environment. These tests require "
    "the compiled core: build and install it with 'pip install .' from the "
    "repository root. This failure is expected on a core-less checkout and "
    "must be green at integration/CI."
)

# Dyadic measurement span and ladders -- identical to the doctest suite
# (cpp/tests/test_integrate.cpp): an exact integer of seconds divisible by
# every ladder step, so accumulated step times carry zero rounding error and
# the measured slopes cannot touch the double-precision roundoff plateau.
SPAN_S = 7168.0
RK4_LADDER = [16.0, 8.0, 4.0, 2.0]
RKF78_LADDER = [512.0, 256.0, 128.0, 64.0]


def _core_or_fail():
    try:
        from star_reacher import _core
    except ImportError:
        pytest.fail(_CORE_MISSING_MESSAGE)
    return _core


def _load_cases(name: str) -> list[dict]:
    with open(GOLDEN_DIR / name, "rb") as fh:
        return tomllib.load(fh)["case"]


def _find_case(cases: list[dict], name: str) -> dict:
    for case in cases:
        if case["name"] == name:
            return case
    raise KeyError(name)


def _hex(v: str) -> float:
    return float.fromhex(v)


def _vec3(case: dict, key: str) -> list[float]:
    return [_hex(x) for x in case[key]]


@pytest.fixture(scope="module")
def orbit() -> dict:
    d = _find_case(_load_cases("kepler_orbit.toml"), "definition")
    return {
        "mu": _hex(d["mu_m3ps2"]),
        "r0": _vec3(d, "r0_m"),
        "v0": _vec3(d, "v0_mps"),
        "period": _hex(d["period_s"]),
        "max_r4": _hex(d["max_r4_mps4"]),
    }


def _loglog_slope(points: list[dict]) -> float:
    xs = [math.log(p["h_s"]) for p in points]
    ys = [math.log(p["err_m"]) for p in points]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / sxx


def test_propagate_kepler_matches_golden(orbit: dict) -> None:
    # The core's analytic reference propagator against the independently
    # generated Python checkpoints; tolerances per the golden manifest.
    core = _core_or_fail()
    checked = 0
    for case in _load_cases("kepler_orbit.toml"):
        if not case["name"].startswith("checkpoint_"):
            continue
        state = core.propagate_kepler(
            orbit["mu"], orbit["r0"], orbit["v0"], _hex(case["t_s"])
        )
        r_gold = _vec3(case, "r_m")
        v_gold = _vec3(case, "v_mps")
        dr = math.dist(state["r_m"], r_gold)
        dv = math.dist(state["v_mps"], v_gold)
        assert dr < 1e-6, f"{case['name']}: position error {dr} m"
        assert dv < 1e-9, f"{case['name']}: velocity error {dv} m/s"
        checked += 1
    assert checked == 8


def test_rk4_kepler_convergence_slope(orbit: dict) -> None:
    # Phase 2 exit criterion 3, RK4 half: measured slope 4.0 +/- 0.2.
    core = _core_or_fail()
    pts = core.kepler_convergence(
        orbit["mu"], orbit["r0"], orbit["v0"], SPAN_S, "rk4", RK4_LADDER
    )
    slope = _loglog_slope(pts)
    assert 3.8 < slope < 4.2, f"RK4 slope {slope}; points {pts}"


def test_rkf78_fixed_kepler_convergence_slope(orbit: dict) -> None:
    # Phase 2 exit criterion 3, RKF7(8) half: fixed-step slope >= 7.5.
    core = _core_or_fail()
    pts = core.kepler_convergence(
        orbit["mu"], orbit["r0"], orbit["v0"], SPAN_S, "rkf78", RKF78_LADDER
    )
    slope = _loglog_slope(pts)
    assert slope >= 7.5, f"RKF7(8) slope {slope}; points {pts}"
    assert slope < 9.5, f"RKF7(8) slope {slope} implausibly high; points {pts}"


def test_rkf78_adaptive_energy_momentum_drift(orbit: dict) -> None:
    # Phase 2 exit criterion 4: energy and |h| drift < 1e-10 relative over
    # 10 orbits at adaptive tolerance 1e-12.
    core = _core_or_fail()
    drift = core.twobody_drift(
        orbit["mu"], orbit["r0"], orbit["v0"], 10.0, 1e-12, 1e-6, 1e-9, 10.0, 600.0
    )
    assert drift["max_energy_rel"] < 1e-10, drift
    assert drift["max_hmag_rel"] < 1e-10, drift
    assert drift["steps_accepted"] > 100, drift


def test_apsis_event_times_analytic(orbit: dict) -> None:
    # Phase 2 exit criterion 5: apsis events within 1 us of analytic times.
    core = _core_or_fail()
    cases = _load_cases("apsis_times.toml")
    t_end = _hex(_find_case(cases, "span")["t_end_s"])
    expected = [
        (_hex(c["t_s"]), c["kind"])
        for c in cases
        if c["name"].startswith("apsis_")
    ]
    hits = core.apsis_events(
        orbit["mu"], orbit["r0"], orbit["v0"], t_end, 1e-12, 1e-6, 1e-9, 5.0, 5.0, 1e-9
    )
    assert len(hits) == len(expected) == 6
    worst_s = 0.0
    for hit, (t_gold, kind_gold) in zip(hits, expected):
        assert hit["kind"] == kind_gold
        err_s = abs(hit["t_s"] - t_gold)
        assert err_s < 1e-6, f"apsis at {t_gold}: error {err_s} s"
        worst_s = max(worst_s, err_s)
    print(f"worst apsis-event time error: {worst_s * 1e6:.6f} us")


def test_dense_output_hermite_midstep_accuracy(orbit: dict) -> None:
    # Dense-output order evidence: midstep error within the a-priori bound
    # (h^4/384) max|r''''| and scaling as h^4 (see ch:integrators).
    core = _core_or_fail()
    err64 = core.hermite_midstep_max_err(
        orbit["mu"], orbit["r0"], orbit["v0"], SPAN_S, 64.0
    )["max_err_m"]
    err32 = core.hermite_midstep_max_err(
        orbit["mu"], orbit["r0"], orbit["v0"], SPAN_S, 32.0
    )["max_err_m"]
    bound64 = 64.0**4 / 384.0 * orbit["max_r4"]
    bound32 = 32.0**4 / 384.0 * orbit["max_r4"]
    assert 0.15 * bound64 < err64 < 1.5 * bound64, (err64, bound64)
    assert 0.15 * bound32 < err32 < 1.5 * bound32, (err32, bound32)
    assert 10.0 < err64 / err32 < 24.0, (err64, err32)
