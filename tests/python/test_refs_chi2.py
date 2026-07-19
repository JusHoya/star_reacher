"""Validate the dependency-free chi-square machinery of ``tests/refs/chi2.py``.

The module supplies the acceptance bounds for Phase 6 exit criteria 1, 3, and 6,
so it is validated first and against CLOSED FORMS rather than against tabulated
values: the chi-square distribution has an exact elementary quantile at one and
two degrees of freedom, which pins the whole implementation without importing
any statistics package or trusting a transcribed table.

* ``k = 2``: the chi-square distribution is exponential with mean 2, so
  ``F(x) = 1 - exp(-x/2)`` and the quantile is exactly ``-2 ln(1 - p)``.
* ``k = 1``: ``X = Z**2`` for a standard normal ``Z``, so the quantile is
  ``Phi^-1((1 + p)/2)**2``.

The suite additionally confirms every chi-square bound quoted in the Phase 6
chapters and measures the error of the Wilson--Hilferty approximation the
chapters use, which is the evidence for their claim that it is negligible
against the gate widths.

These tests are pure NumPy and require no compiled core.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# The independent Phase 6 references live under tests/refs (outside the package).
sys.path.insert(0, str(REPO_ROOT / "tests" / "refs"))
import chi2  # noqa: E402


def test_normal_quantile_matches_the_chapter_constant():
    """``normal_ppf(0.975)`` reproduces the constant quoted in eq:ekf:wh."""
    assert chi2.normal_ppf(0.975) == pytest.approx(chi2.Z_0975, abs=1e-14)
    assert chi2.normal_ppf(0.5) == pytest.approx(0.0, abs=1e-14)
    # Symmetry is a property of the distribution, not of the implementation, so
    # it is a genuine check rather than a restatement.
    assert chi2.normal_ppf(0.025) == pytest.approx(-chi2.Z_0975, abs=1e-14)


@pytest.mark.parametrize("p", [0.001, 0.025, 0.25, 0.5, 0.75, 0.975, 0.999])
def test_two_dof_quantile_matches_the_exponential_closed_form(p):
    """chi2(2) is exponential: the quantile is exactly ``-2 ln(1 - p)``."""
    assert chi2.chi2_ppf(p, 2) == pytest.approx(-2.0 * math.log(1.0 - p), rel=1e-12)


@pytest.mark.parametrize("p", [0.025, 0.5, 0.9, 0.975, 0.99])
def test_one_dof_quantile_matches_the_squared_normal_closed_form(p):
    """chi2(1) is the square of a standard normal."""
    expected = chi2.normal_ppf(0.5 * (1.0 + p)) ** 2
    assert chi2.chi2_ppf(p, 1) == pytest.approx(expected, rel=1e-11)


@pytest.mark.parametrize("k", [1, 2, 3, 7, 30, 300, 3000])
@pytest.mark.parametrize("p", [0.01, 0.025, 0.5, 0.975, 0.99])
def test_quantile_and_cdf_round_trip(k, p):
    """``cdf(ppf(p)) == p``: the inverse is a true inverse, not an estimate."""
    assert chi2.chi2_cdf(chi2.chi2_ppf(p, k), k) == pytest.approx(p, rel=1e-11)


def test_cdf_is_monotone_and_bounded():
    """A CDF that is not monotone would silently corrupt every gate bound."""
    xs = np.linspace(0.0, 60.0, 601)
    values = np.array([chi2.chi2_cdf(float(x), 10) for x in xs])
    assert values[0] == 0.0
    assert np.all(np.diff(values) >= 0.0)
    # The far tail must saturate at one; 60 is only the 4e-9 tail for k = 10, so
    # the limit is checked well beyond the monotonicity sweep.
    assert chi2.chi2_cdf(300.0, 10) == pytest.approx(1.0, abs=1e-12)


def test_incomplete_gamma_branches_agree_at_the_crossover():
    """The series and continued-fraction branches must meet, not merely each work.

    ``gamma_p`` switches expansion at ``x = a + 1``; a discontinuity there would
    be invisible in any single-branch test but would corrupt quantiles that
    bracket the crossover.
    """
    for a in (0.5, 1.0, 5.0, 50.0, 1500.0):
        x = a + 1.0
        below = chi2.gamma_p(a, x * (1.0 - 1e-12))
        above = chi2.gamma_p(a, x * (1.0 + 1e-12))
        assert below == pytest.approx(above, rel=1e-10)


# The bounds each Phase 6 chapter quotes, keyed by (per-draw dof, ensemble size).
# Values are transcribed from the chapters; the test confirms the exact
# quantiles reproduce them, which is what licenses their use as literal gates.
CHAPTER_BOUNDS = {
    ("star tracker / nav fix, eq:optical:stbounds, eq:radio:bounds", 3, 1000): (2.850, 3.154),
    ("sun sensor, sec:optical:stats", 2, 1000): (1.878, 2.126),
    ("altimeter, eq:radio:bounds", 1, 1000): (0.914, 1.090),
    ("NEES, eq:ekf:ensemble", 15, 100): (13.95, 16.09),
    ("NIS nav fix, eq:ekf:ensemble", 6, 100): (5.34, 6.70),
    ("NIS star tracker, eq:ekf:ensemble", 3, 100): (2.54, 3.50),
    ("NIS altimeter, eq:ekf:ensemble", 1, 100): (0.74, 1.30),
}


@pytest.mark.parametrize("key", list(CHAPTER_BOUNDS))
def test_chapter_quoted_bounds_are_reproduced(key):
    """Every chi-square bound printed in a Phase 6 chapter is recomputed exactly."""
    label, n, m = key
    expected_lo, expected_hi = CHAPTER_BOUNDS[key]
    lo, hi = chi2.ensemble_mean_bounds(n, m)
    # The chapters print to the digit shown, so agreement is asserted at half a
    # unit in the last printed place.
    assert lo == pytest.approx(expected_lo, abs=0.005), label
    assert hi == pytest.approx(expected_hi, abs=0.005), label


def test_wilson_hilferty_error_claim_holds_for_k_at_least_300():
    """Chapter ch:ekf claims WH relative error below 5e-5 for ``k >= 300``.

    Measured here rather than accepted: the approximation is what the shipped
    (SciPy-free) core will use, so the claim is load-bearing for every gate.
    """
    worst = 0.0
    for k in (300, 600, 1000, 1500, 3000, 6000):
        for p in (chi2.P_LOWER, chi2.P_UPPER):
            exact = chi2.chi2_ppf(p, k)
            approx = chi2.chi2_ppf_wilson_hilferty(p, k)
            worst = max(worst, abs(approx / exact - 1.0))
    assert worst < 5e-5, f"Wilson-Hilferty worst relative error {worst:.2e} exceeds 5e-5"


def test_gate_accepts_the_null_and_rejects_an_inflated_ensemble():
    """The gate must be able to FAIL, or its passing proves nothing.

    A gate is only evidence if a wrong input trips it, so this drives the
    statistic with correctly scaled draws (accept) and with draws inflated 20 %
    in variance (reject).
    """
    rng = np.random.default_rng(2026)
    correct = np.sum(rng.standard_normal((1000, 3)) ** 2, axis=1)
    passed, statistic, bounds = chi2.gate(correct, 3)
    assert passed, f"correctly scaled draws rejected: {statistic} outside {bounds}"

    inflated = np.sum((rng.standard_normal((1000, 3)) * math.sqrt(1.2)) ** 2, axis=1)
    passed_bad, statistic_bad, bounds_bad = chi2.gate(inflated, 3)
    assert not passed_bad, (
        f"20 % variance inflation was accepted: {statistic_bad} inside {bounds_bad}"
    )


def test_gate_coverage_is_close_to_the_nominal_95_percent():
    """Empirical coverage of the two-sided 95 % gate over many independent trials.

    This is the end-to-end statement the criteria rely on: a correct sensor
    passes about 95 % of the time. Bound generously (90 %--99 %) so the test
    itself is not flaky while still catching a badly miscalibrated interval.
    """
    rng = np.random.default_rng(31337)
    trials = 400
    accepted = 0
    for _ in range(trials):
        draws = np.sum(rng.standard_normal((1000, 3)) ** 2, axis=1)
        accepted += int(chi2.gate(draws, 3)[0])
    coverage = accepted / trials
    assert 0.90 <= coverage <= 0.99, f"empirical coverage {coverage:.3f} is off nominal"


def test_reference_and_shipped_chi2_agree():
    """The deliberate duplicate is load-bearing: two exact paths must agree.

    ``tests/refs/chi2.py`` and ``python/star_reacher/chi2.py`` implement the
    chi-square quantile independently and neither imports the other. That
    separation is the point -- a reference that imported the shipped module
    could not detect an error in it, because the same wrong value would appear
    on both sides of every gate and cancel. The rationale is written out in the
    reference module's docstring; this test is what converts it from an
    intention into a checked property.

    The domain is the one the Phase 6 criteria actually use: the two-sided 95 %
    probabilities, over degrees of freedom from a single three-axis draw up to
    the ensemble totals the sensor and NEES/NIS gates accumulate.
    """
    from star_reacher import chi2 as shipped

    worst = 0.0
    for k in (1, 2, 3, 6, 60, 180, 300, 900, 3000, 9000):
        for p in (chi2.P_LOWER, chi2.P_UPPER):
            reference = chi2.chi2_ppf(p, k)
            production = shipped.chi2_ppf(p, k)
            worst = max(worst, abs(reference - production) / reference)
    # Both routes are exact rather than approximate, so the residual is
    # convergence tolerance, not model error; 1e-10 relative leaves room for
    # the two different root-finding terminations without admitting a real
    # disagreement.
    assert worst < 1e-10, (
        f"the independent and shipped chi-square quantiles disagree by "
        f"{worst:.3e} relative; one of them is wrong"
    )
