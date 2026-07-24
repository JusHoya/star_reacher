"""Anderson-Darling A2 statistic and its p-value (FR-22 layer 6).

The distribution routines are anchored three independent ways, the same
discipline test_consistency.py applies to the chi-square quantile:

- the four adinf values tabulated in Marsaglia & Marsaglia (2004) Table 1;
- the classic asymptotic A2 critical values a printed A-D table lists
  (2.492 at the 5 % point, 3.878 at the 1 % point), recovered here as the
  z with adinf(z) equal to 0.95 and 0.99;
- the closed-form A2 of a tiny hand-computable sample, so the statistic
  itself (not only its tail probability) is pinned.

A calibration test then confirms the p-value is uniform under the null and a
power test confirms it collapses under a shifted alternative, which together
establish that the gate this feeds can both pass a good sample and reject a
bad one.
"""

import math

import numpy as np
import pytest
from numpy.random import PCG64, Generator

from star_reacher.anderson import (
    AndersonDarlingError,
    ad_cdf,
    adinf,
    anderson_darling,
    anderson_darling_uniform,
    errfix,
)

# Marsaglia & Marsaglia, "Evaluating the Anderson-Darling Distribution",
# J. Stat. Soft. 9(2), 2004, Table 1: the limiting CDF adinf(z) at z = 1, 2,
# 3, 4. These are the paper's own published anchor values, printed to four
# decimals; the implementation reproduces them to that precision.
_ADINF_TABLE1 = [
    (1.0, 0.6427),
    (2.0, 0.9082),
    (3.0, 0.9726),
    (4.0, 0.9913),
]


@pytest.mark.parametrize("z, published", _ADINF_TABLE1)
def test_adinf_matches_marsaglia_table1(z, published):
    assert abs(adinf(z) - published) <= 5e-5


# The classic Anderson-Darling asymptotic critical values for a fully
# specified continuous distribution: A2 = 2.492 is the 5 % point and
# A2 = 3.878 the 1 % point (Anderson & Darling 1954; reproduced in every A-D
# table, e.g. Stephens 1974). adinf must therefore cross 0.95 and 0.99 at
# those abscissae; recovering them by inversion cross-checks the paper's
# series against an independent published source.
@pytest.mark.parametrize(
    "prob, published_crit, tol",
    [(0.95, 2.492, 1e-3), (0.99, 3.878, 1e-3)],
)
def test_adinf_inverse_hits_published_critical_values(prob, published_crit, tol):
    lo, hi = 0.5, 8.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if adinf(mid) < prob:
            lo = mid
        else:
            hi = mid
    assert abs(mid - published_crit) <= tol


def test_errfix_vanishes_as_n_grows():
    # The finite-n correction is O(1/n); at large n the finite-sample CDF must
    # collapse onto adinf. Checked at a mid-distribution z.
    x = adinf(1.5)
    assert abs(errfix(10, x)) > abs(errfix(1000, x))
    assert abs(errfix(100000, x)) < 1e-6


def test_ad_cdf_is_monotone_and_bounded():
    # A CDF: nondecreasing in A2 and inside [0, 1] across the whole range.
    prev = -1.0
    for a2 in np.linspace(0.01, 12.0, 200):
        v = ad_cdf(float(a2), 50)
        assert 0.0 <= v <= 1.0
        assert v >= prev - 1e-12
        prev = v


def test_a2_closed_form_on_a_hand_sample():
    """A2 against U(0,1) for the sample {0.1, 0.5, 0.9}, computed by hand.

    With n = 3 and sorted F-values u = [0.1, 0.5, 0.9], the closed form
    A2 = -n - (1/n) sum_i (2i-1)[ln u_i + ln(1 - u_{n+1-i})] is a fixed
    number this recomputes independently, pinning the statistic itself and
    not only its tail probability.
    """
    u = [0.1, 0.5, 0.9]
    n = 3
    total = 0.0
    for i in range(n):
        total += (2 * (i + 1) - 1) * (
            math.log(u[i]) + math.log1p(-u[n - 1 - i])
        )
    expected = -n - total / n
    a2, _p = anderson_darling_uniform(u)
    assert abs(a2 - expected) <= 1e-12


def test_uniform_null_pvalue_is_calibrated():
    """Under the null the p-value is ~U(0,1): its rejection rate matches alpha.

    A seeded batch of exact-uniform samples is A-D tested; the fraction with
    p < 0.01 must sit near 1 %. This is the property the 99 % gate rests on --
    a correct sample is rejected about 1 % of the time, not more.
    """
    rng = Generator(PCG64(20260723))
    trials = 4000
    n = 64
    rejects = 0
    for _ in range(trials):
        sample = rng.random(n)
        _a2, p = anderson_darling_uniform(sample)
        if p < 0.01:
            rejects += 1
    rate = rejects / trials
    # Binomial(4000, 0.01) has sd ~ 0.0016; a +-0.007 window is > 4 sd and
    # still tight enough to catch a mis-scaled distribution.
    assert abs(rate - 0.01) < 0.007


def test_shifted_alternative_is_rejected():
    """A clearly non-uniform sample drives the p-value below the gate.

    Squaring U(0,1) draws concentrates them toward 0, a gross departure from
    uniformity; the A-D p-value must fall under 0.01 with high probability,
    demonstrating the statistic's power (not only its calibration).
    """
    rng = Generator(PCG64(1234))
    trials = 200
    n = 64
    rejects = 0
    for _ in range(trials):
        sample = rng.random(n) ** 2  # CDF F(x) = sqrt(x), not the tested x
        _a2, p = anderson_darling_uniform(sample)
        if p < 0.01:
            rejects += 1
    assert rejects / trials > 0.9


def test_general_cdf_against_standard_normal():
    """anderson_darling with an explicit normal CDF accepts normal data.

    Draws from N(0,1) tested against the standard-normal CDF (built from
    math.erf, no SciPy) must not be rejected at 1 %; the same draws tested as
    if they were U(0,1) must be, confirming the general-CDF path uses the
    supplied F rather than an implicit uniform.
    """
    rng = Generator(PCG64(99))
    sample = rng.standard_normal(128)
    normal_cdf = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    _a2, p_normal = anderson_darling(sample, normal_cdf)
    assert p_normal > 0.01
    # Clamp the same draws into [0,1] and test against U(0,1): a normal sample
    # is emphatically not uniform, so this must reject.
    clamped = np.clip((sample + 5.0) / 10.0, 0.0, 1.0)
    _a2u, p_uniform = anderson_darling_uniform(clamped)
    assert p_uniform < 0.01


def test_too_few_samples_raises():
    with pytest.raises(AndersonDarlingError):
        anderson_darling_uniform([0.5])


def test_boundary_samples_are_clamped_not_infinite():
    # A sample exactly at the support edge would make ln F or ln(1-F) diverge;
    # the documented clamp keeps A2 finite.
    a2, p = anderson_darling_uniform([0.0, 0.5, 1.0])
    assert math.isfinite(a2)
    assert 0.0 <= p <= 1.0


def test_non_finite_cdf_value_raises():
    with pytest.raises(AndersonDarlingError):
        anderson_darling([0.1, 0.2, 0.3], lambda x: float("nan"))
