"""Anderson-Darling goodness-of-fit for a fully specified CDF (FR-22 layer 6).

The Monte Carlo regression gate (``star_reacher.mc_regression``) needs a
distributional test that is sensitive in the tails, where a chi-square mean or
variance statistic is weak: a regression that shifts an ensemble's tail while
leaving its mean and variance intact is exactly the defect a distribution test
catches and a moment test does not. Anderson-Darling supplies it, and it is
implemented here from first principles for the same reason chi2.py is -- the
D-12 runtime allowed-list (numpy, matplotlib, jplephem) excludes SciPy, so its
A-D routine is not available at runtime.

Statistic. For a sample of size n against a fully specified continuous CDF F
(no parameters estimated from the sample), with the sample sorted ascending as
x_(1) <= ... <= x_(n),

    A2 = -n - (1/n) sum_{i=1..n} (2i-1) [ ln F(x_(i)) + ln(1 - F(x_(n+1-i))) ]

(Anderson & Darling, "A Test of Goodness of Fit", J. Amer. Statist. Assoc.
49(268), 1954). This module evaluates A2 and, unlike a critical-value table,
returns a computable p-value, so the regression gate can be set at an exact
probability (99 %, i.e. reject when p < 0.01) rather than interpolated off a
printed table.

P-value. The asymptotic distribution of A2 under the null and its finite-n
correction follow Marsaglia & Marsaglia, "Evaluating the Anderson-Darling
Distribution", Journal of Statistical Software 9(2), 2004
(doi:10.18637/jss.v009.i02). Their ``adinf(z)`` is a piecewise rational/series
approximation of the limiting CDF P(A2 < z) accurate to about 2e-6 absolute,
and ``errfix(n, z)`` is the leading finite-sample correction, so

    AD(n, z) = adinf(z) + errfix(n, adinf(z))

is the finite-n CDF and the p-value is 1 - AD(n, A2). The two functions are
transcribed verbatim from that paper (Section 3, the C listings for ``adinf``,
``errfix``, and ``AD``); the module's tests anchor ``adinf`` against the four
values tabulated in the paper's Table 1.

Domain handling. n < 2 is refused (A2 is undefined for a single point). F is
evaluated at every sample, and a value F(x) that lands exactly at 0 or 1 -- a
sample at or beyond the support edge, or a tie that the CDF maps to the
boundary -- would make ln F or ln(1-F) diverge; those are clamped to
``[_F_CLAMP, 1 - _F_CLAMP]`` with ``_F_CLAMP = 1e-300``, far below any
resolvable probability yet finite, so a boundary sample contributes a large but
finite term instead of an inf. The clamp is documented rather than silent
because it is the one place the statistic departs from the exact definition.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

__all__ = [
    "AndersonDarlingError",
    "ad_cdf",
    "adinf",
    "anderson_darling",
    "anderson_darling_uniform",
    "errfix",
]

# Smallest strictly-positive clamp for F(x): ln(1e-300) ~ -690.8 is finite and
# far below any probability a finite ensemble can resolve, so a sample at the
# support edge contributes a large but finite term rather than a -inf.
_F_CLAMP = 1e-300


class AndersonDarlingError(ValueError):
    """An Anderson-Darling input error (too few points, or a non-finite CDF value)."""


def _adinf_series(z: float) -> float:
    # Marsaglia & Marsaglia (2004), adinf, small-z branch (z < 2): a rational
    # prefactor times an eight-term series in z. Transcribed verbatim.
    return (
        math.exp(-1.2337141 / z)
        / math.sqrt(z)
        * (
            2.00012
            + (
                0.247105
                - (
                    0.0649821
                    - (0.0347962 - (0.011672 - 0.00168691 * z) * z) * z
                )
                * z
            )
            * z
        )
    )


def _adinf_tail(z: float) -> float:
    # Marsaglia & Marsaglia (2004), adinf, large-z branch (z >= 2): a six-term
    # series in z. Transcribed verbatim.
    return math.exp(
        -math.exp(
            1.0776
            - (
                2.30695
                - (
                    0.43424
                    - (0.082433 - (0.008056 - 0.0003146 * z) * z) * z
                )
                * z
            )
            * z
        )
    )


def adinf(z: float) -> float:
    """Limiting Anderson-Darling CDF P(A2 < z) as n -> infinity.

    Marsaglia & Marsaglia (2004) ``adinf``: the two-branch approximation of
    the asymptotic A2 distribution, accurate to about 2e-6 absolute over
    z > 0. Returns 0.0 at z <= 0 (A2 is nonnegative). The module tests anchor
    it against the four values tabulated in that paper.
    """
    if not math.isfinite(z):
        raise AndersonDarlingError(f"adinf requires finite z, got z={z!r}")
    if z <= 0.0:
        return 0.0
    if z < 2.0:
        return _adinf_series(z)
    return _adinf_tail(z)


def errfix(n: int, z: float) -> float:
    """Finite-sample correction AD(n, z) - adinf(z), Marsaglia & Marsaglia (2004).

    ``z`` is the *limiting* CDF value adinf(A2), not A2 itself, and ``n`` the
    sample size; the returned correction is added to adinf to obtain the
    finite-n CDF. The three g-functions and the c(n) breakpoint are transcribed
    verbatim from the paper's ``errfix`` listing.
    """
    if n < 1:
        raise AndersonDarlingError(f"errfix requires n >= 1, got n={n!r}")
    c = 0.01265 + 0.1757 / n
    if z < c:
        t = z / c
        g = t * math.sqrt(t) * (1.0 - t) * (49.0 * t - 102.0)
        return g * (0.0037 / (n * n) + 0.00078 / n + 0.00006) / n
    if z < 0.8:
        t = (z - c) / (0.8 - c)
        g = (
            -0.00022633
            + (6.54034 - (14.6538 - (14.458 - (8.259 - 1.91864 * t) * t) * t) * t)
            * t
        )
        return g * (0.04213 / n + 0.01365 / (n * n)) / n
    # z >= 0.8: the third g-branch, no explicit n^-2 term.
    g = (
        -130.2137
        + (
            745.2337
            - (1705.091 - (1950.646 - (1116.360 - 255.7844 * z) * z) * z) * z
        )
        * z
    )
    return g / n


def ad_cdf(a2: float, n: int) -> float:
    """Finite-n Anderson-Darling CDF AD(n, A2) = adinf(A2) + errfix(n, adinf(A2)).

    Marsaglia & Marsaglia (2004) ``AD``. Clamped to [0, 1]: the correction is
    an approximation and can push the sum a few ulp outside the unit interval
    at extreme A2, which a probability must never report.
    """
    if n < 1:
        raise AndersonDarlingError(f"ad_cdf requires n >= 1, got n={n!r}")
    x = adinf(a2)
    value = x + errfix(n, x)
    return min(1.0, max(0.0, value))


def _a2_statistic(sorted_f: Sequence[float]) -> float:
    # A2 from CDF values already sorted ascending and clamped off the {0, 1}
    # boundary. Pairs the i-th smallest with the i-th largest per the closed
    # form, so one pass computes the whole sum.
    n = len(sorted_f)
    total = 0.0
    for i in range(n):
        f_low = sorted_f[i]
        f_high = sorted_f[n - 1 - i]
        # 1 - f_high can round to exactly 0 for f_high within one ulp of 1
        # (the clamp on f itself cannot prevent this, since 1 - 1e-300 == 1.0
        # in binary64); clamp the complement here so ln stays finite.
        complement = max(1.0 - f_high, _F_CLAMP)
        total += (2 * (i + 1) - 1) * (math.log(f_low) + math.log(complement))
    return -n - total / n


def anderson_darling(
    samples: Sequence[float], cdf: Callable[[float], float]
) -> tuple[float, float]:
    """Anderson-Darling A2 and p-value of ``samples`` against a specified CDF.

    ``cdf`` is a fully specified continuous CDF (no parameters estimated from
    the sample); it is evaluated at each sample and must return a value in
    [0, 1]. Returns ``(A2, p_value)`` with ``p_value = 1 - AD(n, A2)`` from the
    Marsaglia & Marsaglia (2004) finite-n distribution, so a 99 % gate rejects
    when ``p_value < 0.01``.

    Raises :class:`AndersonDarlingError` for fewer than two samples or a
    non-finite CDF value. A CDF value at the {0, 1} boundary is clamped to
    ``[1e-300, 1 - 1e-300]`` (documented in the module docstring), so a sample
    at the support edge contributes a finite term.
    """
    values = [float(x) for x in samples]
    n = len(values)
    if n < 2:
        raise AndersonDarlingError(
            f"Anderson-Darling needs at least two samples, got {n}"
        )
    f_values = []
    for x in sorted(values):
        f = float(cdf(x))
        if not math.isfinite(f):
            raise AndersonDarlingError(
                f"the CDF returned a non-finite value {f!r} at sample {x!r}"
            )
        # Clamp off the open-interval boundary so ln F and ln(1 - F) stay
        # finite; the clamp is far below any resolvable probability.
        f_values.append(min(1.0 - _F_CLAMP, max(_F_CLAMP, f)))
    a2 = _a2_statistic(f_values)
    return a2, 1.0 - ad_cdf(a2, n)


def anderson_darling_uniform(samples: Sequence[float]) -> tuple[float, float]:
    """Anderson-Darling A2 and p-value against the standard uniform U(0, 1).

    The specialization of :func:`anderson_darling` with F(x) = x, the CDF the
    probability integral transform maps any continuous variate onto: pushing a
    metric through its own reference CDF yields U(0, 1) samples under the null,
    which this then tests. Samples are expected in [0, 1]; values landing
    exactly at 0 or 1 are clamped as in :func:`anderson_darling`.
    """
    return anderson_darling(samples, lambda x: x)
