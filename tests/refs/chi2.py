"""Dependency-free chi-square distribution functions for the Phase 6 gates.

This module is part of the Phase 6 independent-reference set: every routine here
was written from the math-library chapters and from published definitions, with
no reference to the C++ sensor or GNC sources it is used to check. See
``tests/refs/manifest.toml`` for the provenance of the whole set.

Chapters ``ch:sensors-optical`` (equation ``eq:optical:stbounds``),
``ch:sensors-radio`` (equation ``eq:radio:bounds``), and ``ch:ekf``
(equations ``eq:ekf:ensemble`` and ``eq:ekf:wh``) all gate an ensemble mean
against two-sided 95 % chi-square bounds. The chapters evaluate the quantiles
with the Wilson--Hilferty transformation because the project ships no SciPy
(decision D-12 restricts the runtime dependency set). An independent reference
should not inherit that approximation, so this module computes the quantiles
*exactly* -- by inverting the regularized lower incomplete gamma function -- and
additionally provides the Wilson--Hilferty form so the two can be compared. The
comparison is what licenses the chapters' approximation, rather than assuming
it.

Only the standard library and NumPy are used; SciPy is deliberately not imported
even where it is installed, so the reference remains usable under the project's
D-12 dependency allowed-list.

Definitions used, written out rather than cited by equation number so that no
citation is invented:

* The chi-square CDF with ``k`` degrees of freedom is the regularized lower
  incomplete gamma function, ``F(x; k) = P(k/2, x/2)``.
* ``P(a, x) = gamma_lower(a, x) / Gamma(a)`` is evaluated for ``x < a + 1`` by
  the everywhere-convergent series

      P(a, x) = exp(-x) x**a / Gamma(a) * sum_{n>=0} x**n / (a (a+1) ... (a+n)),

  and for ``x >= a + 1`` through its complement ``Q = 1 - P`` using Legendre's
  continued fraction

      Q(a, x) = exp(-x) x**a / Gamma(a)
                * 1/(x+1-a - 1*(1-a)/(x+3-a - 2*(2-a)/(x+5-a - ...))),

  evaluated by the modified Lentz recurrence. Both expansions are classical and
  are stated here in full; the pair is the standard way to obtain full double
  precision across the whole domain, because each expansion is used only in the
  half-domain where it converges rapidly.
* The quantile is obtained by bracketing bisection on the CDF, which cannot
  diverge on a monotone function and needs no derivative.

Relationship to ``python/star_reacher/chi2.py`` -- a DELIBERATE duplicate
-----------------------------------------------------------------------

The shipped package carries its own scipy-free exact chi-square module at
``python/star_reacher/chi2.py``, written for FR-26 (``star consistency``). This
module duplicates that capability on purpose and must not be collapsed into it.

The reason is circularity. ``python/star_reacher/chi2.py`` computes the bounds
the shipped NEES/NIS and sensor gates are judged against. A reference used to
CHECK those gates must not import the module that PRODUCES them: a wrong
quantile, a mistaken degrees-of-freedom convention, or an off-by-one in the
ensemble scaling would then appear identically on both sides of the comparison
and cancel, and the gate would pass on a wrong number while looking green. Two
implementations derived independently -- one by bracketing bisection on the
incomplete gamma function, one by the shipped module's own route -- agreeing to
a tight tolerance is real evidence about the special function; one
implementation compared against itself is none.

To keep the duplication honest rather than merely tolerated, the two are
actively cross-checked by
``test_refs_chi2.py::test_reference_and_shipped_chi2_agree`` across the domain
the Phase 6 gates use. A future change that makes either module wrong is caught
there. If that test is ever removed, this module has lost the property that
justifies its existence and the duplication should be revisited rather than
left standing.
"""

from __future__ import annotations

import math

import numpy as np

# Two-sided 95 % gate probabilities used by every Phase 6 chi-square criterion.
P_LOWER = 0.025
P_UPPER = 0.975

# The standard normal 0.975 quantile, quoted to full double precision by
# equation eq:ekf:wh. Reproduced independently by ``normal_ppf`` below; the
# agreement is asserted in the test suite rather than assumed here.
Z_0975 = 1.959963984540054

_MAX_ITER = 1000
_EPS = 1e-16


def _gamma_p_series(a: float, x: float) -> float:
    """Regularized lower incomplete gamma by its convergent power series."""
    # The series is used only for x < a + 1, where every term is positive and
    # the partial sums converge geometrically; there is no cancellation.
    total = 1.0 / a
    term = total
    ap = a
    for _ in range(_MAX_ITER):
        ap += 1.0
        term *= x / ap
        total += term
        if abs(term) < abs(total) * _EPS:
            break
    else:  # pragma: no cover - unreachable for the arguments this module uses
        raise RuntimeError(f"incomplete gamma series failed to converge: a={a}, x={x}")
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gamma_q_continued_fraction(a: float, x: float) -> float:
    """Regularized upper incomplete gamma by Legendre's continued fraction."""
    # Modified Lentz evaluation: the tiny floor keeps a zero denominator from
    # producing an infinity, which is the only failure mode of the recurrence.
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b if b != 0.0 else 1.0 / tiny
    h = d
    for i in range(1, _MAX_ITER):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    else:  # pragma: no cover - unreachable for the arguments this module uses
        raise RuntimeError(f"incomplete gamma fraction failed to converge: a={a}, x={x}")
    return h * math.exp(-x + a * math.log(x) - math.lgamma(a))


def gamma_p(a: float, x: float) -> float:
    """Regularized lower incomplete gamma function ``P(a, x)``."""
    if a <= 0.0:
        raise ValueError(f"shape parameter must be positive, got a={a}")
    if x < 0.0:
        raise ValueError(f"argument must be non-negative, got x={x}")
    if x == 0.0:
        return 0.0
    # The crossover at x = a + 1 is where the two expansions trade places as the
    # rapidly convergent one; either is correct on the other's side but slow.
    if x < a + 1.0:
        return _gamma_p_series(a, x)
    return 1.0 - _gamma_q_continued_fraction(a, x)


def chi2_cdf(x: float, k: float) -> float:
    """Chi-square CDF with ``k`` degrees of freedom."""
    if k <= 0.0:
        raise ValueError(f"degrees of freedom must be positive, got k={k}")
    if x <= 0.0:
        return 0.0
    return gamma_p(0.5 * k, 0.5 * x)


def chi2_ppf(p: float, k: float) -> float:
    """Exact chi-square quantile: the ``x`` with ``chi2_cdf(x, k) == p``.

    Bisection on the (strictly monotone) CDF. The bracket is grown from the
    Wilson--Hilferty estimate, which is accurate enough that the expansion loop
    terminates immediately for every argument this project uses; growing rather
    than assuming the bracket keeps the routine correct for small ``k`` where
    Wilson--Hilferty is poor.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"probability must lie in (0, 1), got p={p}")
    if k <= 0.0:
        raise ValueError(f"degrees of freedom must be positive, got k={k}")

    guess = max(chi2_ppf_wilson_hilferty(p, k), 1e-12)
    lo, hi = 0.5 * guess, 2.0 * guess
    while chi2_cdf(lo, k) > p:
        lo *= 0.5
        if lo < 1e-300:  # pragma: no cover - p below double-precision reach
            raise RuntimeError(f"cannot bracket quantile below: p={p}, k={k}")
    while chi2_cdf(hi, k) < p:
        hi *= 2.0
        if hi > 1e300:  # pragma: no cover - p above double-precision reach
            raise RuntimeError(f"cannot bracket quantile above: p={p}, k={k}")

    # 200 halvings drive the bracket to the last representable bit of any
    # double in the reachable range, so the loop is a fixed-cost operation.
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        if chi2_cdf(mid, k) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def chi2_ppf_wilson_hilferty(p: float, k: float) -> float:
    """Wilson--Hilferty chi-square quantile approximation, equation eq:ekf:wh.

    Reproduced here so the reference can measure the error the chapters accept,
    per Wilson and Hilferty (1931). This is NOT used to compute any gate bound
    in this module: ``chi2_ppf`` is exact.
    """
    if k <= 0.0:
        raise ValueError(f"degrees of freedom must be positive, got k={k}")
    z = normal_ppf(p)
    t = 2.0 / (9.0 * k)
    return k * (1.0 - t + z * math.sqrt(t)) ** 3


def normal_ppf(p: float) -> float:
    """Standard normal quantile, by bisection on ``math.erf``.

    ``Phi(z) = (1 + erf(z / sqrt(2))) / 2`` is the definition; inverting it by
    bisection avoids depending on an ``erfinv`` the standard library does not
    provide, and reaches full double precision in a fixed number of steps.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"probability must lie in (0, 1), got p={p}")
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        if 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0))) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def ensemble_mean_bounds(n: int, m: int) -> tuple[float, float]:
    """Two-sided 95 % acceptance bounds on an ensemble-mean chi-square statistic.

    Implements the common gate form of equations ``eq:optical:stbounds``,
    ``eq:radio:bounds``, and ``eq:ekf:ensemble``: for ``m`` independent draws of
    a chi2(n) statistic, the sum is chi2(n*m) exactly, so the mean of the ``m``
    draws is accepted when it lies in
    ``[chi2_ppf(0.025, n*m) / m, chi2_ppf(0.975, n*m) / m]``.

    Parameters
    ----------
    n : per-draw degrees of freedom (3 for a position fix, 1 for the altimeter).
    m : number of independent draws in the ensemble.
    """
    if n <= 0 or m <= 0:
        raise ValueError(f"n and m must be positive, got n={n}, m={m}")
    dof = float(n * m)
    return chi2_ppf(P_LOWER, dof) / m, chi2_ppf(P_UPPER, dof) / m


def ensemble_mean_statistic(quadratic_forms: np.ndarray) -> float:
    """Mean of a per-draw normalized quadratic form, the gated quantity."""
    values = np.asarray(quadratic_forms, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"expected a non-empty 1-D array, got shape {values.shape}")
    return float(values.mean())


def gate(quadratic_forms: np.ndarray, n: int) -> tuple[bool, float, tuple[float, float]]:
    """Evaluate the two-sided 95 % ensemble-mean gate.

    Returns ``(passed, statistic, (lower, upper))`` so a failing test can report
    the observed value against the bound it missed, per DX-5.
    """
    values = np.asarray(quadratic_forms, dtype=float)
    statistic = ensemble_mean_statistic(values)
    lower, upper = ensemble_mean_bounds(n, values.size)
    return bool(lower <= statistic <= upper), statistic, (lower, upper)
