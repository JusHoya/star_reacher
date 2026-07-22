"""Chi-square and binomial distribution functions from first principles
(FR-26, D-12).

``star consistency`` gates NEES/NIS statistics against two-sided chi-square
bounds, and the D-12 runtime allowed-list (numpy, matplotlib, jplephem)
excludes SciPy, so the chi-square quantile function is implemented here on
top of the regularized lower incomplete gamma function:

    chi2_cdf(x, k) = P(k/2, x/2)
    chi2_ppf(p, k) = the x with chi2_cdf(x, k) = p

``P(a, x)`` follows the classic split of Numerical Recipes section 6.2
(Press, Teukolsky, Vetterling, Flannery, 3rd ed., Cambridge, 2007; routines
``gser``/``gcf``): the power series of the lower incomplete gamma for
x < a + 1 and the modified-Lentz continued fraction of the upper incomplete
gamma for x >= a + 1, both scaled by exp(-x + a ln x - lnGamma(a)) with
lnGamma from ``math.lgamma``. Near x ~ a both routes need O(sqrt(a)) terms,
so the iteration cap scales with sqrt(a) instead of Numerical Recipes' fixed
100: the same exact code path then covers the k ~ 1e6 degrees of freedom
that pooled ensemble NEES statistics reach (R runs x T epochs x n states).

The inverse seeds from the Wilson-Hilferty cube-root normal approximation
(Abramowitz & Stegun, Handbook of Mathematical Functions, eq. 26.4.17, with
the normal quantile from the A&S eq. 26.2.23 rational approximation) and
polishes with a bracketed Newton iteration on the CDF, falling back to
bisection whenever a Newton step would leave the bracket. Because the split
at x = a + 1 always evaluates the smaller of P and Q for tail quantiles, the
CDF's relative rounding error stays near 1e-9 even at k = 1e6, and the
returned quantile is accurate to better than 1e-10 relative in x across
k in [1, 1e6]. That claim is test-anchored in
``tests/python/test_consistency.py`` against published table values, closed
forms at k in {1, 2, 4}, CDF round-trips, and a Wilson-Hilferty large-k
cross-check.

The binomial CDF at the bottom of this module serves the second half of the
FR-26 gate: the ensemble coverage criterion counts how many epochs fall
inside their chi-square interval, and that count is Binomial(n_epochs, q)
under the consistency hypothesis (``consistency.inside_count_threshold``
turns it into an acceptance threshold). It is evaluated by exact summation
of the smaller tail rather than through an incomplete-beta relation: the
epoch counts involved are in the hundreds, so an O(n) sum of log-gamma
probability terms is both cheaper and easier to audit than a continued
fraction, and it is exact to rounding.
"""

from __future__ import annotations

import math

__all__ = [
    "binom_cdf",
    "chi2_cdf",
    "chi2_ppf",
    "gammp",
    "normal_ppf",
    "wilson_hilferty_ppf",
]

_EPS = 2.220446049250313e-16  # IEEE-754 double machine epsilon
# Guard value against zero denominators in the modified Lentz recurrence
# (Numerical Recipes 6.2, FPMIN): tiny but far from the subnormal range.
_FPMIN = 1e-300


def _iter_cap(a: float) -> int:
    # Both the series and the continued fraction need O(sqrt(a)) terms near
    # x ~ a (Numerical Recipes 6.2 notes this as the reason its 3rd edition
    # switches to quadrature for large a); a sqrt-scaled cap keeps the exact
    # series/fraction route convergent up to the k ~ 1e6 pooled-gate scale.
    return int(20.0 * math.sqrt(a)) + 200


def _log_prefactor(a: float, x: float) -> float:
    # ln of the common scale factor x^a e^-x / Gamma(a) shared by the series
    # (NR gser) and continued fraction (NR gcf) forms.
    return a * math.log(x) - x - math.lgamma(a)


def _gser(a: float, x: float) -> float:
    """Lower-tail series for P(a, x), valid for x < a + 1 (NR 6.2 gser)."""
    ap = a
    term = 1.0 / a
    total = term
    for _ in range(_iter_cap(a)):
        ap += 1.0
        term *= x / ap
        total += term
        if abs(term) < abs(total) * _EPS:
            return total * math.exp(_log_prefactor(a, x))
    raise RuntimeError(f"gammp series did not converge for a={a!r}, x={x!r}")


def _gcf(a: float, x: float) -> float:
    """Upper-tail continued fraction for Q(a, x), x >= a + 1 (NR 6.2 gcf).

    Evaluated with the modified Lentz algorithm (NR 5.2), which replaces
    vanishing partial denominators with ``_FPMIN`` instead of dividing by
    zero.
    """
    b = x + 1.0 - a
    c = 1.0 / _FPMIN
    d = 1.0 / b
    h = d
    for i in range(1, _iter_cap(a) + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = b + an / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            return h * math.exp(_log_prefactor(a, x))
    raise RuntimeError(
        f"gammq continued fraction did not converge for a={a!r}, x={x!r}"
    )


def gammp(a: float, x: float) -> float:
    """Regularized lower incomplete gamma function P(a, x).

    P(a, x) = gamma(a, x) / Gamma(a), increasing from 0 at x = 0 to 1 as
    x -> inf. Series/continued-fraction split per Numerical Recipes 6.2.
    Raises ``ValueError`` for a <= 0, x < 0, or non-finite arguments.
    """
    if not (math.isfinite(a) and a > 0.0):
        raise ValueError(f"gammp requires a > 0, got a={a!r}")
    if not (math.isfinite(x) and x >= 0.0):
        raise ValueError(f"gammp requires x >= 0, got x={x!r}")
    if x == 0.0:
        return 0.0
    if x < a + 1.0:
        return _gser(a, x)
    return 1.0 - _gcf(a, x)


def _check_dof(k: float) -> None:
    if not (math.isfinite(k) and k > 0.0):
        raise ValueError(f"chi-square degrees of freedom must be > 0, got k={k!r}")


def chi2_cdf(x: float, k: float) -> float:
    """Chi-square CDF with k degrees of freedom: P(k/2, x/2).

    ``x <= 0`` returns 0.0 (the distribution's support starts at 0);
    non-finite ``x`` and ``k <= 0`` raise ``ValueError``.
    """
    _check_dof(k)
    if not math.isfinite(x):
        raise ValueError(f"chi2_cdf requires finite x, got x={x!r}")
    if x <= 0.0:
        return 0.0
    return gammp(0.5 * k, 0.5 * x)


def normal_ppf(p: float) -> float:
    """Standard normal quantile via A&S eq. 26.2.23 (|error| < 4.5e-4).

    Rational approximation from Abramowitz & Stegun, Handbook of
    Mathematical Functions, eq. 26.2.23 (Hastings). The absolute error
    bound of 4.5e-4 is sufficient for its two roles here: seeding the
    exactly-polished Newton iteration in ``chi2_ppf`` and the coarse
    Wilson-Hilferty cross-checks; it is not a high-precision quantile.
    """
    if not (0.0 < p < 1.0):
        raise ValueError(f"normal_ppf requires 0 < p < 1, got p={p!r}")
    q = min(p, 1.0 - p)
    t = math.sqrt(-2.0 * math.log(q))
    numerator = 2.515517 + t * (0.802853 + t * 0.010328)
    denominator = 1.0 + t * (1.432788 + t * (0.189269 + t * 0.001308))
    z = t - numerator / denominator
    return z if p >= 0.5 else -z


def wilson_hilferty_ppf(p: float, k: float) -> float:
    """Wilson-Hilferty approximate chi-square quantile (A&S eq. 26.4.17).

    chi2_p(k) ~= k * (1 - 2/(9k) + z_p sqrt(2/(9k)))^3 with z_p the standard
    normal quantile. Used as the Newton seed in ``chi2_ppf`` and as an
    independent transcription-error check on published table values in the
    test suite; can go nonpositive for small k and small p, which callers
    must handle.
    """
    _check_dof(k)
    z = normal_ppf(p)
    c = 2.0 / (9.0 * k)
    t = 1.0 - c + z * math.sqrt(c)
    return k * t * t * t


def chi2_ppf(p: float, k: float) -> float:
    """Chi-square inverse CDF (quantile function) with k degrees of freedom.

    Returns the x with ``chi2_cdf(x, k) == p``, accurate to better than
    1e-10 relative in x for k in [1, 1e6]. Bracketed Newton/bisection hybrid
    on the monotone CDF, seeded by Wilson-Hilferty; every iterate stays
    inside a maintained bracket, so convergence is guaranteed and a wrong
    root is impossible. Raises ``ValueError`` outside 0 < p < 1 or k <= 0.
    """
    _check_dof(k)
    if not (0.0 < p < 1.0):
        raise ValueError(f"chi2_ppf requires 0 < p < 1, got p={p!r}")
    a = 0.5 * k

    x = wilson_hilferty_ppf(p, k)
    if x <= 0.0:
        # Wilson-Hilferty can go nonpositive deep in the lower tail of small
        # k; the exact leading term P(a, x/2) ~ (x/2)^a / Gamma(a+1) inverts
        # in closed form and seeds that regime instead.
        x = 2.0 * math.exp((math.log(p) + math.lgamma(a + 1.0)) / a)

    # Establish a bracket [lo, hi] with cdf(lo) < p <= cdf(hi). lo = 0 is a
    # valid lower end (cdf 0 < p); hi doubles from the seed until it covers p.
    lo = 0.0
    hi = max(x, _FPMIN)
    for _ in range(2100):
        if chi2_cdf(hi, k) >= p:
            break
        lo = hi
        hi *= 2.0
    else:
        raise RuntimeError(f"chi2_ppf failed to bracket p={p!r}, k={k!r}")
    if not (lo < x < hi):
        x = 0.5 * (lo + hi)

    log_two = math.log(2.0)
    for _ in range(300):
        f = chi2_cdf(x, k) - p
        if f == 0.0:
            # Exact hit. Returning here matters: the sign update below would
            # make x a bracket endpoint, and the strict-inequality bracket
            # check would then bounce a zero-length Newton step to the
            # midpoint of a possibly still-wide bracket, walking away from a
            # converged root.
            return x
        if f > 0.0:
            hi = x
        else:
            lo = x
        # Newton step uses the log-density so extreme tails cannot overflow
        # or divide by an underflowed pdf; a step that leaves the bracket, or
        # an underflowed density, degrades to bisection.
        log_pdf = (a - 1.0) * math.log(x) - 0.5 * x - a * log_two - math.lgamma(a)
        if log_pdf > -690.0:
            x_next = x - f / math.exp(log_pdf)
            if x_next == x:
                # The correction is below one ulp of x: converged to the
                # working precision of the CDF.
                return x
            if not (lo < x_next < hi):
                x_next = 0.5 * (lo + hi)
        else:
            x_next = 0.5 * (lo + hi)
        # Two stop rules: a converged Newton step, or a bracket already
        # tighter than the CDF's own rounding noise allows to resolve
        # (~1e-11 relative at k ~ 1e6); both sit inside the 1e-10 claim.
        if abs(x_next - x) <= 1e-12 * x or (hi - lo) <= 2e-11 * x:
            return x_next
        x = x_next
    raise RuntimeError(f"chi2_ppf did not converge for p={p!r}, k={k!r}")


# Below exp(-745) a double underflows to zero, so a term that small cannot
# change the sum; skipping it also keeps math.exp from raising on -inf.
_LOG_UNDERFLOW = -745.0


def _log_binom_pmf(j: int, n: int, p: float) -> float:
    # log C(n, j) p^j (1-p)^(n-j), with the binomial coefficient through
    # lgamma so n in the thousands cannot overflow an intermediate factorial.
    return (
        math.lgamma(n + 1.0)
        - math.lgamma(j + 1.0)
        - math.lgamma(n - j + 1.0)
        + j * math.log(p)
        + (n - j) * math.log1p(-p)
    )


def binom_cdf(k: int, n: int, p: float) -> float:
    """Binomial CDF P(X <= k) for X ~ Binomial(n, p), by exact summation.

    Whichever tail is shorter is summed and the other is taken as its
    complement, so the returned value never loses the small tail to
    cancellation against 1.0 — the regime the coverage threshold is solved
    in, where the answer is O(1e-3). ``k < 0`` returns 0.0 and ``k >= n``
    returns 1.0. Raises ``ValueError`` for n < 0 or p outside (0, 1).
    """
    if n < 0:
        raise ValueError(f"binom_cdf requires n >= 0, got n={n!r}")
    if not (math.isfinite(p) and 0.0 < p < 1.0):
        raise ValueError(f"binom_cdf requires 0 < p < 1, got p={p!r}")
    k = int(k)
    n = int(n)
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0

    def tail(lo: int, hi: int) -> float:
        total = 0.0
        for j in range(lo, hi + 1):
            log_pmf = _log_binom_pmf(j, n, p)
            if log_pmf > _LOG_UNDERFLOW:
                total += math.exp(log_pmf)
        return total

    if k < n * p:
        return min(tail(0, k), 1.0)
    return max(0.0, 1.0 - tail(k + 1, n))
