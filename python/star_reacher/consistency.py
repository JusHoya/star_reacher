"""NEES/NIS filter-consistency statistics and chi-square gates (FR-26).

Array-level engine behind ``star consistency``: every public function takes
NumPy arrays and returns plain results, with no file I/O, so the same gates
serve the CLI, the Phase 6 EKF acceptance driver, and any later test
battery. All functions are deterministic (no random number generation).

Statistical conventions (normative; the theory derivation lives in the
math-library EKF chapter, and the definitions follow Bar-Shalom, Li &
Kirubarajan, "Estimation with Applications to Tracking and Navigation",
Wiley, 2001, ch. 5):

- **Per-epoch NEES.** For an estimation error e_k (dimension n) with
  reported covariance P_k, eps_k = e_k^T P_k^{-1} e_k. For a consistent
  filter (zero-mean Gaussian error with honest covariance),
  eps_k ~ chi-square(n). NIS is the same statistic on the innovation y_k
  with innovation covariance S_k, eps_k ~ chi-square(m) per update.
**Which statistic gates.** Exactly one family sets the verdict: the
ensemble statistic of ch:ekf eq:ekf:ensemble, evaluated by
``ensemble_gate``. The per-run time average and the pooled all-epoch mean
are computed and reported as *diagnostics* and are deliberately excluded
from the verdict, because the chi-square bounds they would be checked
against assume the per-epoch values are mutually independent, and within
one run they are not. The reasoning is set out in full in ch:ekf
sec:ekf:consistency and summarized here:

- **Per-run time-averaged diagnostic** (``time_average_gate``). With T
  epochs, the run mean eps_bar = (1/T) sum_k eps_k would satisfy
  T * eps_bar ~ chi-square(T n) *if* the eps_k were independent, giving
  the indicative interval

      [chi2_ppf((1-q)/2, T n) / T,  chi2_ppf((1+q)/2, T n) / T]

  at probability q (default 0.95). A filter's state error is a smooth,
  strongly serially correlated trajectory, so the independence premise
  fails and the interval is far narrower than the true sampling spread of
  eps_bar. The effect is large, not marginal: on the Phase 6
  exit-criterion-3 ensemble — provably consistent, with the ensemble
  statistic inside its bounds at every epoch — only 7 of 100 runs fall
  inside this interval. ``IntervalGate.passed`` is therefore an
  *indicative* flag for a human reader; no caller may use it to decide
  acceptance.
- **Pooled diagnostic** (reported inside ``ensemble_gate`` as ``pooled``).
  The mean over all R*T per-epoch values would satisfy
  R*T*mean ~ chi-square(R T n) under full independence, with interval

      [chi2_ppf((1-q)/2, R T n) / (R T),  chi2_ppf((1+q)/2, R T n) / (R T)].

  It inherits the same across-time correlation defect — at R*T*n degrees
  of freedom the interval is so narrow that the correlation-inflated
  spread dwarfs it — so it too is reported, never gated on.
- **Ensemble gate** (``ensemble_gate`` — the acceptance instrument, per
  the Bar-Shalom Monte Carlo convention, the FR-26 "per-run and ensemble"
  wording, and ch:ekf eq:ekf:ensemble). Over R independent runs the
  ensemble average at epoch k, eps_bar_k = (1/R) sum_r eps_(r,k),
  satisfies R * eps_bar_k ~ chi-square(R n) exactly under the consistency
  hypothesis — averaging is *across runs*, which are independent by
  construction, so no across-time premise is involved. The interval is

      [chi2_ppf((1-q)/2, R n) / R,  chi2_ppf((1+q)/2, R n) / R]

  and two complementary criteria are taken against it, both gating:

  1. **Headline** (``EnsembleGate.headline``): the epoch-average of
     eps_bar_k must lie inside that same interval. Averaging over epochs
     can only shrink the statistic's variance relative to a single epoch's
     (equality would need perfectly correlated epochs), so testing it
     against the single-epoch interval is *conservative under any
     across-time correlation whatsoever* — it never over-rejects because
     of serial correlation. This is the criterion the Phase 6 EKF driver
     gates on, and it is what catches a shifted mean: a mis-scaled
     covariance or a biased estimator moves eps_bar_k by a multiple of the
     interval width.
  2. **Coverage** (``EnsembleGate.coverage_passed``): the number of epochs
     whose eps_bar_k lies inside the interval must be at least
     ``inside_count_threshold`` (see below). This catches a defect
     confined to part of the run, which the epoch-average can dilute.

R = 1 is admitted: at one run the ensemble average degenerates to the
run's own eps_k, which is chi-square(n) at each epoch under consistency,
so both criteria remain exactly valid — merely low-powered. That keeps a
single-log invocation gated rather than unconditionally green.

**The coverage threshold, and why it is not "95 % of epochs"**
(``inside_count_threshold``). Under the consistency hypothesis each epoch
falls inside its two-sided interval with probability q = 0.95, so the count
of epochs inside is

    X ~ Binomial(n_epochs, q),      E[X] = q * n_epochs.

A rule of "at least 95 % of epochs inside" therefore tests X against its
own mean, and a correct filter passes it only about half the time — it is a
coin flip, not a gate. Measured on the exit-criterion-3 ensemble, the three
NIS statistics landed at 93.3 %, 93.3 % and 95.0 % inside; two of the three
"failed" a rule they had no better than even odds of meeting.

The threshold is instead placed in the lower tail of that binomial. For a
target confidence c (default 0.999), the acceptance count is the smallest
t whose lower tail is no larger than the tolerated spurious-failure
probability,

    t = 1 + max{ j : P(X <= j) <= 1 - c },

so that P(reject | consistent filter) = P(X < t) <= 1 - c by construction.
The default c = 0.999 is chosen for the instrument's role rather than by
convention: ``star consistency`` is an automated acceptance gate reporting
one coverage criterion per statistic (four for the reference EKF: NEES plus
three per-sensor NIS), so a per-criterion spurious-failure budget of 1e-3
holds the whole report's false-alarm rate near 4e-3 — roughly one spurious
red in 250 clean runs, rare enough that a red is worth investigating. The
cost in power is small because genuine inconsistency does not nudge the
coverage fraction, it collapses it (see
``tests/python/test_consistency.py`` for the measured bad-filter cases).

**Where the coverage criterion gates, and where it only reports**
(``epochs_independent``). The binomial premise is that the inside/outside
indicators are independent *across epochs*. That is a structural property
of the statistic, known before any data is seen, not something to estimate
per run:

- **NIS: independent.** The innovation sequence of a consistent filter is
  white (the orthogonality principle — Bar-Shalom, Li & Kirubarajan ch. 5),
  so successive NIS epochs are independent and the binomial threshold is an
  exact false-failure bound. Measured on 20 000 synthetic null trials at
  T = 60: 0.0750 % spurious failures against the 0.0738 % predicted, and on
  1000 disjoint ensembles of real reference-EKF output, 0-1 failures per
  statistic.
- **NEES: not independent.** The state error is a smooth trajectory and
  averaging across runs does not whiten it — each run shares the same
  autocorrelation, so the ensemble mean keeps it at every R. Measured on
  the reference EKF, the epoch-mean NEES series has an integrated
  autocorrelation time of about 30 epochs, i.e. roughly 20 effective
  independent samples in a 601-epoch run rather than 601. The count-inside
  is correspondingly over-dispersed, and a binomial threshold applied to it
  rejects consistent output about half the time (21 of 40 disjoint 100-run
  ensembles).

The coverage criterion therefore gates only when ``epochs_independent`` is
true, and is computed and reported as a diagnostic otherwise. Deliberately
no effective-sample-size correction is applied to salvage it in the
correlated case: the correction would have to estimate the autocorrelation
time from the same data being judged, and a filter whose error is stuck or
drifting inflates that estimate and so loosens its own threshold. A gate
must not be a function the system under test can move. The correlated case
is instead covered by the headline criterion, which is conservative under
any correlation whatsoever and cannot be relaxed by the data.

Two further limits, stated so nobody over-reads the guarantee. The
criterion is one-sided by design: it rejects too few epochs inside, never
too many; an implausibly high coverage fraction is not a consistency
failure in the Bar-Shalom sense, and the headline is what constrains the
statistic's level. And the headline's own false-failure rate is not
controlled at 1 - c; it is bounded by 1 - q = 5 % and is far below that
whenever epochs are not perfectly correlated, because averaging over epochs
shrinks the statistic's variance while the interval stays at its
single-epoch width.

Covariance packing: symmetric matrices travel as their row-major upper
triangle, ``[M_00, M_01, ..., M_0(n-1), M_11, ..., M_(n-1)(n-1)]`` — the
FR-26 convention for ``nav.est.P``/``nav.innov.S``, identical to the SRLOG
v1.1 ``mass.inertia_b_kgm2`` packing (docs/formats/srlog_v1.md section 3.1).

Quadratic forms are evaluated through a Cholesky factorization and a
triangular-system solve (``numpy.linalg``), never an explicit matrix
inverse; a non-positive-definite covariance is reported with the index of
the first offending epoch instead of producing garbage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from star_reacher.chi2 import binom_cdf, chi2_ppf

__all__ = [
    "DEFAULT_CONFIDENCE",
    "EnsembleGate",
    "IntervalGate",
    "ensemble_gate",
    "inside_count_threshold",
    "matrix_order",
    "nees",
    "nis",
    "pack_symmetric",
    "packed_length",
    "time_average_gate",
    "unpack_symmetric",
]

# Per-criterion spurious-failure budget of 1e-3; see the module docstring
# for why the instrument's automated-gate role sets it here rather than at
# a conventional 0.95 or 0.99.
DEFAULT_CONFIDENCE = 0.999


def packed_length(n: int) -> int:
    """Packed upper-triangle length n(n+1)/2 for an n x n symmetric matrix."""
    if n < 1:
        raise ValueError(f"matrix order must be >= 1, got {n}")
    return n * (n + 1) // 2


def matrix_order(m: int) -> int:
    """Matrix order n recovered from a packed length m = n(n+1)/2.

    Raises ``ValueError`` when m is not a triangular number, which catches
    a mismatched vector/covariance channel pair early with a clear message.
    """
    n = int((math.isqrt(8 * m + 1) - 1) // 2)
    if n < 1 or n * (n + 1) // 2 != m:
        raise ValueError(
            f"packed symmetric-matrix length {m} is not n(n+1)/2 for any "
            f"integer matrix order n"
        )
    return n


def pack_symmetric(matrices: np.ndarray) -> np.ndarray:
    """Pack symmetric matrices (..., n, n) to row-major upper triangles.

    Output shape (..., n(n+1)/2), element order
    ``[M_00, M_01, ..., M_0(n-1), M_11, ..., M_(n-1)(n-1)]``. Only the
    upper triangle is read; symmetry of the input is the caller's contract.
    """
    matrices = np.asarray(matrices, dtype=np.float64)
    if matrices.ndim < 2 or matrices.shape[-1] != matrices.shape[-2]:
        raise ValueError(
            f"pack_symmetric expects square trailing dimensions, got shape "
            f"{matrices.shape}"
        )
    rows, cols = np.triu_indices(matrices.shape[-1])
    return matrices[..., rows, cols]


def unpack_symmetric(packed: np.ndarray) -> np.ndarray:
    """Unpack row-major upper triangles (..., n(n+1)/2) to full matrices.

    The inverse of ``pack_symmetric``: the returned (..., n, n) arrays are
    exactly symmetric because each off-diagonal pair is written from the
    same packed element.
    """
    packed = np.asarray(packed, dtype=np.float64)
    if packed.ndim < 1:
        raise ValueError("unpack_symmetric expects at least one dimension")
    n = matrix_order(packed.shape[-1])
    rows, cols = np.triu_indices(n)
    full = np.zeros(packed.shape[:-1] + (n, n), dtype=np.float64)
    full[..., rows, cols] = packed
    full[..., cols, rows] = packed
    return full


def _cholesky_or_report(matrices: np.ndarray, matrix_name: str) -> np.ndarray:
    try:
        return np.linalg.cholesky(matrices)
    except np.linalg.LinAlgError:
        # The batched factorization reports no index; rescan one by one so
        # the error names the first offending epoch instead of "somewhere".
        n = matrices.shape[-1]
        flat = matrices.reshape(-1, n, n)
        for index in range(flat.shape[0]):
            try:
                np.linalg.cholesky(flat[index])
            except np.linalg.LinAlgError:
                raise ValueError(
                    f"{matrix_name} at flat epoch index {index} is not "
                    f"positive definite; a reported covariance must be a "
                    f"valid covariance matrix"
                ) from None
        raise  # pragma: no cover - batched failure with no failing member


def _quadratic_form(
    vectors: np.ndarray,
    packed: np.ndarray,
    vector_name: str,
    matrix_name: str,
) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    packed = np.asarray(packed, dtype=np.float64)
    if vectors.ndim < 1 or packed.ndim < 1:
        raise ValueError(
            f"{vector_name} and {matrix_name} must carry at least one dimension"
        )
    n = vectors.shape[-1]
    expected = packed_length(n)
    if packed.shape[-1] != expected:
        raise ValueError(
            f"{matrix_name} packed length {packed.shape[-1]} does not match "
            f"the {vector_name} dimension {n} (expected n(n+1)/2 = {expected})"
        )
    chol = _cholesky_or_report(unpack_symmetric(packed), matrix_name)
    # With M = L L^T, solving L z = v gives z^T z = v^T M^{-1} v: one
    # triangular solve per epoch, never an explicit inverse. Leading batch
    # dimensions broadcast, so per-run (T, n) and ensemble (R, T, n) inputs
    # share one covariance stack (T, n(n+1)/2).
    z = np.linalg.solve(chol, vectors[..., np.newaxis])[..., 0]
    return np.einsum("...i,...i->...", z, z)


def nees(e: np.ndarray, P_packed: np.ndarray) -> np.ndarray:
    """Per-epoch NEES eps_k = e_k^T P_k^{-1} e_k.

    ``e`` holds error vectors with shape (..., n) and ``P_packed`` the
    packed row-major upper-triangle covariances with shape (..., n(n+1)/2);
    leading dimensions broadcast (e.g. e of shape (R, T, n) against
    P_packed of shape (T, n(n+1)/2)). Returns the broadcast shape without
    the last axis. Under consistency, each value ~ chi-square(n).
    """
    return _quadratic_form(e, P_packed, "e", "P")


def nis(y: np.ndarray, S_packed: np.ndarray) -> np.ndarray:
    """Per-update NIS eps_k = y_k^T S_k^{-1} y_k.

    Identical mathematics to ``nees`` with the innovation y (shape
    (..., m)) and packed innovation covariance S (shape (..., m(m+1)/2));
    under consistency each value ~ chi-square(m).
    """
    return _quadratic_form(y, S_packed, "y", "S")


@dataclass(frozen=True)
class IntervalGate:
    """A scalar statistic checked against a two-sided chi-square interval.

    ``mean`` is the averaged statistic, ``lower``/``upper`` the interval,
    ``dof`` the chi-square degrees of freedom the interval came from, and
    ``passed`` is ``lower <= mean <= upper``.

    Whether a given instance is an acceptance gate or a diagnostic depends
    on which statistic it was built from, not on this type: the ensemble
    headline is a gate, while the per-run time average and the pooled mean
    are diagnostics whose ``passed`` must not reach an exit code (see the
    module docstring).
    """

    mean: float
    lower: float
    upper: float
    dof: int
    passed: bool


@dataclass(frozen=True)
class EnsembleGate:
    """Result of the FR-26 acceptance instrument (ch:ekf eq:ekf:ensemble).

    ``epoch_mean`` is the per-epoch ensemble average (shape (T,)) and
    ``lower``/``upper`` the interval from chi-square(``dof`` = R n) scaled
    by 1/R, which the three fractions partition the epochs against.

    Two criteria gate, and ``passed`` is their conjunction:

    - ``headline``, the epoch-average of ``epoch_mean`` against the same
      interval — conservative under across-time correlation;
    - ``coverage_passed``, ``inside_count >= min_inside``, where
      ``min_inside`` is the binomial lower-tail threshold from
      ``inside_count_threshold`` at ``confidence`` and ``min_fraction`` is
      that count expressed as a fraction for reporting. It contributes to
      ``passed`` only when ``coverage_gated`` is set, which mirrors the
      ``epochs_independent`` argument the gate was built with; otherwise it
      is computed for reporting and ``passed`` rests on ``headline`` alone.

    ``pooled`` is the all-epoch statistic at ``dof`` = R T n. It carries its
    own ``passed``, but that flag is a diagnostic: its interval assumes
    epochs are independent. Callers deciding acceptance read ``passed``.
    """

    epoch_mean: np.ndarray
    lower: float
    upper: float
    dof: int
    fraction_inside: float
    fraction_below: float
    fraction_above: float
    inside_count: int
    min_inside: int
    min_fraction: float
    confidence: float
    headline: IntervalGate
    coverage_passed: bool
    coverage_gated: bool
    passed: bool
    pooled: IntervalGate


def _interval(dof: int, scale: float, prob: float) -> tuple[float, float]:
    if not 0.0 < prob < 1.0:
        raise ValueError(f"interval probability must be in (0, 1), got {prob!r}")
    return (
        chi2_ppf(0.5 * (1.0 - prob), dof) / scale,
        chi2_ppf(0.5 * (1.0 + prob), dof) / scale,
    )


def inside_count_threshold(
    epochs: int,
    coverage: float = 0.95,
    confidence: float = DEFAULT_CONFIDENCE,
) -> int:
    """Minimum epochs-inside count for the ensemble coverage criterion.

    Under consistency the count of epochs whose statistic lies inside its
    two-sided ``coverage`` interval is Binomial(``epochs``, ``coverage``).
    Returns the smallest t with P(X < t) <= 1 - ``confidence``, i.e.

        t = 1 + max{ j : binom_cdf(j, epochs, coverage) <= 1 - confidence },

    so a consistent filter fails the criterion with probability at most
    1 - ``confidence``. The module docstring derives this and states the
    two limits on the guarantee. Returns 0 when even an empty count is
    admissible, which happens only for very short runs.
    """
    epochs = int(epochs)
    if epochs < 1:
        raise ValueError(f"epoch count must be >= 1, got {epochs}")
    if not 0.0 < coverage < 1.0:
        raise ValueError(f"coverage must be in (0, 1), got {coverage!r}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence!r}")
    alpha = 1.0 - confidence
    # binom_cdf is nondecreasing in j, so bisect for the last j still inside
    # the tolerated tail. lo = -1 is always admissible (cdf 0 <= alpha) and
    # hi = epochs never is (cdf 1 > alpha), so the invariant holds on entry.
    lo, hi = -1, epochs
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if binom_cdf(mid, epochs, coverage) <= alpha:
            lo = mid
        else:
            hi = mid
    return lo + 1


def time_average_gate(eps: np.ndarray, dim: int, prob: float = 0.95) -> IntervalGate:
    """Per-run time-averaged NEES/NIS DIAGNOSTIC (never an acceptance gate).

    ``eps`` is one run's per-epoch statistic (shape (T,)) and ``dim`` the
    statistic's chi-square dimension (n for NEES, m for NIS). The interval
    is [chi2_ppf((1-prob)/2, T*dim)/T, chi2_ppf((1+prob)/2, T*dim)/T].

    Its ``passed`` flag is indicative only: the interval assumes the T
    per-epoch values are independent, and a filter's state error is a
    serially correlated trajectory, so the interval is much narrower than
    eps_bar's true sampling spread and rejects the large majority of
    perfectly consistent runs (7 of 100 passed on the exit-criterion-3
    ensemble). Use ``ensemble_gate`` to decide acceptance; this function
    exists to report the per-run number to a human reader.
    """
    eps = np.asarray(eps, dtype=np.float64)
    if eps.ndim != 1 or eps.shape[0] < 1:
        raise ValueError(
            f"time_average_gate expects one run's statistic of shape (T,), "
            f"got shape {eps.shape}"
        )
    epochs = eps.shape[0]
    dof = epochs * int(dim)
    lower, upper = _interval(dof, float(epochs), prob)
    mean = float(eps.mean())
    return IntervalGate(
        mean=mean, lower=lower, upper=upper, dof=dof, passed=lower <= mean <= upper
    )


def ensemble_gate(
    eps: np.ndarray,
    dim: int,
    prob: float = 0.95,
    confidence: float = DEFAULT_CONFIDENCE,
    epochs_independent: bool = True,
) -> EnsembleGate:
    """Ensemble NEES/NIS gate over R runs: the FR-26 acceptance instrument.

    ``eps`` holds the per-epoch statistic for every run, shape (R, T), and
    ``dim`` the statistic's chi-square dimension. Each epoch's ensemble
    average is taken against
    [chi2_ppf((1-prob)/2, R*dim)/R, chi2_ppf((1+prob)/2, R*dim)/R], and
    ``passed`` combines the criteria of ch:ekf eq:ekf:ensemble: the
    epoch-averaged headline inside that interval, and — when
    ``epochs_independent`` — at least
    ``inside_count_threshold(T, prob, confidence)`` epochs inside it. The
    pooled all-epoch mean is computed alongside as a diagnostic.

    ``epochs_independent`` declares whether successive epochs of this
    statistic are independent under consistency: true for NIS (a consistent
    filter's innovations are white), false for NEES (the state error is a
    correlated trajectory). It must reflect the statistic's structure, not
    a property estimated from the data; see the module docstring for why a
    data-estimated correction would be a gate the system under test can
    move.

    R = 1 is accepted (the single run is its own ensemble average, still
    chi-square(dim) per epoch), so a one-log invocation stays gated.
    """
    eps = np.asarray(eps, dtype=np.float64)
    if eps.ndim != 2:
        raise ValueError(
            f"ensemble_gate expects per-run statistics of shape (R, T), got "
            f"shape {eps.shape}"
        )
    runs, epochs = eps.shape
    if runs < 1 or epochs < 1:
        raise ValueError(
            f"ensemble statistics need at least one run and one epoch, got "
            f"shape {eps.shape}"
        )
    dof = runs * int(dim)
    lower, upper = _interval(dof, float(runs), prob)
    epoch_mean = eps.mean(axis=0)
    inside_mask = (epoch_mean >= lower) & (epoch_mean <= upper)
    inside_count = int(np.count_nonzero(inside_mask))
    min_inside = inside_count_threshold(epochs, prob, confidence)

    # Averaging epoch_mean over epochs cannot increase its variance beyond a
    # single epoch's, so reusing the single-epoch interval here is a
    # conservative test whatever the across-time correlation is.
    headline_mean = float(epoch_mean.mean())
    headline = IntervalGate(
        mean=headline_mean,
        lower=lower,
        upper=upper,
        dof=dof,
        passed=lower <= headline_mean <= upper,
    )

    pooled_dof = runs * epochs * int(dim)
    pooled_lower, pooled_upper = _interval(pooled_dof, float(runs * epochs), prob)
    pooled_mean = float(eps.mean())
    pooled = IntervalGate(
        mean=pooled_mean,
        lower=pooled_lower,
        upper=pooled_upper,
        dof=pooled_dof,
        passed=pooled_lower <= pooled_mean <= pooled_upper,
    )
    coverage_passed = inside_count >= min_inside
    coverage_gated = bool(epochs_independent)
    return EnsembleGate(
        epoch_mean=epoch_mean,
        lower=lower,
        upper=upper,
        dof=dof,
        fraction_inside=inside_count / epochs,
        fraction_below=float(np.mean(epoch_mean < lower)),
        fraction_above=float(np.mean(epoch_mean > upper)),
        inside_count=inside_count,
        min_inside=min_inside,
        min_fraction=min_inside / epochs,
        confidence=float(confidence),
        headline=headline,
        coverage_passed=coverage_passed,
        coverage_gated=coverage_gated,
        passed=headline.passed and (coverage_passed or not coverage_gated),
        pooled=pooled,
    )
