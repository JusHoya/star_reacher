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
- **Per-run time-averaged gate** (``time_average_gate``). With T epochs,
  the run mean eps_bar = (1/T) sum_k eps_k satisfies
  T * eps_bar ~ chi-square(T n) under the hypothesis that the eps_k are
  independent, so the two-sided interval at probability q (default 0.95) is

      [chi2_ppf((1-q)/2, T n) / T,  chi2_ppf((1+q)/2, T n) / T]

  and the gate passes when eps_bar lies inside it. Caveat: for a filter
  with process noise the single-run eps_k sequence is autocorrelated, so
  this interval is approximate — it is a per-run diagnostic, not the
  acceptance instrument.
- **Ensemble per-epoch gate** (``ensemble_gate`` — the acceptance
  instrument, per the Bar-Shalom Monte Carlo convention and the FR-26
  "per-run and ensemble" wording). Over R independent runs the ensemble
  average at epoch k, eps_bar_k = (1/R) sum_r eps_(r,k), satisfies
  R * eps_bar_k ~ chi-square(R n) exactly under the consistency
  hypothesis, giving the per-epoch two-sided interval

      [chi2_ppf((1-q)/2, R n) / R,  chi2_ppf((1+q)/2, R n) / R].

  PASS criterion: the fraction of epochs whose ensemble average lies
  inside the interval must be at least ``min_fraction`` (default 0.95).
  This is the criterion the Phase 6 EKF driver gates on.
- **Pooled gate** (reported inside ``ensemble_gate``). The mean over all
  R*T per-epoch values satisfies R*T*mean ~ chi-square(R T n) under
  full independence, with interval

      [chi2_ppf((1-q)/2, R T n) / (R T),  chi2_ppf((1+q)/2, R T n) / (R T)].

  The across-time correlation caveat of the time-averaged gate applies to
  the pooled statistic as well.

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

from star_reacher.chi2 import chi2_ppf

__all__ = [
    "EnsembleGate",
    "IntervalGate",
    "ensemble_gate",
    "matrix_order",
    "nees",
    "nis",
    "pack_symmetric",
    "packed_length",
    "time_average_gate",
    "unpack_symmetric",
]


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
    """

    mean: float
    lower: float
    upper: float
    dof: int
    passed: bool


@dataclass(frozen=True)
class EnsembleGate:
    """Ensemble per-epoch gate result plus the pooled all-epoch statistic.

    ``epoch_mean`` is the per-epoch ensemble average (shape (T,)),
    ``lower``/``upper`` the per-epoch interval from chi-square(``dof`` = R n)
    scaled by 1/R, and the three fractions partition the epochs against it.
    ``passed`` is the acceptance criterion ``fraction_inside >=
    min_fraction``; ``pooled`` is the all-epoch ``IntervalGate`` with its
    own independent ``passed``.
    """

    epoch_mean: np.ndarray
    lower: float
    upper: float
    dof: int
    fraction_inside: float
    fraction_below: float
    fraction_above: float
    min_fraction: float
    passed: bool
    pooled: IntervalGate


def _interval(dof: int, scale: float, prob: float) -> tuple[float, float]:
    if not 0.0 < prob < 1.0:
        raise ValueError(f"interval probability must be in (0, 1), got {prob!r}")
    return (
        chi2_ppf(0.5 * (1.0 - prob), dof) / scale,
        chi2_ppf(0.5 * (1.0 + prob), dof) / scale,
    )


def time_average_gate(eps: np.ndarray, dim: int, prob: float = 0.95) -> IntervalGate:
    """Per-run time-averaged NEES/NIS gate (diagnostic; see module docstring).

    ``eps`` is one run's per-epoch statistic (shape (T,)) and ``dim`` the
    statistic's chi-square dimension (n for NEES, m for NIS). The interval
    is [chi2_ppf((1-prob)/2, T*dim)/T, chi2_ppf((1+prob)/2, T*dim)/T] and
    the gate passes when mean(eps) lies inside it.
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
    min_fraction: float = 0.95,
) -> EnsembleGate:
    """Ensemble NEES/NIS gate over R runs: the FR-26 acceptance instrument.

    ``eps`` holds the per-epoch statistic for every run, shape (R, T) with
    R >= 2, and ``dim`` the statistic's chi-square dimension. Each epoch's
    ensemble average is checked against
    [chi2_ppf((1-prob)/2, R*dim)/R, chi2_ppf((1+prob)/2, R*dim)/R];
    the gate passes when at least ``min_fraction`` of the epochs fall
    inside. The pooled all-epoch mean and its chi-square(R*T*dim) interval
    are computed alongside (see module docstring for the correlation
    caveat).
    """
    eps = np.asarray(eps, dtype=np.float64)
    if eps.ndim != 2:
        raise ValueError(
            f"ensemble_gate expects per-run statistics of shape (R, T), got "
            f"shape {eps.shape}"
        )
    runs, epochs = eps.shape
    if runs < 2:
        raise ValueError(f"ensemble statistics need at least two runs, got {runs}")
    if not 0.0 < min_fraction <= 1.0:
        raise ValueError(f"min_fraction must be in (0, 1], got {min_fraction!r}")
    dof = runs * int(dim)
    lower, upper = _interval(dof, float(runs), prob)
    epoch_mean = eps.mean(axis=0)
    inside = float(np.mean((epoch_mean >= lower) & (epoch_mean <= upper)))
    below = float(np.mean(epoch_mean < lower))
    above = float(np.mean(epoch_mean > upper))

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
    return EnsembleGate(
        epoch_mean=epoch_mean,
        lower=lower,
        upper=upper,
        dof=dof,
        fraction_inside=inside,
        fraction_below=below,
        fraction_above=above,
        min_fraction=min_fraction,
        passed=inside >= min_fraction,
        pooled=pooled,
    )
