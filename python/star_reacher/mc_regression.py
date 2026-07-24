"""Seeded Monte Carlo regression gate (FR-22 layer 6, Phase 7 exit criterion 2).

``star mc`` (``star_reacher.mc``) turns a sweep spec into a bit-reproducible
ensemble: the same master seed yields the same per-run seeds, the same logged
bytes, and so the same per-run outcome metric, run after run. This module
freezes the *statistics* of one such ensemble as a golden and gates a re-run's
statistics against it with two complementary 99 % tests, so a change to the
physics or the numerics that moves the outcome distribution is caught while a
bit-identical re-run passes.

The reference ensemble is ``missions/mc_regression_sweep.toml``: a 128-run Latin
hypercube that disperses the initial in-plane velocity of the committed
EGM2008 8x8 LEO mission, each run flying a distinct bound orbit. The per-run
OUTCOME METRIC is the final osculating specific mechanical energy
E = |v|^2 / 2 - GM / |r| of the truth trajectory (``run.elements()``'s
``energy_m2ps2`` at the last epoch) -- a physically meaningful, conservative
quantity dispersed across the ensemble by the initial-velocity dispersion,
computable from the committed gravity data with no fetched ephemeris.

The golden (``tests/golden/mc_regression/energy_stats.toml``) freezes the
ensemble size n, the metric mean mu_g, and the sample standard deviation
sigma_g. Two gates run against it at probability 0.99:

- **chi-square (scale).** Under the regression hypothesis that the re-run's
  metric has the golden mean and variance, the standardized sum of squares
  S = sum_i ((x_i - mu_g) / sigma_g)^2 is chi-square(n) distributed, and it is
  checked against the two-sided 99 % interval
  [chi2_ppf(0.005, n), chi2_ppf(0.995, n)] -- the same two-sided construction
  ``star_reacher.consistency`` gates NEES on. This catches a change in the
  metric's spread: a variance inflation drives S above the upper bound, a
  collapse drives it below the lower.

- **Anderson-Darling (shape and location).** The standardized metric
  z_i = (x_i - mu_g) / sigma_g is Anderson-Darling tested against the standard
  normal CDF (``star_reacher.anderson``), i.e. the reference N(mu_g, sigma_g),
  and the gate passes when the p-value is at least 0.01. A-D weights the tails,
  so it catches a distribution SHIFT that the sum-of-squares chi-square, being
  symmetric in the residual, tolerates: a mean shift of half a standard
  deviation leaves S comfortably inside its interval yet fails the A-D gate
  outright (measured in ``tests/python/test_mc_regression.py``).

Why N(mu_g, sigma_g) as the A-D reference. The regression hypothesis is that
the distribution is UNCHANGED from the golden, and the maximum-entropy
continuous distribution consistent with a frozen mean and variance is the
normal. It is also the distribution the reference ensemble's standardized
metric empirically follows: on the frozen ensemble the A-D statistic against
N(mu_g, sigma_g) is A2 = 1.408 (p = 0.200), comfortably inside the gate, while
against a uniform reference it is A2 = 10.73 (p = 6e-6). The gate is a fixed
function of the frozen seed, so the pass-point statistics are deterministic;
they are recorded in the module tests and in the golden manifest.

No SciPy at runtime (D-12): both gates are built on the project's own
first-principles ``chi2`` and ``anderson`` modules.
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from star_reacher.anderson import anderson_darling
from star_reacher.chi2 import chi2_ppf

__all__ = [
    "GOLDEN_METRIC",
    "GOLDEN_VALUE_FILE",
    "REGRESSION_PROB",
    "GoldenStats",
    "McRegressionError",
    "RegressionGate",
    "ensemble_metric",
    "format_golden_toml",
    "golden_stats_dict",
    "load_golden_stats",
    "regression_gate",
    "summarize_metric",
]

# The golden value file the gate reads and the two-key tooling maintains.
GOLDEN_VALUE_FILE = "energy_stats.toml"

# The gate probability. 0.99 is the exit-criterion-2 figure: both the
# chi-square two-sided interval and the Anderson-Darling p-value threshold
# (reject when p < 1 - 0.99 = 0.01) are taken at 99 %.
REGRESSION_PROB = 0.99

# The frozen sweep's per-run outcome metric: the final osculating specific
# mechanical energy of the truth trajectory. Named as a constant so the golden
# generator, the gate, and the docs name the identical quantity.
GOLDEN_METRIC = "energy_m2ps2"


class McRegressionError(Exception):
    """A Monte Carlo regression input error (bad manifest, empty ensemble, bad golden)."""


def _standard_normal_cdf(x: float) -> float:
    # Phi(x) from math.erf: the A-D reference CDF for the standardized metric,
    # SciPy-free like every other statistical primitive here (D-12).
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass(frozen=True)
class GoldenStats:
    """Frozen reference statistics of the regression ensemble's outcome metric.

    ``n`` is the ensemble size, ``mean``/``std`` the metric's mean and sample
    (ddof=1) standard deviation, ``metric`` the metric name, and ``mission``
    the base mission the sweep dispersed. These are what the gate reads; the
    provenance (date, generation procedure, value hash) lives alongside in the
    directory's ``manifest.toml``.
    """

    n: int
    mean: float
    std: float
    metric: str
    mission: str


def ensemble_metric(manifest: dict, manifest_dir) -> np.ndarray:
    """The per-run outcome metric of a completed ``star mc`` ensemble.

    ``manifest`` is a parsed ``manifest.json`` and ``manifest_dir`` the
    directory holding it (so each run's ``outdir``/``run.srlog`` resolves). The
    metric is the final ``GOLDEN_METRIC`` of every successful run's truth
    trajectory, in run-index order, as a float64 array.

    Raises :class:`McRegressionError` if any run failed (a regression ensemble
    must be complete to be comparable) or if the manifest has no runs.
    """
    from star_reacher.srlog import load

    manifest_dir = Path(manifest_dir)
    runs = manifest.get("runs", [])
    if not runs:
        raise McRegressionError("the manifest records no runs")
    failed = [r["index"] for r in runs if r.get("status") != "success"]
    if failed:
        raise McRegressionError(
            f"{len(failed)} run(s) did not succeed (first index {failed[0]}); a "
            f"regression ensemble must be complete to compare against the golden"
        )
    values = []
    for entry in sorted(runs, key=lambda r: r["index"]):
        log_path = manifest_dir / entry["outdir"] / "run.srlog"
        run = load(log_path)
        # elements() derives the osculating set in the loader (FR-16); the
        # final epoch's specific energy is the run's scalar outcome.
        values.append(float(run.elements("truth")[GOLDEN_METRIC][-1]))
    return np.asarray(values, dtype=np.float64)


def summarize_metric(metric: np.ndarray, *, mission: str) -> GoldenStats:
    """Reduce a metric array to the :class:`GoldenStats` the golden freezes.

    The sample standard deviation uses ddof=1 (Bessel's correction), the
    unbiased estimator of the population sigma the chi-square and A-D gates
    standardize by. Raises :class:`McRegressionError` for fewer than two runs
    (a standard deviation is undefined) or a degenerate zero spread.
    """
    metric = np.asarray(metric, dtype=np.float64)
    n = int(metric.shape[0])
    if n < 2:
        raise McRegressionError(
            f"a regression ensemble needs at least two runs, got {n}"
        )
    std = float(metric.std(ddof=1))
    if not (std > 0.0 and math.isfinite(std)):
        raise McRegressionError(
            f"the metric has a non-positive or non-finite spread ({std!r}); the "
            f"gate standardizes by it, so the ensemble must be dispersed"
        )
    return GoldenStats(
        n=n,
        mean=float(metric.mean()),
        std=std,
        metric=GOLDEN_METRIC,
        mission=mission,
    )


def golden_stats_dict(stats: GoldenStats) -> dict:
    """The golden value file's payload, values as exact binary64 hex literals.

    ``mean`` and ``std`` ride as ``float.hex()`` strings so the frozen golden
    is bit-exact (the regression ensemble is bit-reproducible, so its
    statistics are exact numbers, not rounded ones), the same discipline the
    rng/box_muller golden uses for its float values.
    """
    return {
        "metric": stats.metric,
        "mission": stats.mission,
        "n": stats.n,
        "mean_hex": float(stats.mean).hex(),
        "std_hex": float(stats.std).hex(),
    }


def format_golden_toml(stats: GoldenStats) -> str:
    """Render :class:`GoldenStats` as the golden value file's TOML text.

    A fixed field order and a documenting header, so the value file is a
    stable, diffable artifact whose bytes are a pure function of the frozen
    statistics -- the property the two-key tooling's ``values_sha256`` pins.
    ``mean``/``std`` ride as exact binary64 hex literals; ``mean_readable``/
    ``std_readable`` echo the decimal for a human reader and are NOT read back
    (``load_golden_stats`` uses only the hex fields), so their rounding never
    affects the gate.
    """
    d = golden_stats_dict(stats)
    return (
        "# Frozen golden statistics of the Phase 7 Monte Carlo regression\n"
        "# ensemble (FR-22 layer 6). Written only by scripts/golden_update.py\n"
        "# --apply; provenance and the values_sha256 gate live in manifest.toml.\n"
        "# The mean/std are exact binary64 hex literals (float.hex()); the\n"
        "# *_readable decimals are for the eye only and are never read back.\n"
        "\n"
        f'metric = "{d["metric"]}"\n'
        f'mission = "{d["mission"]}"\n'
        f'n = {d["n"]}\n'
        f'mean_hex = "{d["mean_hex"]}"\n'
        f'std_hex = "{d["std_hex"]}"\n'
        f"mean_readable = {stats.mean!r}\n"
        f"std_readable = {stats.std!r}\n"
    )


def load_golden_stats(path) -> GoldenStats:
    """Parse a frozen golden value file into :class:`GoldenStats`.

    The inverse of :func:`golden_stats_dict`: reads the ``mean_hex``/``std_hex``
    binary64 hex literals back to exact floats. Raises
    :class:`McRegressionError` naming the missing field on a malformed file.
    """
    path = Path(path)
    with path.open("rb") as fh:
        doc = tomllib.load(fh)
    try:
        return GoldenStats(
            n=int(doc["n"]),
            mean=float.fromhex(doc["mean_hex"]),
            std=float.fromhex(doc["std_hex"]),
            metric=str(doc["metric"]),
            mission=str(doc["mission"]),
        )
    except KeyError as exc:
        raise McRegressionError(
            f"{path}: the golden value file is missing field {exc.args[0]!r}"
        ) from None


@dataclass(frozen=True)
class RegressionGate:
    """Result of the two-part Monte Carlo regression gate.

    ``chi2_stat`` is the standardized sum of squares S against the golden mean
    and variance, checked against [``chi2_lower``, ``chi2_upper``] (the
    two-sided ``prob`` chi-square(``dof``) interval); ``chi2_passed`` is
    ``chi2_lower <= chi2_stat <= chi2_upper``. ``ad_stat`` is the
    Anderson-Darling A2 of the standardized metric against N(mu_g, sigma_g) and
    ``ad_pvalue`` its p-value; ``ad_passed`` is ``ad_pvalue >= 1 - prob``.
    ``passed`` is the conjunction: a regression ensemble matches the golden only
    if it holds BOTH its spread (chi-square) and its shape/location (A-D).
    """

    n: int
    dof: int
    prob: float
    chi2_stat: float
    chi2_lower: float
    chi2_upper: float
    chi2_passed: bool
    ad_stat: float
    ad_pvalue: float
    ad_passed: bool
    passed: bool


def regression_gate(
    metric: np.ndarray, golden: GoldenStats, prob: float = REGRESSION_PROB
) -> RegressionGate:
    """Gate an ensemble metric against frozen golden statistics at ``prob``.

    ``metric`` is the per-run outcome array (shape (n,)) and ``golden`` the
    frozen reference. Both the chi-square scale gate and the Anderson-Darling
    shape/location gate must pass for :attr:`RegressionGate.passed`. Raises
    :class:`McRegressionError` if the ensemble size does not match the golden's
    ``n`` (a different N is a different experiment, not a regression of the same
    one).
    """
    metric = np.asarray(metric, dtype=np.float64)
    n = int(metric.shape[0])
    if n != golden.n:
        raise McRegressionError(
            f"the ensemble has {n} runs but the golden was frozen at "
            f"{golden.n}; a regression must compare the same-size ensemble"
        )
    if not 0.0 < prob < 1.0:
        raise McRegressionError(f"prob must be in (0, 1), got {prob!r}")

    z = (metric - golden.mean) / golden.std

    # chi-square scale gate: S ~ chi-square(n) under the golden mean/variance.
    chi2_stat = float(np.sum(z * z))
    chi2_lower = chi2_ppf(0.5 * (1.0 - prob), n)
    chi2_upper = chi2_ppf(0.5 * (1.0 + prob), n)
    chi2_passed = chi2_lower <= chi2_stat <= chi2_upper

    # Anderson-Darling shape/location gate against N(mu_g, sigma_g).
    ad_stat, ad_pvalue = anderson_darling(z, _standard_normal_cdf)
    ad_passed = ad_pvalue >= 1.0 - prob

    return RegressionGate(
        n=n,
        dof=n,
        prob=prob,
        chi2_stat=chi2_stat,
        chi2_lower=chi2_lower,
        chi2_upper=chi2_upper,
        chi2_passed=chi2_passed,
        ad_stat=ad_stat,
        ad_pvalue=ad_pvalue,
        ad_passed=ad_passed,
        passed=chi2_passed and ad_passed,
    )
