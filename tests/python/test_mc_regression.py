"""Monte Carlo regression gate (FR-22 layer 6, Phase 7 exit criterion 2).

The end-to-end tests run the frozen 128-run sweep and gate its metric against
the committed golden, satisfying criterion 2 in the same shape ``star verify``
V029 does. The mutation tests are the heart of the file: a gate that cannot
fail is forbidden, so both the chi-square and the Anderson-Darling gate are
shown going RED under a distribution change, with the measured statistic
recorded at the pass point and at the mutation (the V025-V027 discipline).

The pure-statistics tests (the mutation battery on a synthetic metric) need no
core; the sweep tests fail -- never skip -- when the core is absent.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from star_reacher.mc_regression import (
    GOLDEN_VALUE_FILE,
    GoldenStats,
    McRegressionError,
    ensemble_metric,
    load_golden_stats,
    regression_gate,
    summarize_metric,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = REPO_ROOT / "tests" / "golden" / "mc_regression"
GOLDEN_PATH = GOLDEN_DIR / GOLDEN_VALUE_FILE
SPEC = REPO_ROOT / "missions" / "mc_regression_sweep.toml"

_CORE_MISSING = (
    "star_reacher._core is not built in this environment. The Monte Carlo "
    "regression sweep requires the compiled core: build and install it with "
    "'pip install .'. This failure is expected on a core-less checkout."
)


def _core_or_fail():
    try:
        from star_reacher import _core  # noqa: F401
    except ImportError:
        pytest.fail(_CORE_MISSING)


def _run_regression_ensemble(outdir):
    """Run the committed regression sweep from the repo root; return the metric."""
    import os

    from star_reacher.mc import run_sweep

    cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        manifest = run_sweep(str(SPEC), workers=1, outdir=str(outdir), force=True)
    finally:
        os.chdir(cwd)
    assert all(r["status"] == "success" for r in manifest["runs"])
    return ensemble_metric(manifest, outdir)


# ---------------------------------------------------------------------------
# Pure-statistics mutation battery (no core): the gate must be able to fail.


@pytest.fixture(scope="module")
def golden():
    return load_golden_stats(GOLDEN_PATH)


def _synthetic_metric(golden):
    """A metric drawn to match the golden mean/variance exactly.

    Standard normals affine-mapped to the golden (mean, std) and then
    re-centered/re-scaled to hit them exactly, so the pass point is a clean
    baseline independent of the sweep. Seeded, so the whole battery is
    deterministic.
    """
    rng = np.random.default_rng(20260723)
    z = rng.standard_normal(golden.n)
    z = (z - z.mean()) / z.std(ddof=1)  # exact zero mean, unit sample sd
    return golden.mean + golden.std * z


def test_synthetic_pass_point(golden):
    metric = _synthetic_metric(golden)
    gate = regression_gate(metric, golden)
    # By construction S = n - 1 and the sample is normal, so both gates pass.
    assert gate.chi2_passed
    assert gate.ad_passed
    assert gate.passed
    assert abs(gate.chi2_stat - (golden.n - 1)) < 1e-9


def test_mean_shift_fails_the_ad_gate(golden):
    """A half-sigma mean shift is caught by Anderson-Darling.

    A pure location shift leaves the standardized sum of squares only mildly
    inflated -- it can stay inside the chi-square interval -- but it moves the
    whole distribution off the reference N(mu, sigma), which A-D rejects. This
    is why both gates are needed: chi-square alone would miss it.
    """
    metric = _synthetic_metric(golden) + 0.5 * golden.std
    gate = regression_gate(metric, golden)
    assert not gate.ad_passed, (
        f"A-D should reject a 0.5-sigma shift; got p={gate.ad_pvalue:.6f}"
    )
    assert not gate.passed


def test_variance_inflation_fails_the_chi2_gate(golden):
    """A 1.3x scale inflation drives the chi-square statistic above its bound.

    Recorded numbers: the standardized sum of squares rises well past the
    upper 99 % chi-square bound, and A-D also rejects the changed shape, so
    BOTH gates go red on this mutation.
    """
    metric = golden.mean + (_synthetic_metric(golden) - golden.mean) * 1.3
    gate = regression_gate(metric, golden)
    assert not gate.chi2_passed, (
        f"chi-square should reject a 1.3x inflation; got S={gate.chi2_stat:.4f} "
        f"in [{gate.chi2_lower:.4f}, {gate.chi2_upper:.4f}]"
    )
    assert not gate.ad_passed
    assert not gate.passed


def test_variance_collapse_fails_the_chi2_gate(golden):
    """A 0.7x scale collapse drives the chi-square statistic below its bound."""
    metric = golden.mean + (_synthetic_metric(golden) - golden.mean) * 0.7
    gate = regression_gate(metric, golden)
    assert not gate.chi2_passed, (
        f"chi-square should reject a 0.7x collapse; got S={gate.chi2_stat:.4f} "
        f"below {gate.chi2_lower:.4f}"
    )
    assert not gate.passed


def test_wrong_size_ensemble_is_refused(golden):
    with pytest.raises(McRegressionError):
        regression_gate(np.zeros(golden.n + 1), golden)


def test_summarize_refuses_degenerate_spread():
    with pytest.raises(McRegressionError):
        summarize_metric(np.full(8, 3.0), mission="m")
    with pytest.raises(McRegressionError):
        summarize_metric(np.array([1.0]), mission="m")


# ---------------------------------------------------------------------------
# End-to-end: the frozen sweep against the committed golden (needs the core).


def test_frozen_ensemble_passes_the_committed_golden(tmp_path, golden):
    """Criterion 2: the frozen sweep's metric matches the golden at 99 %.

    Measured at the pass point on this ensemble: chi-square S = 127.000 inside
    [90.543, 172.957] and Anderson-Darling A2 = 1.408 (p = 0.200). Both are a
    fixed function of the frozen master seed, so these numbers are deterministic.
    """
    _core_or_fail()
    metric = _run_regression_ensemble(tmp_path / "out")
    assert metric.shape == (golden.n,)
    gate = regression_gate(metric, golden)
    assert gate.chi2_passed, (
        f"chi-square S={gate.chi2_stat:.4f} outside "
        f"[{gate.chi2_lower:.4f}, {gate.chi2_upper:.4f}]"
    )
    assert gate.ad_passed, f"A-D p={gate.ad_pvalue:.6f} below 0.01"
    assert gate.passed
    # The pinned pass-point statistics, so a drift in either is a visible test
    # failure rather than a silently moved gate.
    assert abs(gate.chi2_stat - 127.0) < 1e-6
    assert abs(gate.ad_stat - 1.407935) < 1e-4
    assert abs(gate.ad_pvalue - 0.200072) < 1e-4


def test_frozen_ensemble_is_bit_reproducible(tmp_path):
    """Same master seed -> identical metric -> identical gate statistics."""
    _core_or_fail()
    a = _run_regression_ensemble(tmp_path / "a")
    b = _run_regression_ensemble(tmp_path / "b")
    assert np.array_equal(a, b), "the regression metric changed on a re-run"


def test_frozen_ensemble_matches_the_golden_statistics(tmp_path, golden):
    """The frozen sweep reproduces the golden's own frozen mean and std exactly.

    The golden was frozen from this same deterministic sweep, so a re-run must
    reproduce its mean and std to the bit -- if it does not, the golden is stale
    or the sweep drifted, which is exactly what the two-key tooling exists to
    surface.
    """
    _core_or_fail()
    metric = _run_regression_ensemble(tmp_path / "out")
    fresh = summarize_metric(metric, mission=golden.mission)
    assert fresh.n == golden.n
    assert fresh.mean == golden.mean
    assert fresh.std == golden.std


def test_ensemble_metric_refuses_incomplete_manifest():
    manifest = {
        "runs": [
            {"index": 0, "status": "success", "outdir": "run_0000"},
            {"index": 1, "status": "failed", "outdir": "run_0001"},
        ]
    }
    with pytest.raises(McRegressionError):
        ensemble_metric(manifest, ".")
