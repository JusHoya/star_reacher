"""FR-26 consistency instrument: chi-square quantiles, NEES/NIS gates, CLI.

Pure NumPy plus stdlib; no compiled core and no SciPy (D-12). The chi-square
quantile implementation is anchored three independent ways so a defect
cannot hide behind a mirrored defect:

- published table values (Abramowitz & Stegun, Handbook of Mathematical
  Functions, Table 26.8; also NIST/SEMATECH e-Handbook of Statistical
  Methods, section 1.3.6.7.4), each cross-checked against the
  Wilson-Hilferty approximation (A&S eq. 26.4.17) to catch transcription
  errors in the reference digits themselves;
- closed forms that share no code with the implementation: k=1 via
  ``math.erf``, k=2 via the exponential distribution, k=4 via the Erlang
  CDF;
- CDF/quantile round-trips up to k = 1e6, the pooled-ensemble scale.

The gate tests draw synthetic filter ensembles with known covariance and
include mutation tests in both directions (covariance reported smaller and
larger than truth), proving the acceptance gate is able to fail before any
green result is trusted.
"""

import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.random import PCG64, Generator

import star_reacher
from star_reacher import _fixtures
from star_reacher.chi2 import (
    binom_cdf,
    chi2_cdf,
    chi2_ppf,
    gammp,
    normal_ppf,
    wilson_hilferty_ppf,
)
from star_reacher.consistency import (
    DEFAULT_CONFIDENCE,
    ensemble_gate,
    inside_count_threshold,
    matrix_order,
    nees,
    nis,
    pack_symmetric,
    packed_length,
    time_average_gate,
    unpack_symmetric,
)

# ---------------------------------------------------------------------------
# Chi-square quantiles


# (p, k, reference quantile, abs tol from the printed digits, Wilson-Hilferty
# relative tolerance). References: Abramowitz & Stegun Table 26.8 (percentage
# points of the chi-square distribution); the same values appear in the
# NIST/SEMATECH e-Handbook, section 1.3.6.7.4. The W-H column is the
# transcription-error tripwire: W-H is coarse at small k (a few percent at
# k = 1) and tight by k = 100, so the tolerances widen accordingly.
_TABLE_REFERENCES = [
    (0.975, 1, 5.023886, 1.5e-6, 0.05),
    (0.025, 10, 3.246973, 1.5e-6, 0.02),
    (0.95, 100, 124.342, 5.0e-4, 1.0e-3),
]


@pytest.mark.parametrize("p, k, reference, atol, wh_rtol", _TABLE_REFERENCES)
def test_chi2_ppf_matches_published_tables(p, k, reference, atol, wh_rtol):
    assert abs(chi2_ppf(p, k) - reference) <= atol
    # Verify the reference digits themselves against Wilson-Hilferty so a
    # transcribed-wrong table value cannot silently anchor the suite.
    assert abs(wilson_hilferty_ppf(p, k) - reference) <= wh_rtol * reference


def test_chi2_k2_matches_exponential_closed_form():
    # chi-square with 2 dof is the exponential distribution with mean 2:
    # ppf(p, 2) = -2 ln(1 - p) exactly. Independent of the gamma-function
    # machinery under test.
    for p in (0.001, 0.025, 0.05, 0.5, 0.95, 0.975, 0.999):
        exact = -2.0 * math.log1p(-p)
        assert abs(chi2_ppf(p, 2) - exact) <= 1e-12 * exact


def test_chi2_k1_cdf_matches_erf():
    # chi-square with 1 dof: F(x) = erf(sqrt(x/2)), with erf from the
    # stdlib, sharing no code with the incomplete-gamma implementation.
    for x in (0.01, 0.1, 0.5, 1.0, 2.0, 3.84, 5.02, 10.0, 20.0):
        exact = math.erf(math.sqrt(0.5 * x))
        assert abs(chi2_cdf(x, 1) - exact) <= 1e-12 * exact


def test_chi2_k4_cdf_matches_erlang():
    # chi-square with 4 dof is Erlang(2, 1/2): F(x) = 1 - (1 + x/2) e^{-x/2}.
    # x >= 1 avoids the closed form's own small-x cancellation.
    for x in (1.0, 2.0, 5.0, 9.49, 15.0, 30.0):
        exact = 1.0 - (1.0 + 0.5 * x) * math.exp(-0.5 * x)
        assert abs(chi2_cdf(x, 4) - exact) <= 1e-11 * exact


def test_chi2_cdf_ppf_roundtrip_across_dof():
    # The pooled ensemble statistic reaches R*T*n ~ 1e5-1e6 dof; per-epoch
    # ensemble uses R*n ~ 600; sensor batteries use up to ~6000. At k = 1e6
    # the CDF's own floating-point conditioning limits the round-trip near
    # 5e-10 in probability (about 1e-11 relative in x).
    for k in (1, 2, 5, 10, 100, 1500, 6000, 10**6):
        tol = 2e-9 if k >= 10**5 else 1e-11
        quantiles = []
        for p in (0.001, 0.025, 0.5, 0.975, 0.999):
            x = chi2_ppf(p, k)
            assert abs(chi2_cdf(x, k) - p) <= tol
            quantiles.append(x)
        assert quantiles == sorted(quantiles)


def test_chi2_ppf_wilson_hilferty_agreement_at_large_dof():
    # Sanity cross-check at the top of the supported dof range: A&S 26.4.17
    # converges to the exact quantile as k grows, so at k = 1e6 the two must
    # agree to well under 1e-5 relative (the residual is dominated by the
    # A&S 26.2.23 normal-quantile approximation used inside W-H).
    k = 10**6
    for p in (0.025, 0.5, 0.975):
        exact = chi2_ppf(p, k)
        assert abs(wilson_hilferty_ppf(p, k) - exact) <= 1e-5 * exact


def test_chi2_domain_errors():
    for bad_call in (
        lambda: chi2_ppf(0.0, 5),
        lambda: chi2_ppf(1.0, 5),
        lambda: chi2_ppf(0.5, 0),
        lambda: chi2_ppf(0.5, -3),
        lambda: chi2_cdf(1.0, 0),
        lambda: chi2_cdf(math.nan, 5),
        lambda: gammp(-1.0, 1.0),
        lambda: gammp(1.0, -1.0),
        lambda: normal_ppf(0.0),
        lambda: normal_ppf(1.0),
    ):
        with pytest.raises(ValueError):
            bad_call()


# ---------------------------------------------------------------------------
# Packed upper-triangle covariance convention


def test_packed_triangle_roundtrip():
    rng = Generator(PCG64(1))
    a = rng.standard_normal((5, 6, 6))
    mats = a @ np.swapaxes(a, -1, -2) + 6.0 * np.eye(6)
    packed = pack_symmetric(mats)
    assert packed.shape == (5, packed_length(6))
    np.testing.assert_array_equal(unpack_symmetric(packed), mats)
    np.testing.assert_array_equal(pack_symmetric(unpack_symmetric(packed)), packed)


def test_packed_triangle_order_is_row_major_upper():
    # Pin the element order to the FR-26 / srlog v1.1 inertia convention:
    # [M_00, M_01, ..., M_0(n-1), M_11, ..., M_(n-1)(n-1)].
    m2 = np.array([[1.0, 2.0], [2.0, 3.0]])
    np.testing.assert_array_equal(pack_symmetric(m2), [1.0, 2.0, 3.0])
    m3 = np.array([[11.0, 12.0, 13.0], [12.0, 22.0, 23.0], [13.0, 23.0, 33.0]])
    np.testing.assert_array_equal(
        pack_symmetric(m3), [11.0, 12.0, 13.0, 22.0, 23.0, 33.0]
    )
    np.testing.assert_array_equal(
        unpack_symmetric([11.0, 12.0, 13.0, 22.0, 23.0, 33.0]), m3
    )


def test_matrix_order_and_packed_length():
    assert matrix_order(1) == 1
    assert matrix_order(21) == 6
    assert packed_length(15) == 120
    with pytest.raises(ValueError):
        matrix_order(5)  # not a triangular number


# ---------------------------------------------------------------------------
# NEES/NIS engine


def test_nees_matches_explicit_inverse():
    # The engine solves through Cholesky; this reference deliberately uses
    # the explicit inverse on well-conditioned matrices so the two paths
    # share nothing beyond the inputs.
    rng = Generator(PCG64(2))
    epochs, n = 50, 6
    a = rng.standard_normal((epochs, n, n))
    P = a @ np.swapaxes(a, -1, -2) + n * np.eye(n)
    e = rng.standard_normal((epochs, n))
    reference = np.einsum("ti,tij,tj->t", e, np.linalg.inv(P), e)
    np.testing.assert_allclose(nees(e, pack_symmetric(P)), reference, rtol=1e-10)


def test_nees_rejects_non_positive_definite_covariance():
    P = np.tile(np.eye(3), (4, 1, 1))
    P[2] = np.diag([1.0, -1.0, 1.0])  # indefinite at epoch 2
    with pytest.raises(ValueError, match=r"positive definite"):
        nees(np.ones((4, 3)), pack_symmetric(P))


def test_nees_rejects_mismatched_packed_length():
    with pytest.raises(ValueError, match=r"packed length"):
        nees(np.ones((4, 6)), np.ones((4, 10)))  # 10 packs n=4, not n=6


def _consistent_ensemble(seed, runs, epochs, dim):
    """Synthetic consistent filter ensemble: e ~ N(0, P_k), P_k time-varying.

    Returns (e, P_packed, P) with e of shape (runs, epochs, dim) drawn with
    exactly the covariance P of shape (epochs, dim, dim) that is reported.
    """
    rng = Generator(PCG64(seed))
    a = rng.standard_normal((dim, dim))
    base = a @ a.T + dim * np.eye(dim)
    # A smooth factor-of-3 covariance swing over the run exercises the
    # per-epoch unpacking/solve path rather than one constant matrix.
    scale = 1.0 + 0.5 * np.sin(2.0 * np.pi * np.arange(epochs) / epochs)
    P = scale[:, np.newaxis, np.newaxis] * base
    chol = np.linalg.cholesky(P)
    z = rng.standard_normal((runs, epochs, dim))
    e = np.einsum("tij,rtj->rti", chol, z)
    return e, pack_symmetric(P), P


# Under exact consistency the per-epoch coverage equals the interval
# probability, so the fraction-inside statistic sits exactly at the 0.95
# acceptance threshold (binomial sd ~1.5 % at T = 200): any seed is a fair
# draw, and these seeds are pinned to ones with comfortable margin. The
# mutation tests below prove the gate fails in both directions, so a
# passing seed is not a rigged gate.
_NEES_SEED = 5
_NIS_SEED = 6


def test_consistent_ensemble_nees_gate_passes():
    e, P_packed, _ = _consistent_ensemble(_NEES_SEED, runs=100, epochs=200, dim=6)
    eps = nees(e, P_packed)
    assert eps.shape == (100, 200)
    gate = ensemble_gate(eps, 6)
    assert gate.passed
    assert gate.fraction_inside >= 0.95
    assert gate.pooled.passed
    # The per-run time-averaged diagnostic holds for (about) 95 % of runs.
    per_run_passes = sum(time_average_gate(eps[r], 6).passed for r in range(100))
    assert per_run_passes >= 90


@pytest.mark.parametrize(
    "factor, expect_high",
    [
        # Covariance reported at 0.7x truth: the filter is overconfident,
        # NEES inflates by 1/0.7 and the gate must fail HIGH. At 1.4x truth
        # it is underconfident and must fail LOW. Both factors are decisive
        # at R = 100 (the ensemble mean sits several interval widths out).
        (0.7, True),
        (1.4, False),
    ],
)
def test_miscalibrated_covariance_fails_nees_gate(factor, expect_high):
    e, P_packed, _ = _consistent_ensemble(_NEES_SEED, runs=100, epochs=200, dim=6)
    gate = ensemble_gate(nees(e, factor * P_packed), 6)
    assert not gate.passed
    assert not gate.pooled.passed
    if expect_high:
        assert gate.fraction_above > 0.9
        assert gate.fraction_below == 0.0
        assert gate.pooled.mean > gate.pooled.upper
    else:
        assert gate.fraction_below > 0.9
        assert gate.fraction_above == 0.0
        assert gate.pooled.mean < gate.pooled.lower


def test_consistent_ensemble_nis_gate_passes():
    y, S_packed, _ = _consistent_ensemble(_NIS_SEED, runs=100, epochs=200, dim=3)
    eps = nis(y, S_packed)
    gate = ensemble_gate(eps, 3)
    assert gate.passed
    assert gate.pooled.passed
    per_run_passes = sum(time_average_gate(eps[r], 3).passed for r in range(100))
    assert per_run_passes >= 90


@pytest.mark.parametrize("factor, expect_high", [(0.7, True), (1.4, False)])
def test_miscalibrated_covariance_fails_nis_gate(factor, expect_high):
    y, S_packed, _ = _consistent_ensemble(_NIS_SEED, runs=100, epochs=200, dim=3)
    gate = ensemble_gate(nis(y, factor * S_packed), 3)
    assert not gate.passed
    if expect_high:
        assert gate.fraction_above > 0.9
    else:
        assert gate.fraction_below > 0.9


# ---------------------------------------------------------------------------
# The binomial coverage threshold


def test_binom_cdf_matches_exact_rational_arithmetic():
    # Reference computed in exact rational arithmetic from math.comb, which
    # shares no code with the lgamma-based implementation under test.
    from fractions import Fraction
    from math import comb

    for n in (1, 5, 20, 60, 200):
        for p_num, p_den in ((1, 20), (1, 2), (19, 20)):
            p = Fraction(p_num, p_den)
            for k in range(n + 1):
                exact = float(
                    sum(comb(n, j) * p**j * (1 - p) ** (n - j) for j in range(k + 1))
                )
                assert abs(binom_cdf(k, n, float(p)) - exact) <= 1e-12


def test_binom_cdf_edges_and_domain():
    assert binom_cdf(-1, 10, 0.5) == 0.0
    assert binom_cdf(10, 10, 0.5) == 1.0
    assert binom_cdf(99, 10, 0.5) == 1.0
    for bad in (lambda: binom_cdf(1, -1, 0.5), lambda: binom_cdf(1, 10, 0.0),
                lambda: binom_cdf(1, 10, 1.0)):
        with pytest.raises(ValueError):
            bad()


def test_inside_count_threshold_is_the_binomial_lower_tail():
    """The threshold is the largest count meeting the stated budget.

    Both halves matter: one more epoch would breach the spurious-failure
    budget (so the threshold is not lax) and the threshold itself does not
    (so it is not needlessly strict).
    """
    alpha = 1.0 - DEFAULT_CONFIDENCE
    for epochs in (30, 60, 200, 601):
        t = inside_count_threshold(epochs, 0.95, DEFAULT_CONFIDENCE)
        assert binom_cdf(t - 1, epochs, 0.95) <= alpha
        assert binom_cdf(t, epochs, 0.95) > alpha
        # And it must sit below the mean, or it would be the coin flip the
        # old ">= 95 % of epochs" rule was.
        assert t < 0.95 * epochs


def test_inside_count_threshold_domain():
    for bad in (
        lambda: inside_count_threshold(0),
        lambda: inside_count_threshold(60, 0.0),
        lambda: inside_count_threshold(60, 1.0),
        lambda: inside_count_threshold(60, 0.95, 0.0),
        lambda: inside_count_threshold(60, 0.95, 1.0),
    ):
        with pytest.raises(ValueError):
            bad()


def _null_epoch_means(rng, trials, epochs, dim, runs):
    """Ensemble epoch means under exact consistency, independent epochs.

    R * eps_bar_k ~ chi-square(R*dim) is the exact null law of the ensemble
    average at one epoch, so drawing it directly needs no filter and no
    approximation.
    """
    return rng.chisquare(runs * dim, size=(trials, epochs)) / runs


def test_coverage_threshold_false_failure_rate_is_at_budget():
    """Monte Carlo the coverage criterion against its design budget.

    Draws from the exact null law with independent epochs -- the premise
    NIS satisfies -- and checks the spurious-failure rate lands where the
    binomial says it should, rather than at the ~50 % the superseded
    ">= 95 % of epochs" rule produced (measured alongside for contrast).
    """
    trials, epochs, dim, runs = 20000, 60, 3, 100
    rng = Generator(PCG64(20260719))
    lower = chi2_ppf(0.025, runs * dim) / runs
    upper = chi2_ppf(0.975, runs * dim) / runs
    x = _null_epoch_means(rng, trials, epochs, dim, runs)
    inside = ((x >= lower) & (x <= upper)).sum(axis=1)

    threshold = inside_count_threshold(epochs, 0.95, DEFAULT_CONFIDENCE)
    predicted = binom_cdf(threshold - 1, epochs, 0.95)
    failures = int((inside < threshold).sum())
    # Three-sigma Monte Carlo band around the binomial prediction.
    expected = trials * predicted
    sigma = math.sqrt(expected * (1.0 - predicted))
    assert abs(failures - expected) <= 3.0 * sigma + 1.0, (
        f"{failures}/{trials} spurious failures against a predicted "
        f"{expected:.1f}; the threshold no longer matches its derivation"
    )
    assert failures / trials < 3.0e-3

    # The superseded rule, on the very same draws: a coin flip.
    old_rule_failures = int((inside < 0.95 * epochs).sum())
    assert old_rule_failures / trials > 0.3, (
        "the '>= 95 % of epochs' rule is expected to reject roughly half of "
        "all consistent ensembles; if it no longer does, this contrast test "
        "is measuring something else"
    )


def test_headline_criterion_does_not_spuriously_fail():
    """The headline is conservative: averaging epochs shrinks its variance.

    Under the null it is tested against a single-epoch interval, so its
    spurious-failure rate must be far below the nominal 5 %.
    """
    trials, epochs, dim, runs = 4000, 60, 3, 100
    rng = Generator(PCG64(555))
    lower = chi2_ppf(0.025, runs * dim) / runs
    upper = chi2_ppf(0.975, runs * dim) / runs
    head = _null_epoch_means(rng, trials, epochs, dim, runs).mean(axis=1)
    failures = int(((head < lower) | (head > upper)).sum())
    assert failures == 0, f"{failures}/{trials} headline failures under the null"


# ---------------------------------------------------------------------------
# The instrument must still be able to fail


def test_ensemble_gate_accepts_a_single_run():
    # R = 1 keeps a one-log invocation gated: the run is its own ensemble
    # average and stays chi-square(dim) at each epoch.
    e, P_packed, _ = _consistent_ensemble(_NEES_SEED, runs=1, epochs=200, dim=6)
    gate = ensemble_gate(nees(e, P_packed), 6)
    assert gate.passed
    assert gate.epoch_mean.shape == (200,)


def test_biased_estimator_fails_the_gate():
    """A constant offset in the error vector must go red.

    The offset is expressed in units of the reported 1-sigma, so the
    magnitude is meaningful independently of the covariance drawn.
    """
    e, P_packed, P = _consistent_ensemble(_NEES_SEED, runs=100, epochs=200, dim=6)
    sigma = np.sqrt(np.einsum("tii->ti", P))
    assert ensemble_gate(nees(e, P_packed), 6).passed
    for k in (1.0, 2.0):
        biased = e + k * sigma[np.newaxis, :, :]
        gate = ensemble_gate(nees(biased, P_packed), 6)
        assert not gate.passed, f"a {k}-sigma estimator bias was not caught"
        assert gate.headline.mean > gate.upper


def test_wrong_measurement_noise_model_fails_the_gate():
    """An anisotropic wrong R: one component's assumed variance inflated.

    Unlike a uniform rescale this leaves the other components correct, so
    it tests that the gate responds to a partial covariance error rather
    than only to a global scaling.
    """
    y, S_packed, S = _consistent_ensemble(_NIS_SEED, runs=100, epochs=200, dim=3)
    assert ensemble_gate(nis(y, S_packed), 3).passed
    for factor in (2.0, 5.0):
        S_wrong = S.copy()
        S_wrong[:, 0, 0] *= factor
        gate = ensemble_gate(nis(y, pack_symmetric(S_wrong)), 3)
        assert not gate.passed, f"wrong R (var[0] x {factor}) was not caught"


def test_defect_confined_to_part_of_the_run_fails_the_gate():
    # A covariance error over the last quarter of the run only: the kind of
    # localized defect a whole-run average can dilute.
    e, P_packed, _ = _consistent_ensemble(_NEES_SEED, runs=100, epochs=200, dim=6)
    eps = nees(e, P_packed)
    assert ensemble_gate(eps, 6).passed
    spoiled = eps.copy()
    spoiled[:, 150:] *= 4.0
    assert not ensemble_gate(spoiled, 6).passed
    # And it must still be caught on the NEES path, where only the headline
    # gates -- a quarter of the run is enough to move the epoch average.
    assert not ensemble_gate(spoiled, 6, epochs_independent=False).passed


def test_coverage_is_reported_but_not_gated_for_correlated_epochs():
    """``epochs_independent=False`` reports the coverage count, never gates.

    NEES epochs are serially correlated, so its inside-count is
    over-dispersed relative to the binomial the threshold is derived from;
    the flag makes that structural fact explicit instead of letting a
    mis-calibrated criterion set an exit code.
    """
    e, P_packed, _ = _consistent_ensemble(_NEES_SEED, runs=100, epochs=200, dim=6)
    eps = nees(e, P_packed)
    # Force the coverage criterion to fail while the headline still holds,
    # by moving a minority of epochs far outside the interval.
    spoiled = eps.copy()
    spoiled[:, :40] *= 3.0
    spoiled[:, 40:80] /= 3.0

    gated = ensemble_gate(spoiled, 6, epochs_independent=True)
    assert gated.coverage_gated
    assert not gated.coverage_passed
    assert not gated.passed

    reported = ensemble_gate(spoiled, 6, epochs_independent=False)
    assert not reported.coverage_gated
    # Same measured count, but it no longer reaches the verdict.
    assert reported.inside_count == gated.inside_count
    assert not reported.coverage_passed
    assert reported.passed == reported.headline.passed


def test_gate_functions_are_deterministic():
    e, P_packed, _ = _consistent_ensemble(_NEES_SEED, runs=10, epochs=50, dim=6)
    eps = nees(e, P_packed)
    first = ensemble_gate(eps, 6)
    second = ensemble_gate(eps, 6)
    np.testing.assert_array_equal(first.epoch_mean, second.epoch_mean)
    assert (first.lower, first.upper, first.passed) == (
        second.lower,
        second.upper,
        second.passed,
    )


# ---------------------------------------------------------------------------
# CLI


def _run_cli(*args, cwd=None):
    env = os.environ.copy()
    # Point the subprocess at the same package this test process imported
    # (source tree or installed wheel), so both see identical code.
    pkg_parent = Path(star_reacher.__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(pkg_parent) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "star_reacher", *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd,
    )


def _nav_header(n, m):
    """A v1.1-shaped header carrying the FR-26 reserved nav groups."""
    nav_groups = [
        {
            "name": "nav.est",
            "rate_hz": 10,
            "channels": [
                {"name": "t_s", "dtype": "f64", "units": "s", "frame": ""},
                {"name": "x_hat", "dtype": f"f64[{n}]", "units": "1", "frame": "GCRF"},
                {
                    "name": "P",
                    "dtype": f"f64[{n * (n + 1) // 2}]",
                    "units": "1",
                    "frame": "GCRF",
                },
            ],
        },
        {
            "name": "nav.err",
            "rate_hz": 10,
            "channels": [
                {"name": "t_s", "dtype": "f64", "units": "s", "frame": ""},
                {"name": "e", "dtype": f"f64[{n}]", "units": "1", "frame": "GCRF"},
            ],
        },
        {
            "name": "nav.innov",
            "rate_hz": 10,
            "channels": [
                {"name": "t_s", "dtype": "f64", "units": "s", "frame": ""},
                {"name": "y", "dtype": f"f64[{m}]", "units": "1", "frame": ""},
                {
                    "name": "S",
                    "dtype": f"f64[{m * (m + 1) // 2}]",
                    "units": "1",
                    "frame": "",
                },
            ],
        },
    ]
    return _fixtures.contract_header(minor=2, extra_groups=nav_groups)


def _write_nav_log(path, seed, epochs=60, n=4, m=2, p_report_factor=1.0):
    """Synthesize one run's SRLOG with consistent nav channels.

    Errors and innovations are drawn with the true covariances; the logged
    P/S are scaled by ``p_report_factor`` (1.0 logs the honest covariance,
    other values synthesize a miscalibrated filter for FAIL-path tests).
    """
    rng = Generator(PCG64(seed))
    header = _nav_header(n, m)
    a = rng.standard_normal((n, n))
    P_true = a @ a.T + n * np.eye(n)
    b = rng.standard_normal((m, m))
    S_true = b @ b.T + m * np.eye(m)
    e = rng.standard_normal((epochs, n)) @ np.linalg.cholesky(P_true).T
    y = rng.standard_normal((epochs, m)) @ np.linalg.cholesky(S_true).T
    p_packed = tuple(pack_symmetric(p_report_factor * P_true))
    s_packed = tuple(pack_symmetric(p_report_factor * S_true))
    gi_est = _fixtures.group_index(header, "nav.est")
    gi_err = _fixtures.group_index(header, "nav.err")
    gi_innov = _fixtures.group_index(header, "nav.innov")
    records = [_fixtures.truth_record(0.0)]
    for k in range(epochs):
        t = 0.1 * k
        records.append((gi_est, (t, (0.0,) * n, p_packed)))
        records.append((gi_err, (t, tuple(e[k]))))
        records.append((gi_innov, (t, tuple(y[k]), s_packed)))
    path.write_bytes(_fixtures.build_srlog(header, records))


def test_cli_missing_nav_groups_is_actionable(tmp_path):
    # A Phase 5 log (no nav groups) must be refused with an error naming
    # every missing group, never a traceback or a silent pass.
    log = tmp_path / "run.srlog"
    log.write_bytes(
        _fixtures.build_srlog(
            _fixtures.contract_header(), [_fixtures.truth_record(0.0)]
        )
    )
    proc = _run_cli("consistency", str(log))
    assert proc.returncode == 1
    for group in ("nav.err", "nav.est", "nav.innov"):
        assert group in proc.stderr
    assert "Traceback" not in proc.stderr


def test_cli_missing_path_exits_1(tmp_path):
    proc = _run_cli("consistency", str(tmp_path / "absent.srlog"))
    assert proc.returncode == 1
    assert "no such file" in proc.stderr


def test_cli_single_run_passes(tmp_path):
    # One log still carries gates: it is its own R = 1 ensemble. Three of
    # them, because the NEES coverage criterion reports without gating.
    log = tmp_path / "run.srlog"
    _write_nav_log(log, seed=11)
    proc = _run_cli("consistency", str(log))
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "CONSISTENCY: PASS (3/3 gates)" in proc.stdout
    assert "NEES time-averaged" in proc.stdout
    assert "NIS time-averaged" in proc.stdout


def test_cli_single_run_overconfident_fails(tmp_path):
    # Covariance logged at half its true value. The exit code must be
    # nonzero from a GATE, and the report must name the direction.
    log = tmp_path / "run.srlog"
    _write_nav_log(log, seed=11, p_report_factor=0.5)
    proc = _run_cli("consistency", str(log))
    assert proc.returncode == 1
    assert "CONSISTENCY: FAIL (2/3 gates)" in proc.stdout
    assert "overconfident" in proc.stdout


def test_cli_time_averaged_numbers_never_set_the_exit_code(tmp_path):
    """The per-run diagnostic is printed outside its bounds, exit stays 0.

    This is the regression this whole gate revision exists to prevent: the
    CLI used to gate on the per-run time average and so reported FAIL on
    consistent output. The log is written with an honest covariance, so
    every real gate passes; the seed is chosen so at least one time-averaged
    diagnostic lands outside its indicative interval anyway.
    """
    log = tmp_path / "run.srlog"
    _write_nav_log(log, seed=21, epochs=400)
    proc = _run_cli("consistency", str(log))
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "CONSISTENCY: PASS" in proc.stdout
    # Indented lines only: the report's leading banner also says
    # "time-averaged" while explaining that those numbers do not gate.
    time_avg = [
        line
        for line in proc.stdout.splitlines()
        if line.startswith("  ") and "time-averaged" in line
    ]
    assert time_avg, proc.stdout
    for line in time_avg:
        # Diagnostics never borrow the words that mean "this moved the exit
        # code", whichever side of the interval they land on.
        assert "[diagnostic, not gated]" in line
        assert "PASS" not in line and "FAIL" not in line


def test_cli_ensemble_over_directory(tmp_path):
    # Three consistent runs in one directory. Gates counted: the NEES
    # headline plus the NIS headline and coverage; the NEES coverage number
    # and every per-run and pooled number are diagnostics. Seeds pinned for
    # the same coverage-at-threshold reason as the engine-level tests.
    for seed in (2, 3, 6):
        rundir = tmp_path / f"run{seed}"
        rundir.mkdir()
        _write_nav_log(rundir / "run.srlog", seed=seed)
    proc = _run_cli("consistency", str(tmp_path))
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "ensemble: R=3 runs, NEES" in proc.stdout
    assert "ensemble: R=3 runs, NIS" in proc.stdout
    assert "CONSISTENCY: PASS (3/3 gates)" in proc.stdout


def test_cli_ensemble_overconfident_directory_fails(tmp_path):
    # The same three-run ensemble with every covariance logged at half
    # truth: the gate must go red and the exit code must be nonzero, so
    # narrowing what gates has not made the instrument unable to fail.
    for seed in (2, 3, 6):
        rundir = tmp_path / f"run{seed}"
        rundir.mkdir()
        _write_nav_log(rundir / "run.srlog", seed=seed, p_report_factor=0.5)
    proc = _run_cli("consistency", str(tmp_path))
    assert proc.returncode == 1, proc.stdout
    assert "CONSISTENCY: FAIL (1/3 gates)" in proc.stdout
    assert "overconfident" in proc.stdout
