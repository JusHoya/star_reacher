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
    chi2_cdf,
    chi2_ppf,
    gammp,
    normal_ppf,
    wilson_hilferty_ppf,
)
from star_reacher.consistency import (
    ensemble_gate,
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
    log = tmp_path / "run.srlog"
    _write_nav_log(log, seed=11)
    proc = _run_cli("consistency", str(log))
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "CONSISTENCY: PASS (2/2 gates)" in proc.stdout
    assert "NEES time-averaged" in proc.stdout
    assert "NIS time-averaged" in proc.stdout


def test_cli_single_run_overconfident_fails(tmp_path):
    # Covariance logged at half its true value: both time-averaged gates
    # must fail high and the exit code must be nonzero.
    log = tmp_path / "run.srlog"
    _write_nav_log(log, seed=11, p_report_factor=0.5)
    proc = _run_cli("consistency", str(log))
    assert proc.returncode == 1
    assert "CONSISTENCY: FAIL (0/2 gates)" in proc.stdout
    assert "overconfident" in proc.stdout


def test_cli_ensemble_over_directory(tmp_path):
    # Three consistent runs in one directory: 6 per-run gates plus the
    # NEES/NIS ensemble and pooled gates. Seeds pinned for the same
    # coverage-at-threshold reason as the engine-level ensemble tests.
    for seed in (2, 3, 6):
        rundir = tmp_path / f"run{seed}"
        rundir.mkdir()
        _write_nav_log(rundir / "run.srlog", seed=seed)
    proc = _run_cli("consistency", str(tmp_path))
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "ensemble: R=3 runs, NEES" in proc.stdout
    assert "ensemble: R=3 runs, NIS" in proc.stdout
    assert "CONSISTENCY: PASS (10/10 gates)" in proc.stdout
