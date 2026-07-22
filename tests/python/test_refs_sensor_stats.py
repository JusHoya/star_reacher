"""Validate the sensor error statistics of ``tests/refs/sensor_stats.py``.

Two Phase 6 exit criteria are served:

* criterion 1 -- star-tracker error statistics inside two-sided 95 % chi-square
  bounds over 1,000 draws;
* criterion 6 -- external-nav-fix and altimeter error statistics inside
  two-sided 95 % chi-square bounds over 1,000 seeded draws.

Both criteria are statements about a STATISTIC, so the reference is validated by
showing that the statistic behaves as claimed on draws whose distribution is
known exactly by construction: it accepts correctly scaled errors at close to
the nominal 95 % rate, and it rejects mis-scaled ones. A gate that has never
been shown to fail is not evidence, so every sensor here has an accompanying
mutation test.

The sharpest structural check is the star tracker's INVARIANCE to the
deterministic aberration factor: the extraction of eq:optical:extract removes
exactly what the measurement of eq:optical:stmodel inserted, so inserting a
20 arcsec field rotation must leave the statistic unchanged. That fails loudly
if the composition order of D-7 is transcribed backwards, which is the defect
Chapter ch:sensors-optical explicitly warns about.

These tests are pure NumPy and require no compiled core.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(REPO_ROOT / "tests" / "refs"))
import sensor_stats as ss  # noqa: E402
from aberration import ARCSEC_PER_RAD, SPEED_OF_LIGHT_MPS  # noqa: E402
from quaternions import quat_conj, quat_mul, quat_normalize, quat_to_dcm  # noqa: E402

# Exit criteria 1 and 6 both specify 1,000 draws.
DRAWS = 1000

# A representative arcsecond-class star tracker: tight about the two transverse
# axes, an order looser about the boresight, per the chapter's note that the
# about-boresight sigma is typically the largest.
ST_SIGMAS = np.array([8.0e-6, 8.0e-6, 5.0e-5])
Q_TRUE = quat_normalize(np.array([0.7, 0.1, -0.3, 0.6]))
BORESIGHT_I = np.array([0.0, 0.0, 1.0])
BETA = np.array([29780.0, 0.0, 0.0]) / SPEED_OF_LIGHT_MPS


# --- Star tracker (criterion 1) --------------------------------------------


def test_aberration_field_rotation_reproduces_the_first_order_displacement():
    """Equation eq:optical:rho's own consistency claim, checked at the boresight.

    ``rho x b = beta - (b . beta) b`` is what licenses treating the aberration
    field as a rigid rotation of the sky over a narrow field of view.
    """
    rng = np.random.default_rng(1)
    for _ in range(200):
        boresight = rng.standard_normal(3)
        boresight = boresight / np.linalg.norm(boresight)
        residual = ss.aberration_rotation_consistency(boresight, BETA)
        assert residual < 1e-18, f"residual {residual:.3e}"


def test_aberration_rotation_magnitude_is_the_expected_arcseconds():
    """``|rho| = beta sin(theta)``, peaking at the 20.49 arcsec of the chapter."""
    rho = ss.aberration_rotation_vector(BORESIGHT_I, BETA)
    # Boresight perpendicular to beta here, so the full beta appears.
    assert float(np.linalg.norm(rho)) * ARCSEC_PER_RAD == pytest.approx(20.49, abs=0.01)


def test_aberrated_boresight_points_away_from_the_velocity():
    """The binding consistency check stated after equation eq:optical:qab.

    The star whose APPARENT position sits on the boresight has its CATALOGUE
    position displaced against the velocity, so the boresight direction implied
    by the aberrated attitude must move away from ``beta``. This is the sign
    check that catches an inverted aberration quaternion.
    """
    boresight_b = quat_to_dcm(Q_TRUE) @ BORESIGHT_I
    rho = ss.aberration_rotation_vector(BORESIGHT_I, BETA)
    q_ab = ss.aberration_quaternion(rho)
    aberrated_attitude = quat_mul(q_ab, Q_TRUE)

    implied_i = quat_to_dcm(aberrated_attitude).T @ boresight_b
    beta_hat = BETA / np.linalg.norm(BETA)
    assert float(implied_i @ beta_hat) < float(BORESIGHT_I @ beta_hat)
    # And the displacement magnitude is the field rotation, to first order.
    displacement = np.linalg.norm(implied_i - BORESIGHT_I)
    assert displacement == pytest.approx(float(np.linalg.norm(rho)), rel=1e-6)


def test_error_extraction_inverts_the_noise_construction_exactly():
    """Equation eq:optical:extract must return the drawn epsilon identically.

    Exactness here is what makes the statistic of eq:optical:ststat exactly
    chi-square rather than approximately so.
    """
    rng = np.random.default_rng(2)
    q_ab = ss.aberration_quaternion(ss.aberration_rotation_vector(BORESIGHT_I, BETA))
    worst = 0.0
    for _ in range(3000):
        epsilon = rng.standard_normal(3) * ST_SIGMAS
        q_meas = ss.star_tracker_measurement(Q_TRUE, epsilon, q_ab)
        recovered = ss.extract_error_vector(q_meas, Q_TRUE, q_ab)
        worst = max(worst, float(np.max(np.abs(recovered - epsilon))))
    assert worst < 1e-14, f"extraction residual {worst:.3e} rad"


def test_statistic_is_invariant_to_the_deterministic_aberration_factor():
    """The structural check on the D-7 composition order of eq:optical:stmodel.

    ``q_meas = q_ab (x) q_true (x) dq_n`` with the extraction removing
    ``q_ab (x) q_true``. Inserting a 20 arcsec field rotation therefore cannot
    move the statistic. Reversing the composition -- the Markley--Crassidis
    reading the chapter warns is a transcription hazard -- would leave a
    residual of order the field rotation, which at these sigmas is a factor of
    two in the statistic.
    """
    with_aberration = ss.star_tracker_draws(
        Q_TRUE, ST_SIGMAS, DRAWS, np.random.default_rng(7), BORESIGHT_I, BETA
    )
    without = ss.star_tracker_draws(Q_TRUE, ST_SIGMAS, DRAWS, np.random.default_rng(7))
    assert np.max(np.abs(with_aberration - without)) < 1e-8 * np.mean(without)


def test_reversed_composition_is_rejected_by_the_gate():
    """Demonstrate the invariance test above has teeth.

    Extracting with ``q_true (x) q_ab`` instead of ``q_ab (x) q_true`` -- the
    Markley--Crassidis reading Chapter ch:sensors-optical flags as a
    transcription hazard -- leaves the 20 arcsec field rotation in the residual.
    The per-draw inflation varies with geometry, so the meaningful statement is
    on the ENSEMBLE: the criterion-1 gate must reject it outright. Without this
    demonstration the invariance test could be passing vacuously.
    """
    from quaternions import rotation_vector_from_quat

    rng = np.random.default_rng(3)
    q_ab = ss.aberration_quaternion(ss.aberration_rotation_vector(BORESIGHT_I, BETA))
    wrong_deterministic = quat_mul(Q_TRUE, q_ab)

    correct = np.empty(DRAWS)
    wrong = np.empty(DRAWS)
    for i in range(DRAWS):
        epsilon = rng.standard_normal(3) * ST_SIGMAS
        q_meas = ss.star_tracker_measurement(Q_TRUE, epsilon, q_ab)
        correct[i] = ss.star_tracker_chi2(q_meas, Q_TRUE, ST_SIGMAS, q_ab)
        wrong_eps = rotation_vector_from_quat(
            quat_mul(quat_conj(wrong_deterministic), q_meas)
        )
        wrong[i] = float(np.sum((wrong_eps / ST_SIGMAS) ** 2))

    assert ss.evaluate_gate("correct composition", correct, 3).passed
    reversed_result = ss.evaluate_gate("reversed composition", wrong, 3)
    assert not reversed_result.passed, reversed_result.describe()
    assert reversed_result.statistic > 10.0 * float(np.mean(correct))


def test_star_tracker_ensemble_gate_accepts_correct_draws():
    """Exit criterion 1's star-tracker clause, on draws known to be correct."""
    statistics = ss.star_tracker_draws(
        Q_TRUE, ST_SIGMAS, DRAWS, np.random.default_rng(11), BORESIGHT_I, BETA
    )
    result = ss.evaluate_gate("star tracker", statistics, 3)
    assert result.passed, result.describe()
    # And the bounds are the ones the chapter prints.
    assert result.lower == pytest.approx(2.850, abs=0.005)
    assert result.upper == pytest.approx(3.154, abs=0.005)


def test_star_tracker_gate_rejects_a_mis_scaled_sigma():
    """Mutation test: evaluating with a 20 % understated sigma must FAIL."""
    statistics = ss.star_tracker_draws(
        Q_TRUE, ST_SIGMAS, DRAWS, np.random.default_rng(12)
    )
    # Recompute the statistic as if the configured sigmas were 20 % small; the
    # normalized quadratic form then inflates by 1/0.8**2.
    inflated = statistics / 0.8**2
    result = ss.evaluate_gate("star tracker (mis-scaled)", inflated, 3)
    assert not result.passed, result.describe()


def test_star_tracker_gate_coverage_is_near_nominal():
    """Repeated independent ensembles must pass about 95 % of the time."""
    accepted = 0
    trials = 120
    for trial in range(trials):
        statistics = ss.star_tracker_draws(
            Q_TRUE, ST_SIGMAS, DRAWS, np.random.default_rng(9000 + trial)
        )
        accepted += int(ss.evaluate_gate("st", statistics, 3).passed)
    coverage = accepted / trials
    assert 0.88 <= coverage <= 1.0, f"empirical coverage {coverage:.3f}"


# --- Sun sensor -------------------------------------------------------------


def test_sun_sensor_gate_accepts_correct_draws():
    """The tangent-plane statistic of sec:optical:stats, bounds [1.878, 2.126]."""
    u_body = np.array([0.3, 0.5, 0.81])
    statistics = ss.sun_sensor_draws(u_body, 1.0e-3, DRAWS, np.random.default_rng(21))
    result = ss.evaluate_gate("sun sensor", statistics, 2)
    assert result.passed, result.describe()
    assert result.lower == pytest.approx(1.878, abs=0.005)
    assert result.upper == pytest.approx(2.126, abs=0.005)


def test_sun_sensor_tangent_basis_is_orthonormal_and_transverse():
    """The basis must be orthonormal and perpendicular to the true direction."""
    rng = np.random.default_rng(22)
    for _ in range(300):
        u = rng.standard_normal(3)
        u = u / np.linalg.norm(u)
        e1, e2 = ss.tangent_basis(u)
        assert float(np.linalg.norm(e1)) == pytest.approx(1.0, abs=1e-14)
        assert float(np.linalg.norm(e2)) == pytest.approx(1.0, abs=1e-14)
        assert abs(float(e1 @ e2)) < 1e-14
        assert abs(float(e1 @ u)) < 1e-14
        assert abs(float(e2 @ u)) < 1e-14


def test_sun_sensor_tangent_basis_handles_the_polar_degeneracy():
    """A line of sight along ``z`` triggers the documented ``x_hat`` fallback."""
    e1, e2 = ss.tangent_basis(np.array([0.0, 0.0, 1.0]))
    assert abs(float(e1 @ np.array([0.0, 0.0, 1.0]))) < 1e-14
    assert abs(float(e1 @ e2)) < 1e-14


# --- External nav fix and altimeter (criterion 6) --------------------------


NAV_CONFIG = ss.NavFixConfig(
    sigma_r_m=np.array([5.0, 5.0, 9.0]),
    sigma_v_mps=np.array([0.05, 0.05, 0.09]),
)
R_TRUTH = np.array([6.9e6, -1.1e6, 2.2e6])
V_TRUTH = np.array([1.2e3, 7.3e3, -0.4e3])


def _nav_fix_statistics(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    q_r = np.empty(DRAWS)
    q_v = np.empty(DRAWS)
    for k in range(DRAWS):
        r_meas, v_meas = ss.nav_fix_measurement(R_TRUTH, V_TRUTH, NAV_CONFIG, rng)
        q_r[k] = ss.nav_fix_chi2(r_meas, R_TRUTH, NAV_CONFIG.sigma_r_m)
        q_v[k] = ss.nav_fix_chi2(v_meas, V_TRUTH, NAV_CONFIG.sigma_v_mps)
    return q_r, q_v


def test_nav_fix_position_and_velocity_gates_accept_correct_draws():
    """Exit criterion 6's nav-fix clause, both fixes, bounds [2.850, 3.154].

    The seed is pinned deliberately. A two-sided 95 % interval rejects about one
    correct ensemble in twenty by construction, so a seeded gate is the only
    form that is both meaningful and reproducible; the coverage test below is
    what establishes the rejection rate is the nominal one rather than a
    systematic error.
    """
    q_r, q_v = _nav_fix_statistics(32)
    for label, statistics in (("position", q_r), ("velocity", q_v)):
        result = ss.evaluate_gate(f"nav fix {label}", statistics, 3)
        assert result.passed, result.describe()
        assert result.lower == pytest.approx(2.850, abs=0.005)
        assert result.upper == pytest.approx(3.154, abs=0.005)


def test_nav_fix_gate_rejects_an_anisotropy_error():
    """Mutation test: swapping the per-axis sigmas must be caught.

    The configuration is anisotropic (9 m on the third axis against 5 m on the
    others), so evaluating the statistic with the axes permuted mis-normalizes
    two of three terms while leaving the total error magnitude untouched -- a
    defect no isotropic check would find.
    """
    rng = np.random.default_rng(32)
    permuted = np.array(
        [NAV_CONFIG.sigma_r_m[2], NAV_CONFIG.sigma_r_m[0], NAV_CONFIG.sigma_r_m[1]]
    )
    statistics = np.empty(DRAWS)
    for k in range(DRAWS):
        r_meas, _ = ss.nav_fix_measurement(R_TRUTH, V_TRUTH, NAV_CONFIG, rng)
        statistics[k] = ss.nav_fix_chi2(r_meas, R_TRUTH, permuted)
    result = ss.evaluate_gate("nav fix (permuted sigmas)", statistics, 3)
    assert not result.passed, result.describe()


def test_altimeter_gate_accepts_correct_draws():
    """Exit criterion 6's altimeter clause, chi2(1), bounds [0.914, 1.090]."""
    rng = np.random.default_rng(41)
    h_true, sigma_h = 412000.0, 3.5
    statistics = np.array(
        [
            ss.altimeter_chi2(ss.altimeter_measurement(h_true, sigma_h, rng), h_true, sigma_h)
            for _ in range(DRAWS)
        ]
    )
    result = ss.evaluate_gate("altimeter", statistics, 1)
    assert result.passed, result.describe()
    assert result.lower == pytest.approx(0.914, abs=0.005)
    assert result.upper == pytest.approx(1.090, abs=0.005)


def test_nav_fix_gate_coverage_is_near_nominal():
    """Repeated independent nav-fix ensembles must pass about 95 % of the time."""
    accepted = 0
    trials = 60
    for trial in range(trials):
        q_r, _ = _nav_fix_statistics(6000 + trial)
        accepted += int(ss.evaluate_gate("nav fix", q_r, 3).passed)
    coverage = accepted / trials
    assert 0.85 <= coverage <= 1.0, f"empirical coverage {coverage:.3f}"


def test_altimeter_per_run_bias_breaks_the_ensemble_mean_gate():
    """The chapter's stated reason for zeroing ``sigma_b`` in the gate scenario.

    A per-run bias makes the per-sample statistics DEPENDENT: every sample
    carries the same offset, so the mean statistic is inflated by
    ``(b/sigma)**2`` rather than averaging out. Demonstrated here so the
    chapter's exclusion is shown to be necessary, not merely stated.
    """
    rng = np.random.default_rng(42)
    h_true, sigma_h = 412000.0, 3.5
    bias = 2.0 * sigma_h
    statistics = np.array(
        [
            ss.altimeter_chi2(
                ss.altimeter_measurement(h_true, sigma_h, rng, bias_m=bias), h_true, sigma_h
            )
            for _ in range(DRAWS)
        ]
    )
    result = ss.evaluate_gate("altimeter (biased)", statistics, 1)
    assert not result.passed, result.describe()
    # The inflation is the squared normalized bias, plus one for the white part.
    assert result.statistic == pytest.approx(1.0 + (bias / sigma_h) ** 2, rel=0.15)


def test_gate_reports_the_observed_value_against_its_bound():
    """DX-2/DX-5: a failing gate must name the number and the bound it missed."""
    result = ss.evaluate_gate("demo", np.full(DRAWS, 5.0), 3)
    text = result.describe()
    assert "FAIL" in text
    assert "statistic=5.000000" in text
    assert "bounds=[" in text
    assert "draws=1000" in text


def test_apparent_sun_body_direction_is_a_unit_vector_in_body_axes():
    """Section sec:optical:sunsensor: ``u^B = C_I2B u'^I``, shared with ch:camera."""
    u_sun_i = np.array([0.6, -0.48, 0.64])
    u_body = ss.apparent_sun_body(Q_TRUE, u_sun_i, BETA)
    assert float(np.linalg.norm(u_body)) == pytest.approx(1.0, abs=1e-14)
    # The aberration must actually change it, at the ~20 arcsec scale.
    geometric = quat_to_dcm(Q_TRUE) @ (u_sun_i / np.linalg.norm(u_sun_i))
    shift_arcsec = float(np.linalg.norm(u_body - geometric)) * ARCSEC_PER_RAD
    assert 0.0 < shift_arcsec <= 20.5


def test_sensor_configuration_errors_are_rejected():
    """Malformed configurations must abort, not silently produce a statistic."""
    with pytest.raises(ValueError, match="three positive sigmas"):
        ss.star_tracker_chi2(Q_TRUE, Q_TRUE, np.array([1.0, 0.0, 1.0]))
    with pytest.raises(ValueError, match="three positive values"):
        ss.NavFixConfig(np.array([1.0, -1.0, 1.0]), np.array([1.0, 1.0, 1.0]))
    with pytest.raises(ValueError, match="sigma must be positive"):
        ss.altimeter_chi2(1.0, 0.0, 0.0)
