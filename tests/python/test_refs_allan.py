"""Validate the independent Allan reference of ``tests/refs/allan.py``.

Phase 6 exit criterion 1 requires that Allan-deviation analysis of a 1e4 s static
IMU record recover the configured angular-random-walk and bias-instability
coefficients to within +/- 10 %. This suite establishes that the ESTIMATOR is
correct before it is ever pointed at the core's IMU, using signals whose
coefficients are known exactly by construction:

* pure white rate noise, whose Allan deviation is exactly ``N / sqrt(tau)``;
* pure quantization, whose Allan deviation is exactly ``q / (2 tau)``;
* a first-order Gauss-Markov bias, whose Allan variance closed form
  (eq:imu:gmadev) is independently re-derived here by direct numerical
  integration of the process autocorrelation -- the check that the chapter's
  algebra is right, not merely self-consistent;
* the full three-term chain, from which the recovery procedure of
  section sec:imu:recovery must return the coefficients it was built from.

The suite also measures the recovery's statistical dispersion, which is the
evidence for (and, in one regime, against) the achievability of the +/- 10 %
criterion.

These tests are pure NumPy and require no compiled core.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(REPO_ROOT / "tests" / "refs"))
import allan  # noqa: E402

# Exit criterion 1's record length and the chapter's worked sample interval.
RECORD_S = 1.0e4
DT_S = 0.01

# A reference gyro preset in data-sheet units: 0.1 deg/sqrt(h) ARW and a
# tactical-grade bias instability, with the chapter's reference correlation time
# of 20 s. The bias instability is chosen above the identifiability threshold
# derived in ``allan.min_bias_instability_for_ratio``; see
# ``test_low_ratio_preset_cannot_meet_the_ten_percent_criterion`` for what
# happens below it.
ARW_DEG_PER_SQRT_HOUR = 0.1
BIAS_INSTABILITY_DEG_PER_HOUR = 5.0
TAU_C_S = 20.0
QUANTUM_RAD = 1.0e-7


def _reference_config() -> allan.ImuAxisConfig:
    return allan.ImuAxisConfig(
        n_coeff=ARW_DEG_PER_SQRT_HOUR * allan.DEG_PER_SQRT_HOUR_TO_RAD_PER_SQRT_S,
        bias_instability=BIAS_INSTABILITY_DEG_PER_HOUR * allan.DEG_PER_HOUR_TO_RAD_PER_S,
        tau_c_s=TAU_C_S,
        quantum=QUANTUM_RAD,
    )


# --- Unit conversions and chapter constants --------------------------------


def test_data_sheet_conversions_are_the_chapter_values():
    """The conversions after eq:imu:arw are exact by definition."""
    assert allan.DEG_PER_SQRT_HOUR_TO_RAD_PER_SQRT_S == pytest.approx(2.9089e-4, rel=1e-4)
    assert allan.DEG_PER_HOUR_TO_RAD_PER_S == pytest.approx(4.8481e-6, rel=1e-4)


def test_gauss_markov_peak_reproduces_the_chapter_constants():
    """Equation eq:imu:gmpeak quotes ``tau* = 1.8926 tau_c``, ``0.6174 sigma``.

    Both are recomputed by golden-section search on the closed form rather than
    transcribed, so a wrong constant in the chapter would show up here.
    """
    x_star, peak = allan.gauss_markov_peak()
    assert x_star == pytest.approx(1.8926, abs=5e-5)
    assert peak == pytest.approx(0.617364, abs=5e-7)


def test_bias_instability_factor_is_sqrt_two_ln_two_over_pi():
    """Equation eq:imu:bi: ``sigma_min = sqrt(2 ln2 / pi) B = 0.664 B``."""
    assert allan.BIAS_INSTABILITY_FACTOR == pytest.approx(0.664282, abs=1e-6)


def test_preset_mapping_makes_the_peak_equal_the_configured_coefficient():
    """Equation eq:imu:presetmap exists so the conventional read-out returns B.

    With ``sigma_GM = 1.0760 B``, the MAXIMUM of the Gauss-Markov Allan
    deviation must equal ``0.664 B`` exactly -- that is the definition the
    mapping is constructed to satisfy, and it is checked numerically here
    rather than assumed.
    """
    bias_instability = 5.0 * allan.DEG_PER_HOUR_TO_RAD_PER_S
    tau_c = 20.0
    sigma_gm = allan.sigma_gm_from_bias_instability(bias_instability)
    assert sigma_gm / bias_instability == pytest.approx(1.0760, abs=5e-5)

    tau = np.geomspace(0.01 * tau_c, 100.0 * tau_c, 20001)
    peak = float(np.max(allan.gauss_markov_adev(tau, sigma_gm, tau_c)))
    assert peak == pytest.approx(allan.BIAS_INSTABILITY_FACTOR * bias_instability, rel=1e-6)
    # And the inverse map returns the configured coefficient.
    assert allan.bias_instability_from_sigma_gm(sigma_gm) == pytest.approx(
        bias_instability, rel=1e-12
    )


def _allan_variance_by_quadrature(tau: float, sigma_gm: float, tau_c: float) -> float:
    """Independently re-derive eq:imu:gmadev from the process autocorrelation.

    ``sigma**2(tau) = (V - C) / tau**2`` with
    ``V = int_0^tau int_0^tau R(t-s)``, ``C = int_0^tau int_tau^{2tau} R(t-s)``
    and ``R(s) = sigma**2 exp(-|s|/tau_c)``. Evaluated by two-dimensional
    trapezoidal quadrature on a fine grid -- no closed form used anywhere -- so
    agreement is evidence that the chapter's algebra is correct.
    """
    n = 601
    s = np.linspace(0.0, tau, n)
    t_first = np.linspace(0.0, tau, n)
    t_second = np.linspace(tau, 2.0 * tau, n)
    kernel_v = sigma_gm**2 * np.exp(-np.abs(t_first[None, :] - s[:, None]) / tau_c)
    kernel_c = sigma_gm**2 * np.exp(-np.abs(t_second[None, :] - s[:, None]) / tau_c)
    v = np.trapezoid(np.trapezoid(kernel_v, t_first, axis=1), s)
    c = np.trapezoid(np.trapezoid(kernel_c, t_second, axis=1), s)
    return float((v - c) / tau**2)


@pytest.mark.parametrize("tau_over_tau_c", [0.25, 1.0, 1.8926, 4.0, 10.0])
def test_gauss_markov_closed_form_matches_direct_quadrature(tau_over_tau_c):
    """Equation eq:imu:gmadev re-derived from ``R(s)`` by numerical integration."""
    sigma_gm, tau_c = 3.0e-6, 20.0
    tau = tau_over_tau_c * tau_c
    closed = float(allan.gauss_markov_allan_variance(np.array([tau]), sigma_gm, tau_c)[0])
    quadrature = _allan_variance_by_quadrature(tau, sigma_gm, tau_c)
    assert closed == pytest.approx(quadrature, rel=1e-5)


@pytest.mark.parametrize(
    ("tau_over_tau_c", "asymptote"),
    [(1e-4, "short"), (1e-3, "short"), (1e3, "long"), (1e4, "long")],
)
def test_gauss_markov_asymptotes(tau_over_tau_c, asymptote):
    """The +1/2 and -1/2 slopes stated below eq:imu:gmadev."""
    value = float(allan.gauss_markov_adev(np.array([tau_over_tau_c]), 1.0, 1.0)[0])
    if asymptote == "short":
        expected = np.sqrt(2.0 * tau_over_tau_c / 3.0)
    else:
        expected = np.sqrt(2.0 / tau_over_tau_c)
    assert value == pytest.approx(expected, rel=1e-3)


# --- The estimator on signals with known coefficients ----------------------


def test_estimator_recovers_pure_white_rate_noise():
    """``sigma(tau) = N / sqrt(tau)``, equation eq:imu:whiteslope.

    A record of pure integrated white noise is the one case where the estimator
    has an exact expectation at every cluster time, so it pins the estimator's
    normalization -- the factor of ``2 tau**2`` in eq:imu:oadev -- with no free
    parameters.
    """
    n_coeff = 2.9089e-5
    rng = np.random.default_rng(101)
    n_samples = int(RECORD_S / DT_S)
    increments = rng.normal(0.0, n_coeff * np.sqrt(DT_S), n_samples)
    phase = allan.phase_from_increments(increments)

    m_grid = allan.octave_m_grid(phase.size, DT_S, RECORD_S / 10.0)
    taus, adevs = allan.adev_curve(phase, DT_S, m_grid)
    expected = allan.white_noise_adev(taus, n_coeff)
    relative = adevs / expected - 1.0
    # Estimator scatter is set by the number of independent clusters, T/tau, so
    # the tolerance must scale with tau rather than being a flat percentage: the
    # standard relative dispersion is about 1/sqrt(2 T/tau), gated here at four
    # times that. At the long end of the grid only ~15 clusters remain and a
    # 20 % departure is ordinary; at one second there are 10,000 and it is not.
    tolerance = 4.0 / np.sqrt(2.0 * RECORD_S / taus)
    assert np.all(np.abs(relative) < tolerance), (
        f"octave errors {relative} exceed the scatter tolerance {tolerance}"
    )
    # At the one-second anchor the estimate is backed by ~5,000 independent
    # second differences and must be much tighter.
    one_second = allan.overlapping_allan_deviation(phase, DT_S, int(round(1.0 / DT_S)))
    assert one_second == pytest.approx(n_coeff, rel=0.03)


def test_quantization_residual_variance_is_a_twelfth_of_the_quantum_squared():
    """The first half of eq:imu:quantadev's derivation, checked directly.

    The chapter asserts the carried residual is uniform over a quantum when the
    signal is dithered, hence of variance ``q**2/12``. Measured over a range of
    dither levels spanning two orders of magnitude, which is what establishes
    the claim is about the quantizer and not about a particular input.
    """
    quantum = 1.0e-6
    for dither_in_quanta in (0.02, 0.2, 1.0, 3.0):
        rng = np.random.default_rng(102)
        values = rng.normal(0.0, quantum * dither_in_quanta, 200000)
        emitted = allan.quantize_with_carry(values, quantum)
        residual = np.cumsum(values) - np.cumsum(emitted)
        assert residual.var() == pytest.approx(quantum**2 / 12.0, rel=0.02), (
            f"dither {dither_in_quanta} q"
        )


def test_white_phase_noise_has_the_minus_one_allan_slope():
    """The second half of eq:imu:quantadev: white PHASE noise gives ``sqrt(3) s/tau``.

    IEEE Std 952's white-phase-noise signature is ``sigma**2(tau) = 3 s**2 /
    tau**2`` for i.i.d. phase samples of standard deviation ``s``. Feeding the
    estimator such a record isolates the -1 slope from every other term, and
    with ``s = q/sqrt(12)`` it reproduces ``q/(2 tau)`` exactly -- which is the
    composition the chapter states.

    Note the composition requires the residual to be white SAMPLE TO SAMPLE. The
    carry of eq:imu:quant correlates it over roughly ``q/sigma_input`` samples,
    so a signal dithered far below one quantum departs from ``q/(2 tau)`` at
    short cluster times; the chapter's parenthetical "when dithered by the noise
    terms" is doing real work.
    """
    quantum = 1.0e-6
    sigma_phase = quantum / np.sqrt(12.0)
    rng = np.random.default_rng(9)
    phase = rng.normal(0.0, sigma_phase, 400001)
    for m in (1, 4, 16, 64, 256, 1024):
        tau = m * DT_S
        estimated = allan.overlapping_allan_deviation(phase, DT_S, m)
        expected = float(allan.quantization_adev(np.array([tau]), quantum)[0])
        assert estimated == pytest.approx(expected, rel=0.02), f"m={m}"


def test_quantizer_carry_is_lossless_in_accumulation():
    """Equation eq:imu:quant: the carry bounds the accumulated error at ``q/2``.

    ``sum y = sum u - rho_K`` with ``rho in (-q/2, q/2]``, which is how a real
    rate-integrating output register behaves and is what keeps quantization from
    contributing a random walk.
    """
    rng = np.random.default_rng(103)
    quantum = 1.0e-6
    values = rng.normal(0.0, quantum * 3.0, 20000)
    emitted = allan.quantize_with_carry(values, quantum)
    residual = np.cumsum(values) - np.cumsum(emitted)
    assert np.max(np.abs(residual)) <= 0.5 * quantum * (1.0 + 1e-12)
    # Every emitted value is an exact integer multiple of the quantum.
    multiples = emitted / quantum
    assert np.max(np.abs(multiples - np.round(multiples))) < 1e-9


def test_quantizer_ties_round_toward_plus_infinity():
    """The chapter's round-half-up rule: ``y = q floor(s/q + 1/2)``."""
    quantum = 2.0
    # A single value at exactly half a quantum, with no carry history.
    assert allan.quantize_with_carry(np.array([1.0]), quantum)[0] == 2.0
    assert allan.quantize_with_carry(np.array([-1.0]), quantum)[0] == 0.0


def test_constant_bias_does_not_change_the_allan_deviation():
    """A turn-on bias is a linear phase ramp, annihilated by the second difference.

    This is why ``synthesize_static_record`` may omit the turn-on bias without
    affecting the recovered coefficients -- asserted rather than assumed.
    """
    rng = np.random.default_rng(104)
    n_samples = 100000
    increments = rng.normal(0.0, 1e-6, n_samples)
    biased = increments + 4.8481e-6 * DT_S  # a 1 deg/h turn-on bias
    for m in (1, 8, 64, 512):
        plain = allan.overlapping_allan_variance(allan.phase_from_increments(increments), DT_S, m)
        with_bias = allan.overlapping_allan_variance(
            allan.phase_from_increments(biased), DT_S, m
        )
        assert with_bias == pytest.approx(plain, rel=1e-9), f"m={m}"


# --- The exit-criterion-1 recovery -----------------------------------------


def test_recovery_meets_the_ten_percent_criterion_on_a_synthetic_record():
    """Exit criterion 1, run end to end on a signal with known coefficients."""
    config = _reference_config()
    # The preset must be identifiable before the gate is meaningful; the ratio
    # is a property of the configuration, not of the seed.
    assert allan.gm_to_white_ratio_at_peak(config) > 4.0

    rng = np.random.default_rng(20260719)
    increments = allan.synthesize_static_record(config, DT_S, RECORD_S, rng)
    result = allan.recover_coefficients(increments, DT_S, config)

    assert abs(result.n_error) <= 0.10, (
        f"ARW recovered as {result.n_hat:.6e} vs configured {config.n_coeff:.6e} "
        f"({100 * result.n_error:+.2f} %)"
    )
    assert abs(result.b_error) <= 0.10, (
        f"bias instability recovered as {result.b_hat:.6e} vs configured "
        f"{config.bias_instability:.6e} ({100 * result.b_error:+.2f} %)"
    )
    assert result.passed


def test_recovery_is_reproducible_for_a_fixed_seed():
    """Determinism: the same seed must give bit-identical recovered coefficients."""
    config = _reference_config()
    first = allan.recover_coefficients(
        allan.synthesize_static_record(config, DT_S, RECORD_S, np.random.default_rng(7)),
        DT_S,
        config,
    )
    second = allan.recover_coefficients(
        allan.synthesize_static_record(config, DT_S, RECORD_S, np.random.default_rng(7)),
        DT_S,
        config,
    )
    assert first.n_hat == second.n_hat
    assert first.b_hat == second.b_hat


def test_recovery_is_reliable_across_seeds_for_an_identifiable_preset():
    """Dispersion of the recovery, measured over independent seeds.

    Exit criterion 1 is a statistical statement, so a single passing seed is not
    evidence. Eight independent records must all pass for the reference preset.
    """
    config = _reference_config()
    errors_n, errors_b = [], []
    for seed in range(8):
        increments = allan.synthesize_static_record(
            config, DT_S, RECORD_S, np.random.default_rng(3000 + seed)
        )
        result = allan.recover_coefficients(increments, DT_S, config)
        errors_n.append(result.n_error)
        errors_b.append(result.b_error)
    assert max(abs(e) for e in errors_n) <= 0.10, f"ARW errors {errors_n}"
    assert max(abs(e) for e in errors_b) <= 0.10, f"bias-instability errors {errors_b}"


def test_low_ratio_preset_cannot_meet_the_ten_percent_criterion():
    """A gate-design constraint Chapter ch:sensors-imu does not state.

    Equation eq:imu:recoverybi recovers ``B`` by subtracting the white term from
    the Allan variance at ``tau*``. When the Gauss-Markov contribution is a
    MINORITY of that variance, the subtraction amplifies the estimator's own
    scatter by ``1 + W/G`` and the +/- 10 % gate becomes unreliable. The
    chapter's dispersion argument (about 6 % scatter at ``tau*`` for a 1e4 s
    record) omits this amplification.

    Documented here as a deterministic regression: with 1 deg/h bias
    instability against 0.1 deg/sqrt(h) ARW at ``tau_c = 20 s`` the ratio is
    below 0.5 and several of eight seeds miss the criterion. The gate scenario
    must therefore be configured above the identifiability threshold, which
    ``allan.min_bias_instability_for_ratio`` computes.
    """
    marginal = allan.ImuAxisConfig(
        n_coeff=0.1 * allan.DEG_PER_SQRT_HOUR_TO_RAD_PER_SQRT_S,
        bias_instability=1.0 * allan.DEG_PER_HOUR_TO_RAD_PER_S,
        tau_c_s=20.0,
        quantum=QUANTUM_RAD,
    )
    assert allan.gm_to_white_ratio_at_peak(marginal) < 0.5

    failures = 0
    for seed in range(8):
        increments = allan.synthesize_static_record(
            marginal, DT_S, RECORD_S, np.random.default_rng(4000 + seed)
        )
        result = allan.recover_coefficients(increments, DT_S, marginal)
        failures += int(abs(result.b_error) > 0.10)
    assert failures >= 2, (
        "the marginal preset unexpectedly passed; the identifiability "
        f"amplification may have changed (failures={failures}/8)"
    )

    # The design rule that fixes it: the threshold B for a ratio of 4.
    threshold = allan.min_bias_instability_for_ratio(marginal.n_coeff, marginal.tau_c_s, 4.0)
    assert threshold / allan.DEG_PER_HOUR_TO_RAD_PER_S == pytest.approx(2.94, abs=0.02)


def test_model_curve_overlays_the_estimated_deviation():
    """The chapter's Allan-recovery validation item, checked directly.

    ``sqrt(N**2/tau + var_GM(tau) + var_Q(tau))`` must overlay the estimated
    ADEV across the octave grid within estimator scatter -- the statement that
    the three-term model is the whole model, with nothing unaccounted for.
    """
    config = _reference_config()
    increments = allan.synthesize_static_record(
        config, DT_S, RECORD_S, np.random.default_rng(55)
    )
    phase = allan.phase_from_increments(increments)
    m_grid = allan.octave_m_grid(phase.size, DT_S, RECORD_S / 10.0)
    taus, adevs = allan.adev_curve(phase, DT_S, m_grid)
    model = allan.model_adev(
        taus, config.n_coeff, config.sigma_gm, config.tau_c_s, config.quantum
    )
    relative = adevs / model - 1.0
    assert np.max(np.abs(relative)) < 0.15, (
        "model overlay departs from the estimate by "
        f"{100 * np.max(np.abs(relative)):.1f} % at tau="
        f"{taus[int(np.argmax(np.abs(relative)))]:.2f} s"
    )


def test_recovery_rejects_a_sample_interval_that_cannot_reach_one_second():
    """Section sec:imu:recovery requires an integer cluster count at 1 s."""
    config = _reference_config()
    increments = np.zeros(1000)
    with pytest.raises(ValueError, match="does not divide 1 s"):
        allan.recover_coefficients(increments, 0.03, config)
