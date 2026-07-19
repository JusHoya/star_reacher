"""Independent sensor error-statistics reference for Phase 6 criteria 1 and 6.

Part of the Phase 6 independent-reference set (``tests/refs/manifest.toml``):
written from Chapters ``ch:sensors-optical`` and ``ch:sensors-radio``, with no
reference to the core sensor implementations it gates. Two exit criteria are
served:

* criterion 1 -- star-tracker error statistics inside two-sided 95 % chi-square
  bounds over 1,000 draws (equations ``eq:optical:extract`` through
  ``eq:optical:stbounds``);
* criterion 6 -- external-nav-fix and altimeter error statistics inside
  two-sided 95 % chi-square bounds over 1,000 seeded draws (equations
  ``eq:radio:chi2`` and ``eq:radio:bounds``).

Two things are provided for each sensor. The MEASUREMENT model
(``star_tracker_measurement``, ``nav_fix_measurement``, ``altimeter_measurement``,
``sun_sensor_measurement``) generates draws from the chapter equations, so the
statistics machinery can be validated end to end against a signal whose error
distribution is known exactly by construction. The STATISTIC
(``star_tracker_chi2``, ``nav_fix_chi2``, ``altimeter_chi2``,
``sun_sensor_chi2``) consumes measurements and truth alone -- it never sees the
draws -- so the same function applies unchanged to the core's logged channels.

Bounds come from ``chi2.ensemble_mean_bounds``, which inverts the chi-square CDF
exactly rather than using the chapters' Wilson--Hilferty approximation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aberration import aberrate_first_order
from chi2 import gate
from quaternions import (
    quat_conj,
    quat_from_rotation_vector,
    quat_mul,
    quat_normalize,
    quat_to_dcm,
    rotation_vector_from_quat,
)

# --- Star tracker -----------------------------------------------------------


def aberration_rotation_vector(
    boresight_i: np.ndarray, beta: np.ndarray
) -> np.ndarray:
    """Field-rotation vector ``rho = b_hat^I x beta``, equation ``eq:optical:rho``.

    Over a narrow field of view the aberration field acts as a rigid rotation of
    the observed sky. The chapter's own verification is reproduced by
    :func:`aberration_rotation_consistency` below.
    """
    b = np.asarray(boresight_i, dtype=float)
    return np.cross(b, np.asarray(beta, dtype=float))


def aberration_quaternion(rho: np.ndarray) -> np.ndarray:
    """``q_ab`` of equation ``eq:optical:qab``: the exact rotation of ``-rho``.

    The chapter writes the components out as
    ``[cos(rho/2), -sin(rho/2) rho/|rho|]``, which is exactly the exponential
    map applied to the NEGATED rotation vector; expressing it that way keeps a
    single exponential-map implementation shared with the noise quaternion.
    """
    return quat_from_rotation_vector(-np.asarray(rho, dtype=float))


def aberration_rotation_consistency(
    boresight_i: np.ndarray, beta: np.ndarray
) -> float:
    """Residual of the chapter's own first-order check at the boresight.

    Equation ``eq:optical:rho`` asserts ``rho x b_hat = beta - (b_hat . beta)
    b_hat``, i.e. that the rigid field rotation reproduces the first-order
    displacement of ``eq:optical:aberration`` exactly on the boresight. This
    returns the norm of the difference, which must vanish to rounding.
    """
    b = np.asarray(boresight_i, dtype=float)
    b = b / np.linalg.norm(b)
    beta = np.asarray(beta, dtype=float)
    rho = aberration_rotation_vector(b, beta)
    return float(np.linalg.norm(np.cross(rho, b) - (beta - np.dot(b, beta) * b)))


def noise_quaternion(epsilon: np.ndarray) -> np.ndarray:
    """``dq_n`` of equation ``eq:optical:noiseq``: exact exponential map of the draw.

    Using the exact map rather than a small-angle construction is what makes the
    extracted statistic exactly chi-square rather than approximately so, because
    the logarithmic map of :func:`extract_error_vector` then recovers the drawn
    vector identically.
    """
    return quat_from_rotation_vector(epsilon)


def star_tracker_measurement(
    q_i2b_true: np.ndarray,
    epsilon: np.ndarray,
    q_ab: np.ndarray | None = None,
) -> np.ndarray:
    """Emitted quaternion, equation ``eq:optical:stmodel``.

    ``q_meas = q_ab (x) q_i2b_true (x) dq_n`` in the project's D-7 composition:
    the aberration is an inertial-side (left) factor and the sensor noise a
    body-side (right) factor. Note that in the Markley--Crassidis composition
    convention the same physical model reads with the factor order reversed;
    this form is the normative one.
    """
    if q_ab is None:
        q_ab = np.array([1.0, 0.0, 0.0, 0.0])
    dq_n = noise_quaternion(epsilon)
    # Normalized after the product, per the chapter's implementation note: the
    # factors are unit to rounding and the explicit step pins the invariant.
    return quat_normalize(quat_mul(quat_mul(q_ab, q_i2b_true), dq_n))


def extract_error_vector(
    q_meas: np.ndarray, q_i2b_true: np.ndarray, q_ab: np.ndarray | None = None
) -> np.ndarray:
    """Recover the error rotation vector, equation ``eq:optical:extract``.

    ``dq = (q_ab (x) q_true)^-1 (x) q_meas``, then the exact logarithmic map.
    By construction this returns the drawn ``epsilon`` identically for
    in-domain sigmas, which is what makes the statistic below exact.
    """
    if q_ab is None:
        q_ab = np.array([1.0, 0.0, 0.0, 0.0])
    deterministic = quat_mul(q_ab, q_i2b_true)
    dq = quat_mul(quat_conj(deterministic), q_meas)
    return rotation_vector_from_quat(dq)


def star_tracker_chi2(
    q_meas: np.ndarray,
    q_i2b_true: np.ndarray,
    sigmas: np.ndarray,
    q_ab: np.ndarray | None = None,
) -> float:
    """Per-draw statistic of equation ``eq:optical:ststat``, distributed chi2(3).

    ``q = eps^T diag(sigma**2)^-1 eps`` on the extracted error vector.
    """
    s = np.asarray(sigmas, dtype=float)
    if s.shape != (3,) or np.any(s <= 0.0):
        raise ValueError(f"three positive sigmas required, got {sigmas}")
    eps = extract_error_vector(q_meas, q_i2b_true, q_ab)
    return float(np.sum((eps / s) ** 2))


# --- Sun sensor -------------------------------------------------------------


def sun_sensor_measurement(u_body: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """Noisy unit vector, equation ``eq:optical:sunsensor``.

    ``u_meas = normalize(u^B + eta)``: the radial noise component drops on
    normalization and the two tangential components give a per-axis direction
    error of standard deviation ``sigma_ss`` to ``O(sigma_ss**2)``.
    """
    u = np.asarray(u_body, dtype=float)
    total = u / np.linalg.norm(u) + np.asarray(eta, dtype=float)
    return total / np.linalg.norm(total)


def tangent_basis(u_body: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic tangent-plane basis of section ``sec:optical:stats``.

    ``e1 = normalize(u^B x z_hat)``, falling back to ``x_hat`` when the cross
    product is shorter than 1e-8 (the near-polar degeneracy), and
    ``e2 = u^B x e1``. The fallback is what keeps the basis single-valued for a
    line of sight along ``z``.
    """
    u = np.asarray(u_body, dtype=float)
    u = u / np.linalg.norm(u)
    cross = np.cross(u, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(cross) < 1e-8:
        cross = np.cross(u, np.array([1.0, 0.0, 0.0]))
    e1 = cross / np.linalg.norm(cross)
    return e1, np.cross(u, e1)


def sun_sensor_chi2(
    u_meas: np.ndarray, u_body_true: np.ndarray, sigma_ss: float
) -> float:
    """Tangent-plane statistic of section ``sec:optical:stats``, chi2(2).

    The measured unit vector is resolved on the tangent basis at the TRUE
    direction; the two components normalized by ``sigma_ss`` give the statistic.
    Exactness is only ``O(sigma_ss**2)`` here, unlike the star tracker, because
    the normalization in ``eq:optical:sunsensor`` is a nonlinear map.
    """
    if sigma_ss <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma_ss}")
    e1, e2 = tangent_basis(u_body_true)
    m = np.asarray(u_meas, dtype=float)
    return float((np.dot(e1, m) ** 2 + np.dot(e2, m) ** 2) / sigma_ss**2)


# --- External nav fix and altimeter ----------------------------------------


@dataclass(frozen=True)
class NavFixConfig:
    """Per-axis GCRF sigmas for the external nav fix, equation ``eq:radio:white``."""

    sigma_r_m: np.ndarray
    sigma_v_mps: np.ndarray

    def __post_init__(self) -> None:
        for name, s in (("sigma_r_m", self.sigma_r_m), ("sigma_v_mps", self.sigma_v_mps)):
            arr = np.asarray(s, dtype=float)
            if arr.shape != (3,) or np.any(arr <= 0.0):
                raise ValueError(f"{name} must be three positive values, got {s}")


def nav_fix_measurement(
    r_i: np.ndarray,
    v_i: np.ndarray,
    config: NavFixConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """One white-only nav fix, equation ``eq:radio:fix``.

    The gate scenario of section ``sec:radio:stats`` disables the optional
    Gauss-Markov component, so the correlated term ``c_k`` is identically zero
    here and the per-sample statistics are independent -- which is the
    precondition for the ensemble-mean gate to be valid at all.
    """
    sr = np.asarray(config.sigma_r_m, dtype=float)
    sv = np.asarray(config.sigma_v_mps, dtype=float)
    r_meas = np.asarray(r_i, dtype=float) + rng.normal(0.0, 1.0, 3) * sr
    v_meas = np.asarray(v_i, dtype=float) + rng.normal(0.0, 1.0, 3) * sv
    return r_meas, v_meas


def nav_fix_chi2(
    measured: np.ndarray, truth: np.ndarray, sigmas: np.ndarray
) -> float:
    """Per-sample statistic of equation ``eq:radio:chi2``, chi2(3).

    ``q = sum_a (meas_a - truth_a)**2 / sigma_a**2`` over the three GCRF axes.
    Applies unchanged to the position and the velocity fix.
    """
    s = np.asarray(sigmas, dtype=float)
    if s.shape != (3,) or np.any(s <= 0.0):
        raise ValueError(f"three positive sigmas required, got {sigmas}")
    residual = np.asarray(measured, dtype=float) - np.asarray(truth, dtype=float)
    return float(np.sum((residual / s) ** 2))


def altimeter_measurement(
    h_true_m: float, sigma_h_m: float, rng: np.random.Generator, bias_m: float = 0.0
) -> float:
    """One altimeter sample, equation ``eq:radio:alt``.

    ``h_meas = h(r) + b_h + eta_h``. The gate scenario sets ``sigma_b = 0`` so
    ``bias_m`` defaults to zero: a per-run bias makes the per-sample statistics
    DEPENDENT and invalidates the ensemble-mean gate, as the chapter states.
    """
    if sigma_h_m <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma_h_m}")
    return float(h_true_m + bias_m + rng.normal(0.0, sigma_h_m))


def altimeter_chi2(h_meas_m: float, h_true_m: float, sigma_h_m: float) -> float:
    """Per-sample statistic of equation ``eq:radio:chi2``, chi2(1)."""
    if sigma_h_m <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma_h_m}")
    return float(((h_meas_m - h_true_m) / sigma_h_m) ** 2)


# --- Common gate ------------------------------------------------------------


@dataclass(frozen=True)
class GateResult:
    """Outcome of an ensemble-mean chi-square gate, reported for DX-5 messages."""

    name: str
    passed: bool
    statistic: float
    lower: float
    upper: float
    draws: int
    dof_per_draw: int

    def describe(self) -> str:
        """One-line report naming the observed value against its bound."""
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"{self.name}: {verdict} statistic={self.statistic:.6f} "
            f"bounds=[{self.lower:.6f}, {self.upper:.6f}] "
            f"draws={self.draws} dof/draw={self.dof_per_draw}"
        )


def evaluate_gate(name: str, statistics: np.ndarray, dof_per_draw: int) -> GateResult:
    """Apply the two-sided 95 % ensemble-mean gate to a statistic sequence.

    The common form of ``eq:optical:stbounds``, ``eq:radio:bounds``, and
    ``eq:ekf:ensemble``: for ``M`` i.i.d. chi2(n) draws the sum is chi2(nM), so
    the accepted interval for the MEAN is the chi2(nM) 2.5 % and 97.5 %
    quantiles divided by ``M``.
    """
    values = np.asarray(statistics, dtype=float)
    passed, statistic, (lower, upper) = gate(values, dof_per_draw)
    return GateResult(
        name=name,
        passed=passed,
        statistic=statistic,
        lower=lower,
        upper=upper,
        draws=int(values.size),
        dof_per_draw=dof_per_draw,
    )


def star_tracker_draws(
    q_i2b_true: np.ndarray,
    sigmas: np.ndarray,
    draws: int,
    rng: np.random.Generator,
    boresight_i: np.ndarray | None = None,
    beta: np.ndarray | None = None,
) -> np.ndarray:
    """Generate ``draws`` star-tracker statistics at fixed truth.

    Applies the aberration factor when a boresight and ``beta`` are supplied,
    which exercises the full ``eq:optical:stmodel`` composition; the statistic
    must be unaffected by it, since the extraction removes the same
    deterministic factor. That invariance is the sharpest available test of the
    composition order.
    """
    s = np.asarray(sigmas, dtype=float)
    if s.shape != (3,) or np.any(s <= 0.0):
        raise ValueError(f"three positive sigmas required, got {sigmas}")
    q_ab = None
    if boresight_i is not None and beta is not None:
        q_ab = aberration_quaternion(aberration_rotation_vector(boresight_i, beta))
    out = np.empty(draws)
    for i in range(draws):
        epsilon = rng.normal(0.0, 1.0, 3) * s
        q_meas = star_tracker_measurement(q_i2b_true, epsilon, q_ab)
        out[i] = star_tracker_chi2(q_meas, q_i2b_true, s, q_ab)
    return out


def sun_sensor_draws(
    u_body_true: np.ndarray, sigma_ss: float, draws: int, rng: np.random.Generator
) -> np.ndarray:
    """Generate ``draws`` sun-sensor tangent-plane statistics at fixed truth."""
    u = np.asarray(u_body_true, dtype=float)
    u = u / np.linalg.norm(u)
    out = np.empty(draws)
    for i in range(draws):
        eta = rng.normal(0.0, sigma_ss, 3)
        out[i] = sun_sensor_chi2(sun_sensor_measurement(u, eta), u, sigma_ss)
    return out


def apparent_sun_body(
    q_i2b: np.ndarray, u_sun_i: np.ndarray, beta: np.ndarray
) -> np.ndarray:
    """Apparent Sun direction in body axes, section ``sec:optical:sunsensor``.

    ``u^B = C_I2B(q_i2b) u'^I``, with ``u'^I`` the aberrated inertial direction
    of equation ``eq:optical:aberration`` -- the same shared function the camera
    hook's Sun vector uses, which is what makes exit criterion 9 a single gate.
    """
    return quat_to_dcm(q_i2b) @ aberrate_first_order(u_sun_i, beta)
