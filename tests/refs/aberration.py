"""Independent velocity-aberration reference for Phase 6 exit criterion 9.

Part of the Phase 6 independent-reference set (``tests/refs/manifest.toml``):
written from Chapter ``ch:sensors-optical`` and from the Lorentz transformation
of a photon four-momentum, with no reference to the core implementation it
gates. Exit criterion 9 requires the core's optical truth directions to match an
independent computation to better than 1 milliarcsecond at Earth orbital speed.

Two formulas are provided:

``aberrate_first_order``
    Equation ``eq:optical:aberration``, the normative project formula:
    ``u' = normalize(u + beta - (u . beta) u)`` -- the geometric direction plus
    the component of ``beta`` perpendicular to it, renormalized. ``u`` points
    FROM the observer TO the source and the apparent direction is displaced
    TOWARD the observer's velocity.

``aberrate_exact``
    The exact special-relativistic result, derived here rather than copied, so
    the chapter's first-order truncation can be measured rather than assumed.
    Let ``n = -u`` be the photon propagation direction in the barycentric frame
    and ``p^mu = E (1, n)`` its four-momentum. Boosting to the observer frame
    moving with velocity ``beta``:

        E'  = gamma E (1 - n . beta)
        p'  = E [ n + (gamma - 1) (n . beta_hat) beta_hat - gamma beta ]

    so ``n' = p'/E'``. Substituting ``u = -n`` and ``u' = -n'`` gives

        u' = [ u + (gamma - 1) (u . beta_hat) beta_hat + gamma beta ]
             / [ gamma (1 + u . beta) ],

    equivalently (dividing through by gamma, using
    ``(gamma - 1)/gamma = gamma beta^2 / (gamma + 1)``)

        u' = [ u/gamma + beta + (gamma/(gamma+1)) (u . beta) beta ]
             / (1 + u . beta),

    which is the form given for stellar aberration by Kaplan (2005), USNO
    Circular 179, section 7.2.3. Expanding to first order in ``beta``
    reproduces ``eq:optical:aberration`` exactly, which is the consistency
    check the test suite asserts.

The chapter states the difference between the two forms as
``(beta**2 / 4) sin(2 theta) + O(beta**3)`` in apparent angle, at most
0.52 mas at ``beta = 1e-4``. ``first_order_error_angle`` measures it directly
and the test suite checks both the magnitude and the claimed functional form.
"""

from __future__ import annotations

import numpy as np

# Speed of light in vacuum: exact by the SI definition of the metre (BIPM SI
# Brochure). Not a measured quantity, so it carries no uncertainty.
SPEED_OF_LIGHT_MPS = 299792458.0

# Earth's mean heliocentric orbital speed, the scale at which Chapter
# ch:sensors-optical quotes 20.49 arcsec and at which exit criterion 9 gates.
EARTH_MEAN_ORBITAL_SPEED_MPS = 29780.0

ARCSEC_PER_RAD = 648000.0 / np.pi
MAS_PER_RAD = 1000.0 * ARCSEC_PER_RAD


def _as_unit(vector: np.ndarray, name: str) -> np.ndarray:
    """Return ``vector`` normalized, rejecting a null direction outright."""
    v = np.asarray(vector, dtype=float)
    if v.shape[-1] != 3:
        raise ValueError(f"{name} must have trailing dimension 3, got {v.shape}")
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    # A zero direction has no aberrated image; abort rather than emit NaN, per
    # the project's abort-on-missing-critical-input rule.
    if np.any(norm == 0.0):
        raise ValueError(f"{name} contains a zero-length direction")
    return v / norm


def beta_vector(
    v_sc_i: np.ndarray,
    v_body_ssb_i: np.ndarray,
    c_mps: float = SPEED_OF_LIGHT_MPS,
) -> np.ndarray:
    """Observer velocity in units of ``c``, equation ``eq:optical:beta``.

    ``beta = (v_sc + v_cb/SSB) / c`` in GCRF axes: the vehicle velocity relative
    to the central body plus the central body's velocity relative to the
    solar-system barycentre. The barycentric composition is required because
    aberration is referred to the frame in which the catalogue directions are
    defined (ICRS, barycentric); the central-body-relative velocity alone omits
    the dominant annual term.
    """
    v = np.asarray(v_sc_i, dtype=float) + np.asarray(v_body_ssb_i, dtype=float)
    return v / c_mps


def aberrate_first_order(u_i: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Apparent direction by equation ``eq:optical:aberration`` (normative)."""
    u = _as_unit(u_i, "u_i")
    b = np.asarray(beta, dtype=float)
    if b.shape[-1] != 3:
        raise ValueError(f"beta must have trailing dimension 3, got {b.shape}")
    u_dot_b = np.sum(u * b, axis=-1, keepdims=True)
    # u + beta_perp: only the component of beta transverse to the line of sight
    # tilts the apparent direction; the parallel part is removed here and the
    # renormalization restores the unit norm.
    shifted = u + b - u_dot_b * u
    return shifted / np.linalg.norm(shifted, axis=-1, keepdims=True)


def aberrate_exact(u_i: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Apparent direction by the exact relativistic aberration derived above."""
    u = _as_unit(u_i, "u_i")
    b = np.asarray(beta, dtype=float)
    if b.shape[-1] != 3:
        raise ValueError(f"beta must have trailing dimension 3, got {b.shape}")
    beta_sq = np.sum(b * b, axis=-1, keepdims=True)
    if np.any(beta_sq >= 1.0):
        raise ValueError("observer speed must be below c")
    gamma = 1.0 / np.sqrt(1.0 - beta_sq)
    u_dot_b = np.sum(u * b, axis=-1, keepdims=True)
    numerator = u / gamma + b + (gamma / (gamma + 1.0)) * u_dot_b * b
    denominator = 1.0 + u_dot_b
    result = numerator / denominator
    # The construction is unit-norm analytically; normalizing pins the
    # invariant against the accumulated rounding of the rational expression.
    return result / np.linalg.norm(result, axis=-1, keepdims=True)


def separation_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle between two vectors, in radians.

    The ``atan2(|a x b|, a . b)`` form of equation ``eq:optical:gating`` is used
    rather than ``arccos`` because it stays accurate for nearly parallel
    vectors -- exactly the regime an aberration difference lives in, where
    ``arccos`` loses half its significant digits.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    cross = np.linalg.norm(np.cross(x, y), axis=-1)
    dot = np.sum(x * y, axis=-1)
    return np.arctan2(cross, dot)


def deflection_angle(u_i: np.ndarray, beta: np.ndarray, exact: bool = False) -> np.ndarray:
    """Angle between the geometric and apparent directions, in radians.

    Equation ``eq:optical:abmag`` gives the first-order magnitude
    ``beta sin(theta)``, maximal at ``theta = pi/2``.
    """
    u = _as_unit(u_i, "u_i")
    aberrated = aberrate_exact(u, beta) if exact else aberrate_first_order(u, beta)
    return separation_angle(u, aberrated)


def first_order_error_angle(u_i: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Angular difference between the first-order and exact apparent directions.

    This is the quantity bounded at 0.52 mas in Chapter ``ch:sensors-optical``,
    assumption 2, and the reason exit criterion 9's 1 mas gate is satisfiable by
    the first-order formula at Earth orbital speed.
    """
    return separation_angle(aberrate_first_order(u_i, beta), aberrate_exact(u_i, beta))


def direction_at_angle(theta_rad: float | np.ndarray, beta_hat: np.ndarray) -> np.ndarray:
    """Unit direction making angle ``theta`` with ``beta_hat``, in its plane.

    A deterministic sweep generator for the exit-criterion-9 grid: the plane is
    spanned by ``beta_hat`` and an arbitrary but reproducible perpendicular, so
    a theta grid sweeps the whole aberration response.
    """
    b = _as_unit(beta_hat, "beta_hat")
    # Pick the coordinate axis least aligned with beta_hat, so the Gram-Schmidt
    # step below is never ill-conditioned.
    seed = np.zeros(3)
    seed[int(np.argmin(np.abs(b)))] = 1.0
    perp = seed - np.dot(seed, b) * b
    perp = perp / np.linalg.norm(perp)
    theta = np.atleast_1d(np.asarray(theta_rad, dtype=float))
    return np.cos(theta)[:, None] * b + np.sin(theta)[:, None] * perp
