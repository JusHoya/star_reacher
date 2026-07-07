"""Loader-side derived quantities: osculating orbital elements (FR-16/FR-17).

Pure NumPy plus stdlib, like ``star_reacher.srlog``: FR-16 requires osculating
elements to be derived in the loader, never logged, and FR-31 requires a log
to remain analyzable without the compiled core, so this module must import on
a NumPy-only machine.

Conventions (documented once here; every function docstring defers to this):

- All angles are radians in ``[0, 2*pi)`` except inclination, which is
  ``[0, pi]``. Angle recovery uses ``atan2`` formulations throughout, never
  ``arccos`` of near-unit arguments, so precision does not collapse near
  0 or pi.
- Singular-geometry fallbacks follow the standard alternate-element
  conventions (Vallado, Fundamentals of Astrodynamics and Applications,
  4th ed., Algorithm 9 "RV2COE"):

  * **circular** (``e`` below ``e_circular_tol``): the argument of periapsis
    is reported as exactly 0 and the true anomaly slot carries the argument
    of latitude (angle from the ascending node to the position vector).
  * **equatorial** (``sin(i)`` below ``sin_i_equatorial_tol``): the RAAN is
    reported as exactly 0 and the node direction is taken as +X, so the
    argument-of-periapsis slot carries the longitude of periapsis and, when
    also circular, the true-anomaly slot carries the true longitude.
  * In-plane angles are measured **in the direction of motion**: the
    in-plane basis is completed with ``h_hat x reference``, so a retrograde
    equatorial orbit reports angles that advance with the motion rather
    than against it.

- **Conic types.** The formulation is conic-agnostic: elliptical, hyperbolic
  (``e > 1``, ``a < 0``, positive energy — Mars-cruise heliocentric legs and
  SOI-exit states), and near-parabolic states all pass through the same
  ``atan2`` recovery. For hyperbolic states the true anomaly stays within
  the asymptote limit ``|nu| < arccos(-1/e)``; incoming (pre-periapsis)
  samples appear in ``(pi, 2*pi)`` under the ``[0, 2*pi)`` convention. At
  exactly zero specific energy the semi-major axis is ``+inf`` (parabolic
  limit); ``p = h^2/mu`` stays finite and is not reported because ``a``
  and ``e`` carry the same information at every non-parabolic sample.
- **Degenerate states** (zero position, zero angular momentum, i.e.
  rectilinear motion): the element angles and eccentricity are undefined;
  those samples yield ``nan`` in ``e``, the angles, and ``a`` (energy and
  ``|h|`` remain valid). No exception is raised so one bad sample cannot
  make a whole run un-analyzable.
"""

from __future__ import annotations

import numpy as np

_TWO_PI = 2.0 * np.pi

# Gravitational parameters GM [m^3/s^2] per central body, keyed by the SRLOG
# header's central_body vocabulary ("earth" | "moon" | "mars" | "sun",
# cpp/include/star/run.hpp). These deliberately duplicate the C++ single-home values in
# cpp/include/star/constants.hpp, because this module must work without the
# compiled core (FR-31); tests/python/test_gm_crosscheck.py compares every
# entry bit-exactly against star_reacher._core.gm() so the copies cannot
# drift silently.
GM_M3_PER_S2 = {
    # IERS Conventions (2010), TN No. 36, Table 1.1 (GM_EARTH_M3_PER_S2).
    "earth": 3.986004418e14,
    # DE440 header constant, Park et al., AJ 161:105 (2021), Table 2
    # (GM_MOON_DE440_M3_PER_S2).
    "moon": 4.902800118e12,
    # DE440 Mars-system GM, same source (GM_MARS_SYS_DE440_M3_PER_S2).
    "mars": 4.2828375816e13,
    # DE440 header constant, same source (GM_SUN_DE440_M3_PER_S2): the
    # heliocentric central body that entered with the Phase 5 Mars-cruise
    # mission.
    "sun": 1.32712440041279419e20,
}


def central_body_gm(central_body) -> float:
    """GM [m^3/s^2] for a header ``central_body`` name.

    Raises ``ValueError`` naming the supported bodies for anything else
    (including ``None``, i.e. a header that carries no central body).
    """
    if isinstance(central_body, str) and central_body in GM_M3_PER_S2:
        return GM_M3_PER_S2[central_body]
    supported = ", ".join(sorted(GM_M3_PER_S2))
    raise ValueError(
        f"no gravitational parameter for central body {central_body!r}; "
        f"supported central bodies: {supported}"
    )


def osculating_elements(
    r_m,
    v_mps,
    gm_m3ps2: float,
    *,
    e_circular_tol: float = 1e-11,
    sin_i_equatorial_tol: float = 1e-11,
) -> dict[str, np.ndarray]:
    """Osculating classical elements from inertial position/velocity samples.

    ``r_m`` and ``v_mps`` are arrays of shape ``(n, 3)`` (or a single
    ``(3,)`` sample, treated as ``n = 1``) in the same inertial frame the
    ``truth`` group logs (GCRF for Earth-centered runs); ``gm_m3ps2`` is the
    central body's gravitational parameter. Returns a dict of 1-D float64
    arrays of length ``n``:

    - ``a_m`` — semi-major axis [m]; negative for hyperbolic states,
      ``+inf`` at exactly parabolic energy.
    - ``e`` — eccentricity magnitude [-].
    - ``i_rad`` — inclination [rad], in ``[0, pi]``.
    - ``raan_rad`` — right ascension of the ascending node [rad],
      ``[0, 2*pi)``; 0 by convention when equatorial.
    - ``argp_rad`` — argument of periapsis [rad], ``[0, 2*pi)``; 0 by
      convention when circular; longitude of periapsis when equatorial.
    - ``nu_rad`` — true anomaly [rad], ``[0, 2*pi)``; argument of latitude
      when circular; true longitude when circular equatorial.
    - ``energy_m2ps2`` — specific orbital energy v^2/2 - mu/r [m^2/s^2].
    - ``hmag_m2ps`` — specific angular momentum magnitude |r x v| [m^2/s].

    Formulation: Vallado, Fundamentals of Astrodynamics and Applications,
    4th ed., Algorithm 9 (RV2COE), with the angle recovery recast in
    ``atan2`` form and the singular-geometry conventions in the module
    docstring. Degenerate samples (|r| = 0 or |h| = 0) yield ``nan``
    elements rather than raising.
    """
    r = np.atleast_2d(np.asarray(r_m, dtype=np.float64))
    v = np.atleast_2d(np.asarray(v_mps, dtype=np.float64))
    if r.shape != v.shape or r.ndim != 2 or r.shape[1] != 3:
        raise ValueError(
            f"r_m and v_mps must both have shape (n, 3); got {np.shape(r_m)} "
            f"and {np.shape(v_mps)}"
        )
    gm = float(gm_m3ps2)
    if not (gm > 0.0):
        raise ValueError(f"gm_m3ps2 must be positive; got {gm_m3ps2!r}")

    rmag = np.linalg.norm(r, axis=1)
    h_vec = np.cross(r, v)
    hmag = np.linalg.norm(h_vec, axis=1)
    # Degenerate rows (rectilinear or zero-radius states) are masked out of
    # every division below and stamped nan afterwards, so a single bad
    # sample cannot raise or poison its neighbors. The two guards are kept
    # separate: a radial state (h = 0, r > 0) still has a well-defined
    # specific energy, which must be computed with the true radius.
    zero_r = rmag == 0.0
    degenerate = zero_r | (hmag == 0.0)
    rmag_safe = np.where(zero_r, 1.0, rmag)
    hmag_safe = np.where(hmag == 0.0, 1.0, hmag)

    energy = 0.5 * np.einsum("ij,ij->i", v, v) - np.where(
        zero_r, np.nan, gm / rmag_safe
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        # a = -mu/(2*energy): the vis-viva inversion; negative for
        # hyperbolic states. The exact parabolic boundary is pinned to +inf
        # explicitly, because -gm/(2*energy) at energy == +0.0 would give
        # -inf from the sign of zero.
        a = np.where(energy == 0.0, np.inf, -gm / (2.0 * energy))

    # Eccentricity vector e = (v x h)/mu - r_hat (Vallado eq. 2-78 rearranged
    # to avoid the cancellation-prone ((v^2 - mu/r) r - (r.v) v)/mu form).
    e_vec = np.cross(v, h_vec) / gm - r / rmag_safe[:, None]
    e = np.linalg.norm(e_vec, axis=1)

    h_hat = h_vec / hmag_safe[:, None]
    cos_i = np.clip(h_hat[:, 2], -1.0, 1.0)
    # Node vector n = z_hat x h; |n|/|h| = sin(i), which is the equatorial
    # discriminator (dimensionless, unlike |n| itself).
    n_x = -h_vec[:, 1]
    n_y = h_vec[:, 0]
    n_mag = np.hypot(n_x, n_y)
    sin_i = n_mag / hmag_safe
    i = np.arctan2(sin_i, cos_i)

    equatorial = sin_i < sin_i_equatorial_tol
    circular = e < e_circular_tol

    raan = np.where(equatorial, 0.0, np.mod(np.arctan2(n_y, n_x), _TWO_PI))
    # Unit node line; +X by convention when equatorial (module docstring).
    n_hat = np.stack([np.cos(raan), np.sin(raan), np.zeros_like(raan)], axis=1)
    # m_hat = h_hat x n_hat completes an in-plane basis 90 degrees ahead of
    # the node in the direction of motion, giving atan2 angle recovery that
    # stays well-conditioned where arccos flattens (angles near 0 or pi).
    m_hat = np.cross(h_hat, n_hat)

    with np.errstate(invalid="ignore"):
        e_hat = e_vec / np.where(circular | degenerate, 1.0, e)[:, None]
    argp = np.where(
        circular,
        0.0,
        np.mod(
            np.arctan2(
                np.einsum("ij,ij->i", e_vec, m_hat),
                np.einsum("ij,ij->i", e_vec, n_hat),
            ),
            _TWO_PI,
        ),
    )

    # True-anomaly reference direction: periapsis when it exists, otherwise
    # the node line (argument of latitude), which is +X when also equatorial
    # (true longitude) — exactly the Vallado alternate-element chain.
    p_hat = np.where(circular[:, None], n_hat, e_hat)
    q_hat = np.cross(h_hat, p_hat)
    nu = np.mod(
        np.arctan2(
            np.einsum("ij,ij->i", r, q_hat),
            np.einsum("ij,ij->i", r, p_hat),
        ),
        _TWO_PI,
    )

    nan = np.float64(np.nan)
    return {
        "a_m": np.where(degenerate, nan, a),
        "e": np.where(degenerate, nan, e),
        "i_rad": np.where(degenerate, nan, i),
        "raan_rad": np.where(degenerate, nan, raan),
        "argp_rad": np.where(degenerate, nan, argp),
        "nu_rad": np.where(degenerate, nan, nu),
        "energy_m2ps2": energy,
        "hmag_m2ps": hmag,
    }
