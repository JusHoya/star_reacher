"""Regenerate the SRP and shadow golden-vector files in this directory.

The values anchor the FR-7 cannonball SRP model with dual-cone conical
shadow (cpp/src/models/srp.cpp). Two files are produced:

- shadow_fraction.toml: illumination fractions nu for committed geometries
  covering every branch of the apparent-disk overlap model (chapter ch:srp,
  eq:srp:nu): full sunlight (nu exactly 1), total umbra (nu exactly 0),
  partial penumbra at three depths, the annular case (occulter apparent
  radius smaller than the solar apparent radius), an off-axis annular case
  (nu independent of the center separation while inside the antumbra), a
  partial lunar occultation, and a lunar-umbra case with the Moon as the
  occulting body.
- accel.toml: cannonball SRP accelerations a = nu P1au (au/d)^2 (Cr A/m) s_hat
  with s_hat the unit vector from the Sun to the spacecraft
  (eq:srp:cannonball), including the exact-zero umbra case and an
  inverse-square check at Mars heliocentric distance.

References are evaluated with mpmath at 60 significant decimal digits from
the exact binary64 inputs and rounded once to binary64. The mpmath
evaluation mirrors the model formulation exactly (same piecewise branches);
it is independent of the C++ code path and of binary64 rounding, so the
committed values check the double-precision implementation to its own
rounding floor. The radiation-pressure constant is evaluated as the exact
rational 1361/299792458 (IAU 2015 Resolution B3 nominal total solar
irradiance over the exact SI speed of light); the C++ constant is the
binary64 rounding of the same quotient, a sub-ulp difference absorbed by
the recorded tolerances.

Geometry constants used to build the committed cases (exact binary64):
solar radius 6.957e8 m (IAU 2015 B3 nominal), au = 149597870700 m (IAU 2012
B2, exact), occulter radii 6378137 m (Earth-like) and 1.7374e6 m
(Moon-like) - representative test geometry; the model takes all radii from
the caller. Penumbra-depth positions are placed from the analytic umbra and
penumbra cone-crossing angles of a 6778.137 km circular orbit with the Sun
on +x, computed here in mpmath by the ch:srp cone construction
(eq:srp:umbracone / eq:srp:penumbracone).

Running this script rewrites both .toml files byte-identically; any diff
after regeneration means the script or the goldens were edited by hand,
which the FR-22 golden-update discipline forbids.
"""

from __future__ import annotations

import pathlib

import mpmath as mp

mp.mp.dps = 60

HERE = pathlib.Path(__file__).resolve().parent

R_SUN = 6.957e8          # [m] IAU 2015 B3 nominal solar radius
AU = 149597870700.0      # [m] IAU 2012 B2, exact
R_EARTH = 6378137.0      # [m] representative Earth-like occulter radius
R_MOON = 1.7374e6        # [m] representative Moon-like occulter radius
A_ORB = 6778137.0        # [m] circular-orbit radius for the penumbra cases


# ---------------------------------------------------------------------------
# mpmath mirror of the shadow model (chapter ch:srp)
# ---------------------------------------------------------------------------


def vsub(a, b):
    return [a[i] - b[i] for i in range(3)]


def vnorm(a):
    return mp.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def apparent_geometry(r_sc, r_sun, rad_sun, r_occ, rad_occ):
    """Apparent angular radii of Sun and occulter and their separation
    (eq:srp:appgeom), from exact binary64 inputs."""
    to_sun = vsub([mp.mpf(x) for x in r_sun], [mp.mpf(x) for x in r_sc])
    to_occ = vsub([mp.mpf(x) for x in r_occ], [mp.mpf(x) for x in r_sc])
    d_sun = vnorm(to_sun)
    d_occ = vnorm(to_occ)
    a = mp.asin(min(mp.mpf(1), mp.mpf(rad_sun) / d_sun))
    b = mp.asin(min(mp.mpf(1), mp.mpf(rad_occ) / d_occ))
    dot = sum(to_sun[i] * to_occ[i] for i in range(3))
    cx = [to_sun[1] * to_occ[2] - to_sun[2] * to_occ[1],
          to_sun[2] * to_occ[0] - to_sun[0] * to_occ[2],
          to_sun[0] * to_occ[1] - to_sun[1] * to_occ[0]]
    c = mp.atan2(vnorm(cx), dot)
    return a, b, c


def shadow_fraction_mp(r_sc, r_sun, rad_sun, r_occ, rad_occ):
    """Illumination fraction (eq:srp:nu): apparent-disk overlap."""
    a, b, c = apparent_geometry(r_sc, r_sun, rad_sun, r_occ, rad_occ)
    if c >= a + b:
        return mp.mpf(1)                    # full sunlight
    if c <= b - a:
        return mp.mpf(0)                    # total umbra
    if c <= a - b:
        return 1 - (b / a) ** 2             # annular (antumbra)
    x = (c * c + a * a - b * b) / (2 * c)   # eq:srp:overlap
    y = mp.sqrt(a * a - x * x)
    area = (a * a * mp.acos(x / a) + b * b * mp.acos((c - x) / b) - c * y)
    return 1 - area / (mp.pi * a * a)


def srp_accel_mp(cr_a_over_m, nu, r_sc, r_sun):
    """Cannonball SRP acceleration (eq:srp:cannonball)."""
    p_1au = mp.mpf(1361) / mp.mpf(299792458)  # eq:srp:pressure, exact ratio
    s_vec = vsub([mp.mpf(x) for x in r_sc], [mp.mpf(x) for x in r_sun])
    d = vnorm(s_vec)
    k = mp.mpf(nu) * p_1au * (mp.mpf(AU) / d) ** 2 * mp.mpf(cr_a_over_m) / d
    return [k * s_vec[i] for i in range(3)]


# ---------------------------------------------------------------------------
# Analytic cone-crossing angles for the penumbra-depth cases
# (ch:srp eq:srp:umbracone / eq:srp:penumbracone; same construction the
# criterion-3 doctest re-derives independently in double precision)
# ---------------------------------------------------------------------------


def umbra_crossing_angle():
    sin_a = (mp.mpf(R_SUN) - R_EARTH) / AU
    t = sin_a / mp.sqrt(1 - sin_a * sin_a)
    ell = mp.mpf(R_EARTH) * AU / (mp.mpf(R_SUN) - R_EARTH)
    aa = mp.mpf(A_ORB) ** 2 * (1 + t * t)
    bb = 2 * t * t * A_ORB * ell
    cc = t * t * ell * ell - mp.mpf(A_ORB) ** 2
    u = (-bb - mp.sqrt(bb * bb - 4 * aa * cc)) / (2 * aa)
    return mp.acos(u)


def penumbra_crossing_angle():
    sin_a = (mp.mpf(R_SUN) + R_EARTH) / AU
    t = sin_a / mp.sqrt(1 - sin_a * sin_a)
    ell = mp.mpf(R_EARTH) * AU / (mp.mpf(R_SUN) + R_EARTH)
    aa = mp.mpf(A_ORB) ** 2 * (1 + t * t)
    bb = -2 * t * t * A_ORB * ell
    cc = t * t * ell * ell - mp.mpf(A_ORB) ** 2
    u = (-bb - mp.sqrt(bb * bb - 4 * aa * cc)) / (2 * aa)
    return mp.acos(u)


def orbit_pos(theta):
    """Circular-orbit position at angle theta, rounded once to binary64."""
    return (float(A_ORB * mp.cos(theta)), float(A_ORB * mp.sin(theta)), 0.0)


# ---------------------------------------------------------------------------
# TOML emission (restricted subset read by cpp/tests/golden_io.hpp)
# ---------------------------------------------------------------------------


def emit(path: pathlib.Path, header: str, cases: list[dict]) -> None:
    lines = [f"# {line}" for line in header.strip().splitlines()]
    for case in cases:
        lines.append("")
        lines.append("[[case]]")
        for key, value in case.items():
            if isinstance(value, list):
                lines.append(f"{key} = [")
                lines.extend(f'  "{item}",' for item in value)
                lines.append("]")
            else:
                lines.append(f'{key} = "{value}"')
    path.write_text("\n".join(lines) + "\n", newline="\n", encoding="utf-8")


def main() -> None:
    sun = (AU, 0.0, 0.0)
    earth = (0.0, 0.0, 0.0)
    th_u = umbra_crossing_angle()
    th_p = penumbra_crossing_angle()

    # (name, r_sc, r_sun, rad_sun, r_occ, rad_occ, kind)
    # kind "exact": nu must be exactly 0 or 1; "partial": compared at the
    # manifest tolerance.
    shadow_cases = [
        ("full_sun_subsolar", (A_ORB, 0.0, 0.0), sun, R_SUN, earth, R_EARTH,
         "exact"),
        ("full_sun_near_penumbra", orbit_pos(th_p - mp.mpf("0.01")), sun,
         R_SUN, earth, R_EARTH, "exact"),
        ("umbra_anti_sun", (-A_ORB, 0.0, 0.0), sun, R_SUN, earth, R_EARTH,
         "exact"),
        ("umbra_near_edge", orbit_pos(th_u + mp.mpf("0.01")), sun, R_SUN,
         earth, R_EARTH, "exact"),
        ("penumbra_shallow", orbit_pos(th_p + mp.mpf("0.1") * (th_u - th_p)),
         sun, R_SUN, earth, R_EARTH, "partial"),
        ("penumbra_mid", orbit_pos(th_p + mp.mpf("0.5") * (th_u - th_p)),
         sun, R_SUN, earth, R_EARTH, "partial"),
        ("penumbra_deep", orbit_pos(th_p + mp.mpf("0.9") * (th_u - th_p)),
         sun, R_SUN, earth, R_EARTH, "partial"),
        # Annular: Moon-like occulter between a LEO spacecraft and the Sun;
        # its apparent radius is below the solar one, so a ring of Sun
        # remains (antumbra) and nu = 1 - (b/a)^2.
        ("annular_moon_transit", (A_ORB, 0.0, 0.0), sun, R_SUN,
         (3.844e8, 0.0, 0.0), R_MOON, "partial"),
        # Off-axis but still inside the antumbra (c < a - b): nu keeps the
        # annular value computed from the slightly shifted distances.
        ("annular_off_axis", (A_ORB, 9.0e3, 0.0), sun, R_SUN,
         (3.844e8, 0.0, 0.0), R_MOON, "partial"),
        # Partial lunar occultation (b < a, |a-b| < c < a+b): the partial
        # branch with the small disk only partly covering the Sun.
        ("partial_moon_transit", (A_ORB, 1.0e6, 0.0), sun, R_SUN,
         (3.844e8, 0.0, 0.0), R_MOON, "partial"),
        # Moon as the occulting body seen from low lunar orbit: total umbra.
        ("moon_umbra_llo", (-1.75e6, -3.5e5, 0.0), sun, R_SUN, earth, R_MOON,
         "exact"),
    ]

    out = []
    for name, r_sc, r_sun_c, rad_sun, r_occ, rad_occ, kind in shadow_cases:
        nu = shadow_fraction_mp(r_sc, r_sun_c, rad_sun, r_occ, rad_occ)
        if kind == "exact":
            assert nu in (0, 1), (name, nu)
        else:
            assert 0 < nu < 1, (name, nu)
        out.append({
            "name": name,
            "kind": kind,
            "r_sc_m": [float(x).hex() for x in r_sc],
            "r_sun_m": [float(x).hex() for x in r_sun_c],
            "radius_sun_m": float(rad_sun).hex(),
            "r_occ_m": [float(x).hex() for x in r_occ],
            "radius_occ_m": float(rad_occ).hex(),
            "nu": float(nu).hex(),
            "nu_decimal": mp.nstr(nu, 17),
        })
        print(f"{name:24s} kind={kind:7s} nu={mp.nstr(nu, 12)}")

    emit(
        HERE / "shadow_fraction.toml",
        "Conical-shadow illumination-fraction golden vectors (FR-7).\n"
        "All positions are relative to a common origin in a common frame;\n"
        "radii are the occulting-body and solar radii the model receives\n"
        "from the caller. nu is the apparent-disk overlap illumination\n"
        "fraction (chapter ch:srp, eq:srp:nu) evaluated with mpmath at 60\n"
        "significant decimal digits from the exact binary64 inputs and\n"
        "rounded once to binary64; kind=exact cases are exactly 0 or 1 by\n"
        "the model's piecewise definition. Provenance and tolerances in\n"
        "manifest.toml. Regenerated by generate.py.",
        out,
    )

    # (name, cr_a_over_m, nu, r_sc, r_sun)
    accel_cases = [
        ("full_sun_1au", 0.02, 1.0, (A_ORB, 0.0, 0.0), sun),
        ("penumbra_half_generic", 0.013, 0.5, (4.1e6, -5.2e6, 1.3e6),
         (-9.0e10, 1.1e11, 4.0e10)),
        ("umbra_exact_zero", 0.02, 0.0, (-A_ORB, 0.0, 0.0), sun),
        ("mars_distance_inverse_square", 0.02, 1.0, (3.6e6, 1.2e6, 0.0),
         (2.28e11, 0.0, 0.0)),
        ("annular_partial_quarter", 0.008, 0.25, (-5.9e6, 3.3e6, 0.4e6),
         (1.4e11, 5.0e10, 2.0e10)),
    ]

    out = []
    for name, cram, nu, r_sc, r_sun_c in accel_cases:
        a = srp_accel_mp(cram, nu, r_sc, r_sun_c)
        out.append({
            "name": name,
            "cr_a_over_m_m2pkg": float(cram).hex(),
            "nu": float(nu).hex(),
            "r_sc_m": [float(x).hex() for x in r_sc],
            "r_sun_m": [float(x).hex() for x in r_sun_c],
            "a_ref_mps2": [float(x).hex() for x in a],
        })
        mag = vnorm([mp.mpf(float(x)) for x in a])
        print(f"{name:30s} |a|={mp.nstr(mag, 6)} m/s^2")

    emit(
        HERE / "accel.toml",
        "Cannonball SRP acceleration golden vectors (FR-7).\n"
        "a_ref_mps2 is nu * P1au * (au/d)^2 * (Cr A/m) * s_hat with s_hat\n"
        "the unit vector from the Sun to the spacecraft and P1au the exact\n"
        "rational 1361/299792458 N/m^2 (chapter ch:srp,\n"
        "eq:srp:cannonball), evaluated with mpmath at 60 significant\n"
        "decimal digits from the exact binary64 inputs and rounded once to\n"
        "binary64. The umbra case is exactly zero. Provenance and\n"
        "tolerances in manifest.toml. Regenerated by generate.py.",
        out,
    )
    print(f"SRP goldens regenerated; mpmath {mp.__version__} at "
          f"{mp.mp.dps} dps")


if __name__ == "__main__":
    main()
