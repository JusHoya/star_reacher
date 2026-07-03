// Physical constants used by the star:: core. Every constant cites its source
// at the definition so a reviewer can trace the value without leaving the file
// (project convention: cite every physical constant at its definition).
#ifndef STAR_CONSTANTS_HPP
#define STAR_CONSTANTS_HPP

namespace star {
namespace constants {

// Geocentric gravitational constant GM_earth [m^3/s^2].
// Source: IERS Conventions (2010), IERS Technical Note No. 36, Table 1.1
// ("Geocentric gravitational constant", GM_E = 3.986004418e14 m^3 s^-2).
// This is the single home for the value: the Python frontend obtains it
// through star_reacher._core.gm("earth") rather than redefining it.
inline constexpr double GM_EARTH_M3_PER_S2 = 3.986004418e14;

// 2*pi rounded to the nearest IEEE-754 binary64. Mathematical constant, not a
// measured quantity; defined here because <cmath> provides no portable pi in
// C++17 and the Box-Muller transform (star/rng.hpp) needs a fixed, documented
// value for cross-platform golden-vector agreement.
inline constexpr double TWO_PI = 6.283185307179586476925286766559;

// pi rounded to the nearest IEEE-754 binary64, for the same portability
// reason as TWO_PI (used by the shadow overlap-area normalization,
// star/models/srp.hpp).
inline constexpr double PI = 3.14159265358979323846264338327950288;

// Speed of light in vacuum c [m/s]. Exact: the SI metre is defined from
// this value. Source: BIPM, The International System of Units (SI),
// 9th edition (2019).
inline constexpr double SPEED_OF_LIGHT_M_PER_S = 299792458.0;

// Astronomical unit [m]. Exact conventional value.
// Source: IAU 2012 Resolution B2 (XXVIII General Assembly, Beijing).
inline constexpr double AU_M = 149597870700.0;

// Nominal total solar irradiance at 1 au [W/m^2]. Source: IAU 2015
// Resolution B3 nominal solar irradiance (Prsa et al. 2016, The
// Astronomical Journal 152, 41).
inline constexpr double SOLAR_IRRADIANCE_1AU_W_PER_M2 = 1361.0;

// Nominal solar radius [m]. Source: IAU 2015 Resolution B3 nominal solar
// radius (Prsa et al. 2016, The Astronomical Journal 152, 41).
inline constexpr double R_SUN_M = 6.957e8;

// Solar radiation pressure at 1 au [N/m^2]: P = Phi/c (docs/mathlib
// eq:srp:pressure), the compile-time quotient of the two cited constants
// above; approximately 4.5398e-6 N/m^2.
inline constexpr double SRP_PRESSURE_1AU_N_PER_M2 =
    SOLAR_IRRADIANCE_1AU_W_PER_M2 / SPEED_OF_LIGHT_M_PER_S;

// Nominal mean angular velocity of the Earth [rad/s].
// Source: IERS Conventions (2010), IERS Technical Note No. 36, Table 1.1
// ("Nominal mean Earth's angular velocity", 7.292115e-5 rad s^-1). Used by
// the EOM layer to form the co-rotating-atmosphere air-relative velocity
// v_rel = v - omega x r (FR-8; eq:drag:vrel) and the co-rotating launch pad
// initial state (FR-14).
inline constexpr double OMEGA_EARTH_RAD_PER_S = 7.292115e-5;

// Mean angular velocity of Mars [rad/s], derived from the IAU 2015 prime-
// meridian rotation rate dW/dt = 350.891982443297 deg/day (Archinal et al.,
// Celest. Mech. Dyn. Astron. 130:22, 2018, with the 2019 erratum values as
// distributed in NAIF pck00011.tpc; the same rate cpp/src/frames.cpp uses
// for the Mars body-fixed frame): omega = dW/dt * (pi/180) / 86400.
// Recorded as the binary64 result of that expression so every consumer
// shares one value instead of re-deriving it (FR-8 co-rotating Mars
// atmosphere; eq:drag:vrel).
inline constexpr double OMEGA_MARS_RAD_PER_S =
    350.891982443297 * (TWO_PI / 2.0 / 180.0) / 86400.0;

// WGS84 reference ellipsoid: semi-major axis [m] and inverse flattening
// (defining constants). Source: NIMA TR8350.2, "Department of Defense World
// Geodetic System 1984", 3rd ed. (2000), Table 3.1. Used for the geodetic
// altitude argument of the Harris-Priester atmosphere (ch:harrispriester;
// the ellipsoid Orekit's cross-tool baseline configuration uses).
inline constexpr double WGS84_A_M = 6378137.0;
inline constexpr double WGS84_INV_F = 298.257223563;

}  // namespace constants
}  // namespace star

#endif  // STAR_CONSTANTS_HPP
