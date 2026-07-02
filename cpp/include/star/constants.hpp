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

}  // namespace constants
}  // namespace star

#endif  // STAR_CONSTANTS_HPP
