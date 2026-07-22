// Harris-Priester upper-atmosphere density model (FR-8: Earth orbit-decay
// drag; deterministic, no space-weather inputs). Montenbruck & Gill,
// "Satellite Orbits" (2000), Sect. 3.5.2 formulation and mean-solar-
// activity table; formulation-compatible with Orekit's HarrisPriester
// class, the frozen D-15 cross-tool baseline for the Phase 3 drag case.
//
// The density function is pure: it takes the geodetic altitude and the
// cosine of the bulge angle, so the EOM layer keeps ownership of frames,
// ephemerides, and the ellipsoid choice. Helpers for the bulge-apex
// direction and the WGS84 geodetic altitude are provided alongside.
//
// Math-library traceability (FR-29): the derivation lives in the
// Harris-Priester chapter of docs/mathlib (ch:harrispriester); the
// implementation echoes its equation labels in atmosphere_hp.cpp.
#ifndef STAR_MODELS_ATMOSPHERE_HP_HPP
#define STAR_MODELS_ATMOSPHERE_HP_HPP

#include <cstddef>

#include <Eigen/Dense>

namespace star {
namespace models {

// Diurnal-bulge cosine exponent: low-inclination orbits ~2, polar orbits
// ~6 (Montenbruck & Gill Sect. 3.5.2). The default of 4 matches the Orekit
// reference implementation so default configurations of the two tools
// coincide; mission configuration overrides it per orbit.
inline constexpr double HP_COS_EXPONENT_MIN = 2.0;
inline constexpr double HP_COS_EXPONENT_MAX = 6.0;
inline constexpr double HP_COS_EXPONENT_DEFAULT = 4.0;

// Density [kg/m^3] at geodetic altitude alt_m above the reference
// ellipsoid, with cos_psi the cosine of the angle between the satellite
// position direction and the diurnal-bulge apex direction, and n the
// bulge exponent in [2, 6]. Throws std::domain_error below the 100 km
// table floor and for invalid cos_psi/n or non-finite inputs; returns
// exactly zero above the 1000 km table ceiling (Orekit-compatible).
double hp_density(double alt_m, double cos_psi, double n);

// Diurnal-bulge apex direction: the unit Sun direction rotated +30 deg
// about the Earth polar (+z) axis (right ascension advanced by the lag,
// declination preserved; eq:hp:apex). Both vectors are resolved in the
// caller's Earth-equatorial frame. sun_dir_unit must be unit norm.
Eigen::Vector3d hp_bulge_apex(const Eigen::Vector3d& sun_dir_unit);

// Geodetic altitude [m] of an Earth-fixed Cartesian position above the
// (a, 1/f) ellipsoid, by Bowring's closed form with one fixed refinement
// pass (eq:hp:geodetic; sub-millimetre for this model's altitude domain).
// The caller chooses the ellipsoid; the Orekit cross-tool configuration
// uses WGS84 (star/constants.hpp: WGS84_A_M, WGS84_INV_F).
double geodetic_altitude(const Eigen::Vector3d& r_ecef_m, double a_m,
                         double inv_f);

// The same Bowring conversion, also returning the geodetic latitude and east
// longitude [rad]. geodetic_altitude() is a thin wrapper over this, so the
// two can never drift apart: a consumer that needs the ellipsoidal normal
// (the reference EKF's altimeter measurement Jacobian, eq:ekf:altH) gets the
// angles from the identical arithmetic that produced the height.
void geodetic_lat_lon_alt(const Eigen::Vector3d& r_ecef_m, double a_m,
                          double inv_f, double& lat_rad, double& lon_rad,
                          double& alt_m);

// Read-only access to the compiled-in 50-row coefficient table, exposed
// so the doctest suite can hold the compiled table and the committed
// golden transcription (tests/golden/atmosphere/
// harris_priester_table.toml) bit-identical (ATM-HP-NODES).
struct HpNode {
  double alt_m;
  double rho_min_kgpm3;
  double rho_max_kgpm3;
};
const HpNode* hp_table(std::size_t* count);

}  // namespace models
}  // namespace star

#endif  // STAR_MODELS_ATMOSPHERE_HP_HPP
