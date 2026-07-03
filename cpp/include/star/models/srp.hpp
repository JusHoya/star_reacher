// Solar radiation pressure with dual-cone conical shadow (FR-7): the
// attitude-independent cannonball acceleration scaled by an illumination
// fraction nu in [0, 1] from the apparent-disk umbra/penumbra overlap
// model, cast by any spherical occulting body the caller configures. The
// shadow functions are separately callable so the event framework (FR-12)
// can screen the signed boundary functions for eclipse entry/exit and the
// force composition can combine nu over multiple occulters.
//
// Math-library traceability (FR-29): the derivation lives in the SRP
// chapter of docs/mathlib (ch:srp); the implementation echoes its equation
// labels `eq:srp:pressure`, `eq:srp:cannonball`, `eq:srp:appgeom`,
// `eq:srp:overlap`, `eq:srp:nu`, and `eq:srp:boundaries` at the
// corresponding code.
#ifndef STAR_MODELS_SRP_HPP
#define STAR_MODELS_SRP_HPP

#include <Eigen/Dense>

namespace star {
namespace models {

// Illumination fraction nu in [0, 1] seen by a spacecraft at r_sc_m: the
// uncovered area fraction of the apparent solar disk (radius radius_sun_m
// at r_sun_m) behind the apparent disk of a spherical occulting body
// (radius radius_occ_m at r_occ_m). Exactly 1.0 in full sunlight, exactly
// 0.0 in total umbra, smooth through the penumbra, and handles the annular
// case (occulter apparent radius smaller than the solar one). All
// positions share one origin and frame; the origin itself is arbitrary
// because only differences enter. Domain: spacecraft outside both spheres
// (inside an occulter the apparent-radius asin clamps; see ch:srp).
double shadow_fraction(const Eigen::Vector3d& r_sc_m,
                       const Eigen::Vector3d& r_sun_m, double radius_sun_m,
                       const Eigen::Vector3d& r_occ_m, double radius_occ_m);

// Signed umbra boundary g_u = theta - (a_occ - a_sun) [rad]: negative
// exactly inside the total umbra, positive wherever any part of the Sun is
// visible (always positive in annular geometry, where no umbra exists).
// Crosses zero transversally along an orbit, so it is usable directly as
// an FR-12 event function (nu itself is constant outside the penumbra and
// cannot be sign-screened).
double shadow_umbra_boundary(const Eigen::Vector3d& r_sc_m,
                             const Eigen::Vector3d& r_sun_m,
                             double radius_sun_m,
                             const Eigen::Vector3d& r_occ_m,
                             double radius_occ_m);

// Signed penumbra boundary g_p = theta - (a_sun + a_occ) [rad]: negative
// exactly where the Sun is at least partially occulted (nu < 1), positive
// in full sunlight. Same event-function contract as the umbra boundary.
double shadow_penumbra_boundary(const Eigen::Vector3d& r_sc_m,
                                const Eigen::Vector3d& r_sun_m,
                                double radius_sun_m,
                                const Eigen::Vector3d& r_occ_m,
                                double radius_occ_m);

// Cannonball SRP acceleration [m/s^2] (eq:srp:cannonball):
//   a = nu * P_1au * (au/d)^2 * (Cr*A/m) * s_hat,
// where s_hat = (r_sc - r_sun)/|r_sc - r_sun| points FROM THE SUN TO THE
// SPACECRAFT (the photon propagation direction), so the acceleration
// pushes the spacecraft away from the Sun for Cr > 0. nu is supplied by
// the caller (shadow_fraction over the configured occulters); P_1au is the
// cited constant SRP_PRESSURE_1AU_N_PER_M2. nu == 0 yields an exactly zero
// vector. No per-call allocation.
Eigen::Vector3d srp_accel(double cr_area_over_mass_m2pkg, double nu,
                          const Eigen::Vector3d& r_sc_m,
                          const Eigen::Vector3d& r_sun_m);

}  // namespace models
}  // namespace star

#endif  // STAR_MODELS_SRP_HPP
