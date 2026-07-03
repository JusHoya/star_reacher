// Cannonball atmospheric-drag acceleration (FR-9: orbital-regime drag).
//
// The model is a pure function of the local density, the ballistic
// parameter Cd*A/m, and the air-relative velocity v_rel = v - omega x r
// (FR-8 co-rotating-atmosphere rule, eq:drag:vrel), which the EOM layer
// computes because it owns frames and planet rotation rates
// (star/constants.hpp: OMEGA_EARTH_RAD_PER_S, OMEGA_MARS_RAD_PER_S).
// The FR-9 domain split: the Phase 4 Mach-table aerodynamic database
// covers continuum ascent flight only; this cannonball model covers the
// orbital free-molecular regime (documented default Cd = 2.2, applied at
// the vehicle-configuration level).
//
// Math-library traceability (FR-29): the derivation lives in the drag
// chapter of docs/mathlib (ch:drag); the implementation echoes its
// equation labels in drag.cpp.
#ifndef STAR_MODELS_DRAG_HPP
#define STAR_MODELS_DRAG_HPP

#include <Eigen/Dense>

namespace star {
namespace models {

// a_drag = -1/2 * rho * (Cd*A/m) * |v_rel| * v_rel  [m/s^2]
// (eq:drag:accel), resolved in whatever frame v_rel is resolved in.
// rho_kgpm3 and cd_a_over_m_m2pkg must be non-negative and finite;
// otherwise throws std::domain_error. A zero v_rel returns the exact
// zero vector (the formulation has no normalization singularity).
Eigen::Vector3d drag_accel(double rho_kgpm3, double cd_a_over_m_m2pkg,
                           const Eigen::Vector3d& v_rel_mps);

}  // namespace models
}  // namespace star

#endif  // STAR_MODELS_DRAG_HPP
