// Cannonball drag acceleration (FR-9). Derivation and validation table:
// docs/mathlib/chapters/drag.tex (ch:drag).
#include "star/models/drag.hpp"

#include <cmath>
#include <stdexcept>

namespace star {
namespace models {

Eigen::Vector3d drag_accel(double rho_kgpm3, double cd_a_over_m_m2pkg,
                           const Eigen::Vector3d& v_rel_mps) {
  if (!std::isfinite(rho_kgpm3) || rho_kgpm3 < 0.0 ||
      !std::isfinite(cd_a_over_m_m2pkg) || cd_a_over_m_m2pkg < 0.0 ||
      !v_rel_mps.allFinite()) {
    throw std::domain_error("drag_accel: invalid argument");
  }
  // eq:drag:accel -- a = -1/2 rho (Cd A/m) |v_rel| v_rel. The caller
  // supplies v_rel per eq:drag:vrel (v - omega x r, co-rotating
  // atmosphere); the product |v|*v combines magnitude and direction with
  // no division, so zero velocity yields the exact zero vector.
  const double factor =
      -0.5 * rho_kgpm3 * cd_a_over_m_m2pkg * v_rel_mps.norm();
  return factor * v_rel_mps;
}

}  // namespace models
}  // namespace star
