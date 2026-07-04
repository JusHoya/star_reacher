// Gravity-gradient torque (FR-1). Derivation and validation evidence:
// docs/mathlib chapter ch:gravgrad. The equation label from that chapter
// is echoed verbatim at the corresponding code (FR-29 traceability).
#include "star/models/gravgrad.hpp"

#include "star/rotation.hpp"

namespace star {
namespace models {

Eigen::Vector3d gravgrad_torque(double mu_m3ps2, const Eigen::Vector3d& r_i_m,
                                const Eigen::Quaterniond& q_i2b,
                                const Eigen::Matrix3d& inertia_b_kgm2) {
  // Normalize in the inertial frame, then map to body axes through the
  // single project code path for frame transformation (ch:rotations), so
  // rhat_b inherits quat_transform's convention guarantees.
  const double rn = r_i_m.norm();
  const Eigen::Vector3d rhat_b =
      rotation::quat_transform(q_i2b, r_i_m / rn);
  // eq:gravgrad:torque -- tau = (3 mu / r^3) rhat_b x (I rhat_b). The
  // scale factor is formed once and applied to the cross product, so the
  // structural zeros (spherical I, principal-axis alignment) stay exact:
  // every product in the cross-product difference carries a zero factor
  // or subtracts two identically rounded copies of the same real product.
  const double k = 3.0 * mu_m3ps2 / (rn * rn * rn);
  return k * rhat_b.cross(inertia_b_kgm2 * rhat_b);
}

}  // namespace models
}  // namespace star
