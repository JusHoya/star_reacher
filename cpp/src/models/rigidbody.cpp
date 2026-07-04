// Rigid-body attitude dynamics (FR-1). Derivation and validation
// evidence: docs/mathlib chapter ch:rigidbody. Equation labels from that
// chapter are echoed verbatim at the corresponding code (FR-29
// traceability). Everything here is a pointwise algebraic map evaluated
// inside the deterministic time loop: fixed evaluation order, no heap
// allocation, IEEE basic operations plus one square root in the
// between-step renormalization helper.
#include "star/models/rigidbody.hpp"

#include <cmath>
#include <stdexcept>

namespace star {
namespace models {

// eq:rigidbody:qdot -- qdot = 1/2 q (x) [0, omega], the Hamilton product
// (eq:notation:hamiltonproduct) written out in scalar-first components so
// the convention is visible at the code, matching
// star::rotation::quat_multiply.
Eigen::Quaterniond rigidbody_qdot(const Eigen::Quaterniond& q_i2b,
                                  const Eigen::Vector3d& omega_b_radps) {
  const double qw = q_i2b.w();
  const double qx = q_i2b.x();
  const double qy = q_i2b.y();
  const double qz = q_i2b.z();
  const double wx = omega_b_radps[0];
  const double wy = omega_b_radps[1];
  const double wz = omega_b_radps[2];
  return Eigen::Quaterniond(0.5 * (-qx * wx - qy * wy - qz * wz),
                            0.5 * (qw * wx + qy * wz - qz * wy),
                            0.5 * (qw * wy - qx * wz + qz * wx),
                            0.5 * (qw * wz + qx * wy - qy * wx));
}

Eigen::Vector3d rigidbody_omega_dot(
    const Eigen::Vector3d& omega_b_radps, const Eigen::Matrix3d& inertia_b_kgm2,
    const Eigen::Matrix3d& inertia_dot_b_kgm2ps,
    const Eigen::Vector3d& torque_b_nm) {
  // eq:rigidbody:euler -- I omega_dot = tau - omega x (I omega) - Idot omega.
  // The instantaneous body-frame angular momentum I*omega is formed once;
  // the gyroscopic cross product and the Idot term are subtracted in a
  // fixed order so the structural zeros of the property tests (principal-
  // axis spin, spherical inertia) stay exact.
  const Eigen::Vector3d h_b = inertia_b_kgm2 * omega_b_radps;
  const Eigen::Vector3d rhs = torque_b_nm - omega_b_radps.cross(h_b) -
                              inertia_dot_b_kgm2ps * omega_b_radps;
  // eq:rigidbody:omegadot -- omega_dot = I^{-1} rhs via the closed-form
  // 3x3 cofactor inverse: IEEE basic operations only, no pivot search, no
  // allocation. Relative error is a small multiple of kappa(I)*eps;
  // physical inertia tensors are well conditioned (ch:rigidbody,
  // implementation notes). A singular I divides by a zero determinant and
  // IEEE non-finite values propagate (documented out-of-domain response).
  return inertia_b_kgm2.inverse() * rhs;
}

void rigidbody_rhs(const double* y_att, const Eigen::Matrix3d& inertia_b_kgm2,
                   const Eigen::Matrix3d& inertia_dot_b_kgm2ps,
                   const Eigen::Vector3d& torque_b_nm, double* ydot_att) {
  // Packed slice layout [q_w, q_x, q_y, q_z, omega_x, omega_y, omega_z];
  // scalar-first per the notation chapter's Eigen mapping rule 1 (the
  // Quaterniond value constructor takes w first).
  const Eigen::Quaterniond q(y_att[0], y_att[1], y_att[2], y_att[3]);
  const Eigen::Vector3d w(y_att[4], y_att[5], y_att[6]);
  const Eigen::Quaterniond qd = rigidbody_qdot(q, w);
  const Eigen::Vector3d wd =
      rigidbody_omega_dot(w, inertia_b_kgm2, inertia_dot_b_kgm2ps,
                          torque_b_nm);
  ydot_att[0] = qd.w();
  ydot_att[1] = qd.x();
  ydot_att[2] = qd.y();
  ydot_att[3] = qd.z();
  ydot_att[4] = wd[0];
  ydot_att[5] = wd[1];
  ydot_att[6] = wd[2];
}

double rigidbody_renormalize(double* y_att) {
  // FR-1 post-step normalization; drift being removed is bounded by
  // eq:rigidbody:normdrift. Rescaling by 1/|q| leaves the represented
  // attitude exactly unchanged (degree-2 homogeneity of the DCM), so this
  // is pure norm bookkeeping, never an attitude correction.
  const double n =
      std::sqrt(y_att[0] * y_att[0] + y_att[1] * y_att[1] +
                y_att[2] * y_att[2] + y_att[3] * y_att[3]);
  if (!(n > 0.0) || !std::isfinite(n)) {
    // A zero or non-finite quaternion has no direction; normalizing it
    // would fabricate an attitude, so fail loudly instead (abort-on-
    // missing-critical-input discipline, mirroring
    // star::rotation::quat_normalize).
    throw std::domain_error(
        "rigidbody_renormalize: zero or non-finite quaternion norm");
  }
  y_att[0] /= n;
  y_att[1] /= n;
  y_att[2] /= n;
  y_att[3] /= n;
  return n - 1.0;
}

}  // namespace models
}  // namespace star
