// Gravity-gradient torque (FR-1): the first-order tidal torque a finite
// rigid body experiences about its center of mass in a point-mass central
// field, tau = (3 mu / r^3) rhat_b x (I rhat_b), about the current central
// body using states and inertia already available. Purely algebraic in its
// inputs: no dependence on the environment model or any vehicle structure,
// and no body constants (mu comes from the caller).
//
// Math-library traceability (FR-29): the derivation lives in the
// gravity-gradient chapter of docs/mathlib (ch:gravgrad); the
// implementation echoes its equation label `eq:gravgrad:torque` at the
// corresponding code, and the validation test echoes `eq:gravgrad:pitch`,
// `eq:gravgrad:libfreq`, and `eq:gravgrad:pendulum`.
#ifndef STAR_MODELS_GRAVGRAD_HPP
#define STAR_MODELS_GRAVGRAD_HPP

#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace star {
namespace models {

// Gravity-gradient torque [N m] in body axes (eq:gravgrad:torque):
//   tau = (3 mu / r^3) rhat_b x (I rhat_b),
// where r_i_m points FROM THE CENTRAL BODY TO THE VEHICLE center of mass
// in the inertial frame, rhat_b is its unit vector expressed in body axes
// through q_i2b, and inertia_b_kgm2 is the body-axes inertia tensor about
// the center of mass (the same tensor the attitude dynamics uses). Unit
// norm of q_i2b is the caller's invariant (FR-1 post-step normalization);
// a non-unit quaternion scales the torque by |q|^4 (ch:gravgrad, domain
// of validity). Exactly zero for a spherical inertia tensor and when
// rhat_b is a principal axis. r = 0 propagates IEEE non-finite values
// rather than throwing (deterministic-loop rule). No per-call allocation.
Eigen::Vector3d gravgrad_torque(double mu_m3ps2, const Eigen::Vector3d& r_i_m,
                                const Eigen::Quaterniond& q_i2b,
                                const Eigen::Matrix3d& inertia_b_kgm2);

}  // namespace models
}  // namespace star

#endif  // STAR_MODELS_GRAVGRAD_HPP
