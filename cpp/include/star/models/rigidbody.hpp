// Rigid-body attitude dynamics (FR-1): quaternion attitude kinematics and
// Euler's rotational equation with time-varying inertia including the
// Idot*omega term, exposed as pure right-hand-side pieces. The Phase 4
// vehicle layer supplies the inertia tensor, its time derivative, and the
// total body torque from mass properties, actuators, and environmental
// torques (star/models/gravgrad.hpp); this module depends on none of those
// structures and owns no physical constants.
//
// Conventions (ch:notation, D-7): q_i2b is the Hamilton, scalar-first,
// inertial-to-body frame-transformation quaternion; omega_b is the angular
// velocity of the body frame with respect to the inertial frame, resolved
// in body axes; all units SI.
//
// Math-library traceability (FR-29): the derivations live in the
// rigid-body chapter of docs/mathlib (ch:rigidbody); the implementation
// echoes its equation labels `eq:rigidbody:qdot`, `eq:rigidbody:euler`,
// `eq:rigidbody:omegadot`, and `eq:rigidbody:normdrift` at the
// corresponding code.
#ifndef STAR_MODELS_RIGIDBODY_HPP
#define STAR_MODELS_RIGIDBODY_HPP

#include <cstddef>

#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace star {
namespace models {

// Packed attitude state slice shared with the Phase 4 vehicle state
// vector: y = [q_w, q_x, q_y, q_z, omega_x, omega_y, omega_z]
// (scalar-first per the notation chapter's Eigen mapping rule 1).
inline constexpr std::size_t kAttitudeStateDim = 7;

// Quaternion attitude kinematics (eq:rigidbody:qdot):
//   qdot = 1/2 q_i2b (x) [0, omega_b].
// Degree-one homogeneous in q and exactly norm-preserving in the
// continuous dynamics (eq:rigidbody:qnorm), so any nonzero q is a valid
// input; unit norm is maintained between steps by rigidbody_renormalize,
// not here (normalizing inside the RHS would change the ODE the
// integrator sees).
Eigen::Quaterniond rigidbody_qdot(const Eigen::Quaterniond& q_i2b,
                                  const Eigen::Vector3d& omega_b_radps);

// Euler's rotational equation with time-varying inertia
// (eq:rigidbody:euler / eq:rigidbody:omegadot):
//   omega_dot = I^{-1} (tau - omega x (I omega) - Idot omega).
// inertia_b_kgm2 and inertia_dot_b_kgm2ps are the body-axes inertia
// tensor about the instantaneous center of mass and its time derivative
// (symmetric; consistency of the pair is the mass-property layer's
// contract); torque_b_nm is the total torque about the same point. A
// singular inertia propagates IEEE non-finite values rather than
// throwing (deterministic-loop rule; see ch:rigidbody, domain of
// validity).
Eigen::Vector3d rigidbody_omega_dot(const Eigen::Vector3d& omega_b_radps,
                                    const Eigen::Matrix3d& inertia_b_kgm2,
                                    const Eigen::Matrix3d& inertia_dot_b_kgm2ps,
                                    const Eigen::Vector3d& torque_b_nm);

// Combined RHS piece on the packed 7-component attitude slice (layout
// above): reads y_att[0..6], writes ydot_att[0..6]. Suitable for direct
// embedding in a larger state vector behind integrate::RhsRef; the caller
// evaluates I, Idot, and tau for the current time/state and passes them
// down as plain inputs. y_att and ydot_att must not alias.
void rigidbody_rhs(const double* y_att, const Eigen::Matrix3d& inertia_b_kgm2,
                   const Eigen::Matrix3d& inertia_dot_b_kgm2ps,
                   const Eigen::Vector3d& torque_b_nm, double* ydot_att);

// Post-step quaternion renormalization on the packed slice (FR-1 norm
// policy): rescales y_att[0..3] to unit norm in place (the rate slice is
// untouched) and returns the signed pre-normalization deviation
// |q| - 1, so callers can monitor the documented drift bound
// (eq:rigidbody:normdrift). Rescaling leaves the represented attitude
// exactly unchanged (the DCM is degree-2 homogeneous in q). Throws
// std::domain_error on a zero or non-finite norm rather than fabricating
// an attitude, mirroring star::rotation::quat_normalize; intended for
// between-step use by the propagation layer, never inside an RHS.
double rigidbody_renormalize(double* y_att);

}  // namespace models
}  // namespace star

#endif  // STAR_MODELS_RIGIDBODY_HPP
