// Attitude actuators (FR-1): on/off RCS thrusters in clusters with
// minimum-impulse-bit enforcement and force+torque coupling about the
// composite CG, and reaction wheels with per-wheel momentum states, exact
// torque/momentum saturation, and the total-system angular-momentum
// bookkeeping that makes conservation testable.
//
// The core never parses text (D-2): the structs below are plain SI-unit
// value types the Python validator fills across the binding; every
// function is pure, allocation-free, and libm-free.
//
// Math-library traceability (FR-29): the derivation lives in the
// actuators chapter of docs/mathlib (ch:actuators); the implementation
// echoes its equation labels eq:actuators:mib, eq:actuators:rcscoupling,
// eq:actuators:wheelclamp, eq:actuators:wheelreaction, and
// eq:actuators:totalmomentum at the corresponding code.
#ifndef STAR_MODELS_ACTUATORS_HPP
#define STAR_MODELS_ACTUATORS_HPP

#include <vector>

#include <Eigen/Dense>

namespace star {
namespace models {

// One on/off thruster, rigid in the FR-13 vehicle frame (+X forward).
// direction is the unit direction of the FORCE the thruster applies to
// the vehicle (opposite the exhaust). All members SI, FR-13 vocabulary.
struct RcsThrusterParams {
  Eigen::Vector3d position_m = Eigen::Vector3d::Zero();
  Eigen::Vector3d direction = Eigen::Vector3d::UnitX();
  double thrust_N = 0.0;
  double mib_Ns = 0.0;  // minimum impulse bit
};

// A cluster is an ordered set of thrusters commanded individually; sums
// iterate in configuration order (D-10 fixed order).
struct RcsClusterParams {
  std::vector<RcsThrusterParams> thrusters;
};

// Instantaneous body force [N] and torque [N m] about the supplied CG
// while a thruster is held on (the integrands of the pulse impulse).
struct RcsForceTorque {
  Eigen::Vector3d force_N = Eigen::Vector3d::Zero();
  Eigen::Vector3d torque_Nm = Eigen::Vector3d::Zero();
};

// Delivered pulse impulse with MIB enforcement (eq:actuators:mib): a
// commanded pulse with thrust * duration below mib_Ns delivers EXACTLY
// zero; at or above it delivers exactly thrust * duration.
struct RcsImpulse {
  double delivered_Ns = 0.0;  // scalar impulse magnitude actually delivered
  Eigen::Vector3d impulse_Ns = Eigen::Vector3d::Zero();
  Eigen::Vector3d angular_impulse_Nms = Eigen::Vector3d::Zero();
};

// Force and torque of one thruster held on, about cg_m
// (eq:actuators:rcscoupling integrands). Throws std::domain_error for
// non-finite inputs, a non-unit direction, or a negative thrust.
RcsForceTorque rcs_force_torque(const RcsThrusterParams& thruster,
                                const Eigen::Vector3d& cg_m);

// Cluster sum over the active thrusters (on.size() must equal the
// thruster count), in configuration order.
RcsForceTorque rcs_cluster_force_torque(const RcsClusterParams& cluster,
                                        const std::vector<bool>& on,
                                        const Eigen::Vector3d& cg_m);

// MIB-enforced pulse (eq:actuators:mib, eq:actuators:rcscoupling).
// Throws std::domain_error for a negative or non-finite duration or MIB.
RcsImpulse rcs_pulse(const RcsThrusterParams& thruster, double duration_s,
                     const Eigen::Vector3d& cg_m);

// One reaction wheel: unit spin axis in the vehicle frame, torque
// saturation, and momentum saturation (the rails +-momentum_max_Nms).
struct WheelParams {
  Eigen::Vector3d axis = Eigen::Vector3d::UnitX();
  double torque_max_Nm = 0.0;
  double momentum_max_Nms = 0.0;
};

// Wheel state: stored spin angular momentum relative to the body, signed
// along the spin axis.
struct WheelState {
  double momentum_Nms = 0.0;
};

// One wheel step result. torque_Nm is the delivered motor torque on the
// wheel (signed scalar along axis); body_torque_Nm = -torque_Nm * axis is
// the reaction on the vehicle (eq:actuators:wheelreaction).
struct WheelStepResult {
  double torque_Nm = 0.0;
  Eigen::Vector3d body_torque_Nm = Eigen::Vector3d::Zero();
  WheelState state;
};

// Advance one wheel by dt under a commanded motor torque
// (eq:actuators:wheelclamp): clamp the command at +-torque_max_Nm, then
// rail-limit the momentum - a step that would cross a rail delivers only
// the torque that lands the momentum EXACTLY on the rail, and a wheel
// already on the rail delivers exactly zero torque for a same-sign
// command (it cannot accept further momentum of that sign; documented
// saturation semantics, ch:actuators). Desaturating commands act
// normally. Throws std::domain_error for invalid params, a non-positive
// or non-finite dt, or an initial momentum outside the rails.
WheelStepResult wheel_step(const WheelParams& wheel, double torque_cmd_Nm,
                           const WheelState& state, double dt_s);

// Total angular momentum of the vehicle-plus-wheels system, body frame
// (eq:actuators:totalmomentum): I * omega + sum_i h_i * axis_i. The
// caller rotates into the inertial frame for conservation checks.
// wheels.size() must equal states.size().
Eigen::Vector3d total_angular_momentum_Nms(
    const Eigen::Matrix3d& inertia_kgm2, const Eigen::Vector3d& omega_radps,
    const std::vector<WheelParams>& wheels,
    const std::vector<WheelState>& states);

}  // namespace models
}  // namespace star

#endif  // STAR_MODELS_ACTUATORS_HPP
