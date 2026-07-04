// RCS thrusters and reaction wheels (FR-1). Derivation: docs/mathlib
// chapter ch:actuators. No libm calls anywhere in this module: every
// operation is an IEEE-754 basic operation in fixed order, so refusal,
// saturation, and rail states are bit-exact and the module is
// bit-portable across platforms (D-10).
#include "star/models/actuators.hpp"

#include <cmath>
#include <stdexcept>

namespace star {
namespace models {

namespace {

bool is_unit(const Eigen::Vector3d& v) {
  return v.allFinite() && std::fabs(v.norm() - 1.0) <= 1e-9;
}

void check_thruster(const RcsThrusterParams& t) {
  if (!t.position_m.allFinite() || !is_unit(t.direction) ||
      !std::isfinite(t.thrust_N) || t.thrust_N < 0.0 ||
      !std::isfinite(t.mib_Ns) || t.mib_Ns < 0.0) {
    throw std::domain_error(
        "actuators: thruster needs a finite position, a unit direction, "
        "and non-negative thrust and MIB");
  }
}

void check_wheel(const WheelParams& w) {
  if (!is_unit(w.axis) || !std::isfinite(w.torque_max_Nm) ||
      w.torque_max_Nm < 0.0 || !std::isfinite(w.momentum_max_Nms) ||
      w.momentum_max_Nms < 0.0) {
    throw std::domain_error(
        "actuators: wheel needs a unit axis and non-negative torque and "
        "momentum limits");
  }
}

}  // namespace

RcsForceTorque rcs_force_torque(const RcsThrusterParams& thruster,
                                const Eigen::Vector3d& cg_m) {
  check_thruster(thruster);
  if (!cg_m.allFinite()) {
    throw std::domain_error("actuators: cg must be finite");
  }
  // eq:actuators:rcscoupling integrands -- constant force F dhat and its
  // moment about the composite CG while the valve is open.
  RcsForceTorque out;
  out.force_N = thruster.thrust_N * thruster.direction;
  out.torque_Nm = (thruster.position_m - cg_m).cross(out.force_N);
  return out;
}

RcsForceTorque rcs_cluster_force_torque(const RcsClusterParams& cluster,
                                        const std::vector<bool>& on,
                                        const Eigen::Vector3d& cg_m) {
  if (on.size() != cluster.thrusters.size()) {
    throw std::domain_error(
        "actuators: cluster needs one on/off flag per thruster");
  }
  // Configuration-order sum (D-10 fixed order).
  RcsForceTorque out;
  for (std::size_t i = 0; i < cluster.thrusters.size(); ++i) {
    if (!on[i]) {
      continue;
    }
    const RcsForceTorque one = rcs_force_torque(cluster.thrusters[i], cg_m);
    out.force_N += one.force_N;
    out.torque_Nm += one.torque_Nm;
  }
  return out;
}

RcsImpulse rcs_pulse(const RcsThrusterParams& thruster, double duration_s,
                     const Eigen::Vector3d& cg_m) {
  check_thruster(thruster);
  if (!std::isfinite(duration_s) || duration_s < 0.0 ||
      !cg_m.allFinite()) {
    throw std::domain_error(
        "actuators: pulse duration must be finite and non-negative and "
        "cg finite");
  }
  // eq:actuators:mib -- the ideal impulse thrust * duration is delivered
  // in full at or above the MIB and refused EXACTLY (literal zeros)
  // below it: the valve never effectively opens, so no residue is
  // possible (Phase 4 exit criterion 7).
  RcsImpulse out;
  const double ideal_Ns = thruster.thrust_N * duration_s;
  if (ideal_Ns < thruster.mib_Ns) {
    return out;
  }
  out.delivered_Ns = ideal_Ns;
  // eq:actuators:rcscoupling -- linear impulse along the unit direction
  // and angular impulse about the composite CG.
  out.impulse_Ns = ideal_Ns * thruster.direction;
  out.angular_impulse_Nms =
      (thruster.position_m - cg_m).cross(out.impulse_Ns);
  return out;
}

WheelStepResult wheel_step(const WheelParams& wheel, double torque_cmd_Nm,
                           const WheelState& state, double dt_s) {
  check_wheel(wheel);
  if (!std::isfinite(torque_cmd_Nm) || !std::isfinite(dt_s) || dt_s <= 0.0 ||
      !std::isfinite(state.momentum_Nms) ||
      std::fabs(state.momentum_Nms) > wheel.momentum_max_Nms) {
    throw std::domain_error(
        "actuators: wheel step needs a finite command, a positive dt, and "
        "a momentum inside the rails");
  }
  // eq:actuators:wheelclamp -- torque clamp first, then the momentum
  // rail. Rail landings assign the configured limit itself (not the
  // accumulated sum), so saturation states are bit-exact; on the rail a
  // same-sign command yields (h_max - h)/dt = 0/dt = 0 exactly.
  double tau = torque_cmd_Nm;
  if (tau > wheel.torque_max_Nm) {
    tau = wheel.torque_max_Nm;
  } else if (tau < -wheel.torque_max_Nm) {
    tau = -wheel.torque_max_Nm;
  }
  WheelStepResult out;
  const double h_unclamped = state.momentum_Nms + tau * dt_s;
  if (h_unclamped > wheel.momentum_max_Nms) {
    out.torque_Nm = (wheel.momentum_max_Nms - state.momentum_Nms) / dt_s;
    out.state.momentum_Nms = wheel.momentum_max_Nms;
  } else if (h_unclamped < -wheel.momentum_max_Nms) {
    out.torque_Nm = (-wheel.momentum_max_Nms - state.momentum_Nms) / dt_s;
    out.state.momentum_Nms = -wheel.momentum_max_Nms;
  } else {
    out.torque_Nm = tau;
    out.state.momentum_Nms = h_unclamped;
  }
  // eq:actuators:wheelreaction -- Newton's third law across the motor:
  // the body receives the negative of the wheel torque along the axis.
  out.body_torque_Nm = -out.torque_Nm * wheel.axis;
  return out;
}

Eigen::Vector3d total_angular_momentum_Nms(
    const Eigen::Matrix3d& inertia_kgm2, const Eigen::Vector3d& omega_radps,
    const std::vector<WheelParams>& wheels,
    const std::vector<WheelState>& states) {
  if (wheels.size() != states.size()) {
    throw std::domain_error(
        "actuators: momentum bookkeeping needs one state per wheel");
  }
  if (!inertia_kgm2.allFinite() || !omega_radps.allFinite()) {
    throw std::domain_error(
        "actuators: inertia and omega must be finite");
  }
  // eq:actuators:totalmomentum -- H = I omega + sum h_i a_i, body frame;
  // wheel terms summed in configuration order (D-10).
  Eigen::Vector3d h = inertia_kgm2 * omega_radps;
  for (std::size_t i = 0; i < wheels.size(); ++i) {
    check_wheel(wheels[i]);
    if (!std::isfinite(states[i].momentum_Nms)) {
      throw std::domain_error("actuators: wheel momentum must be finite");
    }
    h += states[i].momentum_Nms * wheels[i].axis;
  }
  return h;
}

}  // namespace models
}  // namespace star
