// Engine propulsion with thrust-vector control (FR-10): per-engine
// delivered thrust F = lambda F_vac - p_amb Ae with constant vacuum Isp,
// mass flow from the vacuum rating and throttle only, throttle limits,
// linear spool-up/spool-down ramps, ignition-count enforcement, and gimbal
// angle/rate limits. The engine produces a body force and a torque about
// the composite CG (star/models/massprops.hpp) plus the propellant
// consumption rate that drives tank depletion.
//
// The core never parses text (D-2): the structs below are plain SI-unit
// value types the Python validator fills across the binding. The actuator
// state advances once per step (engine_advance); force, torque, and mass
// flow are pure functions of parameters and state so the integrator may
// evaluate them at any stage without side effects.
//
// Math-library traceability (FR-29): the derivation lives in the
// propulsion chapter of docs/mathlib (ch:propulsion); the implementation
// echoes its equation labels eq:propulsion:thrust, eq:propulsion:mdot,
// eq:propulsion:direction, eq:propulsion:forcetorque, eq:propulsion:spool,
// and eq:propulsion:gimbalslew at the corresponding code.
#ifndef STAR_MODELS_PROPULSION_HPP
#define STAR_MODELS_PROPULSION_HPP

#include <Eigen/Dense>

namespace star {
namespace models {

// Standard gravity g0 [m/s^2] for the Isp <-> mass-flow conversion
// (eq:propulsion:mdot). Exact conventional value fixed by the 3rd CGPM
// (1901); published in BIPM, The International System of Units (SI),
// 9th edition (2019). Defined here rather than star/constants.hpp because
// propulsion is its only core consumer.
inline constexpr double STANDARD_GRAVITY_MPS2 = 9.80665;

// One engine, mounted rigidly in the FR-13 vehicle frame (+X forward).
// axis is the nominal thrust direction (unit; +X for an aft-mounted main
// engine); gimbal_axis_1/2 are the orthogonal unit deflection axes
// (perpendicular to axis; defaults +Y pitch, +Z yaw). All members SI,
// FR-13 vocabulary.
struct EngineParams {
  double thrust_vac_N = 0.0;    // rated vacuum thrust at full throttle
  double isp_vac_s = 0.0;       // constant vacuum specific impulse
  double exit_area_m2 = 0.0;    // nozzle exit area Ae
  double throttle_min = 1.0;    // lower throttle limit in [0, 1]
  double throttle_max = 1.0;    // upper throttle limit in [0, 1]
  double spool_up_s = 0.0;      // linear 0->1 spool time; 0 = immediate
  double spool_down_s = 0.0;    // linear 1->0 spool time; 0 = immediate
  int max_ignitions = 1;        // ignition budget (0 = never ignitable)
  double gimbal_limit_rad = 0.0;    // per-axis deflection clamp
  double gimbal_rate_radps = 0.0;   // per-axis slew rate limit
  Eigen::Vector3d position_m = Eigen::Vector3d::Zero();  // thrust point
  Eigen::Vector3d axis = Eigen::Vector3d::UnitX();
  Eigen::Vector3d gimbal_axis_1 = Eigen::Vector3d::UnitY();
  Eigen::Vector3d gimbal_axis_2 = Eigen::Vector3d::UnitZ();
};

// Per-step command. throttle is clamped to the configured limits while
// the engine runs; gimbal_rad are the commanded deflections about
// gimbal_axis_1/2, slewed and clamped per eq:propulsion:gimbalslew.
struct EngineCommand {
  bool run = false;
  double throttle = 1.0;
  Eigen::Vector2d gimbal_rad = Eigen::Vector2d::Zero();
};

// Actuator state. throttle_level is the DELIVERED fraction of the rated
// vacuum thrust: thrust and mass flow follow it (not the running flag),
// so a shut-down engine keeps thrusting through its spool-down ramp.
struct EngineState {
  bool running = false;
  int ignitions_used = 0;
  double throttle_level = 0.0;
  Eigen::Vector2d gimbal_rad = Eigen::Vector2d::Zero();
};

// Body force [N] and torque [N m] about the supplied CG, vehicle frame.
struct EngineForceTorque {
  Eigen::Vector3d force_N = Eigen::Vector3d::Zero();
  Eigen::Vector3d torque_Nm = Eigen::Vector3d::Zero();
};

// Advance the actuator state by one step (eq:propulsion:spool,
// eq:propulsion:gimbalslew): ignition bookkeeping (a run command on a
// non-running engine consumes one ignition and is REFUSED once the budget
// is exhausted - the engine stays off), linear throttle-level slew toward
// the clamped command (toward 0 after shutdown), and per-axis gimbal slew
// at exactly the configured rate with the angle clamped at the limit.
// Throws std::domain_error for invalid params or a negative/non-finite dt.
EngineState engine_advance(const EngineParams& params,
                           const EngineCommand& command,
                           const EngineState& state, double dt_s);

// Gimbal-deflected unit thrust direction (eq:propulsion:direction):
// Rodrigues rotation of the nominal axis about gimbal_axis_1 by
// gimbal_rad[0], then about gimbal_axis_2 by gimbal_rad[1].
Eigen::Vector3d engine_thrust_direction(const EngineParams& params,
                                        const Eigen::Vector2d& gimbal_rad);

// Delivered thrust magnitude [N] (eq:propulsion:thrust):
// lambda F_vac - p_amb Ae for lambda > 0, exactly 0 for lambda == 0 (an
// engine that is off exerts no back-pressure force). May be negative for
// a running engine deeply overexpanded at low throttle - documented, not
// clamped (ch:propulsion assumption 2).
double engine_thrust_N(const EngineParams& params, double throttle_level,
                       double p_amb_Pa);

// Propellant consumption rate [kg/s], POSITIVE while flowing
// (eq:propulsion:mdot): lambda F_vac / (g0 Isp_vac). Set by the vacuum
// rating and throttle only; ambient pressure never enters. The
// integration layer negates this when feeding the signed tank-depletion
// rates (star/models/massprops.hpp).
double engine_mdot_kgps(const EngineParams& params, double throttle_level);

// Body force and torque about cg_m for the current state
// (eq:propulsion:forcetorque): F * dhat and (position - cg) x force.
// Exactly zero at zero throttle level.
EngineForceTorque engine_force_torque(const EngineParams& params,
                                      const EngineState& state,
                                      double p_amb_Pa,
                                      const Eigen::Vector3d& cg_m);

}  // namespace models
}  // namespace star

#endif  // STAR_MODELS_PROPULSION_HPP
