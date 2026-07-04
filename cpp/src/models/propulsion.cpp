// Engine propulsion with TVC (FR-10). Derivation: docs/mathlib chapter
// ch:propulsion. The only libm calls are the sin/cos pair per gimbal axis
// in the thrust-direction rotation; the thrust, mass-flow, spool, and slew
// paths are IEEE-754 basic operations in fixed order (D-10). Zero throttle
// level returns literal zeros so an engine that is off cannot leave
// residue in the force composition.
#include "star/models/propulsion.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace star {
namespace models {

namespace {

bool is_unit(const Eigen::Vector3d& v) {
  return v.allFinite() && std::fabs(v.norm() - 1.0) <= 1e-9;
}

// One shared validation so every entry point enforces the ch:propulsion
// domain identically. The Python layer (D-2) owns user-facing validation;
// this re-checks only what keeps the math well-defined.
void check_params(const EngineParams& p) {
  if (!std::isfinite(p.thrust_vac_N) || p.thrust_vac_N < 0.0 ||
      !std::isfinite(p.isp_vac_s) || p.isp_vac_s <= 0.0 ||
      !std::isfinite(p.exit_area_m2) || p.exit_area_m2 < 0.0) {
    throw std::domain_error(
        "propulsion: thrust_vac_N and exit_area_m2 must be finite and "
        "non-negative, isp_vac_s finite and positive");
  }
  if (!std::isfinite(p.throttle_min) || !std::isfinite(p.throttle_max) ||
      p.throttle_min < 0.0 || p.throttle_max > 1.0 ||
      p.throttle_min > p.throttle_max) {
    throw std::domain_error(
        "propulsion: throttle limits must satisfy 0 <= min <= max <= 1");
  }
  if (!std::isfinite(p.spool_up_s) || p.spool_up_s < 0.0 ||
      !std::isfinite(p.spool_down_s) || p.spool_down_s < 0.0 ||
      p.max_ignitions < 0) {
    throw std::domain_error(
        "propulsion: spool times and ignition budget must be non-negative");
  }
  if (!std::isfinite(p.gimbal_limit_rad) || p.gimbal_limit_rad < 0.0 ||
      !std::isfinite(p.gimbal_rate_radps) || p.gimbal_rate_radps < 0.0) {
    throw std::domain_error(
        "propulsion: gimbal limit and rate must be non-negative");
  }
  if (!p.position_m.allFinite() || !is_unit(p.axis) ||
      !is_unit(p.gimbal_axis_1) || !is_unit(p.gimbal_axis_2) ||
      std::fabs(p.gimbal_axis_1.dot(p.gimbal_axis_2)) > 1e-9 ||
      std::fabs(p.gimbal_axis_1.dot(p.axis)) > 1e-9 ||
      std::fabs(p.gimbal_axis_2.dot(p.axis)) > 1e-9) {
    throw std::domain_error(
        "propulsion: axis and gimbal axes must be finite unit vectors, "
        "gimbal axes orthogonal to each other and to the nominal axis");
  }
}

// Rodrigues rotation of v about the unit axis a by angle
// (eq:propulsion:direction): v cos + (a x v) sin + a (a.v)(1 - cos).
Eigen::Vector3d rodrigues(const Eigen::Vector3d& a, double angle,
                          const Eigen::Vector3d& v) {
  const double c = std::cos(angle);
  const double s = std::sin(angle);
  return c * v + s * a.cross(v) + (a.dot(v) * (1.0 - c)) * a;
}

double clamp(double x, double lo, double hi) {
  return std::min(std::max(x, lo), hi);
}

}  // namespace

EngineState engine_advance(const EngineParams& params,
                           const EngineCommand& command,
                           const EngineState& state, double dt_s) {
  check_params(params);
  if (!std::isfinite(dt_s) || dt_s < 0.0 ||
      !std::isfinite(command.throttle) ||
      !command.gimbal_rad.allFinite() ||
      !std::isfinite(state.throttle_level) ||
      !state.gimbal_rad.allFinite()) {
    throw std::domain_error(
        "propulsion: engine_advance needs finite command/state and a "
        "non-negative dt");
  }
  EngineState out = state;

  // Ignition bookkeeping: a run command on a non-running engine consumes
  // one ignition; an exhausted budget REFUSES the command and the engine
  // stays off (ch:propulsion implementation note 4). Shutdown never
  // consumes anything, and re-igniting during spool-down costs a new
  // count.
  if (command.run && !state.running) {
    if (state.ignitions_used < params.max_ignitions) {
      out.running = true;
      out.ignitions_used = state.ignitions_used + 1;
    }
  } else if (!command.run && state.running) {
    out.running = false;
  }

  // eq:propulsion:spool -- the delivered level slews linearly toward the
  // clamped command while running and toward 0 when off; a zero spool
  // time applies the target immediately. Away from the endpoints each
  // step adds exactly dt/t_up (or subtracts dt/t_down).
  const double target =
      out.running
          ? clamp(command.throttle, params.throttle_min, params.throttle_max)
          : 0.0;
  const double dl = target - state.throttle_level;
  if (dl > 0.0) {
    out.throttle_level = (params.spool_up_s > 0.0)
                             ? state.throttle_level +
                                   std::min(dl, dt_s / params.spool_up_s)
                             : target;
  } else if (dl < 0.0) {
    out.throttle_level = (params.spool_down_s > 0.0)
                             ? state.throttle_level +
                                   std::max(dl, -dt_s / params.spool_down_s)
                             : target;
  }

  // eq:propulsion:gimbalslew -- per-axis slew at exactly the configured
  // rate limit, then the per-axis (square) angle clamp.
  const double max_step = params.gimbal_rate_radps * dt_s;
  for (int i = 0; i < 2; ++i) {
    const double step =
        clamp(command.gimbal_rad[i] - state.gimbal_rad[i], -max_step,
              max_step);
    out.gimbal_rad[i] = clamp(state.gimbal_rad[i] + step,
                              -params.gimbal_limit_rad,
                              params.gimbal_limit_rad);
  }
  return out;
}

Eigen::Vector3d engine_thrust_direction(const EngineParams& params,
                                        const Eigen::Vector2d& gimbal_rad) {
  check_params(params);
  if (!gimbal_rad.allFinite()) {
    throw std::domain_error("propulsion: gimbal angles must be finite");
  }
  // eq:propulsion:direction -- rotate about gimbal_axis_1 first, then
  // gimbal_axis_2 (fixed, documented order; both are proper rotations so
  // the result stays unit-norm to rounding).
  return rodrigues(
      params.gimbal_axis_2, gimbal_rad[1],
      rodrigues(params.gimbal_axis_1, gimbal_rad[0], params.axis));
}

double engine_thrust_N(const EngineParams& params, double throttle_level,
                       double p_amb_Pa) {
  check_params(params);
  if (!std::isfinite(throttle_level) || throttle_level < 0.0 ||
      throttle_level > 1.0 || !std::isfinite(p_amb_Pa) || p_amb_Pa < 0.0) {
    throw std::domain_error(
        "propulsion: throttle level must be in [0, 1] and ambient "
        "pressure non-negative and finite");
  }
  // eq:propulsion:thrust -- literal zero when off: a closed engine is
  // not flowing, so the exit plane carries no pressure imbalance.
  if (throttle_level == 0.0) {
    return 0.0;
  }
  return throttle_level * params.thrust_vac_N -
         p_amb_Pa * params.exit_area_m2;
}

double engine_mdot_kgps(const EngineParams& params, double throttle_level) {
  check_params(params);
  if (!std::isfinite(throttle_level) || throttle_level < 0.0 ||
      throttle_level > 1.0) {
    throw std::domain_error(
        "propulsion: throttle level must be in [0, 1]");
  }
  // eq:propulsion:mdot -- the mass flow follows the vacuum rating and
  // throttle only; back pressure reduces delivered thrust, never the
  // propellant consumption (ch:propulsion).
  return throttle_level * params.thrust_vac_N /
         (STANDARD_GRAVITY_MPS2 * params.isp_vac_s);
}

EngineForceTorque engine_force_torque(const EngineParams& params,
                                      const EngineState& state,
                                      double p_amb_Pa,
                                      const Eigen::Vector3d& cg_m) {
  if (!cg_m.allFinite()) {
    throw std::domain_error("propulsion: cg must be finite");
  }
  EngineForceTorque out;
  const double f = engine_thrust_N(params, state.throttle_level, p_amb_Pa);
  if (f == 0.0) {
    return out;  // exact zeros, including the zero-throttle case
  }
  // eq:propulsion:forcetorque -- F dhat and (r_e - cg) x F.
  out.force_N = f * engine_thrust_direction(params, state.gimbal_rad);
  out.torque_Nm = (params.position_m - cg_m).cross(out.force_N);
  return out;
}

}  // namespace models
}  // namespace star
