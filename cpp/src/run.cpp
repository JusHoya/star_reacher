// Batch runs: config in, SRLOG out. Model math lives in chapter-tracked
// modules under cpp/src/models/; this file owns only the orchestration -
// stepping, decimation/sampling, and logging.
#include "star/run.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "star/constants.hpp"
#include "star/ephemeris.hpp"
#include "star/events.hpp"
#include "star/frames.hpp"
#include "star/integrate.hpp"
#include "star/models/actuators.hpp"
#include "star/models/aero.hpp"
#include "star/models/atmosphere_hp.hpp"
#include "star/models/atmosphere_ussa76.hpp"
#include "star/models/environment.hpp"
#include "star/models/gravgrad.hpp"
#include "star/models/massprops.hpp"
#include "star/models/propulsion.hpp"
#include "star/models/rigidbody.hpp"
#include "star/models/twobody.hpp"
#include "star/models/vehicle6dof.hpp"
#include "star/rotation.hpp"
#include "star/srlog_writer.hpp"
#include "star/time.hpp"
#include "star/version.hpp"

namespace star {

double gm(const std::string& body) {
  if (body == "earth") {
    return models::central_body_gm(models::CentralBody::kEarth);
  }
  if (body == "moon") {
    return models::central_body_gm(models::CentralBody::kMoon);
  }
  if (body == "mars") {
    return models::central_body_gm(models::CentralBody::kMars);
  }
  throw std::invalid_argument(
      "unknown central body: \"" + body +
      "\" (supported: \"earth\", \"moon\", \"mars\")");
}

namespace {

// Defensive re-checks of invariants the Python validator already enforces
// (contract section 3). They exist so a mis-wired caller fails fast with a
// named reason instead of writing a structurally wrong log.
void check_config(const RunConfig& cfg) {
  if (cfg.integrator != "rk4") {
    throw std::invalid_argument("integrator must be \"rk4\" in Phase 1");
  }
  if (!(cfg.duration_s > 0.0)) {
    throw std::invalid_argument("duration_s must be > 0");
  }
  if (!(cfg.dt_s > 0.0)) {
    throw std::invalid_argument("dt_s must be > 0");
  }
  if (!(cfg.mass_kg > 0.0)) {
    throw std::invalid_argument("mass_kg must be > 0");
  }
  if (cfg.truth_rate_hz < 1) {
    throw std::invalid_argument("truth_rate_hz must be >= 1");
  }
  const double steps_exact = cfg.duration_s / cfg.dt_s;
  if (std::fabs(steps_exact - std::llround(steps_exact)) >
      1e-9 * steps_exact) {
    throw std::invalid_argument(
        "duration_s must be an integer multiple of dt_s");
  }
  const double decim_exact =
      1.0 / (cfg.dt_s * static_cast<double>(cfg.truth_rate_hz));
  if (decim_exact < 0.5 ||
      std::fabs(decim_exact - std::llround(decim_exact)) > 1e-9 * decim_exact) {
    throw std::invalid_argument(
        "1/(dt_s*truth_rate_hz) must be a positive integer (decimation only, "
        "never interpolation)");
  }
}

}  // namespace

RunSummary run_twobody(const RunConfig& cfg, const std::string& out_path) {
  check_config(cfg);
  const double mu = gm(cfg.central_body);
  const std::int64_t steps = std::llround(cfg.duration_s / cfg.dt_s);
  const std::int64_t decim = std::llround(
      1.0 / (cfg.dt_s * static_cast<double>(cfg.truth_rate_hz)));

  log::SrlogHeaderFields fields;
  fields.core_version = core_version();
  fields.git_hash = git_hash();
  fields.config_sha256 = cfg.config_sha256;
  fields.master_seed = cfg.master_seed;
  fields.oracle = cfg.oracle;
  fields.epoch_utc = cfg.epoch_utc;
  fields.central_body = cfg.central_body;
  fields.truth_rate_hz = cfg.truth_rate_hz;
  log::SrlogWriter writer(out_path, fields);

  // Placeholder attitude channels (contract section 2): identity quaternion
  // (Hamilton scalar-first, D-7), zero body rates, constant mass. They
  // establish the truth-group schema before attitude dynamics exist; the
  // format document flags them as placeholders.
  const double q_identity[4] = {1.0, 0.0, 0.0, 0.0};
  const Eigen::Vector3d w_zero = Eigen::Vector3d::Zero();

  // Shared fixed-step RK4 from the integrator library (FR-11); the model
  // contributes only its right-hand side. The rhs lambda outlives the loop,
  // satisfying integrate::RhsRef's non-owning lifetime contract.
  auto rhs = [mu](double t, const double* y, double* ydot) {
    models::twobody_rhs(mu, t, y, ydot);
  };
  const integrate::RhsRef f(rhs);
  integrate::Rk4 rk4(6);

  double y[6] = {cfg.r0_m[0],   cfg.r0_m[1],   cfg.r0_m[2],
                 cfg.v0_mps[0], cfg.v0_mps[1], cfg.v0_mps[2]};
  const Eigen::Map<const Eigen::Vector3d> r_m(y);
  const Eigen::Map<const Eigen::Vector3d> v_mps(y + 3);

  RunSummary summary;
  summary.steps = steps;

  writer.write_event(0.0, 1, "run_start");
  summary.event_records += 1;

  writer.write_truth(0.0, r_m, v_mps, q_identity, w_zero, cfg.mass_kg);
  summary.truth_records += 1;

  for (std::int64_t i = 1; i <= steps; ++i) {
    // Step time t = (i-1)*dt as a single multiply (not accumulated
    // addition): one rounding per timestamp keeps step times and logged
    // times well-conditioned and reproducible. The dynamics are autonomous,
    // so t only labels the step here.
    rk4.step(f, static_cast<double>(i - 1) * cfg.dt_s, y, cfg.dt_s, y);
    if (i % decim == 0) {
      // Record decimation semantics are unchanged from Phase 1: log at
      // t = 0 and every decim-th step, never interpolated (FR-16).
      writer.write_truth(static_cast<double>(i) * cfg.dt_s, r_m, v_mps,
                         q_identity, w_zero, cfg.mass_kg);
      summary.truth_records += 1;
    }
  }

  // run_end carries the configured duration verbatim, not steps*dt, so the
  // event timestamp is exactly the user's requested span.
  writer.write_event(cfg.duration_s, 2, "run_end");
  summary.event_records += 1;
  writer.close();

  for (int i = 0; i < 3; ++i) {
    summary.final_r_m[static_cast<std::size_t>(i)] = r_m[i];
    summary.final_v_mps[static_cast<std::size_t>(i)] = v_mps[i];
  }
  return summary;
}

namespace {

// Defensive re-checks for the environment path, mirroring check_config's
// role: a mis-wired caller fails fast with a named reason.
void check_config_env(const RunConfig& cfg) {
  if (!(cfg.duration_s > 0.0)) {
    throw std::invalid_argument("duration_s must be > 0");
  }
  if (!(cfg.mass_kg > 0.0)) {
    throw std::invalid_argument("mass_kg must be > 0");
  }
  if (cfg.truth_rate_hz < 1) {
    throw std::invalid_argument("truth_rate_hz must be >= 1");
  }
  if (cfg.integrator == "rk4") {
    if (!(cfg.dt_s > 0.0)) {
      throw std::invalid_argument("dt_s must be > 0");
    }
    const double steps_exact = cfg.duration_s / cfg.dt_s;
    if (std::fabs(steps_exact - std::llround(steps_exact)) >
        1e-9 * steps_exact) {
      throw std::invalid_argument(
          "duration_s must be an integer multiple of dt_s");
    }
    const double decim_exact =
        1.0 / (cfg.dt_s * static_cast<double>(cfg.truth_rate_hz));
    if (decim_exact < 0.5 || std::fabs(decim_exact - std::llround(
                                 decim_exact)) > 1e-9 * decim_exact) {
      throw std::invalid_argument(
          "1/(dt_s*truth_rate_hz) must be a positive integer (decimation "
          "only, never interpolation)");
    }
  } else if (cfg.integrator == "rkf78") {
    if (!(cfg.rtol > 0.0) || !(cfg.atol_pos_m > 0.0) ||
        !(cfg.atol_vel_mps > 0.0)) {
      throw std::invalid_argument(
          "rkf78 requires positive rtol, atol_pos_m, atol_vel_mps");
    }
    if (!(cfg.h_init_s > 0.0) || !(cfg.h_max_s >= cfg.h_init_s)) {
      throw std::invalid_argument(
          "rkf78 requires 0 < h_init_s <= h_max_s");
    }
    // With adaptive steps the truth log is sampled at k/truth_rate_hz from
    // the dense output; the final record must land on the duration exactly.
    const double records_exact =
        cfg.duration_s * static_cast<double>(cfg.truth_rate_hz);
    if (std::fabs(records_exact - std::llround(records_exact)) >
        1e-9 * records_exact) {
      throw std::invalid_argument(
          "duration_s * truth_rate_hz must be an integer (uniform-rate "
          "sampling of the dense output)");
    }
  } else {
    throw std::invalid_argument(
        "integrator must be \"rk4\" or \"rkf78\"");
  }
}

models::CentralBody central_body_from_name(const std::string& name) {
  if (name == "earth") return models::CentralBody::kEarth;
  if (name == "moon") return models::CentralBody::kMoon;
  if (name == "mars") return models::CentralBody::kMars;
  throw std::invalid_argument(
      "unknown central body: \"" + name +
      "\" (supported: \"earth\", \"moon\", \"mars\")");
}

models::AtmosphereModel atmosphere_from_name(const std::string& name) {
  if (name == "ussa76") return models::AtmosphereModel::kUssa76;
  if (name == "harris_priester") return models::AtmosphereModel::kHarrisPriester;
  if (name == "mars_exponential") return models::AtmosphereModel::kMarsExponential;
  throw std::invalid_argument(
      "unknown atmosphere model: \"" + name +
      "\" (supported: \"ussa76\", \"harris_priester\", \"mars_exponential\")");
}

// Uniform-rate truth sampler over the propagate() dense output: emits every
// record time t_k = k * (1/truth_rate_hz) that falls inside each accepted
// step. Record times use the same one-multiply-per-timestamp arithmetic as
// the fixed-step path, so rk4 and rkf78 runs of the same mission label their
// records identically.
struct TruthSampler {
  log::SrlogWriter* writer = nullptr;
  double inv_rate_s = 0.0;   // 1 / truth_rate_hz
  std::int64_t next_k = 1;   // record 0 (t = 0) is written before the loop
  std::int64_t last_k = 0;   // duration * rate, validated integral
  double mass_kg = 0.0;
  std::int64_t records = 0;

  void operator()(const integrate::DenseStep& d) {
    const double t_end = d.t0 + d.h;
    double y_s[6];
    while (next_k <= last_k) {
      const double t_k = static_cast<double>(next_k) * inv_rate_s;
      if (t_k > t_end) {
        break;
      }
      d.eval(t_k, y_s);
      const Eigen::Map<const Eigen::Vector3d> r(y_s);
      const Eigen::Map<const Eigen::Vector3d> v(y_s + 3);
      const double q_identity[4] = {1.0, 0.0, 0.0, 0.0};
      writer->write_truth(t_k, r, v, q_identity, Eigen::Vector3d::Zero(),
                          mass_kg);
      records += 1;
      next_k += 1;
    }
  }
};

}  // namespace

RunSummary run_env(const RunConfig& cfg, const std::string& out_path) {
  check_config_env(cfg);

  models::EnvironmentSpec spec;
  spec.central_body = central_body_from_name(cfg.central_body);
  spec.epoch_tai = {cfg.epoch_tai_day, cfg.epoch_tai_sec};
  spec.gravity_model = cfg.gravity_model;
  spec.gravity_field_path = cfg.gravity_field_path;
  spec.gravity_degree = cfg.gravity_degree;
  spec.gravity_order = cfg.gravity_order;
  spec.third_bodies = cfg.third_bodies;
  spec.srp_enabled = cfg.srp_enabled;
  spec.cr_a_over_m_m2pkg = cfg.cr_a_over_m_m2pkg;
  spec.srp_occulters = cfg.srp_occulters;
  spec.atmosphere = cfg.drag_enabled ? atmosphere_from_name(cfg.atmosphere)
                                     : models::AtmosphereModel::kNone;
  spec.cd_a_over_m_m2pkg = cfg.cd_a_over_m_m2pkg;
  spec.hp_exponent_n = cfg.hp_exponent_n;
  spec.ephemeris_path = cfg.ephemeris_path;
  models::EnvironmentModel model(spec);

  log::SrlogHeaderFields fields;
  fields.core_version = core_version();
  fields.git_hash = git_hash();
  fields.config_sha256 = cfg.config_sha256;
  fields.master_seed = cfg.master_seed;
  fields.oracle = cfg.oracle;
  fields.epoch_utc = cfg.epoch_utc;
  fields.central_body = cfg.central_body;
  fields.truth_rate_hz = cfg.truth_rate_hz;
  log::SrlogWriter writer(out_path, fields);

  // Placeholder attitude channels, unchanged from Phase 1 (contract
  // section 2): identity quaternion, zero body rates, constant mass.
  const double q_identity[4] = {1.0, 0.0, 0.0, 0.0};
  const Eigen::Vector3d w_zero = Eigen::Vector3d::Zero();

  auto rhs = [&model](double t, const double* y, double* ydot) {
    model.rhs(t, y, ydot);
  };
  const integrate::RhsRef f(rhs);

  double y[6] = {cfg.r0_m[0],   cfg.r0_m[1],   cfg.r0_m[2],
                 cfg.v0_mps[0], cfg.v0_mps[1], cfg.v0_mps[2]};
  const Eigen::Map<const Eigen::Vector3d> r_m(y);
  const Eigen::Map<const Eigen::Vector3d> v_mps(y + 3);

  RunSummary summary;

  writer.write_event(0.0, 1, "run_start");
  summary.event_records += 1;
  writer.write_truth(0.0, r_m, v_mps, q_identity, w_zero, cfg.mass_kg);
  summary.truth_records += 1;

  if (cfg.integrator == "rk4") {
    // Fixed-step decimation semantics identical to run_twobody (FR-16).
    const std::int64_t steps = std::llround(cfg.duration_s / cfg.dt_s);
    const std::int64_t decim = std::llround(
        1.0 / (cfg.dt_s * static_cast<double>(cfg.truth_rate_hz)));
    integrate::Rk4 rk4(6);
    for (std::int64_t i = 1; i <= steps; ++i) {
      rk4.step(f, static_cast<double>(i - 1) * cfg.dt_s, y, cfg.dt_s, y);
      if (i % decim == 0) {
        writer.write_truth(static_cast<double>(i) * cfg.dt_s, r_m, v_mps,
                           q_identity, w_zero, cfg.mass_kg);
        summary.truth_records += 1;
      }
    }
    summary.steps = steps;
  } else {
    // Adaptive RKF7(8) through the shared event-aware driver (no events
    // registered), sampling the truth log from the dense output.
    TruthSampler sampler;
    sampler.writer = &writer;
    sampler.inv_rate_s = 1.0 / static_cast<double>(cfg.truth_rate_hz);
    sampler.last_k = std::llround(cfg.duration_s *
                                  static_cast<double>(cfg.truth_rate_hz));
    sampler.mass_kg = cfg.mass_kg;

    events::PropagateOptions opt;
    opt.method = events::Method::kRkf78;
    opt.mode = events::StepMode::kAdaptive;
    opt.adaptive.groups = {
        {"position", 0, 3, cfg.rtol, cfg.atol_pos_m},
        {"velocity", 3, 3, cfg.rtol, cfg.atol_vel_mps},
    };
    opt.adaptive.h_init = cfg.h_init_s;
    opt.adaptive.h_max = cfg.h_max_s;

    double y_final[6];
    const events::PropagateResult res = events::propagate(
        f, 0.0, cfg.duration_s, y, 6, y_final, opt, nullptr, 0, nullptr,
        sampler);
    for (int i = 0; i < 6; ++i) {
      y[i] = y_final[i];
    }
    // The last record time k*inv_rate can exceed the final step's t0+h by a
    // rounding ulp (e.g. 54000 * (1/10 rounded) > 5400.0), in which case the
    // sampler could not emit it inside any step. The propagation ended at
    // exactly t = duration_s, so the final state IS the state at that record
    // time to within one time ulp; emit it with the canonical k*inv_rate
    // label so the record grid is complete and label arithmetic stays
    // uniform.
    while (sampler.next_k <= sampler.last_k) {
      writer.write_truth(
          static_cast<double>(sampler.next_k) * sampler.inv_rate_s, r_m,
          v_mps, q_identity, w_zero, cfg.mass_kg);
      sampler.records += 1;
      sampler.next_k += 1;
    }
    summary.truth_records += sampler.records;
    summary.steps = res.steps_accepted;
  }

  writer.write_event(cfg.duration_s, 2, "run_end");
  summary.event_records += 1;
  writer.close();

  for (int i = 0; i < 3; ++i) {
    summary.final_r_m[static_cast<std::size_t>(i)] = r_m[i];
    summary.final_v_mps[static_cast<std::size_t>(i)] = v_mps[i];
  }
  return summary;
}

// ===========================================================================
// Phase 4 vehicle 6DOF path. Everything below composes the chapter-tracked
// model modules (environment, massprops, propulsion, actuators, aero,
// gravgrad, rigidbody, vehicle6dof) into a full staged-vehicle run; the
// physics lives in those modules, this file owns only the stepping, the
// control-cycle zero-order hold (D-5), the event sequence, and the logging.
// ===========================================================================
namespace {

double deg2rad_v(double d) { return d * (constants::TWO_PI / 360.0); }

Eigen::Vector3d arr3(const std::array<double, 3>& a) {
  return Eigen::Vector3d(a[0], a[1], a[2]);
}

Eigen::Matrix3d mat3_flat(const std::array<double, 9>& a) {
  Eigen::Matrix3d m;
  m << a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8];
  return m;
}

// Two unit deflection axes orthogonal to a thrust axis (the propulsion module
// requires an orthonormal gimbal frame; for axis == +X these are +Y/+Z).
void gimbal_basis(const Eigen::Vector3d& axis, Eigen::Vector3d& g1,
                  Eigen::Vector3d& g2) {
  const Eigen::Vector3d ref = (std::fabs(axis.z()) < 0.9)
                                  ? Eigen::Vector3d::UnitZ()
                                  : Eigen::Vector3d::UnitX();
  g1 = (ref - ref.dot(axis) * axis).normalized();
  g2 = axis.cross(g1);
}

struct EngineRt {
  models::EngineParams params;
  models::EngineState state;
  models::EngineCommand command;
  int stage_idx = -1;
  int tank_idx = -1;  // global index into VehicleRuntime::tanks, or -1
  std::string name;
};
struct TankRt {
  models::TankParams params;
  double mass_kg = 0.0;
  int stage_idx = -1;
};
struct WheelRt {
  models::WheelParams params;
  models::WheelState state;
  int stage_idx = -1;
};
struct JettRt {
  models::BodyProps body;
  bool attached = true;
  int stage_idx = -1;
  std::string name;
};
struct StageRt {
  models::BodyProps dry;
  bool attached = true;
  std::string name;
  std::vector<int> tanks;
  std::vector<int> engines;
  std::vector<int> jett;
};
struct VehicleRuntime {
  std::vector<StageRt> stages;
  std::vector<TankRt> tanks;
  std::vector<EngineRt> engines;
  std::vector<WheelRt> wheels;
  std::vector<JettRt> jett;
  std::vector<models::AeroTables> aero;
};

VehicleRuntime build_vehicle(const VehicleConfig& v) {
  VehicleRuntime rt;
  int stage_idx = 0;
  for (const StageCfg& s : v.stages) {
    StageRt st;
    st.name = s.name;
    st.dry.mass_kg = s.dry_mass_kg;
    st.dry.cg_m = arr3(s.dry_cg_m);
    st.dry.inertia_kgm2 = mat3_flat(s.dry_inertia_kgm2);
    for (const TankCfg& t : s.tanks) {
      TankRt tr;
      tr.params.radius_m = t.radius_m;
      tr.params.length_m = t.length_m;
      // Aft interior face center = cylinder center - (L/2) xhat (axis +X).
      tr.params.aft_center_m =
          arr3(t.position_m) - Eigen::Vector3d(0.5 * t.length_m, 0.0, 0.0);
      tr.params.density_kgpm3 = t.density_kgpm3;
      tr.params.initial_mass_kg = t.propellant_mass_kg;
      tr.mass_kg = t.propellant_mass_kg;
      tr.stage_idx = stage_idx;
      st.tanks.push_back(static_cast<int>(rt.tanks.size()));
      rt.tanks.push_back(tr);
    }
    for (const EngineCfg& e : s.engines) {
      EngineRt er;
      er.name = e.name;
      er.stage_idx = stage_idx;
      er.params.thrust_vac_N = e.thrust_vac_N;
      er.params.isp_vac_s = e.isp_vac_s;
      er.params.exit_area_m2 = e.exit_area_m2;
      er.params.throttle_min = e.throttle_min;
      er.params.throttle_max = e.throttle_max;
      er.params.spool_up_s = e.spool_time_s;
      er.params.spool_down_s = e.spool_time_s;
      er.params.max_ignitions = e.ignitions;
      er.params.gimbal_limit_rad = deg2rad_v(e.gimbal_max_deg);
      er.params.gimbal_rate_radps = deg2rad_v(e.gimbal_rate_dps);
      er.params.position_m = arr3(e.position_m);
      er.params.axis = arr3(e.axis).normalized();
      gimbal_basis(er.params.axis, er.params.gimbal_axis_1,
                   er.params.gimbal_axis_2);
      er.tank_idx = (e.feeds_tank_index >= 0 &&
                     e.feeds_tank_index < static_cast<int>(st.tanks.size()))
                        ? st.tanks[static_cast<std::size_t>(e.feeds_tank_index)]
                        : -1;
      st.engines.push_back(static_cast<int>(rt.engines.size()));
      rt.engines.push_back(er);
    }
    for (const WheelCfg& w : s.wheels) {
      WheelRt wr;
      wr.stage_idx = stage_idx;
      wr.params.axis = arr3(w.axis).normalized();
      wr.params.torque_max_Nm = w.max_torque_Nm;
      wr.params.momentum_max_Nms = w.max_momentum_Nms;
      rt.wheels.push_back(wr);
    }
    for (const JettisonCfg& j : s.jettison) {
      JettRt jr;
      jr.name = j.name;
      jr.stage_idx = stage_idx;
      jr.body.mass_kg = j.mass_kg;
      jr.body.cg_m = arr3(j.cg_m);
      jr.body.inertia_kgm2 = mat3_flat(j.inertia_kgm2);
      st.jett.push_back(static_cast<int>(rt.jett.size()));
      rt.jett.push_back(jr);
    }
    rt.stages.push_back(st);
    ++stage_idx;
  }
  for (const AeroCfg& a : v.aero) {
    models::AeroTables t;
    t.ref_area_m2 = a.ref_area_m2;
    t.ref_diameter_m = a.ref_diameter_m;
    t.cmq_per_rad = a.cmq_per_rad;
    t.mach = a.mach;
    t.ca = a.ca;
    t.cnalpha_per_rad = a.cnalpha_per_rad;
    t.xcp_m = a.xcp_m;
    rt.aero.push_back(t);
  }
  return rt;
}

struct StackProps {
  models::BodyProps composite;
  models::BodyRates rates;
};

// Compose the currently attached stack in the FIXED FR-10 order (fixed bodies
// -- dry stages then attached jettison items -- then one settled slug per
// attached tank), so the composite wet mass is a stable same-order binary64
// sum every cycle (EC-2 bit-exactness).
StackProps compose_stack(const VehicleRuntime& v) {
  std::vector<models::BodyProps> bodies;
  std::vector<models::BodyRates> rates;
  for (const StageRt& s : v.stages) {
    if (!s.attached) continue;
    bodies.push_back(s.dry);
    rates.emplace_back();
    for (int j : s.jett) {
      if (v.jett[static_cast<std::size_t>(j)].attached) {
        bodies.push_back(v.jett[static_cast<std::size_t>(j)].body);
        rates.emplace_back();
      }
    }
  }
  for (const StageRt& s : v.stages) {
    if (!s.attached) continue;
    for (int ti : s.tanks) {
      const TankRt& tk = v.tanks[static_cast<std::size_t>(ti)];
      double consume = 0.0;  // positive propellant consumption [kg/s]
      if (tk.mass_kg > 0.0) {
        for (int ei : s.engines) {
          const EngineRt& en = v.engines[static_cast<std::size_t>(ei)];
          if (en.tank_idx == ti && en.state.throttle_level > 0.0) {
            consume +=
                models::engine_mdot_kgps(en.params, en.state.throttle_level);
          }
        }
      }
      bodies.push_back(models::tank_slug_props(tk.params, tk.mass_kg));
      // massprops uses SIGNED dm/dt; a draining tank is negative.
      rates.push_back(models::tank_slug_rates(tk.params, tk.mass_kg, -consume));
    }
  }
  StackProps sp;
  sp.composite = models::compose(bodies);
  sp.rates = models::compose_rates(bodies, rates);
  return sp;
}

// ZOH context captured for one control cycle (D-5): everything the frozen RHS
// and the log-instant force breakdown read.
struct CycleCtx {
  const VehicleRuntime* v = nullptr;
  bool earth = true;
  Eigen::Matrix3d c_i2b = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d c_b2i = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d c_gcrf_to_itrf = Eigen::Matrix3d::Identity();
  Eigen::Vector3d omega_planet_i = Eigen::Vector3d::Zero();
  Eigen::Vector3d omega_b = Eigen::Vector3d::Zero();
  Eigen::Vector3d cg_b = Eigen::Vector3d::Zero();
  double mass_kg = 1.0;
  int aero_idx = -1;
  double planet_a_m = 0.0;
  double planet_inv_f = 0.0;
};

struct BodyLoads {
  Eigen::Vector3d f_thrust = Eigen::Vector3d::Zero();
  Eigen::Vector3d tq_thrust = Eigen::Vector3d::Zero();
  Eigen::Vector3d f_aero = Eigen::Vector3d::Zero();
  Eigen::Vector3d tq_aero = Eigen::Vector3d::Zero();
  double alt_m = 0.0;
  double mach = 0.0;
  double q_bar_pa = 0.0;
  double rho = 0.0;
};

// Body-frame thrust and aero force/torque about the composite CG, plus the
// air-data diagnostics (FR-8/FR-9). Air-relative velocity is v - omega x r
// (eq:drag:vrel); Mach and dynamic pressure come straight from the aero model
// (its exact q_bar == 0 at |v_rel| == 0 is what makes the on-pad q_bar exactly
// zero, EC-10).
BodyLoads eval_body_loads(const CycleCtx& c, const Eigen::Vector3d& r_i,
                          const Eigen::Vector3d& v_i) {
  BodyLoads out;
  double p_amb = 0.0;
  double sos = 0.0;
  if (c.earth) {
    const Eigen::Vector3d r_ecef = c.c_gcrf_to_itrf * r_i;
    out.alt_m = models::geodetic_altitude(r_ecef, c.planet_a_m, c.planet_inv_f);
    if (out.alt_m >= -5000.0 && out.alt_m < 86000.0) {
      const models::Ussa76State s = models::ussa76_state(out.alt_m);
      out.rho = s.density_kgpm3;
      sos = s.speed_of_sound_mps;
      p_amb = s.pressure_Pa;
    }
  } else {
    out.alt_m = r_i.norm() - c.planet_a_m;
  }
  for (const EngineRt& e : c.v->engines) {
    if (!c.v->stages[static_cast<std::size_t>(e.stage_idx)].attached) continue;
    if (e.tank_idx >= 0 &&
        c.v->tanks[static_cast<std::size_t>(e.tank_idx)].mass_kg <= 0.0) {
      continue;
    }
    const models::EngineForceTorque ft =
        models::engine_force_torque(e.params, e.state, p_amb, c.cg_b);
    out.f_thrust += ft.force_N;
    out.tq_thrust += ft.torque_Nm;
  }
  if (c.aero_idx >= 0 && out.rho > 0.0 && sos > 0.0) {
    const Eigen::Vector3d v_rel_i = v_i - c.omega_planet_i.cross(r_i);
    const Eigen::Vector3d v_rel_b = c.c_i2b * v_rel_i;
    const models::AeroForceTorque a = models::aero_force_torque(
        c.v->aero[static_cast<std::size_t>(c.aero_idx)], v_rel_b, out.rho, sos,
        c.cg_b.x(), c.omega_b);
    out.f_aero = a.force_N;
    out.tq_aero = a.torque_Nm;
    out.mach = a.mach;
    out.q_bar_pa = a.q_bar_Pa;
  }
  return out;
}

double osc_perigee_alt(double mu, const Eigen::Vector3d& r,
                       const Eigen::Vector3d& v, double radius_m) {
  const double rn = r.norm();
  const double energy = 0.5 * v.squaredNorm() - mu / rn;
  const double a = -mu / (2.0 * energy);
  const Eigen::Vector3d h = r.cross(v);
  const Eigen::Vector3d e_vec = v.cross(h) / mu - r / rn;
  return a * (1.0 - e_vec.norm()) - radius_m;
}

// TDB seconds since J2000 TDB for the epoch shifted by t_s (mirrors
// EnvironmentModel::tdb_s_at; used only for the SOI-transition geometry).
double tdb_s_at(const time::TaiEpoch& epoch, double t_s) {
  const time::TaiEpoch tai = time::tai_add_seconds(epoch, t_s);
  const time::TwoPartJd jd = time::tdb_jd(tai);
  return ((jd.jd1 - 2451545.0) + jd.jd2) * 86400.0;
}

void check_config_vehicle(const RunConfig& cfg) {
  if (cfg.integrator != "rk4") {
    throw std::invalid_argument(
        "run_vehicle: integrator must be \"rk4\" in Phase 4");
  }
  if (!(cfg.duration_s > 0.0)) {
    throw std::invalid_argument("run_vehicle: duration_s must be > 0");
  }
  if (!(cfg.dt_s > 0.0)) {
    throw std::invalid_argument("run_vehicle: dt_s must be > 0");
  }
  if (cfg.truth_rate_hz < 1) {
    throw std::invalid_argument("run_vehicle: truth_rate_hz must be >= 1");
  }
  if (cfg.vehicle.stages.empty()) {
    throw std::invalid_argument(
        "run_vehicle: the vehicle must define at least one stage");
  }
  const double steps_exact = cfg.duration_s / cfg.dt_s;
  if (std::fabs(steps_exact - std::llround(steps_exact)) > 1e-9 * steps_exact) {
    throw std::invalid_argument(
        "run_vehicle: duration_s must be an integer multiple of dt_s");
  }
  const double decim_exact =
      1.0 / (cfg.dt_s * static_cast<double>(cfg.truth_rate_hz));
  if (decim_exact < 0.5 ||
      std::fabs(decim_exact - std::llround(decim_exact)) > 1e-9 * decim_exact) {
    throw std::invalid_argument(
        "run_vehicle: 1/(dt_s*truth_rate_hz) must be a positive integer");
  }
}

}  // namespace

RunSummary run_vehicle(const RunConfig& cfg, const std::string& out_path) {
  check_config_vehicle(cfg);

  // --- environment (reused verbatim from the run_env surface) --------------
  models::EnvironmentSpec spec;
  spec.central_body = central_body_from_name(cfg.central_body);
  spec.epoch_tai = {cfg.epoch_tai_day, cfg.epoch_tai_sec};
  spec.gravity_model = cfg.gravity_model;
  spec.gravity_field_path = cfg.gravity_field_path;
  spec.gravity_degree = cfg.gravity_degree;
  spec.gravity_order = cfg.gravity_order;
  spec.third_bodies = cfg.third_bodies;
  spec.srp_enabled = cfg.srp_enabled;
  spec.cr_a_over_m_m2pkg = cfg.cr_a_over_m_m2pkg;
  spec.srp_occulters = cfg.srp_occulters;
  spec.atmosphere = cfg.drag_enabled ? atmosphere_from_name(cfg.atmosphere)
                                     : models::AtmosphereModel::kNone;
  spec.cd_a_over_m_m2pkg = cfg.cd_a_over_m_m2pkg;
  spec.hp_exponent_n = cfg.hp_exponent_n;
  spec.ephemeris_path = cfg.ephemeris_path;
  models::EnvironmentModel env(spec);

  const models::CentralBody central = spec.central_body;
  const double mu = models::central_body_gm(central);
  const bool earth = central == models::CentralBody::kEarth;
  double planet_a_m = constants::WGS84_A_M;
  double planet_inv_f = constants::WGS84_INV_F;
  if (central == models::CentralBody::kMars) {
    planet_a_m = constants::MARS_ELLIPSOID_A_M;
    planet_inv_f = constants::MARS_ELLIPSOID_INV_F;
  } else if (central == models::CentralBody::kMoon) {
    planet_a_m = constants::R_MOON_M;
    planet_inv_f = 1.0e12;  // Moon treated spherical for altitude reference
  }

  const time::TaiEpoch epoch{cfg.epoch_tai_day, cfg.epoch_tai_sec};
  VehicleRuntime veh = build_vehicle(cfg.vehicle);

  // A Moon ephemeris is loaded only when a soi_transition to the Moon is in the
  // sequence (the geometry needs the Moon's geocentric position).
  bool need_soi_moon = false;
  for (const SequenceEntry& e : cfg.sequence) {
    if (e.trigger == "condition" && e.condition == "soi_transition" &&
        e.body == "moon") {
      need_soi_moon = true;
    }
  }
  std::optional<Ephemeris> soi_eph;
  if (need_soi_moon) {
    soi_eph.emplace(Ephemeris::load_file(cfg.ephemeris_path));
  }

  // --- SRLOG header: enabled force sources in canonical order --------------
  // The forces group is declared only when its rate is nonzero (the writer
  // rejects sources without a rate); a config that disables the group by
  // leaving forces_rate_hz at 0 produces a valid log without it.
  std::vector<std::string> sources;
  if (cfg.forces_rate_hz != 0) {
    sources.push_back("gravity");
    if (!cfg.third_bodies.empty()) sources.push_back("thirdbody");
    if (!veh.aero.empty()) sources.push_back("aero");
    sources.push_back("thrust");
    sources.push_back("gravgrad");
  }

  log::SrlogHeaderFields fields;
  fields.core_version = core_version();
  fields.git_hash = git_hash();
  fields.config_sha256 = cfg.config_sha256;
  fields.master_seed = cfg.master_seed;
  fields.oracle = cfg.oracle;
  fields.epoch_utc = cfg.epoch_utc;
  fields.central_body = cfg.central_body;
  fields.truth_rate_hz = cfg.truth_rate_hz;
  fields.force_sources = sources;
  fields.forces_rate_hz = cfg.forces_rate_hz;
  fields.mass_rate_hz = cfg.mass_rate_hz;
  fields.env_rate_hz = cfg.env_rate_hz;
  log::SrlogWriter writer(out_path, fields);

  // --- initial state, attitude, and control mode ---------------------------
  enum class AttMode {
    kPadFixed,
    kPitchProgram,
    kInertialHold,
    kRateCommand,
    kProgradeHold
  };
  double y[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  Eigen::Map<Eigen::Vector3d> r_m(y);
  Eigen::Map<Eigen::Vector3d> v_mps(y + 3);
  Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
  Eigen::Vector3d omega_b = Eigen::Vector3d::Zero();
  Eigen::Quaterniond q_hold = Eigen::Quaterniond::Identity();
  Eigen::Vector3d rate_omega_b = Eigen::Vector3d::Zero();
  bool rate_frame_gcrf = true;
  bool released = true;
  AttMode mode = AttMode::kInertialHold;
  double pad_lat = 0.0;
  double pad_lon = 0.0;
  double pad_alt = 0.0;
  Eigen::Vector3d up0 = Eigen::Vector3d::UnitZ();
  Eigen::Vector3d east0 = Eigen::Vector3d::UnitX();
  Eigen::Vector3d north0 = Eigen::Vector3d::UnitY();
  double pitch_az_deg = 0.0;
  std::vector<double> pitch_t;
  std::vector<double> pitch_d;

  if (cfg.initial_form == "geodetic") {
    if (!earth) {
      throw std::invalid_argument(
          "run_vehicle: geodetic launch requires central_body earth");
    }
    pad_lat = deg2rad_v(cfg.launch_lat_deg);
    pad_lon = deg2rad_v(cfg.launch_lon_deg);
    pad_alt = cfg.launch_alt_m;
    const Eigen::Matrix3d c_itrf0 = frames::c_gcrf_to_itrf(epoch, 0.0);
    const models::PadState pad = models::geodetic_pad_state(
        pad_lat, pad_lon, pad_alt, c_itrf0, constants::OMEGA_EARTH_RAD_PER_S,
        constants::WGS84_A_M, constants::WGS84_INV_F);
    up0 = pad.up_i;
    east0 = pad.east_i;
    north0 = pad.north_i;
    r_m = pad.r_i_m;
    v_mps = pad.v_i_mps;
    q = models::attitude_from_body_x(up0, north0);
    q_hold = q;
    released = false;
    mode = AttMode::kPadFixed;
  } else {
    r_m = arr3(cfg.r0_m);
    v_mps = arr3(cfg.v0_mps);
    const Eigen::Vector3d vhat =
        v_mps.norm() > 0.0 ? v_mps.normalized() : Eigen::Vector3d::UnitX();
    q = models::attitude_from_body_x(vhat, r_m);  // body +X prograde
    q_hold = q;
    mode = AttMode::kInertialHold;
  }

  const std::int64_t steps = std::llround(cfg.duration_s / cfg.dt_s);
  const double dt = cfg.dt_s;
  const std::int64_t truth_decim = std::llround(
      1.0 / (dt * static_cast<double>(cfg.truth_rate_hz)));
  const std::int64_t forces_decim =
      cfg.forces_rate_hz ? std::llround(1.0 / (dt * cfg.forces_rate_hz)) : 0;
  const std::int64_t mass_decim =
      cfg.mass_rate_hz ? std::llround(1.0 / (dt * cfg.mass_rate_hz)) : 0;
  const std::int64_t env_decim =
      cfg.env_rate_hz ? std::llround(1.0 / (dt * cfg.env_rate_hz)) : 0;

  RunSummary summary;
  writer.write_event(0.0, 1, "run_start");
  summary.event_records += 1;

  std::map<std::string, double> fire_time;
  std::vector<char> fired(cfg.sequence.size(), 0);
  integrate::Rk4 rk4(6);

  CycleCtx ctx;
  ctx.v = &veh;
  ctx.earth = earth;
  ctx.planet_a_m = planet_a_m;
  ctx.planet_inv_f = planet_inv_f;

  auto rhs = [&](double t, const double* yin, double* ydot) {
    const Eigen::Map<const Eigen::Vector3d> r(yin);
    const Eigen::Map<const Eigen::Vector3d> vv(yin + 3);
    const Eigen::Vector3d a_env = env.acceleration(t, r, vv);
    const BodyLoads bl = eval_body_loads(ctx, r, vv);
    const Eigen::Vector3d f_body = bl.f_thrust + bl.f_aero;
    const Eigen::Vector3d a = models::composed_translational_accel(
        a_env, f_body, ctx.c_b2i, ctx.mass_kg);
    ydot[0] = vv[0];
    ydot[1] = vv[1];
    ydot[2] = vv[2];
    ydot[3] = a[0];
    ydot[4] = a[1];
    ydot[5] = a[2];
  };
  const integrate::RhsRef f(rhs);

  bool stop = false;
  int aero_sep_count = 0;

  // One reusable buffer for the whole run: the per-source forces record is
  // rebuilt in place each logged step (clear() retains the capacity) rather
  // than allocating and freeing a fresh vector every cycle. On a high-volume
  // log (order 10^5 records) removing this per-cycle allocation churn is what
  // makes the by-source write path reliable on the build host; the logged
  // bytes are identical either way (see docs/KNOWN_ISSUES.md, KNOWN-ISSUE-P4-1).
  std::vector<log::ForceSourceSample> samples;
  samples.reserve(sources.size());

  for (std::int64_t i = 0; i <= steps && !stop; ++i) {
    const double t = static_cast<double>(i) * dt;
    const time::TaiEpoch tai = time::tai_add_seconds(epoch, t);
    Eigen::Matrix3d c_itrf = Eigen::Matrix3d::Identity();
    if (earth) c_itrf = frames::c_gcrf_to_itrf(tai, 0.0);

    // -- sequence: fire due entries, in file order -------------------------
    for (std::size_t k = 0; k < cfg.sequence.size(); ++k) {
      if (fired[k]) continue;
      const SequenceEntry& e = cfg.sequence[k];
      bool go = false;
      if (e.trigger == "elapsed") {
        go = t >= e.t_s;
      } else if (e.trigger == "after_event") {
        auto it = fire_time.find(e.event);
        go = it != fire_time.end() && t >= it->second + e.offset_s;
      } else {  // condition
        if (e.condition == "altitude_above" || e.condition == "altitude_below") {
          double alt = r_m.norm() - planet_a_m;
          if (earth) alt = models::geodetic_altitude(c_itrf * r_m,
                                                     planet_a_m, planet_inv_f);
          go = (e.condition == "altitude_above") ? alt >= e.altitude_m
                                                 : alt <= e.altitude_m;
        } else if (e.condition == "perigee_above") {
          go = released && osc_perigee_alt(mu, r_m, v_mps, planet_a_m) >=
                               e.perigee_alt_m;
        } else if (e.condition == "soi_transition" && soi_eph.has_value()) {
          const double tdb = tdb_s_at(epoch, t);
          const Eigen::Vector3d r_moon = soi_eph->moon_geocentric(tdb).r_m;
          const double soi = r_moon.norm() *
                             std::pow(constants::GM_MOON_DE440_M3_PER_S2 /
                                          constants::GM_EARTH_M3_PER_S2,
                                      0.4);
          go = (r_m - r_moon).norm() < soi;
        }
      }
      if (!go) continue;
      fired[k] = 1;
      fire_time[e.name] = t;

      // -- apply the action ------------------------------------------------
      if (e.action == "pad_release") {
        released = true;
        q_hold = q;
        mode = AttMode::kInertialHold;  // held until a pitch/rate action fires
      } else if (e.action == "ignite_engine" || e.action == "cutoff_engine") {
        for (EngineRt& en : veh.engines) {
          if (veh.stages[static_cast<std::size_t>(en.stage_idx)].name ==
                  e.stage &&
              en.name == e.engine) {
            en.command.run = (e.action == "ignite_engine");
            if (en.command.run) en.command.throttle = en.params.throttle_max;
          }
        }
      } else if (e.action == "separate_stage" || e.action == "jettison") {
        const Eigen::Vector3d cg_old = compose_stack(veh).composite.cg_m;
        if (e.action == "separate_stage") {
          for (StageRt& s : veh.stages) {
            if (s.name == e.stage) {
              s.attached = false;
              for (int j : s.jett) {
                veh.jett[static_cast<std::size_t>(j)].attached = false;
              }
              for (int ei : s.engines) {
                veh.engines[static_cast<std::size_t>(ei)].command.run = false;
              }
            }
          }
          ++aero_sep_count;
        } else {
          for (JettRt& jt : veh.jett) {
            if (jt.name == e.item &&
                veh.stages[static_cast<std::size_t>(jt.stage_idx)].name ==
                    e.stage) {
              jt.attached = false;
            }
          }
        }
        const Eigen::Vector3d cg_new = compose_stack(veh).composite.cg_m;
        const models::SeparationRemap rm = models::separation_remap(
            cg_old, cg_new, r_m, v_mps, q, omega_b);
        r_m = rm.r_new_i_m;
        v_mps = rm.v_new_i_mps;
      } else if (e.action == "pitch_program") {
        mode = AttMode::kPitchProgram;
        pitch_az_deg = e.azimuth_deg;
        pitch_t = e.pitch_t_s;
        pitch_d = e.pitch_deg;
      } else if (e.action == "attitude_hold") {
        mode = AttMode::kInertialHold;
        q_hold = q;
      } else if (e.action == "prograde_hold") {
        mode = AttMode::kProgradeHold;
      } else if (e.action == "rate_command") {
        mode = AttMode::kRateCommand;
        rate_frame_gcrf = e.frame == "gcrf";
        rate_omega_b = arr3(e.omega_dps) * (constants::TWO_PI / 360.0);
        q_hold = q;
      } else if (e.action == "terminate") {
        stop = true;
      }
      // Event detail names what fired: for a condition trigger the condition
      // leads (so e.g. the orbit-insertion and SOI-transition terminal events
      // are self-describing in the log), otherwise the action leads.
      const std::string detail = (e.trigger == "condition")
                                     ? (e.condition + ":" + e.name)
                                     : (e.action + ":" + e.name);
      writer.write_event(t, 3, detail);
      summary.event_records += 1;
    }

    // -- mass properties for this cycle ------------------------------------
    const StackProps sp = compose_stack(veh);

    // -- attitude command for this cycle -----------------------------------
    Eigen::Vector3d omega_i_planet = Eigen::Vector3d::Zero();
    if (earth) {
      omega_i_planet =
          constants::OMEGA_EARTH_RAD_PER_S * c_itrf.row(2).transpose();
    }
    if (mode == AttMode::kPadFixed) {
      const models::PadState pad = models::geodetic_pad_state(
          pad_lat, pad_lon, pad_alt, c_itrf, constants::OMEGA_EARTH_RAD_PER_S,
          constants::WGS84_A_M, constants::WGS84_INV_F);
      if (!released) {
        r_m = pad.r_i_m;  // clamped to the co-rotating pad until release
        v_mps = pad.v_i_mps;
      }
      q = models::attitude_from_body_x(pad.up_i, pad.north_i);
      // The pad co-rotates with the planet: the logged body rate is the
      // planet spin resolved in body axes.
      omega_b = rotation::dcm_from_quat(q) * omega_i_planet;
    } else if (mode == AttMode::kPitchProgram) {
      auto interp = [](const std::vector<double>& xs,
                       const std::vector<double>& ys, double x) {
        if (x <= xs.front()) return ys.front();
        if (x >= xs.back()) return ys.back();
        for (std::size_t j = 0; j + 1 < xs.size(); ++j) {
          if (x <= xs[j + 1]) {
            const double w = (x - xs[j]) / (xs[j + 1] - xs[j]);
            return ys[j] + w * (ys[j + 1] - ys[j]);
          }
        }
        return ys.back();
      };
      const double az = deg2rad_v(pitch_az_deg);
      const double p0 = deg2rad_v(interp(pitch_t, pitch_d, t));
      const double p1 = deg2rad_v(interp(pitch_t, pitch_d, t + dt));
      const Eigen::Quaterniond q0 = models::attitude_from_body_x(
          models::pitch_program_axis(az, p0, up0, east0, north0), up0);
      const Eigen::Quaterniond q1 = models::attitude_from_body_x(
          models::pitch_program_axis(az, p1, up0, east0, north0), up0);
      q = q0;
      omega_b = models::omega_from_quaternions(q0, q1, dt);
    } else if (mode == AttMode::kRateCommand) {
      q = q_hold;
      omega_b = rate_frame_gcrf ? rotation::dcm_from_quat(q) * rate_omega_b
                                : rate_omega_b;
    } else if (mode == AttMode::kProgradeHold) {
      // Velocity-pointing open-loop steering: body +X tracks the current
      // inertial velocity each control cycle, so a finite burn stays prograde
      // (the fixed-attitude alternative loses cos(theta) of the burn as the
      // velocity rotates ~orbital-rate through a long burn).
      const Eigen::Vector3d vh =
          v_mps.norm() > 0.0 ? v_mps.normalized() : Eigen::Vector3d::UnitX();
      q = models::attitude_from_body_x(vh, r_m);
      omega_b = Eigen::Vector3d::Zero();
    } else {  // inertial hold
      q = q_hold;
      omega_b = Eigen::Vector3d::Zero();
    }

    ctx.c_i2b = rotation::dcm_from_quat(q);
    ctx.c_b2i = ctx.c_i2b.transpose();
    ctx.c_gcrf_to_itrf = c_itrf;
    ctx.omega_planet_i = omega_i_planet;
    ctx.omega_b = omega_b;
    ctx.cg_b = sp.composite.cg_m;
    ctx.mass_kg = sp.composite.mass_kg;
    ctx.aero_idx =
        veh.aero.empty()
            ? -1
            : std::min<int>(aero_sep_count,
                            static_cast<int>(veh.aero.size()) - 1);

    const bool last = (i == steps) || stop;

    // -- logging (decimated from the truth grid) ---------------------------
    if (i % truth_decim == 0 || last) {
      const double q_arr[4] = {q.w(), q.x(), q.y(), q.z()};
      writer.write_truth(t, r_m, v_mps, q_arr, omega_b, sp.composite.mass_kg);
      summary.truth_records += 1;
    }
    const BodyLoads bl = eval_body_loads(ctx, r_m, v_mps);
    if (forces_decim && (i % forces_decim == 0 || last)) {
      const Eigen::Vector3d a_env = env.acceleration(t, r_m, v_mps);
      const Eigen::Vector3d a_grav = models::twobody_accel(mu, r_m);
      const Eigen::Vector3d f_grav_b =
          ctx.c_i2b * (sp.composite.mass_kg * a_grav);
      const Eigen::Vector3d f_pert_b =
          ctx.c_i2b * (sp.composite.mass_kg * (a_env - a_grav));
      const Eigen::Vector3d tq_gg = models::gravgrad_torque(
          mu, r_m, q, sp.composite.inertia_kgm2);
      samples.clear();
      for (const std::string& src : sources) {
        // Eigen vectors are not zero-initialized by their default constructor;
        // zeroing both channels keeps a source that produces only a force or
        // only a torque deterministic (D-10) rather than logging stack garbage.
        log::ForceSourceSample s;
        s.force_b_n = Eigen::Vector3d::Zero();
        s.torque_b_nm = Eigen::Vector3d::Zero();
        if (src == "gravity") {
          s.force_b_n = f_grav_b;
        } else if (src == "thirdbody") {
          s.force_b_n = f_pert_b;
        } else if (src == "aero") {
          s.force_b_n = bl.f_aero;
          s.torque_b_nm = bl.tq_aero;
        } else if (src == "thrust") {
          s.force_b_n = bl.f_thrust;
          s.torque_b_nm = bl.tq_thrust;
        } else if (src == "gravgrad") {
          s.torque_b_nm = tq_gg;
        }
        samples.push_back(s);
      }
      writer.write_forces(t, samples);
    }
    if (mass_decim && (i % mass_decim == 0 || last)) {
      const Eigen::Matrix3d& I = sp.composite.inertia_kgm2;
      const double packed[6] = {I(0, 0), I(0, 1), I(0, 2),
                                I(1, 1), I(1, 2), I(2, 2)};
      writer.write_mass(t, sp.composite.mass_kg, sp.composite.cg_m, packed);
    }
    if (env_decim && (i % env_decim == 0 || last)) {
      const double rn = r_m.norm();
      const double vn = v_mps.norm();
      double fpa = 0.0;
      if (rn > 0.0 && vn > 0.0) {
        double s = r_m.dot(v_mps) / (rn * vn);
        s = std::max(-1.0, std::min(1.0, s));
        fpa = std::asin(s);
      }
      writer.write_env(t, bl.alt_m, bl.mach, bl.q_bar_pa, bl.rho, fpa);
    }

    if (stop) {
      writer.write_event(t, 2, "run_end");
      summary.event_records += 1;
      break;
    }
    if (i == steps) break;

    // -- advance the continuous and discrete state over the cycle ----------
    if (released) {
      rk4.step(f, t, y, dt, y);
      summary.steps += 1;
    }
    // Deplete tanks at the ZOH throttle level held over the cycle, then
    // advance each engine's spool/gimbal/ignition state for the next cycle.
    for (EngineRt& en : veh.engines) {
      if (!veh.stages[static_cast<std::size_t>(en.stage_idx)].attached) {
        continue;
      }
      if (en.tank_idx >= 0 && en.state.throttle_level > 0.0) {
        TankRt& tk = veh.tanks[static_cast<std::size_t>(en.tank_idx)];
        if (tk.mass_kg > 0.0) {
          const double dm =
              models::engine_mdot_kgps(en.params, en.state.throttle_level) * dt;
          tk.mass_kg = std::max(0.0, tk.mass_kg - dm);
        }
      }
    }
    for (EngineRt& en : veh.engines) {
      en.state = models::engine_advance(en.params, en.command, en.state, dt);
    }
    if (mode == AttMode::kRateCommand) {
      // Integrate the commanded body rate to carry q into the next cycle:
      // q_{k+1} = q_k (x) dq, dq the body-frame rotation by omega_b dt
      // (frame-transformation composition, ch:rotations).
      const Eigen::Vector3d w_b =
          rate_frame_gcrf ? (rotation::dcm_from_quat(q_hold) * rate_omega_b)
                          : rate_omega_b;
      const double ang = w_b.norm() * dt;
      Eigen::Quaterniond dq = Eigen::Quaterniond::Identity();
      if (ang > 0.0) {
        const Eigen::Vector3d ax = w_b.normalized();
        const double s = std::sin(0.5 * ang);
        dq = Eigen::Quaterniond(std::cos(0.5 * ang), s * ax.x(), s * ax.y(),
                                s * ax.z());
      }
      q_hold = rotation::quat_normalize(rotation::quat_multiply(q_hold, dq));
    }
  }

  if (!stop) {
    writer.write_event(cfg.duration_s, 2, "run_end");
    summary.event_records += 1;
  }
  writer.close();

  for (int i = 0; i < 3; ++i) {
    summary.final_r_m[static_cast<std::size_t>(i)] = r_m[i];
    summary.final_v_mps[static_cast<std::size_t>(i)] = v_mps[i];
  }
  return summary;
}

}  // namespace star
