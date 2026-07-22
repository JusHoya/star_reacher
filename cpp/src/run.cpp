// Batch runs: config in, SRLOG out. Model math lives in chapter-tracked
// modules under cpp/src/models/; this file owns only the orchestration -
// stepping, decimation/sampling, and logging - for the Phase 1 two-body and
// Phase 3 composed-environment point-mass paths. The Phase 4/6 vehicle path
// lives in the VehicleCycle stepping core (cpp/src/vehicle_cycle.cpp);
// run_vehicle here is a thin batch driver over it.
#include "star/run.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "star/events.hpp"
#include "star/integrate.hpp"
#include "star/models/environment.hpp"
#include "star/models/twobody.hpp"
#include "star/srlog_writer.hpp"
#include "star/vehicle_cycle.hpp"
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
  if (body == "sun") {
    return models::central_body_gm(models::CentralBody::kSun);
  }
  throw std::invalid_argument(
      "unknown central body: \"" + body +
      "\" (supported: \"earth\", \"moon\", \"mars\", \"sun\")");
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
  if (name == "sun") return models::CentralBody::kSun;
  throw std::invalid_argument(
      "unknown central body: \"" + name +
      "\" (supported: \"earth\", \"moon\", \"mars\", \"sun\")");
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


// Propagate the full 6DOF vehicle case (Phase 4; Phase 6 GNC): a batch run
// is literally a loop over the VehicleCycle stepping core - the same
// factoring the Sim stepping API drives one control period at a time - so
// batch and stepped executions of one scenario produce byte-identical logs
// by construction (Phase 6 exit criterion 4).
RunSummary run_vehicle(const RunConfig& cfg, const std::string& out_path) {
  VehicleCycle vc(cfg, out_path);
  while (vc.step()) {
  }
  return vc.summary();
}

}  // namespace star
