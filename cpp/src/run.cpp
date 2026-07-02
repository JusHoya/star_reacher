// Two-body batch run: config in, SRLOG out. Model math lives in
// cpp/src/models/twobody.cpp (chapter-tracked); this file owns only the
// orchestration - stepping, decimation, and logging.
#include "star/run.hpp"

#include <cmath>
#include <stdexcept>

#include <Eigen/Dense>

#include "star/constants.hpp"
#include "star/models/twobody.hpp"
#include "star/srlog_writer.hpp"
#include "star/version.hpp"

namespace star {

double gm(const std::string& body) {
  if (body == "earth") {
    return constants::GM_EARTH_M3_PER_S2;
  }
  throw std::invalid_argument("unknown central body: \"" + body +
                              "\" (Phase 1 supports \"earth\" only)");
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

  models::TwoBodyState state;
  state.r_m = Eigen::Vector3d(cfg.r0_m[0], cfg.r0_m[1], cfg.r0_m[2]);
  state.v_mps = Eigen::Vector3d(cfg.v0_mps[0], cfg.v0_mps[1], cfg.v0_mps[2]);

  RunSummary summary;
  summary.steps = steps;

  writer.write_event(0.0, 1, "run_start");
  summary.event_records += 1;

  writer.write_truth(0.0, state.r_m, state.v_mps, q_identity, w_zero,
                     cfg.mass_kg);
  summary.truth_records += 1;

  for (std::int64_t i = 1; i <= steps; ++i) {
    state = models::rk4_step(mu, state, cfg.dt_s);
    if (i % decim == 0) {
      // t = i*dt as a single multiply (not accumulated addition): one rounding
      // per timestamp keeps logged times well-conditioned and reproducible.
      writer.write_truth(static_cast<double>(i) * cfg.dt_s, state.r_m,
                         state.v_mps, q_identity, w_zero, cfg.mass_kg);
      summary.truth_records += 1;
    }
  }

  // run_end carries the configured duration verbatim, not steps*dt, so the
  // event timestamp is exactly the user's requested span.
  writer.write_event(cfg.duration_s, 2, "run_end");
  summary.event_records += 1;
  writer.close();

  for (int i = 0; i < 3; ++i) {
    summary.final_r_m[static_cast<std::size_t>(i)] = state.r_m[i];
    summary.final_v_mps[static_cast<std::size_t>(i)] = state.v_mps[i];
  }
  return summary;
}

}  // namespace star
