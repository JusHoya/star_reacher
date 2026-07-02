// Batch run entry point for the Phase 1 two-body placeholder mission
// (PRD section 8, Phase 1). The Python frontend validates and canonicalizes
// the mission file (D-2), fills a RunConfig, and calls run_twobody(); the
// core propagates and writes the SRLOG without ever parsing text, touching
// the network, or reading the clock (architecture boundary rule, PRD 7).
#ifndef STAR_RUN_HPP
#define STAR_RUN_HPP

#include <array>
#include <cstdint>
#include <string>

namespace star {

// Mirror of star_reacher._core.RunConfig (Phase 1 contract section 3). All
// user-facing validation happens in Python before this struct is populated;
// the core re-checks only what it needs to stay well-defined and throws
// std::invalid_argument on violations (defensive guard, not user UX).
struct RunConfig {
  std::string epoch_utc;            // ISO-8601, carried verbatim into the log header
  double duration_s = 0.0;          // > 0; integer multiple of dt_s
  double dt_s = 0.0;                // > 0; fixed RK4 step
  std::string integrator = "rk4";   // "rk4" only in Phase 1
  std::string central_body = "earth";  // "earth" only in Phase 1
  std::array<double, 3> r0_m{{0.0, 0.0, 0.0}};     // GCRF position [m]
  std::array<double, 3> v0_mps{{0.0, 0.0, 0.0}};   // GCRF velocity [m/s]
  double mass_kg = 1.0;             // > 0; constant in Phase 1
  std::uint64_t master_seed = 0;    // D-9 master seed, echoed in the header
  std::uint32_t truth_rate_hz = 10; // >= 1; 1/(dt_s*rate) must be an integer
  std::string config_sha256;        // 64-hex resolved-config digest (FR-15)
  bool oracle = false;              // D-11 oracle flag, stamped in the header
};

// Run summary returned across the binding: enough for the CLI to print a
// final-state line and for tests to sanity-check record accounting without
// re-reading the log.
struct RunSummary {
  std::int64_t steps = 0;           // RK4 steps taken
  std::array<double, 3> final_r_m{{0.0, 0.0, 0.0}};
  std::array<double, 3> final_v_mps{{0.0, 0.0, 0.0}};
  std::int64_t truth_records = 0;
  std::int64_t event_records = 0;
};

// Gravitational parameter GM [m^3/s^2] for a named central body. Phase 1
// knows only "earth"; unknown names throw std::invalid_argument. This is the
// single home of the constant (star/constants.hpp) - Python calls it through
// the binding for Keplerian conversions instead of duplicating the value.
double gm(const std::string& body);

// Propagate the two-body case defined by `cfg` and write the SRLOG v1.0 file
// to `out_path`. Truth records are decimated from integrator steps (record at
// t = 0 and every 1/truth_rate_hz; never interpolated); events records are
// code 1 "run_start" at t = 0 and code 2 "run_end" at t = duration_s.
RunSummary run_twobody(const RunConfig& cfg, const std::string& out_path);

}  // namespace star

#endif  // STAR_RUN_HPP
