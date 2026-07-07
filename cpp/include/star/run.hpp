// Batch run entry points: the Phase 1 two-body reference path (run_twobody)
// and the Phase 3 composed-environment path (run_env). The Python frontend
// validates and canonicalizes the mission file (D-2), fills a RunConfig, and
// calls the matching entry point; the core propagates and writes the SRLOG
// without ever parsing text, touching the network, or reading the clock
// (architecture boundary rule, PRD 7).
#ifndef STAR_RUN_HPP
#define STAR_RUN_HPP

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "star/vehicle_config.hpp"

namespace star {

// Mirror of star_reacher._core.RunConfig (Phase 1 contract section 3; Phase 3
// environment extension). All user-facing validation happens in Python before
// this struct is populated; the core re-checks only what it needs to stay
// well-defined and throws std::invalid_argument on violations (defensive
// guard, not user UX). The Phase 1 fields and their defaults are unchanged so
// legacy configurations resolve exactly as before.
struct RunConfig {
  std::string epoch_utc;            // ISO-8601, carried verbatim into the log header
  double duration_s = 0.0;          // > 0; integer multiple of dt_s for rk4
  double dt_s = 0.0;                // > 0; fixed RK4 step (rk4 only)
  std::string integrator = "rk4";   // "rk4" | "rkf78"
  std::string central_body = "earth";  // "earth" | "moon" | "mars" | "sun"
  std::array<double, 3> r0_m{{0.0, 0.0, 0.0}};     // GCRF position [m]
  std::array<double, 3> v0_mps{{0.0, 0.0, 0.0}};   // GCRF velocity [m/s]
  double mass_kg = 1.0;             // > 0; constant through Phase 3
  std::uint64_t master_seed = 0;    // D-9 master seed, echoed in the header
  std::uint32_t truth_rate_hz = 10; // >= 1; 1/(dt_s*rate) must be an integer (rk4)
  std::string config_sha256;        // 64-hex resolved-config digest (FR-15)
  bool oracle = false;              // D-11 oracle flag, stamped in the header

  // --- Phase 3 extension (consumed by run_env only) ------------------------
  // Mission epoch on the TAI scale (two-part, D-6), computed by the Python
  // layer from epoch_utc via the bound time functions so the core never
  // parses the ISO string (D-2).
  std::int64_t epoch_tai_day = 0;
  double epoch_tai_sec = 0.0;

  // Adaptive RKF7(8) controls (integrator == "rkf78"): per-state-group
  // tolerances and explicit initial/maximum step (FR-11; an explicit h_init
  // keeps runs bitwise reproducible under configuration inspection).
  double rtol = 0.0;
  double atol_pos_m = 0.0;
  double atol_vel_mps = 0.0;
  double h_init_s = 0.0;
  double h_max_s = 0.0;

  // Environment surface (mirrors [environment]/[spacecraft], D-2-validated).
  std::string gravity_model = "pointmass";  // "pointmass" | "j2" | "harmonic"
  std::string gravity_field_path;           // SRGRAV file for j2/harmonic
  int gravity_degree = -1;                  // harmonic truncation degree
  int gravity_order = -1;                   // harmonic truncation order
  std::vector<std::string> third_bodies;    // FR-6 perturber names
  bool srp_enabled = false;                 // FR-7
  double cr_a_over_m_m2pkg = 0.0;
  std::vector<std::string> srp_occulters;
  bool drag_enabled = false;                // FR-8/FR-9
  std::string atmosphere;                   // "ussa76" | "harris_priester" | "mars_exponential"
  double cd_a_over_m_m2pkg = 0.0;
  double hp_exponent_n = 4.0;
  std::string ephemeris_path;               // SREPH file when any model needs it

  // --- Phase 4 extension (consumed by run_vehicle only) --------------------
  // The vehicle definition and event sequence as plain data (D-2). run() and
  // run_env() ignore these; a mission takes the vehicle path only when the
  // Python frontend has populated a nonempty vehicle.stages and calls
  // run_vehicle. Every field the vehicle path re-checks throws
  // std::invalid_argument on violation, matching check_config_env.
  VehicleConfig vehicle;
  std::vector<SequenceEntry> sequence;

  // Initial-state form for the vehicle path: "cartesian" | "keplerian" (both
  // fill r0_m/v0_mps) or "geodetic" (fills the launch-site fields below and
  // starts pad-fixed until a pad_release sequence event, FR-14).
  std::string initial_form = "cartesian";
  double launch_lat_deg = 0.0;
  double launch_lon_deg = 0.0;
  double launch_alt_m = 0.0;

  // v1.1 vehicle-group logging rates (FR-16). Nonzero enables the group; each
  // must divide truth_rate_hz (the SRLOG writer re-checks). run_vehicle
  // decides the enabled force-source subset from the configured environment
  // and vehicle (canonical order, format doc section 3.1).
  std::uint32_t forces_rate_hz = 0;
  std::uint32_t mass_rate_hz = 0;
  std::uint32_t env_rate_hz = 0;
};

// Run summary returned across the binding: enough for the CLI to print a
// final-state line and for tests to sanity-check record accounting without
// re-reading the log.
struct RunSummary {
  std::int64_t steps = 0;  // integrator steps taken (accepted steps for rkf78)
  std::array<double, 3> final_r_m{{0.0, 0.0, 0.0}};
  std::array<double, 3> final_v_mps{{0.0, 0.0, 0.0}};
  std::int64_t truth_records = 0;
  std::int64_t event_records = 0;
};

// Gravitational parameter GM [m^3/s^2] for a named central body ("earth",
// "moon", "mars", "sun"); unknown names throw std::invalid_argument. This is
// the single home of the constant (star/constants.hpp) - Python calls it
// through the binding for Keplerian conversions instead of duplicating the
// value. Earth is the IERS TN36 value; Moon, Mars, and the Sun are the DE440
// header values (see the constants.hpp note on the deliberate split).
double gm(const std::string& body);

// Propagate the two-body case defined by `cfg` and write the SRLOG v1.0 file
// to `out_path`. Truth records are decimated from integrator steps (record at
// t = 0 and every 1/truth_rate_hz; never interpolated); events records are
// code 1 "run_start" at t = 0 and code 2 "run_end" at t = duration_s. This
// Phase 1 path is byte-frozen: it ignores the Phase 3 RunConfig extension,
// and its output for a given config is part of the determinism record
// (tests/golden/determinism/cross_platform.toml).
RunSummary run_twobody(const RunConfig& cfg, const std::string& out_path);

// Propagate the composed-environment case (Phase 3): point-mass spacecraft
// about cfg.central_body under the configured gravity tier, third bodies,
// SRP, and drag (star/models/environment.hpp), with either the fixed-step
// RK4 (truth records by step decimation, exactly the run_twobody semantics)
// or the adaptive RKF7(8) (truth records sampled from the Hermite dense
// output at k / truth_rate_hz; interpolation error is bounded by h_max^4,
// see ch:integrators, so h_max_s is the logging-accuracy control). Events
// are run_start/run_end as in run_twobody; the SRLOG schema is unchanged
// (the FR-16 forces group lands with the Phase 4 torque channels).
RunSummary run_env(const RunConfig& cfg, const std::string& out_path);

// Propagate the full 6DOF vehicle case (Phase 4): a staged vehicle
// (star/vehicle_config.hpp) flown under the composed environment
// (star/models/environment.hpp) plus its own thrust, aerodynamics, and
// attitude, driven by the open-loop event [[sequence]]. The translational
// state [r, v] is advanced with fixed-step RK4; attitude, per-tank propellant
// masses, and per-engine spool/gimbal/ignition states are advanced per control
// cycle (D-5 zero-order hold). The SRLOG carries the v1.1 truth (with real q
// and omega), forces, mass, and env groups. Events (pad release, ignition,
// cutoff, staging/jettison with the FR-10 remap, orbit-insertion, and
// SOI-transition) are located at control-cycle boundaries; a terminal event
// stops the run. Defensive re-checks throw std::invalid_argument, matching
// run_env's check_config_env style. run() and run_env() are byte-frozen and
// share none of this path.
RunSummary run_vehicle(const RunConfig& cfg, const std::string& out_path);

}  // namespace star

#endif  // STAR_RUN_HPP
