// VehicleCycle GNC integration tests (FR-25 / Phase 6): the oracle
// privileged-truth gating, the latency_cycles application shift observed
// through the gnc.cmd group (exit criterion 8 in miniature), header
// declarations, defensive config rejection, and double-run byte identity of
// a full GNC-enabled run (FR-21). The scenario is a minimal free-flying
// single-stage vehicle in LEO under an attitude-hold GNC chain, built
// directly as a RunConfig (the core never parses text, D-2).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "star/gnc/component.hpp"
#include "star/run.hpp"
#include "star/vehicle_cycle.hpp"
#include "vendor/doctest.h"

namespace {

std::vector<unsigned char> read_all_bytes(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  REQUIRE(static_cast<bool>(in));
  return std::vector<unsigned char>(std::istreambuf_iterator<char>(in),
                                    std::istreambuf_iterator<char>());
}

template <typename T>
T read_le(const std::vector<unsigned char>& buf, std::size_t offset) {
  REQUIRE(offset + sizeof(T) <= buf.size());
  T v;
  std::memcpy(&v, buf.data() + offset, sizeof(T));
  return v;
}

// Minimal free-flying attitude-hold GNC scenario: one rigid stage (150 kg
// smallsat-class inertia), LEO state, 10 Hz control for 2 s, ideal IMU at
// the control rate, dead-reckoning nav, attitude-hold guidance ~10 deg off
// the initial attitude, PD control with wheel-scale torque authority.
star::RunConfig gnc_scenario(bool oracle, std::uint32_t latency_cycles) {
  star::RunConfig cfg;
  cfg.epoch_utc = "2026-01-01T00:00:00Z";
  cfg.duration_s = 2.0;
  cfg.dt_s = 0.1;
  cfg.integrator = "rk4";
  cfg.central_body = "earth";
  cfg.r0_m = {7.0e6, 0.0, 0.0};
  cfg.v0_mps = {0.0, 7546.0, 0.0};
  cfg.master_seed = 42;
  cfg.truth_rate_hz = 10;
  cfg.config_sha256 = std::string(64, '0');
  cfg.oracle = oracle;

  star::StageCfg stage;
  stage.name = "bus";
  stage.dry_mass_kg = 150.0;
  stage.dry_cg_m = {0.3, 0.0, 0.0};
  stage.dry_inertia_kgm2 = {9.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 11.0};
  cfg.vehicle.stages.push_back(stage);

  cfg.gnc.enabled = true;
  cfg.gnc.control_rate_hz = 10;
  cfg.gnc.latency_cycles = latency_cycles;
  cfg.gnc.nav.component = "dead_reckoning";
  // The configured initial estimate (ch:gnc-builtin: stated explicitly,
  // no implicit truth access): the Phase 4 initial-attitude rule for
  // r = +X, v = +Y gives exactly [0, sqrt(1/2), sqrt(1/2), 0].
  cfg.gnc.nav.vectors["q0"] = {0.0, 0.7071067811865476, 0.7071067811865476,
                               0.0};
  cfg.gnc.guidance.component = "attitude_hold";
  cfg.gnc.guidance.vectors["q_cmd"] = {0.996194698091746, 0.0, 0.0,
                                       0.087155742747658};
  cfg.gnc.control.component = "pd_attitude";
  cfg.gnc.control.vectors["kp_nm_per_rad"] = {0.4, 0.4, 0.4};
  cfg.gnc.control.vectors["kd_nm_per_radps"] = {3.6, 3.6, 3.6};
  cfg.gnc.control.vectors["tau_max_nm"] = {0.05, 0.05, 0.05};
  star::gnc::GncSensorCfg imu;
  imu.kind = "imu";
  imu.sample_rate_hz = 10;
  cfg.gnc.sensors.push_back(imu);
  return cfg;
}

struct CmdRecord {
  double t_s = 0.0;
  double tau[3] = {0.0, 0.0, 0.0};
  double q_cmd[4] = {0.0, 0.0, 0.0, 0.0};
  double w_cmd[3] = {0.0, 0.0, 0.0};
  std::uint32_t valid = 0;
};

// Extract the gnc.cmd records from a log written by the gnc_scenario config
// with the dead_reckoning nav. Group indices and payload sizes follow the
// header declaration order pinned by test_srlog.cpp: truth 0 (120 B),
// events 1 (variable), sensors.imu 2 (56 B), nav.est 3 (n = 7: 288 B),
// nav.err 4 (64 B), gnc.cmd 5 (92 B).
std::vector<CmdRecord> read_gnc_cmd_records(const std::string& path) {
  const std::vector<unsigned char> bytes = read_all_bytes(path);
  const std::uint32_t json_len = read_le<std::uint32_t>(bytes, 12);
  std::size_t off = 16 + json_len;
  std::vector<CmdRecord> out;
  while (off < bytes.size()) {
    const std::uint16_t gi = read_le<std::uint16_t>(bytes, off);
    off += 2;
    if (gi == 0) {
      off += 120;
    } else if (gi == 1) {
      const std::uint16_t len = read_le<std::uint16_t>(bytes, off + 12);
      off += 8 + 4 + 2 + len;
    } else if (gi == 2) {
      off += 56;
    } else if (gi == 3) {
      off += 288;
    } else if (gi == 4) {
      off += 64;
    } else if (gi == 5) {
      CmdRecord rec;
      rec.t_s = read_le<double>(bytes, off);
      for (int i = 0; i < 3; ++i) {
        rec.tau[i] = read_le<double>(bytes, off + 8 + 8 * i);
      }
      for (int i = 0; i < 4; ++i) {
        rec.q_cmd[i] = read_le<double>(bytes, off + 32 + 8 * i);
      }
      for (int i = 0; i < 3; ++i) {
        rec.w_cmd[i] = read_le<double>(bytes, off + 64 + 8 * i);
      }
      rec.valid = read_le<std::uint32_t>(bytes, off + 88);
      out.push_back(rec);
      off += 92;
    } else {
      FAIL("unexpected group index in gnc scenario log: ", gi);
    }
  }
  return out;
}

// Oracle-gating probe: a nav component that records whether the privileged
// truth block was populated on any cycle. Registered once under a
// test-reserved name; the recording state is reset per run.
struct ProbeLog {
  int cycles = 0;
  int oracle_valid_cycles = 0;
  Eigen::Vector3d first_oracle_r = Eigen::Vector3d::Zero();
};
ProbeLog g_probe;

class ProbeNav final : public star::gnc::IGncComponent {
 public:
  void init(const star::gnc::GncInitContext& ctx) override { q0_ = ctx.q0_i2b; }
  star::gnc::GncOutput update(const star::gnc::GncInput& in) override {
    g_probe.cycles += 1;
    if (in.oracle.valid) {
      if (g_probe.oracle_valid_cycles == 0) {
        g_probe.first_oracle_r = in.oracle.r_i_m;
      }
      g_probe.oracle_valid_cycles += 1;
    }
    star::gnc::GncOutput out;
    out.valid = true;
    out.q_i2b = q0_;
    return out;
  }

 private:
  Eigen::Quaterniond q0_ = Eigen::Quaterniond::Identity();
};

std::unique_ptr<star::gnc::IGncComponent> make_probe_nav(
    const star::gnc::GncComponentCfg&) {
  return std::unique_ptr<star::gnc::IGncComponent>(new ProbeNav);
}

void ensure_probe_registered() {
  static const bool once = [] {
    star::gnc::register_component("test_probe_nav", &make_probe_nav);
    return true;
  }();
  (void)once;
}

}  // namespace

TEST_CASE("gnc_cycle_header_declarations") {
  const star::log::SrlogHeaderFields fields =
      star::VehicleCycle::make_header_fields(gnc_scenario(false, 3));
  CHECK(fields.cycle_rate_hz == 10);
  CHECK(fields.latency_cycles == 3);
  REQUIRE(fields.sensors.size() == 1);
  CHECK(fields.sensors[0].kind == "imu");
  CHECK(fields.sensors[0].rate_hz == 10);
  CHECK(fields.nav_est_rate_hz == 10);
  CHECK(fields.nav_state_dim == 7);       // dead_reckoning declares n = 7
  CHECK(fields.nav_err_enabled);
  CHECK_FALSE(fields.nav_innov_enabled);  // no aiding sensors in this phase
  CHECK(fields.gnc_cmd_rate_hz == 10);
  CHECK_FALSE(fields.oracle);
  CHECK(star::VehicleCycle::make_header_fields(gnc_scenario(true, 0)).oracle);

  // A GNC-disabled vehicle config declares no v1.2 group at all.
  star::RunConfig plain = gnc_scenario(false, 0);
  plain.gnc = star::gnc::GncConfig{};
  const star::log::SrlogHeaderFields off =
      star::VehicleCycle::make_header_fields(plain);
  CHECK(off.cycle_rate_hz == 0);
  CHECK(off.sensors.empty());
  CHECK(off.nav_est_rate_hz == 0);
  CHECK(off.gnc_cmd_rate_hz == 0);
}

TEST_CASE("gnc_cycle_config_rejection") {
  // control_rate_hz must equal 1/dt_s (one control cycle per step, D-5).
  {
    star::RunConfig cfg = gnc_scenario(false, 0);
    cfg.gnc.control_rate_hz = 20;
    CHECK_THROWS_AS(star::VehicleCycle::make_header_fields(cfg),
                    std::invalid_argument);
  }
  // The v1 IMU emits one increment pair per major cycle: its rate must
  // EQUAL the control rate (ch:sensors-imu assumption 1) - even an exact
  // divisor is rejected.
  {
    star::RunConfig cfg = gnc_scenario(false, 0);
    cfg.gnc.sensors[0].sample_rate_hz = 5;  // divides 10, still rejected
    CHECK_THROWS_AS(star::VehicleCycle::make_header_fields(cfg),
                    std::invalid_argument);
  }
  // Open-loop attitude sequence actions cannot coexist with [gnc].
  {
    star::RunConfig cfg = gnc_scenario(false, 0);
    star::SequenceEntry e;
    e.name = "hold";
    e.trigger = "elapsed";
    e.t_s = 1.0;
    e.action = "attitude_hold";
    cfg.sequence.push_back(e);
    CHECK_THROWS_AS(star::VehicleCycle::make_header_fields(cfg),
                    std::invalid_argument);
  }
  // Unknown chain component names are rejected by the registry.
  {
    star::RunConfig cfg = gnc_scenario(false, 0);
    cfg.gnc.nav.component = "no_such_nav";
    CHECK_THROWS_AS(star::VehicleCycle::make_header_fields(cfg),
                    std::invalid_argument);
  }
}

TEST_CASE("gnc_cycle_oracle_gating") {
  // FR-25 privileged boundary, both directions: truth enters GncInput if
  // and only if the scenario sets oracle = true. The probe nav observes the
  // GncInput the loop actually builds on every cycle of a real run.
  ensure_probe_registered();
  const std::string path = "test_gnc_cycle_oracle.srlog";

  {
    star::RunConfig cfg = gnc_scenario(false, 0);
    cfg.gnc.nav.component = "test_probe_nav";
    g_probe = ProbeLog{};
    star::VehicleCycle vc(cfg, path);
    while (vc.step()) {
    }
    CHECK(g_probe.cycles == 21);  // 2 s at 10 Hz, cycles 0..20
    CHECK(g_probe.oracle_valid_cycles == 0);
  }
  {
    star::RunConfig cfg = gnc_scenario(true, 0);
    cfg.gnc.nav.component = "test_probe_nav";
    g_probe = ProbeLog{};
    star::VehicleCycle vc(cfg, path);
    while (vc.step()) {
    }
    CHECK(g_probe.cycles == 21);
    CHECK(g_probe.oracle_valid_cycles == 21);
    // The injected truth is the loop's real state: cycle 0 sees the
    // configured initial position exactly.
    CHECK(g_probe.first_oracle_r[0] == 7.0e6);
    CHECK(g_probe.first_oracle_r[1] == 0.0);
    CHECK(g_probe.first_oracle_r[2] == 0.0);
  }
  std::remove(path.c_str());
}

TEST_CASE("gnc_cycle_latency_shifts_application_by_exactly_k") {
  // Exit criterion 8 in miniature: with latency_cycles = 2, the command the
  // k = 0 run applies on cycle 0 appears in the k = 2 run exactly two
  // cycles later, and the first two applied records are pre-fill holds.
  const std::string p0 = "test_gnc_cycle_lat0.srlog";
  const std::string p2 = "test_gnc_cycle_lat2.srlog";
  {
    star::VehicleCycle vc(gnc_scenario(false, 0), p0);
    while (vc.step()) {
    }
  }
  {
    star::VehicleCycle vc(gnc_scenario(false, 2), p2);
    while (vc.step()) {
    }
  }
  const std::vector<CmdRecord> k0 = read_gnc_cmd_records(p0);
  const std::vector<CmdRecord> k2 = read_gnc_cmd_records(p2);
  std::remove(p0.c_str());
  std::remove(p2.c_str());

  REQUIRE(k0.size() == 21);  // one applied command per cycle, 0..20
  REQUIRE(k2.size() == 21);

  // k = 0: the chain output applies immediately; the attitude-hold command
  // has a nonzero attitude error from cycle 0, so every record is a fresh
  // application with nonzero torque.
  CHECK(k0[0].valid == 1);
  CHECK(k0[0].t_s == 0.0);
  const double tau0_norm = std::sqrt(k0[0].tau[0] * k0[0].tau[0] +
                                     k0[0].tau[1] * k0[0].tau[1] +
                                     k0[0].tau[2] * k0[0].tau[2]);
  CHECK(tau0_norm > 0.0);

  // k = 2: cycles 0 and 1 apply the pre-fill hold (zero torque, invalid);
  // the k = 0 run's cycle-0 command applies at t = 2 * dt, bit-identically
  // (both runs compute it from the same initial state).
  CHECK(k2[0].valid == 0);
  CHECK(k2[1].valid == 0);
  CHECK(k2[0].tau[0] == 0.0);
  CHECK(k2[0].tau[1] == 0.0);
  CHECK(k2[0].tau[2] == 0.0);
  CHECK(k2[1].tau[2] == 0.0);
  CHECK(k2[2].valid == 1);
  CHECK(k2[2].t_s == k0[0].t_s + 2 * 0.1);
  for (int i = 0; i < 3; ++i) {
    CHECK(k2[2].tau[i] == k0[0].tau[i]);
  }
  for (int i = 0; i < 4; ++i) {
    CHECK(k2[2].q_cmd[i] == k0[0].q_cmd[i]);
  }
}

TEST_CASE("gnc_cycle_double_run_is_byte_identical") {
  // FR-21 at the whole-file level for a GNC-enabled run: same config, same
  // binary, byte-identical logs (the C++ side of the double-run SHA gate).
  const std::string p1 = "test_gnc_cycle_det_a.srlog";
  const std::string p2 = "test_gnc_cycle_det_b.srlog";
  {
    star::VehicleCycle vc(gnc_scenario(false, 1), p1);
    while (vc.step()) {
    }
  }
  {
    star::VehicleCycle vc(gnc_scenario(false, 1), p2);
    while (vc.step()) {
    }
  }
  const std::vector<unsigned char> a = read_all_bytes(p1);
  const std::vector<unsigned char> b = read_all_bytes(p2);
  std::remove(p1.c_str());
  std::remove(p2.c_str());
  REQUIRE(!a.empty());
  CHECK(a == b);
}

TEST_CASE("gnc_cycle_batch_wrapper_matches_stepping") {
  // Exit-criterion-4 groundwork: run_vehicle is literally a loop over
  // VehicleCycle::step(), so driving the cycle core one step at a time must
  // produce a byte-identical file to the batch entry point.
  const std::string pb = "test_gnc_cycle_batch.srlog";
  const std::string ps = "test_gnc_cycle_stepped.srlog";
  const star::RunConfig cfg = gnc_scenario(false, 1);
  const star::RunSummary batch = star::run_vehicle(cfg, pb);
  star::VehicleCycle vc(cfg, ps);
  std::int64_t manual_steps = 0;
  while (vc.step()) {
    manual_steps += 1;
  }
  CHECK(vc.done());
  const star::RunSummary stepped = vc.summary();
  CHECK(batch.truth_records == stepped.truth_records);
  CHECK(batch.event_records == stepped.event_records);
  CHECK(batch.steps == stepped.steps);
  CHECK(manual_steps == batch.steps);  // 20 advances for 21 cycles
  const std::vector<unsigned char> a = read_all_bytes(pb);
  const std::vector<unsigned char> b = read_all_bytes(ps);
  std::remove(pb.c_str());
  std::remove(ps.c_str());
  REQUIRE(!a.empty());
  CHECK(a == b);
}

// --- FR-24 observation snapshot and external command ----------------------

// Deep equality of two observation snapshots. Written out rather than
// defaulted because Eigen members have no operator== that compares
// element-wise to a bool, and the exit-criterion-4 claim is about VALUES:
// an idempotence test that compared object identity would pass trivially.
bool same_observation(const star::CycleObservation& a,
                      const star::CycleObservation& b) {
  const bool scalars = a.cycle == b.cycle && a.t_s == b.t_s &&
                       a.processed == b.processed && a.done == b.done &&
                       a.gnc_active == b.gnc_active &&
                       a.imu_fresh == b.imu_fresh;
  const bool imu = a.imu.valid == b.imu.valid && a.imu.t_s == b.imu.t_s &&
                   a.imu.dt_s == b.imu.dt_s &&
                   a.imu.dtheta_b_rad == b.imu.dtheta_b_rad &&
                   a.imu.dv_b_mps == b.imu.dv_b_mps;
  const bool chain = a.nav_est.valid == b.nav_est.valid &&
                     a.nav_est.q_i2b.coeffs() == b.nav_est.q_i2b.coeffs() &&
                     a.nav_est.omega_b_radps == b.nav_est.omega_b_radps &&
                     a.att_cmd.q_i2b.coeffs() == b.att_cmd.q_i2b.coeffs() &&
                     a.applied.valid == b.applied.valid &&
                     a.applied.torque_b_nm == b.applied.torque_b_nm;
  const bool aiding = a.navfix.fresh == b.navfix.fresh &&
                      a.navfix.valid == b.navfix.valid &&
                      a.navfix.r_i_m == b.navfix.r_i_m &&
                      a.startracker.fresh == b.startracker.fresh &&
                      a.altimeter.fresh == b.altimeter.fresh &&
                      a.altimeter.h_m == b.altimeter.h_m;
  const bool est = a.nav_x_hat == b.nav_x_hat &&
                   a.nav_p_upper == b.nav_p_upper;
  return scalars && imu && chain && aiding && est;
}

TEST_CASE("gnc_cycle_observation_is_idempotent_without_step") {
  // Exit criterion 4, second clause, at the core: reading the observation
  // must not advance the RNG, consume a sensor sample, or mutate a buffer,
  // so repeated reads without an intervening step() are equal.
  const std::string path = "test_gnc_obs_idempotent.srlog";
  star::VehicleCycle vc(gnc_scenario(false, 0), path);

  // Before the first step the snapshot describes the initial state.
  CHECK(vc.observation().processed == false);
  CHECK(vc.observation().cycle == 0);
  CHECK(same_observation(vc.observation(), vc.observation()));

  for (int k = 0; k < 5; ++k) {
    REQUIRE(vc.step());
    const star::CycleObservation first = vc.observation();
    const star::CycleObservation second = vc.observation();
    const star::CycleObservation third = vc.observation();
    CHECK(same_observation(first, second));
    CHECK(same_observation(second, third));
    CHECK(first.processed == true);
    CHECK(first.cycle == k);
    // Non-vacuous: the snapshot really carries this cycle's chain products.
    CHECK(first.gnc_active == true);
    CHECK(first.nav_x_hat.size() == 7u);
  }
  while (vc.step()) {
  }
  std::remove(path.c_str());
}

TEST_CASE("gnc_cycle_observation_does_not_perturb_the_log") {
  // The stronger statement behind idempotence: observing cannot change the
  // run at all. A log written while observing every cycle must be identical
  // to one written without observing.
  const std::string quiet_path = "test_gnc_obs_quiet.srlog";
  const std::string watched_path = "test_gnc_obs_watched.srlog";
  const star::RunConfig cfg = gnc_scenario(false, 0);

  star::VehicleCycle quiet(cfg, quiet_path);
  while (quiet.step()) {
  }

  star::VehicleCycle watched(cfg, watched_path);
  while (watched.step()) {
    // Three reads per cycle, plus the privileged accessor.
    (void)watched.observation();
    (void)watched.observation();
    (void)watched.observation();
    (void)watched.truth();
  }

  const std::vector<unsigned char> a = read_all_bytes(quiet_path);
  const std::vector<unsigned char> b = read_all_bytes(watched_path);
  std::remove(quiet_path.c_str());
  std::remove(watched_path.c_str());
  REQUIRE(!a.empty());
  CHECK(a == b);
}

TEST_CASE("gnc_cycle_external_command_holds_and_applies") {
  // FR-24 step(commands): the external component is an ordinary chain
  // member, so a driver-supplied torque reaches gnc.cmd and the dynamics,
  // and persists across cycles until replaced (D-5 zero-order hold).
  const std::string path = "test_gnc_external_cmd.srlog";
  star::RunConfig cfg = gnc_scenario(false, 0);
  cfg.gnc.control.component = "external";
  cfg.gnc.control.vectors.clear();

  star::VehicleCycle vc(cfg, path);
  REQUIRE(vc.has_external_command());

  star::gnc::GncOutput cmd;
  cmd.valid = true;
  cmd.torque_b_nm = Eigen::Vector3d(0.001, -0.002, 0.003);
  vc.set_external_command(cmd);
  REQUIRE(vc.step());
  CHECK(vc.observation().applied.torque_b_nm == cmd.torque_b_nm);

  // Held across a step that supplies nothing.
  REQUIRE(vc.step());
  CHECK(vc.observation().applied.torque_b_nm == cmd.torque_b_nm);

  star::gnc::GncOutput zero;
  zero.valid = true;
  vc.set_external_command(zero);
  REQUIRE(vc.step());
  CHECK(vc.observation().applied.torque_b_nm == Eigen::Vector3d::Zero());

  while (vc.step()) {
  }
  std::remove(path.c_str());
}

TEST_CASE("gnc_cycle_external_command_requires_an_external_slot") {
  // Commanding a mission that flies its own built-in chain must fail
  // loudly: a dropped command is indistinguishable from a vehicle that
  // refused to manoeuvre.
  const std::string path = "test_gnc_no_external.srlog";
  star::VehicleCycle vc(gnc_scenario(false, 0), path);
  CHECK(vc.has_external_command() == false);
  CHECK_THROWS_AS(vc.set_external_command(star::gnc::GncOutput()),
                  std::logic_error);
  while (vc.step()) {
  }
  std::remove(path.c_str());
}

TEST_CASE("gnc_cycle_external_component_rejects_parameters") {
  // The external component's command is data from the driver, not
  // configuration; a stray key is a typo, not an accepted setting.
  star::RunConfig cfg = gnc_scenario(false, 0);
  cfg.gnc.control.component = "external";
  cfg.gnc.control.vectors["kp_nm_per_rad"] = {1.0, 1.0, 1.0};
  CHECK_THROWS_AS(star::VehicleCycle(cfg, "test_gnc_external_reject.srlog"),
                  std::invalid_argument);
  std::remove("test_gnc_external_reject.srlog");
}

TEST_CASE("gnc_cycle_close_releases_the_log_of_an_abandoned_run") {
  // A run stopped part way must be able to end its FILE's lifetime without
  // ending its OBJECT's. Otherwise the log stays open for as long as the
  // owner lives, and on Windows an open handle makes a later unlink or a
  // reopen of the same path fail with a sharing violation - which is how
  // this surfaced, as a PermissionError in an unrelated teardown.
  const std::string path = "test_gnc_close_abandoned.srlog";
  {
    star::VehicleCycle vc(gnc_scenario(false, 0), path);
    REQUIRE(vc.step());
    REQUIRE(vc.step());
    CHECK(vc.done() == false);

    vc.close();
    vc.close();  // idempotent

    // The abandoned log is a valid PREFIX: real bytes, no run_end event.
    // Measured after close() because close() is what flushes - before it the
    // records written so far are still in the stream's buffer, which is the
    // other half of why an abandoned run must be closable.
    const std::streamoff written = [&] {
      std::ifstream in(path, std::ios::binary | std::ios::ate);
      return in.tellg();
    }();
    CHECK(written > 0);

    // The behavioural assertion, and the one that fails if close() stops
    // reaching the writer: the file can now be REMOVED. On Windows an open
    // handle refuses this, which is the failure being fixed; on Linux the
    // unlink succeeds either way, so this case gates on Windows only and
    // the throw below carries the rest.
    CHECK(std::remove(path.c_str()) == 0);

    // Stepping a released run is refused by its own message rather than
    // failing later inside the writer with "write failed".
    CHECK_THROWS_WITH_AS(vc.step(), doctest::Contains("after close"),
                         std::logic_error);
  }
  std::remove(path.c_str());
}

TEST_CASE("gnc_cycle_close_after_a_completed_run_is_a_no_op") {
  // finish() already closed the writer, and SrlogWriter::close() is guarded
  // by is_open(), so an explicit close on a finished run must neither throw
  // nor disturb the completed file.
  const std::string path = "test_gnc_close_finished.srlog";
  star::VehicleCycle vc(gnc_scenario(false, 0), path);
  while (vc.step()) {
  }
  REQUIRE(vc.done());
  const std::streamoff before = [&] {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    return in.tellg();
  }();
  CHECK_NOTHROW(vc.close());
  const std::streamoff after = [&] {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    return in.tellg();
  }();
  CHECK(before == after);
  // The run-ended message still wins over the closed one: this file is a
  // complete run, not an abandoned prefix, and the two are different facts.
  CHECK_THROWS_WITH_AS(vc.step(),
                       doctest::Contains("after the run ended"),
                       std::logic_error);
  std::remove(path.c_str());
}
