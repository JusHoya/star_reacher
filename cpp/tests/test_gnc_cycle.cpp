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
  // Per-cycle aiding flags exactly as the loop presented them. Held as char
  // rather than bool because vector<bool> hands back a proxy, not a
  // reference, which reads badly in a CHECK.
  std::vector<char> navfix_valid;
  std::vector<char> navfix_fresh;
};
ProbeLog g_probe;

class ProbeNav final : public star::gnc::IGncComponent {
 public:
  void init(const star::gnc::GncInitContext& ctx) override { q0_ = ctx.q0_i2b; }
  star::gnc::GncOutput update(const star::gnc::GncInput& in) override {
    g_probe.cycles += 1;
    g_probe.navfix_valid.push_back(in.navfix.valid ? 1 : 0);
    g_probe.navfix_fresh.push_back(in.navfix.fresh ? 1 : 0);
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

// --- innovation-reporting probe -------------------------------------------

// What the innovating probe should emit on the next cycle. Set before the
// run; `narrow_first` alternates a short and a full-width sample so both the
// zero-padding and the structural embedding of a short covariance triangle
// execute inside the same run.
struct InnovPlan {
  bool alternate = true;
  std::size_t forced_y = 0;        // 0 = follow the alternation
  std::size_t forced_s_upper = 0;  // 0 = the consistent triangle for forced_y
};
InnovPlan g_innov_plan;

// m_max = 3 with real m of 2 and 3: the short case is what makes the padding
// loop's row stride observable, because a 2-by-2 packed triangle and the
// leading corner of a 3-by-3 one are laid out differently.
constexpr int kInnovMaxDim = 3;

class InnovNav final : public star::gnc::IGncComponent {
 public:
  void init(const star::gnc::GncInitContext& ctx) override { q0_ = ctx.q0_i2b; }

  star::gnc::GncOutput update(const star::gnc::GncInput&) override {
    samples_.clear();
    star::gnc::InnovationSample s;
    s.sensor_id = 0;  // the run's single configured sensor
    if (g_innov_plan.forced_y > 0) {
      // Deliberately malformed, for the loop's own length guards.
      s.y.assign(g_innov_plan.forced_y, 1.0);
      s.s_upper.assign(g_innov_plan.forced_s_upper, 1.0);
    } else if ((cycle_ % 2) == 0) {
      // m = 2 < m_max: y pads to three, and the 2-by-2 triangle
      // [ [4, 0.5], [., 9] ] must land in the leading corner of the 3-by-3.
      s.y = {1.0, 2.0};
      s.s_upper = {4.0, 0.5, 9.0};
    } else {
      s.y = {1.0, 2.0, 3.0};
      s.s_upper = {4.0, 0.5, 0.25, 9.0, 0.75, 16.0};
    }
    cycle_ += 1;
    samples_.push_back(s);

    star::gnc::GncOutput out;
    out.valid = true;
    out.q_i2b = q0_;
    return out;
  }

  int innov_max_dim() const override { return kInnovMaxDim; }

  const std::vector<star::gnc::InnovationSample>& innovations() const override {
    return samples_;
  }

 private:
  Eigen::Quaterniond q0_ = Eigen::Quaterniond::Identity();
  std::vector<star::gnc::InnovationSample> samples_;
  int cycle_ = 0;
};

std::unique_ptr<star::gnc::IGncComponent> make_innov_nav(
    const star::gnc::GncComponentCfg&) {
  return std::unique_ptr<star::gnc::IGncComponent>(new InnovNav);
}

void ensure_innov_registered() {
  static const bool once = [] {
    star::gnc::register_component("test_innov_nav", &make_innov_nav);
    return true;
  }();
  (void)once;
}

struct InnovRecord {
  double t_s = 0.0;
  std::uint32_t sensor_id = 0;
  std::uint32_t m = 0;
  double y[kInnovMaxDim] = {0.0, 0.0, 0.0};
  double s_upper[kInnovMaxDim * (kInnovMaxDim + 1) / 2] = {0.0, 0.0, 0.0,
                                                           0.0, 0.0, 0.0};
};

// Extract the nav.innov records from a log written by the innovating probe.
// The probe declares state_dim() == 0, so no nav.est or nav.err group is
// declared and the order is truth 0, events 1, sensors.imu 2, nav.innov 3,
// gnc.cmd 4 (srlog_writer.cpp declaration order).
std::vector<InnovRecord> read_nav_innov_records(const std::string& path) {
  const std::vector<unsigned char> bytes = read_all_bytes(path);
  const std::uint32_t json_len = read_le<std::uint32_t>(bytes, 12);
  std::size_t off = 16 + json_len;
  std::vector<InnovRecord> out;
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
      InnovRecord rec;
      rec.t_s = read_le<double>(bytes, off);
      rec.sensor_id = read_le<std::uint32_t>(bytes, off + 8);
      rec.m = read_le<std::uint32_t>(bytes, off + 12);
      for (int i = 0; i < kInnovMaxDim; ++i) {
        rec.y[i] = read_le<double>(bytes, off + 16 + 8 * i);
      }
      for (int i = 0; i < kInnovMaxDim * (kInnovMaxDim + 1) / 2; ++i) {
        rec.s_upper[i] = read_le<double>(bytes, off + 40 + 8 * i);
      }
      out.push_back(rec);
      off += 88;
    } else if (gi == 4) {
      off += 92;
    } else {
      FAIL("unexpected group index in innovation scenario log: ", gi);
    }
  }
  return out;
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

TEST_CASE("gnc_cycle_navfix_validity_is_forwarded_not_asserted") {
  // FR-24 observation surface: GncInput.navfix.valid carries the sensor's
  // own flag, as the star tracker's and the altimeter's already do, rather
  // than a constant true. A 1 Hz fix under the scenario's 10 Hz control rate
  // leaves cycles 0..9 with no fix ever taken, which is precisely where a
  // forwarded flag and a hardcoded true disagree.
  ensure_probe_registered();
  const std::string path = "test_gnc_cycle_navfix_valid.srlog";
  star::RunConfig cfg = gnc_scenario(false, 0);
  cfg.gnc.nav.component = "test_probe_nav";
  star::gnc::GncSensorCfg fix;
  fix.kind = "navfix";
  fix.sample_rate_hz = 1;
  cfg.gnc.sensors.push_back(fix);
  g_probe = ProbeLog{};
  {
    star::VehicleCycle vc(cfg, path);
    while (vc.step()) {
    }
  }
  std::remove(path.c_str());

  REQUIRE(g_probe.navfix_valid.size() == 21);  // 2 s at 10 Hz, cycles 0..20
  REQUIRE(g_probe.navfix_fresh.size() == 21);

  // Before the first fix the sensor still holds the zero vectors it was
  // constructed with, so the flag must be false. This is the assertion a
  // hardcoded true cannot satisfy.
  for (std::size_t c = 0; c < 10; ++c) {
    CHECK(g_probe.navfix_valid[c] == 0);
    CHECK(g_probe.navfix_fresh[c] == 0);
  }

  // Cycle 10 is the first sample: fresh and valid together. Validity then
  // persists, because the held fix stays a real measurement, while freshness
  // is true only on the cycles that actually sampled.
  CHECK(g_probe.navfix_fresh[10] == 1);
  CHECK(g_probe.navfix_fresh[11] == 0);
  CHECK(g_probe.navfix_fresh[20] == 1);
  for (std::size_t c = 10; c < 21; ++c) {
    CHECK(g_probe.navfix_valid[c] == 1);
  }
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

// --- nav.innov consumer, driven from the C++ tier -------------------------

TEST_CASE("gnc_cycle_innovation_consumer_pads_and_embeds") {
  // The nav.innov consumer block of the cycle, driven end to end by a real
  // run. Until this case the block ran only under the Python tier, against a
  // wheel carrying no instrumentation, so the embedding loop where this
  // phase's two heap defects lived was outside the reach of the sanitizer
  // and warnings-as-errors legs that build the doctest binary.
  //
  // Fixture non-degeneracy: the probe declares innov_max_dim() == 3 and
  // emits a SHORT (m = 2) sample on even cycles. Padding and embedding are
  // both no-ops when m always equals m_max, so a probe that only ever
  // emitted full-width samples would execute the loop while checking
  // nothing. The two assertions below distinguish the structural embedding
  // from a flat copy, which is the defect the loop exists to prevent.
  ensure_innov_registered();
  const std::string path = "test_gnc_cycle_innov.srlog";
  g_innov_plan = InnovPlan{};

  star::RunConfig cfg = gnc_scenario(false, 0);
  cfg.gnc.nav.component = "test_innov_nav";
  {
    star::VehicleCycle vc(cfg, path);
    while (vc.step()) {
    }
  }

  const std::vector<InnovRecord> recs = read_nav_innov_records(path);
  std::remove(path.c_str());
  REQUIRE(recs.size() == 21);  // one aiding update per cycle, 0..20

  for (std::size_t k = 0; k < recs.size(); ++k) {
    const InnovRecord& r = recs[k];
    CHECK(r.sensor_id == 0);
    CHECK(r.t_s == doctest::Approx(0.1 * static_cast<double>(k)));
    if ((k % 2) == 0) {
      // m = 2: y carries two entries and a zero pad.
      CHECK(r.m == 2);
      CHECK(r.y[0] == 1.0);
      CHECK(r.y[1] == 2.0);
      CHECK(r.y[2] == 0.0);
      // The 2-by-2 triangle [4, 0.5; ., 9] embedded in the leading corner of
      // the 3-by-3 packed upper triangle, whose entry order is
      // (0,0) (0,1) (0,2) (1,1) (1,2) (2,2). Row 0 of the short block
      // occupies the first two slots and row 1 occupies slot 3 - NOT slot 2,
      // which is where a flat copy of the three packed entries would put it.
      CHECK(r.s_upper[0] == 4.0);
      CHECK(r.s_upper[1] == 0.5);
      CHECK(r.s_upper[2] == 0.0);
      CHECK(r.s_upper[3] == 9.0);
      CHECK(r.s_upper[4] == 0.0);
      CHECK(r.s_upper[5] == 0.0);
    } else {
      // m = m_max: the record is the sample verbatim, which is the control
      // showing the padding path above is not simply zeroing everything.
      CHECK(r.m == 3);
      CHECK(r.y[0] == 1.0);
      CHECK(r.y[1] == 2.0);
      CHECK(r.y[2] == 3.0);
      const double expect[6] = {4.0, 0.5, 0.25, 9.0, 0.75, 16.0};
      for (int i = 0; i < 6; ++i) CHECK(r.s_upper[i] == expect[i]);
    }
  }
  g_innov_plan = InnovPlan{};
}

TEST_CASE("gnc_cycle_innovation_consumer_refuses_a_malformed_sample") {
  // The consumer's two length guards, in the tier a sanitizer instruments.
  // Both are asserted from Python against the exception message; what is
  // added here is that they run inside an instrumented binary, because an
  // unguarded copy of an over-wide y writes past a heap buffer sized once at
  // activation.
  // Fixture non-degeneracy: the run is otherwise the same working scenario
  // as the case above, which is shown to complete, so the throw can only
  // come from the malformed sample and not from a broken configuration.
  ensure_innov_registered();
  const std::string path = "test_gnc_cycle_innov_bad.srlog";

  {
    // y wider than the declared maximum.
    g_innov_plan = InnovPlan{};
    g_innov_plan.forced_y = kInnovMaxDim + 1;
    g_innov_plan.forced_s_upper =
        (kInnovMaxDim + 1) * (kInnovMaxDim + 2) / 2;  // self-consistent
    star::RunConfig cfg = gnc_scenario(false, 0);
    cfg.gnc.nav.component = "test_innov_nav";
    star::VehicleCycle vc(cfg, path);
    CHECK_THROWS_WITH_AS(vc.step(), doctest::Contains("innov_max_dim"),
                         std::length_error);
    vc.close();
  }
  {
    // A covariance triangle too short for the sample's own dimension, which
    // the embedding loop would otherwise read past the end of.
    g_innov_plan = InnovPlan{};
    g_innov_plan.forced_y = 3;
    g_innov_plan.forced_s_upper = 5;  // 3-by-3 needs exactly 6
    star::RunConfig cfg = gnc_scenario(false, 0);
    cfg.gnc.nav.component = "test_innov_nav";
    star::VehicleCycle vc(cfg, path);
    CHECK_THROWS_WITH_AS(vc.step(), doctest::Contains("packed upper triangle"),
                         std::length_error);
    vc.close();
  }
  g_innov_plan = InnovPlan{};
  std::remove(path.c_str());
}

TEST_CASE("gnc_cycle_innovation_run_is_byte_identical") {
  // The aiding path carries no clock and no ordering hazard of its own: two
  // runs of the innovating scenario produce identical files, so the nav.innov
  // records are as reproducible as the rest of the log (FR-21).
  ensure_innov_registered();
  const std::string p1 = "test_gnc_innov_det_a.srlog";
  const std::string p2 = "test_gnc_innov_det_b.srlog";
  g_innov_plan = InnovPlan{};
  star::RunConfig cfg = gnc_scenario(false, 0);
  cfg.gnc.nav.component = "test_innov_nav";
  {
    star::VehicleCycle vc(cfg, p1);
    while (vc.step()) {
    }
  }
  {
    star::VehicleCycle vc(cfg, p2);
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
