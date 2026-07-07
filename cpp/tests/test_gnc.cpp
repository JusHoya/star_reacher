// GNC component unit tests (FR-25): registry resolution and rejection, the
// pd_attitude control-law golden vectors (including the sign-unwinding and
// saturation branches), the dead_reckoning composition golden, the
// pitch-program guidance's bit-equality with the Phase 4 open-loop
// machinery, attitude-hold semantics, and the latency FIFO. Golden inputs
// and tolerances: tests/golden/gnc/manifest.toml.
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "golden_io.hpp"
#include "star/constants.hpp"
#include "star/gnc/builtin.hpp"
#include "star/gnc/component.hpp"
#include "star/models/vehicle6dof.hpp"
#include "vendor/doctest.h"

namespace {

using star::gnc::GncComponentCfg;
using star::gnc::GncInitContext;
using star::gnc::GncInput;
using star::gnc::GncOutput;

std::vector<double> golden_vec(const star_tests::GoldenCase& c,
                               const std::string& key) {
  std::vector<double> out;
  for (const std::string& s : c.array(key)) {
    out.push_back(star_tests::parse_hex_double(s));
  }
  return out;
}

Eigen::Quaterniond quat_of(const std::vector<double>& v) {
  REQUIRE(v.size() == 4);
  return Eigen::Quaterniond(v[0], v[1], v[2], v[3]);  // scalar-first (D-7)
}

Eigen::Vector3d vec3_of(const std::vector<double>& v) {
  REQUIRE(v.size() == 3);
  return Eigen::Vector3d(v[0], v[1], v[2]);
}

// Manifest tolerance: |got - expected| <= max(5e-15 |expected|, 1e-18).
void check_close(double got, double expected) {
  const double tol = std::max(5e-15 * std::fabs(expected), 1e-18);
  CHECK(std::fabs(got - expected) <= tol);
}

}  // namespace

TEST_CASE("gnc_registry_resolution_and_rejection") {
  const std::vector<std::string> names = star::gnc::component_names();
  for (const char* built_in :
       {"attitude_hold", "dead_reckoning", "pd_attitude", "pitch_program"}) {
    CHECK(std::find(names.begin(), names.end(), built_in) != names.end());
  }

  // Unknown names are rejected with the registered set in the message so a
  // config typo is self-diagnosing.
  GncComponentCfg bad;
  bad.component = "kalman_9000";
  bool threw = false;
  try {
    star::gnc::make_component(bad);
  } catch (const std::invalid_argument& e) {
    threw = true;
    const std::string msg = e.what();
    CHECK(msg.find("kalman_9000") != std::string::npos);
    CHECK(msg.find("dead_reckoning") != std::string::npos);
    CHECK(msg.find("pd_attitude") != std::string::npos);
  }
  CHECK(threw);

  // Duplicate registration is a determinism hazard and refuses loudly.
  auto factory = [](const GncComponentCfg&) {
    return std::unique_ptr<star::gnc::IGncComponent>();
  };
  CHECK(star::gnc::register_component("test_gnc_dup_probe", factory));
  CHECK_THROWS_AS(star::gnc::register_component("test_gnc_dup_probe", factory),
                  std::logic_error);

  // Components validate their own parameters defensively.
  GncComponentCfg pd;
  pd.component = "pd_attitude";
  pd.vectors["kp_nm_per_rad"] = {1.0, 1.0};  // wrong size
  pd.vectors["kd_nm_per_radps"] = {1.0, 1.0, 1.0};
  pd.vectors["tau_max_nm"] = {1.0, 1.0, 1.0};
  CHECK_THROWS_AS(star::gnc::make_component(pd), std::invalid_argument);
  GncComponentCfg dr;
  dr.component = "dead_reckoning";
  dr.scalars["mystery_knob"] = 1.0;  // unknown parameter
  CHECK_THROWS_AS(star::gnc::make_component(dr), std::invalid_argument);
}

TEST_CASE("gnc_pd_attitude_golden") {
  const std::vector<star_tests::GoldenCase> cases =
      star_tests::load_golden_cases(std::string(STAR_GOLDEN_DIR) +
                                    "/gnc/pd_attitude.toml");
  REQUIRE(cases.size() == 5);
  bool saw_negative_branch = false;
  bool saw_zero_branch = false;
  for (const star_tests::GoldenCase& c : cases) {
    GncComponentCfg cfg;
    cfg.component = "pd_attitude";
    cfg.vectors["kp_nm_per_rad"] = golden_vec(c, "kp");
    cfg.vectors["kd_nm_per_radps"] = golden_vec(c, "kd");
    cfg.vectors["tau_max_nm"] = golden_vec(c, "tau_max");
    std::unique_ptr<star::gnc::IGncComponent> pd =
        star::gnc::make_component(cfg);
    pd->init(GncInitContext{});

    GncInput in;
    in.nav_est.valid = true;
    in.nav_est.q_i2b = quat_of(golden_vec(c, "q_est"));
    in.nav_est.omega_b_radps = vec3_of(golden_vec(c, "w_est"));
    in.att_cmd.valid = true;
    in.att_cmd.q_i2b = quat_of(golden_vec(c, "q_cmd"));
    in.att_cmd.omega_b_radps = vec3_of(golden_vec(c, "w_cmd"));
    const GncOutput out = pd->update(in);
    CHECK(out.valid);

    const std::vector<double> expected = golden_vec(c, "expected_tau_nm");
    for (int i = 0; i < 3; ++i) {
      check_close(out.torque_b_nm[i], expected[static_cast<std::size_t>(i)]);
    }

    // The recorded dq0 documents which sign branch each case exercises; the
    // suite must actually cover the negative branch and the exact-zero
    // (sign(0) = +1) branch.
    const double dq0 = star_tests::parse_hex_double(c.scalar("dq0"));
    if (dq0 < 0.0) saw_negative_branch = true;
    if (dq0 == 0.0) saw_zero_branch = true;

    // Hold semantics: a missing estimate or command yields an invalid
    // (hold) output rather than a torque computed from garbage.
    GncInput no_est = in;
    no_est.nav_est.valid = false;
    CHECK_FALSE(pd->update(no_est).valid);
    GncInput no_cmd = in;
    no_cmd.att_cmd.valid = false;
    CHECK_FALSE(pd->update(no_cmd).valid);
  }
  CHECK(saw_negative_branch);
  CHECK(saw_zero_branch);
}

TEST_CASE("gnc_dead_reckoning_golden") {
  const std::vector<star_tests::GoldenCase> cases =
      star_tests::load_golden_cases(std::string(STAR_GOLDEN_DIR) +
                                    "/gnc/dead_reckoning.toml");
  REQUIRE(cases.size() == 1);
  const star_tests::GoldenCase& c = cases[0];

  GncComponentCfg cfg;
  cfg.component = "dead_reckoning";
  std::unique_ptr<star::gnc::IGncComponent> nav =
      star::gnc::make_component(cfg);
  REQUIRE(nav->state_dim() == 7);
  CHECK(nav->innov_max_dim() == 0);
  CHECK(nav->innovations().empty());

  GncInitContext ictx;
  ictx.q0_i2b = quat_of(golden_vec(c, "q0"));
  ictx.omega0_b_radps = Eigen::Vector3d::Zero();
  nav->init(ictx);

  const double dt = 0.1;
  for (int k = 0; k < 5; ++k) {
    GncInput in;
    in.dt_s = dt;
    in.imu_fresh = true;
    in.imu.valid = true;
    in.imu.dt_s = dt;
    in.imu.dtheta_b_rad =
        vec3_of(golden_vec(c, "dtheta_" + std::to_string(k)));
    const GncOutput out = nav->update(in);
    CHECK(out.valid);
    double x[7];
    nav->state(x);
    const std::vector<double> expected =
        golden_vec(c, "q_after_" + std::to_string(k));
    for (int i = 0; i < 4; ++i) {
      check_close(x[i], expected[static_cast<std::size_t>(i)]);
    }
    // The output's attitude equals the introspected state.
    CHECK(out.q_i2b.w() == x[0]);
    CHECK(out.q_i2b.x() == x[1]);
  }

  // Dead reckoning carries no covariance: P is identically zero.
  double p[28];
  nav->covariance_upper(p);
  for (double v : p) CHECK(v == 0.0);

  // error_state sign-aligns the truth quaternion: -q_hat encodes the same
  // attitude as q_hat, so the attitude error must vanish for both signs.
  double x[7];
  nav->state(x);
  star::gnc::TruthState truth;
  truth.valid = true;
  truth.q_i2b = Eigen::Quaterniond(-x[0], -x[1], -x[2], -x[3]);
  truth.omega_b_radps = Eigen::Vector3d(x[4], x[5], x[6]);
  double e[7];
  nav->error_state(truth, e);
  for (int i = 0; i < 7; ++i) CHECK(e[i] == 0.0);
}

TEST_CASE("gnc_pitch_program_guidance_matches_openloop_machinery") {
  // The closed-loop contract (gnc/builtin.hpp): the guidance component's
  // commanded attitude is computed by the same functions, in the same call
  // sequence, as the Phase 4 open-loop kPitchProgram mode - so the two are
  // bit-identical at equal cycle times. The reference below reproduces the
  // open-loop mode's arithmetic verbatim.
  const double az_deg = 90.0;
  const std::vector<double> t_tab = {0.0, 10.0, 25.0, 60.0};
  const std::vector<double> p_tab = {90.0, 90.0, 75.0, 40.0};

  GncComponentCfg cfg;
  cfg.component = "pitch_program";
  cfg.scalars["azimuth_deg"] = az_deg;
  cfg.vectors["pitch_t_s"] = t_tab;
  cfg.vectors["pitch_deg"] = p_tab;
  std::unique_ptr<star::gnc::IGncComponent> guidance =
      star::gnc::make_component(cfg);

  // An ENU basis representative of a mid-latitude pad (unit, orthogonal).
  GncInitContext ictx;
  ictx.pad_basis_valid = true;
  ictx.up_i = Eigen::Vector3d(0.6, 0.0, 0.8);
  ictx.east_i = Eigen::Vector3d(0.0, 1.0, 0.0);
  ictx.north_i = Eigen::Vector3d(-0.8, 0.0, 0.6);
  guidance->init(ictx);

  auto deg2rad = [](double d) {
    return d * (star::constants::TWO_PI / 360.0);
  };
  const double dt = 0.1;
  for (double t : {0.0, 2.0, 9.95, 10.0, 17.3, 24.999, 25.0, 40.0, 59.9,
                   60.0, 75.0}) {
    GncInput in;
    in.t_s = t;
    in.dt_s = dt;
    const GncOutput out = guidance->update(in);
    CHECK(out.valid);

    const double az = deg2rad(az_deg);
    const double p0 =
        deg2rad(star::models::pwl_interp_clamped(t_tab, p_tab, t));
    const double p1 =
        deg2rad(star::models::pwl_interp_clamped(t_tab, p_tab, t + dt));
    const Eigen::Quaterniond q0 = star::models::attitude_from_body_x(
        star::models::pitch_program_axis(az, p0, ictx.up_i, ictx.east_i,
                                         ictx.north_i),
        ictx.up_i);
    const Eigen::Quaterniond q1 = star::models::attitude_from_body_x(
        star::models::pitch_program_axis(az, p1, ictx.up_i, ictx.east_i,
                                         ictx.north_i),
        ictx.up_i);
    const Eigen::Vector3d w =
        star::models::omega_from_quaternions(q0, q1, dt);
    // Bit equality, not tolerance: same functions, same inputs, same order.
    CHECK(out.q_i2b.w() == q0.w());
    CHECK(out.q_i2b.x() == q0.x());
    CHECK(out.q_i2b.y() == q0.y());
    CHECK(out.q_i2b.z() == q0.z());
    CHECK(out.omega_b_radps[0] == w[0]);
    CHECK(out.omega_b_radps[1] == w[1]);
    CHECK(out.omega_b_radps[2] == w[2]);
  }

  // A free-flying init context has no ENU basis to resolve the commanded
  // axis in; the component refuses rather than guessing a frame.
  std::unique_ptr<star::gnc::IGncComponent> unpadded =
      star::gnc::make_component(cfg);
  CHECK_THROWS_AS(unpadded->init(GncInitContext{}), std::invalid_argument);
}

TEST_CASE("gnc_attitude_hold_guidance") {
  // Explicit target: normalized at construction, echoed on every update
  // with zero commanded rate.
  GncComponentCfg cfg;
  cfg.component = "attitude_hold";
  cfg.vectors["q_cmd"] = {2.0, 0.0, 0.0, 2.0};  // non-unit on purpose
  std::unique_ptr<star::gnc::IGncComponent> hold =
      star::gnc::make_component(cfg);
  hold->init(GncInitContext{});
  const GncOutput out = hold->update(GncInput{});
  CHECK(out.valid);
  const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
  CHECK(std::fabs(out.q_i2b.w() - inv_sqrt2) < 1e-15);
  CHECK(out.q_i2b.x() == 0.0);
  CHECK(std::fabs(out.q_i2b.z() - inv_sqrt2) < 1e-15);
  CHECK(out.omega_b_radps.norm() == 0.0);

  // Default target: the scenario initial attitude from the init context.
  GncComponentCfg dflt;
  dflt.component = "attitude_hold";
  std::unique_ptr<star::gnc::IGncComponent> hold0 =
      star::gnc::make_component(dflt);
  GncInitContext ictx;
  ictx.q0_i2b = Eigen::Quaterniond(0.5, 0.5, 0.5, 0.5);
  hold0->init(ictx);
  const GncOutput out0 = hold0->update(GncInput{});
  CHECK(out0.q_i2b.w() == 0.5);
  CHECK(out0.q_i2b.x() == 0.5);
  CHECK(out0.q_i2b.y() == 0.5);
  CHECK(out0.q_i2b.z() == 0.5);
}

TEST_CASE("gnc_latency_fifo_semantics") {
  GncOutput neutral;
  neutral.q_i2b = Eigen::Quaterniond(0.5, 0.5, 0.5, 0.5);
  neutral.torque_b_nm = Eigen::Vector3d::Zero();

  auto cmd = [](double tx) {
    GncOutput o;
    o.valid = true;
    o.torque_b_nm = Eigen::Vector3d(tx, 0.0, 0.0);
    return o;
  };

  // k = 0: pure passthrough - the output produced this cycle applies this
  // cycle.
  {
    star::gnc::LatencyFifo fifo(0, neutral);
    CHECK_FALSE(fifo.applied().valid);  // neutral until the first push
    const GncOutput a = fifo.push(cmd(1.0));
    CHECK(a.valid);
    CHECK(a.torque_b_nm[0] == 1.0);
    CHECK(fifo.applied().torque_b_nm[0] == 1.0);
  }

  // k = 2: application shifts by exactly two cycles; the two pre-fill holds
  // apply the neutral command with valid == false.
  {
    star::gnc::LatencyFifo fifo(2, neutral);
    const GncOutput a0 = fifo.push(cmd(1.0));
    CHECK_FALSE(a0.valid);
    CHECK(a0.torque_b_nm.norm() == 0.0);
    CHECK(a0.q_i2b.w() == neutral.q_i2b.w());  // held neutral attitude
    const GncOutput a1 = fifo.push(cmd(2.0));
    CHECK_FALSE(a1.valid);
    const GncOutput a2 = fifo.push(cmd(3.0));
    CHECK(a2.valid);
    CHECK(a2.torque_b_nm[0] == 1.0);  // the cycle-0 output, two cycles later
    const GncOutput a3 = fifo.push(cmd(4.0));
    CHECK(a3.valid);
    CHECK(a3.torque_b_nm[0] == 2.0);
  }

  // An invalid produced entry resolves at application time to the previous
  // applied command, flagged as a hold.
  {
    star::gnc::LatencyFifo fifo(0, neutral);
    (void)fifo.push(cmd(5.0));
    GncOutput invalid_cmd;  // valid == false
    const GncOutput held = fifo.push(invalid_cmd);
    CHECK_FALSE(held.valid);
    CHECK(held.torque_b_nm[0] == 5.0);  // ZOH of the last applied command
  }
}
