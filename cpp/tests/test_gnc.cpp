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
#include "star/rotation.hpp"
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
  dr.vectors["q0"] = {1.0, 0.0, 0.0, 0.0};
  dr.scalars["mystery_knob"] = 1.0;  // unknown parameter
  CHECK_THROWS_AS(star::gnc::make_component(dr), std::invalid_argument);
  // The initial estimate is configuration, stated explicitly - a missing q0
  // is rejected rather than silently defaulted from truth (ch:gnc-builtin:
  // no implicit truth access).
  GncComponentCfg dr_no_q0;
  dr_no_q0.component = "dead_reckoning";
  CHECK_THROWS_AS(star::gnc::make_component(dr_no_q0), std::invalid_argument);
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
  // The initial estimate is a configured parameter (no implicit truth
  // access, ch:gnc-builtin); the golden's q0 rides in through the config.
  cfg.vectors["q0"] = golden_vec(c, "q0");
  std::unique_ptr<star::gnc::IGncComponent> nav =
      star::gnc::make_component(cfg);
  REQUIRE(nav->state_dim() == 7);
  CHECK(nav->cov_dim() == 7);  // covariance dimension defaults to the state
  CHECK(nav->innov_max_dim() == 0);
  CHECK(nav->innovations().empty());

  nav->init(GncInitContext{});

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

  // The declared error layout (FR-24): the loop, not the component, forms
  // nav.err. The attitude block is the sign-aligned additive form, so -q_hat
  // encodes the same attitude as q_hat and the error must vanish for both
  // signs.
  double x[7];
  nav->state(x);
  const std::vector<star::gnc::ErrorBlock>& layout = nav->error_layout();
  REQUIRE(layout.size() == 2);
  CHECK(layout[0].quantity == star::gnc::ErrorQuantity::kAttitude);
  CHECK(layout[0].form == star::gnc::ErrorForm::kQuatDifferenceAligned);
  CHECK(layout[0].offset == 0);
  CHECK(layout[1].quantity == star::gnc::ErrorQuantity::kAngularRate);
  CHECK(layout[1].offset == 4);
  star::gnc::validate_error_layout(layout, nav->state_dim(), false);

  star::gnc::TruthState truth;
  truth.valid = true;
  truth.q_i2b = Eigen::Quaterniond(-x[0], -x[1], -x[2], -x[3]);
  truth.omega_b_radps = Eigen::Vector3d(x[4], x[5], x[6]);
  double e[7];
  star::gnc::compute_error_state(layout, truth, x, e);
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
        star::models::pitch_program_roll_ref(az, p0, ictx.up_i, ictx.east_i,
                                             ictx.north_i));
    const Eigen::Quaterniond q1 = star::models::attitude_from_body_x(
        star::models::pitch_program_axis(az, p1, ictx.up_i, ictx.east_i,
                                         ictx.north_i),
        star::models::pitch_program_roll_ref(az, p1, ictx.up_i, ictx.east_i,
                                             ictx.north_i));
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

TEST_CASE("gnc_latency_fifo_full_history_shift") {
  // Exit criterion 8 over the WHOLE history, not one cycle pair.
  //
  // The mission-level gates can only compare cycle 0 between a k = 0 and a
  // k = ell run: the loop is closed, so a delayed torque changes the plant
  // trajectory and from cycle 1 onward the two runs' COMPUTED commands
  // genuinely differ. The criterion's claim - that application is the k = 0
  // history delayed by exactly k cycles - is a statement about the FIFO,
  // and it is only checkable where the produced sequence is held fixed.
  // This drives the FIFO open loop with a recorded sequence and asserts
  // applied[i + k] == produced[i] for every i, including the drain at the
  // end of the run, which no committed test covered.
  GncOutput neutral;
  neutral.q_i2b = Eigen::Quaterniond(0.5, 0.5, 0.5, 0.5);
  neutral.torque_b_nm = Eigen::Vector3d::Zero();

  const int n = 20;
  for (std::uint32_t k = 0; k <= 4; ++k) {
    CAPTURE(k);
    star::gnc::LatencyFifo fifo(k, neutral);
    std::vector<GncOutput> produced;
    std::vector<GncOutput> applied;
    for (int i = 0; i < n; ++i) {
      GncOutput o;
      o.valid = true;
      // Distinct per cycle on all three axes, so an off-by-one shift cannot
      // be masked by two neighbouring commands that happen to agree.
      o.torque_b_nm = Eigen::Vector3d(1.0 + i, 100.0 - i, 2.0 * i + 0.5);
      produced.push_back(o);
      applied.push_back(fifo.push(o));
    }
    // The first k applications are pre-fill holds, flagged invalid.
    for (std::uint32_t i = 0; i < k; ++i) {
      CHECK_FALSE(applied[i].valid);
    }
    // Every later application is the command produced exactly k cycles
    // earlier, on every axis.
    for (int i = 0; i + static_cast<int>(k) < n; ++i) {
      CAPTURE(i);
      CHECK(applied[i + static_cast<int>(k)].valid);
      for (int a = 0; a < 3; ++a) {
        CHECK(applied[i + static_cast<int>(k)].torque_b_nm[a] ==
              produced[i].torque_b_nm[a]);
      }
    }
    // The drain: the last k produced commands are never applied. Asserting
    // this pins the depth from the other side - a FIFO one entry too
    // shallow would have applied produced[n - k] somewhere.
    for (int i = n - static_cast<int>(k); i < n; ++i) {
      for (const GncOutput& a : applied) {
        CHECK(a.torque_b_nm[0] != produced[i].torque_b_nm[0]);
      }
    }
  }
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

TEST_CASE("gnc_base_component_accessors_refuse_an_undeclared_state") {
  // The base-class contract guards. A component that declares an estimator
  // state but does not override the accessors must fail loudly, because the
  // alternative is the loop reading a buffer nothing wrote and logging it as
  // an estimate.
  // Fixture non-degeneracy: the probe DOES declare state_dim() == 3, so the
  // guards are reached along the path a real misuse takes - the loop sizes a
  // buffer from the declaration and then calls an accessor the author forgot
  // to supply. A probe declaring zero would never reach them.
  struct Bare final : star::gnc::IGncComponent {
    void init(const GncInitContext&) override {}
    GncOutput update(const GncInput&) override { return GncOutput(); }
    int state_dim() const override { return 3; }
  };
  Bare c;
  double buf[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  CHECK_THROWS_AS(c.state(buf), std::logic_error);
  CHECK_THROWS_AS(c.covariance_upper(buf), std::logic_error);

  // The rest of the default introspection is well defined and empty, which
  // is what makes the two throws above a deliberate contract rather than a
  // gap in the base class.
  CHECK(c.cov_dim() == 3);  // defaults to the declared state dimension
  CHECK(c.innov_max_dim() == 0);
  CHECK(c.innovations().empty());
  CHECK(c.error_layout().empty());
}

TEST_CASE("gnc_rotation_vector_error_forms_match_the_analytic_reduction") {
  // The rotation-vector attitude error forms, whose write had never been
  // computed in either tier. The expected value is the closed form
  // 2 sgn(dq_w) dq_v of a rotation constructed here, not a regenerated
  // golden: for a rotation of theta about a unit axis, dq_v = sin(theta/2) u
  // exactly, so the reduction is 2 sin(theta/2) u.
  //
  // These cases call compute_error_state directly rather than declaring the
  // layout through validate_error_layout. compute_error_state reads FOUR
  // quaternion slots at the block offset, while error_block_size reports
  // THREE slots for these two forms, so a layout that passes validation
  // cannot supply the fourth slot from within the block. What is pinned here
  // is the arithmetic the function performs on the quaternion it is given.
  const double theta = 1.0e-3;
  const Eigen::Vector3d axis = Eigen::Vector3d(1.0, -2.0, 2.0).normalized();
  const double s = std::sin(0.5 * theta);
  const Eigen::Quaterniond dq(std::cos(0.5 * theta), s * axis.x(),
                              s * axis.y(), s * axis.z());
  // A non-identity estimate, so the two composition sides are genuinely
  // different rotations and neither can pass by accident.
  const Eigen::Quaterniond q_est(0.5, 0.5, -0.5, 0.5);
  const double x_hat[4] = {q_est.w(), q_est.x(), q_est.y(), q_est.z()};

  star::gnc::TruthState truth;
  truth.valid = true;

  // Local: dq = conj(q_est) (x) q_true, so a truth built as q_est (x) dq
  // recovers dq exactly and the expected reduction is analytic.
  {
    const std::vector<star::gnc::ErrorBlock> layout = {
        {star::gnc::ErrorQuantity::kAttitude,
         star::gnc::ErrorForm::kRotationVectorLocal, 0}};
    truth.q_i2b = star::rotation::quat_multiply(q_est, dq);
    double e[3] = {0.0, 0.0, 0.0};
    star::gnc::compute_error_state(layout, truth, x_hat, e);
    for (int i = 0; i < 3; ++i) {
      CHECK(e[i] == doctest::Approx(2.0 * s * axis[i]).epsilon(1e-13));
    }
    // Non-vacuous: at theta = 1e-3 the three components are 6.7e-4, -1.3e-3
    // and 1.3e-3, so a reduction that dropped the factor of 2 or took dq_w
    // instead of dq_v would miss by orders of magnitude, and a sign error on
    // any axis flips a component whose sign differs from its neighbours'.
    CHECK(e[0] > 0.0);
    CHECK(e[1] < 0.0);
    CHECK(e[2] > 0.0);

    // The +w canonicalization the reduction depends on: negating the truth
    // quaternion names the same attitude, so the reported error must not
    // change. Without the canonicalization every component would flip.
    star::gnc::TruthState flipped = truth;
    flipped.q_i2b = Eigen::Quaterniond(-truth.q_i2b.w(), -truth.q_i2b.x(),
                                       -truth.q_i2b.y(), -truth.q_i2b.z());
    double e_flipped[3] = {0.0, 0.0, 0.0};
    star::gnc::compute_error_state(layout, flipped, x_hat, e_flipped);
    for (int i = 0; i < 3; ++i) {
      CHECK(e_flipped[i] == doctest::Approx(e[i]).epsilon(1e-13));
    }
  }

  // Global: dq = q_true (x) conj(q_est), so the truth is built on the other
  // side. Using the SAME dq makes the two cases differ only in composition
  // side, which is the convention this form exists to carry.
  {
    const std::vector<star::gnc::ErrorBlock> layout = {
        {star::gnc::ErrorQuantity::kAttitude,
         star::gnc::ErrorForm::kRotationVectorGlobal, 0}};
    truth.q_i2b = star::rotation::quat_multiply(dq, q_est);
    double e[3] = {0.0, 0.0, 0.0};
    star::gnc::compute_error_state(layout, truth, x_hat, e);
    for (int i = 0; i < 3; ++i) {
      CHECK(e[i] == doctest::Approx(2.0 * s * axis[i]).epsilon(1e-13));
    }

    // The composition side is real: reading the SAME globally-composed truth
    // through the local form gives a different rotation axis. q_est is a
    // 120 deg rotation, so the two reductions are far apart rather than
    // marginally so.
    const std::vector<star::gnc::ErrorBlock> local_layout = {
        {star::gnc::ErrorQuantity::kAttitude,
         star::gnc::ErrorForm::kRotationVectorLocal, 0}};
    double e_local[3] = {0.0, 0.0, 0.0};
    star::gnc::compute_error_state(local_layout, truth, x_hat, e_local);
    double diff = 0.0;
    for (int i = 0; i < 3; ++i) diff += std::fabs(e_local[i] - e[i]);
    CHECK(diff > 0.5 * theta);
    // Both sides describe the same rotation ANGLE, only resolved in
    // different frames, so the reduction's magnitude is preserved.
    const double n_local = std::sqrt(e_local[0] * e_local[0] +
                                     e_local[1] * e_local[1] +
                                     e_local[2] * e_local[2]);
    const double n_global =
        std::sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
    CHECK(n_local == doctest::Approx(n_global).epsilon(1e-12));
  }

  // The declared block width for both forms is three slots, against four for
  // every quaternion form; this is what a component author selects by, and
  // it is the value validate_error_layout tiles the state vector with.
  CHECK(star::gnc::error_block_size(
            star::gnc::ErrorQuantity::kAttitude,
            star::gnc::ErrorForm::kRotationVectorLocal) == 3);
  CHECK(star::gnc::error_block_size(
            star::gnc::ErrorQuantity::kAttitude,
            star::gnc::ErrorForm::kRotationVectorGlobal) == 3);
  CHECK(star::gnc::error_block_size(
            star::gnc::ErrorQuantity::kAttitude,
            star::gnc::ErrorForm::kQuatErrorLocal) == 4);
}
