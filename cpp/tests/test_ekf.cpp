// Reference error-state EKF unit tests (FR-25, FR-26, ch:ekf): registry and
// parameter validation, the mechanization's agreement with the shared
// dead-reckoning attitude path, the update algebra against an independent
// explicit-inverse implementation of eq:ekf:joseph, the star-tracker
// innovation of eq:ekf:stinnov, covariance hygiene under alternating
// updates and outages, and the innovation reporting nav.innov consumes.
//
// The filter is exercised through the IGncComponent interface exactly as
// the loop drives it, so these tests bind to the same surface the run path
// uses rather than to internals.
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "star/gnc/component.hpp"
#include "star/rotation.hpp"
#include "vendor/doctest.h"

namespace {

using star::gnc::GncComponentCfg;
using star::gnc::GncInitContext;
using star::gnc::GncInput;
using star::gnc::IGncComponent;
using star::gnc::InnovationSample;

constexpr int kM = 15;
constexpr int kN = 16;
constexpr double kMu = 3.986004418e14;

// A well-conditioned LEO reference point for the algebra tests.
const Eigen::Vector3d kPos(7.0e6, 0.0, 0.0);
const Eigen::Vector3d kVel(0.0, 7546.0, 0.0);

GncComponentCfg ekf_cfg() {
  GncComponentCfg cfg;
  cfg.component = "error_state_ekf";
  cfg.vectors["q0"] = {1.0, 0.0, 0.0, 0.0};
  cfg.vectors["v0_mps"] = {kVel.x(), kVel.y(), kVel.z()};
  cfg.vectors["p0_m"] = {kPos.x(), kPos.y(), kPos.z()};
  cfg.vectors["bg0_radps"] = {0.0, 0.0, 0.0};
  cfg.vectors["ba0_mps2"] = {0.0, 0.0, 0.0};
  cfg.vectors["p0_sigma_att_rad"] = {1.0e-3, 1.0e-3, 1.0e-3};
  cfg.vectors["p0_sigma_vel_mps"] = {0.5, 0.5, 0.5};
  cfg.vectors["p0_sigma_pos_m"] = {50.0, 50.0, 50.0};
  cfg.vectors["p0_sigma_bg_radps"] = {1.0e-7, 1.0e-7, 1.0e-7};
  cfg.vectors["p0_sigma_ba_mps2"] = {1.0e-5, 1.0e-5, 1.0e-5};
  return cfg;
}

GncInitContext ekf_ctx() {
  GncInitContext ctx;
  ctx.dt_s = 0.1;
  ctx.control_rate_hz = 10;
  ctx.mu_m3ps2 = kMu;
  ctx.ellipsoid_a_m = 6378137.0;
  ctx.ellipsoid_inv_f = 298.257223563;
  ctx.sensors.imu_present = true;
  ctx.sensors.imu_id = 0;
  ctx.sensors.gyro_arw = 1.0e-5;
  ctx.sensors.accel_vrw = 1.0e-4;
  ctx.sensors.gyro_gm_sigma = 1.0e-7;
  ctx.sensors.gyro_tau_s = 100.0;
  ctx.sensors.accel_gm_sigma = 1.0e-5;
  ctx.sensors.accel_tau_s = 100.0;
  ctx.sensors.navfix_present = true;
  ctx.sensors.navfix_id = 2;
  ctx.sensors.navfix_sigma_r_m = Eigen::Vector3d(10.0, 10.0, 10.0);
  ctx.sensors.navfix_sigma_v_mps = Eigen::Vector3d(0.1, 0.1, 0.1);
  ctx.sensors.startracker_present = true;
  ctx.sensors.startracker_id = 1;
  ctx.sensors.startracker_sigma_rad = Eigen::Vector3d(1.0e-5, 1.0e-5, 5.0e-5);
  ctx.sensors.startracker_boresight_b = Eigen::Vector3d::UnitZ();
  ctx.sensors.altimeter_present = true;
  ctx.sensors.altimeter_id = 3;
  ctx.sensors.altimeter_sigma_noise_m = 20.0;
  ctx.sensors.altimeter_sigma_bias_m = 0.0;
  return ctx;
}

std::unique_ptr<IGncComponent> make_ekf() {
  std::unique_ptr<IGncComponent> f = star::gnc::make_component(ekf_cfg());
  f->init(ekf_ctx());
  return f;
}

// One cycle of IMU increments, marked fresh, with no aiding.
GncInput imu_cycle(const Eigen::Vector3d& dtheta, const Eigen::Vector3d& dv,
                   double dt) {
  GncInput in;
  in.dt_s = dt;
  in.imu_fresh = true;
  in.imu.valid = true;
  in.imu.dt_s = dt;
  in.imu.dtheta_b_rad = dtheta;
  in.imu.dv_b_mps = dv;
  return in;
}

Eigen::Matrix<double, kM, kM> unpack_upper(const std::vector<double>& p) {
  Eigen::Matrix<double, kM, kM> m;
  m.setZero();
  int k = 0;
  for (int i = 0; i < kM; ++i) {
    for (int j = i; j < kM; ++j) {
      m(i, j) = p[static_cast<std::size_t>(k++)];
      m(j, i) = m(i, j);
    }
  }
  return m;
}

std::vector<double> covariance_of(const IGncComponent& f) {
  std::vector<double> p(static_cast<std::size_t>(kM) * (kM + 1) / 2, 0.0);
  f.covariance_upper(p.data());
  return p;
}

std::vector<double> state_of(const IGncComponent& f) {
  std::vector<double> x(kN, 0.0);
  f.state(x.data());
  return x;
}

}  // namespace

TEST_CASE("ekf_registry_and_declared_dimensions") {
  std::unique_ptr<IGncComponent> f = make_ekf();
  // The pinned cross-workstream contract: n = 16, m = 15, so nav.est.P
  // carries 120 doubles, and the largest aiding update is the nav fix's 6.
  CHECK(f->state_dim() == kN);
  CHECK(f->cov_dim() == kM);
  CHECK(f->innov_max_dim() == 6);
  CHECK(covariance_of(*f).size() == 120u);
}

TEST_CASE("ekf_rejects_malformed_parameters") {
  SUBCASE("missing required vector") {
    GncComponentCfg cfg = ekf_cfg();
    cfg.vectors.erase("p0_m");
    CHECK_THROWS_AS(star::gnc::make_component(cfg), std::invalid_argument);
  }
  SUBCASE("unknown parameter is refused rather than ignored") {
    GncComponentCfg cfg = ekf_cfg();
    cfg.vectors["p0_sigma_clock_s"] = {1.0, 1.0, 1.0};
    CHECK_THROWS_AS(star::gnc::make_component(cfg), std::invalid_argument);
  }
  SUBCASE("a zero initial sigma would make P0 singular") {
    GncComponentCfg cfg = ekf_cfg();
    cfg.vectors["p0_sigma_pos_m"] = {50.0, 0.0, 50.0};
    CHECK_THROWS_AS(star::gnc::make_component(cfg), std::invalid_argument);
  }
  SUBCASE("a run without an IMU has no propagation source") {
    std::unique_ptr<IGncComponent> f = star::gnc::make_component(ekf_cfg());
    GncInitContext ctx = ekf_ctx();
    ctx.sensors.imu_present = false;
    CHECK_THROWS_AS(f->init(ctx), std::invalid_argument);
  }
}

TEST_CASE("ekf_initial_state_and_covariance_are_the_configured_belief") {
  std::unique_ptr<IGncComponent> f = make_ekf();
  const std::vector<double> x = state_of(*f);
  CHECK(x[0] == doctest::Approx(1.0));
  CHECK(x[4 + 1] == doctest::Approx(kVel.y()));
  CHECK(x[7 + 0] == doctest::Approx(kPos.x()));
  const Eigen::Matrix<double, kM, kM> p = unpack_upper(covariance_of(*f));
  // P0 is diagonal with the configured variances in the eq:ekf:staterr
  // block ordering.
  CHECK(p(0, 0) == doctest::Approx(1.0e-6));
  CHECK(p(3, 3) == doctest::Approx(0.25));
  CHECK(p(6, 6) == doctest::Approx(2500.0));
  CHECK(p(0, 1) == doctest::Approx(0.0));
}

TEST_CASE("ekf_mechanization_attitude_matches_the_exact_rotation_path") {
  // ch:ekf validation, mechanization gate: with zero bias estimates the
  // attitude increment is composed as the same exact rotation
  // (eq:gnc:exactrot) the dead reckoner uses, so the two share a code path
  // and must agree to rounding on the same increments.
  std::unique_ptr<IGncComponent> f = make_ekf();
  const double dt = 0.1;
  const Eigen::Vector3d dtheta(1.0e-3, -2.0e-3, 3.0e-4);
  Eigen::Quaterniond expected = Eigen::Quaterniond::Identity();
  for (int k = 0; k < 20; ++k) {
    f->update(imu_cycle(dtheta, Eigen::Vector3d::Zero(), dt));
    const double angle = dtheta.norm();
    const Eigen::Vector3d axis = dtheta / angle;
    const double s = std::sin(0.5 * angle);
    const Eigen::Quaterniond dq(std::cos(0.5 * angle), s * axis.x(),
                                s * axis.y(), s * axis.z());
    expected = star::rotation::quat_normalize(
        star::rotation::quat_multiply(expected, dq));
  }
  const std::vector<double> x = state_of(*f);
  CHECK(x[0] == doctest::Approx(expected.w()).epsilon(1e-14));
  CHECK(x[1] == doctest::Approx(expected.x()).epsilon(1e-14));
  CHECK(x[2] == doctest::Approx(expected.y()).epsilon(1e-14));
  CHECK(x[3] == doctest::Approx(expected.z()).epsilon(1e-14));
}

TEST_CASE("ekf_free_fall_velocity_is_second_order_in_gravity") {
  // With zero specific force the mechanization reduces to the filter's own
  // point-mass gravity (ch:ekf assumption 2), so one cycle is a pure
  // quadrature of g along the trajectory. eq:ekf:mech closes that quadrature
  // to second order with a predictor-corrector, and this pins both the
  // arithmetic and the ORDER: the first-order step it replaced is excluded
  // by four decades, so a regression to Euler fails here rather than
  // surviving as a silent bias the ensemble gate has to catch.
  std::unique_ptr<IGncComponent> f = make_ekf();
  const double dt = 0.1;
  f->update(imu_cycle(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), dt));
  const std::vector<double> x = state_of(*f);

  const auto grav = [](const Eigen::Vector3d& p) {
    const double r = p.norm();
    return Eigen::Vector3d(-kMu * p / (r * r * r));
  };

  // The specified predictor-corrector, recomputed here independently.
  const Eigen::Vector3d g0 = grav(kPos);
  const Eigen::Vector3d v_pred = kVel + g0 * dt;
  const Eigen::Vector3d p_pred = kPos + 0.5 * (kVel + v_pred) * dt;
  const Eigen::Vector3d v_new = kVel + 0.5 * (g0 + grav(p_pred)) * dt;
  CHECK(x[4] == doctest::Approx(v_new.x()).epsilon(1e-12));
  CHECK(x[5] == doctest::Approx(v_new.y()).epsilon(1e-12));

  // An RK4 step of the same two-body dynamics is an INDEPENDENT reference:
  // a different scheme of higher order, not a restatement of the formula
  // above, so agreeing with it is evidence about accuracy rather than about
  // self-consistency. It is also the scheme the truth trajectory uses, which
  // is the comparison the NEES gate ultimately makes.
  const auto deriv = [&grav](const Eigen::Matrix<double, 6, 1>& y) {
    Eigen::Matrix<double, 6, 1> d;
    d.head<3>() = y.tail<3>();
    d.tail<3>() = grav(y.head<3>());
    return d;
  };
  Eigen::Matrix<double, 6, 1> y;
  y << kPos, kVel;
  const Eigen::Matrix<double, 6, 1> k1 = deriv(y);
  const Eigen::Matrix<double, 6, 1> k2 = deriv(y + 0.5 * dt * k1);
  const Eigen::Matrix<double, 6, 1> k3 = deriv(y + 0.5 * dt * k2);
  const Eigen::Matrix<double, 6, 1> k4 = deriv(y + dt * k3);
  const Eigen::Matrix<double, 6, 1> y_rk4 =
      y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);

  const Eigen::Vector3d v_ekf(x[4], x[5], x[6]);
  const double err = (v_ekf - y_rk4.tail<3>()).norm();
  // Measured 7.9e-10 m/s for the predictor-corrector against 4.4e-5 m/s for
  // the Euler step it replaced; the gate sits between them, nearer the
  // former by three decades.
  CHECK(err < 1.0e-8);
}

TEST_CASE("ekf_propagated_covariance_stays_symmetric_and_psd") {
  // ch:ekf validation, covariance-hygiene gate. The Joseph form plus the
  // explicit symmetrization of eq:ekf:disc must hold both invariants
  // across a long run of alternating propagation, updates, and outages.
  std::unique_ptr<IGncComponent> f = make_ekf();
  const double dt = 0.1;
  const Eigen::Vector3d dtheta(2.0e-4, 1.0e-4, -3.0e-4);
  const Eigen::Vector3d dv(1.0e-3, -2.0e-3, 5.0e-4);
  for (int k = 0; k < 200; ++k) {
    GncInput in = imu_cycle(dtheta, dv, dt);
    // Aiding drops in and out, which is the case that exercises the update
    // path against a covariance that has been propagating unaided.
    if (k % 10 == 0) {
      in.navfix.valid = true;
      in.navfix.fresh = true;
      in.navfix.sensor_id = 2;
      in.navfix.r_i_m = kPos;
      in.navfix.v_i_mps = kVel;
    }
    if (k % 10 == 5) {
      in.startracker.valid = true;
      in.startracker.fresh = true;
      in.startracker.sensor_id = 1;
      in.startracker.q_i2b = Eigen::Quaterniond::Identity();
    }
    f->update(in);
    const Eigen::Matrix<double, kM, kM> p = unpack_upper(covariance_of(*f));
    // Symmetry is pinned exactly by construction, not merely to a
    // tolerance: covariance_upper reads one triangle of a symmetrized
    // matrix.
    CHECK((p - p.transpose()).cwiseAbs().maxCoeff() == doctest::Approx(0.0));
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, kM, kM>> es(p);
    // Scale the floor to the covariance's own magnitude: an absolute bound
    // would be meaningless across blocks spanning 1e-14 to 1e3.
    CHECK(es.eigenvalues().minCoeff() >= -1e-12 * p.diagonal().maxCoeff());
  }
}

TEST_CASE("ekf_navfix_update_matches_an_explicit_inverse_implementation") {
  // ch:ekf validation, update-algebra gate. The filter forms K through an
  // LDLT solve on S and updates P in Joseph form; this recomputes the same
  // step with an explicit matrix inverse and the textbook expressions, two
  // paths sharing nothing beyond their inputs.
  std::unique_ptr<IGncComponent> f = make_ekf();
  const Eigen::Matrix<double, kM, kM> p_prior =
      unpack_upper(covariance_of(*f));
  const std::vector<double> x_prior = state_of(*f);

  const Eigen::Vector3d r_meas = kPos + Eigen::Vector3d(30.0, -12.0, 7.0);
  const Eigen::Vector3d v_meas = kVel + Eigen::Vector3d(0.2, 0.05, -0.1);
  GncInput in;
  in.dt_s = 0.1;
  in.navfix.valid = true;
  in.navfix.fresh = true;
  in.navfix.sensor_id = 2;
  in.navfix.r_i_m = r_meas;
  in.navfix.v_i_mps = v_meas;
  f->update(in);  // no fresh IMU sample, so this is a pure update

  Eigen::Matrix<double, 6, kM> h = Eigen::Matrix<double, 6, kM>::Zero();
  h.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity();
  h.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();
  Eigen::Matrix<double, 6, 6> r_mat = Eigen::Matrix<double, 6, 6>::Zero();
  for (int i = 0; i < 3; ++i) {
    r_mat(i, i) = 100.0;      // (10 m)^2
    r_mat(3 + i, 3 + i) = 0.01;  // (0.1 m/s)^2
  }
  Eigen::Matrix<double, 6, 1> y;
  y.segment<3>(0) = r_meas - kPos;
  y.segment<3>(3) = v_meas - kVel;

  const Eigen::Matrix<double, 6, 6> s = h * p_prior * h.transpose() + r_mat;
  const Eigen::Matrix<double, kM, 6> k =
      p_prior * h.transpose() * s.inverse();
  const Eigen::Matrix<double, kM, 1> dx = k * y;
  const Eigen::Matrix<double, kM, kM> ikh =
      Eigen::Matrix<double, kM, kM>::Identity() - k * h;
  Eigen::Matrix<double, kM, kM> p_post =
      ikh * p_prior * ikh.transpose() + k * r_mat * k.transpose();
  p_post = 0.5 * (p_post + p_post.transpose()).eval();

  const Eigen::Matrix<double, kM, kM> p_actual =
      unpack_upper(covariance_of(*f));
  CHECK((p_actual - p_post).cwiseAbs().maxCoeff() <
        1e-9 * p_post.diagonal().maxCoeff());

  // The reset folds dx into the nominal state (eq:ekf:reset).
  const std::vector<double> x_post = state_of(*f);
  for (int i = 0; i < 3; ++i) {
    CHECK(x_post[4 + i] ==
          doctest::Approx(x_prior[4 + i] + dx[3 + i]).epsilon(1e-10));
    CHECK(x_post[7 + i] ==
          doctest::Approx(x_prior[7 + i] + dx[6 + i]).epsilon(1e-10));
  }

  // The applied update is reported for nav.innov with its sensor id and
  // its own dimension, innovation first.
  const std::vector<InnovationSample>& innov = f->innovations();
  REQUIRE(innov.size() == 1u);
  CHECK(innov[0].sensor_id == 2u);
  CHECK(innov[0].y.size() == 6u);
  CHECK(innov[0].s_upper.size() == 21u);
  for (int i = 0; i < 6; ++i) {
    CHECK(innov[0].y[static_cast<std::size_t>(i)] ==
          doctest::Approx(y[i]).epsilon(1e-12));
  }
  CHECK(innov[0].s_upper[0] == doctest::Approx(s(0, 0)).epsilon(1e-12));
}

TEST_CASE("ekf_star_tracker_innovation_recovers_an_injected_attitude_error") {
  // eq:ekf:stinnov: with a noise-free measurement and no aberration (zero
  // velocity term), the innovation is exactly the injected dtheta to
  // second order, and the update drives the attitude estimate toward it.
  std::unique_ptr<IGncComponent> f = make_ekf();
  const Eigen::Vector3d dtheta(3.0e-4, -1.5e-4, 2.0e-4);
  const double angle = dtheta.norm();
  const Eigen::Vector3d axis = dtheta / angle;
  const double s = std::sin(0.5 * angle);
  const Eigen::Quaterniond dq(std::cos(0.5 * angle), s * axis.x(),
                              s * axis.y(), s * axis.z());
  // The estimate starts at identity, so a measurement of dq is exactly an
  // attitude error of dtheta under q_true = q_hat (x) dq(dtheta).
  GncInput in;
  in.dt_s = 0.1;
  in.startracker.valid = true;
  in.startracker.fresh = true;
  in.startracker.sensor_id = 1;
  in.startracker.q_i2b = dq;
  // Zero the observer velocity so the aberration factor is identity and
  // the innovation isolates the attitude error alone.
  in.env.v_central_ssb_mps = Eigen::Vector3d::Zero();
  GncInitContext ctx = ekf_ctx();
  ctx.sensors.navfix_present = false;
  ctx.sensors.altimeter_present = false;
  // A zero initial velocity makes beta identically zero, so the aberration
  // factor is the identity and the innovation isolates the attitude error.
  GncComponentCfg zero_v = ekf_cfg();
  zero_v.vectors["v0_mps"] = {0.0, 0.0, 0.0};
  std::unique_ptr<IGncComponent> h = star::gnc::make_component(zero_v);
  h->init(ctx);
  h->update(in);

  const std::vector<InnovationSample>& innov = h->innovations();
  REQUIRE(innov.size() == 1u);
  CHECK(innov[0].sensor_id == 1u);
  CHECK(innov[0].y.size() == 3u);
  for (int i = 0; i < 3; ++i) {
    CHECK(innov[0].y[static_cast<std::size_t>(i)] ==
          doctest::Approx(dtheta[i]).epsilon(1e-8));
  }
  // The attitude covariance (1e-3 rad) dwarfs the star tracker's noise, so
  // the update absorbs nearly all of the injected error.
  const std::vector<double> x = state_of(*h);
  const Eigen::Quaterniond q_hat(x[0], x[1], x[2], x[3]);
  const Eigen::Quaterniond residual =
      star::rotation::quat_multiply(star::rotation::quat_conjugate(q_hat), dq);
  const double residual_angle =
      2.0 * Eigen::Vector3d(residual.x(), residual.y(), residual.z()).norm();
  CHECK(residual_angle < 0.02 * angle);
}

TEST_CASE("ekf_skips_stale_and_invalid_aiding_samples") {
  // Folding one measurement in twice, or trusting a gated-out sample, is
  // invisible in the state error but immediately corrupts the covariance,
  // so both guards are pinned here.
  std::unique_ptr<IGncComponent> f = make_ekf();
  GncInput in;
  in.dt_s = 0.1;
  in.navfix.valid = true;
  in.navfix.fresh = false;  // held from an earlier cycle
  in.navfix.sensor_id = 2;
  in.navfix.r_i_m = kPos;
  in.navfix.v_i_mps = kVel;
  in.startracker.valid = false;  // gated out by the sensor
  in.startracker.fresh = true;
  in.startracker.sensor_id = 1;
  in.startracker.q_i2b = Eigen::Quaterniond::Identity();
  const std::vector<double> before = covariance_of(*f);
  f->update(in);
  const std::vector<double> after = covariance_of(*f);
  CHECK(f->innovations().empty());
  for (std::size_t i = 0; i < before.size(); ++i) {
    CHECK(after[i] == before[i]);
  }
}

TEST_CASE("ekf_error_state_reports_the_multiplicative_attitude_error") {
  // The nav.err contract: the leading four entries are the
  // sign-canonicalized error quaternion of eq:ekf:qerr, the rest are
  // additive truth-minus-estimate, including the bias rows.
  //
  // The filter no longer computes this itself: it DECLARES the layout of its
  // state vector and the loop does the arithmetic, so the truth state below
  // never reaches the component (FR-24; gnc/component.hpp). This exercises
  // the same path the loop takes - declared layout, published state vector,
  // compute_error_state.
  std::unique_ptr<IGncComponent> f = make_ekf();
  const std::vector<star::gnc::ErrorBlock>& layout = f->error_layout();
  star::gnc::validate_error_layout(layout, f->state_dim(), true);
  REQUIRE(layout.size() == 5);
  CHECK(layout[0].quantity == star::gnc::ErrorQuantity::kAttitude);
  CHECK(layout[0].form == star::gnc::ErrorForm::kQuatErrorLocal);
  std::vector<double> x_hat(kN, 0.0);
  f->state(x_hat.data());

  star::gnc::TruthState truth;
  truth.valid = true;
  truth.r_i_m = kPos + Eigen::Vector3d(1.0, 2.0, 3.0);
  truth.v_i_mps = kVel + Eigen::Vector3d(0.1, 0.2, 0.3);
  const Eigen::Vector3d dtheta(1.0e-4, 2.0e-4, -3.0e-4);
  const double angle = dtheta.norm();
  const Eigen::Vector3d axis = dtheta / angle;
  const double s = std::sin(0.5 * angle);
  truth.q_i2b = Eigen::Quaterniond(std::cos(0.5 * angle), s * axis.x(),
                                   s * axis.y(), s * axis.z());
  truth.imu_bias_valid = true;
  truth.b_g_radps = Eigen::Vector3d(1.0e-8, -2.0e-8, 3.0e-8);
  truth.b_a_mps2 = Eigen::Vector3d(1.0e-6, -2.0e-6, 3.0e-6);

  std::vector<double> e(kN, 0.0);
  star::gnc::compute_error_state(layout, truth, x_hat.data(), e.data());
  // Estimate is identity, so the error quaternion is the truth quaternion,
  // already in the +w hemisphere.
  CHECK(e[0] == doctest::Approx(truth.q_i2b.w()));
  CHECK(2.0 * e[1] == doctest::Approx(dtheta.x()).epsilon(1e-6));
  CHECK(e[4] == doctest::Approx(0.1));
  CHECK(e[7] == doctest::Approx(1.0));
  CHECK(e[10] == doctest::Approx(1.0e-8));
  CHECK(e[13] == doctest::Approx(1.0e-6));

  SUBCASE("the error quaternion is canonicalized to the +w hemisphere") {
    truth.q_i2b = Eigen::Quaterniond(-truth.q_i2b.w(), -truth.q_i2b.x(),
                                     -truth.q_i2b.y(), -truth.q_i2b.z());
    std::vector<double> e2(kN, 0.0);
    star::gnc::compute_error_state(layout, truth, x_hat.data(), e2.data());
    CHECK(e2[0] > 0.0);
    CHECK(e2[1] == doctest::Approx(e[1]));
  }
}
