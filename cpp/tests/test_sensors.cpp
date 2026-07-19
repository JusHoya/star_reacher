// FR-23 sensor-layer unit tests: the IMU's exact accumulation of the loop's
// held per-cycle kinematics, its sample/reset semantics, the error chain of
// eq:imu:gyro--eq:imu:quant, and the sensor factory's kind vocabulary.
// Written against the contracts in sensors/imu.hpp; record-level byte layout
// is covered by test_srlog.cpp, and in-loop scheduling by test_gnc_cycle.cpp.
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

#include <Eigen/Dense>

#include "star/constants.hpp"
#include "star/gnc/config.hpp"
#include "star/rng.hpp"
#include "star/sensors/camera.hpp"
#include "star/sensors/imu.hpp"
#include "star/sensors/optical.hpp"
#include "star/sensors/radio.hpp"
#include "star/sensors/sensor.hpp"
#include "star/srlog_writer.hpp"
#include "vendor/doctest.h"

namespace {

// A writer with only sensors.imu declared, for sample() calls that need a
// record sink; the emitted bytes themselves are covered by test_srlog.cpp.
star::log::SrlogWriter make_imu_writer(const std::string& path,
                                       std::uint32_t rate_hz = 10) {
  star::log::SrlogHeaderFields f;
  f.core_version = "0.6.0-test";
  f.git_hash = "unknown";
  f.config_sha256 = std::string(64, '0');
  f.master_seed = 1;
  f.oracle = false;
  f.epoch_utc = "2026-01-01T00:00:00Z";
  f.central_body = "earth";
  f.truth_rate_hz = rate_hz;
  f.cycle_rate_hz = rate_hz;
  f.sensors = {{"imu", rate_hz, 0}};
  return star::log::SrlogWriter(path, f);
}

// One cycle of held truth: constant endpoints make the eq:imu:quadrature
// trapezoid equal dt * value exactly.
star::sensors::SensorCycleTruth held_cycle(double dt, double w, double f) {
  star::sensors::SensorCycleTruth c;
  c.dt_s = dt;
  c.omega_b_start_radps = Eigen::Vector3d(w, w, w);
  c.omega_b_end_radps = Eigen::Vector3d(w, w, w);
  c.sf_b_start_mps2 = Eigen::Vector3d(f, f, f);
  c.sf_b_end_mps2 = Eigen::Vector3d(f, f, f);
  return c;
}

}  // namespace

TEST_CASE("sensors_ideal_imu_trapezoidal_accumulation_and_reset") {
  // The zero-error configuration is the model's ideal special case: every
  // error term degenerates to an exact identity, so the emitted increments
  // are the raw eq:imu:quadrature trapezoids.
  star::sensors::Imu imu(10, star::sensors::ImuErrorCfg(), 7);
  CHECK(std::string(imu.kind()) == "imu");
  CHECK(imu.sample_rate_hz() == 10);
  CHECK_FALSE(imu.last_sample().valid);

  // Endpoint values and dt chosen exactly representable in binary64, so
  // the trapezoidal sums (h/2)(x_start + x_end) are exact and asserted
  // with bit equality (eq:imu:quadrature; sensors/imu.hpp).
  const double dt = 0.25;
  star::sensors::SensorCycleTruth c1;
  c1.t_s = 0.0;
  c1.dt_s = dt;
  c1.omega_b_start_radps = Eigen::Vector3d(0.25, -0.5, 0.125);
  c1.omega_b_end_radps = Eigen::Vector3d(0.75, -0.25, 0.375);
  c1.sf_b_start_mps2 = Eigen::Vector3d(8.0, 0.0, -2.0);
  c1.sf_b_end_mps2 = Eigen::Vector3d(4.0, 2.0, -6.0);
  star::sensors::SensorCycleTruth c2 = c1;
  c2.t_s = dt;
  c2.omega_b_start_radps = Eigen::Vector3d(-0.125, 0.25, 0.5);
  c2.omega_b_end_radps = Eigen::Vector3d(-0.375, 0.75, 0.25);
  c2.sf_b_start_mps2 = Eigen::Vector3d(0.0, 4.0, 2.0);
  c2.sf_b_end_mps2 = Eigen::Vector3d(2.0, 6.0, 4.0);

  imu.accumulate(c1);
  imu.accumulate(c2);

  const std::string path = "test_sensors_imu.srlog";
  {
    star::log::SrlogWriter writer = make_imu_writer(path);
    imu.sample(0.5, writer);

    const star::gnc::ImuSample& s = imu.last_sample();
    CHECK(s.valid);
    CHECK(s.t_s == 0.5);
    CHECK(s.dt_s == 0.5);  // two exact quarter-second cycles
    // dtheta = 0.125*(1.0, -0.75, 0.5) + 0.125*(-0.5, 1.0, 0.75), exact.
    CHECK(s.dtheta_b_rad[0] == 0.0625);
    CHECK(s.dtheta_b_rad[1] == 0.03125);
    CHECK(s.dtheta_b_rad[2] == 0.15625);
    // dv = 0.125*(12, 2, -8) + 0.125*(2, 10, 6), exact.
    CHECK(s.dv_b_mps[0] == 1.75);
    CHECK(s.dv_b_mps[1] == 1.5);
    CHECK(s.dv_b_mps[2] == -0.25);

    // sample() resets the accumulators: the next interval starts clean.
    star::sensors::SensorCycleTruth c3 = c1;
    c3.t_s = 0.5;
    imu.accumulate(c3);
    imu.sample(0.75, writer);
    const star::gnc::ImuSample& s2 = imu.last_sample();
    CHECK(s2.dt_s == 0.25);
    CHECK(s2.dtheta_b_rad[0] == 0.125);  // 0.125 * (0.25 + 0.75) only
    CHECK(s2.dv_b_mps[0] == 1.5);        // 0.125 * (8 + 4) only
    writer.close();
  }
  std::remove(path.c_str());
}

TEST_CASE("sensors_imu_turnon_bias_and_draw_schedule") {
  // Turn-on bias only (eq:imu:turnon): with every other coefficient zero the
  // measured increment is exactly truth + b0 * dt, and b0 is the first three
  // draws of the "sensors.imu" stream. Replicating the stream here pins the
  // normative initialization draw schedule of ch:sensors-imu note 3, not
  // just the arithmetic.
  const std::uint64_t seed = 20260719;
  const double sigma0 = 1.0e-3;
  star::sensors::ImuErrorCfg err;
  err.gyro.turnon_bias_sigma = sigma0;

  star::rng::NormalSampler ref(star::rng::make_stream(seed, "sensors.imu"));
  Eigen::Vector3d b0;
  for (int i = 0; i < 3; ++i) b0[i] = sigma0 * ref.next();
  // The remaining nine initialization draws are consumed unconditionally
  // even though their coefficients are zero; the accelerometer turn-on bias
  // is the seventh through ninth, and is zero here by configuration.
  for (int i = 0; i < 9; ++i) (void)ref.next();

  star::sensors::Imu imu(4, err, seed);
  const double dt = 0.25;
  imu.accumulate(held_cycle(dt, 2.0, 3.0));

  const std::string path = "test_sensors_imu_bias.srlog";
  {
    star::log::SrlogWriter writer = make_imu_writer(path, 4);
    imu.sample(dt, writer);
    const star::gnc::ImuSample& s = imu.last_sample();
    for (int i = 0; i < 3; ++i) {
      // Bit equality: with a zero distortion matrix the M*truth product is
      // exactly the zero vector, so the chain reduces to truth + b0*dt.
      CHECK(s.dtheta_b_rad[i] == 2.0 * dt + b0[i] * dt);
      CHECK(s.dv_b_mps[i] == 3.0 * dt);  // accelerometer left ideal
    }
    writer.close();
  }
  std::remove(path.c_str());
}

TEST_CASE("sensors_imu_scale_factor_and_misalignment") {
  // eq:imu:mis: M = S + Gamma applied as (I + M) dtheta. Scale factors are
  // configured in ppm, misalignments in rad in the order xy, xz, yx, yz,
  // zx, zy.
  star::gnc::GncSensorCfg cfg;
  cfg.kind = "imu";
  cfg.sample_rate_hz = 4;
  cfg.vectors["gyro_scale_factor_ppm"] = {100.0, -200.0, 300.0};
  cfg.vectors["gyro_misalignment_rad"] = {1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5};
  const star::sensors::ImuErrorCfg err = star::sensors::parse_imu_error_cfg(cfg);
  CHECK(err.gyro.distortion(0, 0) == doctest::Approx(1.0e-4));
  CHECK(err.gyro.distortion(1, 1) == doctest::Approx(-2.0e-4));
  CHECK(err.gyro.distortion(2, 2) == doctest::Approx(3.0e-4));
  CHECK(err.gyro.distortion(0, 1) == 1e-5);
  CHECK(err.gyro.distortion(0, 2) == 2e-5);
  CHECK(err.gyro.distortion(1, 0) == 3e-5);
  CHECK(err.gyro.distortion(1, 2) == 4e-5);
  CHECK(err.gyro.distortion(2, 0) == 5e-5);
  CHECK(err.gyro.distortion(2, 1) == 6e-5);

  star::sensors::Imu imu(4, err, 11);
  const double dt = 0.25;
  const double w = 2.0;
  imu.accumulate(held_cycle(dt, w, 0.0));
  const Eigen::Vector3d truth(w * dt, w * dt, w * dt);
  const Eigen::Vector3d expect = truth + err.gyro.distortion * truth;

  const std::string path = "test_sensors_imu_mis.srlog";
  {
    star::log::SrlogWriter writer = make_imu_writer(path, 4);
    imu.sample(dt, writer);
    const star::gnc::ImuSample& s = imu.last_sample();
    for (int i = 0; i < 3; ++i) {
      CHECK(s.dtheta_b_rad[i] == doctest::Approx(expect[i]).epsilon(1e-15));
    }
    writer.close();
  }
  std::remove(path.c_str());

  // A typo must fail rather than silently disable the term it names.
  star::gnc::GncSensorCfg typo;
  typo.kind = "imu";
  typo.sample_rate_hz = 4;
  typo.scalars["gyro_arw_rad_per_sqrt_sec"] = 1e-5;
  CHECK_THROWS_AS(star::sensors::parse_imu_error_cfg(typo),
                  std::invalid_argument);
  star::gnc::GncSensorCfg badlen;
  badlen.kind = "imu";
  badlen.sample_rate_hz = 4;
  badlen.vectors["gyro_scale_factor_ppm"] = {1.0, 2.0};
  CHECK_THROWS_AS(star::sensors::parse_imu_error_cfg(badlen),
                  std::invalid_argument);
  star::gnc::GncSensorCfg neg;
  neg.kind = "imu";
  neg.sample_rate_hz = 4;
  neg.scalars["gyro_quantum_rad"] = -1.0;
  CHECK_THROWS_AS(star::sensors::parse_imu_error_cfg(neg),
                  std::invalid_argument);
}

TEST_CASE("sensors_imu_quantizer_carry_and_ties") {
  // eq:imu:quant with every value an exact binary fraction, so the whole
  // sequence is asserted with bit equality. Per cycle the truth increment is
  // (0.25/2)(0.5 + 0.5) = 0.125 rad, exactly half the quantum q = 0.25:
  //   k=1: s = 0.125 + 0     -> s/q + 1/2 = 1.0 -> y = 0.25, rho = -0.125
  //   k=2: s = 0.125 - 0.125 -> s/q + 1/2 = 0.5 -> y = 0.0,  rho =  0.0
  // which exercises the tie (rounding toward +infinity by construction) and
  // the residual carry that makes quantization lossless in accumulation.
  star::sensors::ImuErrorCfg err;
  err.gyro.quantum = 0.25;
  star::sensors::Imu imu(4, err, 3);

  const std::string path = "test_sensors_imu_quant.srlog";
  {
    star::log::SrlogWriter writer = make_imu_writer(path, 4);
    double truth_sum = 0.0;
    double emitted_sum = 0.0;
    for (int k = 0; k < 6; ++k) {
      imu.accumulate(held_cycle(0.25, 0.5, 0.0));
      imu.sample(0.25 * (k + 1), writer);
      const double y = imu.last_sample().dtheta_b_rad[0];
      // Every emitted increment is an exact multiple of the quantum.
      CHECK(y == (k % 2 == 0 ? 0.25 : 0.0));
      truth_sum += 0.125;
      emitted_sum += y;
      // The carry bounds the accumulated difference by half a quantum at
      // every sample - the rate-integrating output register's behavior.
      CHECK(std::fabs(emitted_sum - truth_sum) <= 0.5 * 0.25);
    }
    CHECK(truth_sum == 0.75);
    CHECK(emitted_sum == 0.75);  // lossless over a whole number of quanta
    writer.close();
  }
  std::remove(path.c_str());
}

TEST_CASE("sensors_imu_gauss_markov_recursion_and_preset_map") {
  // eq:imu:presetmap: sigma_GM = (0.664282 / 0.617364) B = 1.0760 B, chosen
  // so the Gauss-Markov ADEV of eq:imu:gmadev peaks at exactly 0.664282 B at
  // the eq:imu:gmpeak abscissa tau* = 1.8926 tau_c. Both halves of that
  // claim are checked here against the closed form, so the constant cannot
  // drift from the convention it encodes.
  const double b_inst = 3.0e-6;  // rad/s
  const double sigma_gm = star::sensors::gm_sigma_from_bias_instability(b_inst);
  CHECK(sigma_gm == doctest::Approx(1.0760 * b_inst).epsilon(1e-4));

  const double tau_c = 20.0;
  const auto gm_adev = [sigma_gm, tau_c](double tau) {
    // eq:imu:gmadev, the Allan variance of the first-order Gauss-Markov
    // process, evaluated as its deviation.
    const double x = tau / tau_c;
    const double bracket =
        1.0 - (tau_c / (2.0 * tau)) *
                  (3.0 - 4.0 * std::exp(-x) + std::exp(-2.0 * x));
    return std::sqrt((2.0 * sigma_gm * sigma_gm * tau_c / tau) * bracket);
  };
  const double tau_star = 1.8926 * tau_c;
  CHECK(gm_adev(tau_star) == doctest::Approx(0.664282 * b_inst).epsilon(1e-4));
  // tau* is a maximum: the curve is below the peak on both sides.
  CHECK(gm_adev(0.5 * tau_star) < gm_adev(tau_star));
  CHECK(gm_adev(2.0 * tau_star) < gm_adev(tau_star));

  // The in-run bias state follows eq:imu:gm exactly, driven by the stream's
  // own draws in the normative per-sample order (gyro drive first).
  const std::uint64_t seed = 4242;
  star::sensors::ImuErrorCfg err;
  err.gyro.bias_instability = b_inst;
  err.gyro.bias_tau_s = tau_c;
  const std::uint32_t rate = 10;
  const double dt = 1.0 / rate;
  const double phi = std::exp(-dt / tau_c);
  const double w_sigma = sigma_gm * std::sqrt(1.0 - phi * phi);

  star::rng::NormalSampler ref(star::rng::make_stream(seed, "sensors.imu"));
  for (int i = 0; i < 3; ++i) (void)ref.next();  // gyro turn-on bias
  Eigen::Vector3d b;
  for (int i = 0; i < 3; ++i) b[i] = sigma_gm * ref.next();  // stationary init
  for (int i = 0; i < 6; ++i) (void)ref.next();  // accel turn-on + GM init

  star::sensors::Imu imu(rate, err, seed);
  for (int i = 0; i < 3; ++i) {
    CHECK(imu.gyro_bias_radps()[i] == b[i]);
  }

  const std::string path = "test_sensors_imu_gm.srlog";
  {
    star::log::SrlogWriter writer = make_imu_writer(path, rate);
    for (int k = 0; k < 5; ++k) {
      for (int i = 0; i < 3; ++i) b[i] = phi * b[i] + w_sigma * ref.next();
      for (int i = 0; i < 9; ++i) (void)ref.next();  // ARW, accel GM, accel VRW
      imu.accumulate(held_cycle(dt, 0.0, 0.0));
      imu.sample(dt * (k + 1), writer);
      for (int i = 0; i < 3; ++i) {
        CHECK(imu.gyro_bias_radps()[i] == b[i]);
        // Static truth: the whole measured increment is the bias integral.
        CHECK(imu.last_sample().dtheta_b_rad[i] == doctest::Approx(b[i] * dt)
                                                       .epsilon(1e-13));
      }
    }
    writer.close();
  }
  std::remove(path.c_str());
}

TEST_CASE("sensors_imu_zero_error_config_is_the_ideal_special_case") {
  // The error chain's zero-coefficient configuration must reproduce the
  // ideal increments bit-for-bit: every stage is an exact identity, so a
  // run that configures no errors logs exactly what the pre-error-chain
  // reference logged. Two differently seeded instances must also agree,
  // because a zero coefficient multiplies its draw by zero.
  star::sensors::Imu a(4, star::sensors::ImuErrorCfg(), 1);
  star::sensors::Imu b(4, star::sensors::ImuErrorCfg(), 999999);
  const std::string path = "test_sensors_imu_zero.srlog";
  {
    star::log::SrlogWriter writer = make_imu_writer(path, 4);
    for (int k = 0; k < 4; ++k) {
      star::sensors::SensorCycleTruth c;
      c.dt_s = 0.25;
      c.omega_b_start_radps = Eigen::Vector3d(0.125 * k, -0.25, 0.5);
      c.omega_b_end_radps = Eigen::Vector3d(0.25, 0.125 * k, -0.5);
      c.sf_b_start_mps2 = Eigen::Vector3d(1.5, -2.5, 0.25 * k);
      c.sf_b_end_mps2 = Eigen::Vector3d(-0.5, 3.0, 0.5);
      a.accumulate(c);
      b.accumulate(c);
      a.sample(0.25 * (k + 1), writer);
      b.sample(0.25 * (k + 1), writer);
      for (int i = 0; i < 3; ++i) {
        CHECK(a.last_sample().dtheta_b_rad[i] == b.last_sample().dtheta_b_rad[i]);
        CHECK(a.last_sample().dv_b_mps[i] == b.last_sample().dv_b_mps[i]);
        // The ideal value is the raw trapezoid, reconstructed here.
        CHECK(a.last_sample().dtheta_b_rad[i] ==
              0.125 * (c.omega_b_start_radps[i] + c.omega_b_end_radps[i]));
      }
    }
    writer.close();
  }
  std::remove(path.c_str());
}

TEST_CASE("sensors_aberration_magnitude_and_direction") {
  // eq:optical:aberration against the closed-form magnitude of
  // eq:optical:abmag. The headline number of ch:sensors-optical: at Earth's
  // mean heliocentric speed of 29.78 km/s, a source perpendicular to the
  // velocity is displaced by 20.49 arcsec.
  const double c = star::constants::SPEED_OF_LIGHT_M_PER_S;
  const double v = 29780.0;
  const double arcsec = star::constants::TWO_PI / (360.0 * 3600.0);
  const Eigen::Vector3d beta = Eigen::Vector3d(v, 0.0, 0.0) / c;

  const Eigen::Vector3d u_perp(0.0, 0.0, 1.0);
  const Eigen::Vector3d u_ab = star::sensors::aberrate(u_perp, beta);
  CHECK(u_ab.norm() == doctest::Approx(1.0).epsilon(1e-15));
  const double disp = star::sensors::angle_between(u_perp, u_ab);
  CHECK(disp / arcsec == doctest::Approx(20.49).epsilon(1e-3));

  // The apparent direction is displaced TOWARD the velocity: the component
  // of the apparent direction along beta is positive where the geometric
  // direction had none. This is the sign convention ch:sensors-optical
  // calls out as the usual defect, so it is asserted rather than assumed.
  CHECK(u_ab.dot(beta.normalized()) > 0.0);
  CHECK(u_ab.z() > 0.0);  // still predominantly the original direction

  // Parallel and antiparallel sources are undisplaced: beta has no
  // perpendicular component to add (eq:optical:abmag's sin(theta) factor).
  for (double sign : {1.0, -1.0}) {
    const Eigen::Vector3d u_par(sign, 0.0, 0.0);
    const Eigen::Vector3d out = star::sensors::aberrate(u_par, beta);
    CHECK(star::sensors::angle_between(u_par, out) <
          1e-12 * arcsec);
  }

  // The first-order displacement is beta*sin(theta) across the full range.
  for (int k = 0; k <= 8; ++k) {
    const double th = star::constants::TWO_PI * 0.5 * k / 8.0;
    const Eigen::Vector3d u(std::cos(th), 0.0, std::sin(th));
    const double got =
        star::sensors::angle_between(u, star::sensors::aberrate(u, beta));
    // sin(theta) here is the angle to +X, and u is built with that angle.
    CHECK(got == doctest::Approx((v / c) * std::sin(th)).epsilon(2e-4));
  }

  // beta = 0 returns the input untouched, exactly (a run without ephemeris
  // must not perturb a direction).
  const Eigen::Vector3d u0(0.0, 1.0, 0.0);
  const Eigen::Vector3d same =
      star::sensors::aberrate(u0, Eigen::Vector3d::Zero());
  CHECK(same.x() == u0.x());
  CHECK(same.y() == u0.y());
  CHECK(same.z() == u0.z());

  // eq:optical:beta composes the spacecraft and central-body barycentric
  // velocities before dividing by c.
  const Eigen::Vector3d b = star::sensors::aberration_beta(
      Eigen::Vector3d(1000.0, 0.0, 0.0), Eigen::Vector3d(29000.0, 0.0, 0.0));
  CHECK(b.x() == doctest::Approx(30000.0 / c).epsilon(1e-15));
}

TEST_CASE("sensors_startracker_noise_extraction_is_the_drawn_rotation") {
  // eq:optical:stmodel with the deterministic factors removed
  // (eq:optical:extract) must return the drawn rotation vector identically -
  // that exactness is what makes the acceptance statistic chi-square rather
  // than approximately so.
  //
  // The observer is held at rest so beta, and hence the eq:optical:rho field
  // rotation, is exactly zero and q_ab is the identity: that isolates the
  // noise quaternion, which is what this case is about. Note that beta
  // depends on the truth VELOCITY, not on whether an ephemeris is loaded -
  // a moving observer aberrates even with no barycentric term available -
  // so zeroing the velocity is the way to remove the factor, and the
  // aberration itself is covered by its own case above.
  const std::uint64_t seed = 90210;
  star::sensors::StarTrackerCfg cfg;
  cfg.sigma_rad = Eigen::Vector3d(2e-5, 3e-5, 9e-5);
  star::sensors::StarTracker st(4, cfg, seed);

  star::sensors::SensorCycleTruth truth;
  truth.dt_s = 0.25;
  // A non-identity truth attitude, so the extraction cannot pass by
  // accident on an identity quaternion.
  truth.q_end_i2b =
      Eigen::Quaterniond(0.5, 0.5, -0.5, 0.5).normalized();
  truth.v_end_i_mps = Eigen::Vector3d::Zero();
  st.accumulate(truth);

  star::rng::NormalSampler ref(
      star::rng::make_stream(seed, "sensors.startracker"));

  const std::string path = "test_sensors_st.srlog";
  {
    star::log::SrlogHeaderFields f;
    f.core_version = "0.6.0-test";
    f.git_hash = "unknown";
    f.config_sha256 = std::string(64, '0');
    f.master_seed = seed;
    f.oracle = false;
    f.epoch_utc = "2026-01-01T00:00:00Z";
    f.central_body = "earth";
    f.truth_rate_hz = 4;
    f.cycle_rate_hz = 4;
    f.sensors = {{"startracker", 4, 0}};
    star::log::SrlogWriter writer(path, f);

    for (int k = 0; k < 8; ++k) {
      st.accumulate(truth);
      st.sample(0.25 * (k + 1), writer);
      Eigen::Vector3d eps;
      for (int i = 0; i < 3; ++i) eps[i] = cfg.sigma_rad[i] * ref.next();
      // beta is zero, so rho is zero and q_ab is identity: the extraction
      // dq = q_true^-1 (x) q_meas must be exactly the drawn rotation.
      const Eigen::Quaterniond dq =
          truth.q_end_i2b.conjugate() * st.last_measurement();
      // eq:optical:extract, with every intermediate bound to a named type:
      // an Eigen product held in a deduced type can outlive its temporary
      // operands, so the vector part and the sign-corrected axis are
      // materialized explicitly.
      const Eigen::Vector3d dqv = dq.vec();
      const double vn = dqv.norm();
      const double theta = 2.0 * std::atan2(vn, std::fabs(dq.w()));
      Eigen::Vector3d axis = Eigen::Vector3d::Zero();
      if (vn > 0.0) {
        // sgn(dq_w) resolves the double cover; sgn(0) = +1 by convention.
        axis = (dq.w() >= 0.0) ? Eigen::Vector3d(dqv / vn)
                               : Eigen::Vector3d(-dqv / vn);
      }
      const Eigen::Vector3d extracted = theta * axis;
      for (int i = 0; i < 3; ++i) {
        CHECK(extracted[i] == doctest::Approx(eps[i]).epsilon(1e-11));
      }
    }
    writer.close();
  }
  std::remove(path.c_str());
}

TEST_CASE("sensors_camera_pinhole_projection_and_visibility") {
  // eq:camera:proj on constructed geometry, and each visibility test across
  // its boundary. The camera looks along +Z_body with an identity mount, so
  // a landmark straight ahead lands on the principal point.
  star::sensors::CameraCfg cfg;
  cfg.fx = 800.0;
  cfg.fy = 600.0;  // anisotropic, per the exit-criterion-7 scenario
  cfg.cx = 511.5;
  cfg.cy = 383.5;
  cfg.width_px = 1024;
  cfg.height_px = 768;
  // Body -> camera identity means camera +Z is body +Z. The landmark sits
  // off the body center: a landmark AT the center is degenerate for the
  // eq:camera:nearside test, whose dot product against (l - r_T) is
  // identically zero there.
  cfg.landmarks_fixed_m = {Eigen::Vector3d(0.0, 0.0, -1000.0)};

  star::sensors::CameraHook cam(4, cfg);
  star::sensors::SensorCycleTruth truth;
  truth.dt_s = 0.25;
  truth.q_end_i2b = Eigen::Quaterniond::Identity();
  truth.geom.c_gcrf_to_bodyfixed = Eigen::Matrix3d::Identity();
  truth.geom.bodyfixed_valid = true;

  const std::string path = "test_sensors_cam.srlog";
  star::log::SrlogHeaderFields f;
  f.core_version = "0.6.0-test";
  f.git_hash = "unknown";
  f.config_sha256 = std::string(64, '0');
  f.master_seed = 1;
  f.oracle = false;
  f.epoch_utc = "2026-01-01T00:00:00Z";
  f.central_body = "earth";
  f.truth_rate_hz = 4;
  f.cycle_rate_hz = 4;
  f.sensors = {{"camera", 4, 1}};
  {
    star::log::SrlogWriter writer(path, f);

    // Camera at z = -5000 looking along +Z, landmark at z = -1000: dead
    // ahead at 4000 m range, so it projects to the principal point.
    truth.r_end_i_m = Eigen::Vector3d(0.0, 0.0, -5000.0);
    cam.accumulate(truth);
    cam.sample(0.25, writer);
    CHECK(cam.last_pixels()[0] == doctest::Approx(cfg.cx).epsilon(1e-12));
    CHECK(cam.last_pixels()[1] == doctest::Approx(cfg.cy).epsilon(1e-12));
    CHECK(cam.last_visible()[0] == 1);

    // Offset the camera to +X by 100 m: the landmark sits at X = -100 in
    // camera axes at Z = 4000, so u shifts by fx * (-100/4000) = -20 px.
    // Anisotropy is exercised by fy != fx on the v axis elsewhere.
    truth.r_end_i_m = Eigen::Vector3d(100.0, 0.0, -5000.0);
    cam.accumulate(truth);
    cam.sample(0.5, writer);
    CHECK(cam.last_pixels()[0] ==
          doctest::Approx(cfg.cx - 20.0).epsilon(1e-12));
    CHECK(cam.last_pixels()[1] == doctest::Approx(cfg.cy).epsilon(1e-12));
    CHECK(cam.last_visible()[0] == 1);

    // A +Y camera offset moves v by fy * (-100/4000) = -15 px, which is a
    // different shift than u took for the same metric offset - the
    // anisotropic-focal-length case exit criterion 7 calls for.
    truth.r_end_i_m = Eigen::Vector3d(0.0, 100.0, -5000.0);
    cam.accumulate(truth);
    cam.sample(0.75, writer);
    CHECK(cam.last_pixels()[1] ==
          doctest::Approx(cfg.cy - 15.0).epsilon(1e-12));

    // Test 1 (in front of the camera): move the camera past the landmark so
    // the line of sight runs backward along the boresight; Z goes negative.
    truth.r_end_i_m = Eigen::Vector3d(0.0, 0.0, 0.0);
    cam.accumulate(truth);
    cam.sample(1.0, writer);
    CHECK(cam.last_visible()[0] == 0);

    // Test 2 (within the sensor) at the half-pixel edge: with Z = 4000,
    // u = cx + fx*(-x_off/4000) = 511.5 - 0.2*x_off, so the right edge
    // u = W - 1/2 = 1023.5 is crossed at x_off = -2560.
    truth.r_end_i_m = Eigen::Vector3d(-2559.0, 0.0, -5000.0);
    cam.accumulate(truth);
    cam.sample(1.25, writer);
    CHECK(cam.last_pixels()[0] < 1023.5);
    CHECK(cam.last_visible()[0] == 1);
    truth.r_end_i_m = Eigen::Vector3d(-2561.0, 0.0, -5000.0);
    cam.accumulate(truth);
    cam.sample(1.5, writer);
    CHECK(cam.last_pixels()[0] > 1023.5);
    CHECK(cam.last_visible()[0] == 0);
    writer.close();
  }
  std::remove(path.c_str());

  // Test 3 (eq:camera:nearside) in isolation: one camera pose, two
  // landmarks on opposite sides of the body center, BOTH in front of the
  // camera and both inside the sensor - only the near-side one is visible.
  // The camera sits at +Z with a 180 degree roll about X, so its boresight
  // (body +Z) points along inertial -Z toward the body.
  star::sensors::CameraCfg c2 = cfg;
  c2.landmarks_fixed_m = {Eigen::Vector3d(0.0, 0.0, 1000.0),
                          Eigen::Vector3d(0.0, 0.0, -1000.0)};
  star::sensors::CameraHook cam2(4, c2);
  star::sensors::SensorCycleTruth t2;
  t2.dt_s = 0.25;
  t2.geom.bodyfixed_valid = true;
  t2.geom.c_gcrf_to_bodyfixed = Eigen::Matrix3d::Identity();
  t2.q_end_i2b = Eigen::Quaterniond(0.0, 1.0, 0.0, 0.0);  // 180 deg about X
  t2.r_end_i_m = Eigen::Vector3d(0.0, 0.0, 5000.0);
  star::log::SrlogHeaderFields f2 = f;
  f2.sensors = {{"camera", 4, 2}};
  {
    star::log::SrlogWriter writer(path, f2);
    cam2.accumulate(t2);
    cam2.sample(0.25, writer);
    // Near-side landmark (z = +1000, same side as the camera): visible.
    CHECK(cam2.last_visible()[0] == 1);
    // Far-side landmark (z = -1000, body between it and the camera): the
    // camera is below its local tangent plane, so the flag drops even
    // though the projection itself is well inside the sensor.
    CHECK(cam2.last_visible()[1] == 0);
    CHECK(cam2.last_pixels()[2] == doctest::Approx(cfg.cx).epsilon(1e-9));
    CHECK(cam2.last_pixels()[3] == doctest::Approx(cfg.cy).epsilon(1e-9));
    writer.close();
  }
  std::remove(path.c_str());
}

TEST_CASE("sensors_navfix_and_altimeter_error_terms") {
  // eq:radio:fix and eq:radio:alt against independently replicated streams,
  // which pins both the arithmetic and the normative draw schedules.
  const std::uint64_t seed = 5150;
  const Eigen::Vector3d r_true(7.0e6, 1.0e6, -2.0e6);
  const Eigen::Vector3d v_true(1.0e3, 7.4e3, 0.5e3);

  star::sensors::NavFixCfg fix;
  fix.sigma_r_m = Eigen::Vector3d(5.0, 5.0, 9.0);
  fix.sigma_v_mps = Eigen::Vector3d(0.05, 0.05, 0.09);
  star::sensors::NavFix nav(4, fix, seed);

  star::sensors::SensorCycleTruth truth;
  truth.dt_s = 0.25;
  truth.r_end_i_m = r_true;
  truth.v_end_i_mps = v_true;

  star::rng::NormalSampler ref(
      star::rng::make_stream(seed, "sensors.navfix"));

  const std::string path = "test_sensors_radio.srlog";
  star::log::SrlogHeaderFields f;
  f.core_version = "0.6.0-test";
  f.git_hash = "unknown";
  f.config_sha256 = std::string(64, '0');
  f.master_seed = seed;
  f.oracle = false;
  f.epoch_utc = "2026-01-01T00:00:00Z";
  f.central_body = "earth";
  f.truth_rate_hz = 4;
  f.cycle_rate_hz = 4;
  f.sensors = {{"navfix", 4, 0}, {"altimeter", 4, 0}};
  {
    star::log::SrlogWriter writer(path, f);
    for (int k = 0; k < 5; ++k) {
      nav.accumulate(truth);
      nav.sample(0.25 * (k + 1), writer);
      // With the correlated components off, the schedule is position white
      // then velocity white - three draws each, no Gauss-Markov draws.
      for (int i = 0; i < 3; ++i) {
        CHECK(nav.last_position_m()[i] ==
              r_true[i] + fix.sigma_r_m[i] * ref.next());
      }
      for (int i = 0; i < 3; ++i) {
        CHECK(nav.last_velocity_mps()[i] ==
              v_true[i] + fix.sigma_v_mps[i] * ref.next());
      }
    }

    // Altimeter: turn-on bias is one draw at construction, then one white
    // draw per sample (eq:radio:alt). A spherical central body makes the
    // truth altitude exactly norm(r) - a.
    star::sensors::AltimeterCfg alt_cfg;
    alt_cfg.sigma_bias_m = 3.0;
    alt_cfg.sigma_noise_m = 0.5;
    alt_cfg.h_min_m = 0.0;
    alt_cfg.h_max_m = 1.0e6;
    star::rng::NormalSampler aref(
        star::rng::make_stream(seed, "sensors.altimeter"));
    const double bias = alt_cfg.sigma_bias_m * aref.next();
    star::sensors::Altimeter alt(4, alt_cfg, seed);
    CHECK(alt.turnon_bias_m() == bias);

    star::sensors::SensorCycleTruth at = truth;
    at.geom.bodyfixed_valid = true;
    at.geom.c_gcrf_to_bodyfixed = Eigen::Matrix3d::Identity();
    at.geom.ellipsoid_a_m = 6378137.0;
    at.geom.ellipsoid_inv_f = 0.0;  // sphere: h = norm(r) - a exactly
    at.r_end_i_m = Eigen::Vector3d(6378137.0 + 400000.0, 0.0, 0.0);
    for (int k = 0; k < 5; ++k) {
      alt.accumulate(at);
      alt.sample(0.25 * (k + 1), writer);
      const double expect =
          400000.0 + bias + alt_cfg.sigma_noise_m * aref.next();
      CHECK(alt.last_measurement_m() == doctest::Approx(expect).epsilon(1e-14));
      CHECK(alt.last_valid());  // inside the configured band
    }
    // eq:radio:altgate: outside the band the flag drops, and the draw is
    // still consumed so the stream schedule is gate-independent.
    at.r_end_i_m = Eigen::Vector3d(6378137.0 + 2.0e6, 0.0, 0.0);
    alt.accumulate(at);
    alt.sample(2.0, writer);
    CHECK_FALSE(alt.last_valid());
    const double expect_out =
        2.0e6 + bias + alt_cfg.sigma_noise_m * aref.next();
    CHECK(alt.last_measurement_m() ==
          doctest::Approx(expect_out).epsilon(1e-14));
    writer.close();
  }
  std::remove(path.c_str());
}

TEST_CASE("sensors_factory_kind_vocabulary") {
  // Every canonical FR-23 kind constructs at its configured rate and reports
  // its own group name, so the log declaration and the sensor cannot drift
  // apart. The camera needs positive intrinsics to be well defined; the rest
  // are fully specified by their defaults (the ideal instrument).
  for (const char* kind : {"imu", "startracker", "sunsensor", "navfix",
                           "altimeter", "camera"}) {
    star::gnc::GncSensorCfg cfg;
    cfg.kind = kind;
    cfg.sample_rate_hz = 100;
    if (cfg.kind == "camera") {
      cfg.scalars["fx_px"] = 800.0;
      cfg.scalars["fy_px"] = 800.0;
      cfg.scalars["width_px"] = 1024.0;
      cfg.scalars["height_px"] = 768.0;
    }
    const auto s = star::sensors::make_sensor(cfg, 42);
    CHECK(std::string(s->kind()) == kind);
    CHECK(s->sample_rate_hz() == 100);
  }

  // A name outside the canonical vocabulary is refused by name rather than
  // silently resolved to a stub, so a typo cannot enter a run.
  star::gnc::GncSensorCfg bad;
  bad.kind = "lidar";
  bad.sample_rate_hz = 10;
  CHECK_THROWS_AS(star::sensors::make_sensor(bad, 42), std::invalid_argument);

  star::gnc::GncSensorCfg zero;
  zero.kind = "imu";
  zero.sample_rate_hz = 0;
  CHECK_THROWS_AS(star::sensors::make_sensor(zero, 42),
                  std::invalid_argument);
}
