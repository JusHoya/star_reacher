// FR-23 sensor-layer unit tests: the IMU's exact accumulation of the loop's
// held per-cycle kinematics, its sample/reset semantics, the error chain of
// eq:imu:gyro--eq:imu:quant, and the sensor factory's kind vocabulary.
// Written against the contracts in sensors/imu.hpp; record-level byte layout
// is covered by test_srlog.cpp, and in-loop scheduling by test_gnc_cycle.cpp.
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

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

// A writer declaring exactly one sensor group, for sample() calls that need
// a record sink; the emitted bytes are covered by test_srlog.cpp.
star::log::SrlogWriter make_one_sensor_writer(const std::string& path,
                                              const std::string& kind,
                                              std::uint32_t rate_hz) {
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
  f.sensors = {{kind, rate_hz, 0}};
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

namespace {

// Overlapping Allan variance of a phase (cumulative-angle) record,
// eq:imu:oadev:
//
//   sigma^2(tau) = 1/(2 tau^2 (N - 2m)) sum (theta_{n+2m} - 2 theta_{n+m}
//                                            + theta_n)^2
//
// the maximally overlapped estimator, which reuses every stride-m second
// difference and has the smallest variance of the standard family.
double oadev(const std::vector<double>& theta, std::size_t m, double dt) {
  const std::size_t n_theta = theta.size();
  REQUIRE(n_theta > 2 * m);
  const double tau = static_cast<double>(m) * dt;
  double acc = 0.0;
  const std::size_t count = n_theta - 2 * m;
  for (std::size_t n = 0; n < count; ++n) {
    const double d = theta[n + 2 * m] - 2.0 * theta[n + m] + theta[n];
    acc += d * d;
  }
  return std::sqrt(acc / (2.0 * tau * tau * static_cast<double>(count)));
}

// eq:imu:gmadev, the Gauss-Markov contribution at cluster time tau.
double gm_adev_at(double sigma_gm, double tau_c, double tau) {
  const double x = tau / tau_c;
  const double bracket =
      1.0 - (tau_c / (2.0 * tau)) *
                (3.0 - 4.0 * std::exp(-x) + std::exp(-2.0 * x));
  return std::sqrt((2.0 * sigma_gm * sigma_gm * tau_c / tau) * bracket);
}

}  // namespace

TEST_CASE("sensors_imu_allan_recovers_arw_and_bias_instability") {
  // Phase 6 exit criterion 1: a 1e4 s static record must return the
  // configured ARW and bias-instability coefficients within +/- 10 %, by
  // the ch:sensors-imu section on the recovery procedure.
  //
  // GATE PRESET DESIGN, deliberate and load-bearing. eq:imu:recoverybi
  // subtracts the white-noise term before inverting eq:imu:bi, and that
  // subtraction amplifies estimator scatter: writing A = adev^2(tau*),
  // W = N^2/tau*, and G = A - W, the relative error propagates as
  // delta_B/B = (A/G)(delta_adev/adev). A preset where the Gauss-Markov
  // term is a MINORITY of the Allan variance at tau* therefore produces a
  // flaky gate. At N = 0.1 deg/sqrt(h) and tau_c = 20 s the ratio G/W runs
  // 0.46 at B = 1 deg/h, 1.86 at 2, 4.18 at 3, and 7.42 at 4; the
  // corresponding B_hat scatter falls from about 12 % (which would fail a
  // 10 % gate roughly 40 % of the time) to about 4.5 %. This preset takes
  // B = 4 deg/h, G/W = 7.42, so the gate has better than two sigma of
  // margin. The design rule is B >= sqrt(ratio) N / (0.664282 sqrt(tau*)).
  const double deg = star::constants::TWO_PI / 360.0;
  const double n_gyro = 0.1 * deg / 60.0;      // 0.1 deg/sqrt(h) in rad/sqrt(s)
  const double b_gyro = 4.0 * deg / 3600.0;    // 4 deg/h in rad/s
  const double n_accel = 1.0e-4;               // (m/s)/sqrt(s)
  const double b_accel = 1.0e-4;               // m/s^2
  const double tau_c = 20.0;

  star::sensors::ImuErrorCfg err;
  err.gyro.random_walk = n_gyro;
  err.gyro.bias_instability = b_gyro;
  err.gyro.bias_tau_s = tau_c;
  err.accel.random_walk = n_accel;
  err.accel.bias_instability = b_accel;
  err.accel.bias_tau_s = tau_c;

  const std::uint32_t rate = 10;
  const double dt = 1.0 / rate;
  const std::size_t n_samples = 100000;  // 1e4 s of record
  star::sensors::Imu imu(rate, err, 20260719);

  // Cumulative-angle ("phase") records with theta_0 = 0 prepended, one axis
  // per instrument - the recovery is per axis and per instrument, and one
  // axis of each exercises both chains.
  std::vector<double> theta_g;
  std::vector<double> theta_a;
  theta_g.reserve(n_samples + 1);
  theta_a.reserve(n_samples + 1);
  theta_g.push_back(0.0);
  theta_a.push_back(0.0);

  const std::string path = "test_sensors_imu_allan.srlog";
  {
    star::log::SrlogWriter writer = make_imu_writer(path, rate);
    // Static truth: zero body rate and zero specific force, so every
    // emitted increment is pure sensor error.
    const star::sensors::SensorCycleTruth statics = held_cycle(dt, 0.0, 0.0);
    for (std::size_t k = 0; k < n_samples; ++k) {
      imu.accumulate(statics);
      imu.sample(dt * static_cast<double>(k + 1), writer);
      theta_g.push_back(theta_g.back() + imu.last_sample().dtheta_b_rad[0]);
      theta_a.push_back(theta_a.back() + imu.last_sample().dv_b_mps[0]);
    }
    writer.close();
  }
  std::remove(path.c_str());

  const double tau_star = 1.8926 * tau_c;
  const std::size_t m_one = static_cast<std::size_t>(1.0 / dt);
  const std::size_t m_star = static_cast<std::size_t>(tau_star / dt + 0.5);

  struct Case {
    const std::vector<double>* theta;
    double n_coeff;
    double b_coeff;
    const char* name;
  };
  const Case cases[] = {{&theta_g, n_gyro, b_gyro, "gyro"},
                        {&theta_a, n_accel, b_accel, "accel"}};

  for (const Case& c : cases) {
    CAPTURE(c.name);
    const double sigma_gm =
        star::sensors::gm_sigma_from_bias_instability(c.b_coeff);

    // Step 2: recover the random walk at the one-second anchor by
    // subtracting the known Gauss-Markov contribution in quadrature
    // (eq:imu:recovery). The quantizer is disabled here, so its term is
    // exactly zero and is omitted rather than added as a zero.
    const double a_one = oadev(*c.theta, m_one, dt);
    const double gm_one = gm_adev_at(sigma_gm, tau_c, 1.0);
    const double n_hat = std::sqrt(a_one * a_one - gm_one * gm_one);
    CHECK(std::fabs(n_hat / c.n_coeff - 1.0) <= 0.10);

    // Step 3: recover the bias instability at the peak anchor by removing
    // the recovered white-noise term and inverting eq:imu:bi
    // (eq:imu:recoverybi).
    const double a_star = oadev(*c.theta, m_star, dt);
    const double white_at_star = n_hat * n_hat / tau_star;
    const double gm_part = a_star * a_star - white_at_star;
    // The designed margin: the Gauss-Markov term must dominate at tau*, or
    // the subtraction above amplifies scatter into the gate.
    CHECK(gm_part / white_at_star >= 4.0);
    const double b_hat = std::sqrt(gm_part) / 0.664282;
    CHECK(std::fabs(b_hat / c.b_coeff - 1.0) <= 0.10);

    // The analytic curve overlays the estimate across the octave grid
    // (the chapter's second Allan gate). Tolerance widens with cluster
    // time because the number of independent clusters falls as 1/tau.
    for (std::size_t m = 1; static_cast<double>(m) * dt <= 1000.0; m *= 2) {
      const double tau = static_cast<double>(m) * dt;
      const double model = std::sqrt(c.n_coeff * c.n_coeff / tau +
                                     std::pow(gm_adev_at(sigma_gm, tau_c, tau),
                                              2.0));
      const double est = oadev(*c.theta, m, dt);
      const double clusters = 1.0e4 / (2.0 * tau);
      const double tol = 4.0 / std::sqrt(clusters);  // ~4 sigma of scatter
      CHECK(std::fabs(est / model - 1.0) <= tol);
    }
  }
}

TEST_CASE("sensors_startracker_chi_square_over_1000_draws") {
  // Exit criterion 1: the eq:optical:ststat statistic over M = 1000 seeded
  // draws must fall inside the eq:optical:stbounds two-sided 95 % interval
  // [2.850, 3.154]. The extraction is exact (the log map inverts
  // eq:optical:noiseq), so the statistic is exactly chi-square with three
  // degrees of freedom rather than approximately so.
  const int m_draws = 1000;
  star::sensors::StarTrackerCfg cfg;
  cfg.sigma_rad = Eigen::Vector3d(1.0e-5, 1.0e-5, 5.0e-5);
  star::sensors::StarTracker st(4, cfg, 31337);

  star::sensors::SensorCycleTruth truth;
  truth.dt_s = 0.25;
  truth.q_end_i2b = Eigen::Quaterniond(0.5, 0.5, -0.5, 0.5).normalized();
  truth.v_end_i_mps = Eigen::Vector3d::Zero();  // isolates the noise factor

  const std::string path = "test_sensors_st_chi2.srlog";
  double q_sum = 0.0;
  std::vector<double> per_axis_sq(3, 0.0);
  {
    star::log::SrlogHeaderFields f;
    f.core_version = "0.6.0-test";
    f.git_hash = "unknown";
    f.config_sha256 = std::string(64, '0');
    f.master_seed = 31337;
    f.oracle = false;
    f.epoch_utc = "2026-01-01T00:00:00Z";
    f.central_body = "earth";
    f.truth_rate_hz = 4;
    f.cycle_rate_hz = 4;
    f.sensors = {{"startracker", 4, 0}};
    star::log::SrlogWriter writer(path, f);
    for (int i = 0; i < m_draws; ++i) {
      st.accumulate(truth);
      st.sample(0.25 * (i + 1), writer);
      // eq:optical:extract with the deterministic factors removed.
      const Eigen::Quaterniond dq =
          truth.q_end_i2b.conjugate() * st.last_measurement();
      const Eigen::Vector3d dqv = dq.vec();
      const double vn = dqv.norm();
      const double theta = 2.0 * std::atan2(vn, std::fabs(dq.w()));
      Eigen::Vector3d axis = Eigen::Vector3d::Zero();
      if (vn > 0.0) {
        axis = (dq.w() >= 0.0) ? Eigen::Vector3d(dqv / vn)
                               : Eigen::Vector3d(-dqv / vn);
      }
      const Eigen::Vector3d eps = theta * axis;
      for (int a = 0; a < 3; ++a) {
        const double z = eps[a] / cfg.sigma_rad[a];
        q_sum += z * z;
        per_axis_sq[static_cast<std::size_t>(a)] += eps[a] * eps[a];
      }
    }
    writer.close();
  }
  std::remove(path.c_str());

  const double q_mean = q_sum / m_draws;
  CHECK(q_mean >= 2.850);
  CHECK(q_mean <= 3.154);

  // Per-axis sample variances match the configured sigmas within their own
  // two-sided 95 % bounds: chi^2_{0.025,1000}/1000 = 0.9137 and
  // chi^2_{0.975,1000}/1000 = 1.0900.
  for (int a = 0; a < 3; ++a) {
    const double var = per_axis_sq[static_cast<std::size_t>(a)] / m_draws;
    const double ratio =
        var / (cfg.sigma_rad[a] * cfg.sigma_rad[a]);
    CHECK(ratio >= 0.9137);
    CHECK(ratio <= 1.0900);
  }
}

TEST_CASE("sensors_navfix_altimeter_chi_square_over_1000_draws") {
  // Exit criterion 6: the eq:radio:chi2 statistics over M = 1000 seeded
  // draws inside the eq:radio:bounds two-sided 95 % intervals - [2.850,
  // 3.154] for the three-degree-of-freedom position and velocity fixes and
  // [0.914, 1.090] for the one-degree-of-freedom altimeter. The gate
  // scenario holds truth fixed and disables the correlated components, as
  // the chapter's acceptance section specifies: a per-run bias would make
  // the per-sample statistics dependent and the mean gate invalid.
  const int m_draws = 1000;
  const Eigen::Vector3d r_true(7.0e6, 1.0e6, -2.0e6);
  const Eigen::Vector3d v_true(1.0e3, 7.4e3, 0.5e3);

  star::sensors::NavFixCfg fix;
  fix.sigma_r_m = Eigen::Vector3d(5.0, 8.0, 12.0);
  fix.sigma_v_mps = Eigen::Vector3d(0.05, 0.08, 0.12);
  star::sensors::NavFix nav(4, fix, 8675309);

  star::sensors::AltimeterCfg alt_cfg;
  alt_cfg.sigma_bias_m = 0.0;  // gate scenario: white part only
  alt_cfg.sigma_noise_m = 2.5;
  star::sensors::Altimeter alt(4, alt_cfg, 8675309);

  star::sensors::SensorCycleTruth truth;
  truth.dt_s = 0.25;
  truth.r_end_i_m = r_true;
  truth.v_end_i_mps = v_true;
  truth.geom.bodyfixed_valid = true;
  truth.geom.c_gcrf_to_bodyfixed = Eigen::Matrix3d::Identity();
  truth.geom.ellipsoid_a_m = 6378137.0;
  truth.geom.ellipsoid_inv_f = 0.0;  // sphere: h = norm(r) - a exactly
  const double h_true = r_true.norm() - 6378137.0;

  double qr = 0.0;
  double qv = 0.0;
  double qh = 0.0;
  const std::string path = "test_sensors_radio_chi2.srlog";
  {
    star::log::SrlogHeaderFields f;
    f.core_version = "0.6.0-test";
    f.git_hash = "unknown";
    f.config_sha256 = std::string(64, '0');
    f.master_seed = 8675309;
    f.oracle = false;
    f.epoch_utc = "2026-01-01T00:00:00Z";
    f.central_body = "earth";
    f.truth_rate_hz = 4;
    f.cycle_rate_hz = 4;
    f.sensors = {{"navfix", 4, 0}, {"altimeter", 4, 0}};
    star::log::SrlogWriter writer(path, f);
    for (int k = 0; k < m_draws; ++k) {
      nav.accumulate(truth);
      nav.sample(0.25 * (k + 1), writer);
      alt.accumulate(truth);
      alt.sample(0.25 * (k + 1), writer);
      for (int a = 0; a < 3; ++a) {
        const double dr = nav.last_position_m()[a] - r_true[a];
        const double dv = nav.last_velocity_mps()[a] - v_true[a];
        qr += (dr * dr) / (fix.sigma_r_m[a] * fix.sigma_r_m[a]);
        qv += (dv * dv) / (fix.sigma_v_mps[a] * fix.sigma_v_mps[a]);
      }
      const double dh = alt.last_measurement_m() - h_true;
      qh += (dh * dh) / (alt_cfg.sigma_noise_m * alt_cfg.sigma_noise_m);
    }
    writer.close();
  }
  std::remove(path.c_str());

  CHECK(qr / m_draws >= 2.850);
  CHECK(qr / m_draws <= 3.154);
  CHECK(qv / m_draws >= 2.850);
  CHECK(qv / m_draws <= 3.154);
  CHECK(qh / m_draws >= 0.914);
  CHECK(qh / m_draws <= 1.090);
}

TEST_CASE("sensors_sunsensor_chi_square_over_1000_draws") {
  // ch:sensors-optical acceptance statistics: the tangent-plane statistic
  // over M = 1000 draws inside [1.878, 2.126], the two-sided 95 % interval
  // for a chi-square with two degrees of freedom.
  const int m_draws = 1000;
  star::sensors::SunSensorCfg cfg;
  cfg.sigma_rad = 2.0e-3;
  star::sensors::SunSensor ss(4, cfg, 112358);

  star::sensors::SensorCycleTruth truth;
  truth.dt_s = 0.25;
  truth.q_end_i2b = Eigen::Quaterniond::Identity();
  truth.v_end_i_mps = Eigen::Vector3d::Zero();
  truth.geom.ephemeris_valid = true;
  truth.geom.illumination_nu = 1.0;
  truth.r_end_i_m = Eigen::Vector3d::Zero();
  truth.geom.r_sun_m = Eigen::Vector3d(1.5e11, 0.0, 0.0);
  const Eigen::Vector3d u_true(1.0, 0.0, 0.0);

  // Deterministic tangent-plane basis about the true direction.
  const Eigen::Vector3d z = Eigen::Vector3d::UnitZ();
  const Eigen::Vector3d e1 = u_true.cross(z).normalized();
  const Eigen::Vector3d e2 = u_true.cross(e1);

  double q_sum = 0.0;
  const std::string path = "test_sensors_sun_chi2.srlog";
  {
    star::log::SrlogHeaderFields f;
    f.core_version = "0.6.0-test";
    f.git_hash = "unknown";
    f.config_sha256 = std::string(64, '0');
    f.master_seed = 112358;
    f.oracle = false;
    f.epoch_utc = "2026-01-01T00:00:00Z";
    f.central_body = "earth";
    f.truth_rate_hz = 4;
    f.cycle_rate_hz = 4;
    f.sensors = {{"sunsensor", 4, 0}};
    star::log::SrlogWriter writer(path, f);
    for (int k = 0; k < m_draws; ++k) {
      ss.accumulate(truth);
      ss.sample(0.25 * (k + 1), writer);
      const double a1 = e1.dot(ss.last_measurement());
      const double a2 = e2.dot(ss.last_measurement());
      q_sum += (a1 * a1 + a2 * a2) / (cfg.sigma_rad * cfg.sigma_rad);
    }
    writer.close();
  }
  std::remove(path.c_str());

  const double q_mean = q_sum / m_draws;
  CHECK(q_mean >= 1.878);
  CHECK(q_mean <= 2.126);
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

TEST_CASE("sensors_startracker_exclusion_gating") {
  // The two exclusion terms of eq:optical:gating, both directions each.
  //
  // Fixture non-degeneracy, stated because this gate was previously
  // unreachable in every mission that configured it. Three properties make
  // the flag depend on the exclusion arithmetic and on nothing else:
  //   - geom.ephemeris_valid is TRUE, so the guard above the exclusion block
  //     admits the sample. The mission that configured these radii carried
  //     an invalid ephemeris and was rejected before reaching them, and the
  //     mission with a valid ephemeris set both radii to zero.
  //   - both velocities are zero, so beta is zero, aberrate() returns its
  //     input unchanged, and the tested angle is exactly the geometric
  //     separation. The expected flag is therefore analytic.
  //   - sigma_rad and slew_limit_radps are zero, so no draw perturbs the
  //     sample and the slew term cannot mask an exclusion result.
  // Each radius is exercised with the other set to zero, so neither term can
  // stand in for the other.
  const std::string path = "test_sensors_st_gating.srlog";
  const double deg = star::constants::PI / 180.0;
  const double kSun = 30.0 * deg;
  const double kCb = 25.0 * deg;

  star::sensors::SensorCycleTruth truth;
  truth.dt_s = 0.25;
  // Identity truth attitude makes C_i2b the identity, so the inertial
  // boresight IS the configured body boresight - the separations below are
  // read straight off the configuration.
  truth.q_end_i2b = Eigen::Quaterniond::Identity();
  truth.r_end_i_m = Eigen::Vector3d(7.0e6, 0.0, 0.0);
  truth.v_end_i_mps = Eigen::Vector3d::Zero();
  truth.geom.ephemeris_valid = true;
  truth.geom.v_central_ssb_mps = Eigen::Vector3d::Zero();
  // Sun far along +X: normalize(r_sun - r) is +X to within 5e-5 of a degree.
  truth.geom.r_sun_m = Eigen::Vector3d(1.5e11, 0.0, 0.0);

  // Boresights at a chosen separation from the Sun direction (+X) and from
  // the central-body direction (-X, the origin of the propagation frame).
  const auto from_sun = [deg](double a_deg) {
    return Eigen::Vector3d(std::cos(a_deg * deg), std::sin(a_deg * deg), 0.0);
  };
  const auto from_central_body = [deg](double a_deg) {
    return Eigen::Vector3d(-std::cos(a_deg * deg), std::sin(a_deg * deg), 0.0);
  };

  star::log::SrlogWriter writer =
      make_one_sensor_writer(path, "startracker", 4);
  double t = 0.25;
  const auto flag_for = [&](const star::sensors::StarTrackerCfg& cfg,
                            const star::sensors::SensorCycleTruth& tr) {
    star::sensors::StarTracker st(4, cfg, 31337);
    st.accumulate(tr);
    st.sample(t, writer);
    t += 0.25;
    return st.last_valid();
  };

  {
    star::sensors::StarTrackerCfg cfg;
    cfg.sun_exclusion_rad = kSun;  // central-body term and slew term off
    // Clear of the cone, then inside it.
    cfg.boresight_b = from_sun(40.0);
    CHECK(flag_for(cfg, truth));
    cfg.boresight_b = from_sun(20.0);
    CHECK_FALSE(flag_for(cfg, truth));
    // Straddling the threshold by half a degree either side. This pins the
    // comparison to the configured radius, in radians, against the APPARENT
    // Sun direction: a gate reading degrees, the complementary angle, or the
    // other excluded body's direction cannot place a transition within one
    // degree of 30, and an inverted comparison reverses both answers.
    cfg.boresight_b = from_sun(30.5);
    CHECK(flag_for(cfg, truth));
    cfg.boresight_b = from_sun(29.5);
    CHECK_FALSE(flag_for(cfg, truth));
  }
  {
    star::sensors::StarTrackerCfg cfg;
    cfg.central_body_exclusion_rad = kCb;  // Sun term and slew term off
    cfg.boresight_b = from_central_body(35.0);
    CHECK(flag_for(cfg, truth));
    cfg.boresight_b = from_central_body(15.0);
    CHECK_FALSE(flag_for(cfg, truth));
    cfg.boresight_b = from_central_body(25.5);
    CHECK(flag_for(cfg, truth));
    cfg.boresight_b = from_central_body(24.5);
    CHECK_FALSE(flag_for(cfg, truth));
  }
  {
    // The terms compose with AND: a boresight 165 deg from the Sun - clear
    // of that cone by any measure - is still excluded by the central body.
    star::sensors::StarTrackerCfg cfg;
    cfg.sun_exclusion_rad = kSun;
    cfg.central_body_exclusion_rad = kCb;
    cfg.boresight_b = from_central_body(15.0);
    CHECK_FALSE(flag_for(cfg, truth));
    cfg.boresight_b = from_sun(90.0);  // 90 deg from both directions
    CHECK(flag_for(cfg, truth));

    // Without an ephemeris there is no excluded direction to measure
    // against, so the same excluded geometry reports valid. This is the
    // configuration that made the block above unexecuted: a mission may
    // configure both radii and never evaluate either.
    star::sensors::SensorCycleTruth blind = truth;
    blind.geom.ephemeris_valid = false;
    cfg.boresight_b = from_central_body(15.0);
    CHECK(flag_for(cfg, blind));
  }
  writer.close();
  std::remove(path.c_str());
}

TEST_CASE("sensors_startracker_slew_limit_flag") {
  // The slew term of eq:optical:gating executes on every sample of every run
  // and its result was asserted nowhere; this asserts it both ways, matching
  // the standard the altimeter's band flag is held to above.
  // Fixture non-degeneracy: geom.ephemeris_valid is left false and both
  // exclusion radii are zero, so the flag is the slew comparison alone, and
  // the rates below straddle the configured limit rather than sitting on one
  // side of it.
  const std::string path = "test_sensors_st_slew.srlog";
  star::sensors::StarTrackerCfg cfg;
  cfg.slew_limit_radps = 0.01;
  star::sensors::StarTracker st(4, cfg, 4242);

  star::sensors::SensorCycleTruth truth;
  truth.dt_s = 0.25;
  truth.q_end_i2b = Eigen::Quaterniond::Identity();

  star::log::SrlogWriter writer =
      make_one_sensor_writer(path, "startracker", 4);
  truth.omega_b_end_radps = Eigen::Vector3d(0.006, 0.0, 0.0);
  st.accumulate(truth);
  st.sample(0.25, writer);
  CHECK(st.last_valid());

  truth.omega_b_end_radps = Eigen::Vector3d(0.0, 0.02, 0.0);
  st.accumulate(truth);
  st.sample(0.5, writer);
  CHECK_FALSE(st.last_valid());

  // The comparison is on the rate NORM: three components each individually
  // inside the limit whose norm (0.01386 rad/s) exceeds it are rejected.
  truth.omega_b_end_radps = Eigen::Vector3d(0.008, 0.008, 0.008);
  st.accumulate(truth);
  st.sample(0.75, writer);
  CHECK_FALSE(st.last_valid());

  // A zero limit is "no slew gate", not "reject everything".
  star::sensors::StarTrackerCfg ungated;
  star::sensors::StarTracker open(4, ungated, 4242);
  open.accumulate(truth);
  open.sample(1.0, writer);
  CHECK(open.last_valid());
  writer.close();
  std::remove(path.c_str());
}

TEST_CASE("sensors_sunsensor_validity_flag") {
  // eq:optical:sungate, whose three independent reasons to drop the flag -
  // no ephemeris, total umbra, and a Sun outside the field of view - were
  // computed on every sample and asserted by no test.
  // Fixture non-degeneracy: sigma_rad is zero, so the measured direction is
  // exactly the true one and the field-of-view angle is analytic; the truth
  // attitude is the identity and the Sun lies along +X, so the body-frame
  // Sun direction is +X and the separation from the boresight is the
  // boresight's own tilt. Each reason is exercised with the other two
  // satisfied, so no single condition can carry the result.
  const std::string path = "test_sensors_sun_gating.srlog";
  const double deg = star::constants::PI / 180.0;

  star::sensors::SensorCycleTruth truth;
  truth.dt_s = 0.25;
  truth.q_end_i2b = Eigen::Quaterniond::Identity();
  truth.r_end_i_m = Eigen::Vector3d(7.0e6, 0.0, 0.0);
  truth.v_end_i_mps = Eigen::Vector3d::Zero();
  truth.geom.ephemeris_valid = true;
  truth.geom.illumination_nu = 1.0;
  truth.geom.r_sun_m = Eigen::Vector3d(1.5e11, 0.0, 0.0);

  star::log::SrlogWriter writer =
      make_one_sensor_writer(path, "sunsensor", 4);
  double t = 0.25;
  const auto flag_for = [&](const star::sensors::SunSensorCfg& cfg,
                            const star::sensors::SensorCycleTruth& tr) {
    star::sensors::SunSensor ss(4, cfg, 112358);
    ss.accumulate(tr);
    ss.sample(t, writer);
    t += 0.25;
    return ss.last_valid();
  };

  star::sensors::SunSensorCfg cfg;
  cfg.fov_half_angle_rad = 20.0 * deg;
  const auto tilted = [deg](double a_deg) {
    return Eigen::Vector3d(std::cos(a_deg * deg), std::sin(a_deg * deg), 0.0);
  };

  cfg.boresight_b = tilted(10.0);
  CHECK(flag_for(cfg, truth));
  cfg.boresight_b = tilted(30.0);
  CHECK_FALSE(flag_for(cfg, truth));
  // Half a degree either side of the configured half-angle, which places the
  // transition and rules out a degrees-for-radians or full-angle reading.
  cfg.boresight_b = tilted(19.5);
  CHECK(flag_for(cfg, truth));
  cfg.boresight_b = tilted(20.5);
  CHECK_FALSE(flag_for(cfg, truth));

  // In-view but in total umbra: the illumination fraction shared with the
  // ch:srp shadow model gates the flag independently of geometry.
  cfg.boresight_b = tilted(10.0);
  star::sensors::SensorCycleTruth dark = truth;
  dark.geom.illumination_nu = 0.0;
  CHECK_FALSE(flag_for(cfg, dark));
  // Partial illumination counts as visible, so the penumbra is not a gap.
  dark.geom.illumination_nu = 0.25;
  CHECK(flag_for(cfg, dark));

  // Without an ephemeris there is no Sun direction to measure at all.
  star::sensors::SensorCycleTruth blind = truth;
  blind.geom.ephemeris_valid = false;
  CHECK_FALSE(flag_for(cfg, blind));

  // A zero half-angle is "no field-of-view gate", not "reject everything":
  // an illuminated sensor pointing away from the Sun still reports valid.
  star::sensors::SunSensorCfg wide;
  wide.boresight_b = tilted(150.0);
  CHECK(flag_for(wide, truth));
  writer.close();
  std::remove(path.c_str());
}

TEST_CASE("sensors_navfix_gauss_markov_correlated_errors") {
  // eq:radio:gm, the correlated component of the external nav fix. The whole
  // model - both stationary initializations, both recursions, and
  // advance_gm itself - was entered by no test in either tier, while its
  // three parameters are documented mission configuration.
  //
  // Fixture non-degeneracy: the WHITE standard deviations are zero, so the
  // measured fix minus truth is exactly the correlated component and the
  // model under test is the only thing the assertions can see. The draws for
  // the white terms are still consumed (they are unconditional), which is
  // what makes the schedule below the real one rather than a simplified one.
  const std::uint64_t seed = 90210;
  const std::uint32_t rate_hz = 4;
  const double dt = 1.0 / static_cast<double>(rate_hz);
  const double sigma_r = 40.0;
  const double tau_r = 10.0;
  const double sigma_v = 0.3;
  const double tau_v = 25.0;
  const Eigen::Vector3d r_true(7.0e6, 1.0e6, -2.0e6);
  const Eigen::Vector3d v_true(1.0e3, 7.4e3, 0.5e3);

  star::sensors::NavFixCfg cfg;
  cfg.gm_r.sigma = sigma_r;
  cfg.gm_r.tau_s = tau_r;
  cfg.gm_v.sigma = sigma_v;
  cfg.gm_v.tau_s = tau_v;
  star::sensors::NavFix nav(rate_hz, cfg, seed);

  star::sensors::SensorCycleTruth truth;
  truth.dt_s = dt;
  truth.r_end_i_m = r_true;
  truth.v_end_i_mps = v_true;

  // The coefficients are recomputed here from the chapter's relations rather
  // than read off the implementation.
  const double phi_r = std::exp(-dt / tau_r);
  const double phi_v = std::exp(-dt / tau_v);
  const double w_r = sigma_r * std::sqrt(1.0 - phi_r * phi_r);
  const double w_v = sigma_v * std::sqrt(1.0 - phi_v * phi_v);

  star::rng::NormalSampler ref(star::rng::make_stream(seed, "sensors.navfix"));
  // Construction-time stationary initialization, position then velocity
  // (ch:sensors-radio draw schedule).
  Eigen::Vector3d c_r;
  Eigen::Vector3d c_v;
  for (int i = 0; i < 3; ++i) c_r[i] = sigma_r * ref.next();
  for (int i = 0; i < 3; ++i) c_v[i] = sigma_v * ref.next();

  // Two phases against one continuous run of the sensor. The first replays
  // the reference stream sample by sample, which pins the recursion and the
  // draw schedule exactly; the second only records the sequence, so the
  // statistical assertions below cost a handful of checks rather than one
  // per sample.
  const int n_exact = 25;
  const int n_samples = 20000;
  std::vector<double> series_r;
  series_r.reserve(static_cast<std::size_t>(n_samples));
  const std::string path = "test_sensors_navfix_gm.srlog";
  {
    star::log::SrlogWriter writer =
        make_one_sensor_writer(path, "navfix", rate_hz);
    for (int k = 0; k < n_samples; ++k) {
      nav.accumulate(truth);
      nav.sample(dt * (k + 1), writer);
      if (k < n_exact) {
        // Per-sample schedule: position drive, velocity drive, position
        // white, velocity white. The white draws are consumed even at zero
        // sigma, so omitting them here would desynchronize the reference
        // stream and the second sample would already disagree.
        for (int i = 0; i < 3; ++i) c_r[i] = phi_r * c_r[i] + w_r * ref.next();
        for (int i = 0; i < 3; ++i) c_v[i] = phi_v * c_v[i] + w_v * ref.next();
        for (int i = 0; i < 3; ++i) (void)ref.next();
        for (int i = 0; i < 3; ++i) (void)ref.next();
        for (int i = 0; i < 3; ++i) {
          CHECK(nav.last_position_m()[i] ==
                doctest::Approx(r_true[i] + c_r[i]).epsilon(1e-12));
          CHECK(nav.last_velocity_mps()[i] ==
                doctest::Approx(v_true[i] + c_v[i]).epsilon(1e-12));
        }
      }
      series_r.push_back(nav.last_position_m()[0] - r_true[0]);
    }
    writer.close();
  }
  std::remove(path.c_str());

  // Stationarity: the sequence is initialized at the stationary variance and
  // driven by w_sigma = sigma * sqrt(1 - phi^2), so its sample variance sits
  // at sigma^2 with no drift. This is the independent check of the two
  // relations the exact comparison above assumes - a drive variance missing
  // the sqrt, or a zero initialization, changes this ratio by orders of
  // magnitude. 20000 samples at dt/tau = 0.025 is 500 correlation times.
  double sum = 0.0;
  double sum_sq = 0.0;
  double lag1 = 0.0;
  for (int k = 0; k < n_samples; ++k) {
    const double c = series_r[static_cast<std::size_t>(k)];
    sum += c;
    sum_sq += c * c;
    if (k + 1 < n_samples) lag1 += c * series_r[static_cast<std::size_t>(k + 1)];
  }
  const double mean = sum / n_samples;
  const double var = sum_sq / n_samples - mean * mean;
  CHECK(var / (sigma_r * sigma_r) >= 0.85);
  CHECK(var / (sigma_r * sigma_r) <= 1.15);
  // Lag-one autocorrelation recovers phi = exp(-dt/tau), which is the
  // parameter tau_s actually reaching the recursion.
  CHECK(lag1 / sum_sq == doctest::Approx(phi_r).epsilon(0.02));

  // Both components are independently switchable: a zero tau leaves the
  // correlated term off, and the fix is then truth exactly at zero white
  // sigma - the case every existing test ran without knowing it.
  star::sensors::NavFixCfg off;
  off.gm_r.sigma = sigma_r;  // sigma alone does not enable the model
  star::sensors::NavFix plain(rate_hz, off, seed);
  {
    star::log::SrlogWriter writer =
        make_one_sensor_writer(path, "navfix", rate_hz);
    plain.accumulate(truth);
    plain.sample(dt, writer);
    writer.close();
  }
  std::remove(path.c_str());
  for (int i = 0; i < 3; ++i) {
    CHECK(plain.last_position_m()[i] == r_true[i]);
    CHECK(plain.last_velocity_mps()[i] == v_true[i]);
  }
}

TEST_CASE("sensors_parsers_reject_an_unknown_parameter_name") {
  // The reject_unknown allow-list in each sensor parser. A silently ignored
  // typo is the failure mode this guards: the named term stays at its
  // default and the run looks entirely plausible, which is a different and
  // worse outcome than a confusing error message.
  // Fixture non-degeneracy: each case starts from a configuration the parser
  // ACCEPTS - asserted first - and then adds one misspelling of that
  // sensor's own vocabulary, so a rejection can only come from the unknown
  // name and not from an otherwise invalid config.
  struct Case {
    const char* kind;
    const char* typo;
  };
  const Case cases[] = {
      {"startracker", "sun_exclusion_deg"},   // real name ends _rad
      {"sunsensor", "fov_half_angle_deg"},    // real name ends _rad
      {"navfix", "gm_position_tau"},          // real name ends _tau_s
      {"altimeter", "sigma_noise_meters"},    // real name ends _m
      {"camera", "focal_px"},                 // real name is fx_px / fy_px
      {"imu", "arw_rad_per_sqrt_hour"},       // not the configured spelling
  };
  for (const Case& c : cases) {
    star::gnc::GncSensorCfg base;
    base.kind = c.kind;
    base.sample_rate_hz = 10;
    if (base.kind == "camera") {
      base.scalars["fx_px"] = 800.0;
      base.scalars["fy_px"] = 800.0;
      base.scalars["width_px"] = 1024.0;
      base.scalars["height_px"] = 768.0;
    }
    CHECK_NOTHROW(star::sensors::make_sensor(base, 42));

    star::gnc::GncSensorCfg typo = base;
    typo.scalars[c.typo] = 1.0;
    CHECK_THROWS_AS(star::sensors::make_sensor(typo, 42),
                    std::invalid_argument);
  }

  // Vector-valued parameters go through the same allow-list.
  star::gnc::GncSensorCfg vec;
  vec.kind = "startracker";
  vec.sample_rate_hz = 10;
  vec.vectors["boresight_body"] = {0.0, 0.0, 1.0};  // real name is boresight_b
  CHECK_THROWS_AS(star::sensors::make_sensor(vec, 42), std::invalid_argument);
}
