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

#include "star/gnc/config.hpp"
#include "star/rng.hpp"
#include "star/sensors/imu.hpp"
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

TEST_CASE("sensors_factory_kind_vocabulary") {
  star::gnc::GncSensorCfg cfg;
  cfg.kind = "imu";
  cfg.sample_rate_hz = 100;
  const auto imu = star::sensors::make_sensor(cfg, 42);
  CHECK(std::string(imu->kind()) == "imu");
  CHECK(imu->sample_rate_hz() == 100);

  // The remaining FR-23 kinds land in a later workstream; until then the
  // factory refuses them by name rather than returning a stub.
  star::gnc::GncSensorCfg bad;
  bad.kind = "startracker";
  bad.sample_rate_hz = 10;
  CHECK_THROWS_AS(star::sensors::make_sensor(bad, 42), std::invalid_argument);

  star::gnc::GncSensorCfg zero;
  zero.kind = "imu";
  zero.sample_rate_hz = 0;
  CHECK_THROWS_AS(star::sensors::make_sensor(zero, 42),
                  std::invalid_argument);
}
