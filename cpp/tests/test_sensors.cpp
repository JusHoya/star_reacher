// FR-23 sensor-layer unit tests: the ideal IMU's exact accumulation of the
// loop's held per-cycle kinematics, its sample/reset semantics, and the
// sensor factory's kind vocabulary. Written against the contracts in
// sensors/imu_ideal.hpp; record-level byte layout is covered by
// test_srlog.cpp, and in-loop scheduling by test_gnc_cycle.cpp.
#include <cstdio>
#include <stdexcept>
#include <string>

#include <Eigen/Dense>

#include "star/gnc/config.hpp"
#include "star/sensors/imu_ideal.hpp"
#include "star/sensors/sensor.hpp"
#include "star/srlog_writer.hpp"
#include "vendor/doctest.h"

namespace {

// A writer with only sensors.imu declared, for sample() calls that need a
// record sink; the emitted bytes themselves are covered by test_srlog.cpp.
star::log::SrlogWriter make_imu_writer(const std::string& path) {
  star::log::SrlogHeaderFields f;
  f.core_version = "0.6.0-test";
  f.git_hash = "unknown";
  f.config_sha256 = std::string(64, '0');
  f.master_seed = 1;
  f.oracle = false;
  f.epoch_utc = "2026-01-01T00:00:00Z";
  f.central_body = "earth";
  f.truth_rate_hz = 10;
  f.cycle_rate_hz = 10;
  f.sensors = {{"imu", 10, 0}};
  return star::log::SrlogWriter(path, f);
}

}  // namespace

TEST_CASE("sensors_ideal_imu_trapezoidal_accumulation_and_reset") {
  star::sensors::IdealImu imu(10);
  CHECK(std::string(imu.kind()) == "imu");
  CHECK(imu.sample_rate_hz() == 10);
  CHECK_FALSE(imu.last_sample().valid);

  // Endpoint values and dt chosen exactly representable in binary64, so
  // the trapezoidal sums (h/2)(x_start + x_end) are exact and asserted
  // with bit equality (eq:imu:quadrature; sensors/imu_ideal.hpp).
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
