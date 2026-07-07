// Ideal IMU implementation (contracts in sensors/imu_ideal.hpp).
#include "star/sensors/imu_ideal.hpp"

#include <stdexcept>

#include "star/srlog_writer.hpp"

namespace star {
namespace sensors {

IdealImu::IdealImu(std::uint32_t sample_rate_hz) : rate_hz_(sample_rate_hz) {
  if (rate_hz_ == 0) {
    throw std::invalid_argument("IdealImu: sample_rate_hz must be >= 1");
  }
}

void IdealImu::accumulate(const SensorCycleTruth& truth) {
  // Exact integrals of the held per-cycle values (imu_ideal.hpp): one
  // product and one sum per component per cycle, fixed order (D-10).
  dtheta_ += truth.omega_b_radps * truth.dt_s;
  dv_ += truth.sf_b_mps2 * truth.dt_s;
  accum_dt_ += truth.dt_s;
}

void IdealImu::sample(double t_s, log::SrlogWriter& writer) {
  writer.write_sensor_imu(t_s, dtheta_, dv_);
  last_.valid = true;
  last_.t_s = t_s;
  last_.dt_s = accum_dt_;
  last_.dtheta_b_rad = dtheta_;
  last_.dv_b_mps = dv_;
  dtheta_ = Eigen::Vector3d::Zero();
  dv_ = Eigen::Vector3d::Zero();
  accum_dt_ = 0.0;
}

std::unique_ptr<ISensor> make_sensor(const gnc::GncSensorCfg& cfg,
                                     std::uint64_t /*master_seed*/) {
  // The master seed threads through so error-bearing sensors can derive
  // their D-9 named stream ("sensors.<kind>") here without a signature
  // change; the ideal IMU consumes zero draws.
  if (cfg.kind == "imu") {
    return std::unique_ptr<ISensor>(new IdealImu(cfg.sample_rate_hz));
  }
  throw std::invalid_argument(
      "make_sensor: unknown sensor kind '" + cfg.kind +
      "'; supported in this phase: {imu} (the remaining FR-23 kinds land "
      "with the sensor error-model workstream)");
}

}  // namespace sensors
}  // namespace star
