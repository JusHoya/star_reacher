// FR-23 sensor abstraction: core-side ISensor modules sampled on the
// control-cycle grid (D-5 major cycle), fed truth kinematics/dynamics
// context by the loop, emitting records into their `sensors.<kind>` SRLOG
// group (format doc section 3.2).
//
// Scheduling contract (normative here; the format doc states the on-disk
// consequence): a sensor declares a sample rate that must be an exact
// integer divisor of the control rate, so it is sampled every
// control_rate_hz / sample_rate_hz cycles, on the cycle grid. Each cycle the
// loop calls accumulate() once with that cycle's zero-order-held truth;
// at the sensor's sample instants (t = k / sample_rate_hz after GNC
// activation, k >= 1) the loop calls sample(), which emits the record and
// resets any accumulation state. Determinism (D-10): sensors draw
// randomness only from their D-9 named stream ("sensors.<kind>"), seeded
// from the master seed at construction; the ideal IMU consumes zero draws,
// but the seeding plumbing is part of this interface so the FR-23 error
// models land later with no interface change (they add config fields of
// their own through the plain-data GncSensorCfg, D-2).
#ifndef STAR_SENSORS_SENSOR_HPP
#define STAR_SENSORS_SENSOR_HPP

#include <cstdint>
#include <memory>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "star/gnc/config.hpp"

namespace star {
namespace log {
class SrlogWriter;
}  // namespace log

namespace sensors {

// One control cycle's zero-order-held truth, captured at the cycle start
// and held over [t_s, t_s + dt_s) (D-5). omega_b_radps is the cycle's held
// body rate; sf_b_mps2 is the cycle's held specific force in the body frame
// (body surface forces / mass; domain bound in the format doc section 3.2).
struct SensorCycleTruth {
  double t_s = 0.0;
  double dt_s = 0.0;
  Eigen::Vector3d r_i_m = Eigen::Vector3d::Zero();
  Eigen::Vector3d v_i_mps = Eigen::Vector3d::Zero();
  Eigen::Quaterniond q_i2b = Eigen::Quaterniond::Identity();
  Eigen::Vector3d omega_b_radps = Eigen::Vector3d::Zero();
  Eigen::Vector3d sf_b_mps2 = Eigen::Vector3d::Zero();
};

class ISensor {
 public:
  virtual ~ISensor() = default;

  // Canonical sensor kind (log::kSensorKinds); names the SRLOG group.
  virtual const char* kind() const = 0;

  // Declared sample rate [Hz]; the loop validated divisibility against the
  // control rate before construction.
  virtual std::uint32_t sample_rate_hz() const = 0;

  // Integrate one control cycle of held truth into the sensor's internal
  // accumulation state. Called exactly once per cycle, in cycle order.
  virtual void accumulate(const SensorCycleTruth& truth) = 0;

  // Emit the sample for the instant t_s into the sensor's SRLOG group and
  // reset accumulation state. Called only at the sensor's sample instants.
  virtual void sample(double t_s, log::SrlogWriter& writer) = 0;
};

// Instantiate a sensor from its resolved config (D-2 plain data). The
// master seed derives the sensor's D-9 named stream; the ideal IMU stores
// no generator (zero error draws) but every sensor kind constructs through
// this one signature. Unknown kinds throw std::invalid_argument naming the
// supported set (Phase 6 WS1: "imu").
std::unique_ptr<ISensor> make_sensor(const gnc::GncSensorCfg& cfg,
                                     std::uint64_t master_seed);

}  // namespace sensors
}  // namespace star

#endif  // STAR_SENSORS_SENSOR_HPP
