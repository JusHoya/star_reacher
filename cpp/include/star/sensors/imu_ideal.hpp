// Ideal IMU (FR-23 reference implementation): accumulated angle and
// velocity increments with zero errors.
//
//   dtheta = integral of the true body rate over the sample interval
//   dv     = integral of the true specific force (body frame) over the
//            sample interval
//
// The loop's kinematics are zero-order-held per control cycle (D-5), so the
// integrals reduce to exact sums of held values:
//   dtheta = sum_cycles omega_b * dt,   dv = sum_cycles sf_b * dt
// - "exact" meaning the increments are the exact integrals of the loop's
// piecewise-constant rate and specific-force histories (one rounding per
// product and sum; no quadrature error). In the torque-driven attitude mode
// the body rate additionally varies inside a cycle; the sensor layer
// samples the cycle-start hold, so the within-cycle variation is not
// resolved (bounded by |omega_dot| dt^2 / 2 per cycle - the documented
// v1.2 sensor-truth scheme, format doc section 3.2).
//
// The full FR-23 IMU error model (turn-on bias, Gauss-Markov in-run bias,
// scale factor, misalignment, ARW/VRW, quantization, coning/sculling
// preservation) lands in a later workstream against this same ISensor
// interface; the ideal IMU consumes zero draws from its D-9 stream.
#ifndef STAR_SENSORS_IMU_IDEAL_HPP
#define STAR_SENSORS_IMU_IDEAL_HPP

#include <cstdint>

#include <Eigen/Dense>

#include "star/gnc/component.hpp"
#include "star/sensors/sensor.hpp"

namespace star {
namespace sensors {

class IdealImu final : public ISensor {
 public:
  explicit IdealImu(std::uint32_t sample_rate_hz);

  const char* kind() const override { return "imu"; }
  std::uint32_t sample_rate_hz() const override { return rate_hz_; }
  void accumulate(const SensorCycleTruth& truth) override;
  void sample(double t_s, log::SrlogWriter& writer) override;

  // Latest emitted sample, for the loop's typed GncInput wiring (valid is
  // false until the first sample instant). Later sensor kinds expose their
  // own typed accessors the same way - no ISensor change required.
  const gnc::ImuSample& last_sample() const { return last_; }

 private:
  std::uint32_t rate_hz_;
  Eigen::Vector3d dtheta_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d dv_ = Eigen::Vector3d::Zero();
  double accum_dt_ = 0.0;
  gnc::ImuSample last_;
};

}  // namespace sensors
}  // namespace star

#endif  // STAR_SENSORS_IMU_IDEAL_HPP
