// Ideal IMU (FR-23 reference implementation, ch:sensors-imu): accumulated
// angle and velocity increments with zero errors,
//
//   dtheta_k = integral over the sample interval of the true body rate
//   dv_k     = integral over the sample interval of the true specific
//              force in body axes                       (eq:imu:dtheta)
//
// evaluated by TRAPEZOIDAL accumulation over the accepted integrator steps
// tiling the interval (eq:imu:quadrature): for steps of size h_j with
// endpoint values x_j, x_j+1,
//
//   increment += sum_j (h_j / 2) (x_j + x_j+1),
//
// which carries intra-interval motion (an increment interface, distinct
// from a point-sampled rate times dt) with local error bounded by
// eq:imu:quaderr. The loop supplies the endpoint values per cycle through
// SensorCycleTruth (one accepted rk4 step per cycle in this phase, D-5).
// The v1 IMU samples exactly once per major cycle: sample_rate_hz must
// equal the control rate (ch:sensors-imu assumption 1).
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
