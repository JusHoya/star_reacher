// IMU sensor (FR-23, ch:sensors-imu): accumulated angle and velocity
// increments carrying the full seeded error chain.
//
// Truth increments (eq:imu:dtheta) are
//
//   dtheta_k = integral over the sample interval of the true body rate
//   dv_k     = integral over the sample interval of the true specific
//              force in body axes
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
// The measured increments then follow eq:imu:gyro / eq:imu:accel,
//
//   dtheta~ = Q_q[ (I + M_g) dtheta + (b_g0 + b_gk) dt + eta_g ],
//
// with the terms applied in exactly that order (ch:sensors-imu
// implementation note 2): Gauss-Markov advance (eq:imu:gm), then the
// linear distortion, bias, and noise sum, then the carry-preserving
// quantizer (eq:imu:quant). A configuration with every error coefficient
// zero reproduces the ideal (zero-error) increments bit-for-bit - the
// quantizer, the bias terms, and the noise terms each degenerate to an
// exact identity - which is the model's zero-error special case and is
// pinned by a differential test.
#ifndef STAR_SENSORS_IMU_HPP
#define STAR_SENSORS_IMU_HPP

#include <cstdint>

#include <Eigen/Dense>

#include "star/gnc/component.hpp"
#include "star/gnc/config.hpp"
#include "star/rng.hpp"
#include "star/sensors/sensor.hpp"

namespace star {
namespace sensors {

// Error coefficients for one instrument triad (gyro or accelerometer). All
// SI: the gyro's bias/noise units are rad/s and rad/sqrt(s), the
// accelerometer's m/s^2 and (m/s)/sqrt(s). Every coefficient defaults to
// zero, so a default-constructed config is the ideal instrument.
struct ImuTriadErrorCfg {
  // Turn-on bias standard deviation (eq:imu:turnon); one draw per run.
  double turnon_bias_sigma = 0.0;
  // Data-sheet bias instability B (eq:imu:bi). The process strength is
  // sigma_GM = 1.0760 * B by the preset mapping of eq:imu:presetmap, so a
  // configured B is what the conventional ADEV read-out returns.
  double bias_instability = 0.0;
  // Gauss-Markov correlation time tau_c [s] (eq:imu:gm). Non-positive
  // disables the in-run bias state while leaving the draw schedule intact.
  double bias_tau_s = 0.0;
  // Random-walk coefficient N (eq:imu:arw): ARW for the gyro, VRW for the
  // accelerometer. The increment noise sigma is N * sqrt(dt).
  double random_walk = 0.0;
  // Output quantum q (eq:imu:quant); zero disables the quantizer exactly.
  double quantum = 0.0;
  // Combined linear distortion M = S + Gamma (eq:imu:mis): the diagonal
  // carries the dimensionless scale-factor errors, the off-diagonal the
  // six small-angle misalignment entries [rad].
  Eigen::Matrix3d distortion = Eigen::Matrix3d::Zero();
};

struct ImuErrorCfg {
  ImuTriadErrorCfg gyro;
  ImuTriadErrorCfg accel;
};

// Parse the FR-23 IMU error coefficients out of a resolved sensor config's
// flat parameter maps. Unit-suffixed keys per DX-3; every key is optional
// and defaults to the ideal instrument. Throws std::invalid_argument naming
// the offending key on an unknown key, a wrong-length vector, or a negative
// sigma/quantum/correlation time - a defensive re-check of what the FR-15
// validator already enforces, not the user-facing error path.
ImuErrorCfg parse_imu_error_cfg(const gnc::GncSensorCfg& cfg);

// Gauss-Markov process strength from the data-sheet bias instability
// (eq:imu:presetmap): sigma_GM = (0.664282 / 0.617364) B = 1.0760 B, the
// ratio of the flicker-noise flat-region coefficient of eq:imu:bi to the
// Gauss-Markov ADEV peak value of eq:imu:gmpeak, so the model's ADEV peaks
// at exactly 0.664 B and the conventional read-out returns the configured
// coefficient.
double gm_sigma_from_bias_instability(double bias_instability);

class Imu final : public ISensor {
 public:
  // master_seed derives the D-9 "sensors.imu" stream; construction consumes
  // the twelve initialization draws of the normative schedule.
  Imu(std::uint32_t sample_rate_hz, const ImuErrorCfg& err,
      std::uint64_t master_seed);

  const char* kind() const override { return "imu"; }
  std::uint32_t sample_rate_hz() const override { return rate_hz_; }
  void accumulate(const SensorCycleTruth& truth) override;
  void sample(double t_s, log::SrlogWriter& writer) override;

  // Latest emitted sample, for the loop's typed GncInput wiring (valid is
  // false until the first sample instant).
  const gnc::ImuSample& last_sample() const { return last_; }

  // True in-run Gauss-Markov bias states after the most recent sample, the
  // truth-side diagnostic of ch:sensors-imu implementation note 4: the
  // consistency evaluation forms full-state estimation errors from these
  // without reaching into core internals.
  const Eigen::Vector3d& gyro_bias_radps() const { return gyro_.b; }
  const Eigen::Vector3d& accel_bias_mps2() const { return accel_.b; }

  // Total true bias: the turn-on draw plus the in-run Gauss-Markov state
  // (eq:imu:gyro / eq:imu:accel apply their sum). This, not the
  // Gauss-Markov part alone, is the quantity a filter's bias state
  // estimates, so it is what a truth-minus-estimate bias error must
  // difference against. The two agree exactly when the scenario disables
  // turn-on biases, which is the reference consistency scenario's
  // configuration (ch:ekf assumption 3).
  Eigen::Vector3d gyro_total_bias_radps() const { return gyro_.b0 + gyro_.b; }
  Eigen::Vector3d accel_total_bias_mps2() const {
    return accel_.b0 + accel_.b;
  }

 private:
  // One instrument's error coefficients and per-sample error state.
  struct Triad {
    Eigen::Matrix3d distortion = Eigen::Matrix3d::Zero();
    Eigen::Vector3d b0 = Eigen::Vector3d::Zero();      // turn-on bias
    Eigen::Vector3d b = Eigen::Vector3d::Zero();       // GM in-run bias
    Eigen::Vector3d carry = Eigen::Vector3d::Zero();   // quantizer residual
    double phi = 0.0;          // exp(-dt/tau_c), eq:imu:gm
    double w_sigma = 0.0;      // sigma_GM * sqrt(1 - phi^2), eq:imu:gm
    double noise_sigma = 0.0;  // N * sqrt(dt), eq:imu:arw
    double quantum = 0.0;      // q, eq:imu:quant
  };

  // Advance one triad's Gauss-Markov state and form its measured increment.
  Eigen::Vector3d measure(Triad& tri, const Eigen::Vector3d& truth,
                          double dt_s);

  std::uint32_t rate_hz_;
  rng::NormalSampler normals_;
  Triad gyro_;
  Triad accel_;
  Eigen::Vector3d dtheta_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d dv_ = Eigen::Vector3d::Zero();
  double accum_dt_ = 0.0;
  gnc::ImuSample last_;
};

}  // namespace sensors
}  // namespace star

#endif  // STAR_SENSORS_IMU_HPP
