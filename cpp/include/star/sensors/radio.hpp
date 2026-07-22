// Radio-class sensors (FR-23, A-6, ch:sensors-radio): the external nav fix -
// the project's generalization of GNSS, an abstract service delivering GCRF
// position and velocity fixes - and the altimeter, a scalar geodetic-altitude
// measurement over the central body's reference ellipsoid.
#ifndef STAR_SENSORS_RADIO_HPP
#define STAR_SENSORS_RADIO_HPP

#include <cstdint>

#include <Eigen/Dense>

#include "star/gnc/config.hpp"
#include "star/rng.hpp"
#include "star/sensors/sensor.hpp"

namespace star {
namespace sensors {

// One optional first-order Gauss-Markov error component (eq:radio:gm),
// following exactly the recursion of ch:sensors-imu equation eq:imu:gm.
// Defaulted off: sigma == 0 or tau <= 0 leaves the component identically
// zero while its draws are still consumed.
struct GaussMarkovCfg {
  double sigma = 0.0;
  double tau_s = 0.0;
};

struct NavFixCfg {
  // Per-GCRF-axis white error standard deviations (eq:radio:white).
  Eigen::Vector3d sigma_r_m = Eigen::Vector3d::Zero();
  Eigen::Vector3d sigma_v_mps = Eigen::Vector3d::Zero();
  GaussMarkovCfg gm_r;  // correlated position component, defaulted off
  GaussMarkovCfg gm_v;  // correlated velocity component, defaulted off
};

NavFixCfg parse_nav_fix_cfg(const gnc::GncSensorCfg& cfg);

class NavFix final : public ISensor {
 public:
  NavFix(std::uint32_t sample_rate_hz, const NavFixCfg& cfg,
         std::uint64_t master_seed);

  const char* kind() const override { return "navfix"; }
  std::uint32_t sample_rate_hz() const override { return rate_hz_; }
  void accumulate(const SensorCycleTruth& truth) override;
  void sample(double t_s, log::SrlogWriter& writer) override;

  const Eigen::Vector3d& last_position_m() const { return r_meas_; }
  const Eigen::Vector3d& last_velocity_mps() const { return v_meas_; }
  // The fix service carries no operating-band or exclusion gate, so validity
  // here reports only that the held pair is a measurement at all: before the
  // first sample() the accessors above return the zero vectors they were
  // constructed with, which is not a position anywhere near a central body.
  bool last_valid() const { return sampled_; }

 private:
  // Advance one three-axis Gauss-Markov component (eq:radio:gm) in place.
  void advance_gm(Eigen::Vector3d& c, double phi, double w_sigma);

  std::uint32_t rate_hz_;
  NavFixCfg cfg_;
  rng::NormalSampler normals_;
  SensorCycleTruth latest_;
  double phi_r_ = 0.0;
  double phi_v_ = 0.0;
  double w_sigma_r_ = 0.0;
  double w_sigma_v_ = 0.0;
  Eigen::Vector3d c_r_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d c_v_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d r_meas_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d v_meas_ = Eigen::Vector3d::Zero();
  bool sampled_ = false;
};

struct AltimeterCfg {
  double sigma_bias_m = 0.0;   // turn-on bias sigma, one draw per run
  double sigma_noise_m = 0.0;  // per-sample white noise sigma
  // Operating band (eq:radio:altgate). h_max <= h_min disables the gate,
  // which is how a scenario says "always in band".
  double h_min_m = 0.0;
  double h_max_m = 0.0;
};

AltimeterCfg parse_altimeter_cfg(const gnc::GncSensorCfg& cfg);

class Altimeter final : public ISensor {
 public:
  Altimeter(std::uint32_t sample_rate_hz, const AltimeterCfg& cfg,
            std::uint64_t master_seed);

  const char* kind() const override { return "altimeter"; }
  std::uint32_t sample_rate_hz() const override { return rate_hz_; }
  void accumulate(const SensorCycleTruth& truth) override;
  void sample(double t_s, log::SrlogWriter& writer) override;

  double last_measurement_m() const { return alt_meas_; }
  bool last_valid() const { return valid_; }
  double turnon_bias_m() const { return bias_m_; }

 private:
  std::uint32_t rate_hz_;
  AltimeterCfg cfg_;
  rng::NormalSampler normals_;
  SensorCycleTruth latest_;
  double bias_m_ = 0.0;
  double alt_meas_ = 0.0;
  bool valid_ = false;
};

}  // namespace sensors
}  // namespace star

#endif  // STAR_SENSORS_RADIO_HPP
