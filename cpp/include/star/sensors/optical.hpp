// Optical attitude sensors (FR-23, ch:sensors-optical): the star tracker and
// the sun sensor, plus the velocity-aberration correction this project
// applies to EVERY optical truth direction.
//
// Aberration is specified once here and consumed by both sensors and by the
// camera hook (ch:camera) through the single function aberrate() - one code
// path is what makes Phase 6 exit criterion 9 a single gate rather than
// three independently drifting implementations.
#ifndef STAR_SENSORS_OPTICAL_HPP
#define STAR_SENSORS_OPTICAL_HPP

#include <cstdint>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "star/gnc/config.hpp"
#include "star/rng.hpp"
#include "star/sensors/sensor.hpp"

namespace star {
namespace sensors {

// Observer velocity over c for the aberration formula (eq:optical:beta):
// beta = (v_sc + v_cb/SSB) / c, GCRF axes. The barycentric composition is
// required because catalog directions are defined in the barycentric (ICRS)
// frame; the planet-relative velocity alone would omit the dominant annual
// term of about 30 km/s.
Eigen::Vector3d aberration_beta(const Eigen::Vector3d& v_sc_i_mps,
                                const Eigen::Vector3d& v_central_ssb_mps);

// First-order velocity aberration, eq:optical:aberration:
//
//   u' = normalize(u + beta - (u . beta) u)
//
// the geometric direction plus the component of beta perpendicular to it,
// renormalized. The apparent direction is displaced from the geometric one
// TOWARD the observer's velocity, by beta sin(theta) to first order
// (eq:optical:abmag) - 20.49 arcsec at Earth's mean heliocentric speed of
// 29.78 km/s. `u_hat` must be a unit vector; a zero-length beta returns u
// unchanged. The neglected second-order term is at most 0.52 mas at
// beta = 1e-4 (ch:sensors-optical domain item 2).
Eigen::Vector3d aberrate(const Eigen::Vector3d& u_hat,
                         const Eigen::Vector3d& beta);

// Angle between two vectors by the numerically well-conditioned form
// atan2(norm(a x b), a . b), used for every gating comparison.
double angle_between(const Eigen::Vector3d& a, const Eigen::Vector3d& b);

// --- star tracker ---------------------------------------------------------

struct StarTrackerCfg {
  Eigen::Vector3d boresight_b = Eigen::Vector3d::UnitZ();  // unit, body axes
  // Per-axis error standard deviations [rad], body axes (about-boresight is
  // typically the largest).
  Eigen::Vector3d sigma_rad = Eigen::Vector3d::Zero();
  // Exclusion half-angles [rad] against the apparent Sun direction and the
  // central body's apparent direction; each is understood to include the
  // body's angular radius plus stray-light margin (ch:sensors-optical
  // assumption 5). A non-positive value disables that exclusion.
  double sun_exclusion_rad = 0.0;
  double central_body_exclusion_rad = 0.0;
  // Slew-rate limit [rad/s]; non-positive disables the rate gate.
  double slew_limit_radps = 0.0;
};

StarTrackerCfg parse_star_tracker_cfg(const gnc::GncSensorCfg& cfg);

class StarTracker final : public ISensor {
 public:
  StarTracker(std::uint32_t sample_rate_hz, const StarTrackerCfg& cfg,
              std::uint64_t master_seed);

  const char* kind() const override { return "startracker"; }
  std::uint32_t sample_rate_hz() const override { return rate_hz_; }
  void accumulate(const SensorCycleTruth& truth) override;
  void sample(double t_s, log::SrlogWriter& writer) override;

  const Eigen::Quaterniond& last_measurement() const { return q_meas_; }
  bool last_valid() const { return valid_; }

 private:
  std::uint32_t rate_hz_;
  StarTrackerCfg cfg_;
  rng::NormalSampler normals_;
  SensorCycleTruth latest_;  // most recent cycle's truth (sampled at instants)
  Eigen::Quaterniond q_meas_ = Eigen::Quaterniond::Identity();
  bool valid_ = false;
};

// --- sun sensor -----------------------------------------------------------

struct SunSensorCfg {
  Eigen::Vector3d boresight_b = Eigen::Vector3d::UnitZ();
  double fov_half_angle_rad = 0.0;  // non-positive disables the FOV gate
  double sigma_rad = 0.0;
};

SunSensorCfg parse_sun_sensor_cfg(const gnc::GncSensorCfg& cfg);

class SunSensor final : public ISensor {
 public:
  SunSensor(std::uint32_t sample_rate_hz, const SunSensorCfg& cfg,
            std::uint64_t master_seed);

  const char* kind() const override { return "sunsensor"; }
  std::uint32_t sample_rate_hz() const override { return rate_hz_; }
  void accumulate(const SensorCycleTruth& truth) override;
  void sample(double t_s, log::SrlogWriter& writer) override;

  const Eigen::Vector3d& last_measurement() const { return u_meas_; }
  bool last_valid() const { return valid_; }

 private:
  std::uint32_t rate_hz_;
  SunSensorCfg cfg_;
  rng::NormalSampler normals_;
  SensorCycleTruth latest_;
  Eigen::Vector3d u_meas_ = Eigen::Vector3d::Zero();
  bool valid_ = false;
};

}  // namespace sensors
}  // namespace star

#endif  // STAR_SENSORS_OPTICAL_HPP
