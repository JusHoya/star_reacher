// Optical sensor implementation (contracts in sensors/optical.hpp, model in
// ch:sensors-optical).
#include "star/sensors/optical.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "star/constants.hpp"
#include "star/rotation.hpp"
#include "star/srlog_writer.hpp"

namespace star {
namespace sensors {

namespace {

double scalar_or(const gnc::GncSensorCfg& cfg, const char* key, double dflt) {
  const auto it = cfg.scalars.find(key);
  return it == cfg.scalars.end() ? dflt : it->second;
}

double nonneg(const gnc::GncSensorCfg& cfg, const char* key, const char* kind) {
  const double v = scalar_or(cfg, key, 0.0);
  if (v < 0.0) {
    throw std::invalid_argument("sensors." + std::string(kind) + ": " +
                                std::string(key) + " must be >= 0");
  }
  return v;
}

void reject_unknown(const gnc::GncSensorCfg& cfg, const char* kind,
                    const std::vector<std::string>& scalars,
                    const std::vector<std::string>& vectors) {
  // A typo must not silently disable the term it names (DX-2).
  for (const auto& kv : cfg.scalars) {
    bool known = false;
    for (const auto& k : scalars) known = known || kv.first == k;
    if (!known) {
      throw std::invalid_argument("sensors." + std::string(kind) +
                                  ": unknown parameter '" + kv.first + "'");
    }
  }
  for (const auto& kv : cfg.vectors) {
    bool known = false;
    for (const auto& k : vectors) known = known || kv.first == k;
    if (!known) {
      throw std::invalid_argument("sensors." + std::string(kind) +
                                  ": unknown parameter '" + kv.first + "'");
    }
  }
}

// A configured direction must be a usable unit vector; a zero vector would
// make every downstream angle undefined, so it is refused rather than
// silently replaced by a default axis.
Eigen::Vector3d unit_vector(const gnc::GncSensorCfg& cfg, const char* key,
                            const char* kind, const Eigen::Vector3d& dflt) {
  const auto it = cfg.vectors.find(key);
  if (it == cfg.vectors.end()) return dflt;
  if (it->second.size() != 3) {
    throw std::invalid_argument("sensors." + std::string(kind) + ": " +
                                std::string(key) +
                                " must have exactly 3 entries");
  }
  const Eigen::Vector3d v(it->second[0], it->second[1], it->second[2]);
  const double n = v.norm();
  if (n <= 0.0) {
    throw std::invalid_argument("sensors." + std::string(kind) + ": " +
                                std::string(key) + " must be nonzero");
  }
  return v / n;
}

Eigen::Vector3d sigma_vector(const gnc::GncSensorCfg& cfg, const char* key,
                             const char* kind) {
  const auto it = cfg.vectors.find(key);
  if (it == cfg.vectors.end()) return Eigen::Vector3d::Zero();
  if (it->second.size() != 3) {
    throw std::invalid_argument("sensors." + std::string(kind) + ": " +
                                std::string(key) +
                                " must have exactly 3 entries");
  }
  const Eigen::Vector3d v(it->second[0], it->second[1], it->second[2]);
  if (v.minCoeff() < 0.0) {
    throw std::invalid_argument("sensors." + std::string(kind) + ": " +
                                std::string(key) + " entries must be >= 0");
  }
  return v;
}

// Exact exponential map of a rotation vector to a unit quaternion
// (eq:optical:noiseq / eq:optical:qab): [cos(th/2), sin(th/2) * v/th], with
// the identity at th == 0. Using the exact map rather than a small-angle
// construction is what makes the star tracker's acceptance statistic
// exactly chi-square: the log map of eq:optical:extract recovers the drawn
// rotation vector identically.
Eigen::Quaterniond quat_exp(const Eigen::Vector3d& v) {
  const double theta = v.norm();
  if (theta == 0.0) return Eigen::Quaterniond::Identity();
  const double half = 0.5 * theta;
  const double s = std::sin(half) / theta;
  return Eigen::Quaterniond(std::cos(half), s * v.x(), s * v.y(), s * v.z());
}

}  // namespace

Eigen::Vector3d aberration_beta(const Eigen::Vector3d& v_sc_i_mps,
                                const Eigen::Vector3d& v_central_ssb_mps) {
  // eq:optical:beta: beta = (v_sc + v_cb/SSB) / c, GCRF axes.
  return (v_sc_i_mps + v_central_ssb_mps) / constants::SPEED_OF_LIGHT_M_PER_S;
}

Eigen::Vector3d aberrate(const Eigen::Vector3d& u_hat,
                         const Eigen::Vector3d& beta) {
  // eq:optical:aberration: u' = normalize(u + beta - (u . beta) u), the
  // geometric direction plus the component of beta perpendicular to it.
  // The apparent direction is displaced TOWARD the velocity.
  //
  // First order is NORMATIVE here, not an approximation awaiting an upgrade
  // (ch:sensors-optical domain item 2 declares this equation THE formula and
  // exit criterion 9 recomputes THIS form independently). The exact
  // relativistic form differs by (beta^2/4) sin(2 theta): 0.51 mas at Earth's
  // mean heliocentric speed and 0.81 mas once LEO speed is added - which
  // would spend most of criterion 9's 1 mas budget on a formula change that
  // buys no physical fidelity at this suite's arcsecond-class noise levels.
  const Eigen::Vector3d w = u_hat + beta - u_hat.dot(beta) * u_hat;
  const double n = w.norm();
  if (n == 0.0) return u_hat;  // unreachable for |beta| < 1; total function
  return w / n;
}

double angle_between(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
  return std::atan2(a.cross(b).norm(), a.dot(b));
}

// --- star tracker ---------------------------------------------------------

StarTrackerCfg parse_star_tracker_cfg(const gnc::GncSensorCfg& cfg) {
  reject_unknown(cfg, "startracker",
                 {"sun_exclusion_rad", "central_body_exclusion_rad",
                  "slew_limit_radps"},
                 {"boresight_b", "sigma_rad"});
  StarTrackerCfg c;
  c.boresight_b =
      unit_vector(cfg, "boresight_b", "startracker", Eigen::Vector3d::UnitZ());
  c.sigma_rad = sigma_vector(cfg, "sigma_rad", "startracker");
  c.sun_exclusion_rad = nonneg(cfg, "sun_exclusion_rad", "startracker");
  c.central_body_exclusion_rad =
      nonneg(cfg, "central_body_exclusion_rad", "startracker");
  c.slew_limit_radps = nonneg(cfg, "slew_limit_radps", "startracker");
  return c;
}

StarTracker::StarTracker(std::uint32_t sample_rate_hz,
                         const StarTrackerCfg& cfg, std::uint64_t master_seed)
    : rate_hz_(sample_rate_hz),
      cfg_(cfg),
      normals_(rng::make_stream(master_seed, "sensors.startracker")) {
  if (rate_hz_ == 0) {
    throw std::invalid_argument("StarTracker: sample_rate_hz must be >= 1");
  }
}

void StarTracker::accumulate(const SensorCycleTruth& truth) {
  // A point sensor holds the latest cycle's truth rather than integrating;
  // the cycle-end fields carry the state at the next sample instant.
  latest_ = truth;
}

void StarTracker::sample(double t_s, log::SrlogWriter& writer) {
  const Eigen::Matrix3d c_i2b = rotation::dcm_from_quat(latest_.q_end_i2b);
  const Eigen::Vector3d b_i = c_i2b.transpose() * cfg_.boresight_b;
  const Eigen::Vector3d beta =
      aberration_beta(latest_.v_end_i_mps, latest_.geom.v_central_ssb_mps);

  // Aberration as a rigid field rotation (eq:optical:rho): rho = b_I x beta,
  // whose cross product with the boresight reproduces the first-order
  // displacement of eq:optical:aberration exactly at the boresight. The
  // reported attitude is relative to the APPARENT inertial frame, so the
  // aberration factor is the exact unit quaternion of the rotation vector
  // -rho (eq:optical:qab).
  const Eigen::Vector3d rho = b_i.cross(beta);
  const Eigen::Quaterniond q_ab = quat_exp(-rho);

  // Noise quaternion (eq:optical:noiseq): three unconditional draws in body
  // axes, mapped exactly.
  Eigen::Vector3d eps;
  for (int i = 0; i < 3; ++i) eps[i] = cfg_.sigma_rad[i] * normals_.next();
  const Eigen::Quaterniond dq_n = quat_exp(eps);

  // eq:optical:stmodel: q_meas = q_ab (x) q_true (x) dq_n, in the D-7
  // frame-transformation composition - the aberration is an inertial-side
  // (left) factor, the sensor noise a body-side (right) factor. The explicit
  // normalization pins the unit invariant; the sign is NOT canonicalized,
  // because consumers own the double cover (ch:sensors-optical note 3).
  const Eigen::Quaterniond q_prod = rotation::quat_multiply(
      rotation::quat_multiply(q_ab, latest_.q_end_i2b), dq_n);
  q_meas_ = rotation::quat_normalize(q_prod);

  // Validity gating (eq:optical:gating): boresight against each configured
  // excluded body's APPARENT direction, and the true body rate against the
  // slew limit. The measurement is computed, logged, and flagged every
  // sample regardless of validity, and the draws above are unconditional,
  // so the stream schedule does not depend on the gate.
  bool valid = true;
  if (latest_.geom.ephemeris_valid) {
    if (cfg_.sun_exclusion_rad > 0.0) {
      const Eigen::Vector3d u_sun =
          (latest_.geom.r_sun_m - latest_.r_end_i_m).normalized();
      valid = valid && angle_between(b_i, aberrate(u_sun, beta)) >=
                           cfg_.sun_exclusion_rad;
    }
    if (cfg_.central_body_exclusion_rad > 0.0) {
      // The central body sits at the origin of the propagation frame, so
      // the direction to its center is simply -r.
      const Eigen::Vector3d u_cb = (-latest_.r_end_i_m).normalized();
      valid = valid && angle_between(b_i, aberrate(u_cb, beta)) >=
                           cfg_.central_body_exclusion_rad;
    }
  }
  if (cfg_.slew_limit_radps > 0.0) {
    valid = valid && latest_.omega_b_end_radps.norm() <= cfg_.slew_limit_radps;
  }
  valid_ = valid;

  const double q[4] = {q_meas_.w(), q_meas_.x(), q_meas_.y(), q_meas_.z()};
  writer.write_sensor_startracker(t_s, q, valid_ ? 1u : 0u);
}

// --- sun sensor -----------------------------------------------------------

SunSensorCfg parse_sun_sensor_cfg(const gnc::GncSensorCfg& cfg) {
  reject_unknown(cfg, "sunsensor", {"fov_half_angle_rad", "sigma_rad"},
                 {"boresight_b"});
  SunSensorCfg c;
  c.boresight_b =
      unit_vector(cfg, "boresight_b", "sunsensor", Eigen::Vector3d::UnitZ());
  c.fov_half_angle_rad = nonneg(cfg, "fov_half_angle_rad", "sunsensor");
  c.sigma_rad = nonneg(cfg, "sigma_rad", "sunsensor");
  return c;
}

SunSensor::SunSensor(std::uint32_t sample_rate_hz, const SunSensorCfg& cfg,
                     std::uint64_t master_seed)
    : rate_hz_(sample_rate_hz),
      cfg_(cfg),
      normals_(rng::make_stream(master_seed, "sensors.sunsensor")) {
  if (rate_hz_ == 0) {
    throw std::invalid_argument("SunSensor: sample_rate_hz must be >= 1");
  }
}

void SunSensor::accumulate(const SensorCycleTruth& truth) { latest_ = truth; }

void SunSensor::sample(double t_s, log::SrlogWriter& writer) {
  const Eigen::Matrix3d c_i2b = rotation::dcm_from_quat(latest_.q_end_i2b);
  const Eigen::Vector3d beta =
      aberration_beta(latest_.v_end_i_mps, latest_.geom.v_central_ssb_mps);

  // Apparent Sun direction through the shared aberration path, resolved in
  // body axes. Without an ephemeris there is no Sun direction to measure, so
  // the sample is emitted invalid rather than fabricated.
  Eigen::Vector3d u_b = Eigen::Vector3d::Zero();
  bool geometry = latest_.geom.ephemeris_valid;
  if (geometry) {
    const Eigen::Vector3d u_sun_i =
        (latest_.geom.r_sun_m - latest_.r_end_i_m).normalized();
    u_b = c_i2b * aberrate(u_sun_i, beta);
  }

  // eq:optical:sunsensor: u_meas = normalize(u_b + eta), the standard
  // unit-vector error model - the radial noise component drops on
  // normalization and the two tangential components give a per-axis
  // direction error of standard deviation sigma to O(sigma^2). Draws are
  // unconditional, as everywhere in this suite.
  Eigen::Vector3d eta;
  for (int i = 0; i < 3; ++i) eta[i] = cfg_.sigma_rad * normals_.next();
  const Eigen::Vector3d sum = u_b + eta;
  const double n = sum.norm();
  u_meas_ = (n > 0.0) ? (sum / n).eval() : u_b;

  // eq:optical:sungate: inside the field of view AND illuminated. The
  // illumination fraction is the ch:srp conical shadow model's nu, shared
  // with the SRP force term: a sensor in total umbra sees no Sun, while
  // partial illumination counts as visible.
  bool valid = geometry && latest_.geom.illumination_nu > 0.0;
  if (valid && cfg_.fov_half_angle_rad > 0.0) {
    valid = angle_between(u_b, cfg_.boresight_b) <= cfg_.fov_half_angle_rad;
  }
  valid_ = valid;

  writer.write_sensor_sunsensor(t_s, u_meas_, valid_ ? 1u : 0u);
}

}  // namespace sensors
}  // namespace star
