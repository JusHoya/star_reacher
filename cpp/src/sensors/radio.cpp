// Radio-class sensor implementation (contracts in sensors/radio.hpp, model
// in ch:sensors-radio).
#include "star/sensors/radio.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "star/models/atmosphere_hp.hpp"
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

}  // namespace

// --- external nav fix -----------------------------------------------------

NavFixCfg parse_nav_fix_cfg(const gnc::GncSensorCfg& cfg) {
  reject_unknown(cfg, "navfix",
                 {"gm_position_sigma_m", "gm_position_tau_s",
                  "gm_velocity_sigma_mps", "gm_velocity_tau_s"},
                 {"sigma_r_m", "sigma_v_mps"});
  NavFixCfg c;
  c.sigma_r_m = sigma_vector(cfg, "sigma_r_m", "navfix");
  c.sigma_v_mps = sigma_vector(cfg, "sigma_v_mps", "navfix");
  c.gm_r.sigma = nonneg(cfg, "gm_position_sigma_m", "navfix");
  c.gm_r.tau_s = nonneg(cfg, "gm_position_tau_s", "navfix");
  c.gm_v.sigma = nonneg(cfg, "gm_velocity_sigma_mps", "navfix");
  c.gm_v.tau_s = nonneg(cfg, "gm_velocity_tau_s", "navfix");
  return c;
}

NavFix::NavFix(std::uint32_t sample_rate_hz, const NavFixCfg& cfg,
               std::uint64_t master_seed)
    : rate_hz_(sample_rate_hz),
      cfg_(cfg),
      normals_(rng::make_stream(master_seed, "sensors.navfix")) {
  if (rate_hz_ == 0) {
    throw std::invalid_argument("NavFix: sample_rate_hz must be >= 1");
  }
  const double dt_nom = 1.0 / static_cast<double>(rate_hz_);
  // eq:radio:gm, the ch:sensors-imu recursion: phi = exp(-dt/tau) with drive
  // variance sigma^2 (1 - phi^2), which makes the discrete sequence exactly
  // stationary.
  if (cfg_.gm_r.sigma > 0.0 && cfg_.gm_r.tau_s > 0.0) {
    phi_r_ = std::exp(-dt_nom / cfg_.gm_r.tau_s);
    w_sigma_r_ = cfg_.gm_r.sigma * std::sqrt(1.0 - phi_r_ * phi_r_);
    // Stationary initialization, consumed at run start (position then
    // velocity) per the ch:sensors-radio draw schedule.
    for (int i = 0; i < 3; ++i) c_r_[i] = cfg_.gm_r.sigma * normals_.next();
  }
  if (cfg_.gm_v.sigma > 0.0 && cfg_.gm_v.tau_s > 0.0) {
    phi_v_ = std::exp(-dt_nom / cfg_.gm_v.tau_s);
    w_sigma_v_ = cfg_.gm_v.sigma * std::sqrt(1.0 - phi_v_ * phi_v_);
    for (int i = 0; i < 3; ++i) c_v_[i] = cfg_.gm_v.sigma * normals_.next();
  }
}

void NavFix::accumulate(const SensorCycleTruth& truth) { latest_ = truth; }

void NavFix::advance_gm(Eigen::Vector3d& c, double phi, double w_sigma) {
  for (int i = 0; i < 3; ++i) c[i] = phi * c[i] + w_sigma * normals_.next();
}

void NavFix::sample(double t_s, log::SrlogWriter& writer) {
  // Draw schedule (ch:sensors-radio note 2), in order: position
  // Gauss-Markov drive if enabled, velocity Gauss-Markov drive if enabled,
  // position white, velocity white. Unlike the IMU's fully unconditional
  // rule, the optional correlated components consume draws only when
  // enabled - enabling one is a configuration change that lands in the
  // FR-15 resolved config hash, which is the reproducibility anchor.
  if (w_sigma_r_ > 0.0) advance_gm(c_r_, phi_r_, w_sigma_r_);
  if (w_sigma_v_ > 0.0) advance_gm(c_v_, phi_v_, w_sigma_v_);

  // eq:radio:fix: r_meas = r + c_r + eta_r, v_meas = v + c_v + eta_v, with
  // independent white errors per GCRF axis (eq:radio:white).
  for (int i = 0; i < 3; ++i) {
    r_meas_[i] = latest_.r_end_i_m[i] + c_r_[i] +
                 cfg_.sigma_r_m[i] * normals_.next();
  }
  for (int i = 0; i < 3; ++i) {
    v_meas_[i] = latest_.v_end_i_mps[i] + c_v_[i] +
                 cfg_.sigma_v_mps[i] * normals_.next();
  }
  writer.write_sensor_navfix(t_s, r_meas_, v_meas_);
}

// --- altimeter ------------------------------------------------------------

AltimeterCfg parse_altimeter_cfg(const gnc::GncSensorCfg& cfg) {
  reject_unknown(cfg, "altimeter",
                 {"sigma_bias_m", "sigma_noise_m", "h_min_m", "h_max_m"}, {});
  AltimeterCfg c;
  c.sigma_bias_m = nonneg(cfg, "sigma_bias_m", "altimeter");
  c.sigma_noise_m = nonneg(cfg, "sigma_noise_m", "altimeter");
  // The band bounds are signed: a geodetic altitude may legitimately be
  // negative over the ellipsoid, so they are not passed through nonneg().
  c.h_min_m = scalar_or(cfg, "h_min_m", 0.0);
  c.h_max_m = scalar_or(cfg, "h_max_m", 0.0);
  return c;
}

Altimeter::Altimeter(std::uint32_t sample_rate_hz, const AltimeterCfg& cfg,
                     std::uint64_t master_seed)
    : rate_hz_(sample_rate_hz),
      cfg_(cfg),
      normals_(rng::make_stream(master_seed, "sensors.altimeter")) {
  if (rate_hz_ == 0) {
    throw std::invalid_argument("Altimeter: sample_rate_hz must be >= 1");
  }
  // eq:radio:alt turn-on bias: one unconditional draw at run start.
  bias_m_ = cfg_.sigma_bias_m * normals_.next();
}

void Altimeter::accumulate(const SensorCycleTruth& truth) { latest_ = truth; }

void Altimeter::sample(double t_s, log::SrlogWriter& writer) {
  // Geodetic altitude of the truth position over the central body's
  // reference ellipsoid, through the same Bowring two-pass conversion
  // (eq:hp:geodetic) and the same body-fixed rotation the atmosphere and
  // environment models use - one conversion path, per ch:sensors-radio
  // note 3. A spherical body (inv_f == 0) degenerates exactly to
  // norm(r) - a, which is what the conversion returns for zero flattening.
  // A spherical body is handled by the closed spherical form rather than by
  // passing zero flattening into the Bowring conversion: ch:sensors-radio
  // assumption 4 notes that the conversion degenerates exactly to
  // norm(r) - a at f = 0, which is true of the mathematics but NOT of the
  // implementation - geodetic_altitude() requires inv_f > 1 and rejects a
  // sphere outright. The two branches agree in the limit, and the spherical
  // one is also rotation-independent, so it needs no body-fixed frame.
  const bool spherical = !(latest_.geom.ellipsoid_inv_f > 1.0);
  double h_true = 0.0;
  if (spherical || !latest_.geom.bodyfixed_valid) {
    h_true = latest_.r_end_i_m.norm() - latest_.geom.ellipsoid_a_m;
  } else {
    const Eigen::Vector3d r_bf =
        latest_.geom.c_gcrf_to_bodyfixed * latest_.r_end_i_m;
    h_true = models::geodetic_altitude(r_bf, latest_.geom.ellipsoid_a_m,
                                       latest_.geom.ellipsoid_inv_f);
  }

  // eq:radio:alt: h_meas = h(r) + b_h + eta_h, the white draw unconditional.
  alt_meas_ = h_true + bias_m_ + cfg_.sigma_noise_m * normals_.next();

  // eq:radio:altgate: the configured operating band of the instrument. The
  // measurement is computed and logged regardless of the flag.
  valid_ = (cfg_.h_max_m <= cfg_.h_min_m)
               ? true
               : (h_true >= cfg_.h_min_m && h_true <= cfg_.h_max_m);

  writer.write_sensor_altimeter(t_s, alt_meas_);
}

}  // namespace sensors
}  // namespace star
