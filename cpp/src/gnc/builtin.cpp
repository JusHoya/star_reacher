// Built-in reference GNC components. Registry names, parameters, and the
// normative control-law arithmetic are documented in gnc/builtin.hpp; the
// derivations live in the math-library chapter ch:gnc-builtin (authored in a
// parallel workstream against the same equations).
#include "star/gnc/builtin.hpp"

#include <cmath>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "star/constants.hpp"
#include "star/gnc/component.hpp"
#include "star/models/vehicle6dof.hpp"
#include "star/rotation.hpp"

namespace star {
namespace gnc {

namespace {

// --- parameter plumbing (defensive re-checks; user UX lives in Python) ----

void check_param_keys(const GncComponentCfg& cfg,
                      const std::set<std::string>& scalars,
                      const std::set<std::string>& vectors) {
  for (const auto& kv : cfg.scalars) {
    if (scalars.find(kv.first) == scalars.end()) {
      throw std::invalid_argument("gnc component '" + cfg.component +
                                  "': unknown scalar parameter '" + kv.first +
                                  "'");
    }
  }
  for (const auto& kv : cfg.vectors) {
    if (vectors.find(kv.first) == vectors.end()) {
      throw std::invalid_argument("gnc component '" + cfg.component +
                                  "': unknown vector parameter '" + kv.first +
                                  "'");
    }
  }
}

double require_scalar(const GncComponentCfg& cfg, const std::string& key) {
  const auto it = cfg.scalars.find(key);
  if (it == cfg.scalars.end() || !std::isfinite(it->second)) {
    throw std::invalid_argument("gnc component '" + cfg.component +
                                "': missing or non-finite scalar parameter '" +
                                key + "'");
  }
  return it->second;
}

std::vector<double> require_vector(const GncComponentCfg& cfg,
                                   const std::string& key, std::size_t size) {
  const auto it = cfg.vectors.find(key);
  if (it == cfg.vectors.end() || it->second.size() != size) {
    throw std::invalid_argument(
        "gnc component '" + cfg.component + "': vector parameter '" + key +
        "' must be present with exactly " + std::to_string(size) +
        " entries");
  }
  for (double v : it->second) {
    if (!std::isfinite(v)) {
      throw std::invalid_argument("gnc component '" + cfg.component +
                                  "': vector parameter '" + key +
                                  "' has a non-finite entry");
    }
  }
  return it->second;
}

// Identical expression to the Phase 4 loop's deg2rad_v so the guidance and
// open-loop paths convert angles with the same rounding.
double deg2rad_g(double d) { return d * (constants::TWO_PI / 360.0); }

// --- "dead_reckoning" navigation -------------------------------------------

class DeadReckoningNav final : public IGncComponent {
 public:
  explicit DeadReckoningNav(const GncComponentCfg& cfg) {
    check_param_keys(cfg, {}, {"q0"});
    // The initial estimate comes from configuration, stated explicitly in
    // the mission file - no implicit truth access (ch:gnc-builtin,
    // sec:gnc:deadreckoning). The reference scenarios set it to the true
    // initial attitude and document the derivation.
    const std::vector<double> q0 = require_vector(cfg, "q0", 4);
    q_hat_ = rotation::quat_normalize(
        Eigen::Quaterniond(q0[0], q0[1], q0[2], q0[3]));
  }

  void init(const GncInitContext&) override {
    // Attitude comes from the configured q0 (constructor); the rate
    // estimate starts at zero and is defined by eq:gnc:drprop from the
    // first IMU sample on.
    omega_hat_ = Eigen::Vector3d::Zero();
  }

  GncOutput update(const GncInput& input) override {
    if (input.imu_fresh && input.imu.valid) {
      // Compose the accumulated increment as one exact rotation
      // (eq:gnc:exactrot): angle = |dtheta|, axis = dtheta/|dtheta|,
      // identity when the increment is exactly zero.
      const Eigen::Vector3d dtheta = input.imu.dtheta_b_rad;
      const double angle = dtheta.norm();
      Eigen::Quaterniond dq = Eigen::Quaterniond::Identity();
      if (angle > 0.0) {
        const Eigen::Vector3d axis = dtheta / angle;
        const double s = std::sin(0.5 * angle);
        dq = Eigen::Quaterniond(std::cos(0.5 * angle), s * axis.x(),
                                s * axis.y(), s * axis.z());
      }
      // Propagation and rate estimate, eq:gnc:drprop: body-side
      // composition, normalized every cycle; interval-mean rate.
      q_hat_ = rotation::quat_normalize(rotation::quat_multiply(q_hat_, dq));
      if (input.imu.dt_s > 0.0) {
        omega_hat_ = input.imu.dtheta_b_rad / input.imu.dt_s;
      }
    }
    GncOutput out;
    out.valid = true;
    out.q_i2b = q_hat_;
    out.omega_b_radps = omega_hat_;
    return out;
  }

  int state_dim() const override { return 7; }

  void state(double* x_hat) const override {
    x_hat[0] = q_hat_.w();
    x_hat[1] = q_hat_.x();
    x_hat[2] = q_hat_.y();
    x_hat[3] = q_hat_.z();
    x_hat[4] = omega_hat_[0];
    x_hat[5] = omega_hat_[1];
    x_hat[6] = omega_hat_[2];
  }

  void covariance_upper(double* p) const override {
    // Dead reckoning carries no covariance (format doc section 3.2); the
    // packed upper triangle is identically zero by contract, not by
    // accident of uninitialized memory.
    for (int i = 0; i < 7 * (7 + 1) / 2; ++i) p[i] = 0.0;
  }

  void error_state(const TruthState& truth, double* e) const override {
    // Truth minus estimate in the state convention, with the truth
    // quaternion sign-aligned to the estimate first (q and -q encode the
    // same attitude; alignment keeps e continuous).
    Eigen::Quaterniond qt = truth.q_i2b;
    const double dot = qt.w() * q_hat_.w() + qt.x() * q_hat_.x() +
                       qt.y() * q_hat_.y() + qt.z() * q_hat_.z();
    if (dot < 0.0) {
      qt = Eigen::Quaterniond(-qt.w(), -qt.x(), -qt.y(), -qt.z());
    }
    e[0] = qt.w() - q_hat_.w();
    e[1] = qt.x() - q_hat_.x();
    e[2] = qt.y() - q_hat_.y();
    e[3] = qt.z() - q_hat_.z();
    e[4] = truth.omega_b_radps[0] - omega_hat_[0];
    e[5] = truth.omega_b_radps[1] - omega_hat_[1];
    e[6] = truth.omega_b_radps[2] - omega_hat_[2];
  }

 private:
  Eigen::Quaterniond q_hat_ = Eigen::Quaterniond::Identity();
  Eigen::Vector3d omega_hat_ = Eigen::Vector3d::Zero();
};

// --- "pitch_program" guidance ----------------------------------------------

class PitchProgramGuidance final : public IGncComponent {
 public:
  explicit PitchProgramGuidance(const GncComponentCfg& cfg)
      : az_rad_(0.0) {
    check_param_keys(cfg, {"azimuth_deg"}, {"pitch_t_s", "pitch_deg"});
    az_rad_ = deg2rad_g(require_scalar(cfg, "azimuth_deg"));
    pitch_t_s_ = require_vector(cfg, "pitch_t_s",
                                cfg.vectors.count("pitch_t_s")
                                    ? cfg.vectors.at("pitch_t_s").size()
                                    : 0);
    pitch_deg_ = require_vector(cfg, "pitch_deg", pitch_t_s_.size());
    if (pitch_t_s_.size() < 2) {
      throw std::invalid_argument(
          "gnc component 'pitch_program': pitch_t_s/pitch_deg need at least "
          "2 entries");
    }
    for (std::size_t j = 0; j + 1 < pitch_t_s_.size(); ++j) {
      if (!(pitch_t_s_[j] < pitch_t_s_[j + 1])) {
        throw std::invalid_argument(
            "gnc component 'pitch_program': pitch_t_s must be strictly "
            "increasing");
      }
    }
  }

  void init(const GncInitContext& ctx) override {
    if (!ctx.pad_basis_valid) {
      throw std::invalid_argument(
          "gnc component 'pitch_program': requires a geodetic launch "
          "mission (the commanded axis is resolved in the launch-site ENU "
          "basis, FR-14)");
    }
    up_ = ctx.up_i;
    east_ = ctx.east_i;
    north_ = ctx.north_i;
  }

  GncOutput update(const GncInput& input) override {
    // Same machinery, same call sequence, same absolute-time table lookups
    // as the Phase 4 open-loop kPitchProgram mode, so the commanded
    // attitude equals the open-loop command bit-for-bit at equal cycle
    // times (gnc/builtin.hpp contract): clamped interpolation per
    // eq:gnc:interp, commanded rate per eq:gnc:cmdrate
    // (models::omega_from_quaternions, resolved in the commanded frame).
    const double p0 = deg2rad_g(
        models::pwl_interp_clamped(pitch_t_s_, pitch_deg_, input.t_s));
    const double p1 = deg2rad_g(models::pwl_interp_clamped(
        pitch_t_s_, pitch_deg_, input.t_s + input.dt_s));
    const Eigen::Quaterniond q0 = models::attitude_from_body_x(
        models::pitch_program_axis(az_rad_, p0, up_, east_, north_), up_);
    const Eigen::Quaterniond q1 = models::attitude_from_body_x(
        models::pitch_program_axis(az_rad_, p1, up_, east_, north_), up_);
    GncOutput out;
    out.valid = true;
    out.q_i2b = q0;
    out.omega_b_radps = models::omega_from_quaternions(q0, q1, input.dt_s);
    return out;
  }

 private:
  double az_rad_;
  std::vector<double> pitch_t_s_;
  std::vector<double> pitch_deg_;
  Eigen::Vector3d up_ = Eigen::Vector3d::UnitZ();
  Eigen::Vector3d east_ = Eigen::Vector3d::UnitX();
  Eigen::Vector3d north_ = Eigen::Vector3d::UnitY();
};

// --- "attitude_hold" guidance ----------------------------------------------

class AttitudeHoldGuidance final : public IGncComponent {
 public:
  explicit AttitudeHoldGuidance(const GncComponentCfg& cfg) {
    check_param_keys(cfg, {}, {"q_cmd"});
    if (cfg.vectors.count("q_cmd") != 0) {
      const std::vector<double> q = require_vector(cfg, "q_cmd", 4);
      // Scalar-first per D-7; quat_normalize rejects a zero/non-finite norm.
      q_cmd_ = rotation::quat_normalize(
          Eigen::Quaterniond(q[0], q[1], q[2], q[3]));
      explicit_q_ = true;
    }
  }

  void init(const GncInitContext& ctx) override {
    if (!explicit_q_) {
      q_cmd_ = ctx.q0_i2b;  // hold the scenario initial attitude
    }
  }

  GncOutput update(const GncInput&) override {
    GncOutput out;
    out.valid = true;
    out.q_i2b = q_cmd_;
    out.omega_b_radps = Eigen::Vector3d::Zero();
    return out;
  }

 private:
  Eigen::Quaterniond q_cmd_ = Eigen::Quaterniond::Identity();
  bool explicit_q_ = false;
};

// --- "pd_attitude" control --------------------------------------------------

class PdAttitudeControl final : public IGncComponent {
 public:
  explicit PdAttitudeControl(const GncComponentCfg& cfg) {
    check_param_keys(cfg, {},
                     {"kp_nm_per_rad", "kd_nm_per_radps", "tau_max_nm"});
    const std::vector<double> kp = require_vector(cfg, "kp_nm_per_rad", 3);
    const std::vector<double> kd = require_vector(cfg, "kd_nm_per_radps", 3);
    const std::vector<double> tm = require_vector(cfg, "tau_max_nm", 3);
    for (int i = 0; i < 3; ++i) {
      if (kp[static_cast<std::size_t>(i)] < 0.0 ||
          kd[static_cast<std::size_t>(i)] < 0.0) {
        throw std::invalid_argument(
            "gnc component 'pd_attitude': kp_nm_per_rad and kd_nm_per_radps "
            "entries must be >= 0");
      }
      if (!(tm[static_cast<std::size_t>(i)] > 0.0)) {
        throw std::invalid_argument(
            "gnc component 'pd_attitude': tau_max_nm entries must be > 0");
      }
      kp_[i] = kp[static_cast<std::size_t>(i)];
      kd_[i] = kd[static_cast<std::size_t>(i)];
      tau_max_[i] = tm[static_cast<std::size_t>(i)];
    }
  }

  void init(const GncInitContext&) override {}

  GncOutput update(const GncInput& input) override {
    GncOutput out;
    if (!input.nav_est.valid || !input.att_cmd.valid) {
      return out;  // hold: no estimate or no command to track
    }
    // Normative arithmetic (gnc/builtin.hpp; ch:gnc-builtin sec:gnc:pd;
    // Phase 6 exit criterion 2). No renormalization of dq - inputs are
    // used as received.
    //   dq    = q_cmd^* (x) q_est                       (eq:gnc:deltaq)
    //   s     = (dq_0 >= 0) ? +1 : -1                   (eq:gnc:sign)
    //   w_err = w_est - C(dq) * w_cmd                   (eq:gnc:werr)
    //   tau_i = -kp_i * s * dq_vec_i - kd_i * w_err_i   (eq:gnc:pd)
    //   tau_i = clamp(tau_i, -tau_max_i, +tau_max_i)    (eq:gnc:sat)
    const Eigen::Quaterniond dq = rotation::quat_multiply(
        rotation::quat_conjugate(input.att_cmd.q_i2b), input.nav_est.q_i2b);
    const double s = (dq.w() >= 0.0) ? 1.0 : -1.0;
    const double dq_vec[3] = {dq.x(), dq.y(), dq.z()};
    // C(dq) is cmd-to-body for dq = q_cmd2b (eq:notation:quat2dcm), so it
    // resolves the commanded rate, expressed in the commanded frame, into
    // the estimated body frame the rate estimate lives in.
    const Eigen::Vector3d w_cmd_b =
        rotation::dcm_from_quat(dq) * input.att_cmd.omega_b_radps;
    for (int i = 0; i < 3; ++i) {
      double tau =
          -kp_[i] * s * dq_vec[i] -
          kd_[i] * (input.nav_est.omega_b_radps[i] - w_cmd_b[i]);
      if (tau > tau_max_[i]) tau = tau_max_[i];
      if (tau < -tau_max_[i]) tau = -tau_max_[i];
      out.torque_b_nm[i] = tau;
    }
    out.valid = true;
    out.q_i2b = input.att_cmd.q_i2b;            // echo of the tracked command
    out.omega_b_radps = input.att_cmd.omega_b_radps;
    return out;
  }

 private:
  double kp_[3] = {0.0, 0.0, 0.0};
  double kd_[3] = {0.0, 0.0, 0.0};
  double tau_max_[3] = {0.0, 0.0, 0.0};
};

// --- factories --------------------------------------------------------------

std::unique_ptr<IGncComponent> make_dead_reckoning(const GncComponentCfg& c) {
  return std::unique_ptr<IGncComponent>(new DeadReckoningNav(c));
}
std::unique_ptr<IGncComponent> make_pitch_program(const GncComponentCfg& c) {
  return std::unique_ptr<IGncComponent>(new PitchProgramGuidance(c));
}
std::unique_ptr<IGncComponent> make_attitude_hold(const GncComponentCfg& c) {
  return std::unique_ptr<IGncComponent>(new AttitudeHoldGuidance(c));
}
std::unique_ptr<IGncComponent> make_pd_attitude(const GncComponentCfg& c) {
  return std::unique_ptr<IGncComponent>(new PdAttitudeControl(c));
}

}  // namespace

void register_builtin_components() {
  static const bool once = [] {
    register_component("dead_reckoning", &make_dead_reckoning);
    register_component("pitch_program", &make_pitch_program);
    register_component("attitude_hold", &make_attitude_hold);
    register_component("pd_attitude", &make_pd_attitude);
    return true;
  }();
  (void)once;
}

}  // namespace gnc
}  // namespace star
