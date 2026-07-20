// The built-in reference error-state EKF. Registry name, parameters, and the
// logged-channel conventions are documented in gnc/ekf.hpp; the derivation
// and the normative arithmetic live in the math-library chapter ch:ekf,
// whose equation labels are echoed at each step below (FR-29 traceability).
#include "star/gnc/ekf.hpp"

#include <cmath>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "star/gnc/component.hpp"
#include "star/models/atmosphere_hp.hpp"
#include "star/rotation.hpp"
#include "star/sensors/optical.hpp"

namespace star {
namespace gnc {

namespace {

constexpr int kM = 15;  // error-state dimension (eq:ekf:staterr)
constexpr int kN = 16;  // total-state dimension (quaternion-led)

using Matrix15d = Eigen::Matrix<double, kM, kM>;
using Vector15d = Eigen::Matrix<double, kM, 1>;

// --- parameter plumbing (defensive re-checks; user UX lives in Python) ----

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

// A 1-sigma parameter must be strictly positive: a zero initial variance
// makes P0 singular, and NEES is then undefined rather than merely large.
Eigen::Vector3d require_sigma3(const GncComponentCfg& cfg,
                               const std::string& key) {
  const std::vector<double> v = require_vector(cfg, key, 3);
  for (double x : v) {
    if (!(x > 0.0)) {
      throw std::invalid_argument("gnc component '" + cfg.component +
                                  "': vector parameter '" + key +
                                  "' entries must be > 0");
    }
  }
  return Eigen::Vector3d(v[0], v[1], v[2]);
}

Eigen::Vector3d vec3(const std::vector<double>& v) {
  return Eigen::Vector3d(v[0], v[1], v[2]);
}

Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
  Eigen::Matrix3d s;
  s << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
  return s;
}

// Pack the row-major upper triangle of a symmetric matrix, the interchange
// order the nav.est.P channel and the consistency engine share.
void pack_upper(const Matrix15d& m, double* out) {
  int k = 0;
  for (int i = 0; i < kM; ++i) {
    for (int j = i; j < kM; ++j) out[k++] = m(i, j);
  }
}

// Symmetry pinning after every propagation and update (eq:ekf:disc): keeps
// rounding drift from accumulating into an asymmetric P, which would make
// the logged covariance depend on which triangle a reader trusts.
void symmetrize(Matrix15d& p) { p = 0.5 * (p + p.transpose()).eval(); }

// The reference filter's Gauss-Markov decay factor. A non-positive
// correlation time means the sensor chapter disabled the in-run bias state
// (ch:sensors-imu), so the filter models a non-decaying bias: phi = 1 with
// zero drive, which is the correct limit rather than a special case.
double gm_phi(double dt_s, double tau_s) {
  return tau_s > 0.0 ? std::exp(-dt_s / tau_s) : 1.0;
}

class ErrorStateEkf final : public IGncComponent {
 public:
  explicit ErrorStateEkf(const GncComponentCfg& cfg) {
    static const std::set<std::string> kVectors = {
        "q0",           "v0_mps",           "p0_m",
        "bg0_radps",    "ba0_mps2",         "p0_sigma_att_rad",
        "p0_sigma_vel_mps", "p0_sigma_pos_m", "p0_sigma_bg_radps",
        "p0_sigma_ba_mps2"};
    if (!cfg.scalars.empty()) {
      throw std::invalid_argument(
          "gnc component 'error_state_ekf': unknown scalar parameter '" +
          cfg.scalars.begin()->first + "'");
    }
    for (const auto& kv : cfg.vectors) {
      if (kVectors.find(kv.first) == kVectors.end()) {
        throw std::invalid_argument(
            "gnc component 'error_state_ekf': unknown vector parameter '" +
            kv.first + "'");
      }
    }
    const std::vector<double> q0 = require_vector(cfg, "q0", 4);
    q0_ = rotation::quat_normalize(
        Eigen::Quaterniond(q0[0], q0[1], q0[2], q0[3]));
    v0_ = vec3(require_vector(cfg, "v0_mps", 3));
    p0_ = vec3(require_vector(cfg, "p0_m", 3));
    bg0_ = vec3(require_vector(cfg, "bg0_radps", 3));
    ba0_ = vec3(require_vector(cfg, "ba0_mps2", 3));

    // P0 is diagonal in the eq:ekf:staterr ordering.
    const Eigen::Vector3d s_att = require_sigma3(cfg, "p0_sigma_att_rad");
    const Eigen::Vector3d s_vel = require_sigma3(cfg, "p0_sigma_vel_mps");
    const Eigen::Vector3d s_pos = require_sigma3(cfg, "p0_sigma_pos_m");
    const Eigen::Vector3d s_bg = require_sigma3(cfg, "p0_sigma_bg_radps");
    const Eigen::Vector3d s_ba = require_sigma3(cfg, "p0_sigma_ba_mps2");
    p0_diag_.setZero();
    for (int i = 0; i < 3; ++i) {
      p0_diag_[i] = s_att[i] * s_att[i];
      p0_diag_[3 + i] = s_vel[i] * s_vel[i];
      p0_diag_[6 + i] = s_pos[i] * s_pos[i];
      p0_diag_[9 + i] = s_bg[i] * s_bg[i];
      p0_diag_[12 + i] = s_ba[i] * s_ba[i];
    }
    innov_.reserve(3);
  }

  void init(const GncInitContext& ctx) override {
    q_hat_ = q0_;
    v_hat_ = v0_;
    p_hat_ = p0_;
    bg_hat_ = bg0_;
    ba_hat_ = ba0_;
    omega_hat_.setZero();
    p_.setZero();
    p_.diagonal() = p0_diag_;

    mu_ = ctx.mu_m3ps2;
    ellipsoid_a_m_ = ctx.ellipsoid_a_m;
    ellipsoid_inv_f_ = ctx.ellipsoid_inv_f;
    sensors_ = ctx.sensors;
    if (!(mu_ > 0.0)) {
      throw std::invalid_argument(
          "gnc component 'error_state_ekf': the run supplied a non-positive "
          "central-body gravity parameter; the filter's point-mass dynamics "
          "model (eq:ekf:mech) is undefined without it");
    }
    if (!sensors_.imu_present) {
      throw std::invalid_argument(
          "gnc component 'error_state_ekf': the run configures no IMU; the "
          "filter mechanizes from IMU increments (eq:ekf:mech) and has no "
          "propagation source without one");
    }
    innov_.clear();
  }

  GncOutput update(const GncInput& input) override {
    innov_.clear();
    if (input.imu_fresh && input.imu.valid && input.imu.dt_s > 0.0) {
      propagate(input.imu);
    }
    // Updates run in the fixed order nav fix, star tracker, altimeter, each
    // against the running post-update covariance (ch:ekf implementation note
    // 2). The order is normative: processing the same measurements in a
    // different order gives a different - though equally valid - trajectory,
    // and the ensemble gate is only reproducible if the order is pinned.
    if (input.navfix.fresh && input.navfix.valid && sensors_.navfix_present) {
      update_navfix(input.navfix);
    }
    if (input.startracker.fresh && input.startracker.valid &&
        sensors_.startracker_present) {
      update_startracker(input.startracker, input.env);
    }
    if (input.altimeter.fresh && input.altimeter.valid &&
        sensors_.altimeter_present) {
      update_altimeter(input.altimeter, input.env);
    }

    GncOutput out;
    out.valid = true;
    out.q_i2b = q_hat_;
    out.omega_b_radps = omega_hat_;
    return out;
  }

  int state_dim() const override { return kN; }
  int cov_dim() const override { return kM; }
  // The largest innovation this filter can ever produce is the nav fix's
  // six. Declared as a constant because the log header is built before
  // init() supplies the configured sensor suite; a run with fewer sensors
  // simply never emits a six-wide record, and each record carries its own
  // valid dimension m (format doc section 3.2).
  int innov_max_dim() const override { return 6; }

  const std::vector<InnovationSample>& innovations() const override {
    return innov_;
  }

  void state(double* x_hat) const override {
    x_hat[0] = q_hat_.w();
    x_hat[1] = q_hat_.x();
    x_hat[2] = q_hat_.y();
    x_hat[3] = q_hat_.z();
    for (int i = 0; i < 3; ++i) {
      x_hat[4 + i] = v_hat_[i];
      x_hat[7 + i] = p_hat_[i];
      x_hat[10 + i] = bg_hat_[i];
      x_hat[13 + i] = ba_hat_[i];
    }
  }

  void covariance_upper(double* p) const override { pack_upper(p_, p); }

  const std::vector<ErrorBlock>& error_layout() const override {
    // The state vector this filter publishes through state(), read block by
    // block, so the loop can form nav.err without this component ever
    // holding the truth state (the FR-24 boundary; see the descriptor
    // commentary in gnc/component.hpp). The blocks tile x_hat in the
    // eq:ekf:staterr ordering, quaternion-led.
    //
    // The attitude block declares kQuatErrorLocal, which is the
    // MULTIPLICATIVE error of eq:ekf:qerr, dq = q_hat^-1 (x) q_true,
    // sign-canonicalized to the +w hemisphere so the double cover cannot
    // flip the logged error between neighbouring epochs. The consistency
    // evaluator reduces it to the three-component dtheta the covariance
    // describes by dtheta = 2 sgn(dq_w) dq_v.
    //
    // The bias blocks are why init() rejects a run without an IMU: without
    // one there is no true bias to difference and the layout is refused.
    static const std::vector<ErrorBlock> kLayout = {
        {ErrorQuantity::kAttitude, ErrorForm::kQuatErrorLocal, 0},
        {ErrorQuantity::kVelocity, ErrorForm::kDifference, 4},
        {ErrorQuantity::kPosition, ErrorForm::kDifference, 7},
        {ErrorQuantity::kGyroBias, ErrorForm::kDifference, 10},
        {ErrorQuantity::kAccelBias, ErrorForm::kDifference, 13},
    };
    return kLayout;
  }

 private:
  // --- propagation -------------------------------------------------------

  void propagate(const ImuSample& imu) {
    const double dt = imu.dt_s;
    // Pre-cycle nominal: F is evaluated here (ch:ekf implementation note 2),
    // so these are captured before the mechanization overwrites the state.
    const Eigen::Matrix3d c_hat =
        rotation::dcm_from_quat(q_hat_).transpose();  // body -> inertial
    const Eigen::Vector3d p_pre = p_hat_;
    const Eigen::Vector3d v_pre = v_hat_;

    // eq:ekf:mech, bias-corrected increments and the cycle averages that
    // drive F.
    const Eigen::Vector3d dtheta = imu.dtheta_b_rad - bg_hat_ * dt;
    const Eigen::Vector3d dv = imu.dv_b_mps - ba_hat_ * dt;
    const Eigen::Vector3d omega_avg = dtheta / dt;
    const Eigen::Vector3d f_avg = dv / dt;

    // Attitude: the increment composed as one exact rotation, the same map
    // the dead reckoner uses (eq:gnc:exactrot) - one code path.
    const Eigen::Quaterniond dq = sensors::quat_exp(dtheta);
    q_hat_ = rotation::quat_normalize(rotation::quat_multiply(q_hat_, dq));
    omega_hat_ = omega_avg;

    // Velocity and position both trapezoidal, the predictor-corrector of
    // eq:ekf:mech. Gravity is the one acceleration the IMU cannot sense, so
    // it is the filter's own quadrature - not a sensor - that sets its
    // truncation error. An explicit Euler step here leaves a deterministic
    // velocity drift no term of Q describes (sec:ekf:gravityorder), so the
    // step is closed to second order to match the position step's order.
    const Eigen::Vector3d dv_rot = dv + 0.5 * dtheta.cross(dv);
    const Eigen::Vector3d sensed = c_hat * dv_rot;
    const Eigen::Vector3d g_pre = gravity(p_pre);
    const Eigen::Vector3d v_pred = v_pre + sensed + g_pre * dt;
    const Eigen::Vector3d p_pred = p_pre + 0.5 * (v_pre + v_pred) * dt;
    const Eigen::Vector3d v_new =
        v_pre + sensed + 0.5 * (g_pre + gravity(p_pred)) * dt;
    p_hat_ = p_pre + 0.5 * (v_pre + v_new) * dt;
    v_hat_ = v_new;

    // Bias estimates decay with their Gauss-Markov model.
    const double phi_g = gm_phi(dt, sensors_.gyro_tau_s);
    const double phi_a = gm_phi(dt, sensors_.accel_tau_s);
    bg_hat_ = phi_g * bg_hat_;
    ba_hat_ = phi_a * ba_hat_;

    // --- linearized error dynamics, eq:ekf:errdyn / eq:ekf:F -------------
    Matrix15d f_mat = Matrix15d::Zero();
    const Eigen::Matrix3d i3 = Eigen::Matrix3d::Identity();
    f_mat.block<3, 3>(0, 0) = -skew(omega_avg);
    f_mat.block<3, 3>(0, 9) = -i3;
    const Eigen::Matrix3d c_skew_f = c_hat * skew(f_avg);
    f_mat.block<3, 3>(3, 0) = -c_skew_f;
    f_mat.block<3, 3>(3, 6) = gravity_gradient(p_pre);
    f_mat.block<3, 3>(3, 12) = -c_hat;
    f_mat.block<3, 3>(6, 3) = i3;
    if (sensors_.gyro_tau_s > 0.0) {
      f_mat.block<3, 3>(9, 9) = -i3 / sensors_.gyro_tau_s;
    }
    if (sensors_.accel_tau_s > 0.0) {
      f_mat.block<3, 3>(12, 12) = -i3 / sensors_.accel_tau_s;
    }

    // eq:ekf:disc, first order by decision (ch:ekf assumption 5).
    const Matrix15d phi = Matrix15d::Identity() + f_mat * dt;

    // Q_k = Gamma Q_c Gamma^T dt (eq:ekf:G). Gamma is block-sparse and its
    // only non-identity block is -C_hat on the velocity row, so the product
    // is assembled directly rather than through a 15x12 multiply: the
    // attitude block picks up N_g^2, the velocity block C N_a^2 C^T, and the
    // bias blocks their Gauss-Markov drive densities. The signs in Gamma
    // square out, as they must for a covariance.
    Matrix15d q_mat = Matrix15d::Zero();
    const double ng2 = sensors_.gyro_arw * sensors_.gyro_arw;
    const double na2 = sensors_.accel_vrw * sensors_.accel_vrw;
    q_mat.block<3, 3>(0, 0) = ng2 * i3;
    const Eigen::Matrix3d c_ct = c_hat * c_hat.transpose();
    q_mat.block<3, 3>(3, 3) = na2 * c_ct;
    // q = 2 sigma_GM^2 / tau_c, the continuous drive density consistent with
    // the discrete recursion eq:imu:gm (Maybeck).
    if (sensors_.gyro_tau_s > 0.0) {
      const double qg = 2.0 * sensors_.gyro_gm_sigma * sensors_.gyro_gm_sigma /
                        sensors_.gyro_tau_s;
      q_mat.block<3, 3>(9, 9) = qg * i3;
    }
    if (sensors_.accel_tau_s > 0.0) {
      const double qa = 2.0 * sensors_.accel_gm_sigma *
                        sensors_.accel_gm_sigma / sensors_.accel_tau_s;
      q_mat.block<3, 3>(12, 12) = qa * i3;
    }
    q_mat *= dt;

    const Matrix15d p_prop = phi * p_ * phi.transpose() + q_mat;
    p_ = p_prop;
    symmetrize(p_);
  }

  Eigen::Vector3d gravity(const Eigen::Vector3d& p) const {
    // The navigator's internal gravity model is the central-body point mass
    // (ch:ekf assumption 2).
    const double r = p.norm();
    return -mu_ * p / (r * r * r);
  }

  Eigen::Matrix3d gravity_gradient(const Eigen::Vector3d& p) const {
    // eq:ekf:gravgrad.
    const double r = p.norm();
    const Eigen::Vector3d u = p / r;
    const Eigen::Matrix3d uut = u * u.transpose();
    return (mu_ / (r * r * r)) *
           (3.0 * uut - Eigen::Matrix3d::Identity());
  }

  // --- the generic Joseph-form update, eq:ekf:joseph ---------------------
  //
  // Templated on the measurement dimension so every update runs on
  // fixed-size Eigen types with no heap allocation in the cycle path (D-10),
  // and so the three sensors cannot drift into three different update
  // algebras. Returns the innovation covariance for the nav.innov record.
  template <int M>
  Eigen::Matrix<double, M, M> joseph_update(
      const Eigen::Matrix<double, M, kM>& h,
      const Eigen::Matrix<double, M, M>& r,
      const Eigen::Matrix<double, M, 1>& y, Vector15d& dx) {
    const Eigen::Matrix<double, kM, M> pht = p_ * h.transpose();
    const Eigen::Matrix<double, M, M> s = h * pht + r;
    // LDLT on the (at most 6x6) innovation covariance: deterministic with no
    // pivoting hazards at these sizes (ch:ekf implementation note 3). K is
    // formed as K^T = S^-1 (P H^T)^T, exploiting S's symmetry.
    const Eigen::LDLT<Eigen::Matrix<double, M, M>> ldlt(s);
    // GCC 13 raises a false -Warray-bounds on the M = 1 instantiation of the
    // line below, reporting writes at offsets 120-232 into the 120-byte kt.
    // Those offsets are dst.row(1) of a 1x15 destination. They come from
    // Eigen's row permutation `dst = m_transpositions * rhs`
    // (Eigen/src/Cholesky/LDLT.h), which swaps via dst.row(k).swap(dst.row(j))
    // with j = tr.coeff(k) (Eigen/src/Core/ProductEvaluators.h). For M = 1
    // that swap is provably the identity: ldlt_inplace::unblocked returns
    // early on size <= 1 having called transpositions.setIdentity(), so
    // j == k == 0 and only row 0 is ever touched. Under -DNDEBUG the
    // eigen_assert carrying that index bound is compiled out, leaving GCC to
    // model the unreachable j >= 1 branch; removing -DNDEBUG alone silences
    // the warning. Recorded in docs/ci/phase6_crossplatform.md.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#endif
    const Eigen::Matrix<double, M, kM> kt = ldlt.solve(pht.transpose());
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
    const Eigen::Matrix<double, kM, M> k = kt.transpose();
    dx = k * y;
    const Matrix15d ikh = Matrix15d::Identity() - k * h;
    const Matrix15d p_post =
        ikh * p_ * ikh.transpose() + k * r * k.transpose();
    p_ = p_post;
    symmetrize(p_);
    return s;
  }

  // Fold the correction into the nominal state and zero the error mean
  // (eq:ekf:reset). The exact reset also transports the covariance by a
  // Jacobian that differs from identity only in the attitude block, by
  // O(dtheta); the reference filter OMITS that correction by specification
  // (ch:ekf, closed-loop reset paragraph) - it is second order in the
  // post-update error.
  void reset(const Vector15d& dx) {
    const Eigen::Vector3d dtheta = dx.segment<3>(0);
    const Eigen::Quaterniond dq = sensors::quat_exp(dtheta);
    q_hat_ = rotation::quat_normalize(rotation::quat_multiply(q_hat_, dq));
    v_hat_ += dx.segment<3>(3);
    p_hat_ += dx.segment<3>(6);
    bg_hat_ += dx.segment<3>(9);
    ba_hat_ += dx.segment<3>(12);
  }

  // Record one applied update for the nav.innov channel. The packed upper
  // triangle matches the channel's documented interchange order.
  template <int M>
  void record(std::uint32_t sensor_id, const Eigen::Matrix<double, M, 1>& y,
              const Eigen::Matrix<double, M, M>& s) {
    InnovationSample rec;
    rec.sensor_id = sensor_id;
    rec.y.resize(M);
    for (int i = 0; i < M; ++i) rec.y[static_cast<std::size_t>(i)] = y[i];
    rec.s_upper.resize(static_cast<std::size_t>(M) * (M + 1) / 2);
    std::size_t k = 0;
    for (int i = 0; i < M; ++i) {
      for (int j = i; j < M; ++j) rec.s_upper[k++] = s(i, j);
    }
    innov_.push_back(rec);
  }

  // --- the three aiding updates ------------------------------------------

  void update_navfix(const NavFixSample& z) {
    // eq:ekf:navfixH, position rows first.
    Eigen::Matrix<double, 6, kM> h = Eigen::Matrix<double, 6, kM>::Zero();
    h.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity();
    h.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();
    Eigen::Matrix<double, 6, 6> r = Eigen::Matrix<double, 6, 6>::Zero();
    for (int i = 0; i < 3; ++i) {
      const double sr = sensors_.navfix_sigma_r_m[i];
      const double sv = sensors_.navfix_sigma_v_mps[i];
      r(i, i) = sr * sr;
      r(3 + i, 3 + i) = sv * sv;
    }
    Eigen::Matrix<double, 6, 1> y;
    y.segment<3>(0) = z.r_i_m - p_hat_;
    y.segment<3>(3) = z.v_i_mps - v_hat_;

    Vector15d dx = Vector15d::Zero();
    const Eigen::Matrix<double, 6, 6> s = joseph_update<6>(h, r, y, dx);
    reset(dx);
    record<6>(z.sensor_id, y, s);
  }

  void update_startracker(const StarTrackerSample& z,
                          const NavEnvironment& env) {
    // The filter predicts the APPARENT attitude with its own state: the
    // aberration factor of eq:optical:qab evaluated at the estimated
    // rotation vector -rho_hat, with beta_hat assembled from the ESTIMATED
    // velocity and the ephemeris chain (eq:ekf:stinnov discussion). The
    // velocity-estimate contribution to the aberration error is
    // |dv|/c ~ 3.3e-9 rad per m/s, negligible against sigma.
    const Eigen::Matrix3d c_i2b = rotation::dcm_from_quat(q_hat_);
    const Eigen::Vector3d b_i =
        c_i2b.transpose() * sensors_.startracker_boresight_b;
    const Eigen::Vector3d beta =
        sensors::aberration_beta(v_hat_, env.v_central_ssb_mps);
    const Eigen::Vector3d rho = b_i.cross(beta);
    const Eigen::Quaterniond q_ab = sensors::quat_exp(-rho);
    const Eigen::Quaterniond q_pred =
        rotation::quat_multiply(q_ab, q_hat_);

    // eq:ekf:stinnov: exact error extraction, sgn(0) = +1.
    const Eigen::Quaterniond dq_y = rotation::quat_multiply(
        rotation::quat_conjugate(q_pred), z.q_i2b);
    const double sgn = dq_y.w() >= 0.0 ? 1.0 : -1.0;
    Eigen::Vector3d y;
    y << 2.0 * sgn * dq_y.x(), 2.0 * sgn * dq_y.y(), 2.0 * sgn * dq_y.z();

    // eq:ekf:stH.
    Eigen::Matrix<double, 3, kM> h = Eigen::Matrix<double, 3, kM>::Zero();
    h.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d r = Eigen::Matrix3d::Zero();
    for (int i = 0; i < 3; ++i) {
      const double sd = sensors_.startracker_sigma_rad[i];
      r(i, i) = sd * sd;
    }

    Vector15d dx = Vector15d::Zero();
    const Eigen::Matrix3d s = joseph_update<3>(h, r, y, dx);
    reset(dx);
    record<3>(z.sensor_id, y, s);
  }

  void update_altimeter(const AltimeterSample& z, const NavEnvironment& env) {
    if (!env.bodyfixed_valid || !(ellipsoid_a_m_ > 0.0)) {
      // Without a body-fixed frame there is no ellipsoid to measure against;
      // skipping is the honest response, and the run's altimeter records
      // simply go unused rather than being folded in against a wrong frame.
      return;
    }
    // h(x) is the same Bowring conversion the sensor used (eq:radio:alt),
    // through the one shared implementation.
    const Eigen::Vector3d r_bf = env.c_gcrf_to_bodyfixed * p_hat_;
    double lat = 0.0;
    double lon = 0.0;
    double alt = 0.0;
    models::geodetic_lat_lon_alt(r_bf, ellipsoid_a_m_, ellipsoid_inv_f_, lat,
                                 lon, alt);
    // eq:ekf:altH: the gradient of geodetic height with respect to Cartesian
    // position is exactly the unit ellipsoidal normal at the sub-vehicle
    // point, rotated back into the inertial frame.
    Eigen::Vector3d n_bf;
    n_bf << std::cos(lat) * std::cos(lon), std::cos(lat) * std::sin(lon),
        std::sin(lat);
    const Eigen::Vector3d n_i = env.c_gcrf_to_bodyfixed.transpose() * n_bf;

    Eigen::Matrix<double, 1, kM> h = Eigen::Matrix<double, 1, kM>::Zero();
    h.block<1, 3>(0, 6) = n_i.transpose();
    // The turn-on bias is not estimated; when enabled its variance inflates
    // R and the residual mismatch is accepted (ch:ekf, altimeter paragraph).
    const double sn = sensors_.altimeter_sigma_noise_m;
    const double sb = sensors_.altimeter_sigma_bias_m;
    Eigen::Matrix<double, 1, 1> r;
    r(0, 0) = sn * sn + sb * sb;
    Eigen::Matrix<double, 1, 1> y;
    y(0) = z.h_m - alt;

    Vector15d dx = Vector15d::Zero();
    const Eigen::Matrix<double, 1, 1> s = joseph_update<1>(h, r, y, dx);
    reset(dx);
    record<1>(z.sensor_id, y, s);
  }

  // configured initial belief
  Eigen::Quaterniond q0_ = Eigen::Quaterniond::Identity();
  Eigen::Vector3d v0_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d p0_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d bg0_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d ba0_ = Eigen::Vector3d::Zero();
  Vector15d p0_diag_ = Vector15d::Zero();

  // nominal state and covariance
  Eigen::Quaterniond q_hat_ = Eigen::Quaterniond::Identity();
  Eigen::Vector3d v_hat_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d p_hat_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d bg_hat_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d ba_hat_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d omega_hat_ = Eigen::Vector3d::Zero();
  Matrix15d p_ = Matrix15d::Zero();

  // run context captured at init
  double mu_ = 0.0;
  double ellipsoid_a_m_ = 0.0;
  double ellipsoid_inv_f_ = 0.0;
  NavSensorModel sensors_;

  // Rebuilt every update(); capacity is reserved once at construction, so
  // the only per-cycle allocation is the innovation payload of a cycle that
  // actually applied an update - aiding sensors run far slower than the
  // control cycle, so most cycles allocate nothing.
  std::vector<InnovationSample> innov_;
};

std::unique_ptr<IGncComponent> make_error_state_ekf(
    const GncComponentCfg& c) {
  return std::unique_ptr<IGncComponent>(new ErrorStateEkf(c));
}

}  // namespace

void register_ekf_component() {
  static const bool once = [] {
    register_component("error_state_ekf", &make_error_state_ekf);
    return true;
  }();
  (void)once;
}

}  // namespace gnc
}  // namespace star
