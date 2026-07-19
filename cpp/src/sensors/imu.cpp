// IMU implementation (contracts in sensors/imu.hpp, model in ch:sensors-imu).
#include "star/sensors/imu.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "star/srlog_writer.hpp"

namespace star {
namespace sensors {

namespace {

// eq:imu:bi flicker-noise flat-region coefficient sqrt(2 ln 2 / pi), and the
// eq:imu:gmpeak Gauss-Markov ADEV peak value, both to the precision the
// chapter states. Their ratio is the eq:imu:presetmap constant; spelling it
// as the quotient keeps the derivation visible at the constant itself.
constexpr double kBiasInstabilityCoeff = 0.664282;
constexpr double kGaussMarkovPeak = 0.617364;

double scalar_or(const gnc::GncSensorCfg& cfg, const char* key, double dflt) {
  const auto it = cfg.scalars.find(key);
  return it == cfg.scalars.end() ? dflt : it->second;
}

// Sigmas, quanta, and correlation times are magnitudes: a negative one would
// silently invert a noise term rather than fail, so it is refused here too
// (the FR-15 validator owns the user-facing message; this is the core-side
// guard that keeps the model well-defined).
double nonneg(const gnc::GncSensorCfg& cfg, const char* key) {
  const double v = scalar_or(cfg, key, 0.0);
  if (v < 0.0) {
    throw std::invalid_argument("sensors.imu: " + std::string(key) +
                                " must be >= 0");
  }
  return v;
}

const std::vector<double>* vector_or_null(const gnc::GncSensorCfg& cfg,
                                          const char* key, std::size_t n) {
  const auto it = cfg.vectors.find(key);
  if (it == cfg.vectors.end()) return nullptr;
  if (it->second.size() != n) {
    throw std::invalid_argument("sensors.imu: " + std::string(key) +
                                " must have exactly " + std::to_string(n) +
                                " entries, got " +
                                std::to_string(it->second.size()));
  }
  return &it->second;
}

// M = S + Gamma (eq:imu:mis): scale factors on the diagonal (parts per
// million in the config, dimensionless in the model), the six independent
// misalignment entries off-diagonal in the fixed order xy, xz, yx, yz, zx,
// zy - row-major skipping the diagonal, so the config order is readable
// straight off the matrix.
Eigen::Matrix3d build_distortion(const gnc::GncSensorCfg& cfg,
                                 const char* sf_key, const char* mis_key) {
  Eigen::Matrix3d m = Eigen::Matrix3d::Zero();
  if (const std::vector<double>* sf = vector_or_null(cfg, sf_key, 3)) {
    for (int i = 0; i < 3; ++i) {
      m(i, i) = (*sf)[static_cast<std::size_t>(i)] * 1.0e-6;
    }
  }
  if (const std::vector<double>* g = vector_or_null(cfg, mis_key, 6)) {
    m(0, 1) = (*g)[0];
    m(0, 2) = (*g)[1];
    m(1, 0) = (*g)[2];
    m(1, 2) = (*g)[3];
    m(2, 0) = (*g)[4];
    m(2, 1) = (*g)[5];
  }
  return m;
}

// Carry-preserving quantizer, eq:imu:quant, one component:
//   s = u + rho_prev;  y = q floor(s/q + 1/2);  rho = s - y
// round-half-up through floor - one branch-free IEEE-754 expression, ties
// toward +infinity by construction. q == 0 disables the stage exactly
// (identity, carry untouched), which is what lets an all-zero error
// configuration reproduce the ideal increments bit-for-bit.
double quantize(double u, double q, double& carry) {
  if (q == 0.0) return u;
  const double s = u + carry;
  const double y = q * std::floor(s / q + 0.5);
  carry = s - y;
  return y;
}

}  // namespace

double gm_sigma_from_bias_instability(double bias_instability) {
  // eq:imu:presetmap: anchor the Gauss-Markov ADEV peak at the conventional
  // flat-region read-out 0.664 B, so a configured B is what the recovery
  // procedure returns.
  return (kBiasInstabilityCoeff / kGaussMarkovPeak) * bias_instability;
}

ImuErrorCfg parse_imu_error_cfg(const gnc::GncSensorCfg& cfg) {
  static const char* const kKnownScalars[] = {
      "gyro_turnon_bias_sigma_radps", "gyro_bias_instability_radps",
      "gyro_bias_tau_s",              "gyro_arw_rad_per_sqrt_s",
      "gyro_quantum_rad",             "accel_turnon_bias_sigma_mps2",
      "accel_bias_instability_mps2",  "accel_bias_tau_s",
      "accel_vrw_mps_per_sqrt_s",     "accel_quantum_mps"};
  static const char* const kKnownVectors[] = {
      "gyro_scale_factor_ppm", "gyro_misalignment_rad",
      "accel_scale_factor_ppm", "accel_misalignment_rad"};
  // A typo must not silently disable an error term the user believes is
  // configured, so unknown keys are refused rather than ignored (DX-2).
  for (const auto& kv : cfg.scalars) {
    bool known = false;
    for (const char* k : kKnownScalars) known = known || kv.first == k;
    if (!known) {
      throw std::invalid_argument("sensors.imu: unknown parameter '" +
                                  kv.first + "'");
    }
  }
  for (const auto& kv : cfg.vectors) {
    bool known = false;
    for (const char* k : kKnownVectors) known = known || kv.first == k;
    if (!known) {
      throw std::invalid_argument("sensors.imu: unknown parameter '" +
                                  kv.first + "'");
    }
  }

  ImuErrorCfg e;
  e.gyro.turnon_bias_sigma = nonneg(cfg, "gyro_turnon_bias_sigma_radps");
  e.gyro.bias_instability = nonneg(cfg, "gyro_bias_instability_radps");
  e.gyro.bias_tau_s = nonneg(cfg, "gyro_bias_tau_s");
  e.gyro.random_walk = nonneg(cfg, "gyro_arw_rad_per_sqrt_s");
  e.gyro.quantum = nonneg(cfg, "gyro_quantum_rad");
  e.gyro.distortion =
      build_distortion(cfg, "gyro_scale_factor_ppm", "gyro_misalignment_rad");
  e.accel.turnon_bias_sigma = nonneg(cfg, "accel_turnon_bias_sigma_mps2");
  e.accel.bias_instability = nonneg(cfg, "accel_bias_instability_mps2");
  e.accel.bias_tau_s = nonneg(cfg, "accel_bias_tau_s");
  e.accel.random_walk = nonneg(cfg, "accel_vrw_mps_per_sqrt_s");
  e.accel.quantum = nonneg(cfg, "accel_quantum_mps");
  e.accel.distortion =
      build_distortion(cfg, "accel_scale_factor_ppm", "accel_misalignment_rad");
  return e;
}

Imu::Imu(std::uint32_t sample_rate_hz, const ImuErrorCfg& err,
         std::uint64_t master_seed)
    : rate_hz_(sample_rate_hz),
      normals_(rng::make_stream(master_seed, "sensors.imu")) {
  if (rate_hz_ == 0) {
    throw std::invalid_argument("Imu: sample_rate_hz must be >= 1");
  }
  // The sample interval is a configuration constant - ch:sensors-imu
  // assumption 1 pins it to the control period and the loop rejects any
  // other rate - so the eq:imu:gm discretization and the eq:imu:arw scaling
  // are formed once here instead of per sample.
  const double dt_nom = 1.0 / static_cast<double>(rate_hz_);

  const auto setup = [dt_nom](Triad& tri, const ImuTriadErrorCfg& c) {
    tri.distortion = c.distortion;
    tri.quantum = c.quantum;
    tri.noise_sigma = c.random_walk * std::sqrt(dt_nom);  // eq:imu:arw
    const double sigma_gm = gm_sigma_from_bias_instability(c.bias_instability);
    if (c.bias_tau_s > 0.0 && sigma_gm > 0.0) {
      tri.phi = std::exp(-dt_nom / c.bias_tau_s);
      // Drive variance sigma_GM^2 (1 - phi^2) is what makes the discrete
      // sequence exactly stationary at every sample (eq:imu:gm).
      tri.w_sigma = sigma_gm * std::sqrt(1.0 - tri.phi * tri.phi);
    }
    return sigma_gm;
  };
  const double sigma_gm_g = setup(gyro_, err.gyro);
  const double sigma_gm_a = setup(accel_, err.accel);

  // Initialization draw schedule (ch:sensors-imu implementation note 3),
  // normative and unconditional: a disabled term multiplies a drawn normal
  // by zero rather than skipping the draw, so the stream schedule does not
  // depend on which error terms are enabled.
  const double init_g = gyro_.phi > 0.0 ? sigma_gm_g : 0.0;
  const double init_a = accel_.phi > 0.0 ? sigma_gm_a : 0.0;
  for (int i = 0; i < 3; ++i) {
    gyro_.b0[i] = err.gyro.turnon_bias_sigma * normals_.next();  // eq:imu:turnon
  }
  for (int i = 0; i < 3; ++i) {
    // Stationary initialization of eq:imu:gm: b_0 ~ N(0, sigma_GM^2).
    gyro_.b[i] = init_g * normals_.next();
  }
  for (int i = 0; i < 3; ++i) {
    accel_.b0[i] = err.accel.turnon_bias_sigma * normals_.next();
  }
  for (int i = 0; i < 3; ++i) {
    accel_.b[i] = init_a * normals_.next();
  }
}

void Imu::accumulate(const SensorCycleTruth& truth) {
  // Trapezoidal rule over the cycle's accepted step (eq:imu:quadrature):
  // (h/2)(x_start + x_end), evaluated as (x_start + x_end) * (0.5 * h) -
  // one add and two multiplies per component in fixed order (D-10). The
  // half-step factor 0.5 * dt is exact (power-of-two scaling), which is
  // what lets the pytest suite reconstruct these increments bit-exactly
  // from the logged truth rates.
  const double half_dt = 0.5 * truth.dt_s;
  dtheta_ += (truth.omega_b_start_radps + truth.omega_b_end_radps) * half_dt;
  dv_ += (truth.sf_b_start_mps2 + truth.sf_b_end_mps2) * half_dt;
  accum_dt_ += truth.dt_s;
}

Eigen::Vector3d Imu::measure(Triad& tri, const Eigen::Vector3d& truth,
                             double dt_s) {
  // (1) Advance the Gauss-Markov state BEFORE the increment is formed
  // (eq:imu:gm, exact Ornstein-Uhlenbeck sampling): b_k = phi b_{k-1} + w.
  // The drive draws are unconditional; w_sigma is zero when the process is
  // disabled, which holds b at its (also zero) initial value.
  for (int i = 0; i < 3; ++i) {
    tri.b[i] = tri.phi * tri.b[i] + tri.w_sigma * normals_.next();
  }
  // (2) Form the bracketed value of eq:imu:gyro / eq:imu:accel in the
  // normative term order: the linear distortion (I + M) applied to the
  // truth increment, then the turn-on and in-run biases integrated over the
  // interval, then the random-walk noise of eq:imu:arw. The distortion is
  // written as truth + M*truth rather than (I + M)*truth so a zero M is an
  // exact no-op on every component.
  const Eigen::Vector3d distorted = tri.distortion * truth;
  Eigen::Vector3d u = truth + distorted;
  u += (tri.b0 + tri.b) * dt_s;
  for (int i = 0; i < 3; ++i) {
    u[i] += tri.noise_sigma * normals_.next();
  }
  // (3) Quantize with residual carry (eq:imu:quant).
  Eigen::Vector3d y;
  for (int i = 0; i < 3; ++i) {
    y[i] = quantize(u[i], tri.quantum, tri.carry[i]);
  }
  return y;
}

void Imu::sample(double t_s, log::SrlogWriter& writer) {
  // The per-sample draw schedule (ch:sensors-imu note 3) is gyro
  // Gauss-Markov drive, gyro ARW, accelerometer Gauss-Markov drive,
  // accelerometer VRW - which is exactly the gyro measure() call followed
  // by the accelerometer one, so the order is carried by the call sequence
  // rather than by a separate draw-ordering step.
  const Eigen::Vector3d dtheta_meas = measure(gyro_, dtheta_, accum_dt_);
  const Eigen::Vector3d dv_meas = measure(accel_, dv_, accum_dt_);

  writer.write_sensor_imu(t_s, dtheta_meas, dv_meas);
  last_.valid = true;
  last_.t_s = t_s;
  last_.dt_s = accum_dt_;
  last_.dtheta_b_rad = dtheta_meas;
  last_.dv_b_mps = dv_meas;
  dtheta_ = Eigen::Vector3d::Zero();
  dv_ = Eigen::Vector3d::Zero();
  accum_dt_ = 0.0;
}

std::unique_ptr<ISensor> make_sensor(const gnc::GncSensorCfg& cfg,
                                     std::uint64_t master_seed) {
  // Each kind derives its own D-9 named stream from the master seed, so
  // adding or reconfiguring one sensor never perturbs another's draws.
  if (cfg.kind == "imu") {
    return std::unique_ptr<ISensor>(
        new Imu(cfg.sample_rate_hz, parse_imu_error_cfg(cfg), master_seed));
  }
  throw std::invalid_argument(
      "make_sensor: unknown sensor kind '" + cfg.kind +
      "'; supported in this phase: {imu} (the remaining FR-23 kinds land "
      "with the sensor error-model workstream)");
}

}  // namespace sensors
}  // namespace star
