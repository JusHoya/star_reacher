// Camera hook implementation (contracts in sensors/camera.hpp, model in
// ch:camera).
#include "star/sensors/camera.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include "star/rotation.hpp"
#include "star/sensors/optical.hpp"
#include "star/srlog_writer.hpp"

namespace star {
namespace sensors {

namespace {

double scalar_or(const gnc::GncSensorCfg& cfg, const char* key, double dflt) {
  const auto it = cfg.scalars.find(key);
  return it == cfg.scalars.end() ? dflt : it->second;
}

double positive(const gnc::GncSensorCfg& cfg, const char* key) {
  const double v = scalar_or(cfg, key, 0.0);
  if (!(v > 0.0)) {
    throw std::invalid_argument("sensors.camera: " + std::string(key) +
                                " must be > 0");
  }
  return v;
}

}  // namespace

CameraCfg parse_camera_cfg(const gnc::GncSensorCfg& cfg) {
  static const char* const kScalars[] = {"fx_px", "fy_px",    "cx_px",
                                         "cy_px", "width_px", "height_px"};
  static const char* const kVectors[] = {"r_cam_b_m", "q_b2c",
                                         "landmarks_fixed_m"};
  for (const auto& kv : cfg.scalars) {
    bool known = false;
    for (const char* k : kScalars) known = known || kv.first == k;
    if (!known) {
      throw std::invalid_argument("sensors.camera: unknown parameter '" +
                                  kv.first + "'");
    }
  }
  for (const auto& kv : cfg.vectors) {
    bool known = false;
    for (const char* k : kVectors) known = known || kv.first == k;
    if (!known) {
      throw std::invalid_argument("sensors.camera: unknown parameter '" +
                                  kv.first + "'");
    }
  }

  CameraCfg c;
  c.fx = positive(cfg, "fx_px");
  c.fy = positive(cfg, "fy_px");
  // Principal-point coordinates are free within the sensor, including
  // negative values for a deliberately decentred model, so they are read
  // without a sign constraint.
  c.cx = scalar_or(cfg, "cx_px", 0.0);
  c.cy = scalar_or(cfg, "cy_px", 0.0);
  c.width_px = static_cast<std::uint32_t>(positive(cfg, "width_px"));
  c.height_px = static_cast<std::uint32_t>(positive(cfg, "height_px"));

  const auto rit = cfg.vectors.find("r_cam_b_m");
  if (rit != cfg.vectors.end()) {
    if (rit->second.size() != 3) {
      throw std::invalid_argument(
          "sensors.camera: r_cam_b_m must have exactly 3 entries");
    }
    c.r_cam_b_m =
        Eigen::Vector3d(rit->second[0], rit->second[1], rit->second[2]);
  }
  const auto qit = cfg.vectors.find("q_b2c");
  if (qit != cfg.vectors.end()) {
    if (qit->second.size() != 4) {
      throw std::invalid_argument(
          "sensors.camera: q_b2c must have exactly 4 entries "
          "(Hamilton scalar-first, D-7)");
    }
    const Eigen::Quaterniond q(qit->second[0], qit->second[1], qit->second[2],
                               qit->second[3]);
    if (!(q.norm() > 0.0)) {
      throw std::invalid_argument("sensors.camera: q_b2c must be nonzero");
    }
    c.q_b2c = rotation::quat_normalize(q);
  }
  const auto lit = cfg.vectors.find("landmarks_fixed_m");
  if (lit != cfg.vectors.end()) {
    if (lit->second.size() % 3 != 0) {
      throw std::invalid_argument(
          "sensors.camera: landmarks_fixed_m must be a flat list of "
          "3-component central-body-fixed positions");
    }
    for (std::size_t i = 0; i < lit->second.size(); i += 3) {
      c.landmarks_fixed_m.push_back(Eigen::Vector3d(
          lit->second[i], lit->second[i + 1], lit->second[i + 2]));
    }
  }
  return c;
}

CameraHook::CameraHook(std::uint32_t sample_rate_hz, const CameraCfg& cfg)
    : rate_hz_(sample_rate_hz), cfg_(cfg) {
  if (rate_hz_ == 0) {
    throw std::invalid_argument("CameraHook: sample_rate_hz must be >= 1");
  }
  px_.assign(2 * cfg_.landmarks_fixed_m.size(), 0.0);
  visible_.assign(cfg_.landmarks_fixed_m.size(), 0);
}

void CameraHook::accumulate(const SensorCycleTruth& truth) { latest_ = truth; }

void CameraHook::sample(double t_s, log::SrlogWriter& writer) {
  // eq:camera:pose. The emitted pose channels are the truth doubles
  // themselves - r_end_i_m and q_end_i2b are copied, never recomputed - so
  // exit criterion 7's bit-exactness clause holds by construction rather
  // than by a tolerance.
  const Eigen::Vector3d& r_i = latest_.r_end_i_m;
  const Eigen::Quaterniond& q_i2b = latest_.q_end_i2b;
  const double q_out[4] = {q_i2b.w(), q_i2b.x(), q_i2b.y(), q_i2b.z()};

  const Eigen::Matrix3d c_i2b = rotation::dcm_from_quat(q_i2b);
  const Eigen::Matrix3d c_b2c = rotation::dcm_from_quat(cfg_.q_b2c);
  // The camera station in inertial axes; the mount offset is CG-relative
  // (ch:camera assumption 2).
  const Eigen::Vector3d r_cam_i =
      r_i + c_i2b.transpose() * cfg_.r_cam_b_m;
  const Eigen::Vector3d beta =
      aberration_beta(latest_.v_end_i_mps, latest_.geom.v_central_ssb_mps);

  // Half-pixel sensor bounds of the pixel-center convention
  // (ch:camera section on intrinsics).
  const double u_lo = -0.5;
  const double v_lo = -0.5;
  const double u_hi = static_cast<double>(cfg_.width_px) - 0.5;
  const double v_hi = static_cast<double>(cfg_.height_px) - 0.5;

  for (std::size_t k = 0; k < cfg_.landmarks_fixed_m.size(); ++k) {
    // eq:camera:landmark: a body-fixed landmark rotated into inertial axes.
    // The central body is at the origin of the propagation frame, so its
    // inertial position contributes nothing to the sum.
    const Eigen::Vector3d l_i =
        latest_.geom.c_gcrf_to_bodyfixed.transpose() *
        cfg_.landmarks_fixed_m[k];
    // eq:camera:los, then eq:camera:apparent: the apparent direction carried
    // at the geometric range, through the shared aberration path.
    const Eigen::Vector3d d_i = l_i - r_cam_i;
    const double range = d_i.norm();
    const Eigen::Vector3d u_hat = (range > 0.0)
                                      ? (d_i / range).eval()
                                      : Eigen::Vector3d::UnitZ();
    const Eigen::Vector3d p_c =
        range * (c_b2c * (c_i2b * aberrate(u_hat, beta)));

    // eq:camera:proj, the exact central projection.
    const double u = cfg_.fx * (p_c.x() / p_c.z()) + cfg_.cx;
    const double v = cfg_.fy * (p_c.y() / p_c.z()) + cfg_.cy;
    px_[2 * k] = u;
    px_[2 * k + 1] = v;

    // Visibility tests in the normative order. The projection values are
    // logged even when a later test fails; consumers filter on the flag.
    bool vis = p_c.z() > 0.0;                                    // (1) in front
    vis = vis && u >= u_lo && u <= u_hi && v >= v_lo && v <= v_hi;  // (2) sensor
    // (3) eq:camera:nearside: the camera lies above the landmark's local
    // tangent plane. For a spherical body and a surface landmark this is
    // exactly equivalent to non-occlusion by the body itself, which is why
    // no separate self-occlusion test exists.
    vis = vis && (r_cam_i - l_i).dot(l_i) > 0.0;
    visible_[k] = vis ? 1 : 0;
  }

  // eq:camera:sunvec: the apparent Sun direction in camera axes, from the
  // same aberration path the sun sensor uses, so the two agree by
  // construction. Not a logged channel - the SRLOG camera record carries
  // pose and pixels only - but exposed for consumers.
  if (latest_.geom.ephemeris_valid) {
    const Eigen::Vector3d u_sun_i = (latest_.geom.r_sun_m - r_cam_i).normalized();
    sun_c_ = c_b2c * (c_i2b * aberrate(u_sun_i, beta));
  } else {
    sun_c_ = Eigen::Vector3d::Zero();
  }

  writer.write_sensor_camera(t_s, r_i, q_out, px_.data(), px_.size());
}

}  // namespace sensors
}  // namespace star
