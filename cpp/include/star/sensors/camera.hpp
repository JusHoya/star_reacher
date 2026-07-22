// Camera hook (FR-23, ch:camera): a draw-free, noise-free sensor emitting
// geometric truth for offline external rendering and optical-navigation
// research - the camera pose, the pinhole intrinsics, and optional landmark
// pixel projections. There is no in-core rendering.
//
// Scope note, binding on this module: landmarks are body-fixed points on the
// CENTRAL body. That is the surface-relative optical-navigation case the
// hook targets, and it is exactly served by the central-body-fixed rotation
// the environment model already composes for the dynamics. Landmarks on a
// third body, and the multi-body occlusion test of eq:camera:occlusion, need
// a configured body table this module does not plumb; the near-side test of
// eq:camera:nearside is exact for a spherical central body and a surface
// landmark, so self-occlusion is handled without it.
#ifndef STAR_SENSORS_CAMERA_HPP
#define STAR_SENSORS_CAMERA_HPP

#include <cstdint>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "star/gnc/config.hpp"
#include "star/sensors/sensor.hpp"

namespace star {
namespace sensors {

struct CameraCfg {
  // Pinhole intrinsics (eq:camera:K), pixels. The pixel origin is the CENTER
  // of the top-left pixel, u rightward along columns, v downward along rows;
  // the sensor spans u in [-1/2, W - 1/2], v in [-1/2, H - 1/2].
  double fx = 0.0;
  double fy = 0.0;
  double cx = 0.0;
  double cy = 0.0;
  std::uint32_t width_px = 0;
  std::uint32_t height_px = 0;
  // Extrinsics: mount offset relative to the composite center of mass (the
  // point whose state the truth channels carry) and the body-to-camera
  // rotation. Camera axes are the computer-vision convention: +Z along the
  // boresight, +X toward increasing u, +Y toward increasing v (downward).
  Eigen::Vector3d r_cam_b_m = Eigen::Vector3d::Zero();
  Eigen::Quaterniond q_b2c = Eigen::Quaterniond::Identity();
  // Landmarks in central-body-fixed axes, in the configured order. The
  // declared count is fixed at header-write time (the log record carries
  // 2L interleaved pixel coordinates).
  std::vector<Eigen::Vector3d> landmarks_fixed_m;
};

CameraCfg parse_camera_cfg(const gnc::GncSensorCfg& cfg);

class CameraHook final : public ISensor {
 public:
  // No PRNG stream: the hook is noise-free truth and consumes no draws
  // (ch:camera implementation note 1), so it takes no seed.
  CameraHook(std::uint32_t sample_rate_hz, const CameraCfg& cfg);

  const char* kind() const override { return "camera"; }
  std::uint32_t sample_rate_hz() const override { return rate_hz_; }
  void accumulate(const SensorCycleTruth& truth) override;
  void sample(double t_s, log::SrlogWriter& writer) override;

  std::size_t landmark_count() const { return cfg_.landmarks_fixed_m.size(); }
  // Interleaved (u0, v0, u1, v1, ...) of the most recent sample.
  const std::vector<double>& last_pixels() const { return px_; }
  // Per-landmark visibility of the most recent sample. The pixel values are
  // emitted even when a landmark is not visible - consumers filter on this
  // flag (ch:camera implementation note 3).
  const std::vector<char>& last_visible() const { return visible_; }
  // Apparent Sun unit direction in camera axes (eq:camera:sunvec). Computed
  // for consumers; the SRLOG camera record layout carries pose and pixels
  // only, so this is not a logged channel.
  const Eigen::Vector3d& last_sun_c() const { return sun_c_; }

 private:
  std::uint32_t rate_hz_;
  CameraCfg cfg_;
  SensorCycleTruth latest_;
  std::vector<double> px_;
  std::vector<char> visible_;
  Eigen::Vector3d sun_c_ = Eigen::Vector3d::Zero();
};

}  // namespace sensors
}  // namespace star

#endif  // STAR_SENSORS_CAMERA_HPP
