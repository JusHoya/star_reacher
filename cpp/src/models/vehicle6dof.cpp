// Vehicle 6DOF composition layer (Phase 4). Derivation: docs/mathlib chapter
// ch:vehicle6dof. The geometry here (pad state, pitch-program attitude, staging
// remap, EOM summation order) is the load-bearing new physics of the phase;
// every force/torque term itself lives in its own chapter-tracked module. Only
// the sin/cos of the pad latitude/longitude and the pitch/azimuth angles reach
// libm; the remap and acceleration assembly are IEEE-754 basic operations in a
// fixed order (D-10).
#include "star/models/vehicle6dof.hpp"

#include <cmath>
#include <stdexcept>

#include "star/rotation.hpp"

namespace star {
namespace models {

double pwl_interp_clamped(const std::vector<double>& xs,
                          const std::vector<double>& ys, double x) {
  // Verbatim arithmetic of the original Phase 4 in-loop lambda (run.cpp
  // pitch-program mode): clamp at the endpoints, then one linear segment.
  // Changing any operation or its order here would break both the Phase 4
  // byte-freeze and the Phase 6 open-loop/closed-loop command equality.
  if (x <= xs.front()) return ys.front();
  if (x >= xs.back()) return ys.back();
  for (std::size_t j = 0; j + 1 < xs.size(); ++j) {
    if (x <= xs[j + 1]) {
      const double w = (x - xs[j]) / (xs[j + 1] - xs[j]);
      return ys[j] + w * (ys[j + 1] - ys[j]);
    }
  }
  return ys.back();
}

PadState geodetic_pad_state(double lat_rad, double lon_rad, double alt_m,
                            const Eigen::Matrix3d& c_gcrf_to_itrf,
                            double omega_earth_radps, double a_m,
                            double inv_f) {
  if (!(a_m > 0.0) || !(inv_f > 0.0)) {
    throw std::invalid_argument(
        "vehicle6dof: pad ellipsoid needs positive a_m and inv_f");
  }
  const double f = 1.0 / inv_f;
  const double e2 = f * (2.0 - f);
  const double slat = std::sin(lat_rad);
  const double clat = std::cos(lat_rad);
  const double slon = std::sin(lon_rad);
  const double clon = std::cos(lon_rad);
  // eq:vehicle6dof:padstate -- prime-vertical radius of curvature and the
  // geodetic-to-ECEF forward transform (Vallado Alg. 51 forward form).
  const double n_rad = a_m / std::sqrt(1.0 - e2 * slat * slat);
  const Eigen::Vector3d r_ecef((n_rad + alt_m) * clat * clon,
                               (n_rad + alt_m) * clat * slon,
                               (n_rad * (1.0 - e2) + alt_m) * slat);
  // ITRF -> GCRF is the transpose of the Earth-fixed rotation (orthonormal).
  const Eigen::Matrix3d c_itrf_to_gcrf = c_gcrf_to_itrf.transpose();
  const Eigen::Vector3d r_i = c_itrf_to_gcrf * r_ecef;
  // Earth angular-velocity vector in GCRF: omega * (ITRF z-axis expressed in
  // GCRF), the third row of C_GCRF->ITRF (eq:drag:vrel plumbing). v = w x r is
  // then exactly the pad co-rotation velocity (EC-10).
  const Eigen::Vector3d omega_i =
      omega_earth_radps * c_gcrf_to_itrf.row(2).transpose();
  PadState p;
  p.r_i_m = r_i;
  p.v_i_mps = omega_i.cross(r_i);
  // Ellipsoid-normal ENU basis in ECEF, rotated to GCRF.
  p.up_i = c_itrf_to_gcrf * Eigen::Vector3d(clat * clon, clat * slon, slat);
  p.east_i = c_itrf_to_gcrf * Eigen::Vector3d(-slon, clon, 0.0);
  p.north_i =
      c_itrf_to_gcrf * Eigen::Vector3d(-slat * clon, -slat * slon, clat);
  return p;
}

Eigen::Vector3d pitch_program_axis(double az_rad, double pitch_rad,
                                   const Eigen::Vector3d& up_i,
                                   const Eigen::Vector3d& east_i,
                                   const Eigen::Vector3d& north_i) {
  const double cp = std::cos(pitch_rad);
  const double sp = std::sin(pitch_rad);
  const double sa = std::sin(az_rad);
  const double ca = std::cos(az_rad);
  // eq:vehicle6dof:pitchaxis -- elevation pitch above the horizontal, in the
  // ground-track direction set by the flight azimuth (east of north).
  return cp * (sa * east_i + ca * north_i) + sp * up_i;
}

Eigen::Quaterniond attitude_from_body_x(const Eigen::Vector3d& xb_i,
                                        const Eigen::Vector3d& ref_i) {
  // eq:vehicle6dof:attitude -- body +X along xb_i, body +Y from Gram-Schmidt of
  // ref_i, body +Z completing the right-handed triad. The columns of C_b2i are
  // the body axes in inertial; q_i2b is extracted from its transpose C_i2b.
  const double xn = xb_i.norm();
  if (!(xn > 0.0) || !xb_i.allFinite()) {
    throw std::invalid_argument(
        "vehicle6dof: attitude_from_body_x needs a nonzero finite body-X");
  }
  const Eigen::Vector3d ex = xb_i / xn;
  Eigen::Vector3d ref = ref_i;
  Eigen::Vector3d ey = ref - ref.dot(ex) * ex;
  if (!(ey.norm() > 1e-8)) {
    // ref_i is (near-)parallel to xb_i: pick a fixed alternate so the triad is
    // always well-conditioned (roll is a free axisymmetric choice).
    ref = std::fabs(ex.z()) < 0.9 ? Eigen::Vector3d::UnitZ()
                                  : Eigen::Vector3d::UnitX();
    ey = ref - ref.dot(ex) * ex;
  }
  ey.normalize();
  const Eigen::Vector3d ez = ex.cross(ey);
  Eigen::Matrix3d c_b2i;
  c_b2i.col(0) = ex;
  c_b2i.col(1) = ey;
  c_b2i.col(2) = ez;
  return rotation::quat_from_dcm(c_b2i.transpose());
}

Eigen::Vector3d omega_from_quaternions(const Eigen::Quaterniond& q0_i2b,
                                       const Eigen::Quaterniond& q1_i2b,
                                       double dt_s) {
  if (!(dt_s > 0.0)) {
    throw std::invalid_argument("vehicle6dof: omega_from_quaternions dt > 0");
  }
  // Relative body rotation dq = q0^{-1} (x) q1 (frame-transformation
  // composition, ch:rotations); its vector part is sin(theta/2) about the body
  // rotation axis, so omega_b = 2 dq_vec / dt to first order. Small-angle over
  // one control cycle, used for logging only.
  const Eigen::Quaterniond dq =
      rotation::quat_multiply(rotation::quat_conjugate(q0_i2b), q1_i2b);
  Eigen::Vector3d v(dq.x(), dq.y(), dq.z());
  if (dq.w() < 0.0) {
    v = -v;  // take the short rotation (w >= 0 half of the double cover)
  }
  return (2.0 / dt_s) * v;
}

SeparationRemap separation_remap(const Eigen::Vector3d& cg_old_b_m,
                                 const Eigen::Vector3d& cg_new_b_m,
                                 const Eigen::Vector3d& r_old_i_m,
                                 const Eigen::Vector3d& v_old_i_mps,
                                 const Eigen::Quaterniond& q_i2b,
                                 const Eigen::Vector3d& omega_b_radps) {
  // eq:vehicle6dof:remap -- the tracked CG jumps by Delta r_cg (body), rotated
  // to inertial; the new CG is a material point of the retained body, so its
  // velocity is v_old + omega x Delta r_cg (omega unchanged, torque-free).
  const Eigen::Matrix3d c_i2b = rotation::dcm_from_quat(q_i2b);
  const Eigen::Matrix3d c_b2i = c_i2b.transpose();
  const Eigen::Vector3d dcg_i = c_b2i * (cg_new_b_m - cg_old_b_m);
  const Eigen::Vector3d omega_i = c_b2i * omega_b_radps;
  SeparationRemap out;
  out.r_new_i_m = r_old_i_m + dcg_i;
  out.v_new_i_mps = v_old_i_mps + omega_i.cross(dcg_i);
  return out;
}

Eigen::Vector3d composed_translational_accel(const Eigen::Vector3d& a_env_mps2,
                                             const Eigen::Vector3d& f_body_b_n,
                                             const Eigen::Matrix3d& c_b2i,
                                             double mass_kg) {
  if (!(mass_kg > 0.0)) {
    throw std::invalid_argument(
        "vehicle6dof: composed acceleration needs positive mass");
  }
  // eq:vehicle6dof:transaccel -- environment acceleration first (its own fixed
  // internal order, ch:environment), then the body force rotated into GCRF and
  // divided by the current composite mass.
  return a_env_mps2 + (c_b2i * f_body_b_n) / mass_kg;
}

}  // namespace models
}  // namespace star
