// Rotation kernel implementation (FR-3, D-7). Derivation and validation
// evidence: math library chapter ch:rotations. Equation labels from that
// chapter are echoed verbatim at the corresponding code (FR-29
// traceability).
#include "star/rotation.hpp"

#include <cmath>
#include <stdexcept>

namespace star {
namespace rotation {

// Elementary frame rotations (eq:rotations:r1r2r3). Signs follow the
// coordinate-transformation convention: R3(t) has +sin above the diagonal,
// matching the IAU SOFA Rx/Ry/Rz primitives the frames chain composes.
Eigen::Matrix3d r1(double angle_rad) {
  const double c = std::cos(angle_rad);
  const double s = std::sin(angle_rad);
  Eigen::Matrix3d m;
  m << 1.0, 0.0, 0.0,  //
      0.0, c, s,       //
      0.0, -s, c;
  return m;
}

Eigen::Matrix3d r2(double angle_rad) {
  const double c = std::cos(angle_rad);
  const double s = std::sin(angle_rad);
  Eigen::Matrix3d m;
  m << c, 0.0, -s,  //
      0.0, 1.0, 0.0,  //
      s, 0.0, c;
  return m;
}

Eigen::Matrix3d r3(double angle_rad) {
  const double c = std::cos(angle_rad);
  const double s = std::sin(angle_rad);
  Eigen::Matrix3d m;
  m << c, s, 0.0,  //
      -s, c, 0.0,  //
      0.0, 0.0, 1.0;
  return m;
}

// Hamilton product (eq:rotations:hamprod), written out in scalar-first
// components rather than delegating to Eigen's operator* so the convention
// is visible at the code and cannot silently follow an Eigen semantics
// change. (Eigen's Quaterniond product is in fact the Hamilton product;
// the golden tests pin this equivalence.)
Eigen::Quaterniond quat_multiply(const Eigen::Quaterniond& p,
                                 const Eigen::Quaterniond& q) {
  return Eigen::Quaterniond(
      p.w() * q.w() - p.x() * q.x() - p.y() * q.y() - p.z() * q.z(),
      p.w() * q.x() + p.x() * q.w() + p.y() * q.z() - p.z() * q.y(),
      p.w() * q.y() - p.x() * q.z() + p.y() * q.w() + p.z() * q.x(),
      p.w() * q.z() + p.x() * q.y() - p.y() * q.x() + p.z() * q.w());
}

Eigen::Quaterniond quat_conjugate(const Eigen::Quaterniond& q) {
  return Eigen::Quaterniond(q.w(), -q.x(), -q.y(), -q.z());
}

Eigen::Quaterniond quat_normalize(const Eigen::Quaterniond& q) {
  const double n = std::sqrt(q.w() * q.w() + q.x() * q.x() + q.y() * q.y() +
                             q.z() * q.z());
  if (!(n > 0.0) || !std::isfinite(n)) {
    // A zero or non-finite quaternion has no direction; normalizing it
    // would fabricate an attitude, so fail loudly instead (abort-on-
    // missing-critical-input discipline).
    throw std::domain_error("quat_normalize: zero or non-finite quaternion");
  }
  return Eigen::Quaterniond(q.w() / n, q.x() / n, q.y() / n, q.z() / n);
}

// DCM of a frame-transformation quaternion (eq:rotations:quat2dcm):
// C = (w^2 - v.v) I + 2 v v^T - 2 w [v]x.
Eigen::Matrix3d dcm_from_quat(const Eigen::Quaterniond& q_a2b) {
  const double w = q_a2b.w();
  const double x = q_a2b.x();
  const double y = q_a2b.y();
  const double z = q_a2b.z();
  const double ww = w * w;
  const double xx = x * x;
  const double yy = y * y;
  const double zz = z * z;
  Eigen::Matrix3d c;
  c(0, 0) = ww + xx - yy - zz;
  c(0, 1) = 2.0 * (x * y + w * z);
  c(0, 2) = 2.0 * (x * z - w * y);
  c(1, 0) = 2.0 * (x * y - w * z);
  c(1, 1) = ww - xx + yy - zz;
  c(1, 2) = 2.0 * (y * z + w * x);
  c(2, 0) = 2.0 * (x * z + w * y);
  c(2, 1) = 2.0 * (y * z - w * x);
  c(2, 2) = ww - xx - yy + zz;
  return c;
}

// v^B = C_A^B v^A via the quaternion sandwich, evaluated through the DCM so
// coordinate transformation and dcm_from_quat are one code path (their
// agreement is then structural, not a test-only property).
Eigen::Vector3d quat_transform(const Eigen::Quaterniond& q_a2b,
                               const Eigen::Vector3d& v_a) {
  return dcm_from_quat(q_a2b) * v_a;
}

// Shepperd's method (eq:rotations:shepperd; Shepperd 1978, J. Guidance and
// Control 1(3) 223-224; also Markley & Crassidis 2014, sec. 2.9.3). Of the
// four quantities 4w^2 = 1+tr(C) and 4x^2, 4y^2, 4z^2 = 1+2C(i,i)-tr(C),
// the largest is computed by square root and the remaining components by
// division, so no divisor is smaller than the largest component - the
// extraction never cancels catastrophically, for any input attitude.
Eigen::Quaterniond quat_from_dcm(const Eigen::Matrix3d& c) {
  const double trace = c(0, 0) + c(1, 1) + c(2, 2);
  // Index of the largest of (trace, C00, C11, C22) selects the branch.
  int pivot = 3;
  double best = trace;
  for (int i = 0; i < 3; ++i) {
    if (c(i, i) > best) {
      best = c(i, i);
      pivot = i;
    }
  }
  double w;
  double x;
  double y;
  double z;
  if (pivot == 3) {
    w = 0.5 * std::sqrt(1.0 + trace);
    const double d = 4.0 * w;
    // Off-diagonal differences isolate the vector part; signs follow the
    // frame-transformation DCM of eq:rotations:quat2dcm (e.g. C12 - C21 =
    // 4wx).
    x = (c(1, 2) - c(2, 1)) / d;
    y = (c(2, 0) - c(0, 2)) / d;
    z = (c(0, 1) - c(1, 0)) / d;
  } else if (pivot == 0) {
    x = 0.5 * std::sqrt(1.0 + 2.0 * c(0, 0) - trace);
    const double d = 4.0 * x;
    w = (c(1, 2) - c(2, 1)) / d;
    y = (c(0, 1) + c(1, 0)) / d;
    z = (c(2, 0) + c(0, 2)) / d;
  } else if (pivot == 1) {
    y = 0.5 * std::sqrt(1.0 + 2.0 * c(1, 1) - trace);
    const double d = 4.0 * y;
    w = (c(2, 0) - c(0, 2)) / d;
    x = (c(0, 1) + c(1, 0)) / d;
    z = (c(1, 2) + c(2, 1)) / d;
  } else {
    z = 0.5 * std::sqrt(1.0 + 2.0 * c(2, 2) - trace);
    const double d = 4.0 * z;
    w = (c(0, 1) - c(1, 0)) / d;
    x = (c(2, 0) + c(0, 2)) / d;
    y = (c(1, 2) + c(2, 1)) / d;
  }
  // q and -q are the same rotation; the project convention resolves the
  // sign to w >= 0 so extractions are reproducible bit for bit.
  if (w < 0.0) {
    w = -w;
    x = -x;
    y = -y;
    z = -z;
  }
  return Eigen::Quaterniond(w, x, y, z);
}

// 3-2-1: C = R1(a3) R2(a2) R3(a1) (eq:rotations:e321).
Eigen::Matrix3d dcm_from_euler321(double a1_rad, double a2_rad,
                                  double a3_rad) {
  return r1(a3_rad) * (r2(a2_rad) * r3(a1_rad));
}

// 3-1-3: C = R3(a3) R1(a2) R3(a1) (eq:rotations:e313).
Eigen::Matrix3d dcm_from_euler313(double a1_rad, double a2_rad,
                                  double a3_rad) {
  return r3(a3_rad) * (r1(a2_rad) * r3(a1_rad));
}

// 3-2-1 extraction. Element layout of eq:rotations:e321:
//   C02 = -sin a2, C00 = cos a2 cos a1, C01 = cos a2 sin a1,
//   C12 = sin a3 cos a2, C22 = cos a3 cos a2.
void euler321_from_dcm(const Eigen::Matrix3d& c, double& a1_rad,
                       double& a2_rad, double& a3_rad) {
  // hypot-form middle angle: full precision near |a2| = pi/2 where the
  // asin form would lose half the significand (chapter, implementation
  // notes).
  const double cos_a2 = std::sqrt(c(0, 0) * c(0, 0) + c(0, 1) * c(0, 1));
  a2_rad = std::atan2(-c(0, 2), cos_a2);
  if (cos_a2 > 0.0) {
    a1_rad = std::atan2(c(0, 1), c(0, 0));
    a3_rad = std::atan2(c(1, 2), c(2, 2));
  } else {
    // Exact gimbal lock: axes 3 and 1 coincide and only a3 -/+ a1 is
    // observable. Deterministic policy a1 = 0; the sub-block C10, C11 then
    // yields a3 directly: at a2 = +pi/2, C10 = sin(a3 - a1) and
    // C11 = cos(a3 - a1); at a2 = -pi/2, C10 = -sin(a3 + a1).
    a1_rad = 0.0;
    if (c(0, 2) < 0.0) {  // sin a2 = +1
      a3_rad = std::atan2(c(1, 0), c(1, 1));
    } else {  // sin a2 = -1
      a3_rad = std::atan2(-c(1, 0), c(1, 1));
    }
  }
}

// 3-1-3 extraction. Element layout of eq:rotations:e313:
//   C22 = cos a2, C20 = sin a1 sin a2, C21 = -cos a1 sin a2,
//   C02 = sin a3 sin a2, C12 = cos a3 sin a2.
void euler313_from_dcm(const Eigen::Matrix3d& c, double& a1_rad,
                       double& a2_rad, double& a3_rad) {
  const double sin_a2 = std::sqrt(c(0, 2) * c(0, 2) + c(1, 2) * c(1, 2));
  a2_rad = std::atan2(sin_a2, c(2, 2));
  if (sin_a2 > 0.0) {
    a1_rad = std::atan2(c(2, 0), -c(2, 1));
    a3_rad = std::atan2(c(0, 2), c(1, 2));
  } else {
    // Exact lock (a2 = 0 or pi): both z-rotations share an axis. Policy
    // a1 = 0; the upper-left block is then a pure z-rotation by a3 + a1
    // (a2 = 0) or by a3 - a1 (a2 = pi), read off as before.
    a1_rad = 0.0;
    if (c(2, 2) > 0.0) {  // a2 = 0
      a3_rad = std::atan2(c(0, 1), c(0, 0));
    } else {  // a2 = pi
      a3_rad = std::atan2(-c(0, 1), c(0, 0));
    }
  }
}

}  // namespace rotation
}  // namespace star
