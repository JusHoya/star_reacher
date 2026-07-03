// Rotation kernel for the star:: core (FR-3, decision D-7): Hamilton
// quaternions, direction-cosine matrices, elementary frame rotations, and
// the 3-2-1 / 3-1-3 Euler sequences. Derivations, domain bounds, and
// validation evidence are in the math library chapter ch:rotations
// (docs/mathlib/chapters/rotations.tex).
//
// Conventions (ratified in the notation chapter; D-7):
//
// - Quaternions are HAMILTON quaternions (ij = k), components ordered
//   SCALAR-FIRST [w, x, y, z] in every log, API, and document. The Hamilton
//   product is eq:rotations:hamprod. The JPL scalar-last convention is not
//   used anywhere in this project.
// - An attitude quaternion q_a2b is a frame transformation from frame A to
//   frame B. Its DCM C_A^B transforms coordinates, v^B = C_A^B v^A
//   (eq:rotations:quat2dcm), and successive transformations compose as
//   C_A^C = C_B^C * C_A^B; on quaternions q_a2c = q_a2b (x) q_b2c.
// - Elementary rotations R1/R2/R3 (eq:rotations:r1r2r3) are FRAME rotations
//   (coordinate transformations): R3(t) maps the coordinates of a fixed
//   vector into a frame rotated by +t about +z. This matches the IAU SOFA
//   Rx/Ry/Rz convention used by the Earth-orientation chain in
//   star/frames.hpp.
//
// Mandatory Eigen note (D-7). The core stores quaternions as
// Eigen::Quaterniond, whose component ordering differs between constructor
// and storage:
//   - the value constructor takes the scalar FIRST:  Quaterniond(w, x, y, z);
//   - the internal storage exposed by .coeffs() is scalar-LAST: [x, y, z, w].
// Consequences, binding on all core and bindings code:
//   1. Never serialize .coeffs() verbatim; externally visible quaternions
//      are emitted scalar-first through the named accessors .w() .x() .y()
//      .z() (notation chapter, Table "Eigen mapping").
//   2. Eigen's operator* on vectors and .toRotationMatrix() implement the
//      ACTIVE rotation v -> q v q^-1; applied to a frame transformation
//      q_a2b they produce C_B^A, the TRANSPOSE of eq:rotations:quat2dcm.
//      This kernel therefore provides dcm_from_quat()/quat_transform() with
//      the project (frame transformation) semantics; core code uses these,
//      not .toRotationMatrix(), for frame mappings.
#ifndef STAR_ROTATION_HPP
#define STAR_ROTATION_HPP

#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace star {
namespace rotation {

// Elementary frame rotations about the x, y, z axes (eq:rotations:r1r2r3).
// R1/R2/R3 name the rotation axis by index, the aerospace convention used in
// Euler sequence names like "3-2-1".
Eigen::Matrix3d r1(double angle_rad);
Eigen::Matrix3d r2(double angle_rad);
Eigen::Matrix3d r3(double angle_rad);

// Hamilton product p (x) q, scalar-first semantics (eq:rotations:hamprod).
// Composition order: q_a2c = quat_multiply(q_a2b, q_b2c).
Eigen::Quaterniond quat_multiply(const Eigen::Quaterniond& p,
                                 const Eigen::Quaterniond& q);

// Conjugate [w, -x, -y, -z]; equals the inverse for unit quaternions.
Eigen::Quaterniond quat_conjugate(const Eigen::Quaterniond& q);

// q / |q|. Throws std::domain_error on a zero or non-finite norm rather
// than returning a silently invalid attitude.
Eigen::Quaterniond quat_normalize(const Eigen::Quaterniond& q);

// Coordinate transformation by a unit frame-transformation quaternion:
// v^B = quat_transform(q_a2b, v^A), the sandwich q^-1 (x) v (x) q of
// eq:rotations:quat2dcm.
Eigen::Vector3d quat_transform(const Eigen::Quaterniond& q_a2b,
                               const Eigen::Vector3d& v_a);

// DCM C_A^B of a unit frame-transformation quaternion q_a2b
// (eq:rotations:quat2dcm). Unit norm is the caller's invariant, mirroring
// quat_from_dcm below: a non-unit input is not detected and yields the
// rotation scaled by |q|^2 (not orthonormal); normalize first via
// quat_normalize when the invariant is not already established.
Eigen::Matrix3d dcm_from_quat(const Eigen::Quaterniond& q_a2b);

// Inverse mapping via Shepperd's method (eq:rotations:shepperd): the four
// candidate denominators 1+trace and 1+2*C(i,i)-trace are compared and the
// largest is used, so the extraction is numerically robust for every
// attitude (no small-divisor branch). Sign convention: the returned
// quaternion has w >= 0 (of the +/-q pair representing the same rotation).
// The input must be a proper rotation matrix; orthonormality is the
// caller's invariant and is gated by the property tests, not re-checked
// here.
Eigen::Quaterniond quat_from_dcm(const Eigen::Matrix3d& c_a2b);

// Euler sequences. Angles are in radians and in APPLICATION order: the
// sequence name lists the rotation axes in the order applied, so
// 3-2-1 is C = R1(a3) R2(a2) R3(a1)   (eq:rotations:e321)
// 3-1-3 is C = R3(a3) R1(a2) R3(a1)   (eq:rotations:e313).
Eigen::Matrix3d dcm_from_euler321(double a1_rad, double a2_rad,
                                  double a3_rad);
Eigen::Matrix3d dcm_from_euler313(double a1_rad, double a2_rad,
                                  double a3_rad);

// Euler extraction. The middle angle is recovered through atan2 of a
// hypot-form pair (never asin alone), so a2 itself stays accurate to
// machine precision arbitrarily close to gimbal lock. At lock the first
// and third rotations share an axis and only their sum (or difference) is
// observable; the deterministic policy is a1 = 0 with the whole degenerate
// rotation assigned to a3. Near (not at) lock the individual a1/a3 values
// are ill-conditioned - their errors grow as eps/delta, delta the angular
// distance from lock - while the recomposed rotation stays accurate; the
// quantitative bound is derived and tested per the chapter (ch:rotations,
// domain of validity).
//
// Ranges: 3-2-1 returns a2 in [-pi/2, pi/2]; 3-1-3 returns a2 in [0, pi];
// a1 and a3 are in [-pi, pi] (the atan2 range).
void euler321_from_dcm(const Eigen::Matrix3d& c, double& a1_rad,
                       double& a2_rad, double& a3_rad);
void euler313_from_dcm(const Eigen::Matrix3d& c, double& a1_rad,
                       double& a2_rad, double& a3_rad);

}  // namespace rotation
}  // namespace star

#endif  // STAR_ROTATION_HPP
