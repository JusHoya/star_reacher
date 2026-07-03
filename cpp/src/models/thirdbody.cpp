// Battin f(q) third-body differential acceleration (FR-6). Derivation and
// conditioning analysis: docs/mathlib chapter ch:thirdbody. This file uses
// only IEEE-754 basic operations (+, -, *, /, sqrt) in a fixed order, so
// under the D-10 flags (no FMA contraction, no fast-math) it evaluates
// bit-identically across platforms; the golden generator's Python mirror
// (tests/golden/thirdbody/generate.py, battin_double) reproduces the same
// operation sequence statement for statement, which is what makes its
// recorded generation-time margins transferable to every CI leg.
#include "star/models/thirdbody.hpp"

#include <cmath>

namespace star {
namespace models {

Eigen::Vector3d thirdbody_accel(double gm_third_m3ps2,
                                const Eigen::Vector3d& r_sc_m,
                                const Eigen::Vector3d& r_third_m) {
  // eq:thirdbody:q -- q = r.(r - 2 s) / (s.s), the exact fractional excess
  // of |s - r|^2 over |s|^2. Products only: no like-magnitude subtraction
  // for |r| < |s|, unlike the naive direct-minus-indirect difference.
  const double s2 = r_third_m.squaredNorm();
  const double q = r_sc_m.dot(r_sc_m - 2.0 * r_third_m) / s2;

  // eq:thirdbody:fq -- f(q) = (1+q)^{3/2} - 1, rationalized so the
  // subtraction of 1 never happens: f = q (3 + 3q + q^2)/(1 + (1+q)^{3/2}).
  const double opq = 1.0 + q;
  const double f = q * (3.0 + 3.0 * q + q * q) / (1.0 + opq * std::sqrt(opq));

  // eq:thirdbody:accel -- p = -(mu3 / |r - s|^3) (r + f(q) s). The addition
  // r + f s combines operands of order |r| and <= 3|r| (chapter,
  // conditioning note), never the O(|s|) operands the naive form subtracts.
  const Eigen::Vector3d d = r_sc_m - r_third_m;
  const double dn = d.norm();
  const double d3 = dn * dn * dn;
  return (-gm_third_m3ps2 / d3) * (r_sc_m + f * r_third_m);
}

}  // namespace models
}  // namespace star
