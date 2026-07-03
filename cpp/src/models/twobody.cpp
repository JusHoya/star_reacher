// Two-body point-mass dynamics. This file is the chapter-manifest anchor for
// the two-body model (docs/mathlib chapter `ch:twobody`); the equation labels
// referenced below are defined there. Time stepping is NOT implemented here:
// propagation goes through the shared integrator library (star/integrate.hpp,
// chapter ch:integrators), so every model supplies only its right-hand side.
//
// Derivation source: Vallado, Fundamentals of Astrodynamics and Applications,
// two-body relative motion.
#include "star/models/twobody.hpp"

namespace star {
namespace models {

Eigen::Vector3d twobody_accel(double gm_m3ps2, const Eigen::Vector3d& r_m) {
  // eq:twobody:accel  a = -mu * r / |r|^3
  // The norm is computed once and cubed by multiplication (not pow) so the
  // operation sequence is fixed and identical across platforms (D-10).
  const double rn = r_m.norm();
  return (-gm_m3ps2 / (rn * rn * rn)) * r_m;
}

void twobody_rhs(double gm_m3ps2, double /*t*/, const double* y,
                 double* ydot) {
  // eq:twobody:firstorder  y = (r, v), ydot = (v, a(r)). Component order is
  // fixed: position derivative first, then the acceleration (D-10).
  const Eigen::Map<const Eigen::Vector3d> r(y);
  const Eigen::Vector3d a = twobody_accel(gm_m3ps2, r);
  ydot[0] = y[3];
  ydot[1] = y[4];
  ydot[2] = y[5];
  ydot[3] = a[0];
  ydot[4] = a[1];
  ydot[5] = a[2];
}

}  // namespace models
}  // namespace star
