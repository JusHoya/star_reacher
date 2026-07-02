// Two-body point-mass dynamics and classical RK4 stepping. This file is the
// chapter-manifest anchor for the two-body model (docs/mathlib chapter
// `ch:twobody`); the equation labels referenced below are defined there.
//
// Derivation source: Vallado, Fundamentals of Astrodynamics and Applications,
// two-body relative motion; classical RK4 per Hairer, Norsett, and Wanner,
// Solving Ordinary Differential Equations I.
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

TwoBodyState rk4_step(double gm_m3ps2, const TwoBodyState& state, double dt_s) {
  // eq:twobody:rk4  classical Runge-Kutta 4 on y' = f(y), y = (r, v),
  // f = (v, a(r)). Stage order and the summation order in the final combine
  // are fixed; do not reorder (bit-identity contract, D-10).
  const Eigen::Vector3d k1_r = state.v_mps;
  const Eigen::Vector3d k1_v = twobody_accel(gm_m3ps2, state.r_m);

  const Eigen::Vector3d k2_r = state.v_mps + 0.5 * dt_s * k1_v;
  const Eigen::Vector3d k2_v =
      twobody_accel(gm_m3ps2, state.r_m + 0.5 * dt_s * k1_r);

  const Eigen::Vector3d k3_r = state.v_mps + 0.5 * dt_s * k2_v;
  const Eigen::Vector3d k3_v =
      twobody_accel(gm_m3ps2, state.r_m + 0.5 * dt_s * k2_r);

  const Eigen::Vector3d k4_r = state.v_mps + dt_s * k3_v;
  const Eigen::Vector3d k4_v = twobody_accel(gm_m3ps2, state.r_m + dt_s * k3_r);

  TwoBodyState out;
  out.r_m = state.r_m + (dt_s / 6.0) * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r);
  out.v_mps =
      state.v_mps + (dt_s / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v);
  return out;
}

}  // namespace models
}  // namespace star
