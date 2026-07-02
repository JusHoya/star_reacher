// Two-body point-mass gravity model with fixed-step classical RK4 propagation
// (PRD Phase 1 placeholder dynamics; FR-11 fixed-step integrator tier).
//
// Math-library traceability (FR-29): the derivation lives in the two-body
// chapter of docs/mathlib; the implementation echoes its equation labels
// `eq:twobody:accel` and `eq:twobody:rk4` at the corresponding code.
#ifndef STAR_MODELS_TWOBODY_HPP
#define STAR_MODELS_TWOBODY_HPP

#include <Eigen/Dense>

namespace star {
namespace models {

// Translational state in the GCRF-oriented inertial frame (D-7/FR-3; Phase 1
// carries no attitude dynamics, so position and velocity are the whole state).
struct TwoBodyState {
  Eigen::Vector3d r_m;    // position [m], GCRF
  Eigen::Vector3d v_mps;  // velocity [m/s], GCRF
};

// Point-mass gravitational acceleration a = -mu * r / |r|^3 (eq:twobody:accel).
// gm_m3ps2 is the central body's gravitational parameter [m^3/s^2].
Eigen::Vector3d twobody_accel(double gm_m3ps2, const Eigen::Vector3d& r_m);

// One classical fixed-step RK4 step of size dt_s (eq:twobody:rk4). The
// dynamics are time-invariant, so the stage evaluations carry no explicit
// time argument.
TwoBodyState rk4_step(double gm_m3ps2, const TwoBodyState& state, double dt_s);

}  // namespace models
}  // namespace star

#endif  // STAR_MODELS_TWOBODY_HPP
