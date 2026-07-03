// Two-body point-mass gravity model (PRD Phase 1 placeholder dynamics).
// Propagation uses the shared integrator library (star/integrate.hpp,
// FR-11); this module owns only the dynamics.
//
// Math-library traceability (FR-29): the derivation lives in the two-body
// chapter of docs/mathlib; the implementation echoes its equation labels
// `eq:twobody:accel` and `eq:twobody:firstorder` at the corresponding code.
// The Runge-Kutta discretization is derived in the integrators chapter
// (ch:integrators).
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

// First-order ODE right-hand side for the shared integrators
// (eq:twobody:firstorder): y = [r, v], ydot = [v, a(r)]. The dynamics are
// time-invariant; t is accepted to satisfy the integrate::RhsRef signature.
void twobody_rhs(double gm_m3ps2, double t, const double* y, double* ydot);

}  // namespace models
}  // namespace star

#endif  // STAR_MODELS_TWOBODY_HPP
