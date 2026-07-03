// Third-body gravitational perturbation (FR-6): the differential
// acceleration on a spacecraft, relative to its central body, from a
// point-mass third body, evaluated in Battin's cancellation-safe f(q)
// formulation. The naive difference of the direct and indirect terms loses
// up to log10(|r_third|/|r_sc|) significant digits in double precision;
// the f(q) form is algebraically identical but stays at a few ulp for all
// geometries with |r_sc| < |r_third| (Phase 3 exit criterion 7).
//
// Math-library traceability (FR-29): the derivation lives in the
// third-body chapter of docs/mathlib (ch:thirdbody); the implementation
// echoes its equation labels `eq:thirdbody:q`, `eq:thirdbody:fq`, and
// `eq:thirdbody:accel` at the corresponding code.
#ifndef STAR_MODELS_THIRDBODY_HPP
#define STAR_MODELS_THIRDBODY_HPP

#include <Eigen/Dense>

namespace star {
namespace models {

// Differential third-body acceleration [m/s^2] on a spacecraft at r_sc_m
// due to a third body of gravitational parameter gm_third_m3ps2 at
// r_third_m. Both positions are relative to the central body's center,
// resolved in a common (GCRF-oriented) frame; GM values come from the
// caller (DE440 header values wired by the force-composition layer) - the
// model hardcodes no body data. No per-call allocation; returns non-finite
// values (never throws) for the physically impossible inputs
// r_sc == r_third or r_third == 0, per the chapter's out-of-domain rule.
Eigen::Vector3d thirdbody_accel(double gm_third_m3ps2,
                                const Eigen::Vector3d& r_sc_m,
                                const Eigen::Vector3d& r_third_m);

}  // namespace models
}  // namespace star

#endif  // STAR_MODELS_THIRDBODY_HPP
