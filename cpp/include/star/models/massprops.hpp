// Analytic settled-tank vehicle mass properties (FR-10, assumption A-2):
// draining-cylinder propellant slugs, mass-weighted CG and parallel-axis
// inertia composition about the composite CG, exact analytic depletion
// rates (the Idot the FR-1 rigid-body dynamics consumes is never a finite
// difference), and the closed-form single-body removal that models a
// staging/jettison event. All geometry is resolved in the single FR-13
// structural frame (+X forward); tank axes are restricted to +X (the
// closed forms are specific to that orientation - documented in ch:massprops).
//
// The core never parses text (D-2): the structs below are plain SI-unit
// value types the Python validator fills across the binding; every
// function is pure and allocation-free apart from reading the caller's
// vectors.
//
// Math-library traceability (FR-29): the derivation lives in the mass-
// properties chapter of docs/mathlib (ch:massprops); the implementation
// echoes its equation labels eq:massprops:fillheight, eq:massprops:slug,
// eq:massprops:slugrates, eq:massprops:parallelaxis, eq:massprops:compose,
// eq:massprops:composerates, and eq:massprops:remove at the corresponding
// code.
#ifndef STAR_MODELS_MASSPROPS_HPP
#define STAR_MODELS_MASSPROPS_HPP

#include <vector>

#include <Eigen/Dense>

namespace star {
namespace models {

// Cylindrical propellant tank, axis along vehicle +X (v1 restriction:
// off-X tank positions are supported, off-X orientations are not).
// aft_center_m is the center of the aft (-X) interior face the settled
// liquid rests against (A-2). All members SI, FR-13 vocabulary.
struct TankParams {
  double radius_m = 0.0;      // interior radius
  double length_m = 0.0;      // interior length along +X
  Eigen::Vector3d aft_center_m = Eigen::Vector3d::Zero();
  double density_kgpm3 = 0.0;  // propellant bulk density
  double initial_mass_kg = 0.0;  // propellant load at t0 (validated by
                                 // the Python layer against capacity)
};

// One rigid body of a composition: a stage dry body, a jettisonable
// item, or a propellant slug. cg_m is in the vehicle frame; inertia_kgm2
// is about the body's own CG, resolved in vehicle-frame axes.
struct BodyProps {
  double mass_kg = 0.0;
  Eigen::Vector3d cg_m = Eigen::Vector3d::Zero();
  Eigen::Matrix3d inertia_kgm2 = Eigen::Matrix3d::Zero();
};

// Time derivatives of a BodyProps. mdot_kgps is the SIGNED d(mass)/dt -
// negative while a tank drains. Propulsion reports consumption as a
// positive flow (ch:propulsion); the integration layer negates it when
// feeding tank depletion, and this sign convention is the reason the
// chain-rule rates below need no special cases.
struct BodyRates {
  double mdot_kgps = 0.0;
  Eigen::Vector3d cg_rate_mps = Eigen::Vector3d::Zero();
  Eigen::Matrix3d inertia_rate_kgm2ps = Eigen::Matrix3d::Zero();
};

// Propellant capacity rho * (pi R^2) * L [kg], evaluated in this fixed
// order so every caller and the domain check share one binary64 value.
// Throws std::domain_error for non-finite or non-positive dimensions or
// density.
double tank_capacity_kg(const TankParams& tank);

// Settled fill height h = m_p / (rho pi R^2) [m]
// (eq:massprops:fillheight). Throws std::domain_error if propellant_kg
// is negative, non-finite, or exceeds the tank capacity.
double tank_fill_height_m(const TankParams& tank, double propellant_kg);

// Settled-slug mass, CG, and own-CG inertia (eq:massprops:slug). The
// returned mass is propellant_kg verbatim (never reconstructed from the
// fill height), which is what makes the composite wet-mass identity
// bit-exact. An empty tank returns zero mass and inertia with the CG at
// the aft-face center.
BodyProps tank_slug_props(const TankParams& tank, double propellant_kg);

// Exact slug depletion rates by the chain rule through the fill height
// (eq:massprops:slugrates). mdot_kgps is the signed d(m_p)/dt.
BodyRates tank_slug_rates(const TankParams& tank, double propellant_kg,
                          double mdot_kgps);

// Composite mass, CG, and inertia about the composite CG
// (eq:massprops:compose). Masses are summed in vector order - the
// documented FR-10 order is fixed bodies first, then one slug per tank -
// so the wet mass equals the same-order sum of the inputs bit-exactly.
// Throws std::domain_error for an empty vector, any negative or
// non-finite mass, or a non-positive total mass.
BodyProps compose(const std::vector<BodyProps>& bodies);

// Composite rates (eq:massprops:composerates): rates[i] belongs to
// bodies[i] (fixed bodies pass default-constructed zero rates). Fixed
// bodies still contribute to the inertia rate through the motion of the
// composite CG underneath them. Throws std::domain_error on size
// mismatch or the compose() domain violations.
BodyRates compose_rates(const std::vector<BodyProps>& bodies,
                        const std::vector<BodyRates>& rates);

// Closed-form composite after removing one body (eq:massprops:remove):
// the discrete staging/jettison update of FR-10. `composite` must be a
// composition that contains `item`; throws std::domain_error if the
// remainder mass would be non-positive.
BodyProps remove_body(const BodyProps& composite, const BodyProps& item);

}  // namespace models
}  // namespace star

#endif  // STAR_MODELS_MASSPROPS_HPP
