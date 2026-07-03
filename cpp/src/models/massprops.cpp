// Analytic settled-tank mass properties (FR-10). Derivation: docs/mathlib
// chapter ch:massprops. Every quantity is a closed form in IEEE-754 basic
// operations with no libm calls, evaluated in a fixed order (D-10), so the
// module is bit-portable across platforms. The slug mass is the propellant
// mass argument verbatim and compositions sum masses in vector order, which
// is what makes the exit-criterion-2 wet-mass identity bit-exact.
#include "star/models/massprops.hpp"

#include <cmath>
#include <stdexcept>

#include "star/constants.hpp"

namespace star {
namespace models {

namespace {

// Cross-sectional area pi R^2 shared by the fill-height, capacity, and
// rate paths so they cannot disagree about the geometry.
double cross_section_m2(const TankParams& tank) {
  if (!std::isfinite(tank.radius_m) || tank.radius_m <= 0.0 ||
      !std::isfinite(tank.length_m) || tank.length_m <= 0.0 ||
      !std::isfinite(tank.density_kgpm3) || tank.density_kgpm3 <= 0.0) {
    throw std::domain_error(
        "massprops: tank radius, length, and density must be finite and "
        "positive");
  }
  return constants::PI * tank.radius_m * tank.radius_m;
}

// eq:massprops:parallelaxis -- P(m, d) = m (|d|^2 E - d d^T), the
// parallel-axis inertia increment for a point offset d.
Eigen::Matrix3d parallel_axis(double m, const Eigen::Vector3d& d) {
  return m * (d.squaredNorm() * Eigen::Matrix3d::Identity() -
              d * d.transpose());
}

// eq:massprops:composerates -- exact time derivative of the parallel-axis
// increment: mdot (|d|^2 E - d d^T) + m (2 (d . ddot) E - ddot d^T -
// d ddot^T).
Eigen::Matrix3d parallel_axis_rate(double m, double mdot,
                                   const Eigen::Vector3d& d,
                                   const Eigen::Vector3d& ddot) {
  return mdot * (d.squaredNorm() * Eigen::Matrix3d::Identity() -
                 d * d.transpose()) +
         m * (2.0 * d.dot(ddot) * Eigen::Matrix3d::Identity() -
              ddot * d.transpose() - d * ddot.transpose());
}

void check_body(const BodyProps& body) {
  if (!std::isfinite(body.mass_kg) || body.mass_kg < 0.0 ||
      !body.cg_m.allFinite() || !body.inertia_kgm2.allFinite()) {
    throw std::domain_error(
        "massprops: body mass must be finite and non-negative with finite "
        "CG and inertia");
  }
}

}  // namespace

double tank_capacity_kg(const TankParams& tank) {
  // Fixed evaluation order rho * (pi R^2) * L, shared with the domain
  // check in tank_fill_height_m, so "at capacity" is one binary64 value
  // everywhere (ch:massprops domain rule).
  return tank.density_kgpm3 * cross_section_m2(tank) * tank.length_m;
}

double tank_fill_height_m(const TankParams& tank, double propellant_kg) {
  const double area = cross_section_m2(tank);
  if (!std::isfinite(propellant_kg) || propellant_kg < 0.0 ||
      propellant_kg > tank.density_kgpm3 * area * tank.length_m) {
    throw std::domain_error(
        "massprops: propellant mass must be finite, non-negative, and "
        "within the tank capacity");
  }
  // eq:massprops:fillheight -- h = m_p / (rho pi R^2).
  return propellant_kg / (tank.density_kgpm3 * area);
}

BodyProps tank_slug_props(const TankParams& tank, double propellant_kg) {
  const double h = tank_fill_height_m(tank, propellant_kg);
  BodyProps slug;
  // The mass is the argument verbatim, never rho pi R^2 h: the fill
  // height round trip would add two roundings and break the bit-exact
  // wet-mass identity (ch:massprops implementation note).
  slug.mass_kg = propellant_kg;
  // eq:massprops:slug -- CG on the tank axis at half the fill height.
  slug.cg_m = tank.aft_center_m + Eigen::Vector3d(0.5 * h, 0.0, 0.0);
  // eq:massprops:slug -- solid-cylinder inertia about the slug CG:
  // Ixx = m R^2 / 2 about the symmetry axis, Iyy = Izz =
  // m (3 R^2 + h^2) / 12 transverse; off-diagonals exactly zero.
  const double r2 = tank.radius_m * tank.radius_m;
  slug.inertia_kgm2(0, 0) = 0.5 * propellant_kg * r2;
  slug.inertia_kgm2(1, 1) = propellant_kg * (3.0 * r2 + h * h) / 12.0;
  slug.inertia_kgm2(2, 2) = slug.inertia_kgm2(1, 1);
  return slug;
}

BodyRates tank_slug_rates(const TankParams& tank, double propellant_kg,
                          double mdot_kgps) {
  const double h = tank_fill_height_m(tank, propellant_kg);
  if (!std::isfinite(mdot_kgps)) {
    throw std::domain_error("massprops: mdot must be finite");
  }
  // eq:massprops:slugrates -- chain rule through the fill height:
  // hdot = mdot / (rho pi R^2), cgdot = hdot/2 xhat,
  // Ixxdot = mdot R^2 / 2,
  // Iyydot = Izzdot = mdot (3 R^2 + h^2) / 12 + m h hdot / 6.
  const double hdot =
      mdot_kgps / (tank.density_kgpm3 * cross_section_m2(tank));
  BodyRates rates;
  rates.mdot_kgps = mdot_kgps;
  rates.cg_rate_mps = Eigen::Vector3d(0.5 * hdot, 0.0, 0.0);
  const double r2 = tank.radius_m * tank.radius_m;
  rates.inertia_rate_kgm2ps(0, 0) = 0.5 * mdot_kgps * r2;
  rates.inertia_rate_kgm2ps(1, 1) =
      mdot_kgps * (3.0 * r2 + h * h) / 12.0 +
      propellant_kg * h * hdot / 6.0;
  rates.inertia_rate_kgm2ps(2, 2) = rates.inertia_rate_kgm2ps(1, 1);
  return rates;
}

BodyProps compose(const std::vector<BodyProps>& bodies) {
  if (bodies.empty()) {
    throw std::domain_error("massprops: compose needs at least one body");
  }
  // eq:massprops:compose -- masses and moments summed in vector order
  // (the documented FR-10 order: fixed bodies, then tank slugs), so the
  // composite mass is the same-order binary64 sum of the inputs.
  double mass = 0.0;
  Eigen::Vector3d moment = Eigen::Vector3d::Zero();
  for (const BodyProps& body : bodies) {
    check_body(body);
    mass += body.mass_kg;
    moment += body.mass_kg * body.cg_m;
  }
  if (mass <= 0.0) {
    throw std::domain_error("massprops: composite mass must be positive");
  }
  BodyProps composite;
  composite.mass_kg = mass;
  composite.cg_m = moment / mass;
  for (const BodyProps& body : bodies) {
    composite.inertia_kgm2 +=
        body.inertia_kgm2 +
        parallel_axis(body.mass_kg, body.cg_m - composite.cg_m);
  }
  return composite;
}

BodyRates compose_rates(const std::vector<BodyProps>& bodies,
                        const std::vector<BodyRates>& rates) {
  if (bodies.size() != rates.size()) {
    throw std::domain_error(
        "massprops: compose_rates needs one rate entry per body");
  }
  const BodyProps composite = compose(bodies);
  // eq:massprops:composerates -- Mdot and the CG rate first (the inertia
  // rate needs ddot = cgdot_b - cgdot_composite).
  double mdot = 0.0;
  Eigen::Vector3d moment_rate = Eigen::Vector3d::Zero();
  for (std::size_t i = 0; i < bodies.size(); ++i) {
    if (!std::isfinite(rates[i].mdot_kgps) ||
        !rates[i].cg_rate_mps.allFinite() ||
        !rates[i].inertia_rate_kgm2ps.allFinite()) {
      throw std::domain_error("massprops: rates must be finite");
    }
    mdot += rates[i].mdot_kgps;
    moment_rate += rates[i].mdot_kgps * bodies[i].cg_m +
                   bodies[i].mass_kg * rates[i].cg_rate_mps;
  }
  BodyRates out;
  out.mdot_kgps = mdot;
  out.cg_rate_mps =
      (moment_rate - mdot * composite.cg_m) / composite.mass_kg;
  for (std::size_t i = 0; i < bodies.size(); ++i) {
    // Fixed bodies (zero own rates) still contribute here through
    // ddot = -cg_rate: the composite CG moves underneath them
    // (ch:massprops).
    const Eigen::Vector3d d = bodies[i].cg_m - composite.cg_m;
    const Eigen::Vector3d ddot = rates[i].cg_rate_mps - out.cg_rate_mps;
    out.inertia_rate_kgm2ps +=
        rates[i].inertia_rate_kgm2ps +
        parallel_axis_rate(bodies[i].mass_kg, rates[i].mdot_kgps, d, ddot);
  }
  return out;
}

BodyProps remove_body(const BodyProps& composite, const BodyProps& item) {
  check_body(composite);
  check_body(item);
  const double mass = composite.mass_kg - item.mass_kg;
  if (!(mass > 0.0)) {
    throw std::domain_error(
        "massprops: removal must leave a positive remainder mass");
  }
  // eq:massprops:remove -- inversion of the two-body composition: the
  // retained remainder's CG from the moment balance, then its inertia by
  // subtracting the departed item's own and parallel-axis terms and
  // re-referencing to the new CG.
  BodyProps out;
  out.mass_kg = mass;
  out.cg_m =
      (composite.mass_kg * composite.cg_m - item.mass_kg * item.cg_m) /
      mass;
  out.inertia_kgm2 =
      composite.inertia_kgm2 - item.inertia_kgm2 -
      parallel_axis(item.mass_kg, item.cg_m - composite.cg_m) -
      parallel_axis(mass, out.cg_m - composite.cg_m);
  return out;
}

}  // namespace models
}  // namespace star
