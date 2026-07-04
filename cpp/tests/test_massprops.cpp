// Mass-properties tests (FR-10, Phase 4 exit criterion 2): golden slug and
// composite vectors against the independent 60-digit mpmath references
// (FR-22 layer 1), the bit-exact wet-mass identity, CG continuity across a
// depletion sweep, the closed-form jettison delta against an independent
// recomposition, and the analytic-rate/central-difference cross-check
// (layer 2). Test IDs are cited by the math-library validation table
// (ch:massprops); do not rename them. Golden provenance and tolerances:
// tests/golden/massprop/manifest.toml.
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "golden_io.hpp"
#include "star/models/massprops.hpp"
#include "vendor/doctest.h"

namespace {

using star::models::BodyProps;
using star::models::BodyRates;
using star::models::TankParams;
using star_tests::load_golden_cases;
using star_tests::parse_hex_double;

const std::string kGoldenDir = STAR_GOLDEN_DIR;

Eigen::Vector3d parse_vec3(const star_tests::GoldenCase& c,
                           const std::string& key) {
  const auto& a = c.array(key);
  REQUIRE(a.size() == 3);
  return Eigen::Vector3d(parse_hex_double(a[0]), parse_hex_double(a[1]),
                         parse_hex_double(a[2]));
}

Eigen::Matrix3d parse_mat3(const star_tests::GoldenCase& c,
                           const std::string& key) {
  const auto& a = c.array(key);
  REQUIRE(a.size() == 9);
  Eigen::Matrix3d m;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      m(i, j) = parse_hex_double(a[3 * i + j]);
    }
  }
  return m;
}

// Norm-relative comparison with the exact-zero contract: a zero reference
// requires an exactly zero value (the model writes literal zeros there),
// a nonzero reference is gated at tol. Returns the measured error so the
// caller can track the worst case.
double check_vec(const Eigen::Vector3d& value, const Eigen::Vector3d& ref,
                 double tol) {
  if (ref.norm() == 0.0) {
    CHECK(value.norm() == 0.0);
    return 0.0;
  }
  const double err = (value - ref).norm() / ref.norm();
  CHECK(err <= tol);
  return err;
}

double check_mat(const Eigen::Matrix3d& value, const Eigen::Matrix3d& ref,
                 double tol) {
  if (ref.norm() == 0.0) {
    CHECK(value.norm() == 0.0);
    return 0.0;
  }
  const double err = (value - ref).norm() / ref.norm();
  CHECK(err <= tol);
  return err;
}

double check_scalar(double value, double ref, double tol) {
  if (ref == 0.0) {
    CHECK(value == 0.0);
    return 0.0;
  }
  const double err = std::fabs(value - ref) / std::fabs(ref);
  CHECK(err <= tol);
  return err;
}

// prefix is "" for the slug file's unprefixed keys, "tank0_"/"tank1_"
// for the composite file.
TankParams parse_tank(const star_tests::GoldenCase& c,
                      const std::string& prefix) {
  TankParams tank;
  tank.radius_m = parse_hex_double(c.scalar(prefix + "radius_m"));
  tank.length_m = parse_hex_double(c.scalar(prefix + "length_m"));
  tank.aft_center_m = parse_vec3(c, prefix + "aft_center_m");
  tank.density_kgpm3 = parse_hex_double(c.scalar(prefix + "density_kgpm3"));
  return tank;
}

// Reference stack shared by the identity/continuity/jettison/rate tests:
// two dry bodies and two +X tanks, values chosen physically admissible
// (SPD inertia, fills inside capacity) and unrelated to the golden cases.
struct Stack {
  std::vector<BodyProps> fixed;
  std::vector<TankParams> tanks;
};

Stack reference_stack() {
  Stack s;
  BodyProps core;
  core.mass_kg = 950.0;
  core.cg_m = Eigen::Vector3d(2.4, 0.05, -0.03);
  core.inertia_kgm2 << 700.0, 8.0, -4.0, 8.0, 3100.0, 6.0, -4.0, 6.0,
      3150.0;
  BodyProps payload;
  payload.mass_kg = 120.0;
  payload.cg_m = Eigen::Vector3d(5.8, -0.02, 0.04);
  payload.inertia_kgm2 << 30.0, 0.5, 0.0, 0.5, 42.0, -0.8, 0.0, -0.8, 45.0;
  s.fixed = {core, payload};
  TankParams fuel;
  fuel.radius_m = 0.8;
  fuel.length_m = 2.8;
  fuel.aft_center_m = Eigen::Vector3d(0.9, 0.0, 0.0);
  fuel.density_kgpm3 = 810.0;
  fuel.initial_mass_kg = 3200.0;
  TankParams ox;
  ox.radius_m = 0.8;
  ox.length_m = 1.9;
  ox.aft_center_m = Eigen::Vector3d(3.9, 0.0, 0.0);
  ox.density_kgpm3 = 1140.0;
  ox.initial_mass_kg = 3000.0;
  s.tanks = {fuel, ox};
  return s;
}

// Bodies vector in the documented FR-10 order (fixed bodies, then one
// slug per tank) at the given propellant loads.
std::vector<BodyProps> stack_bodies(const Stack& s,
                                    const std::vector<double>& props) {
  std::vector<BodyProps> bodies = s.fixed;
  for (std::size_t i = 0; i < s.tanks.size(); ++i) {
    bodies.push_back(star::models::tank_slug_props(s.tanks[i], props[i]));
  }
  return bodies;
}

}  // namespace

TEST_CASE("MASSPROP-SLUG-GOLDEN") {
  // Golden gate (FR-22 layer 1): draining-cylinder slug closed forms
  // (eq:massprops:slug, eq:massprops:slugrates) against the independent
  // 60-digit mpmath references at 1e-12 relative (exit criterion 2; the
  // C++ path accumulates <10 IEEE roundings, floor ~1e-15, manifest
  // derivation). Exactly-zero references (empty tank) must be exactly
  // zero: the model multiplies the propellant mass into every term.
  const auto cases =
      load_golden_cases(kGoldenDir + "/massprop/slug.toml");
  REQUIRE(cases.size() == 5);
  double worst = 0.0;
  for (const auto& c : cases) {
    CAPTURE(c.scalar("name"));
    const TankParams t = parse_tank(c, "");
    const double m_p = parse_hex_double(c.scalar("propellant_kg"));
    const double mdot = parse_hex_double(c.scalar("mdot_kgps"));

    const double h = star::models::tank_fill_height_m(t, m_p);
    worst = std::max(
        worst,
        check_scalar(h, parse_hex_double(c.scalar("fill_height_m")), 1e-12));

    const BodyProps slug = star::models::tank_slug_props(t, m_p);
    CHECK(slug.mass_kg == m_p);  // verbatim pass-through, bit-exact
    worst = std::max(worst, check_vec(slug.cg_m, parse_vec3(c, "cg_m"),
                                      1e-12));
    worst = std::max(worst, check_mat(slug.inertia_kgm2,
                                      parse_mat3(c, "inertia_kgm2"), 1e-12));

    const BodyRates rates = star::models::tank_slug_rates(t, m_p, mdot);
    CHECK(rates.mdot_kgps == mdot);
    worst = std::max(worst, check_vec(rates.cg_rate_mps,
                                      parse_vec3(c, "cg_rate_mps"), 1e-12));
    worst = std::max(
        worst, check_mat(rates.inertia_rate_kgm2ps,
                         parse_mat3(c, "inertia_rate_kgm2ps"), 1e-12));
  }
  CAPTURE(worst);
  CHECK(worst <= 1e-12);  // prints the observed margin under -s
}

TEST_CASE("MASSPROP-COMPOSITE-GOLDEN") {
  // Golden gate (FR-22 layer 1): parallel-axis composition, composite
  // rates, and closed-form removal (eq:massprops:compose,
  // eq:massprops:composerates, eq:massprops:remove) against the 60-digit
  // mpmath references. Masses compare bit-exact (round-decimal inputs
  // with exact binary64 sums, manifest derivation); everything else at
  // norm-relative 1e-12, the exit-criterion-2 gate.
  const auto cases =
      load_golden_cases(kGoldenDir + "/massprop/composite.toml");
  REQUIRE(cases.size() == 2);
  double worst = 0.0;
  for (const auto& c : cases) {
    CAPTURE(c.scalar("name"));
    std::vector<BodyProps> bodies;
    std::vector<BodyRates> rates;
    for (int k = 0; k < 2; ++k) {
      const std::string p = "body" + std::to_string(k);
      BodyProps body;
      body.mass_kg = parse_hex_double(c.scalar(p + "_mass_kg"));
      body.cg_m = parse_vec3(c, p + "_cg_m");
      body.inertia_kgm2 = parse_mat3(c, p + "_inertia_kgm2");
      bodies.push_back(body);
      rates.emplace_back();  // fixed body: zero rates
    }
    for (int k = 0; k < 2; ++k) {
      const std::string p = "tank" + std::to_string(k) + "_";
      const TankParams tank = parse_tank(c, p);
      const double m_p = parse_hex_double(c.scalar(p + "propellant_kg"));
      const double mdot = parse_hex_double(c.scalar(p + "mdot_kgps"));
      bodies.push_back(star::models::tank_slug_props(tank, m_p));
      rates.push_back(star::models::tank_slug_rates(tank, m_p, mdot));
    }

    const BodyProps composite = star::models::compose(bodies);
    CHECK(composite.mass_kg ==
          parse_hex_double(c.scalar("mass_kg")));  // bit-exact
    worst = std::max(worst,
                     check_vec(composite.cg_m, parse_vec3(c, "cg_m"), 1e-12));
    worst = std::max(worst, check_mat(composite.inertia_kgm2,
                                      parse_mat3(c, "inertia_kgm2"), 1e-12));

    const BodyRates crates = star::models::compose_rates(bodies, rates);
    worst = std::max(
        worst, check_scalar(crates.mdot_kgps,
                            parse_hex_double(c.scalar("mdot_kgps")), 1e-12));
    worst = std::max(worst, check_vec(crates.cg_rate_mps,
                                      parse_vec3(c, "cg_rate_mps"), 1e-12));
    worst = std::max(
        worst, check_mat(crates.inertia_rate_kgm2ps,
                         parse_mat3(c, "inertia_rate_kgm2ps"), 1e-12));

    REQUIRE(c.scalar("jettison_body_index") == "1");
    const BodyProps after = star::models::remove_body(composite, bodies[1]);
    CHECK(after.mass_kg ==
          parse_hex_double(c.scalar("post_jettison_mass_kg")));
    worst = std::max(worst, check_vec(after.cg_m,
                                      parse_vec3(c, "post_jettison_cg_m"),
                                      1e-12));
    worst = std::max(
        worst, check_mat(after.inertia_kgm2,
                         parse_mat3(c, "post_jettison_inertia_kgm2"), 1e-12));
  }
  CAPTURE(worst);
  CHECK(worst <= 1e-12);
}

TEST_CASE("MASSPROP-WET-MASS-IDENTITY") {
  // Exit criterion 2: wet mass at t0 equals dry + Sigma propellant
  // EXACTLY. The model sums masses in vector order and passes slug
  // masses through verbatim, so the identity is bit-level for the
  // same-order binary64 sum - at t0 and at every point of a depletion
  // sweep.
  const Stack s = reference_stack();
  const double m0_fuel = s.tanks[0].initial_mass_kg;
  const double m0_ox = s.tanks[1].initial_mass_kg;
  for (int k = 0; k <= 100; ++k) {
    const double f = static_cast<double>(k) / 100.0;
    const double m_fuel = m0_fuel * (1.0 - f);
    const double m_ox = m0_ox * (1.0 - f);
    const BodyProps composite =
        star::models::compose(stack_bodies(s, {m_fuel, m_ox}));
    // Same-order reference sum: fixed bodies, then tank slugs.
    const double expected =
        s.fixed[0].mass_kg + s.fixed[1].mass_kg + m_fuel + m_ox;
    CHECK(composite.mass_kg == expected);
  }
}

TEST_CASE("MASSPROP-CG-CONTINUITY") {
  // Exit criterion 2: CG continuous across depletion. Drain the fuel
  // tank from full to empty in 2000 samples and bound every per-sample
  // CG step by the analytic rate (eq:massprops:composerates) at the
  // interval endpoints: |dcg| <= max(|rate|) * dm with a 2x margin for
  // the rate's variation inside the interval. A discontinuity (branch
  // error, capacity clamp, sign flip) fails at the full jump size.
  const Stack s = reference_stack();
  const double m0 = s.tanks[0].initial_mass_kg;
  const int n = 2000;
  const double dm = m0 / static_cast<double>(n);
  Eigen::Vector3d prev_cg;
  double prev_rate = 0.0;
  double max_jump = 0.0;
  double max_bound_ratio = 0.0;
  for (int k = 0; k <= n; ++k) {
    const double m_fuel = m0 - dm * static_cast<double>(k);
    const std::vector<BodyProps> bodies =
        stack_bodies(s, {std::max(0.0, m_fuel), s.tanks[1].initial_mass_kg});
    std::vector<BodyRates> rates(bodies.size());
    // Unit drain rate: |cg_rate| then equals |dcg/dm_p| in m/kg.
    rates[2] = star::models::tank_slug_rates(
        s.tanks[0], std::max(0.0, m_fuel), -1.0);
    const BodyProps composite = star::models::compose(bodies);
    const BodyRates crates = star::models::compose_rates(bodies, rates);
    const double rate = crates.cg_rate_mps.norm();
    if (k > 0) {
      const double jump = (composite.cg_m - prev_cg).norm();
      const double bound = 2.0 * std::max(rate, prev_rate) * dm;
      max_jump = std::max(max_jump, jump);
      if (bound > 0.0) {
        max_bound_ratio = std::max(max_bound_ratio, jump / bound);
      }
      CHECK(jump <= bound);
    }
    prev_cg = composite.cg_m;
    prev_rate = rate;
  }
  CAPTURE(max_jump);
  CAPTURE(max_bound_ratio);
  // The trajectory must actually move (a frozen CG would pass the jump
  // gate vacuously): full-to-empty fuel drain shifts the composite CG by
  // centimeters at least for this stack.
  CHECK(max_jump > 0.0);
}

TEST_CASE("MASSPROP-JETTISON-CLOSED-FORM") {
  // Exit criterion 2: the staging CG jump matches the closed-form mass
  // properties to 1e-12 relative. Reference by DIFFERENT arithmetic:
  // remove_body (eq:massprops:remove) vs recomposing the retained bodies
  // directly with eq:massprops:compose - the two paths share no
  // intermediate values beyond the inputs.
  const Stack s = reference_stack();
  const std::vector<BodyProps> bodies = stack_bodies(s, {2600.0, 2100.0});
  const BodyProps composite = star::models::compose(bodies);

  const BodyProps after =
      star::models::remove_body(composite, bodies[1]);  // drop the payload
  std::vector<BodyProps> retained = {bodies[0], bodies[2], bodies[3]};
  const BodyProps recomposed = star::models::compose(retained);

  CHECK(after.mass_kg == recomposed.mass_kg);  // both exact binary64 sums
  const double cg_err =
      (after.cg_m - recomposed.cg_m).norm() / recomposed.cg_m.norm();
  const double inertia_err = (after.inertia_kgm2 - recomposed.inertia_kgm2)
                                 .norm() /
                             recomposed.inertia_kgm2.norm();
  CAPTURE(cg_err);
  CAPTURE(inertia_err);
  CHECK(cg_err <= 1e-12);
  CHECK(inertia_err <= 1e-12);

  // The jump itself must be nonzero (dropping an off-CG payload moves
  // the composite CG), so the 1e-12 agreement is not vacuous.
  CHECK((after.cg_m - composite.cg_m).norm() > 1e-3);
}

TEST_CASE("MASSPROP-IDOT-CONSISTENCY") {
  // FR-22 layer 2: the analytic composite rates
  // (eq:massprops:composerates) against central differences of the
  // composite properties - an independent reference that exercises the
  // full chain (slug rates, CG rate, parallel-axis rate) without reusing
  // its algebra. Central differencing converges as O(dt^2), so the error
  // must fall by ~100x per decade of dt until the conditioning floor;
  // the smallest step is gated at 1e-6 relative, far above the floor
  // (~1e-11 for these magnitudes) and far below any term error (O(1)).
  const Stack s = reference_stack();
  const double m_fuel = 2600.0;
  const double m_ox = 2100.0;
  const double mdot_fuel = -85.0;
  const double mdot_ox = -40.0;

  const std::vector<BodyProps> bodies = stack_bodies(s, {m_fuel, m_ox});
  std::vector<BodyRates> rates(bodies.size());
  rates[2] = star::models::tank_slug_rates(s.tanks[0], m_fuel, mdot_fuel);
  rates[3] = star::models::tank_slug_rates(s.tanks[1], m_ox, mdot_ox);
  const BodyRates analytic = star::models::compose_rates(bodies, rates);

  const auto composite_at = [&](double dt) {
    return star::models::compose(stack_bodies(
        s, {m_fuel + mdot_fuel * dt, m_ox + mdot_ox * dt}));
  };

  double prev_err = -1.0;
  double final_err = -1.0;
  for (const double dt : {1.0, 0.1, 0.01}) {
    const BodyProps plus = composite_at(dt);
    const BodyProps minus = composite_at(-dt);
    const double mdot_fd = (plus.mass_kg - minus.mass_kg) / (2.0 * dt);
    const Eigen::Vector3d cg_fd = (plus.cg_m - minus.cg_m) / (2.0 * dt);
    const Eigen::Matrix3d idot_fd =
        (plus.inertia_kgm2 - minus.inertia_kgm2) / (2.0 * dt);
    const double err_mdot =
        std::fabs(mdot_fd - analytic.mdot_kgps) /
        std::fabs(analytic.mdot_kgps);
    const double err_cg = (cg_fd - analytic.cg_rate_mps).norm() /
                          analytic.cg_rate_mps.norm();
    const double err_idot = (idot_fd - analytic.inertia_rate_kgm2ps).norm() /
                            analytic.inertia_rate_kgm2ps.norm();
    CAPTURE(dt);
    CAPTURE(err_mdot);
    CAPTURE(err_cg);
    CAPTURE(err_idot);
    // Total mass is linear in t, so its central difference is exact up
    // to rounding at every step size.
    CHECK(err_mdot <= 1e-12);
    const double err = std::max(err_cg, err_idot);
    if (prev_err >= 0.0) {
      CHECK(err < prev_err);  // O(dt^2) convergence, no floor yet
    }
    prev_err = err;
    final_err = err;
  }
  CAPTURE(final_err);
  CHECK(final_err <= 1e-6);
}

TEST_CASE("MASSPROP-DOMAIN") {
  // Out-of-domain behavior per ch:massprops: std::domain_error, matching
  // the chapter's domain section exactly; in-domain boundaries evaluate.
  TankParams tank;
  tank.radius_m = 0.5;
  tank.length_m = 2.0;
  tank.density_kgpm3 = 1000.0;
  const double capacity = star::models::tank_capacity_kg(tank);
  CHECK(capacity > 0.0);

  CHECK_THROWS_AS(star::models::tank_fill_height_m(tank, -1.0),
                  std::domain_error);
  CHECK_THROWS_AS(star::models::tank_fill_height_m(tank, capacity * 1.01),
                  std::domain_error);
  CHECK_THROWS_AS(
      star::models::tank_fill_height_m(
          tank, std::numeric_limits<double>::quiet_NaN()),
      std::domain_error);
  // Exactly-at-capacity and exactly-empty are in-domain boundaries.
  CHECK(star::models::tank_fill_height_m(tank, capacity) ==
        doctest::Approx(tank.length_m).epsilon(1e-12));
  CHECK(star::models::tank_fill_height_m(tank, 0.0) == 0.0);

  TankParams bad = tank;
  bad.radius_m = 0.0;
  CHECK_THROWS_AS(star::models::tank_capacity_kg(bad), std::domain_error);
  bad = tank;
  bad.density_kgpm3 = -1.0;
  CHECK_THROWS_AS(star::models::tank_slug_props(bad, 1.0),
                  std::domain_error);

  CHECK_THROWS_AS(star::models::compose({}), std::domain_error);
  BodyProps negative;
  negative.mass_kg = -1.0;
  CHECK_THROWS_AS(star::models::compose({negative}), std::domain_error);
  BodyProps zero;  // zero-mass-only composite: total mass not positive
  CHECK_THROWS_AS(star::models::compose({zero}), std::domain_error);

  BodyProps a;
  a.mass_kg = 10.0;
  a.cg_m = Eigen::Vector3d(1.0, 0.0, 0.0);
  CHECK_THROWS_AS(star::models::remove_body(a, a), std::domain_error);

  CHECK_THROWS_AS(
      star::models::compose_rates({a}, {BodyRates{}, BodyRates{}}),
      std::domain_error);
}
