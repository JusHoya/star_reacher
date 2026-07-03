// Atmosphere model tests (FR-8; Phase 3 exit criteria 4, 8, 9): USSA76
// published-row reproduction, Harris-Priester node/off-node checks, and
// Mars node-exactness/continuity gates. Test IDs are cited by the
// math-library validation tables (ch:ussa76, ch:harrispriester,
// ch:marsatmosphere); do not rename them. Golden provenance and
// tolerances: tests/golden/atmosphere/manifest.toml.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

#include <Eigen/Dense>

#include "golden_io.hpp"
#include "star/constants.hpp"
#include "star/models/atmosphere_hp.hpp"
#include "star/models/atmosphere_mars.hpp"
#include "star/models/atmosphere_ussa76.hpp"
#include "vendor/doctest.h"

namespace {

using star_tests::GoldenCase;
using star_tests::load_golden_cases;
using star_tests::parse_hex_double;

std::string golden_path(const char* file) {
  return std::string(STAR_GOLDEN_DIR) + "/atmosphere/" + file;
}

// strtod handles both the decimal transcription strings and hex literals;
// a dedicated name keeps call sites readable for the decimal case.
double parse_double(const std::string& s) { return parse_hex_double(s); }

// Half a unit in the last place of a 4-significant-figure print: the
// "print precision" gate of Phase 3 exit criterion 4. The 1e-9 slack
// covers printed values that sit exactly on a rounding boundary.
double half_ulp4(double printed) {
  return 0.5 * std::pow(10.0, std::floor(std::log10(std::fabs(printed))) - 3.0);
}

void check_print_precision(double model, double printed, double* worst) {
  const double gate = half_ulp4(printed);
  const double err = std::fabs(model - printed);
  if (worst != nullptr && err / gate > *worst) {
    *worst = err / gate;
  }
  CAPTURE(model);
  CAPTURE(printed);
  CHECK(err <= gate * (1.0 + 1e-9));
}

}  // namespace

TEST_CASE("ATM-USSA76-ROWS") {
  // Phase 3 exit criterion 4: reproduce published USSA76 Table I rows to
  // print precision (4 significant figures). Below 86 km the analytic
  // model is exercised on temperature, pressure, and density; above 86 km
  // the committed-node interpolation is exercised on density. The
  // measured worst-case margins are CAPTUREd for the acceptance report.
  const auto cases = load_golden_cases(golden_path("ussa76_rows.toml"));
  CHECK(cases.size() == 35);
  double worst_analytic = 0.0;  // fraction of the print-precision gate
  double worst_table = 0.0;
  for (const auto& c : cases) {
    const double z = parse_double(c.scalar("z_m"));
    const double rho_ref = parse_double(c.scalar("rho_kgpm3"));
    CAPTURE(c.scalar("name"));
    if (c.scalar("region") == "analytic") {
      const auto s = star::models::ussa76_state(z);
      check_print_precision(s.temperature_K, parse_double(c.scalar("t_K")),
                            &worst_analytic);
      // The printed pressure is in millibar; 1 mb = 100 Pa exactly.
      check_print_precision(s.pressure_Pa,
                            parse_double(c.scalar("p_mb")) * 100.0,
                            &worst_analytic);
      check_print_precision(s.density_kgpm3, rho_ref, &worst_analytic);
      // The density-only entry point must agree with the full state.
      CHECK(star::models::ussa76_density(z) == s.density_kgpm3);
      // Sanity anchor for the speed of sound (eq:ussa76:speedofsound):
      // sea level must give the familiar ~340.29 m/s.
      if (z == 0.0) {
        CHECK(s.speed_of_sound_mps == doctest::Approx(340.294).epsilon(1e-4));
      }
    } else {
      check_print_precision(star::models::ussa76_density(z), rho_ref,
                            &worst_table);
    }
  }
  CAPTURE(worst_analytic);
  CAPTURE(worst_table);
  // The analytic margin is a regression canary well inside the gate: the
  // generation-time measurement was 0.199 of the gate (manifest).
  CHECK(worst_analytic < 0.5);
  // Above 86 km the gated rows are committed nodes, exact by design.
  CHECK(worst_table == 0.0);
}

TEST_CASE("ATM-USSA76-UPPER-NODES") {
  // The compiled-in 86-1000 km node grid must equal the committed
  // transcription check copy bit for bit, and evaluation at every node
  // must return the node density exactly (eq:ussa76:interp with zero
  // fractional offset).
  const auto cases = load_golden_cases(golden_path("ussa76_upper_nodes.toml"));
  std::size_t count = 0;
  const star::models::Ussa76Node* nodes =
      star::models::ussa76_upper_nodes(&count);
  REQUIRE(cases.size() == count);
  for (std::size_t i = 0; i < count; ++i) {
    CAPTURE(i);
    CHECK(parse_double(cases[i].scalar("z_m")) == nodes[i].z_m);
    CHECK(parse_double(cases[i].scalar("rho_kgpm3")) == nodes[i].rho_kgpm3);
    CHECK(star::models::ussa76_density(nodes[i].z_m) == nodes[i].rho_kgpm3);
  }
}

TEST_CASE("ATM-USSA76-CONTINUITY") {
  // Analytic/table seam at 86 km (ch:ussa76, domain section): the gap is
  // print rounding of the first node, measured 2.6e-5 relative.
  const double below =
      star::models::ussa76_density(std::nextafter(86000.0, 0.0));
  const double at = star::models::ussa76_density(86000.0);
  const double rel_gap = std::fabs(below - at) / at;
  CAPTURE(rel_gap);
  CHECK(rel_gap < 2e-4);
}

TEST_CASE("ATM-USSA76-DOMAIN") {
  // Out-of-domain behavior documented in ch:ussa76: hard errors, never
  // fabricated values (DX-2).
  CHECK_THROWS_AS(star::models::ussa76_density(-5000.1), std::domain_error);
  CHECK_THROWS_AS(star::models::ussa76_density(1.0000001e6),
                  std::domain_error);
  CHECK_THROWS_AS(star::models::ussa76_density(
                      std::numeric_limits<double>::quiet_NaN()),
                  std::domain_error);
  CHECK_THROWS_AS(star::models::ussa76_state(86000.0), std::domain_error);
  CHECK_THROWS_AS(star::models::ussa76_state(500000.0), std::domain_error);
  // Domain endpoints are valid.
  CHECK(star::models::ussa76_density(-5000.0) > 1.2250);
  CHECK(star::models::ussa76_density(1.0e6) == 3.561e-15);
  CHECK(star::models::ussa76_state(85999.9).density_kgpm3 > 0.0);
}

TEST_CASE("ATM-HP-NODES") {
  // Phase 3 exit criterion 8: with the bulge pinned to its minimum
  // (cos_psi = -1) and maximum (cos_psi = +1), the model at every node
  // altitude reproduces the published rho_min/rho_max to 4 significant
  // figures. The design makes at-node evaluation exact, so the check is
  // bit equality -- infinitely inside the criterion gate -- plus the
  // compiled-table-vs-golden transcription equality.
  const auto cases =
      load_golden_cases(golden_path("harris_priester_table.toml"));
  std::size_t count = 0;
  const star::models::HpNode* table = star::models::hp_table(&count);
  REQUIRE(cases.size() == count);
  REQUIRE(count == 50);
  for (std::size_t i = 0; i < count; ++i) {
    CAPTURE(i);
    const double alt = parse_double(cases[i].scalar("alt_m"));
    const double rho_min = parse_double(cases[i].scalar("rho_min_kgpm3"));
    const double rho_max = parse_double(cases[i].scalar("rho_max_kgpm3"));
    CHECK(alt == table[i].alt_m);
    CHECK(rho_min == table[i].rho_min_kgpm3);
    CHECK(rho_max == table[i].rho_max_kgpm3);
    // Pinned bulge minimum and maximum, default exponent (the exponent is
    // irrelevant at the pinned endpoints).
    CHECK(star::models::hp_density(
              alt, -1.0, star::models::HP_COS_EXPONENT_DEFAULT) == rho_min);
    CHECK(star::models::hp_density(
              alt, 1.0, star::models::HP_COS_EXPONENT_DEFAULT) == rho_max);
  }
}

TEST_CASE("ATM-HP-OFFNODE") {
  // Off-node densities vs the independent 50-digit mpmath reference
  // (manifest tolerance 1e-13 relative), plus the bulge-apex construction
  // (eq:hp:apex) and geodetic-altitude closure (eq:hp:geodetic).
  const auto cases = load_golden_cases(golden_path("hp_offnode.toml"));
  CHECK(cases.size() == 8);
  double worst_rel = 0.0;
  for (const auto& c : cases) {
    CAPTURE(c.scalar("name"));
    const double rho = star::models::hp_density(
        parse_double(c.scalar("alt_m")), parse_double(c.scalar("cos_psi")),
        parse_double(c.scalar("n")));
    const double ref = parse_hex_double(c.scalar("rho_kgpm3"));
    const double rel = std::fabs(rho - ref) / ref;
    if (rel > worst_rel) {
      worst_rel = rel;
    }
    CHECK(rel <= 1e-13);
  }
  CAPTURE(worst_rel);

  // Bulge apex: RA advances by the 30 deg lag, declination preserved.
  const double alpha = 10.0 * star::constants::TWO_PI / 360.0;
  const double delta = 20.0 * star::constants::TWO_PI / 360.0;
  const Eigen::Vector3d sun(std::cos(delta) * std::cos(alpha),
                            std::cos(delta) * std::sin(alpha),
                            std::sin(delta));
  const Eigen::Vector3d apex = star::models::hp_bulge_apex(sun);
  const double alpha_b = alpha + 30.0 * star::constants::TWO_PI / 360.0;
  const Eigen::Vector3d expect(std::cos(delta) * std::cos(alpha_b),
                               std::cos(delta) * std::sin(alpha_b),
                               std::sin(delta));
  CHECK((apex - expect).norm() < 1e-15);

  // Geodetic-altitude closure: synthesize Earth-fixed points from exact
  // geodetic coordinates on WGS84 and require recovery to < 1e-6 m
  // (Bowring's two fixed passes are sub-millimetre in this domain).
  const double a = star::constants::WGS84_A_M;
  const double inv_f = star::constants::WGS84_INV_F;
  const double f = 1.0 / inv_f;
  const double e2 = f * (2.0 - f);
  const double lats_deg[] = {0.0, 45.0, 89.0, -60.0};
  const double alts_m[] = {150000.0, 420000.0, 800000.0, 250000.0};
  for (int i = 0; i < 4; ++i) {
    const double phi = lats_deg[i] * star::constants::TWO_PI / 360.0;
    const double lam = 0.4 * (i + 1);
    const double h = alts_m[i];
    const double sp = std::sin(phi);
    const double n_rad = a / std::sqrt(1.0 - e2 * sp * sp);
    const Eigen::Vector3d r((n_rad + h) * std::cos(phi) * std::cos(lam),
                            (n_rad + h) * std::cos(phi) * std::sin(lam),
                            (n_rad * (1.0 - e2) + h) * sp);
    const double h_rec = star::models::geodetic_altitude(r, a, inv_f);
    CAPTURE(lats_deg[i]);
    CHECK(std::fabs(h_rec - h) < 1e-6);
  }
}

TEST_CASE("ATM-HP-DOMAIN") {
  // Out-of-domain behavior documented in ch:harrispriester: throw below
  // the table, exact zero above it (Orekit-compatible), reject invalid
  // bulge arguments.
  const double n = star::models::HP_COS_EXPONENT_DEFAULT;
  CHECK_THROWS_AS(star::models::hp_density(99999.9, 0.0, n),
                  std::domain_error);
  CHECK_THROWS_AS(star::models::hp_density(300000.0, 1.5, n),
                  std::domain_error);
  CHECK_THROWS_AS(star::models::hp_density(300000.0, 0.0, 1.9),
                  std::domain_error);
  CHECK_THROWS_AS(star::models::hp_density(300000.0, 0.0, 6.1),
                  std::domain_error);
  CHECK_THROWS_AS(star::models::hp_density(
                      std::numeric_limits<double>::quiet_NaN(), 0.0, n),
                  std::domain_error);
  CHECK(star::models::hp_density(1000000.1, 0.5, n) == 0.0);
  CHECK(star::models::hp_density(5.0e6, 0.5, n) == 0.0);
  // Exponent bounds are inclusive.
  CHECK(star::models::hp_density(300000.0, 0.3, 2.0) > 0.0);
  CHECK(star::models::hp_density(300000.0, 0.3, 6.0) > 0.0);
}

TEST_CASE("ATM-MARS-NODES") {
  // Phase 3 exit criterion 9(a): the model returns the committed node
  // values with bit-exact double equality; off-node evaluations match the
  // independent mpmath reference to 1e-13 relative. (Model provenance is
  // flagged confidence: low per PRD A-3; see the golden manifest.)
  const auto cases = load_golden_cases(golden_path("mars_nodes.toml"));
  std::size_t count = 0;
  const star::models::MarsNode* nodes = star::models::mars_nodes(&count);
  REQUIRE(cases.size() == count);
  REQUIRE(count == 21);
  for (std::size_t i = 0; i < count; ++i) {
    CAPTURE(i);
    const double z = parse_double(cases[i].scalar("z_m"));
    const double rho = parse_hex_double(cases[i].scalar("rho_kgpm3"));
    CHECK(z == nodes[i].z_m);
    CHECK(rho == nodes[i].rho_kgpm3);
    CHECK(star::models::mars_density(z) == rho);  // bit-exact by design
  }
  const auto off = load_golden_cases(golden_path("mars_offnode.toml"));
  CHECK(off.size() == 7);
  double worst_rel = 0.0;
  for (const auto& c : off) {
    CAPTURE(c.scalar("name"));
    const double rho = star::models::mars_density(
        parse_double(c.scalar("z_m")));
    const double ref = parse_hex_double(c.scalar("rho_kgpm3"));
    const double rel = std::fabs(rho - ref) / ref;
    if (rel > worst_rel) {
      worst_rel = rel;
    }
    CHECK(rel <= 1e-13);
  }
  CAPTURE(worst_rel);
  CHECK(worst_rel <= 1e-13);  // prints the observed margin under -s
}

TEST_CASE("ATM-MARS-CONT") {
  // Phase 3 exit criterion 9(b): density is continuous at every segment
  // boundary -- approaching from both sides with the smallest
  // representable increments yields a relative jump < 1e-12. The
  // committed 100 km top node (boundary to the documented extrapolation)
  // is included.
  std::size_t count = 0;
  const star::models::MarsNode* nodes = star::models::mars_nodes(&count);
  double worst_jump = 0.0;
  for (std::size_t i = 1; i < count; ++i) {
    const double zb = nodes[i].z_m;
    const double at = star::models::mars_density(zb);
    const double below = star::models::mars_density(
        std::nextafter(zb, -std::numeric_limits<double>::infinity()));
    const double above = star::models::mars_density(
        std::nextafter(zb, std::numeric_limits<double>::infinity()));
    const double jump_below = std::fabs(below - at) / at;
    const double jump_above = std::fabs(above - at) / at;
    const double jump = std::max(jump_below, jump_above);
    if (jump > worst_jump) {
      worst_jump = jump;
    }
    CAPTURE(zb);
    CHECK(jump < 1e-12);
  }
  CAPTURE(worst_jump);
  CHECK(worst_jump < 1e-12);  // prints the observed margin under -s
}

TEST_CASE("ATM-MARS-DOMAIN") {
  // Out-of-domain behavior documented in ch:marsatmosphere: documented
  // extrapolations at both ends, hard error below the -8 km guard.
  CHECK_THROWS_AS(star::models::mars_density(-8000.1), std::domain_error);
  CHECK_THROWS_AS(star::models::mars_density(
                      std::numeric_limits<double>::quiet_NaN()),
                  std::domain_error);
  // Downward extrapolation: denser than the surface node, finite.
  const double rho_hellas = star::models::mars_density(-8000.0);
  CHECK(rho_hellas > star::models::mars_density(0.0));
  CHECK(std::isfinite(rho_hellas));
  // Upward extrapolation: monotone vanishing continuation.
  const double rho_100 = star::models::mars_density(100000.0);
  const double rho_150 = star::models::mars_density(150000.0);
  const double rho_300 = star::models::mars_density(300000.0);
  CHECK(rho_150 < rho_100);
  CHECK(rho_300 < rho_150);
  CHECK(rho_300 > 0.0);
}
