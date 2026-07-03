// Cannonball drag tests (FR-9): golden acceleration vectors against the
// independent mpmath reference (FR-22 layer 1) and formulation invariants
// (layer 2). Test IDs are cited by the math-library validation table
// (ch:drag); do not rename them. Golden provenance and tolerances:
// tests/golden/atmosphere/manifest.toml.
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include <Eigen/Dense>

#include "golden_io.hpp"
#include "star/models/drag.hpp"
#include "vendor/doctest.h"

namespace {

using star_tests::load_golden_cases;
using star_tests::parse_hex_double;

Eigen::Vector3d parse_vec(const std::vector<std::string>& items) {
  REQUIRE(items.size() == 3);
  return Eigen::Vector3d(parse_hex_double(items[0]),
                         parse_hex_double(items[1]),
                         parse_hex_double(items[2]));
}

}  // namespace

TEST_CASE("DRAG-CANNONBALL-GOLDEN") {
  // Committed states vs the 50-digit mpmath reference: component-wise
  // relative agreement to 1e-14 (manifest); exactly-zero components and
  // the zero-velocity case must be exactly zero (no normalization
  // singularity exists in eq:drag:accel).
  const auto cases = load_golden_cases(std::string(STAR_GOLDEN_DIR) +
                                       "/atmosphere/drag_vectors.toml");
  CHECK(cases.size() == 5);
  double worst_rel = 0.0;
  for (const auto& c : cases) {
    CAPTURE(c.scalar("name"));
    const double rho = parse_hex_double(c.scalar("rho_kgpm3"));
    const double cdam = parse_hex_double(c.scalar("cd_a_over_m_m2pkg"));
    const Eigen::Vector3d v_rel = parse_vec(c.array("v_rel_mps"));
    const Eigen::Vector3d a_ref = parse_vec(c.array("a_mps2"));
    const Eigen::Vector3d a = star::models::drag_accel(rho, cdam, v_rel);
    for (int i = 0; i < 3; ++i) {
      CAPTURE(i);
      if (a_ref[i] == 0.0) {
        CHECK(a[i] == 0.0);
      } else {
        const double rel = std::fabs(a[i] - a_ref[i]) / std::fabs(a_ref[i]);
        if (rel > worst_rel) {
          worst_rel = rel;
        }
        CHECK(rel <= 1e-14);
      }
    }
  }
  CAPTURE(worst_rel);
  CHECK(worst_rel <= 1e-14);  // prints the observed margin under -s
}

TEST_CASE("DRAG-PROPERTIES") {
  // FR-22 layer-2 invariants of eq:drag:accel and the domain guards of
  // ch:drag.
  const double rho = 5.215e-13;
  const double cdam = 0.0044;  // Cd = 2.2 (documented default), A/m = 1/500
  const Eigen::Vector3d v(-7100.0, 1200.0, -350.0);
  const Eigen::Vector3d a = star::models::drag_accel(rho, cdam, v);

  // Anti-parallel to the relative velocity: negative projection and a
  // vanishing normalized cross product (exact parallelism up to the
  // rounding of three scalar multiplies).
  CHECK(a.dot(v) < 0.0);
  CHECK(a.cross(v).norm() <= 1e-15 * a.norm() * v.norm());

  // Magnitude equals the dynamic-pressure form 1/2 rho (CdA/m) |v|^2.
  const double mag_expected = 0.5 * rho * cdam * v.squaredNorm();
  CHECK(std::fabs(a.norm() - mag_expected) <= 1e-14 * mag_expected);

  // Linear in density and ballistic parameter; doubling either input
  // scales the result by exactly 2 (multiplication by 2 is exact in
  // binary64 and the evaluation order is fixed).
  CHECK(star::models::drag_accel(2.0 * rho, cdam, v) == 2.0 * a);
  CHECK(star::models::drag_accel(rho, 2.0 * cdam, v) == 2.0 * a);

  // Quadratic in speed: doubling v_rel scales the acceleration by exactly
  // 4 (|2v| = 2|v| and the prefactor chain are all power-of-two exact).
  CHECK(star::models::drag_accel(rho, cdam, 2.0 * v) == 4.0 * a);

  // Vacuum and zero-velocity limits are exact zeros.
  CHECK(star::models::drag_accel(0.0, cdam, v) == Eigen::Vector3d::Zero());
  CHECK(star::models::drag_accel(rho, cdam, Eigen::Vector3d::Zero()) ==
        Eigen::Vector3d::Zero());

  // Domain guards: negative or non-finite inputs are hard errors, never
  // silently absorbed (DX-2).
  CHECK_THROWS_AS(star::models::drag_accel(-1.0, cdam, v),
                  std::domain_error);
  CHECK_THROWS_AS(star::models::drag_accel(rho, -1.0, v),
                  std::domain_error);
  CHECK_THROWS_AS(
      star::models::drag_accel(
          std::numeric_limits<double>::quiet_NaN(), cdam, v),
      std::domain_error);
  CHECK_THROWS_AS(
      star::models::drag_accel(
          rho, cdam,
          Eigen::Vector3d(std::numeric_limits<double>::infinity(), 0.0,
                          0.0)),
      std::domain_error);
}
