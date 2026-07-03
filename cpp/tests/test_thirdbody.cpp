// Third-body model tests (FR-6, FR-22 layer 1, Phase 3 exit criterion 7):
// the Battin f(q) implementation against extended-precision references at
// the 10 committed golden states, the near-alignment digit-loss
// demonstration, and limit properties. Test IDs are cited by the
// math-library validation table (ch:thirdbody); do not rename them.
//
// The golden references in tests/golden/thirdbody/states.toml are the
// naive two-vector difference (eq:thirdbody:naive) evaluated with mpmath
// at 60 significant decimal digits from the exact committed binary64
// inputs (provenance: tests/golden/thirdbody/manifest.toml). The naive and
// f(q) forms are algebraically identical, so the comparison isolates
// floating-point behavior - exactly what criterion 7 gates.
#include <cmath>
#include <string>

#include <Eigen/Dense>

#include "golden_io.hpp"
#include "star/models/thirdbody.hpp"
#include "vendor/doctest.h"

namespace {

const std::string kGoldenDir = STAR_GOLDEN_DIR;
const std::string kStates = kGoldenDir + "/thirdbody/states.toml";

Eigen::Vector3d parse_vec3(const star_tests::GoldenCase& c,
                           const std::string& key) {
  const auto& a = c.array(key);
  REQUIRE(a.size() == 3);
  return Eigen::Vector3d(star_tests::parse_hex_double(a[0]),
                         star_tests::parse_hex_double(a[1]),
                         star_tests::parse_hex_double(a[2]));
}

// Naive direct-minus-indirect difference (eq:thirdbody:naive) in double
// precision, in an EXPLICIT scalar operation order that mirrors
// naive_double() in tests/golden/thirdbody/generate.py statement for
// statement. The order is load-bearing: only IEEE-754 basic operations
// appear, so under the D-10 flags the digit loss demonstrated below is the
// bit-exact value recorded in the golden manifest on every platform. This
// evaluation exists only here, to demonstrate the cancellation the shipped
// f(q) formulation avoids; it is never part of the model.
Eigen::Vector3d naive_thirdbody_double(double gm, const Eigen::Vector3d& r,
                                       const Eigen::Vector3d& s) {
  const double d0 = s[0] - r[0];
  const double d1 = s[1] - r[1];
  const double d2 = s[2] - r[2];
  const double dn = std::sqrt(d0 * d0 + d1 * d1 + d2 * d2);
  const double sn = std::sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
  const double dn3 = dn * dn * dn;
  const double sn3 = sn * sn * sn;
  return Eigen::Vector3d(gm * (d0 / dn3 - s[0] / sn3),
                         gm * (d1 / dn3 - s[1] / sn3),
                         gm * (d2 / dn3 - s[2] / sn3));
}

double rel_err(const Eigen::Vector3d& a, const Eigen::Vector3d& ref) {
  return (a - ref).norm() / ref.norm();
}

}  // namespace

TEST_CASE("thirdbody_battin_extended_reference_golden") {
  // Phase 3 exit criterion 7: Battin f(q) matches the extended-precision
  // naive reference to < 1e-12 norm-relative at all 10 committed states.
  // Generation-time margin of the operation-order mirror: 3.4e-16 worst
  // case, so the gate carries ~3.5 orders of headroom while a formulation
  // error (wrong sign, wrong f(q)) shows as O(1).
  const auto cases = star_tests::load_golden_cases(kStates);
  REQUIRE(cases.size() == 10);
  double worst = 0.0;
  for (const auto& c : cases) {
    const std::string name = c.scalar("name");
    CAPTURE(name);
    const double gm = star_tests::parse_hex_double(c.scalar("gm_m3ps2"));
    const Eigen::Vector3d r = parse_vec3(c, "r_sc_m");
    const Eigen::Vector3d s = parse_vec3(c, "r_third_m");
    const Eigen::Vector3d ref = parse_vec3(c, "a_ref_mps2");
    const double err = rel_err(star::models::thirdbody_accel(gm, r, s), ref);
    CAPTURE(err);
    CHECK(err < 1e-12);  // criterion-7 gate
    worst = std::max(worst, err);
  }
  CAPTURE(worst);
  CHECK(worst < 1e-12);
}

TEST_CASE("thirdbody_naive_cancellation_digit_loss") {
  // Phase 3 exit criterion 7, near-alignment clause: at the flagged golden
  // state (low lunar orbiter 0.0012 rad off the Moon-Jupiter line, Jupiter
  // at a far conjunction, |s|/|r| ~ 5.4e5) the naive double-precision
  // difference loses >= 6 of binary64's ~15.95 significant decimal digits
  // against the extended-precision reference (rel err >= 1e-10; recorded
  // generation value 1.373e-10, i.e. 6.14 digits), while the Battin
  // implementation stays under the criterion gate at the same state. This
  // is the measured justification for shipping the f(q) formulation.
  const auto cases = star_tests::load_golden_cases(kStates);
  int demos = 0;
  for (const auto& c : cases) {
    if (c.scalar("digit_loss_demo") != "true") {
      continue;
    }
    ++demos;
    const std::string name = c.scalar("name");
    CAPTURE(name);
    const double gm = star_tests::parse_hex_double(c.scalar("gm_m3ps2"));
    const Eigen::Vector3d r = parse_vec3(c, "r_sc_m");
    const Eigen::Vector3d s = parse_vec3(c, "r_third_m");
    const Eigen::Vector3d ref = parse_vec3(c, "a_ref_mps2");
    const double err_naive = rel_err(naive_thirdbody_double(gm, r, s), ref);
    const double err_battin =
        rel_err(star::models::thirdbody_accel(gm, r, s), ref);
    CAPTURE(err_naive);
    CAPTURE(err_battin);
    CHECK(err_naive >= 1e-10);   // >= 6 significant digits lost
    CHECK(err_battin < 1e-12);   // criterion-7 gate at the same state
  }
  CHECK(demos >= 1);
}

TEST_CASE("thirdbody_limit_properties") {
  // Exact-zero limit: at the central body's center the direct and indirect
  // terms coincide, and the implementation reproduces that exactly - q = 0
  // gives f(q) = 0 (eq:thirdbody:q, eq:thirdbody:fq) and r contributes
  // nothing, so every component is exactly +-0. A tolerance here would hide
  // a spurious constant offset.
  const Eigen::Vector3d s(1.495978707e11, 2.0e10, -8.0e9);
  const Eigen::Vector3d zero = star::models::thirdbody_accel(
      1.327e20, Eigen::Vector3d::Zero(), s);
  CHECK(zero[0] == 0.0);
  CHECK(zero[1] == 0.0);
  CHECK(zero[2] == 0.0);

  // Cross-formulation agreement at a benign geometry: for the committed
  // earth_from_llo_generic state (|s|/|r| ~ 200) the naive double
  // evaluation is itself accurate to ~1e-13 (mild cancellation), so the
  // two algebraically identical forms must agree in double precision. The
  // 1e-11 bound is ~100x the expected naive rounding error and catches a
  // formulation divergence without flaking on rounding.
  const auto cases = star_tests::load_golden_cases(kStates);
  bool found = false;
  for (const auto& c : cases) {
    if (c.scalar("name") != "earth_from_llo_generic") {
      continue;
    }
    found = true;
    const double gm = star_tests::parse_hex_double(c.scalar("gm_m3ps2"));
    const Eigen::Vector3d r = parse_vec3(c, "r_sc_m");
    const Eigen::Vector3d s_case = parse_vec3(c, "r_third_m");
    const Eigen::Vector3d battin =
        star::models::thirdbody_accel(gm, r, s_case);
    const Eigen::Vector3d naive = naive_thirdbody_double(gm, r, s_case);
    const double agree = rel_err(naive, battin);
    CAPTURE(agree);
    CHECK(agree < 1e-11);
  }
  CHECK(found);
}
