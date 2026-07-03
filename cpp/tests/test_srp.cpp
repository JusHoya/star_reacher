// SRP and conical-shadow tests (FR-7, FR-22 layers 1-3, Phase 3 exit
// criterion 3): golden illumination fractions and cannonball
// accelerations, the analytic shadow-cone timing benchmark, and the
// shadow-fraction continuity property. Test IDs are cited by the
// math-library validation table (ch:srp); do not rename them.
//
// The golden references in tests/golden/srp/ are mpmath evaluations (60
// significant decimal digits) of the same model formulation from the exact
// committed binary64 inputs (provenance: tests/golden/srp/manifest.toml).
// The timing benchmark's reference is DIFFERENT mathematics: the classical
// tangent-cone construction (eq:srp:umbracone / eq:srp:penumbracone) and
// its closed-form circular-orbit crossing quadratic
// (eq:srp:conequadratic), derived in ch:srp and implemented only here, so
// the reference cannot share a defect with the model's apparent-disk path.
#include <cmath>
#include <string>

#include <Eigen/Dense>

#include "golden_io.hpp"
#include "star/constants.hpp"
#include "star/models/srp.hpp"
#include "vendor/doctest.h"

namespace {

const std::string kGoldenDir = STAR_GOLDEN_DIR;
const std::string kShadow = kGoldenDir + "/srp/shadow_fraction.toml";
const std::string kAccel = kGoldenDir + "/srp/accel.toml";

Eigen::Vector3d parse_vec3(const star_tests::GoldenCase& c,
                           const std::string& key) {
  const auto& a = c.array(key);
  REQUIRE(a.size() == 3);
  return Eigen::Vector3d(star_tests::parse_hex_double(a[0]),
                         star_tests::parse_hex_double(a[1]),
                         star_tests::parse_hex_double(a[2]));
}

// Reference shadow-crossing scenario for the timing and continuity tests:
// a circular orbit of radius kAOrb about an Earth-like occulter at the
// origin, orbit plane containing the Sun line, Sun fixed on +x at 1 au.
// The occulter radius is self-consistency test geometry (representative of
// Earth): both the model path and the analytic reference receive the same
// value, which is all the comparison requires. The solar radius and au are
// the model's cited constants.
constexpr double kROcc = 6378137.0;
constexpr double kAOrb = 6778137.0;
const double kRSun = star::constants::R_SUN_M;
const double kDSun = star::constants::AU_M;
const double kMu = star::constants::GM_EARTH_M3_PER_S2;

Eigen::Vector3d orbit_pos(double n, double t) {
  return Eigen::Vector3d(kAOrb * std::cos(n * t), kAOrb * std::sin(n * t),
                         0.0);
}

// Bisect f for a root in [lo, hi] (f(lo), f(hi) of opposite signs,
// asserted by the caller) to a time tolerance tol_s. Bisection is
// deterministic and needs no derivative; ~31 iterations resolve a
// quarter-orbit bracket to 1e-6 s, four orders below the 0.1 s gate.
template <typename F>
double bisect(const F& f, double lo, double hi, double tol_s) {
  double flo = f(lo);
  while (hi - lo > tol_s) {
    const double mid = 0.5 * (lo + hi);
    const double fm = f(mid);
    if ((flo < 0.0) == (fm < 0.0)) {
      lo = mid;
      flo = fm;
    } else {
      hi = mid;
    }
  }
  return 0.5 * (lo + hi);
}

// Analytic cone-crossing angle psi (in (pi/2, pi)) for the circular
// reference orbit, from the closed-form quadratic in u = cos(psi)
// (eq:srp:conequadratic): a^2(1+T^2)u^2 + 2 a ell_signed T^2 u +
// (ell^2 T^2 - a^2) = 0, where T = tan(alpha) and ell_signed is the apex
// abscissa measured so the cone radius at the spacecraft's axial station
// is T*(a*cos(psi) - ell_signed) with the appropriate sign. For the umbra
// cone (eq:srp:umbracone) the apex sits at x = -ell_u (anti-Sun side) and
// the crossing condition is |a sin psi| = T_u (a cos psi + ell_u); for the
// penumbra cone (eq:srp:penumbracone) the apex sits at x = +ell_p and the
// condition is |a sin psi| = T_p (ell_p - a cos psi). Both square to the
// same quadratic with ell_signed = -ell_u and +ell_p respectively. The
// physical (shadow-side) root is the negative one; the positive root is
// the squaring artifact on the sunlit side (ch:srp).
double cone_crossing_psi(double tan_alpha, double ell_signed) {
  const double t2 = tan_alpha * tan_alpha;
  const double qa = kAOrb * kAOrb * (1.0 + t2);
  const double qb = -2.0 * kAOrb * ell_signed * t2;
  const double qc = ell_signed * ell_signed * t2 - kAOrb * kAOrb;
  const double disc = qb * qb - 4.0 * qa * qc;
  REQUIRE(disc > 0.0);
  const double u = (-qb - std::sqrt(disc)) / (2.0 * qa);
  REQUIRE(u > -1.0);
  REQUIRE(u < 0.0);  // shadow side: behind the occulter
  return std::acos(u);
}

}  // namespace

TEST_CASE("srp_shadow_fraction_golden") {
  // Golden gate (FR-22 layer 1): every branch of the piecewise
  // illumination fraction (eq:srp:nu) against extended-precision
  // references. kind=exact cases (full sunlight, total umbra) must return
  // the exact constants - the model returns literals in those branches,
  // so any deviation is a branch-selection bug, not roundoff. Partial
  // cases compare at abs 1e-8: in the lens formula (eq:srp:overlap) the
  // occulter-disk term b^2*acos((c-x)/b) has (c-x)/b ~ 1 for a
  // large-occulter geometry (LEO Earth shadow: b ~ 1.2 rad vs a ~ 4.7e-3
  // rad), so ulp-level angle errors are amplified by
  // b^2/(pi a^2 sqrt(1-((c-x)/b)^2)) ~ 1e6..1e7 - observed 1.0e-9 worst
  // case on MSVC x64 at the committed penumbra states (manifest
  // derivation). 1e-8 gives an order of libm headroom across platforms
  // while a branch or geometry error still fails at O(1e-3) or worse; a
  // 1e-9 nu error is ~1e-9 of the SRP force for a seconds-long transit,
  // physically negligible against the C_R uncertainty (ch:srp).
  const auto cases = star_tests::load_golden_cases(kShadow);
  REQUIRE(cases.size() >= 11);
  double worst_partial = 0.0;
  for (const auto& c : cases) {
    const std::string name = c.scalar("name");
    CAPTURE(name);
    const double nu_ref = star_tests::parse_hex_double(c.scalar("nu"));
    const double nu = star::models::shadow_fraction(
        parse_vec3(c, "r_sc_m"), parse_vec3(c, "r_sun_m"),
        star_tests::parse_hex_double(c.scalar("radius_sun_m")),
        parse_vec3(c, "r_occ_m"),
        star_tests::parse_hex_double(c.scalar("radius_occ_m")));
    CAPTURE(nu);
    CAPTURE(nu_ref);
    if (c.scalar("kind") == "exact") {
      CHECK(nu == nu_ref);  // exactly 0.0 or exactly 1.0
    } else {
      const double err = std::fabs(nu - nu_ref);
      CAPTURE(err);
      CHECK(err <= 1e-8);
      worst_partial = std::max(worst_partial, err);
    }
  }
  CAPTURE(worst_partial);
  CHECK(worst_partial <= 1e-8);
}

TEST_CASE("srp_cannonball_accel_golden") {
  // Golden gate (FR-22 layer 1): the cannonball acceleration
  // (eq:srp:cannonball) against extended-precision references at
  // norm-relative 1e-12 (C++ accumulates ~5 IEEE roundings plus the
  // sub-ulp constant-quotient difference; manifest derivation). The umbra
  // case must be exactly zero in every component: nu scales the scalar
  // coefficient, so nu = 0 cannot leave residue.
  const auto cases = star_tests::load_golden_cases(kAccel);
  REQUIRE(cases.size() >= 5);
  double worst = 0.0;
  for (const auto& c : cases) {
    const std::string name = c.scalar("name");
    CAPTURE(name);
    const double cram =
        star_tests::parse_hex_double(c.scalar("cr_a_over_m_m2pkg"));
    const double nu = star_tests::parse_hex_double(c.scalar("nu"));
    const Eigen::Vector3d a = star::models::srp_accel(
        cram, nu, parse_vec3(c, "r_sc_m"), parse_vec3(c, "r_sun_m"));
    const Eigen::Vector3d ref = parse_vec3(c, "a_ref_mps2");
    if (nu == 0.0) {
      CHECK(a[0] == 0.0);
      CHECK(a[1] == 0.0);
      CHECK(a[2] == 0.0);
      continue;
    }
    const double err = (a - ref).norm() / ref.norm();
    CAPTURE(err);
    CHECK(err <= 1e-12);
    worst = std::max(worst, err);
  }
  CAPTURE(worst);
  CHECK(worst <= 1e-12);
}

TEST_CASE("srp_shadow_cone_timing") {
  // Phase 3 exit criterion 3: umbra (and penumbra) entry/exit times of the
  // reference circular-orbit shadow crossing, root-found on the model's
  // signed boundary functions (eq:srp:boundaries) by bisection along the
  // analytic orbit, must match the closed-form tangent-cone times to
  // < 0.1 s. The analytic path shares no code with the model: it is the
  // cone geometry of ch:srp Sec. "tangent cones" evaluated right here.
  // Also gate the conical-vs-cylindrical separation, so a cylindrical
  // shadow (the classic shortcut) cannot silently pass: the cone narrows
  // the umbra at orbit altitude by ~13 km, shifting each crossing by ~4 s.
  const double n = std::sqrt(kMu / (kAOrb * kAOrb * kAOrb));
  const double period = star::constants::TWO_PI / n;
  const Eigen::Vector3d r_sun(kDSun, 0.0, 0.0);
  const Eigen::Vector3d r_occ = Eigen::Vector3d::Zero();

  // Analytic umbra cone (eq:srp:umbracone): apex at x = -ell_u,
  // sin(alpha_u) = (R_sun - R_occ)/D.
  const double sin_au = (kRSun - kROcc) / kDSun;
  const double tan_au = sin_au / std::sqrt(1.0 - sin_au * sin_au);
  const double ell_u = kROcc * kDSun / (kRSun - kROcc);
  const double psi_u = cone_crossing_psi(tan_au, -ell_u);
  const double t_umb_in_analytic = psi_u / n;
  const double t_umb_out_analytic = (star::constants::TWO_PI - psi_u) / n;

  // Analytic penumbra cone (eq:srp:penumbracone): apex at x = +ell_p,
  // sin(alpha_p) = (R_sun + R_occ)/D.
  const double sin_ap = (kRSun + kROcc) / kDSun;
  const double tan_ap = sin_ap / std::sqrt(1.0 - sin_ap * sin_ap);
  const double ell_p = kROcc * kDSun / (kRSun + kROcc);
  const double psi_p = cone_crossing_psi(tan_ap, ell_p);
  const double t_pen_in_analytic = psi_p / n;
  const double t_pen_out_analytic = (star::constants::TWO_PI - psi_p) / n;

  // Model boundary functions along the orbit.
  const auto g_umbra = [&](double t) {
    return star::models::shadow_umbra_boundary(orbit_pos(n, t), r_sun, kRSun,
                                               r_occ, kROcc);
  };
  const auto g_penumbra = [&](double t) {
    return star::models::shadow_penumbra_boundary(orbit_pos(n, t), r_sun,
                                                  kRSun, r_occ, kROcc);
  };

  // Brackets: quarter orbit (sunlit, g > 0) to half orbit (anti-Sun,
  // g < 0) for entry; the mirror for exit. REQUIRE the signs so a broken
  // boundary function fails loudly instead of "converging" nowhere.
  REQUIRE(g_umbra(0.25 * period) > 0.0);
  REQUIRE(g_umbra(0.50 * period) < 0.0);
  REQUIRE(g_umbra(0.75 * period) > 0.0);
  REQUIRE(g_penumbra(0.25 * period) > 0.0);
  REQUIRE(g_penumbra(0.50 * period) < 0.0);
  REQUIRE(g_penumbra(0.75 * period) > 0.0);
  const double tol_s = 1e-6;  // four orders below the 0.1 s gate
  const double t_umb_in = bisect(g_umbra, 0.25 * period, 0.50 * period,
                                 tol_s);
  const double t_umb_out = bisect(g_umbra, 0.50 * period, 0.75 * period,
                                  tol_s);
  const double t_pen_in = bisect(g_penumbra, 0.25 * period, 0.50 * period,
                                 tol_s);
  const double t_pen_out = bisect(g_penumbra, 0.50 * period, 0.75 * period,
                                  tol_s);

  // Criterion-3 gate: 0.1 s. The disk-tangency boundary and the tangent
  // cone are the same surface for spheres (ch:srp equivalence argument),
  // so the observed difference is bisection tolerance, ~1e-6 s.
  const double d_umb_in = std::fabs(t_umb_in - t_umb_in_analytic);
  const double d_umb_out = std::fabs(t_umb_out - t_umb_out_analytic);
  const double d_pen_in = std::fabs(t_pen_in - t_pen_in_analytic);
  const double d_pen_out = std::fabs(t_pen_out - t_pen_out_analytic);
  CAPTURE(d_umb_in);
  CAPTURE(d_umb_out);
  CAPTURE(d_pen_in);
  CAPTURE(d_pen_out);
  CHECK(d_umb_in < 0.1);
  CHECK(d_umb_out < 0.1);
  CHECK(d_pen_in < 0.1);
  CHECK(d_pen_out < 0.1);

  // Cylinder guard: a cylindrical shadow of radius R_occ enters at
  // psi = pi - asin(R_occ/a). The conical times must differ measurably
  // (expected ~4 s here), or the test would accept the cylinder shortcut.
  const double psi_cyl = star::constants::PI - std::asin(kROcc / kAOrb);
  const double t_cyl_in = psi_cyl / n;
  const double t_cyl_out = (star::constants::TWO_PI - psi_cyl) / n;
  const double d_cyl_in = std::fabs(t_umb_in - t_cyl_in);
  const double d_cyl_out = std::fabs(t_umb_out - t_cyl_out);
  CAPTURE(d_cyl_in);
  CAPTURE(d_cyl_out);
  CHECK(d_cyl_in > 1.0);
  CHECK(d_cyl_out > 1.0);
}

TEST_CASE("srp_shadow_fraction_continuity") {
  // Property gate (FR-22 layer 2): along the reference shadow crossing,
  // nu is exactly 1 before the penumbra, exactly 0 inside the umbra,
  // strictly interior at the penumbra midpoint, within [0, 1] everywhere,
  // and continuous: with a ~8.2 s penumbra transit, 1 ms sampling bounds
  // the physical per-sample change by ~2e-4, so a 1e-3 jump gate passes
  // smooth evolution and fails any branch-boundary discontinuity.
  const double n = std::sqrt(kMu / (kAOrb * kAOrb * kAOrb));
  const double period = star::constants::TWO_PI / n;
  const Eigen::Vector3d r_sun(kDSun, 0.0, 0.0);
  const Eigen::Vector3d r_occ = Eigen::Vector3d::Zero();
  const auto nu_at = [&](double t) {
    return star::models::shadow_fraction(orbit_pos(n, t), r_sun, kRSun,
                                         r_occ, kROcc);
  };
  const auto g_umbra = [&](double t) {
    return star::models::shadow_umbra_boundary(orbit_pos(n, t), r_sun, kRSun,
                                               r_occ, kROcc);
  };
  const auto g_penumbra = [&](double t) {
    return star::models::shadow_penumbra_boundary(orbit_pos(n, t), r_sun,
                                                  kRSun, r_occ, kROcc);
  };
  const double t_pen_in = bisect(g_penumbra, 0.25 * period, 0.50 * period,
                                 1e-6);
  const double t_umb_in = bisect(g_umbra, 0.25 * period, 0.50 * period,
                                 1e-6);

  CHECK(nu_at(t_pen_in - 2.0) == 1.0);  // fully lit before the penumbra
  CHECK(nu_at(t_umb_in + 2.0) == 0.0);  // total umbra after entry
  const double nu_mid = nu_at(0.5 * (t_pen_in + t_umb_in));
  CAPTURE(nu_mid);
  CHECK(nu_mid > 0.0);
  CHECK(nu_mid < 1.0);

  const double t0 = t_pen_in - 2.0;
  const double t1 = t_umb_in + 2.0;
  const double dt = 1e-3;
  double prev = nu_at(t0);
  double max_jump = 0.0;
  for (double t = t0 + dt; t <= t1; t += dt) {
    const double nu = nu_at(t);
    CHECK(nu >= 0.0);
    CHECK(nu <= 1.0);
    max_jump = std::max(max_jump, std::fabs(nu - prev));
    prev = nu;
  }
  CAPTURE(max_jump);
  CHECK(max_jump < 1e-3);
}
