// Gravity model tests (FR-5, FR-22 layers 1-3): cross-tool golden agreement
// against independently synthesized pyshtools accelerations (Phase 3 exit
// criterion 1, GRAV-XTOOL-20), the 30-day J2 secular-rate analytic benchmark
// (Phase 3 exit criterion 2, GRAV-J2-SECULAR), tier semantics, pole
// regularity, truncation consistency, and SRGRAV loader error paths. Test
// IDs are cited by the math-library validation table; do not rename them.
//
// Golden provenance: tests/golden/gravity/manifest.toml. The committed
// pyshtools accelerations were produced over the SAME committed excerpt
// coefficient files these tests load, so the comparison isolates the
// evaluation algorithm (Pines here, colatitude Legendre recursion there).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "golden_io.hpp"
#include "star/constants.hpp"
#include "star/events.hpp"
#include "star/integrate.hpp"
#include "star/models/gravity.hpp"
#include "vendor/doctest.h"

namespace {

using star::models::GravityField;
using star::models::GravityTier;
using star::models::PinesGravity;

const std::string kGoldenDir = STAR_GOLDEN_DIR;
const std::string kGravityDir = kGoldenDir + "/gravity";
const std::string kEarthN20 = kGravityDir + "/earth_egm2008_n20.srgrav";

// <cmath> provides no portable pi in C++17 (the reason constants.hpp defines
// TWO_PI); degree inputs below convert through it.
constexpr double kDegToRad = star::constants::TWO_PI / 360.0;

bool same_vec_bits(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
  return a.x() == b.x() && a.y() == b.y() && a.z() == b.z();
}

Eigen::Vector3d vec3_from(const star_tests::GoldenCase& c,
                          const std::string& key) {
  const auto& a = c.array(key);
  REQUIRE(a.size() == 3);
  return Eigen::Vector3d(star_tests::parse_hex_double(a[0]),
                         star_tests::parse_hex_double(a[1]),
                         star_tests::parse_hex_double(a[2]));
}

// Osculating classical elements from a Cartesian state (Vallado,
// Fundamentals of Astrodynamics and Applications, RV -> COE algorithm).
// Local to this test: production element conversion lands with the loader
// phase; the acceptance gate only needs a, e, i and the unwrapped secular
// angles here.
struct Elements {
  double a, e, i, raan, argp;
};

Elements elements_from_rv(double mu, const Eigen::Vector3d& r,
                          const Eigen::Vector3d& v) {
  const Eigen::Vector3d h = r.cross(v);
  const Eigen::Vector3d node(-h.y(), h.x(), 0.0);  // z-hat x h
  const double rn = r.norm();
  const Eigen::Vector3d evec =
      ((v.squaredNorm() - mu / rn) * r - r.dot(v) * v) / mu;
  Elements el;
  el.e = evec.norm();
  el.a = 1.0 / (2.0 / rn - v.squaredNorm() / mu);
  el.i = std::acos(h.z() / h.norm());
  el.raan = std::atan2(node.y(), node.x());
  const double cos_argp =
      node.dot(evec) / (node.norm() * el.e);
  el.argp = std::acos(std::min(1.0, std::max(-1.0, cos_argp)));
  if (evec.z() < 0.0) {
    el.argp = -el.argp;  // measured from the ascending node, southward e
  }
  return el;
}

// Least-squares slope of x(t); the secular-rate estimator for the fitted
// nodal-regression and apsidal rates.
double fitted_slope(const std::vector<double>& t, const std::vector<double>& x) {
  const std::size_t n = t.size();
  double tm = 0.0;
  double xm = 0.0;
  for (std::size_t k = 0; k < n; ++k) {
    tm += t[k];
    xm += x[k];
  }
  tm /= static_cast<double>(n);
  xm /= static_cast<double>(n);
  double sxy = 0.0;
  double sxx = 0.0;
  for (std::size_t k = 0; k < n; ++k) {
    sxy += (t[k] - tm) * (x[k] - xm);
    sxx += (t[k] - tm) * (t[k] - tm);
  }
  return sxy / sxx;
}

}  // namespace

TEST_CASE("GRAV-XTOOL-20") {
  // Phase 3 exit criterion 1: harmonic acceleration matches independently
  // synthesized values at 20 test states to < 1e-12 relative. The golden
  // side is pyshtools 4.14.1 MakeGravGridPoint (colatitude-recursion
  // algorithm, independent authorship) over the SAME committed excerpt
  // coefficients; positions are committed as binary64 hex, so both sides
  // evaluate the identical point and the identical field. Measured worst
  // case at generation time between pyshtools and a Python mirror of this
  // Pines implementation: 2.7e-15 (manifest.toml), so the 1e-12 gate holds
  // ~2.5 orders of margin while any recursion or normalization defect
  // (which produces >= 1e-9 here) fails decisively.
  const auto cases =
      star_tests::load_golden_cases(kGravityDir + "/pyshtools_accel.toml");
  REQUIRE(cases.size() == 20);

  // Load each referenced excerpt once; evaluator per field (workspace
  // reuse across states is the production usage pattern).
  std::vector<std::string> files;
  std::vector<PinesGravity> models;
  for (const auto& c : cases) {
    const std::string f = c.scalar("field_file");
    if (std::find(files.begin(), files.end(), f) == files.end()) {
      files.push_back(f);
      models.emplace_back(GravityField::load_file(kGravityDir + "/" + f));
    }
  }

  double worst_rel = 0.0;
  for (const auto& c : cases) {
    const std::string name = c.scalar("name");
    CAPTURE(name);
    const std::string f = c.scalar("field_file");
    const std::size_t fi = static_cast<std::size_t>(
        std::find(files.begin(), files.end(), f) - files.begin());
    const int n_eval = std::stoi(c.scalar("n_eval"));
    const int m_eval = std::stoi(c.scalar("m_eval"));
    const Eigen::Vector3d r_bf = vec3_from(c, "r_bf_m");
    const Eigen::Vector3d golden = vec3_from(c, "accel_mps2");
    const Eigen::Vector3d got =
        models[fi].acceleration(r_bf, GravityTier::kFull, n_eval, m_eval);
    const double rel = (got - golden).norm() / golden.norm();
    CAPTURE(rel);
    CHECK(rel < 1e-12);
    worst_rel = std::max(worst_rel, rel);
  }
  MESSAGE("GRAV-XTOOL-20 worst relative difference over 20 states: "
          << worst_rel << " (gate 1e-12)");
}

TEST_CASE("GRAV-J2-SECULAR") {
  // Phase 3 exit criterion 2: fitted 30-day nodal-regression and apsidal
  // rates from a J2-only propagation match the first-order analytic secular
  // formulas (Vallado, Fundamentals of Astrodynamics and Applications,
  // 4th ed., the J2 secular rates
  //   dRAAN/dt = -(3/2) n J2 (R/p)^2 cos i,
  //   dargp/dt = +(3/4) n J2 (R/p)^2 (5 cos^2 i - 1))
  // within 0.5 %. J2 and (GM, R) come from the committed EGM2008 excerpt
  // itself, so the numerical and analytic sides share one field definition.
  //
  // Frame note: the J2-only tier is zonal, hence axially symmetric about
  // the body z axis; a propagation that holds the body-fixed axes inertial
  // (no Earth rotation) produces the same orbit-plane dynamics, so no frame
  // chain is needed for this benchmark.
  //
  // Orbit: sun-synchronous-class LEO (a = 7078.137 km, e = 0.02,
  // i = 97.8 deg), deliberately away from the critical inclination
  // (63.435 deg) where the apsidal rate crosses zero and a relative gate
  // would be ill-posed.
  //
  // First-order theory predicts rates in MEAN elements; the fit is over
  // osculating samples. Mean a, e, i are estimated by averaging the
  // osculating elements over the first orbital period (their short-period
  // J2 oscillations have zero first-order mean), after which the residual
  // model error is O(J2^2) ~ 1e-6 relative -- far inside the 0.5 % gate.
  // The long-arc (30-day) least-squares fit averages the short-period
  // oscillation of the fitted angles themselves down by ~(T_orbit/T_span).
  PinesGravity model(GravityField::load_file(kEarthN20));
  const double mu = model.field().gm_m3ps2;
  const double R = model.field().ref_radius_m;
  const double j2 = star::models::j2_from_field(model.field());
  CHECK(j2 == doctest::Approx(1.0826e-3).epsilon(1e-3));  // EGM2008 sanity

  // Initial osculating elements -> perifocal -> inertial state.
  const double a0 = 7078137.0;
  const double e0 = 0.02;
  const double i0 = 97.8 * kDegToRad;
  const double raan0 = 30.0 * kDegToRad;
  const double argp0 = 50.0 * kDegToRad;
  const double nu0 = 0.0;
  const double p0 = a0 * (1.0 - e0 * e0);
  const double r0n = p0 / (1.0 + e0 * std::cos(nu0));
  const Eigen::Vector3d r_pf(r0n * std::cos(nu0), r0n * std::sin(nu0), 0.0);
  const Eigen::Vector3d v_pf(-std::sqrt(mu / p0) * std::sin(nu0),
                             std::sqrt(mu / p0) * (e0 + std::cos(nu0)), 0.0);
  const Eigen::Matrix3d rot =
      (Eigen::AngleAxisd(raan0, Eigen::Vector3d::UnitZ()) *
       Eigen::AngleAxisd(i0, Eigen::Vector3d::UnitX()) *
       Eigen::AngleAxisd(argp0, Eigen::Vector3d::UnitZ()))
          .toRotationMatrix();
  const Eigen::Vector3d r0v = rot * r_pf;
  const Eigen::Vector3d v0v = rot * v_pf;

  auto rhs = [&model](double /*t*/, const double* y, double* ydot) {
    const Eigen::Map<const Eigen::Vector3d> r(y);
    const Eigen::Vector3d acc =
        model.acceleration(r, GravityTier::kJ2Only);
    ydot[0] = y[3];
    ydot[1] = y[4];
    ydot[2] = y[5];
    ydot[3] = acc.x();
    ydot[4] = acc.y();
    ydot[5] = acc.z();
  };
  const star::integrate::RhsRef f(rhs);

  // Sample osculating elements at every accepted step endpoint.
  std::vector<double> ts, raans, argps, as, es, is;
  auto observer = [&](const star::integrate::DenseStep& d) {
    const Eigen::Map<const Eigen::Vector3d> r(d.y1);
    const Eigen::Map<const Eigen::Vector3d> v(d.y1 + 3);
    const Elements el = elements_from_rv(mu, r, v);
    const double t = d.t0 + d.h;
    // Unwrap both angles against their previous sample; per-step secular
    // motion is << pi, so branch jumps are unambiguous.
    double raan = el.raan;
    double argp = el.argp;
    if (!ts.empty()) {
      const double two_pi = star::constants::TWO_PI;
      raan += two_pi * std::round((raans.back() - raan) / two_pi);
      argp += two_pi * std::round((argps.back() - argp) / two_pi);
    }
    ts.push_back(t);
    raans.push_back(raan);
    argps.push_back(argp);
    as.push_back(el.a);
    es.push_back(el.e);
    is.push_back(el.i);
  };
  const star::events::StepObserverRef obs(observer);

  star::events::PropagateOptions opt;
  opt.method = star::events::Method::kRkf78;
  opt.mode = star::events::StepMode::kAdaptive;
  opt.adaptive.groups = {
      {"position", 0, 3, 1e-10, 1e-4},
      {"velocity", 3, 3, 1e-10, 1e-7},
  };
  opt.adaptive.h_init = 60.0;
  opt.adaptive.h_max = 600.0;
  const double t_span = 30.0 * 86400.0;
  double y0[6] = {r0v[0], r0v[1], r0v[2], v0v[0], v0v[1], v0v[2]};
  double yf[6];
  const auto res = star::events::propagate(f, 0.0, t_span, y0, 6, yf, opt,
                                           nullptr, 0, nullptr, obs);
  CHECK(res.t_final == t_span);
  REQUIRE(ts.size() > 1000);  // the fit needs dense multi-orbit coverage

  // Mean elements from the first orbital period.
  const double period0 =
      star::constants::TWO_PI * std::sqrt(a0 * a0 * a0 / mu);
  double am = 0.0, em = 0.0, im = 0.0;
  std::size_t nm = 0;
  for (std::size_t k = 0; k < ts.size() && ts[k] <= period0; ++k) {
    am += as[k];
    em += es[k];
    im += is[k];
    ++nm;
  }
  REQUIRE(nm > 10);
  am /= static_cast<double>(nm);
  em /= static_cast<double>(nm);
  im /= static_cast<double>(nm);

  const double n_mean = std::sqrt(mu / (am * am * am));
  const double p_mean = am * (1.0 - em * em);
  const double rp2 = (R / p_mean) * (R / p_mean);
  const double raan_dot_analytic =
      -1.5 * n_mean * j2 * rp2 * std::cos(im);
  const double argp_dot_analytic =
      0.75 * n_mean * j2 * rp2 * (5.0 * std::cos(im) * std::cos(im) - 1.0);

  const double raan_dot_fit = fitted_slope(ts, raans);
  const double argp_dot_fit = fitted_slope(ts, argps);
  const double raan_rel_err =
      std::fabs(raan_dot_fit / raan_dot_analytic - 1.0);
  const double argp_rel_err =
      std::fabs(argp_dot_fit / argp_dot_analytic - 1.0);
  CAPTURE(raan_dot_analytic);
  CAPTURE(raan_dot_fit);
  CAPTURE(argp_dot_analytic);
  CAPTURE(argp_dot_fit);
  MESSAGE("GRAV-J2-SECULAR relative errors vs analytic: RAAN "
          << raan_rel_err << ", argp " << argp_rel_err << " (gate 5e-3)");
  CHECK(raan_rel_err < 5e-3);  // Phase 3 exit criterion 2: within 0.5 %
  CHECK(argp_rel_err < 5e-3);
  // Sanity on sign and regime: retrograde sun-sync-class orbit regresses
  // the node eastward (positive rate) and rotates the apsis backward.
  CHECK(raan_dot_fit > 0.0);
  CHECK(argp_dot_fit < 0.0);
}

TEST_CASE("gravity_pointmass_tier") {
  // FR-5 point-mass tier: the degree-0 evaluation must equal the closed
  // form -GM/r^2 r-hat through the same Pines assembly (the central term is
  // carried by the a4 radial sum), at exact bit level for the magnitude
  // along each axis where the closed form is one division and multiply.
  PinesGravity model(GravityField::load_file(kEarthN20));
  const double mu = model.field().gm_m3ps2;
  const Eigen::Vector3d r(7000e3, -1200e3, 3400e3);
  const Eigen::Vector3d a = model.acceleration(r, GravityTier::kPointMass);
  const double rn = r.norm();
  const Eigen::Vector3d expect = -mu / (rn * rn) * (r / rn);
  CHECK((a - expect).norm() / expect.norm() < 1e-15);

  // A pure point-mass field (factory) must agree with the tier exactly.
  PinesGravity pm(GravityField::point_mass("pm", mu));
  const Eigen::Vector3d a_pm = pm.acceleration(r, GravityTier::kFull);
  CHECK(same_vec_bits(a_pm, a));
}

TEST_CASE("gravity_j2_tier_closed_form") {
  // FR-5 J2-only tier vs the closed-form J2 perturbation (Vallado, 4th ed.,
  // the J2 acceleration in body-axes with z the symmetry axis):
  //   a_J2 = -(3/2) J2 (mu/r^2) (R/r)^2 *
  //          [ (1 - 5 u^2) x/r, (1 - 5 u^2) y/r, (3 - 5 u^2) z/r ],
  // u = z/r, added to the central term. Both sides here share one operation
  // count small enough that agreement to 1e-14 relative isolates the
  // normalization chain (J2 = -sqrt(5) C-bar(2,0)) and the (2,0) term of
  // the Pines assembly.
  PinesGravity model(GravityField::load_file(kEarthN20));
  const double mu = model.field().gm_m3ps2;
  const double R = model.field().ref_radius_m;
  const double j2 = star::models::j2_from_field(model.field());

  const Eigen::Vector3d pts[] = {
      {7000e3, 0.0, 0.0},          // equatorial
      {5000e3, -3000e3, 4000e3},   // generic
      {0.0, 0.0, 7000e3},          // polar axis
      {-4500e3, 2500e3, -5200e3},  // southern generic
  };
  for (const Eigen::Vector3d& r : pts) {
    CAPTURE(r.transpose());
    const Eigen::Vector3d got = model.acceleration(r, GravityTier::kJ2Only);
    const double rn = r.norm();
    const double u = r.z() / rn;
    const double k = -1.5 * j2 * (mu / (rn * rn)) * (R / rn) * (R / rn);
    const Eigen::Vector3d central = -mu / (rn * rn) * (r / rn);
    const Eigen::Vector3d j2acc(k * (1.0 - 5.0 * u * u) * r.x() / rn,
                                k * (1.0 - 5.0 * u * u) * r.y() / rn,
                                k * (3.0 - 5.0 * u * u) * r.z() / rn);
    const Eigen::Vector3d expect = central + j2acc;
    const double rel = (got - expect).norm() / expect.norm();
    CAPTURE(rel);
    CHECK(rel < 1e-14);
  }
}

TEST_CASE("gravity_pole_regularity") {
  // The Pines formulation's reason for existence (FR-5): evaluation exactly
  // on the poles (s = t = 0, u = +/-1) is a regular point. The check is
  // continuity against a point 1 m off-axis: the acceleration gradient is
  // bounded by ~3 mu / r^3 ~ 4e-6 1/s per meter here, so 1e-4 m/s^2 is a
  // generous continuity bound that still fails on any pole singularity
  // (which produces NaN or O(1) discontinuities).
  PinesGravity model(GravityField::load_file(kEarthN20));
  const double r = 6778137.0;
  for (const double sgn : {1.0, -1.0}) {
    CAPTURE(sgn);
    const Eigen::Vector3d pole(0.0, 0.0, sgn * r);
    const Eigen::Vector3d a_pole =
        model.acceleration(pole, GravityTier::kFull, 20, 20);
    CHECK(std::isfinite(a_pole.x()));
    CHECK(std::isfinite(a_pole.y()));
    CHECK(std::isfinite(a_pole.z()));
    // Radial dominance at the pole: the transverse component is set by the
    // (2,1)-class coefficients (~1e-6 of g), never O(g).
    CHECK(std::fabs(a_pole.z()) > 8.0);
    CHECK(std::hypot(a_pole.x(), a_pole.y()) < 1e-3);
    const Eigen::Vector3d near(1.0, 0.0, sgn * r);
    const Eigen::Vector3d a_near =
        model.acceleration(near, GravityTier::kFull, 20, 20);
    CHECK((a_pole - a_near).norm() < 1e-4);
  }
}

TEST_CASE("gravity_truncation_consistency") {
  // Runtime degree/order selection (FR-5 configurable n x m) must be
  // arithmetically identical to evaluating a field truncated at load time:
  // same recursions, same coefficients, same summation order => the results
  // are required BIT-identical, which pins the truncation logic to "sum
  // fewer terms" with no other effect.
  const GravityField full = GravityField::load_file(kEarthN20);
  std::vector<double> c8, s8;
  for (int n = 0; n <= 8; ++n) {
    for (int m = 0; m <= n; ++m) {
      c8.push_back(full.cnm(n, m));
      s8.push_back(full.snm(n, m));
    }
  }
  PinesGravity model_full(full);
  PinesGravity model_8(GravityField::from_coefficients(
      "EGM2008_8", full.gm_m3ps2, full.ref_radius_m, 8, 8, c8, s8,
      full.tide_system));

  const Eigen::Vector3d pts[] = {
      {6778137.0, 0.0, 0.0},
      {-2500e3, 5800e3, 3100e3},
      {0.0, 0.0, -6778137.0},
  };
  for (const Eigen::Vector3d& r : pts) {
    CAPTURE(r.transpose());
    const Eigen::Vector3d a_runtime =
        model_full.acceleration(r, GravityTier::kFull, 8, 8);
    const Eigen::Vector3d a_loadtime =
        model_8.acceleration(r, GravityTier::kFull);
    CHECK(same_vec_bits(a_runtime, a_loadtime));  // bitwise, no tolerance
  }

  // Determinism spot check (D-10): identical calls give identical bits.
  const Eigen::Vector3d r(6778137.0, 1000e3, -2000e3);
  const Eigen::Vector3d a1 = model_full.acceleration(r, GravityTier::kFull);
  const Eigen::Vector3d a2 = model_full.acceleration(r, GravityTier::kFull);
  CHECK(same_vec_bits(a1, a2));

  // Out-of-band requests are refused, never silently degraded (FR-5 /
  // srgrav_v1.md section 5).
  CHECK_THROWS_AS(model_full.acceleration(r, GravityTier::kFull, 21, 21),
                  std::invalid_argument);
  CHECK_THROWS_AS(model_full.acceleration(r, GravityTier::kFull, 10, 11),
                  std::invalid_argument);
  PinesGravity pm(GravityField::point_mass("pm", full.gm_m3ps2));
  CHECK_THROWS_AS(pm.acceleration(r, GravityTier::kJ2Only),
                  std::invalid_argument);
  CHECK_THROWS_AS(model_full.acceleration(Eigen::Vector3d::Zero(),
                                          GravityTier::kFull),
                  std::domain_error);
}

TEST_CASE("gravity_srgrav_error_paths") {
  // SRGRAV v1 loader rejections (docs/formats/srgrav_v1.md): every
  // malformed input is refused with a specific error, never read as
  // garbage coefficients.
  std::ifstream in(kEarthN20, std::ios::binary);
  REQUIRE(in.good());
  std::vector<char> bytes((std::istreambuf_iterator<char>(in)),
                          std::istreambuf_iterator<char>());
  REQUIRE(bytes.size() > 96);

  auto write_temp = [](const std::string& name, const std::vector<char>& data) {
    // The test binary's working directory (the build tree) is writable;
    // unique names keep parallel ctest invocations from colliding.
    std::ofstream out(name, std::ios::binary | std::ios::trunc);
    out.write(data.data(), static_cast<std::streamsize>(data.size()));
    return name;
  };

  std::vector<char> bad_magic = bytes;
  bad_magic[0] ^= 0x40;
  const std::string p1 = write_temp("gravity_test_bad_magic.srgrav", bad_magic);
  CHECK_THROWS_AS(GravityField::load_file(p1), std::runtime_error);

  std::vector<char> bad_major = bytes;
  bad_major[8] = 2;
  const std::string p2 = write_temp("gravity_test_bad_major.srgrav", bad_major);
  CHECK_THROWS_AS(GravityField::load_file(p2), std::runtime_error);

  std::vector<char> truncated(bytes.begin(),
                              bytes.begin() + static_cast<long>(bytes.size() / 2));
  const std::string p3 = write_temp("gravity_test_truncated.srgrav", truncated);
  CHECK_THROWS_AS(GravityField::load_file(p3), std::runtime_error);

  std::vector<char> bad_tide = bytes;
  bad_tide[20] = 9;
  const std::string p4 = write_temp("gravity_test_bad_tide.srgrav", bad_tide);
  CHECK_THROWS_AS(GravityField::load_file(p4), std::runtime_error);

  CHECK_THROWS_AS(GravityField::load_file("gravity_test_does_not_exist.srgrav"),
                  std::runtime_error);

  // The specific-defect messages name the failure class (DX-2 spirit).
  try {
    GravityField::load_file(p2);
    FAIL("expected std::runtime_error");
  } catch (const std::runtime_error& exc) {
    const std::string msg = exc.what();
    CHECK(msg.find("major version 2") != std::string::npos);
  }

  std::remove(p1.c_str());
  std::remove(p2.c_str());
  std::remove(p3.c_str());
  std::remove(p4.c_str());
}
