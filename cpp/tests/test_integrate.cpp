// Integrator library acceptance suite (FR-11; Phase 2 exit criteria 3-4).
// Test IDs are cited by the math-library validation table (ch:integrators);
// do not rename them. The reference-orbit numbers come from the committed
// goldens in tests/golden/integrators/ (provenance in manifest.toml there);
// the headline measurements are computed by the shared drivers in
// star/testsupport/acceptance.hpp, the same code the Python evidence uses.
#include <cmath>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "golden_io.hpp"
#include "star/integrate.hpp"
#include "star/testsupport/acceptance.hpp"
#include "star/testsupport/kepler_ref.hpp"
#include "vendor/doctest.h"

namespace {

const star_tests::GoldenCase& find_case(
    const std::vector<star_tests::GoldenCase>& cases, const std::string& name) {
  for (const star_tests::GoldenCase& c : cases) {
    if (c.scalar("name") == name) {
      return c;
    }
  }
  throw std::runtime_error("golden case not found: " + name);
}

Eigen::Vector3d vec3(const star_tests::GoldenCase& c, const std::string& key) {
  const std::vector<std::string>& a = c.array(key);
  if (a.size() != 3) {
    throw std::runtime_error("golden array " + key + " is not length 3");
  }
  return Eigen::Vector3d(star_tests::parse_hex_double(a[0]),
                         star_tests::parse_hex_double(a[1]),
                         star_tests::parse_hex_double(a[2]));
}

struct ReferenceOrbit {
  double mu;
  Eigen::Vector3d r0, v0;
  double period, max_r4;
};

ReferenceOrbit load_reference_orbit() {
  const auto cases = star_tests::load_golden_cases(
      std::string(STAR_GOLDEN_DIR) + "/integrators/kepler_orbit.toml");
  const star_tests::GoldenCase& def = find_case(cases, "definition");
  ReferenceOrbit orb;
  orb.mu = star_tests::parse_hex_double(def.scalar("mu_m3ps2"));
  orb.r0 = vec3(def, "r0_m");
  orb.v0 = vec3(def, "v0_mps");
  orb.period = star_tests::parse_hex_double(def.scalar("period_s"));
  orb.max_r4 = star_tests::parse_hex_double(def.scalar("max_r4_mps4"));
  return orb;
}

// Dyadic measurement span (~1.007 orbital periods): an exact integer of
// seconds divisible by every ladder step below, so accumulated step times
// carry zero rounding error and the measured errors are pure truncation
// (see the time-accounting note in star/testsupport/acceptance.hpp).
constexpr double kSpanS = 7168.0;  // = 2^10 * 7

}  // namespace

TEST_CASE("rkf78_tableau_order_conditions") {
  // Machine gate against tableau transcription error (the classic failure
  // mode for RKF7(8)): the exact rational coefficients of Fehlberg,
  // NASA TR R-287 must satisfy the row-sum conditions sum_j a_ij = c_i, the
  // quadrature order conditions sum_i b_i c_i^q = 1/(q+1) for q up to
  // order-1 (7th-order weights: q <= 6; 8th-order weights: q <= 7), and the
  // order-3/4 coupling conditions listed below (Hairer-Norsett-Wanner,
  // Solving ODEs I, ch. II.2, Butcher order conditions). A single wrong
  // digit in any coefficient breaks at least one of these identities.
  // Tolerance: the identities are sums of ~13 double-rounded rationals of
  // magnitude O(1..20); 1e-13 absorbs that rounding while any transcription
  // error registers at >= 1e-4.
  using star::integrate::rkf78::kA;
  using star::integrate::rkf78::kB7;
  using star::integrate::rkf78::kB8;
  using star::integrate::rkf78::kC;
  using star::integrate::rkf78::kStages;
  const double tol = 1e-13;

  for (int i = 0; i < kStages; ++i) {
    double row = 0.0;
    for (int j = 0; j < kStages - 1; ++j) {
      row += kA[i][j];
    }
    CAPTURE(i);
    CHECK(std::fabs(row - kC[i]) < tol);
  }

  for (int order = 7; order <= 8; ++order) {
    const double* b = (order == 7) ? kB7 : kB8;
    for (int q = 0; q <= order - 1; ++q) {
      double s = 0.0;
      for (int i = 0; i < kStages; ++i) {
        s += b[i] * std::pow(kC[i], q);
      }
      CAPTURE(order);
      CAPTURE(q);
      CHECK(std::fabs(s - 1.0 / static_cast<double>(q + 1)) < tol);
    }
    // Coupling conditions (rooted trees of orders 3 and 4):
    //   sum b_i a_ij c_j = 1/6,      sum b_i c_i a_ij c_j = 1/8,
    //   sum b_i a_ij c_j^2 = 1/12,   sum b_i a_ij a_jk c_k = 1/24.
    double bac = 0.0, bcac = 0.0, bac2 = 0.0, baac = 0.0;
    for (int i = 0; i < kStages; ++i) {
      double ac = 0.0, ac2 = 0.0, aac = 0.0;
      for (int j = 0; j < kStages - 1; ++j) {
        ac += kA[i][j] * kC[j];
        ac2 += kA[i][j] * kC[j] * kC[j];
        double inner = 0.0;
        for (int k = 0; k < kStages - 1; ++k) {
          inner += kA[j][k] * kC[k];
        }
        aac += kA[i][j] * inner;
      }
      bac += b[i] * ac;
      bcac += b[i] * kC[i] * ac;
      bac2 += b[i] * ac2;
      baac += b[i] * aac;
    }
    CAPTURE(order);
    CHECK(std::fabs(bac - 1.0 / 6.0) < tol);
    CHECK(std::fabs(bcac - 1.0 / 8.0) < tol);
    CHECK(std::fabs(bac2 - 1.0 / 12.0) < tol);
    CHECK(std::fabs(baac - 1.0 / 24.0) < tol);
  }
}

TEST_CASE("kepler_reference_propagator_golden") {
  // The C++ analytic reference propagator (star/testsupport/kepler_ref.hpp)
  // must reproduce the independently generated Python checkpoints. Tolerance
  // per tests/golden/integrators/manifest.toml: 1e-6 m / 1e-9 m/s (two
  // independent implementations of the closed-form solution agree to
  // ~1e-8 m; the bound leaves margin for cross-platform libm spread).
  const ReferenceOrbit orb = load_reference_orbit();
  const star::testsupport::EllipticOrbitRef kepler(orb.mu, orb.r0, orb.v0);
  CHECK(std::fabs(kepler.period - orb.period) < 1e-9 * orb.period);

  const auto cases = star_tests::load_golden_cases(
      std::string(STAR_GOLDEN_DIR) + "/integrators/kepler_orbit.toml");
  int checked = 0;
  for (const star_tests::GoldenCase& c : cases) {
    if (c.scalar("name").rfind("checkpoint_", 0) != 0) {
      continue;
    }
    const double t = star_tests::parse_hex_double(c.scalar("t_s"));
    Eigen::Vector3d r, v;
    kepler.state_at(t, &r, &v);
    const double dr = (r - vec3(c, "r_m")).norm();
    const double dv = (v - vec3(c, "v_mps")).norm();
    CAPTURE(t);
    CAPTURE(dr);
    CAPTURE(dv);
    CHECK(dr < 1e-6);
    CHECK(dv < 1e-9);
    ++checked;
  }
  CHECK(checked == 8);
}

TEST_CASE("rk4_kepler_convergence_slope") {
  // Phase 2 exit criterion 3 (RK4 half): global position error at the end
  // of the dyadic span vs step size over a dt-halving ladder, fitted
  // log-log slope 4.0 +/- 0.2. Ladder h = 16..2 s: measured truncation
  // error spans ~8e-2 m (h=16) down to ~1.9e-5 m (h=2), keeping every point
  // >= 2.5 orders above the ~6e-8 m roundoff floor (sqrt(N)*ulp(|r|)) --
  // the documented plateau-avoidance choice.
  const ReferenceOrbit orb = load_reference_orbit();
  const std::vector<double> ladder = {16.0, 8.0, 4.0, 2.0};
  const auto pts = star::testsupport::kepler_convergence(
      orb.mu, orb.r0, orb.v0, kSpanS, star::events::Method::kRk4, ladder);
  for (const auto& p : pts) {
    CAPTURE(p.h_s);
    CAPTURE(p.err_m);
    CHECK(p.err_m > 1e-6);  // still truncation-dominated at the fine end
  }
  const double slope = star::testsupport::fitted_loglog_slope(pts);
  CAPTURE(slope);
  CHECK(slope > 3.8);
  CHECK(slope < 4.2);
}

TEST_CASE("rkf78_fixed_kepler_convergence_slope") {
  // Phase 2 exit criterion 3 (RKF7(8) half): fixed-step mode, fitted slope
  // >= 7.5. With local extrapolation the propagated solution is the
  // 8th-order combination, so the expected slope is ~8. Ladder h = 512..64 s
  // keeps truncation between ~50 m and ~3e-6 m, again well above the
  // roundoff floor.
  const ReferenceOrbit orb = load_reference_orbit();
  const std::vector<double> ladder = {512.0, 256.0, 128.0, 64.0};
  const auto pts = star::testsupport::kepler_convergence(
      orb.mu, orb.r0, orb.v0, kSpanS, star::events::Method::kRkf78, ladder);
  for (const auto& p : pts) {
    CAPTURE(p.h_s);
    CAPTURE(p.err_m);
    CHECK(p.err_m > 1e-7);  // above the roundoff plateau
  }
  const double slope = star::testsupport::fitted_loglog_slope(pts);
  CAPTURE(slope);
  CHECK(slope >= 7.5);
  CHECK(slope < 9.5);  // sanity: a slope far above 8 would mean a broken ladder
}

TEST_CASE("rkf78_adaptive_energy_momentum_drift") {
  // Phase 2 exit criterion 4: specific orbital energy and |h| drift
  // < 1e-10 relative over 10 orbits of the reference eccentric orbit at
  // adaptive tolerance 1e-12 (relative; absolute floors are set orders
  // below rtol*|state| so the relative tolerance governs).
  const ReferenceOrbit orb = load_reference_orbit();
  const auto drift = star::testsupport::twobody_drift(
      orb.mu, orb.r0, orb.v0, /*n_orbits=*/10.0, /*rtol=*/1e-12,
      /*atol_pos_m=*/1e-6, /*atol_vel_mps=*/1e-9, /*h_init=*/10.0,
      /*h_max=*/600.0);
  CAPTURE(drift.max_energy_rel);
  CAPTURE(drift.max_hmag_rel);
  CAPTURE(drift.steps_accepted);
  CAPTURE(drift.steps_rejected);
  CHECK(drift.max_energy_rel < 1e-10);
  CHECK(drift.max_hmag_rel < 1e-10);
  CHECK(drift.steps_accepted > 100);  // sanity: the controller actually ran
}

TEST_CASE("dense_output_hermite_midstep_accuracy") {
  // Dense-output validation: the cubic Hermite interpolant's midstep
  // position error must be consistent with its documented order -- error
  // bound (h^4/384) max|r''''| (derived in ch:integrators) and fourth-order
  // step-size scaling. The trajectory comes from fixed-step RKF7(8), whose
  // O(h^8) global error (~3e-6 m at h = 64 s) is negligible next to the
  // interpolant error measured here (~0.5 m), so the measurement isolates
  // the interpolant. max|r''''| is the committed golden value (numerically
  // differentiated analytic solution; see manifest.toml).
  const ReferenceOrbit orb = load_reference_orbit();
  const double err64 = star::testsupport::hermite_midstep_max_err(
      orb.mu, orb.r0, orb.v0, kSpanS, 64.0);
  const double err32 = star::testsupport::hermite_midstep_max_err(
      orb.mu, orb.r0, orb.v0, kSpanS, 32.0);
  const double bound64 = std::pow(64.0, 4) / 384.0 * orb.max_r4;
  const double bound32 = std::pow(32.0, 4) / 384.0 * orb.max_r4;
  CAPTURE(err64);
  CAPTURE(err32);
  CAPTURE(bound64);
  CAPTURE(bound32);
  // Within the a-priori bound (1.5x headroom for the bound's ~4-digit
  // numerical differentiation) and not vacuously small (>= 0.15x bound:
  // the max over a period must approach the periapsis-region bound).
  CHECK(err64 < 1.5 * bound64);
  CHECK(err64 > 0.15 * bound64);
  CHECK(err32 < 1.5 * bound32);
  CHECK(err32 > 0.15 * bound32);
  // Fourth-order scaling: halving h divides the error by ~16. The band
  // [10, 24] tolerates the max relocating between steps across the two
  // grids while excluding any other integer order.
  const double ratio = err64 / err32;
  CAPTURE(ratio);
  CHECK(ratio > 10.0);
  CHECK(ratio < 24.0);
}

TEST_CASE("rkf78_adaptive_double_run_bitwise") {
  // D-10 determinism at the integrator level: two identical adaptive runs
  // (controller history, rejections, dense endpoints and all) must produce
  // bit-identical final states. This is the same-binary/same-platform gate;
  // cross-platform spread is measured elsewhere (exit criterion 8).
  const ReferenceOrbit orb = load_reference_orbit();
  Eigen::Matrix<double, 6, 1> yf[2];
  for (int run = 0; run < 2; ++run) {
    auto rhs = [&orb](double t, const double* y, double* ydot) {
      star::models::twobody_rhs(orb.mu, t, y, ydot);
    };
    const star::integrate::RhsRef f(rhs);
    star::events::PropagateOptions opt;
    opt.method = star::events::Method::kRkf78;
    opt.mode = star::events::StepMode::kAdaptive;
    opt.adaptive.groups = star::testsupport::twobody_groups(1e-12, 1e-6, 1e-9);
    opt.adaptive.h_init = 10.0;
    opt.adaptive.h_max = 600.0;
    double y0[6] = {orb.r0[0], orb.r0[1], orb.r0[2],
                    orb.v0[0], orb.v0[1], orb.v0[2]};
    star::events::propagate(f, 0.0, 3.0 * orb.period, y0, 6, yf[run].data(),
                            opt, nullptr, 0, nullptr);
  }
  for (int i = 0; i < 6; ++i) {
    CAPTURE(i);
    CHECK(yf[0][i] == yf[1][i]);  // bitwise, no tolerance (D-10)
  }
}
