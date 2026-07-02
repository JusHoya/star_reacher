// Event-detection framework acceptance suite (FR-12; Phase 2 exit
// criterion 5). Test IDs are cited by the math-library validation table
// (ch:events); do not rename them. Analytic apsis times come from the
// committed goldens in tests/golden/integrators/ (provenance in
// manifest.toml there).
#include <cmath>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "golden_io.hpp"
#include "star/constants.hpp"
#include "star/events.hpp"
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
};

ReferenceOrbit load_reference_orbit() {
  const auto cases = star_tests::load_golden_cases(
      std::string(STAR_GOLDEN_DIR) + "/integrators/kepler_orbit.toml");
  const star_tests::GoldenCase& def = find_case(cases, "definition");
  return ReferenceOrbit{star_tests::parse_hex_double(def.scalar("mu_m3ps2")),
                        vec3(def, "r0_m"), vec3(def, "v0_mps")};
}

}  // namespace

TEST_CASE("brent_root_known_functions") {
  // Brent's method (Brent 1973, ch. 4) against functions with known roots.
  // Acceptance: located root within tol_x + 4*eps*|root| of the true root
  // (the documented bracket-width guarantee) and iteration counts far below
  // the cap (superlinear convergence on smooth roots).
  using star::events::brent_root;
  using star::events::BrentResult;

  {  // quadratic root at exactly 2
    auto fn = [](double x) { return x * x - 4.0; };
    const star::events::ScalarFnRef f(fn);
    const BrentResult r = brent_root(f, 0.0, 10.0, fn(0.0), fn(10.0), 1e-12);
    CAPTURE(r.root);
    CAPTURE(r.iterations);
    CHECK(std::fabs(r.root - 2.0) < 1e-12 + 4e-16 * 2.0);
    CHECK(r.iterations < 30);
  }
  {  // transcendental root at pi/2
    auto fn = [](double x) { return std::cos(x); };
    const star::events::ScalarFnRef f(fn);
    const BrentResult r = brent_root(f, 1.0, 2.0, fn(1.0), fn(2.0), 1e-12);
    const double half_pi = 0.25 * star::constants::TWO_PI;
    CAPTURE(r.root);
    CHECK(std::fabs(r.root - half_pi) < 1e-12 + 4e-16 * half_pi);
    CHECK(r.iterations < 30);
  }
  {  // triple root (odd multiplicity, flat): bisection fallback must hold
    auto fn = [](double x) {
      const double d = x - 0.75;
      return d * d * d;
    };
    const star::events::ScalarFnRef f(fn);
    const BrentResult r = brent_root(f, 0.0, 1.0, fn(0.0), fn(1.0), 1e-9);
    CAPTURE(r.root);
    CAPTURE(r.iterations);
    CHECK(std::fabs(r.root - 0.75) < 1e-9);
    CHECK(r.iterations < 128);
  }
  {  // exact-zero endpoint is returned immediately
    auto fn = [](double x) { return x; };
    const star::events::ScalarFnRef f(fn);
    const BrentResult r = brent_root(f, 0.0, 1.0, 0.0, 1.0, 1e-12);
    CHECK(r.root == 0.0);
    CHECK(r.iterations == 0);
  }
  {  // non-bracketing endpoints are rejected loudly
    auto fn = [](double x) { return x * x + 1.0; };
    const star::events::ScalarFnRef f(fn);
    CHECK_THROWS_AS(brent_root(f, 0.0, 1.0, fn(0.0), fn(1.0), 1e-12),
                    std::invalid_argument);
  }
}

TEST_CASE("timer_event_location") {
  // Timer events g = t - t_event are linear in t and independent of the
  // interpolated state, so the located root isolates the root-finding and
  // restart machinery: acceptance is the FR-12 location tolerance (1e-9 s)
  // exactly. Also exercises a terminal event: propagation must stop at the
  // located root, not at tf.
  const ReferenceOrbit orb = load_reference_orbit();
  auto rhs = [&orb](double t, const double* y, double* ydot) {
    star::models::twobody_rhs(orb.mu, t, y, ydot);
  };
  const star::integrate::RhsRef f(rhs);

  star::events::TimerEvent mark{3000.0};
  star::events::TimerEvent stop{3500.0};
  star::events::EventSpec specs[2] = {
      {"mark", star::events::EventFnRef(mark),
       star::events::Direction::kIncreasing, {}, false},
      {"stop", star::events::EventFnRef(stop),
       star::events::Direction::kIncreasing, {}, true},
  };

  star::events::PropagateOptions opt;
  opt.method = star::events::Method::kRkf78;
  opt.mode = star::events::StepMode::kAdaptive;
  opt.adaptive.groups = star::testsupport::twobody_groups(1e-12, 1e-6, 1e-9);
  opt.adaptive.h_init = 10.0;
  opt.adaptive.h_max = 600.0;

  double y0[6] = {orb.r0[0], orb.r0[1], orb.r0[2],
                  orb.v0[0], orb.v0[1], orb.v0[2]};
  double yf[6];
  std::vector<star::events::EventRecord> log;
  const auto res = star::events::propagate(f, 0.0, 7000.0, y0, 6, yf, opt,
                                           specs, 2, &log);

  REQUIRE(log.size() == 2);
  CAPTURE(log[0].t);
  CAPTURE(log[1].t);
  CHECK(log[0].event_index == 0);
  CHECK(std::fabs(log[0].t - 3000.0) <= 1e-9);
  CHECK(log[1].event_index == 1);
  CHECK(std::fabs(log[1].t - 3500.0) <= 1e-9);
  CHECK(res.terminated_by_event);
  CHECK(res.terminal_event_index == 1);
  CHECK(res.t_final == log[1].t);
}

TEST_CASE("event_restart_discrete_update") {
  // The restart-with-discrete-update mechanism (what staging/SOI events use
  // later): a timer event applies an impulsive +50 m/s along-velocity kick;
  // the final state must match the two-segment analytic composition
  // (Kepler to the event, kick, Kepler to the end). Tolerance: adaptive
  // tolerance 1e-12 accumulates ~1e-4 m of trajectory error over the span,
  // and the 1e-9 s event-time error contributes |dv|*tol ~ 5e-8 m; 2e-3 m
  // gives an order of margin while a broken restart (kick not applied, or
  // applied at interpolant accuracy on a coarse step) fails by meters or
  // kilometers.
  const ReferenceOrbit orb = load_reference_orbit();
  const double t_kick = 3000.0;
  const double t_end = 6400.0;
  const double dv = 50.0;

  auto rhs = [&orb](double t, const double* y, double* ydot) {
    star::models::twobody_rhs(orb.mu, t, y, ydot);
  };
  const star::integrate::RhsRef f(rhs);

  star::events::TimerEvent kick_timer{t_kick};
  int hook_calls = 0;
  auto kick = [dv, &hook_calls](double /*t*/, double* y) {
    Eigen::Map<Eigen::Vector3d> v(y + 3);
    const Eigen::Vector3d vhat = v.normalized();
    v += dv * vhat;
    ++hook_calls;
  };
  star::events::EventSpec specs[1] = {
      {"kick", star::events::EventFnRef(kick_timer),
       star::events::Direction::kIncreasing,
       star::events::UpdateFnRef(kick), false},
  };

  star::events::PropagateOptions opt;
  opt.method = star::events::Method::kRkf78;
  opt.mode = star::events::StepMode::kAdaptive;
  opt.adaptive.groups = star::testsupport::twobody_groups(1e-12, 1e-6, 1e-9);
  opt.adaptive.h_init = 10.0;
  opt.adaptive.h_max = 600.0;

  double y0[6] = {orb.r0[0], orb.r0[1], orb.r0[2],
                  orb.v0[0], orb.v0[1], orb.v0[2]};
  double yf[6];
  std::vector<star::events::EventRecord> log;
  star::events::propagate(f, 0.0, t_end, y0, 6, yf, opt, specs, 1, &log);

  REQUIRE(log.size() == 1);  // exactly one kick: no duplicate re-fire
  CHECK(hook_calls == 1);
  CHECK(std::fabs(log[0].t - t_kick) <= 1e-9);

  // Analytic composition of the two Keplerian arcs around the kick.
  const star::testsupport::EllipticOrbitRef arc1(orb.mu, orb.r0, orb.v0);
  Eigen::Vector3d r1, v1;
  arc1.state_at(t_kick, &r1, &v1);
  v1 += dv * v1.normalized();
  const star::testsupport::EllipticOrbitRef arc2(orb.mu, r1, v1);
  Eigen::Vector3d r2, v2;
  arc2.state_at(t_end - t_kick, &r2, &v2);

  const Eigen::Map<const Eigen::Vector3d> rf(yf);
  const Eigen::Map<const Eigen::Vector3d> vf(yf + 3);
  const double dr = (rf - r2).norm();
  const double dvel = (vf - v2).norm();
  CAPTURE(dr);
  CAPTURE(dvel);
  CHECK(dr < 2e-3);
  CHECK(dvel < 2e-6);
}

TEST_CASE("apsis_event_times_analytic") {
  // Phase 2 exit criterion 5: periapsis and apoapsis passage times located
  // by the event framework (g = r.v with direction filters, Brent on dense
  // output) within 1 microsecond of the analytic times, over 3.2 orbits of
  // the eccentric reference orbit. h_max = 5 s caps the O(h^4) dense-output
  // contribution to the located times at ~5e-11 s (error law in ch:events),
  // so the budget is dominated by the 1e-9 s root tolerance -- three orders
  // inside the criterion.
  const ReferenceOrbit orb = load_reference_orbit();
  const auto cases = star_tests::load_golden_cases(
      std::string(STAR_GOLDEN_DIR) + "/integrators/apsis_times.toml");
  const double t_end =
      star_tests::parse_hex_double(find_case(cases, "span").scalar("t_end_s"));

  const auto scan = star::testsupport::apsis_event_scan(
      orb.mu, orb.r0, orb.v0, t_end, /*rtol=*/1e-12, /*atol_pos_m=*/1e-6,
      /*atol_vel_mps=*/1e-9, /*h_init=*/5.0, /*h_max=*/5.0,
      /*event_tol_s=*/1e-9);

  std::vector<std::pair<double, bool>> expected;  // (t, is_periapsis)
  for (const star_tests::GoldenCase& c : cases) {
    if (c.scalar("name").rfind("apsis_", 0) != 0) {
      continue;
    }
    expected.emplace_back(star_tests::parse_hex_double(c.scalar("t_s")),
                          c.scalar("kind") == "periapsis");
  }
  REQUIRE(expected.size() == 6);
  REQUIRE(scan.hits.size() == expected.size());

  double worst_us = 0.0;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    const double err_s = std::fabs(scan.hits[i].t_s - expected[i].first);
    CAPTURE(i);
    CAPTURE(scan.hits[i].t_s);
    CAPTURE(expected[i].first);
    CHECK(scan.hits[i].periapsis == expected[i].second);
    CHECK(err_s < 1e-6);  // the exit-criterion bound
    worst_us = std::fmax(worst_us, err_s * 1e6);
  }
  MESSAGE("worst apsis-event time error: " << worst_us << " us over "
                                           << expected.size() << " apsides");
}
