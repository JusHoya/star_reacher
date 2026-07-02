// Two-body model tests: analytic circular-orbit benchmark (FR-22 layer 3)
// and energy/angular-momentum property invariants (FR-22 layer 2). Test IDs
// are cited by the math-library validation table; do not rename them.
#include <cmath>
#include <cstdint>

#include <Eigen/Dense>

#include "star/constants.hpp"
#include "star/models/twobody.hpp"
#include "vendor/doctest.h"

namespace {

constexpr double kMu = star::constants::GM_EARTH_M3_PER_S2;

star::models::TwoBodyState circular_state(double a_m) {
  // Circular orbit in the xy-plane: r = a x_hat, v = sqrt(mu/a) y_hat
  // (Vallado, two-body relative motion: circular speed v = sqrt(mu/r)).
  star::models::TwoBodyState s;
  s.r_m = Eigen::Vector3d(a_m, 0.0, 0.0);
  s.v_mps = Eigen::Vector3d(0.0, std::sqrt(kMu / a_m), 0.0);
  return s;
}

}  // namespace

TEST_CASE("twobody_circular_orbit_analytic") {
  // Benchmark: a = 6778137 m (LEO, ~400 km altitude), period
  // T = 2*pi*sqrt(a^3/mu) ~ 5553.6 s, dt = 0.1 s. Propagate
  // N = round(T/dt) steps and compare against the closed-form circular
  // solution evaluated at exactly t = N*dt:
  //   r(t) = a*(cos(n t), sin(n t), 0),  v(t) = v_c*(-sin(n t), cos(n t), 0)
  // with mean motion n = sqrt(mu/a^3). Comparing at N*dt (not at T) removes
  // the up-to-dt/2 time-quantization offset, which would otherwise alias into
  // ~383 m of along-track displacement (v_c ~ 7668.6 m/s) and swamp the
  // integrator error being measured.
  //
  // Tolerance derivation (recorded per FR-22): RK4 local truncation error is
  // ~(dt^5/120)*|d5y/dt5| with |d5r/dt5| = a*n^5 ~ 6.78e6*(1.1313e-3)^5
  // ~ 1.3e-8 m/s^5, i.e. ~1e-15 m per step; accumulated over N ~ 55536 steps
  // with secular along-track growth this stays below ~1e-8 m, and
  // floating-point round-off random-walk adds ~sqrt(N)*ulp(a) ~ 2e-7 m.
  // Measured error on this machine (MSVC x64, /fp:strict): 2.8e-7 m,
  // i.e. round-off-dominated, consistent with the estimate. The contract
  // bound of 1 m therefore carries ~6 orders of margin and fails loudly on
  // any structural defect (a wrong RK4 stage coefficient produces >= 1e2 m
  // here).
  const double a = 6778137.0;
  const double n = std::sqrt(kMu / (a * a * a));
  const double period_s = star::constants::TWO_PI / n;
  const double dt = 0.1;
  const std::int64_t steps = std::llround(period_s / dt);

  star::models::TwoBodyState s = circular_state(a);
  for (std::int64_t i = 0; i < steps; ++i) {
    s = star::models::rk4_step(kMu, s, dt);
  }

  const double t_end = static_cast<double>(steps) * dt;
  const Eigen::Vector3d r_analytic(a * std::cos(n * t_end),
                                   a * std::sin(n * t_end), 0.0);
  const double err_m = (s.r_m - r_analytic).norm();
  CAPTURE(err_m);
  CHECK(err_m < 1.0);  // contract bound; see derivation above
}

TEST_CASE("twobody_energy_momentum_drift") {
  // Property invariants (FR-22 layer 2): specific orbital energy
  // eps = v^2/2 - mu/r and specific angular momentum magnitude |h| = |r x v|
  // are exact constants of two-body motion (Vallado, conservation of energy
  // and angular momentum). RK4 does not conserve them exactly; the drift over
  // a fixed span bounds the integrator's dissipation.
  //
  // Span/step: 3 orbits at dt = 1.0 s (n*dt ~ 1.13e-3 rad). RK4 truncation
  // drift scales as (n*dt)^4 per orbit; a circular orbit sits at the scheme's
  // benign point, so the measured maxima on this machine (MSVC x64,
  // /fp:strict) are round-off-dominated: energy 1.1e-14, |h| 5.2e-15 over
  // the whole span. Bounds are set ~3 orders above the measurement so
  // cross-platform libm/round-off spread cannot flake the test, while a
  // stage-coefficient or force-model defect (drift >= 1e-8) still fails
  // decisively.
  const double a = 6778137.0;
  const double dt = 1.0;
  const double n = std::sqrt(kMu / (a * a * a));
  const double period_s = star::constants::TWO_PI / n;
  const std::int64_t steps = std::llround(3.0 * period_s / dt);

  star::models::TwoBodyState s = circular_state(a);
  const double eps0 = 0.5 * s.v_mps.squaredNorm() - kMu / s.r_m.norm();
  const double h0 = s.r_m.cross(s.v_mps).norm();

  double max_eps_rel = 0.0;
  double max_h_rel = 0.0;
  for (std::int64_t i = 0; i < steps; ++i) {
    s = star::models::rk4_step(kMu, s, dt);
    const double eps = 0.5 * s.v_mps.squaredNorm() - kMu / s.r_m.norm();
    const double h = s.r_m.cross(s.v_mps).norm();
    max_eps_rel = std::max(max_eps_rel, std::fabs((eps - eps0) / eps0));
    max_h_rel = std::max(max_h_rel, std::fabs((h - h0) / h0));
  }

  CAPTURE(max_eps_rel);
  CAPTURE(max_h_rel);
  CHECK(max_eps_rel < 1e-11);  // ~1000x measured 1.1e-14; see derivation above
  CHECK(max_h_rel < 1e-11);    // ~2000x measured 5.2e-15; see derivation above
}
