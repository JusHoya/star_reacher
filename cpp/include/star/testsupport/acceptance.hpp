// Acceptance-measurement drivers for the Phase 2 integrator/event exit
// criteria -- TEST SUPPORT ONLY.
//
// One implementation of each headline measurement (convergence slope,
// invariant drift, apsis event scan, dense-output midstep error), shared by
// the doctest suites and the thin pybind11 test-support entry points so the
// C++ and Python evidence are the same numbers by construction. Header-only;
// never part of the deterministic mission run path.
//
// Time-accounting design for the convergence and dense-output measurements:
// spans and step sizes are restricted to dyadic values (t_end an integer
// number of seconds, h a power of two), so every step time k*h is exactly
// representable and accumulated time carries zero rounding error. This
// isolates integrator truncation error from time-base roundoff; the chosen
// ladders keep the smallest truncation error >= 4 orders of magnitude above
// the position roundoff floor (~sqrt(N)*ulp(|r|), a few 1e-8 m here), which
// is how the measured slopes avoid the double-precision plateau (documented
// in ch:integrators).
#ifndef STAR_TESTSUPPORT_ACCEPTANCE_HPP
#define STAR_TESTSUPPORT_ACCEPTANCE_HPP

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>

#include "star/events.hpp"
#include "star/integrate.hpp"
#include "star/models/twobody.hpp"
#include "star/testsupport/kepler_ref.hpp"

namespace star {
namespace testsupport {

// Default state groups for the two-body state y = [r(3), v(3)]: absolute
// tolerances are set far below rtol*|state| at orbital scales (|r| ~ 1e7 m,
// |v| ~ 1e3..1e4 m/s), so the relative tolerance governs -- matching the
// "adaptive tol 1e-12" phrasing of the Phase 2 exit criteria.
inline std::vector<integrate::StateGroup> twobody_groups(double rtol,
                                                         double atol_pos_m,
                                                         double atol_vel_mps) {
  return {
      integrate::StateGroup{"position", 0, 3, rtol, atol_pos_m},
      integrate::StateGroup{"velocity", 3, 3, rtol, atol_vel_mps},
  };
}

struct ConvergencePoint {
  double h_s = 0.0;
  double err_m = 0.0;  // |r_numeric(t_end) - r_analytic(t_end)|
};

// Fixed-step global-error ladder on the reference Kepler problem: propagate
// y' = f(y) with constant step h over [0, t_end] and measure the final
// position error against the closed-form solution. Method selects RK4 or
// RKF7(8)-in-fixed-step (the mode required to measure the high-order slope).
// t_end must be an exact multiple of every h in the ladder (throws
// otherwise): exact step counts are what make the measurement clean.
inline std::vector<ConvergencePoint> kepler_convergence(
    double mu, const Eigen::Vector3d& r0, const Eigen::Vector3d& v0,
    double t_end, events::Method method, const std::vector<double>& ladder) {
  const EllipticOrbitRef orbit(mu, r0, v0);
  Eigen::Vector3d r_true, v_true;
  orbit.state_at(t_end, &r_true, &v_true);

  auto rhs = [mu](double t, const double* y, double* ydot) {
    models::twobody_rhs(mu, t, y, ydot);
  };
  const integrate::RhsRef f(rhs);

  std::vector<ConvergencePoint> out;
  out.reserve(ladder.size());
  integrate::Rk4 rk4(6);
  integrate::Rkf78 rkf78(6);
  double y[6];
  double errv[6];
  for (const double h : ladder) {
    const double steps_exact = t_end / h;
    const auto steps = static_cast<std::int64_t>(std::llround(steps_exact));
    if (steps <= 0 ||
        steps_exact != static_cast<double>(steps)) {  // exact divisibility
      throw std::invalid_argument(
          "kepler_convergence: t_end must be an exact multiple of h");
    }
    for (int i = 0; i < 3; ++i) {
      y[i] = r0[i];
      y[i + 3] = v0[i];
    }
    for (std::int64_t i = 0; i < steps; ++i) {
      const double t = static_cast<double>(i) * h;  // exact for dyadic h
      if (method == events::Method::kRk4) {
        rk4.step(f, t, y, h, y);
      } else {
        rkf78.step(f, t, y, h, y, errv);
      }
    }
    const Eigen::Map<const Eigen::Vector3d> r_num(y);
    out.push_back(ConvergencePoint{h, (r_num - r_true).norm()});
  }
  return out;
}

// Least-squares slope of log(err) vs log(h) -- the measured order.
inline double fitted_loglog_slope(const std::vector<ConvergencePoint>& pts) {
  if (pts.size() < 2) {
    throw std::invalid_argument("fitted_loglog_slope: need >= 2 points");
  }
  double sx = 0.0, sy = 0.0;
  for (const ConvergencePoint& p : pts) {
    sx += std::log(p.h_s);
    sy += std::log(p.err_m);
  }
  const double mx = sx / static_cast<double>(pts.size());
  const double my = sy / static_cast<double>(pts.size());
  double sxx = 0.0, sxy = 0.0;
  for (const ConvergencePoint& p : pts) {
    const double dx = std::log(p.h_s) - mx;
    sxy += dx * (std::log(p.err_m) - my);
    sxx += dx * dx;
  }
  return sxy / sxx;
}

struct DriftResult {
  double max_energy_rel = 0.0;
  double max_hmag_rel = 0.0;
  std::int64_t steps_accepted = 0;
  std::int64_t steps_rejected = 0;
};

// Conserved-quantity drift over n_orbits of the reference orbit under
// adaptive RKF7(8): specific orbital energy eps = v^2/2 - mu/r and specific
// angular momentum magnitude |h| = |r x v| are exact invariants of two-body
// motion, so their worst relative deviation at accepted-step endpoints
// bounds the integrator's dissipation (Phase 2 exit criterion 4).
inline DriftResult twobody_drift(double mu, const Eigen::Vector3d& r0,
                                 const Eigen::Vector3d& v0, double n_orbits,
                                 double rtol, double atol_pos_m,
                                 double atol_vel_mps, double h_init,
                                 double h_max) {
  const EllipticOrbitRef orbit(mu, r0, v0);
  const double eps0 = 0.5 * v0.squaredNorm() - mu / r0.norm();
  const double h0 = r0.cross(v0).norm();

  auto rhs = [mu](double t, const double* y, double* ydot) {
    models::twobody_rhs(mu, t, y, ydot);
  };
  const integrate::RhsRef f(rhs);

  DriftResult res;
  auto observer = [&](const integrate::DenseStep& d) {
    const Eigen::Map<const Eigen::Vector3d> r(d.y1);
    const Eigen::Map<const Eigen::Vector3d> v(d.y1 + 3);
    const double eps = 0.5 * v.squaredNorm() - mu / r.norm();
    const double hm = r.cross(v).norm();
    res.max_energy_rel =
        std::fmax(res.max_energy_rel, std::fabs((eps - eps0) / eps0));
    res.max_hmag_rel = std::fmax(res.max_hmag_rel, std::fabs((hm - h0) / h0));
  };

  events::PropagateOptions opt;
  opt.method = events::Method::kRkf78;
  opt.mode = events::StepMode::kAdaptive;
  opt.adaptive.groups = twobody_groups(rtol, atol_pos_m, atol_vel_mps);
  opt.adaptive.h_init = h_init;
  opt.adaptive.h_max = h_max;

  double y0[6];
  double yf[6];
  for (int i = 0; i < 3; ++i) {
    y0[i] = r0[i];
    y0[i + 3] = v0[i];
  }
  events::StepObserverRef obs(observer);
  const events::PropagateResult pr =
      events::propagate(f, 0.0, n_orbits * orbit.period, y0, 6, yf, opt,
                        nullptr, 0, nullptr, obs);
  res.steps_accepted = pr.steps_accepted;
  res.steps_rejected = pr.steps_rejected;
  return res;
}

struct ApsisHit {
  double t_s = 0.0;
  bool periapsis = false;
};

struct ApsisScanResult {
  std::vector<ApsisHit> hits;
  std::int64_t steps_accepted = 0;
};

// Locate every apsis passage in (0, t_end] with the event framework:
// g = r.v screened as an increasing crossing for periapsis and a decreasing
// crossing for apoapsis (Phase 2 exit criterion 5). h_max caps the step so
// the O(h^4) dense-output error keeps the located times inside the
// microsecond budget (error law derived in ch:events).
inline ApsisScanResult apsis_event_scan(double mu, const Eigen::Vector3d& r0,
                                        const Eigen::Vector3d& v0,
                                        double t_end, double rtol,
                                        double atol_pos_m, double atol_vel_mps,
                                        double h_init, double h_max,
                                        double event_tol_s) {
  auto rhs = [mu](double t, const double* y, double* ydot) {
    models::twobody_rhs(mu, t, y, ydot);
  };
  const integrate::RhsRef f(rhs);

  events::ApsisEvent apsis_fn;  // g = r.v on the [r, v] state layout
  events::EventSpec specs[2] = {
      {"periapsis", events::EventFnRef(apsis_fn),
       events::Direction::kIncreasing, {}, false},
      {"apoapsis", events::EventFnRef(apsis_fn),
       events::Direction::kDecreasing, {}, false},
  };

  events::PropagateOptions opt;
  opt.method = events::Method::kRkf78;
  opt.mode = events::StepMode::kAdaptive;
  opt.adaptive.groups = twobody_groups(rtol, atol_pos_m, atol_vel_mps);
  opt.adaptive.h_init = h_init;
  opt.adaptive.h_max = h_max;
  opt.event_time_tol_s = event_tol_s;

  double y0[6];
  double yf[6];
  for (int i = 0; i < 3; ++i) {
    y0[i] = r0[i];
    y0[i + 3] = v0[i];
  }
  std::vector<events::EventRecord> log;
  log.reserve(64);
  const events::PropagateResult pr =
      events::propagate(f, 0.0, t_end, y0, 6, yf, opt, specs, 2, &log);

  ApsisScanResult res;
  res.steps_accepted = pr.steps_accepted;
  res.hits.reserve(log.size());
  for (const events::EventRecord& rec : log) {
    res.hits.push_back(ApsisHit{rec.t, rec.event_index == 0});
  }
  return res;
}

// Worst-case position error of the cubic Hermite dense output at step
// midpoints, against the analytic solution, for a fixed-step RKF7(8)
// propagation: at 8th order the trajectory error is negligible next to the
// O(h^4) interpolant error, so this isolates the quantity the dense-output
// error bound (h^4/384) max|r''''| describes. t_end and h must satisfy the
// same dyadic-divisibility rule as kepler_convergence.
inline double hermite_midstep_max_err(double mu, const Eigen::Vector3d& r0,
                                      const Eigen::Vector3d& v0, double t_end,
                                      double h) {
  const EllipticOrbitRef orbit(mu, r0, v0);
  auto rhs = [mu](double t, const double* y, double* ydot) {
    models::twobody_rhs(mu, t, y, ydot);
  };
  const integrate::RhsRef f(rhs);

  const double steps_exact = t_end / h;
  const auto steps = static_cast<std::int64_t>(std::llround(steps_exact));
  if (steps <= 0 || steps_exact != static_cast<double>(steps)) {
    throw std::invalid_argument(
        "hermite_midstep_max_err: t_end must be an exact multiple of h");
  }

  integrate::Rkf78 rkf78(6);
  double y[6], ynew[6], f0[6], f1[6], errv[6], umid[6];
  for (int i = 0; i < 3; ++i) {
    y[i] = r0[i];
    y[i + 3] = v0[i];
  }
  double max_err = 0.0;
  for (std::int64_t i = 0; i < steps; ++i) {
    const double t = static_cast<double>(i) * h;  // exact for dyadic h
    f(t, y, f0);
    rkf78.step(f, t, y, h, ynew, errv, f0);
    f(t + h, ynew, f1);

    integrate::DenseStep dense{t, h, y, f0, ynew, f1, 6};
    const double t_mid = t + 0.5 * h;  // exact for dyadic h
    dense.eval(t_mid, umid);

    Eigen::Vector3d r_true, v_true;
    orbit.state_at(t_mid, &r_true, &v_true);
    const Eigen::Map<const Eigen::Vector3d> r_interp(umid);
    max_err = std::fmax(max_err, (r_interp - r_true).norm());

    for (int j = 0; j < 6; ++j) {
      y[j] = ynew[j];
    }
  }
  return max_err;
}

}  // namespace testsupport
}  // namespace star

#endif  // STAR_TESTSUPPORT_ACCEPTANCE_HPP
