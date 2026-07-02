// Event detection and event-aware propagation (FR-12). This file is the
// chapter-manifest anchor for the events chapter (docs/mathlib chapter
// ch:events); the equation labels referenced below are defined there.
//
// Sources: Brent's method per Brent, Algorithms for Minimization without
// Derivatives (1973), ch. 4; the screening/relocation architecture follows
// the standard practice for ODE event location on dense output (see the
// chapter for the error analysis).
#include "star/events.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace star {
namespace events {

// ---------------------------------------------------------------------------
// Brent root finder
// ---------------------------------------------------------------------------

BrentResult brent_root(const ScalarFnRef& f, double a, double b, double fa,
                       double fb, double tol_x, int max_iter) {
  if (!(tol_x > 0.0)) {
    throw std::invalid_argument("brent_root: tol_x must be > 0");
  }
  // Exact-zero endpoints are already roots; handle before the sign test so
  // callers can pass a bracket that closed on a node.
  if (fa == 0.0) {
    return BrentResult{a, 0.0, a, 0.0, 0};
  }
  if (fb == 0.0) {
    return BrentResult{b, 0.0, b, 0.0, 0};
  }
  if ((fa > 0.0) == (fb > 0.0)) {
    throw std::invalid_argument(
        "brent_root: endpoints do not bracket a sign change");
  }

  // Brent (1973), ch. 4: the zero-finding algorithm combining bisection,
  // secant, and inverse quadratic interpolation, with the interpolation
  // acceptance tests that guarantee convergence in at most O(log^2) function
  // evaluations while retaining superlinear convergence on smooth roots.
  // Variable roles follow the published algorithm: b is the current best
  // iterate, a the previous one, c the counterpoint keeping [b, c] a
  // bracket; d is the proposed step, e the one before it.
  constexpr double kEps = std::numeric_limits<double>::epsilon();
  double c = a;
  double fc = fa;
  double d = b - a;
  double e = d;

  int iter = 0;
  for (; iter < max_iter; ++iter) {
    if (std::fabs(fc) < std::fabs(fb)) {
      a = b;
      b = c;
      c = a;
      fa = fb;
      fb = fc;
      fc = fa;
    }
    // Convergence tolerance: tol_x/2 plus the resolution of double at the
    // iterate's magnitude, so tiny tol_x degrades gracefully to machine
    // precision instead of looping forever.
    const double tol = 2.0 * kEps * std::fabs(b) + 0.5 * tol_x;
    const double m = 0.5 * (c - b);
    if (std::fabs(m) <= tol || fb == 0.0) {
      break;
    }
    if (std::fabs(e) < tol || std::fabs(fa) <= std::fabs(fb)) {
      // No usable interpolation history: bisect.
      d = m;
      e = m;
    } else {
      double p;
      double q;
      const double s = fb / fa;
      if (a == c) {
        // Two distinct points: secant (linear interpolation).
        p = 2.0 * m * s;
        q = 1.0 - s;
      } else {
        // Three distinct points: inverse quadratic interpolation.
        const double qq = fa / fc;
        const double r = fb / fc;
        p = s * (2.0 * m * qq * (qq - r) - (b - a) * (r - 1.0));
        q = (qq - 1.0) * (r - 1.0) * (s - 1.0);
      }
      if (p > 0.0) {
        q = -q;
      } else {
        p = -p;
      }
      // Accept the interpolation only if it stays well inside the bracket
      // and shrinks faster than the previous-but-one step; otherwise bisect
      // (Brent's convergence guarantee).
      if (2.0 * p < 3.0 * m * q - std::fabs(tol * q) &&
          p < std::fabs(0.5 * e * q)) {
        e = d;
        d = p / q;
      } else {
        d = m;
        e = m;
      }
    }
    a = b;
    fa = fb;
    if (std::fabs(d) > tol) {
      b += d;
    } else {
      b += (m > 0.0) ? tol : -tol;
    }
    fb = f(b);
    if ((fb > 0.0) == (fc > 0.0)) {
      c = a;
      fc = fa;
      d = b - a;
      e = d;
    }
  }
  return BrentResult{b, fb, c, fc, iter};
}

// ---------------------------------------------------------------------------
// Event-aware propagation driver
// ---------------------------------------------------------------------------

namespace {

// Screening outcome for one event over one accepted step.
struct Crossing {
  bool found = false;
  bool increasing = false;  // direction of the detected sign change
};

// eq:events:screen  A crossing exists in (t0, t1] when g moves from strictly
// one sign to the other sign or to zero, filtered by the event's direction.
// g_prev == 0 never screens as a crossing: the framework restarts exactly at
// located roots, and an event must leave zero before it can fire again.
Crossing screen(double g_prev, double g_now, Direction dir) {
  Crossing c;
  if (g_prev < 0.0 && g_now >= 0.0 &&
      (dir == Direction::kAny || dir == Direction::kIncreasing)) {
    c.found = true;
    c.increasing = true;
  } else if (g_prev > 0.0 && g_now <= 0.0 &&
             (dir == Direction::kAny || dir == Direction::kDecreasing)) {
    c.found = true;
    c.increasing = false;
  }
  return c;
}

void validate_options(const PropagateOptions& opt, std::size_t n,
                      const double* y0) {
  if (opt.mode == StepMode::kFixed) {
    if (!(opt.h_fixed > 0.0)) {
      throw std::invalid_argument("propagate: h_fixed must be > 0");
    }
    return;
  }
  if (opt.method != Method::kRkf78) {
    throw std::invalid_argument(
        "propagate: adaptive mode requires the RKF7(8) method");
  }
  const integrate::AdaptiveOptions& ao = opt.adaptive;
  if (!(ao.h_init > 0.0) || !(ao.h_max > 0.0) || !(ao.h_min > 0.0) ||
      ao.h_min > ao.h_max) {
    throw std::invalid_argument(
        "propagate: adaptive h_init/h_min/h_max must be positive with "
        "h_min <= h_max");
  }
  // Validate the group partition once up front (error_norm re-checks per
  // step, but failing before any integration gives a cleaner contract).
  std::vector<double> zeros(n, 0.0);
  (void)integrate::error_norm(zeros.data(), y0, y0, n, ao.groups);
}

}  // namespace

PropagateResult propagate(const integrate::RhsRef& f, double t0, double tf,
                          const double* y0, std::size_t n, double* y_final,
                          const PropagateOptions& opt, EventSpec* events,
                          std::size_t n_events,
                          std::vector<EventRecord>* event_log,
                          StepObserverRef observer) {
  if (n == 0 || y0 == nullptr || y_final == nullptr) {
    throw std::invalid_argument("propagate: invalid state buffer");
  }
  if (!(tf >= t0)) {
    throw std::invalid_argument("propagate: tf must be >= t0");
  }
  if (n_events > 0 && events == nullptr) {
    throw std::invalid_argument("propagate: null events with n_events > 0");
  }
  if (!(opt.event_time_tol_s > 0.0)) {
    throw std::invalid_argument("propagate: event_time_tol_s must be > 0");
  }
  validate_options(opt, n, y0);

  PropagateResult result;

  // Count every right-hand-side evaluation by routing calls through a
  // counting wrapper; the count is acceptance-suite evidence (stage counts
  // per step are part of the determinism story).
  auto counting_rhs = [&f, &result](double t, const double* y, double* yd) {
    f(t, y, yd);
    result.rhs_evals += 1;
  };
  const integrate::RhsRef cf(counting_rhs);

  const auto ni = static_cast<Eigen::Index>(n);
  Eigen::VectorXd y(ni), ynew(ni), yevt(ni), fa(ni), fb(ni), errv(ni),
      ydense(ni);
  y = Eigen::Map<const Eigen::VectorXd>(y0, ni);

  // Steppers are constructed unconditionally: both are cheap (workspace
  // vectors) and it keeps the step dispatch branch-only.
  integrate::Rk4 rk4(n);
  integrate::Rkf78 rkf78(n);
  // Error-estimate order k = 8: the embedded difference estimates the local
  // error of the 7th-order formula, O(h^8) (see ch:integrators).
  integrate::PiController pi(opt.adaptive.safety, opt.adaptive.fac_min,
                             opt.adaptive.fac_max, 8);
  const bool adaptive = (opt.mode == StepMode::kAdaptive);

  std::vector<double> g_prev(n_events, 0.0);
  std::vector<double> g_now(n_events, 0.0);

  double t = t0;
  cf(t, y.data(), fa.data());
  for (std::size_t i = 0; i < n_events; ++i) {
    g_prev[i] = events[i].g(t, y.data());
  }

  double h = adaptive ? std::fmin(opt.adaptive.h_init, opt.adaptive.h_max)
                      : opt.h_fixed;

  while (t < tf) {
    const bool final_step = (t + h >= tf);
    const double h_use = final_step ? (tf - t) : h;
    if (!final_step && t + h_use == t) {
      // A step below the resolution of double at |t| would loop forever;
      // this can only be reached by a pathologically small fixed step or
      // h_min, so fail loudly instead of spinning.
      throw std::runtime_error(
          "propagate: step size underflow (h below the time resolution of "
          "double at |t|)");
    }

    // --- one step attempt -------------------------------------------------
    double h_next = h;
    if (opt.method == Method::kRk4) {
      rk4.step(cf, t, y.data(), h_use, ynew.data(), fa.data());
    } else {
      rkf78.step(cf, t, y.data(), h_use, ynew.data(), errv.data(), fa.data());
      if (adaptive) {
        const double err = integrate::error_norm(
            errv.data(), y.data(), ynew.data(), n, opt.adaptive.groups);
        if (err > 1.0) {
          if (h_use <= opt.adaptive.h_min) {
            throw std::runtime_error(
                "propagate: adaptive error test failed at h_min; the "
                "requested tolerance is unattainable for these dynamics "
                "(documented out-of-domain response, see ch:integrators)");
          }
          result.steps_rejected += 1;
          h = std::fmax(opt.adaptive.h_min, h_use * pi.factor_reject(err));
          continue;
        }
        h_next = std::fmin(opt.adaptive.h_max,
                           std::fmax(opt.adaptive.h_min,
                                     h_use * pi.factor_accept(err)));
      }
    }

    // --- accepted: dense output over [t, t1] ------------------------------
    const double t1 = final_step ? tf : (t + h_use);
    cf(t1, ynew.data(), fb.data());
    integrate::DenseStep dense;
    dense.t0 = t;
    dense.h = h_use;
    dense.y0 = y.data();
    dense.f0 = fa.data();
    dense.y1 = ynew.data();
    dense.f1 = fb.data();
    dense.n = n;

    // --- event screening and location -------------------------------------
    bool have_root = false;
    double t_root = 0.0;
    std::size_t root_idx = 0;
    bool root_increasing = false;
    for (std::size_t i = 0; i < n_events; ++i) {
      g_now[i] = events[i].g(t1, ynew.data());
      const Crossing c = screen(g_prev[i], g_now[i], events[i].direction);
      if (!c.found) {
        continue;
      }
      // Root of the interpolated event function g(tau, u(tau)) on [t, t1]
      // (eq:events:ghat). The interpolant reproduces the endpoints exactly,
      // so the bracket values are g_prev and g_now verbatim.
      const EventFnRef& gi = events[i].g;
      auto g_on_dense = [&dense, &gi, &ydense](double tq) -> double {
        dense.eval(tq, ydense.data());
        return gi(tq, ydense.data());
      };
      const ScalarFnRef gd(g_on_dense);
      const BrentResult br = brent_root(gd, t, t1, g_prev[i], g_now[i],
                                        opt.event_time_tol_s);
      // Fire at the bracket endpoint on the POST-crossing side (or exactly
      // at a zero), so the restart begins with the event's sign already
      // crossed and a hooked event whose g the hook does not change cannot
      // re-fire on an endpoint that landed an ulp before the root. The
      // side selection costs at most the bracket width (<= the location
      // tolerance) of reported-time accuracy.
      double t_evt;
      if (br.f_root == 0.0) {
        t_evt = br.root;
      } else if (c.increasing ? (br.f_root > 0.0) : (br.f_root < 0.0)) {
        t_evt = br.root;
      } else {
        t_evt = br.other;
      }
      if (t_evt < t) t_evt = t;
      if (t_evt > t1) t_evt = t1;
      // Earliest root wins; exact ties fire the lowest index (ascending
      // iteration + strict inequality), deterministically.
      if (!have_root || t_evt < t_root) {
        have_root = true;
        t_root = t_evt;
        root_idx = i;
        root_increasing = c.increasing;
      }
    }

    if (have_root) {
      // Re-integrate from the step start to the root so the event-time
      // state carries integrator accuracy, not interpolant accuracy. The
      // shortened step needs no error re-test: it is interior to a step
      // that already passed (fixed modes never test).
      const double h_evt = t_root - t;
      if (h_evt > 0.0) {
        if (opt.method == Method::kRk4) {
          rk4.step(cf, t, y.data(), h_evt, yevt.data(), fa.data());
        } else {
          rkf78.step(cf, t, y.data(), h_evt, yevt.data(), errv.data(),
                     fa.data());
        }
      } else {
        yevt = y;
      }
      cf(t_root, yevt.data(), fb.data());

      if (observer) {
        integrate::DenseStep trunc;
        trunc.t0 = t;
        trunc.h = h_evt;
        trunc.y0 = y.data();
        trunc.f0 = fa.data();
        trunc.y1 = yevt.data();
        trunc.f1 = fb.data();
        trunc.n = n;
        observer(trunc);
      }
      result.steps_accepted += 1;
      if (event_log != nullptr) {
        event_log->push_back(EventRecord{t_root, root_idx});
      }

      EventSpec& fired = events[root_idx];
      const bool had_update = static_cast<bool>(fired.on_event);
      if (had_update) {
        fired.on_event(t_root, yevt.data());
      }
      if (fired.terminal) {
        Eigen::Map<Eigen::VectorXd>(y_final, ni) = yevt;
        result.t_final = t_root;
        result.terminated_by_event = true;
        result.terminal_event_index = root_idx;
        return result;
      }

      // Restart at the root. The derivative is recomputed only when a
      // discrete update changed the state; otherwise fb (evaluated at yevt
      // above) is already f(t_root, y).
      t = t_root;
      y.swap(yevt);
      if (had_update) {
        cf(t, y.data(), fa.data());
      } else {
        fa.swap(fb);
      }
      for (std::size_t i = 0; i < n_events; ++i) {
        const double gi_val = events[i].g(t, y.data());
        if (i == root_idx && !had_update) {
          // Duplicate suppression: force the stored sign to the crossed
          // side. Interpolant-level noise at the root cannot re-fire the
          // event; a genuine re-cross must first move g through zero
          // again. When an update ran, the evaluated value stands -- the
          // hook may legitimately have changed the regime.
          const double floor_mag = std::numeric_limits<double>::min();
          if (root_increasing) {
            g_prev[i] = (gi_val > 0.0) ? gi_val : floor_mag;
          } else {
            g_prev[i] = (gi_val < 0.0) ? gi_val : -floor_mag;
          }
        } else {
          g_prev[i] = gi_val;
        }
      }
      if (adaptive) {
        h = h_next;
      }
      continue;
    }

    // --- no event: commit the full step ------------------------------------
    if (observer) {
      observer(dense);
    }
    result.steps_accepted += 1;
    t = t1;
    y.swap(ynew);
    fa.swap(fb);
    g_prev.swap(g_now);
    if (adaptive) {
      h = h_next;
    }
  }

  Eigen::Map<Eigen::VectorXd>(y_final, ni) = y;
  result.t_final = t;
  return result;
}

}  // namespace events
}  // namespace star
