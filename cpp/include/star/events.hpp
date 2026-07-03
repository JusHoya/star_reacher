// Event-detection framework and event-aware propagation driver (FR-12):
// scalar event functions g(t, y), per-step sign screening on accepted steps,
// Brent root location on the integrator's dense output to a configurable
// time tolerance (1e-9 s default), direction filters, and integrator restart
// at the located root with a discrete-update hook -- the mechanism the
// staging/SOI events of later phases plug into. New event types are added by
// supplying new g callables; the framework itself does not change.
//
// Math-library traceability (FR-29): derivations live in the events chapter
// of docs/mathlib (ch:events); equation labels (eq:events:*) are echoed at
// the corresponding code.
#ifndef STAR_EVENTS_HPP
#define STAR_EVENTS_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include "star/integrate.hpp"

namespace star {
namespace events {

// Non-owning reference to a scalar event function g(t, y). Same lifetime
// contract as integrate::RhsRef: the callable must outlive the reference.
class EventFnRef {
 public:
  template <typename F>
  EventFnRef(F& f)  // NOLINT: implicit by design, mirrors RhsRef
      : ctx_(&f), fn_([](void* ctx, double t, const double* y) -> double {
          return (*static_cast<F*>(ctx))(t, y);
        }) {}
  template <typename F>
  EventFnRef(const F&& f) = delete;

  double operator()(double t, const double* y) const {
    return fn_(ctx_, t, y);
  }

 private:
  void* ctx_;
  double (*fn_)(void*, double, const double*);
};

// Non-owning reference to a scalar function of one variable, for the root
// finder.
class ScalarFnRef {
 public:
  template <typename F>
  ScalarFnRef(F& f)  // NOLINT: implicit by design
      : ctx_(&f), fn_([](void* ctx, double x) -> double {
          return (*static_cast<F*>(ctx))(x);
        }) {}
  template <typename F>
  ScalarFnRef(const F&& f) = delete;

  double operator()(double x) const { return fn_(ctx_, x); }

 private:
  void* ctx_;
  double (*fn_)(void*, double);
};

// Optional non-owning reference to a discrete state update applied at a
// located event: the hook may modify y in place (mass drop, impulsive dv,
// origin re-centering in later phases). Default-constructed means "absent".
class UpdateFnRef {
 public:
  UpdateFnRef() = default;
  template <typename F>
  UpdateFnRef(F& f)  // NOLINT: implicit by design
      : ctx_(&f), fn_([](void* ctx, double t, double* y) {
          (*static_cast<F*>(ctx))(t, y);
        }) {}
  template <typename F>
  UpdateFnRef(const F&& f) = delete;

  explicit operator bool() const { return fn_ != nullptr; }
  void operator()(double t, double* y) const { fn_(ctx_, t, y); }

 private:
  void* ctx_ = nullptr;
  void (*fn_)(void*, double, double*) = nullptr;
};

// Which sign changes of g fire the event (eq:events:screen).
enum class Direction { kAny, kIncreasing, kDecreasing };

// One registered event. The framework knows nothing about what g means;
// Phase 3+ event types (staging, SOI, ground impact) are just more g
// callables with the appropriate direction and hook.
struct EventSpec {
  const char* name;  // diagnostic only; not owned
  EventFnRef g;
  Direction direction = Direction::kAny;
  UpdateFnRef on_event{};  // optional discrete update at the root
  bool terminal = false;   // true: propagation stops at the located root
};

// Brent's method (Brent 1973, Algorithms for Minimization without
// Derivatives, ch. 4): guaranteed-convergent root bracketing combining
// bisection, secant, and inverse quadratic interpolation. Requires
// fa = f(a), fb = f(b) with opposite signs (throws std::invalid_argument
// otherwise). Iterates until the final bracket width is at most
// tol_x + 4*eps*|root|, i.e. tol_x is honored down to the resolution of
// double at the root's magnitude. root is the endpoint with the smaller
// residual |f|; (other, f_other) is the surviving counterpoint of the final
// bracket, exposed so callers that need a specific SIDE of the root (the
// event driver restarts on the post-crossing side) can pick it.
struct BrentResult {
  double root = 0.0;
  double f_root = 0.0;
  double other = 0.0;
  double f_other = 0.0;
  int iterations = 0;
};
BrentResult brent_root(const ScalarFnRef& f, double a, double b, double fa,
                       double fb, double tol_x, int max_iter = 128);

// Ready-made event functions for the Phase 2 event set. Both are plain
// callables handed to EventSpec::g by reference, exactly like user events.

// Timer: g = t - t_event (eq:events:timer). Increasing by construction.
struct TimerEvent {
  double t_event_s;
  double operator()(double t, const double* /*y*/) const {
    return t - t_event_s;
  }
};

// Apsis crossing for a translational state laid out [r(3), v(3)] starting at
// state_offset: g = r . v (eq:events:apsis). On a bound eccentric orbit
// g < 0 approaching periapsis and g > 0 leaving it, so periapsis is an
// increasing crossing and apoapsis a decreasing one. Degenerate for e -> 0
// (g stays near zero everywhere): documented out-of-domain in the chapter.
struct ApsisEvent {
  std::size_t state_offset = 0;
  double operator()(double /*t*/, const double* y) const {
    const double* r = y + state_offset;
    const double* v = y + state_offset + 3;
    return r[0] * v[0] + r[1] * v[1] + r[2] * v[2];
  }
};

// Optional non-owning observer invoked once per accepted step (after event
// truncation, before the discrete update), receiving the step's dense
// interpolant. Used for uniform-rate logging and for the acceptance suite's
// drift/interpolation measurements. Default-constructed means "absent".
class StepObserverRef {
 public:
  StepObserverRef() = default;
  template <typename F>
  StepObserverRef(F& f)  // NOLINT: implicit by design
      : ctx_(&f), fn_([](void* ctx, const integrate::DenseStep& d) {
          (*static_cast<F*>(ctx))(d);
        }) {}
  template <typename F>
  StepObserverRef(const F&& f) = delete;

  explicit operator bool() const { return fn_ != nullptr; }
  void operator()(const integrate::DenseStep& d) const { fn_(ctx_, d); }

 private:
  void* ctx_ = nullptr;
  void (*fn_)(void*, const integrate::DenseStep&) = nullptr;
};

enum class Method { kRk4, kRkf78 };
enum class StepMode { kFixed, kAdaptive };  // kAdaptive requires kRkf78

struct PropagateOptions {
  Method method = Method::kRkf78;
  StepMode mode = StepMode::kAdaptive;
  double h_fixed = 0.0;  // required > 0 in kFixed mode
  integrate::AdaptiveOptions adaptive{};  // used in kAdaptive mode
  double event_time_tol_s = 1e-9;  // FR-12 root-location tolerance
};

// One located event occurrence.
struct EventRecord {
  double t = 0.0;
  std::size_t event_index = 0;  // index into the events array
};

struct PropagateResult {
  double t_final = 0.0;
  std::int64_t steps_accepted = 0;
  std::int64_t steps_rejected = 0;
  std::int64_t rhs_evals = 0;
  bool terminated_by_event = false;
  std::size_t terminal_event_index = 0;  // valid iff terminated_by_event
};

// Propagate y' = f(t, y) from (t0, y0) toward tf with event detection.
//
// Loop per accepted step: integrate [t, t+h]; build the dense interpolant
// from endpoint states/derivatives; screen every event's sign change
// against its direction filter (eq:events:screen); if any event crossed,
// locate the earliest root with Brent on the interpolant to
// event_time_tol_s, RE-INTEGRATE from the step start to the root (so the
// event-time state carries integrator accuracy, not interpolant accuracy),
// invoke the observer on the truncated step, record the event, apply its
// discrete update, and restart from the root. Ties (multiple events rooted
// within the time tolerance) fire lowest-index-first, deterministically.
//
// Duplicate suppression: after an event without a discrete update fires,
// its stored screening sign is forced to the post-crossing side, so
// interpolant-level noise at the root cannot re-fire it; roots of the same
// event closer than event_time_tol_s are indistinguishable by construction.
// When a discrete update ran, the sign is re-evaluated from the updated
// state instead (the hook may legitimately change the regime).
//
// event_log may be null (events located and applied but not recorded).
// Determinism: all workspace is pre-allocated; the only allocation after
// setup is event_log growth on an actual event (callers may reserve()).
// Throws std::invalid_argument on malformed options and std::runtime_error
// if the adaptive error test cannot be satisfied at h_min.
PropagateResult propagate(const integrate::RhsRef& f, double t0, double tf,
                          const double* y0, std::size_t n, double* y_final,
                          const PropagateOptions& opt, EventSpec* events,
                          std::size_t n_events,
                          std::vector<EventRecord>* event_log,
                          StepObserverRef observer = {});

}  // namespace events
}  // namespace star

#endif  // STAR_EVENTS_HPP
