// Shared ODE integrator library (FR-11): fixed-step classical RK4, the
// Fehlberg RKF7(8) embedded pair (NASA TR R-287) in fixed-step and adaptive
// modes with PI step-size control and per-state-group tolerances, and cubic
// Hermite dense output across each accepted step.
//
// Math-library traceability (FR-29): the derivations live in the integrators
// chapter of docs/mathlib (ch:integrators); the implementation echoes its
// equation labels (eq:integrators:*) at the corresponding code.
//
// Determinism contract (D-10): every routine here is single-threaded, uses a
// fixed evaluation and summation order, and performs no heap allocation
// inside the step loop -- all workspace is sized at construction. The
// right-hand-side interface is a non-owning function reference so arbitrary
// callables plug in with one indirect call and zero allocation.
#ifndef STAR_INTEGRATE_HPP
#define STAR_INTEGRATE_HPP

#include <cstddef>
#include <string>
#include <vector>

#include <Eigen/Dense>

namespace star {
namespace integrate {

// Non-owning reference to the ODE right-hand side ydot = f(t, y). Raw
// pointers (not Eigen types) keep the interface storage-agnostic; y and ydot
// each hold dim() doubles. The referenced callable must outlive the RhsRef
// (it is captured by address) -- callers hold their lambda in a local and
// pass it down, which is the only supported pattern. This sits inside the
// hot loop for the life of the project: one indirect call per evaluation,
// nothing else.
class RhsRef {
 public:
  template <typename F>
  RhsRef(F& f)  // NOLINT: implicit so call sites read propagate(rhs, ...)
      : ctx_(&f), fn_([](void* ctx, double t, const double* y, double* ydot) {
          (*static_cast<F*>(ctx))(t, y, ydot);
        }) {}
  // Binding a temporary would dangle immediately; forbid it at compile time.
  template <typename F>
  RhsRef(const F&& f) = delete;

  void operator()(double t, const double* y, double* ydot) const {
    fn_(ctx_, t, y, ydot);
  }

 private:
  void* ctx_;
  void (*fn_)(void*, double, const double*, double*);
};

// One named slice of the state vector with its own error tolerances
// (FR-11 per-state-group tolerances). Groups exist so position, velocity,
// and later attitude components -- which live on wildly different scales --
// each get a physically meaningful error weight instead of sharing one.
struct StateGroup {
  std::string name;    // e.g. "position"; diagnostic only
  std::size_t offset;  // first component index in y
  std::size_t size;    // number of components
  double rtol;         // relative tolerance
  double atol;         // absolute tolerance, in the group's own units
};

// Weighted RMS error norm over the whole state (eq:integrators:errnorm):
//   err = sqrt( (1/n) * sum_i (e_i / d_i)^2 ),
//   d_i  = atol_g + rtol_g * max(|y0_i|, |y1_i|)   for i in group g,
// following the norm of Hairer, Norsett, and Wanner, Solving ODEs I,
// section II.4. A step is acceptable iff err <= 1. The groups must exactly
// partition [0, n) in ascending offset order (validated; throws
// std::invalid_argument otherwise), which also fixes the summation order.
double error_norm(const double* e, const double* y0, const double* y1,
                  std::size_t n, const std::vector<StateGroup>& groups);

// Classical fourth-order Runge-Kutta, fixed step (eq:integrators:rk4).
// Workspace is allocated once at construction for dimension n; step() is
// allocation-free and y_out may alias y.
class Rk4 {
 public:
  explicit Rk4(std::size_t n);

  std::size_t dim() const { return static_cast<std::size_t>(k1_.size()); }

  // Advance one step of size h from (t, y) to y_out. If f0 is non-null it
  // must equal f(t, y) and stage 1 reuses it (the caller already evaluated
  // it for dense output or logging); the arithmetic is identical either way.
  void step(const RhsRef& f, double t, const double* y, double h,
            double* y_out, const double* f0 = nullptr);

 private:
  Eigen::VectorXd k1_, k2_, k3_, k4_, ytmp_, acc_;
};

// Fehlberg 7(8) coefficients, transcribed as exact rational expressions from
// the 13-stage seventh-order pair with eighth-order error-estimation formula
// of Fehlberg, NASA TR R-287 (1968). Exposed so the doctest suite can assert
// the row-sum and quadrature order conditions directly on the tableau -- the
// machine check that guards against transcription error.
namespace rkf78 {
inline constexpr int kStages = 13;
extern const double kC[kStages];               // nodes (Fehlberg's alpha)
extern const double kA[kStages][kStages - 1];  // coupling matrix (beta)
extern const double kB7[kStages];              // 7th-order weights (c)
extern const double kB8[kStages];              // 8th-order weights (c-hat)
}  // namespace rkf78

// Fehlberg RKF7(8) embedded pair (eq:integrators:rkf78:stages). The step
// propagates the eighth-order solution (local extrapolation) and returns the
// embedded difference y8 - y7 as the componentwise error estimate
// (eq:integrators:rkf78:err), which estimates the local error of the
// seventh-order formula. Local extrapolation is what the Phase 2 exit
// criterion's fixed-step order slope (>= 7.5) measures.
class Rkf78 {
 public:
  explicit Rkf78(std::size_t n);

  std::size_t dim() const { return static_cast<std::size_t>(ytmp_.size()); }

  // One raw step of size h from (t, y): the 8th-order solution lands in
  // y_out (may alias y) and the embedded error estimate in err_out. If f0 is
  // non-null it must equal f(t, y) and stage 0 reuses it.
  void step(const RhsRef& f, double t, const double* y, double h,
            double* y_out, double* err_out, const double* f0 = nullptr);

 private:
  Eigen::VectorXd k_[rkf78::kStages];
  Eigen::VectorXd ytmp_, acc_;
};

// Adaptive step-size options. h_init is required (there is deliberately no
// automatic initial-step heuristic: an explicit value keeps runs bitwise
// reproducible under configuration inspection). Reaching h_min with the
// error test still failing throws std::runtime_error -- the documented
// out-of-domain response for dynamics this pair cannot resolve (e.g. stiff
// systems), rather than silently delivering an out-of-tolerance state.
struct AdaptiveOptions {
  std::vector<StateGroup> groups;  // must partition [0, n)
  double h_init = 0.0;             // required > 0
  double h_min = 1e-12;            // floor before the error test aborts
  double h_max = 0.0;              // required > 0; also caps h_init
  double safety = 0.9;             // Hairer-Norsett-Wanner "fac" safety
  double fac_min = 0.2;            // max step decrease per controller call
  double fac_max = 5.0;            // max step increase per controller call
};

// PI step-size controller (eq:integrators:pi), per Gustafsson (1991) with
// the exponent pair recommended in Hairer and Wanner, Solving ODEs II,
// section IV.2: on acceptance
//   h_new = h * clamp( safety * err_n^(-0.7/k) * err_(n-1)^(0.4/k) )
// with k = 8 (the order of the embedded error estimate), and the elementary
// controller h_new = h * clamp(safety * err^(-1/k), <= 1) on rejection. The
// increase factor is additionally clamped to 1 on the first acceptance after
// a rejection (Hairer-Wanner's anti-thrashing rule). err_prev is seeded to 1
// so the first step reduces to the elementary controller. Deterministic:
// pure double arithmetic on the error history, no state beyond err_prev.
class PiController {
 public:
  PiController(double safety, double fac_min, double fac_max, int order_k);

  double factor_accept(double err);  // err <= 1; updates the error history
  double factor_reject(double err);  // err > 1

 private:
  double safety_, fac_min_, fac_max_;
  double beta1_, beta2_, inv_k_;
  double err_prev_ = 1.0;
  bool last_rejected_ = false;
};

// Cubic Hermite dense output over one accepted step [t0, t0 + h], built from
// the endpoint states and derivatives the integrator already has
// (eq:integrators:hermite). With theta = (t - t0)/h:
//   u(theta) = (1-theta) y0 + theta y1
//            + theta (theta-1) [ (1-2 theta)(y1-y0)
//                                + (theta-1) h f0 + theta h f1 ].
// Interpolation error is O(h^4) -- bounded by (h^4/384) max|y''''| for exact
// endpoint data -- which is BELOW the integrator orders (4 and 7/8). Event
// location and uniform-rate logging accuracy are therefore interpolant-
// limited and scale as h^4; capping h_max is the documented control when
// microsecond-level event timing matters (see the events chapter).
// Non-owning: the four pointers must outlive the evaluation.
struct DenseStep {
  double t0 = 0.0;
  double h = 0.0;
  const double* y0 = nullptr;
  const double* f0 = nullptr;
  const double* y1 = nullptr;
  const double* f1 = nullptr;
  std::size_t n = 0;

  // Evaluate u(t) into y_out (size n). t is clamped to [t0, t0 + h]: the
  // cubic extrapolates poorly and no caller has a legitimate reason to
  // sample outside the step.
  void eval(double t, double* y_out) const;
};

}  // namespace integrate
}  // namespace star

#endif  // STAR_INTEGRATE_HPP
