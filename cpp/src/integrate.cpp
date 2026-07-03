// Shared ODE integrator library (FR-11). This file is the chapter-manifest
// anchor for the integrators chapter (docs/mathlib chapter ch:integrators);
// the equation labels referenced below are defined there.
//
// Derivation sources: classical RK4 and the error norm per Hairer, Norsett,
// and Wanner, Solving Ordinary Differential Equations I; RKF7(8) tableau per
// Fehlberg, NASA TR R-287 (1968); PI step-size control per Gustafsson (1991)
// and Hairer and Wanner, Solving Ordinary Differential Equations II,
// section IV.2.
#include "star/integrate.hpp"

#include <cmath>
#include <stdexcept>

namespace star {
namespace integrate {

// ---------------------------------------------------------------------------
// Error norm
// ---------------------------------------------------------------------------

double error_norm(const double* e, const double* y0, const double* y1,
                  std::size_t n, const std::vector<StateGroup>& groups) {
  // The partition check runs on every call because the norm is the safety
  // net of the adaptive loop: a mis-declared group silently mis-weighting
  // half the state would be worse than the O(n_groups) comparison cost.
  std::size_t expected_offset = 0;
  for (const StateGroup& g : groups) {
    if (g.offset != expected_offset || g.size == 0) {
      throw std::invalid_argument(
          "state groups must partition [0, n) contiguously in order");
    }
    if (!(g.rtol >= 0.0) || !(g.atol >= 0.0) || !(g.rtol + g.atol > 0.0)) {
      throw std::invalid_argument(
          "state group tolerances must be non-negative and not both zero");
    }
    expected_offset += g.size;
  }
  if (expected_offset != n) {
    throw std::invalid_argument("state groups do not cover the state vector");
  }

  // eq:integrators:errnorm  err = sqrt((1/n) sum_i (e_i/d_i)^2) with
  // d_i = atol_g + rtol_g * max(|y0_i|, |y1_i|). Summation order is the
  // component order, fixed by the partition check above (D-10).
  double sum = 0.0;
  for (const StateGroup& g : groups) {
    for (std::size_t i = g.offset; i < g.offset + g.size; ++i) {
      const double scale =
          g.atol + g.rtol * std::fmax(std::fabs(y0[i]), std::fabs(y1[i]));
      const double q = e[i] / scale;
      sum += q * q;
    }
  }
  return std::sqrt(sum / static_cast<double>(n));
}

// ---------------------------------------------------------------------------
// Classical RK4
// ---------------------------------------------------------------------------

Rk4::Rk4(std::size_t n)
    : k1_(n), k2_(n), k3_(n), k4_(n), ytmp_(n), acc_(n) {
  if (n == 0) {
    throw std::invalid_argument("state dimension must be > 0");
  }
}

void Rk4::step(const RhsRef& f, double t, const double* y, double h,
               double* y_out, const double* f0) {
  const auto n = static_cast<Eigen::Index>(dim());
  const Eigen::Map<const Eigen::VectorXd> ym(y, n);

  // eq:integrators:rk4  classical Runge-Kutta 4 stages. Stage order and the
  // summation order in the final combine are fixed; do not reorder (D-10).
  if (f0 != nullptr) {
    k1_ = Eigen::Map<const Eigen::VectorXd>(f0, n);
  } else {
    f(t, y, k1_.data());
  }

  ytmp_ = ym + (0.5 * h) * k1_;
  f(t + 0.5 * h, ytmp_.data(), k2_.data());

  ytmp_ = ym + (0.5 * h) * k2_;
  f(t + 0.5 * h, ytmp_.data(), k3_.data());

  ytmp_ = ym + h * k3_;
  f(t + h, ytmp_.data(), k4_.data());

  // acc_ buffers the result so y_out may alias y.
  acc_ = ym + (h / 6.0) * (k1_ + 2.0 * k2_ + 2.0 * k3_ + k4_);
  Eigen::Map<Eigen::VectorXd>(y_out, n) = acc_;
}

// ---------------------------------------------------------------------------
// Fehlberg RKF7(8) tableau
// ---------------------------------------------------------------------------

namespace rkf78 {

// Transcribed from the 13-stage seventh-order pair with eighth-order
// error-estimation formula of Fehlberg, NASA TR R-287 (1968), as exact
// rational expressions evaluated in double precision. Transcription is
// machine-verified: the doctest case rkf78_tableau_order_conditions asserts
// the row-sum conditions sum_j a_ij = c_i, the weight normalizations, the
// quadrature order conditions sum_i b_i c_i^q = 1/(q+1) through the order of
// each formula, and a set of deeper coupling conditions.
const double kC[kStages] = {
    0.0,        2.0 / 27.0, 1.0 / 9.0, 1.0 / 6.0, 5.0 / 12.0,
    1.0 / 2.0,  5.0 / 6.0,  1.0 / 6.0, 2.0 / 3.0, 1.0 / 3.0,
    1.0,        0.0,        1.0};

const double kA[kStages][kStages - 1] = {
    // stage 0 row is unused (no predecessors); kept for index alignment.
    {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {2.0 / 27.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {1.0 / 36.0, 1.0 / 12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0},
    {1.0 / 24.0, 0.0, 1.0 / 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {5.0 / 12.0, 0.0, -25.0 / 16.0, 25.0 / 16.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0, 0.0},
    {1.0 / 20.0, 0.0, 0.0, 1.0 / 4.0, 1.0 / 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0},
    {-25.0 / 108.0, 0.0, 0.0, 125.0 / 108.0, -65.0 / 27.0, 125.0 / 54.0, 0.0,
     0.0, 0.0, 0.0, 0.0, 0.0},
    {31.0 / 300.0, 0.0, 0.0, 0.0, 61.0 / 225.0, -2.0 / 9.0, 13.0 / 900.0, 0.0,
     0.0, 0.0, 0.0, 0.0},
    {2.0, 0.0, 0.0, -53.0 / 6.0, 704.0 / 45.0, -107.0 / 9.0, 67.0 / 90.0, 3.0,
     0.0, 0.0, 0.0, 0.0},
    {-91.0 / 108.0, 0.0, 0.0, 23.0 / 108.0, -976.0 / 135.0, 311.0 / 54.0,
     -19.0 / 60.0, 17.0 / 6.0, -1.0 / 12.0, 0.0, 0.0, 0.0},
    {2383.0 / 4100.0, 0.0, 0.0, -341.0 / 164.0, 4496.0 / 1025.0,
     -301.0 / 82.0, 2133.0 / 4100.0, 45.0 / 82.0, 45.0 / 164.0, 18.0 / 41.0,
     0.0, 0.0},
    {3.0 / 205.0, 0.0, 0.0, 0.0, 0.0, -6.0 / 41.0, -3.0 / 205.0, -3.0 / 41.0,
     3.0 / 41.0, 6.0 / 41.0, 0.0, 0.0},
    {-1777.0 / 4100.0, 0.0, 0.0, -341.0 / 164.0, 4496.0 / 1025.0,
     -289.0 / 82.0, 2193.0 / 4100.0, 51.0 / 82.0, 33.0 / 164.0, 12.0 / 41.0,
     0.0, 1.0}};

const double kB7[kStages] = {41.0 / 840.0, 0.0,         0.0,
                             0.0,          0.0,         34.0 / 105.0,
                             9.0 / 35.0,   9.0 / 35.0,  9.0 / 280.0,
                             9.0 / 280.0,  41.0 / 840.0, 0.0,
                             0.0};

const double kB8[kStages] = {0.0,          0.0,         0.0,
                             0.0,          0.0,         34.0 / 105.0,
                             9.0 / 35.0,   9.0 / 35.0,  9.0 / 280.0,
                             9.0 / 280.0,  0.0,         41.0 / 840.0,
                             41.0 / 840.0};

}  // namespace rkf78

// ---------------------------------------------------------------------------
// RKF7(8) stepper
// ---------------------------------------------------------------------------

Rkf78::Rkf78(std::size_t n) : ytmp_(n), acc_(n) {
  if (n == 0) {
    throw std::invalid_argument("state dimension must be > 0");
  }
  for (auto& k : k_) {
    k.resize(static_cast<Eigen::Index>(n));
  }
}

void Rkf78::step(const RhsRef& f, double t, const double* y, double h,
                 double* y_out, double* err_out, const double* f0) {
  const auto n = static_cast<Eigen::Index>(dim());
  const Eigen::Map<const Eigen::VectorXd> ym(y, n);

  // eq:integrators:rkf78:stages  k_i = f(t + c_i h, y + h sum_j a_ij k_j).
  // The j-loop skips entries that are exactly zero in the tableau; the
  // skipped set is a compile-time constant, so the summation order is fixed
  // across runs and platforms (D-10).
  if (f0 != nullptr) {
    k_[0] = Eigen::Map<const Eigen::VectorXd>(f0, n);
  } else {
    f(t, y, k_[0].data());
  }
  for (int i = 1; i < rkf78::kStages; ++i) {
    ytmp_ = ym;
    for (int j = 0; j < i; ++j) {
      if (rkf78::kA[i][j] != 0.0) {
        ytmp_ += (h * rkf78::kA[i][j]) * k_[j];
      }
    }
    f(t + rkf78::kC[i] * h, ytmp_.data(), k_[i].data());
  }

  // Local extrapolation: propagate the 8th-order combination.
  acc_ = ym;
  for (int i = 0; i < rkf78::kStages; ++i) {
    if (rkf78::kB8[i] != 0.0) {
      acc_ += (h * rkf78::kB8[i]) * k_[i];
    }
  }

  // eq:integrators:rkf78:err  y7 - y8 = (41 h / 840)(k0 + k10 - k11 - k12):
  // the closed form of h * sum_i (b7_i - b8_i) k_i, estimating the local
  // error of the 7th-order formula. Written into err_out before y_out so an
  // aliased y_out cannot clobber inputs (k vectors are internal anyway).
  const double w = 41.0 / 840.0 * h;
  Eigen::Map<Eigen::VectorXd>(err_out, n) =
      w * (k_[0] + k_[10] - k_[11] - k_[12]);
  Eigen::Map<Eigen::VectorXd>(y_out, n) = acc_;
}

// ---------------------------------------------------------------------------
// PI step-size controller
// ---------------------------------------------------------------------------

PiController::PiController(double safety, double fac_min, double fac_max,
                           int order_k)
    : safety_(safety), fac_min_(fac_min), fac_max_(fac_max) {
  if (!(safety > 0.0) || !(fac_min > 0.0) || !(fac_max >= 1.0) ||
      order_k < 1) {
    throw std::invalid_argument("invalid PI controller parameters");
  }
  // eq:integrators:pi exponents: beta1 = 0.7/k, beta2 = 0.4/k per
  // Gustafsson (1991) / Hairer-Wanner II.IV.2 (kI = 0.3/k, kP = 0.4/k in
  // integral/proportional form; beta1 = kI + kP, beta2 = kP).
  beta1_ = 0.7 / static_cast<double>(order_k);
  beta2_ = 0.4 / static_cast<double>(order_k);
  inv_k_ = 1.0 / static_cast<double>(order_k);
}

double PiController::factor_accept(double err) {
  double fac;
  if (err == 0.0) {
    // The estimate can be exactly zero (e.g. linear dynamics the pair
    // integrates exactly); grow at the cap rather than dividing by zero.
    fac = fac_max_;
  } else {
    // eq:integrators:pi  fac = S * err_n^(-beta1) * err_(n-1)^(beta2).
    fac = safety_ * std::pow(err, -beta1_) * std::pow(err_prev_, beta2_);
    if (fac < fac_min_) fac = fac_min_;
    if (fac > fac_max_) fac = fac_max_;
  }
  if (last_rejected_ && fac > 1.0) {
    // Anti-thrashing rule (Hairer-Wanner): never grow the step on the first
    // acceptance after a rejection.
    fac = 1.0;
  }
  // Floor the stored history at a tiny positive value so a zero error does
  // not poison the next PI evaluation with pow(0, beta2) = 0.
  err_prev_ = (err > 1e-300) ? err : 1e-300;
  last_rejected_ = false;
  return fac;
}

double PiController::factor_reject(double err) {
  // Elementary controller on rejection (standard practice: the PI history
  // is not meaningful across a rejected step); never allows growth.
  double fac = safety_ * std::pow(err, -inv_k_);
  if (fac < fac_min_) fac = fac_min_;
  if (fac > 1.0) fac = 1.0;
  last_rejected_ = true;
  return fac;
}

// ---------------------------------------------------------------------------
// Dense output
// ---------------------------------------------------------------------------

void DenseStep::eval(double t, double* y_out) const {
  // Clamp instead of throwing: callers hand t values produced by root
  // finding, whose last bisection can sit an ulp outside [t0, t0+h].
  double theta = (t - t0) / h;
  if (theta < 0.0) theta = 0.0;
  if (theta > 1.0) theta = 1.0;

  // eq:integrators:hermite  u(theta) = (1-theta) y0 + theta y1
  //   + theta(theta-1)[(1-2theta)(y1-y0) + (theta-1) h f0 + theta h f1].
  // Fixed per-component evaluation order (D-10); no allocation.
  const double tm1 = theta - 1.0;
  const double w_lin0 = 1.0 - theta;
  const double q = theta * tm1;  // theta(theta-1)
  const double w_dy = q * (1.0 - 2.0 * theta);
  const double w_f0 = q * tm1 * h;
  const double w_f1 = q * theta * h;
  for (std::size_t i = 0; i < n; ++i) {
    const double dy = y1[i] - y0[i];
    y_out[i] = w_lin0 * y0[i] + theta * y1[i] + w_dy * dy + w_f0 * f0[i] +
               w_f1 * f1[i];
  }
}

}  // namespace integrate
}  // namespace star
