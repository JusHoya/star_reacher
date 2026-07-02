// Analytic elliptic two-body reference propagator -- TEST SUPPORT ONLY.
//
// This header exists so the doctest acceptance suite and the thin
// test-support bindings share one implementation of the closed-form Kepler
// solution used as truth by the Phase 2 exit criteria (convergence slopes,
// drift, apsis event times). It is header-only, lives outside the
// deterministic run path, and is never called by mission propagation. It is
// itself validated against the independently generated golden checkpoints
// in tests/golden/integrators/ (doctest kepler_reference_propagator_golden).
//
// Method (Vallado, Fundamentals of Astrodynamics and Applications, 4th ed.,
// ch. 2 -- Kepler's equation and Kepler's problem): derive a, e, and the
// perifocal basis from the input state; advance the mean anomaly; solve
// M = E - e sin E by Newton's method to machine precision; reconstruct the
// state in the perifocal frame. Domain: bound, eccentric orbits
// (0 < e < 1); anything else throws std::domain_error.
#ifndef STAR_TESTSUPPORT_KEPLER_REF_HPP
#define STAR_TESTSUPPORT_KEPLER_REF_HPP

#include <cmath>
#include <stdexcept>

#include <Eigen/Dense>

#include "star/constants.hpp"

namespace star {
namespace testsupport {

struct EllipticOrbitRef {
  double mu = 0.0;
  double a = 0.0;       // semi-major axis [m]
  double e = 0.0;       // eccentricity
  double n = 0.0;       // mean motion [rad/s]
  double period = 0.0;  // [s]
  double m0 = 0.0;      // mean anomaly at the defining epoch [rad]
  Eigen::Vector3d p_hat, q_hat, w_hat;  // perifocal basis in the input frame

  EllipticOrbitRef(double mu_in, const Eigen::Vector3d& r0,
                   const Eigen::Vector3d& v0)
      : mu(mu_in) {
    const double r0n = r0.norm();
    const double alpha = 2.0 / r0n - v0.squaredNorm() / mu;  // 1/a
    if (!(alpha > 0.0)) {
      throw std::domain_error("kepler_ref: orbit must be elliptic");
    }
    a = 1.0 / alpha;
    const Eigen::Vector3d h_vec = r0.cross(v0);
    const Eigen::Vector3d e_vec = v0.cross(h_vec) / mu - r0 / r0n;
    e = e_vec.norm();
    if (!(e > 1e-12) || !(e < 1.0)) {
      throw std::domain_error(
          "kepler_ref: orbit must be eccentric (0 < e < 1); the perifocal "
          "basis is undefined for circular orbits");
    }
    n = std::sqrt(mu / (a * a * a));
    // constants::TWO_PI rather than M_PI: the latter is not standard C++17
    // and is absent on MSVC without feature macros.
    period = constants::TWO_PI / n;
    p_hat = e_vec / e;
    w_hat = h_vec / h_vec.norm();
    q_hat = w_hat.cross(p_hat);
    // Eccentric anomaly of the defining state: cos E = (1 - r/a)/e,
    // sin E = (r.v)/(e sqrt(mu a)) (Vallado ch. 2).
    const double cos_e0 = (1.0 - r0n / a) / e;
    const double sin_e0 = r0.dot(v0) / (e * std::sqrt(mu * a));
    const double e0 = std::atan2(sin_e0, cos_e0);
    m0 = e0 - e * std::sin(e0);
  }

  // Newton solution of Kepler's equation M = E - e sin E. The fixed
  // convergence threshold and iteration cap keep the solve deterministic;
  // for e < 1 Newton from E = M converges in a handful of iterations
  // (Vallado ch. 2). Throws if the residual fails to reach solver noise --
  // that would mean the truth reference itself is broken.
  double solve_kepler(double m_anom) const {
    double e_anom = m_anom;
    for (int i = 0; i < 64; ++i) {
      const double fval = e_anom - e * std::sin(e_anom) - m_anom;
      const double fprime = 1.0 - e * std::cos(e_anom);
      const double delta = fval / fprime;
      e_anom -= delta;
      if (std::fabs(delta) < 5e-16) {
        break;
      }
    }
    const double residual = e_anom - e * std::sin(e_anom) - m_anom;
    // Residual scale: M grows linearly with time, so allow ulp-level error
    // relative to |M| plus an absolute floor.
    if (std::fabs(residual) > 1e-13 * std::fmax(1.0, std::fabs(m_anom))) {
      throw std::runtime_error("kepler_ref: Kepler solve failed to converge");
    }
    return e_anom;
  }

  // Analytic state at time t past the defining epoch.
  void state_at(double t, Eigen::Vector3d* r_out,
                Eigen::Vector3d* v_out) const {
    const double m_anom = m0 + n * t;
    const double e_anom = solve_kepler(m_anom);
    const double ce = std::cos(e_anom);
    const double se = std::sin(e_anom);
    const double b_semi = a * std::sqrt(1.0 - e * e);  // semi-minor axis
    *r_out = (a * (ce - e)) * p_hat + (b_semi * se) * q_hat;
    const double r_mag = a * (1.0 - e * ce);
    const double vs = std::sqrt(mu * a) / r_mag;
    *v_out = (-vs * se) * p_hat + (vs * std::sqrt(1.0 - e * e) * ce) * q_hat;
  }
};

// One-call convenience mirroring the Python-facing binding.
inline void propagate_kepler(double mu, const Eigen::Vector3d& r0,
                             const Eigen::Vector3d& v0, double t,
                             Eigen::Vector3d* r_out, Eigen::Vector3d* v_out) {
  EllipticOrbitRef(mu, r0, v0).state_at(t, r_out, v_out);
}

}  // namespace testsupport
}  // namespace star

#endif  // STAR_TESTSUPPORT_KEPLER_REF_HPP
