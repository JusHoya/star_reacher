// Cannonball SRP and apparent-disk conical shadow (FR-7). Derivation:
// docs/mathlib chapter ch:srp. The three shadow functions share one
// apparent-geometry helper so they cannot disagree about the geometry; the
// full-sun and umbra branches return literal 1.0/0.0 before any area
// arithmetic runs, so nu is bit-exact in both regimes. The libm calls
// (asin, acos, atan2) are the only non-correctly-rounded operations and
// therefore this module's cross-platform divergence surface (D-10);
// tests/golden/srp/manifest.toml budgets their few-ulp spread.
#include "star/models/srp.hpp"

#include <algorithm>
#include <cmath>

#include "star/constants.hpp"

namespace star {
namespace models {

namespace {

// Apparent angular radii of the Sun (*a_sun) and occulter (*a_occ) and the
// angular separation of their centers (*sep), seen from r_sc
// (eq:srp:appgeom). The separation uses atan2(|u x v|, u.v), accurate for
// all separations where acos of the normalized dot loses precision near 0
// and pi; the asin arguments clamp to 1 for the out-of-domain sub-surface
// states (ch:srp domain rule).
void apparent_geometry(const Eigen::Vector3d& r_sc_m,
                       const Eigen::Vector3d& r_sun_m, double radius_sun_m,
                       const Eigen::Vector3d& r_occ_m, double radius_occ_m,
                       double* a_sun, double* a_occ, double* sep) {
  const Eigen::Vector3d to_sun = r_sun_m - r_sc_m;
  const Eigen::Vector3d to_occ = r_occ_m - r_sc_m;
  const double d_sun = to_sun.norm();
  const double d_occ = to_occ.norm();
  *a_sun = std::asin(std::min(1.0, radius_sun_m / d_sun));
  *a_occ = std::asin(std::min(1.0, radius_occ_m / d_occ));
  *sep = std::atan2(to_sun.cross(to_occ).norm(), to_sun.dot(to_occ));
}

}  // namespace

double shadow_fraction(const Eigen::Vector3d& r_sc_m,
                       const Eigen::Vector3d& r_sun_m, double radius_sun_m,
                       const Eigen::Vector3d& r_occ_m, double radius_occ_m) {
  double a = 0.0;  // solar apparent radius
  double b = 0.0;  // occulter apparent radius
  double c = 0.0;  // apparent center separation
  apparent_geometry(r_sc_m, r_sun_m, radius_sun_m, r_occ_m, radius_occ_m,
                    &a, &b, &c);
  // eq:srp:nu -- piecewise by disk containment/tangency. The 1.0 and 0.0
  // branches are literals so full sunlight and total umbra are bit-exact.
  if (c >= a + b) {
    return 1.0;  // disks disjoint: full sunlight
  }
  if (c <= b - a) {
    return 0.0;  // solar disk inside occulter disk: total umbra
  }
  if (c <= a - b) {
    // Annular (antumbra): the whole occulter disk covers part of the Sun;
    // nu is independent of c while the containment lasts (eq:srp:annular).
    return 1.0 - (b / a) * (b / a);
  }
  // Partial overlap: lens area of two planar disks (eq:srp:overlap). The
  // acos arguments are clamped and the chord height floored at zero to
  // protect boundary-adjacent states from sub-ulp domain excursions.
  const double x = (c * c + a * a - b * b) / (2.0 * c);
  const double y = std::sqrt(std::max(0.0, a * a - x * x));
  const double area =
      a * a * std::acos(std::max(-1.0, std::min(1.0, x / a))) +
      b * b * std::acos(std::max(-1.0, std::min(1.0, (c - x) / b))) - c * y;
  return 1.0 - area / (constants::PI * a * a);
}

double shadow_umbra_boundary(const Eigen::Vector3d& r_sc_m,
                             const Eigen::Vector3d& r_sun_m,
                             double radius_sun_m,
                             const Eigen::Vector3d& r_occ_m,
                             double radius_occ_m) {
  double a = 0.0;
  double b = 0.0;
  double c = 0.0;
  apparent_geometry(r_sc_m, r_sun_m, radius_sun_m, r_occ_m, radius_occ_m,
                    &a, &b, &c);
  // eq:srp:boundaries -- g_u = theta - (a_occ - a_sun): negative exactly in
  // the total umbra (internal disk tangency = umbra cone surface, ch:srp).
  return c - (b - a);
}

double shadow_penumbra_boundary(const Eigen::Vector3d& r_sc_m,
                                const Eigen::Vector3d& r_sun_m,
                                double radius_sun_m,
                                const Eigen::Vector3d& r_occ_m,
                                double radius_occ_m) {
  double a = 0.0;
  double b = 0.0;
  double c = 0.0;
  apparent_geometry(r_sc_m, r_sun_m, radius_sun_m, r_occ_m, radius_occ_m,
                    &a, &b, &c);
  // eq:srp:boundaries -- g_p = theta - (a_sun + a_occ): negative exactly
  // where any part of the Sun is occulted (external disk tangency =
  // penumbra cone surface).
  return c - (a + b);
}

Eigen::Vector3d srp_accel(double cr_area_over_mass_m2pkg, double nu,
                          const Eigen::Vector3d& r_sc_m,
                          const Eigen::Vector3d& r_sun_m) {
  // eq:srp:cannonball -- a = nu P_1au (au/d)^2 (Cr A/m) s_hat with s_hat
  // the Sun-to-spacecraft unit vector (header direction convention). The
  // scalar coefficient carries nu, so nu == 0 gives an exactly zero
  // vector. P_1au is the compile-time quotient of the cited irradiance and
  // speed-of-light constants (eq:srp:pressure).
  const Eigen::Vector3d s_vec = r_sc_m - r_sun_m;
  const double d = s_vec.norm();
  const double au_over_d = constants::AU_M / d;
  const double k = nu * constants::SRP_PRESSURE_1AU_N_PER_M2 * au_over_d *
                   au_over_d * cr_area_over_mass_m2pkg / d;
  return k * s_vec;
}

}  // namespace models
}  // namespace star
