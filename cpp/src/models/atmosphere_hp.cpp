// Harris-Priester density model (FR-8). Derivation and validation table:
// docs/mathlib/chapters/harris_priester.tex (ch:harrispriester). The
// evaluation order deliberately mirrors Orekit's HarrisPriester class (the
// frozen D-15 cross-tool baseline) so cross-tool differences reduce to
// floating-point rounding.
#include "star/models/atmosphere_hp.hpp"

#include <cmath>
#include <stdexcept>

namespace star {
namespace models {

namespace {

// Harris-Priester lower/upper (antapex/apex) density coefficients for mean
// solar activity, 100-1000 km. Source: Montenbruck & Gill, Satellite
// Orbits (2000), Sect. 3.5.2, pp. 89-91 (printed there in g/km^3 = 1e-12
// kg/m^3); recorded here in kg/m^3, digit-for-digit identical to the
// Orekit HarrisPriester reference table. The committed golden
// tests/golden/atmosphere/harris_priester_table.toml is the independent
// transcription check copy; ATM-HP-NODES holds the two bit-identical.
constexpr HpNode kTable[] = {
    {100000.0, 4.974e-07, 4.974e-07}, {120000.0, 2.490e-08, 2.490e-08},
    {130000.0, 8.377e-09, 8.710e-09}, {140000.0, 3.899e-09, 4.059e-09},
    {150000.0, 2.122e-09, 2.215e-09}, {160000.0, 1.263e-09, 1.344e-09},
    {170000.0, 8.008e-10, 8.758e-10}, {180000.0, 5.283e-10, 6.010e-10},
    {190000.0, 3.617e-10, 4.297e-10}, {200000.0, 2.557e-10, 3.162e-10},
    {210000.0, 1.839e-10, 2.396e-10}, {220000.0, 1.341e-10, 1.853e-10},
    {230000.0, 9.949e-11, 1.455e-10}, {240000.0, 7.488e-11, 1.157e-10},
    {250000.0, 5.709e-11, 9.308e-11}, {260000.0, 4.403e-11, 7.555e-11},
    {270000.0, 3.430e-11, 6.182e-11}, {280000.0, 2.697e-11, 5.095e-11},
    {290000.0, 2.139e-11, 4.226e-11}, {300000.0, 1.708e-11, 3.526e-11},
    {320000.0, 1.099e-11, 2.511e-11}, {340000.0, 7.214e-12, 1.819e-11},
    {360000.0, 4.824e-12, 1.337e-11}, {380000.0, 3.274e-12, 9.955e-12},
    {400000.0, 2.249e-12, 7.492e-12}, {420000.0, 1.558e-12, 5.684e-12},
    {440000.0, 1.091e-12, 4.355e-12}, {460000.0, 7.701e-13, 3.362e-12},
    {480000.0, 5.474e-13, 2.612e-12}, {500000.0, 3.916e-13, 2.042e-12},
    {520000.0, 2.819e-13, 1.605e-12}, {540000.0, 2.042e-13, 1.267e-12},
    {560000.0, 1.488e-13, 1.005e-12}, {580000.0, 1.092e-13, 7.997e-13},
    {600000.0, 8.070e-14, 6.390e-13}, {620000.0, 6.012e-14, 5.123e-13},
    {640000.0, 4.519e-14, 4.121e-13}, {660000.0, 3.430e-14, 3.325e-13},
    {680000.0, 2.632e-14, 2.691e-13}, {700000.0, 2.043e-14, 2.185e-13},
    {720000.0, 1.607e-14, 1.779e-13}, {740000.0, 1.281e-14, 1.452e-13},
    {760000.0, 1.036e-14, 1.190e-13}, {780000.0, 8.496e-15, 9.776e-14},
    {800000.0, 7.069e-15, 8.059e-14}, {840000.0, 4.680e-15, 5.741e-14},
    {880000.0, 3.200e-15, 4.210e-14}, {920000.0, 2.210e-15, 3.130e-14},
    {960000.0, 1.560e-15, 2.360e-14}, {1000000.0, 1.150e-15, 1.810e-14},
};
constexpr std::size_t kTableSize = sizeof(kTable) / sizeof(kTable[0]);

// Bulge apex lag behind the Sun, 30 degrees in right ascension
// (Montenbruck & Gill Sect. 3.5.2; same constant as Orekit). Sine/cosine
// recorded as the binary64 results of the double-precision expressions so
// every evaluation shares one value.
constexpr double kLagRad = 30.0 * 3.14159265358979323846 / 180.0;

// Antapex guard, mirroring Orekit: within double rounding of psi == pi,
// cos(psi/2) underflows the power form; the bulge weight is exactly zero
// there (eq:hp:cospow).
constexpr double kMinCosHalf = 1.0e-12;

// eq:hp:density -- antapex profile plus bulge modulation. The pinned
// endpoints return the profile values bit-exactly (weight 0 -> rho_min;
// weight 1 -> rho_max, where a general fused expression could round),
// which is what lets ATM-HP-NODES gate the published node values with
// zero observed error.
double mix(double rho_min, double rho_max, double cos_pow) {
  if (cos_pow == 0.0) {
    return rho_min;
  }
  if (cos_pow == 1.0) {
    return rho_max;
  }
  return rho_min + (rho_max - rho_min) * cos_pow;
}

}  // namespace

double hp_density(double alt_m, double cos_psi, double n) {
  if (!std::isfinite(alt_m) || !std::isfinite(cos_psi) ||
      !std::isfinite(n) || std::fabs(cos_psi) > 1.0 ||
      n < HP_COS_EXPONENT_MIN || n > HP_COS_EXPONENT_MAX) {
    throw std::domain_error("harris_priester: invalid argument");
  }
  if (alt_m < kTable[0].alt_m) {
    // The table has no information below 100 km; launch aerodynamics use
    // USSA76 (ch:ussa76). Refusing beats extrapolating (DX-2).
    throw std::domain_error(
        "harris_priester: altitude below the 100 km table floor");
  }
  if (alt_m > kTable[kTableSize - 1].alt_m) {
    // Orekit-compatible ceiling: zero density above 1000 km keeps the
    // cross-tool trajectories comparable (ch:harrispriester, domain).
    return 0.0;
  }

  // eq:hp:cospow -- bulge weight cos^n(psi/2) via the half-angle identity,
  // evaluated exactly as the Orekit reference: c2 * sqrt(c2)^(n-2).
  const double c2 = 0.5 * (1.0 + cos_psi);
  const double c_half = std::sqrt(c2);
  const double cos_pow =
      (c_half > kMinCosHalf) ? c2 * std::pow(c_half, n - 2.0) : 0.0;

  // Fixed-order bracket scan. For off-node altitudes this selects the
  // same bracket as the Orekit reference; at an exact node altitude it
  // selects the segment STARTING at the node (>= rather than >), so the
  // interpolation exponent is exactly zero and the tabulated value is
  // reproduced bit-for-bit (ATM-HP-NODES relies on it; the reference's
  // choice differs there by ~1 ulp of pow rounding only).
  std::size_t ia = 0;
  while (ia < kTableSize - 2 && alt_m >= kTable[ia + 1].alt_m) {
    ++ia;
  }

  // eq:hp:interp -- exponential interpolation in the power form.
  const HpNode& a = kTable[ia];
  const HpNode& b = kTable[ia + 1];
  // The 1000 km table top is the one node that is never a segment start;
  // return its row directly so node exactness holds there too.
  if (alt_m == b.alt_m) {
    return mix(b.rho_min_kgpm3, b.rho_max_kgpm3, cos_pow);
  }
  const double dh = (a.alt_m - alt_m) / (a.alt_m - b.alt_m);
  const double rho_min =
      a.rho_min_kgpm3 * std::pow(b.rho_min_kgpm3 / a.rho_min_kgpm3, dh);
  if (cos_pow == 0.0) {
    return rho_min;
  }
  const double rho_max =
      a.rho_max_kgpm3 * std::pow(b.rho_max_kgpm3 / a.rho_max_kgpm3, dh);
  return mix(rho_min, rho_max, cos_pow);
}

Eigen::Vector3d hp_bulge_apex(const Eigen::Vector3d& sun_dir_unit) {
  // eq:hp:apex -- rotate the Sun direction by the lag about the polar
  // (+z) axis: right ascension advances by 30 deg, declination unchanged.
  const double c = std::cos(kLagRad);
  const double s = std::sin(kLagRad);
  return Eigen::Vector3d(sun_dir_unit.x() * c - sun_dir_unit.y() * s,
                         sun_dir_unit.x() * s + sun_dir_unit.y() * c,
                         sun_dir_unit.z());
}

double geodetic_altitude(const Eigen::Vector3d& r_ecef_m, double a_m,
                         double inv_f) {
  if (!r_ecef_m.allFinite() || !(a_m > 0.0) || !(inv_f > 1.0)) {
    throw std::domain_error("geodetic_altitude: invalid argument");
  }
  // eq:hp:geodetic -- Bowring's closed form with one fixed refinement
  // pass. Fixed pass count keeps the evaluation deterministic (D-10);
  // Bowring's cubic convergence makes two passes sub-millimetre over this
  // model's altitude domain (ch:harrispriester, derivation).
  const double f = 1.0 / inv_f;
  const double b_m = a_m * (1.0 - f);
  const double e2 = f * (2.0 - f);
  const double ep2 = e2 / (1.0 - e2);
  const double p = std::sqrt(r_ecef_m.x() * r_ecef_m.x() +
                             r_ecef_m.y() * r_ecef_m.y());
  const double z = r_ecef_m.z();
  // Starting parametric latitude, then two identical Bowring passes.
  double u = std::atan2(z * a_m, p * b_m);
  double phi = 0.0;
  for (int pass = 0; pass < 2; ++pass) {
    const double su = std::sin(u);
    const double cu = std::cos(u);
    phi = std::atan2(z + ep2 * b_m * su * su * su,
                     p - e2 * a_m * cu * cu * cu);
    u = std::atan2(b_m * std::sin(phi), a_m * std::cos(phi));
  }
  const double sphi = std::sin(phi);
  const double n_rad = a_m / std::sqrt(1.0 - e2 * sphi * sphi);
  // Off the poles use the robust p/cos(phi) form; near them fall back to
  // the z/sin(phi) form (cos(phi) -> 0 would lose all precision).
  if (std::fabs(sphi) < 0.99) {
    return p / std::cos(phi) - n_rad;
  }
  return z / sphi - n_rad * (1.0 - e2);
}

const HpNode* hp_table(std::size_t* count) {
  *count = kTableSize;
  return kTable;
}

}  // namespace models
}  // namespace star
