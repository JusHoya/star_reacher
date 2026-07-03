// U.S. Standard Atmosphere 1976 (FR-8). Derivation and validation table:
// docs/mathlib/chapters/ussa76.tex (ch:ussa76). Every constant below is a
// USSA76-defining value cited at its definition (project convention); they
// are private to this model on purpose -- they define the 1976 document's
// atmosphere, not the project physical-constant set (star/constants.hpp).
#include "star/models/atmosphere_ussa76.hpp"

#include <cmath>
#include <stdexcept>

namespace star {
namespace models {

namespace {

// Defining constants, U.S. Standard Atmosphere 1976 (NOAA-S/T 76-1562 /
// NASA-TM-X-74335), Part 1, Table 2 (p. 2) unless noted.
constexpr double kT0K = 288.15;        // sea-level temperature [K]
constexpr double kP0Pa = 101325.0;     // sea-level pressure [Pa]
constexpr double kG0p = 9.80665;       // g0' [m^2/(s^2 m')]
constexpr double kRstar = 8314.32;     // R* [N m/(kmol K)] (8.31432e3)
constexpr double kM0 = 28.9644;        // sea-level molecular weight
                                       //   [kg/kmol] (Part 1, p. 9)
constexpr double kR0M = 6356766.0;     // effective Earth radius [m]
                                       //   (Part 1, p. 8)
constexpr double kGamma = 1.40;        // ratio of specific heats (exact)

// Part 1, Table 4 (p. 3): geopotential layer bases H_b [m'] and
// molecular-scale temperature gradients L_M,b [K/m'] (eq:ussa76:tm).
constexpr int kNumLayers = 7;
constexpr double kHb[kNumLayers + 1] = {0.0,     11000.0, 20000.0,
                                        32000.0, 47000.0, 51000.0,
                                        71000.0, 84852.0};
constexpr double kLb[kNumLayers] = {-0.0065, 0.0,     0.0010, 0.0028,
                                    0.0,     -0.0028, -0.0020};

// The geometric branch point. The document equates H_7 = 84852 m' with
// 86 km geometric as a rounded convention (H(86000 m) = 84852.046 m');
// branching on z at exactly 86 km lets the layer-7 formulas extend ~5 cm
// of geopotential past H_7 so the analytic and tabulated regions meet
// without a gap (ch:ussa76, domain section).
constexpr double kBranchZM = 86000.0;
constexpr double kMinZM = -5000.0;   // Table I lower limit
constexpr double kMaxZM = 1.0e6;     // Table I upper limit

// Committed 86-1000 km density node grid, transcribed from USSA76 Part 4,
// Table I (geometric altitude, metric units), document pages 68-73. The
// committed golden tests/golden/atmosphere/ussa76_upper_nodes.toml is the
// independent transcription check copy; ATM-USSA76-UPPER-NODES holds the
// two bit-identical. Grid spacing follows the log-density curvature so the
// between-node interpolation error stays below ~0.1 % (ch:ussa76,
// eq:ussa76:interp error budget).
constexpr Ussa76Node kUpperNodes[] = {
    {86000.0, 6.958e-6},   {87000.0, 5.824e-6},   {88000.0, 4.875e-6},
    {89000.0, 4.081e-6},   {90000.0, 3.416e-6},   {91000.0, 2.860e-6},
    {92000.0, 2.393e-6},   {93000.0, 2.000e-6},   {94000.0, 1.670e-6},
    {95000.0, 1.393e-6},   {96000.0, 1.162e-6},   {97000.0, 9.685e-7},
    {98000.0, 8.071e-7},   {99000.0, 6.725e-7},   {100000.0, 5.604e-7},
    {102000.0, 3.935e-7},  {104000.0, 2.769e-7},  {106000.0, 1.954e-7},
    {108000.0, 1.381e-7},  {110000.0, 9.708e-8},  {112000.0, 6.838e-8},
    {114000.0, 4.975e-8},  {116000.0, 3.720e-8},  {118000.0, 2.847e-8},
    {120000.0, 2.222e-8},  {122000.0, 1.767e-8},  {124000.0, 1.428e-8},
    {126000.0, 1.171e-8},  {128000.0, 9.717e-9},  {130000.0, 8.152e-9},
    {132000.0, 6.904e-9},  {134000.0, 5.897e-9},  {136000.0, 5.074e-9},
    {138000.0, 4.396e-9},  {140000.0, 3.831e-9},  {142000.0, 3.358e-9},
    {144000.0, 2.958e-9},  {146000.0, 2.618e-9},  {148000.0, 2.326e-9},
    {150000.0, 2.076e-9},  {155000.0, 1.585e-9},  {160000.0, 1.233e-9},
    {165000.0, 9.750e-10}, {170000.0, 7.815e-10}, {175000.0, 6.339e-10},
    {180000.0, 5.194e-10}, {185000.0, 4.295e-10}, {190000.0, 3.581e-10},
    {195000.0, 3.006e-10}, {200000.0, 2.541e-10}, {205000.0, 2.160e-10},
    {210000.0, 1.846e-10}, {215000.0, 1.585e-10}, {220000.0, 1.367e-10},
    {225000.0, 1.184e-10}, {230000.0, 1.029e-10}, {235000.0, 8.979e-11},
    {240000.0, 7.858e-11}, {245000.0, 6.898e-11}, {250000.0, 6.073e-11},
    {255000.0, 5.360e-11}, {260000.0, 4.742e-11}, {265000.0, 4.206e-11},
    {270000.0, 3.738e-11}, {275000.0, 3.329e-11}, {280000.0, 2.971e-11},
    {285000.0, 2.656e-11}, {290000.0, 2.378e-11}, {295000.0, 2.133e-11},
    {300000.0, 1.916e-11}, {310000.0, 1.552e-11}, {320000.0, 1.264e-11},
    {340000.0, 8.503e-12}, {360000.0, 5.805e-12}, {380000.0, 4.013e-12},
    {400000.0, 2.803e-12}, {420000.0, 1.975e-12}, {440000.0, 1.402e-12},
    {460000.0, 1.002e-12}, {480000.0, 7.208e-13}, {500000.0, 5.215e-13},
    {525000.0, 3.509e-13}, {550000.0, 2.384e-13}, {575000.0, 1.637e-13},
    {600000.0, 1.137e-13}, {625000.0, 7.998e-14}, {650000.0, 5.712e-14},
    {675000.0, 4.148e-14}, {700000.0, 3.070e-14}, {725000.0, 2.318e-14},
    {750000.0, 1.788e-14}, {775000.0, 1.410e-14}, {800000.0, 1.136e-14},
    {825000.0, 9.339e-15}, {850000.0, 7.824e-15}, {875000.0, 6.664e-15},
    {900000.0, 5.759e-15}, {925000.0, 5.038e-15}, {950000.0, 4.453e-15},
    {975000.0, 3.968e-15}, {1000000.0, 3.561e-15},
};
constexpr std::size_t kNumUpperNodes =
    sizeof(kUpperNodes) / sizeof(kUpperNodes[0]);

// Layer base values (T_M,b, P_b), computed once in fixed order b = 0..6 by
// successive application of eq:ussa76:gradient / eq:ussa76:isothermal at
// each layer top (D-10: fixed evaluation order; single-threaded core, so
// the function-local static involves no synchronization concerns beyond
// the guaranteed-once C++11 semantics).
struct LayerBases {
  double t[kNumLayers];
  double p[kNumLayers];
};

const LayerBases& layer_bases() {
  static const LayerBases bases = [] {
    LayerBases b{};
    double t = kT0K;
    double p = kP0Pa;
    for (int i = 0; i < kNumLayers; ++i) {
      b.t[i] = t;
      b.p[i] = p;
      const double dh = kHb[i + 1] - kHb[i];
      if (kLb[i] == 0.0) {
        // eq:ussa76:isothermal at the layer top
        p *= std::exp(-kG0p * kM0 * dh / (kRstar * t));
      } else {
        // eq:ussa76:gradient at the layer top
        const double t_next = t + kLb[i] * dh;
        p *= std::pow(t / t_next, kG0p * kM0 / (kRstar * kLb[i]));
        t = t_next;
      }
    }
    return b;
  }();
  return bases;
}

Ussa76State analytic_state(double z_m) {
  // eq:ussa76:geopotential -- geopotential altitude, Gamma = 1 m'/m.
  const double h = kR0M * z_m / (kR0M + z_m);
  // Fixed-order layer scan (the branch guard guarantees h is below the
  // extended layer-7 top).
  int b = 0;
  while (b < kNumLayers - 1 && h > kHb[b + 1]) {
    ++b;
  }
  const LayerBases& bases = layer_bases();
  const double dh = h - kHb[b];
  // eq:ussa76:tm -- molecular-scale temperature, linear in the layer.
  const double tm = bases.t[b] + kLb[b] * dh;
  double p;
  if (kLb[b] == 0.0) {
    // eq:ussa76:isothermal
    p = bases.p[b] * std::exp(-kG0p * kM0 * dh / (kRstar * bases.t[b]));
  } else {
    // eq:ussa76:gradient
    p = bases.p[b] * std::pow(bases.t[b] / tm, kG0p * kM0 / (kRstar * kLb[b]));
  }
  Ussa76State s{};
  s.temperature_K = tm;
  s.pressure_Pa = p;
  // eq:ussa76:density
  s.density_kgpm3 = p * kM0 / (kRstar * tm);
  // eq:ussa76:speedofsound
  s.speed_of_sound_mps = std::sqrt(kGamma * kRstar * tm / kM0);
  return s;
}

void check_range(double z_m) {
  if (!std::isfinite(z_m) || z_m < kMinZM || z_m > kMaxZM) {
    throw std::domain_error(
        "ussa76: geometric altitude outside [-5 km, 1000 km]");
  }
}

}  // namespace

Ussa76State ussa76_state(double z_m) {
  check_range(z_m);
  if (z_m >= kBranchZM) {
    // Temperature/pressure above 86 km would require the document's
    // species number-density integrals (out of scope, ch:ussa76);
    // refusing beats returning a fabricated value (DX-2).
    throw std::domain_error(
        "ussa76: full thermodynamic state is only defined below 86 km");
  }
  return analytic_state(z_m);
}

double ussa76_density(double z_m) {
  check_range(z_m);
  if (z_m < kBranchZM) {
    return analytic_state(z_m).density_kgpm3;
  }
  // Fixed-order scan for the segment whose start node satisfies z_i <= z.
  // The scan caps at the second-to-last node, so the top of the table is
  // the final segment's end point rather than a segment start.
  std::size_t i = 0;
  while (i < kNumUpperNodes - 2 && z_m >= kUpperNodes[i + 1].z_m) {
    ++i;
  }
  // eq:ussa76:interp -- log-linear (piecewise-exponential) interpolation.
  // At z == z_i the offset is exactly zero and exp(0) == 1, so the node
  // density is returned bit-exactly (ATM-USSA76-UPPER-NODES relies on it).
  // The 1000 km table top is the one node that is never a segment start;
  // return it directly so node exactness holds there too.
  const Ussa76Node& a = kUpperNodes[i];
  const Ussa76Node& b = kUpperNodes[i + 1];
  if (z_m == b.z_m) {
    return b.rho_kgpm3;
  }
  const double slope =
      (std::log(b.rho_kgpm3) - std::log(a.rho_kgpm3)) / (b.z_m - a.z_m);
  return a.rho_kgpm3 * std::exp(slope * (z_m - a.z_m));
}

const Ussa76Node* ussa76_upper_nodes(std::size_t* count) {
  *count = kNumUpperNodes;
  return kUpperNodes;
}

}  // namespace models
}  // namespace star
