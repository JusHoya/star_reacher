// Mars piecewise-exponential atmosphere (FR-8, PRD assumption A-3).
// Derivation and validation table: docs/mathlib/chapters/
// mars_atmosphere.tex (ch:marsatmosphere).
//
// CONFIDENCE: LOW -- provenance provisional per PRD A-3. Node values are
// samples of the NASA Glenn Research Center "Mars Atmosphere Model"
// curve fits (eq:mars:glenn; educational fit to Mars Global Surveyor
// data, retrieved 2026-07-02); the golden manifest carries the same flag.
#include "star/models/atmosphere_mars.hpp"

#include <cmath>
#include <stdexcept>

namespace star {
namespace models {

namespace {

// Committed node densities at 5 km spacing, 0-100 km (eq:mars:glenn
// evaluated by tests/golden/atmosphere/generate.py). Hexadecimal
// floating literals so the compiled table is bit-identical to the
// committed golden mars_nodes.toml, whose hex fields are normative
// (ATM-MARS-NODES gates the equality); the decimal comments are
// readability aids only.
constexpr MarsNode kNodes[] = {
    {0.0, 0x1.ec7fa0fce9ad5p-7},       // 0.015029862983124623
    {5000.0, 0x1.40a3c68595509p-7},    // 0.0097851485581182
    {10000.0, 0x1.aa2ba0eb4573bp-8},   // 0.006502844607231324
    {15000.0, 0x1.1dad4120bbce6p-8},   // 0.004359081650121532
    {20000.0, 0x1.8002264416184p-9},   // 0.002929751559389073
    {25000.0, 0x1.02d99025afa4ap-9},   // 0.0019748676739646297
    {30000.0, 0x1.5e1c86936c5ddp-10},  // 0.0013355691088289192
    {35000.0, 0x1.db4ba318f594ap-11},  // 0.000906554140892615
    {40000.0, 0x1.43f814fd34db9p-11},  // 0.0006179219632493272
    {45000.0, 0x1.bbc1cd94a0d51p-12},  // 0.00042319969478673165
    {50000.0, 0x1.3198a73ef6063p-12},  // 0.0002914393443780359
    {55000.0, 0x1.a79d72915f352p-13},  // 0.00020199538679923718
    {60000.0, 0x1.27d83546f5083p-13},  // 0.000141069680664633
    {65000.0, 0x1.a1028a114ae8dp-14},  // 9.942291241731901e-05
    {70000.0, 0x1.2932254a8c6c8p-14},  // 7.085701978650904e-05
    {75000.0, 0x1.ad9093321c69ep-15},  // 5.1208108190159795e-05
    {80000.0, 0x1.3c121675cec00p-15},  // 3.767855825605876e-05
    {85000.0, 0x1.dc6ab4aa65bfdp-16},  // 2.8396655275732947e-05
    {90000.0, 0x1.7359b1ab7264ep-16},  // 2.2134206728618905e-05
    {95000.0, 0x1.3085e625ecf09p-16},  // 1.8150987805844727e-05
    {100000.0, 0x1.0ffb4d4537243p-16},  // 1.62113695026825e-05
};
constexpr std::size_t kNumNodes = sizeof(kNodes) / sizeof(kNodes[0]);
constexpr std::size_t kNumSegments = kNumNodes - 1;

// Engineering guard, not a physical datum: -8 km sits comfortably below
// the deepest Martian terrain, so any input beyond it is certainly an
// error upstream (ch:marsatmosphere, domain section).
constexpr double kMinZM = -8000.0;

// Per-segment inverse scale lengths k_i (eq:mars:scaleheight), computed
// once from the committed nodes in fixed order and cached. Deriving them
// from adjacent nodes is what makes the profile continuous at every
// boundary by construction (Phase 3 exit criterion 9b).
struct SegmentCoeffs {
  double k[kNumSegments];
};

const SegmentCoeffs& segment_coeffs() {
  static const SegmentCoeffs coeffs = [] {
    SegmentCoeffs c{};
    for (std::size_t i = 0; i < kNumSegments; ++i) {
      c.k[i] = (std::log(kNodes[i + 1].rho_kgpm3) -
                std::log(kNodes[i].rho_kgpm3)) /
               (kNodes[i + 1].z_m - kNodes[i].z_m);
    }
    return c;
  }();
  return coeffs;
}

}  // namespace

double mars_density(double z_m) {
  if (!std::isfinite(z_m) || z_m < kMinZM) {
    throw std::domain_error(
        "mars_atmosphere: altitude below the -8 km guard");
  }
  // Fixed-order scan for the segment whose start node satisfies
  // z_i <= z; below node 0 the first segment extends downward, and at or
  // above the last node the final segment extends upward (the documented
  // extrapolations).
  std::size_t i = 0;
  while (i < kNumSegments - 1 && z_m >= kNodes[i + 1].z_m) {
    ++i;
  }
  // The 100 km table top is the one node that is never a segment start;
  // return it directly so node exactness holds there too (bit-exact node
  // return is exit criterion 9a).
  if (z_m == kNodes[i + 1].z_m) {
    return kNodes[i + 1].rho_kgpm3;
  }
  // eq:mars:segment -- piecewise exponential. At z == z_i the exponent is
  // identically zero and exp(0) == 1, so the committed node density is
  // returned bit-exactly.
  return kNodes[i].rho_kgpm3 *
         std::exp(segment_coeffs().k[i] * (z_m - kNodes[i].z_m));
}

const MarsNode* mars_nodes(std::size_t* count) {
  *count = kNumNodes;
  return kNodes;
}

}  // namespace models
}  // namespace star
