// Mars piecewise-exponential atmosphere (FR-8, PRD assumption A-3).
//
// CONFIDENCE: LOW -- provenance provisional per PRD A-3. The committed
// node densities derive from the NASA Glenn Research Center "Mars
// Atmosphere Model" educational curve fits (no stated uncertainty or
// altitude datum); until a validated engineering provenance replaces
// them, results that depend on absolute Mars density are sensitivity
// studies, not predictions. Mars EDL is out of scope; the model serves
// aerobraking-class studies only.
//
// Math-library traceability (FR-29): the derivation lives in the Mars
// atmosphere chapter of docs/mathlib (ch:marsatmosphere); the
// implementation echoes its equation labels in atmosphere_mars.cpp.
#ifndef STAR_MODELS_ATMOSPHERE_MARS_HPP
#define STAR_MODELS_ATMOSPHERE_MARS_HPP

#include <cstddef>

namespace star {
namespace models {

// Density [kg/m^3] at altitude z_m above the source model's surface
// reference. Piecewise exponential between committed nodes (0-100 km,
// 5 km spacing); evaluation at a node returns the committed value
// bit-exactly, and the profile is continuous at every segment boundary
// by construction. Above 100 km the topmost segment's exponential is
// continued (documented non-physical continuity aid); below 0 the first
// segment's exponential is continued to -8 km. Throws std::domain_error
// below -8 km and for non-finite z.
double mars_density(double z_m);

// Read-only access to the compiled-in node table, exposed so the doctest
// suite can hold the compiled table and the committed golden
// (tests/golden/atmosphere/mars_nodes.toml) bit-identical
// (ATM-MARS-NODES).
struct MarsNode {
  double z_m;
  double rho_kgpm3;
};
const MarsNode* mars_nodes(std::size_t* count);

}  // namespace models
}  // namespace star

#endif  // STAR_MODELS_ATMOSPHERE_MARS_HPP
