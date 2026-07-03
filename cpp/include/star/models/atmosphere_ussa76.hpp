// U.S. Standard Atmosphere 1976 (FR-8: Earth launch atmosphere).
//
// Below 86 km geometric altitude the model is fully analytic from the
// USSA76 defining constants and returns temperature, pressure, density,
// and speed of sound (the Phase 4 aerodynamics model derives Mach number
// from the speed of sound; propulsion derives back-pressure thrust from
// the pressure). From 86 km to 1000 km it returns density only, by
// log-linear interpolation of a committed node grid transcribed from the
// published USSA76 tables -- the kinetic species integrals the document
// uses up there are deliberately out of scope, and orbit-decay drag uses
// the Harris-Priester model (star/models/atmosphere_hp.hpp) instead.
//
// Math-library traceability (FR-29): the derivation lives in the USSA76
// chapter of docs/mathlib (ch:ussa76); the implementation echoes its
// equation labels at the corresponding code in atmosphere_ussa76.cpp.
#ifndef STAR_MODELS_ATMOSPHERE_USSA76_HPP
#define STAR_MODELS_ATMOSPHERE_USSA76_HPP

#include <cstddef>

namespace star {
namespace models {

// Thermodynamic state below 86 km. temperature_K is the USSA76
// molecular-scale temperature T_M, which equals the printed Table I
// temperature everywhere below 86 km (the rigorously corrected kinetic
// temperature is at most 0.04 % lower between 80 and 86 km; ch:ussa76,
// domain section).
struct Ussa76State {
  double temperature_K;
  double pressure_Pa;
  double density_kgpm3;
  double speed_of_sound_mps;
};

// Full analytic state for geometric altitude z in [-5 km, 86 km).
// Throws std::domain_error outside that range (including at and above
// 86 km, where only density exists in this model) and for non-finite z.
Ussa76State ussa76_state(double z_m);

// Density for geometric altitude z in [-5 km, 1000 km]: analytic below
// 86 km, committed-node log-linear interpolation at and above. Throws
// std::domain_error outside the range and for non-finite z.
double ussa76_density(double z_m);

// Read-only access to the compiled-in 86-1000 km node grid, exposed so
// the doctest suite can hold the compiled table and the committed golden
// transcription (tests/golden/atmosphere/ussa76_upper_nodes.toml)
// bit-identical (ATM-USSA76-UPPER-NODES).
struct Ussa76Node {
  double z_m;
  double rho_kgpm3;
};
const Ussa76Node* ussa76_upper_nodes(std::size_t* count);

}  // namespace models
}  // namespace star

#endif  // STAR_MODELS_ATMOSPHERE_USSA76_HPP
