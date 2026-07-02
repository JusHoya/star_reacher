// Deterministic random-number generation for the star:: core (PRD D-9).
//
// The core owns its PRNG end to end because C++ standard-library engines and
// distributions are implementation-defined: the same seed produces different
// streams on different standard libraries, which would break the bit-identity
// contract (PRD D-10). The composition is:
//
//   stream key  = FNV-1a-64(stream_name)            [IETF draft-eastlake-fnv]
//   expansion   = SplitMix64(master_seed XOR key)   [Vigna, splitmix64.c]
//   generator   = PCG64 (pcg_setseq_128 XSL-RR 128/64) [O'Neill, HMC-CS-2014-0905]
//   normals     = trigonometric Box-Muller           [Box & Muller, 1958]
//
// Named streams (e.g. "sensors.imu", "dispersions.mass") give each subsystem
// an independent, reproducible sequence derived from one master seed, so
// adding a consumer never perturbs another consumer's draws.
#ifndef STAR_RNG_HPP
#define STAR_RNG_HPP

#include <cstdint>
#include <string_view>

namespace star {
namespace rng {

// Unsigned 128-bit integer emulated as two 64-bit halves. PCG64's state and
// multiplier are 128-bit; MSVC has no __int128, so the arithmetic is spelled
// out portably (pure unsigned 64/32-bit operations, identical on every
// conforming platform - a determinism requirement, not an optimization).
struct U128 {
  std::uint64_t hi;
  std::uint64_t lo;
};

// (a * b) mod 2^128 and (a + b) mod 2^128; unsigned wraparound is defined
// behavior and is exactly the modular arithmetic PCG specifies.
U128 mul_u128(U128 a, U128 b);
U128 add_u128(U128 a, U128 b);

// FNV-1a 64-bit hash of the UTF-8 bytes of `data`.
// Parameters per IETF draft-eastlake-fnv: offset basis 14695981039346656037
// (0xCBF29CE484222325), prime 1099511628211 (0x100000001B3).
std::uint64_t fnv1a64(std::string_view data);

// SplitMix64 (Vigna's reference splitmix64.c, public domain; algorithm from
// Steele, Lea, and Flood, "Fast Splittable Pseudorandom Number Generators",
// OOPSLA 2014). Used only to expand one 64-bit seed into the 256 bits PCG64
// needs; never used as the simulation generator itself.
class SplitMix64 {
 public:
  explicit SplitMix64(std::uint64_t seed) : state_(seed) {}
  std::uint64_t next();

 private:
  std::uint64_t state_;
};

// PCG64: pcg_setseq_128 with the XSL-RR 128/64 output function (O'Neill,
// "PCG: A Family of Simple Fast Space-Efficient Statistically Good Algorithms
// for Random Number Generation", HMC-CS-2014-0905; reference implementation
// at pcg-random.org). Bit-compatible with numpy.random.PCG64 for the same
// (state, inc) pair, which is what lets the Python side cross-validate the
// core's streams (Phase 1 contract section 4).
class Pcg64 {
 public:
  // Reference `srandom` seeding: state=0; inc=(initseq<<1)|1; step;
  // state+=initstate; step.
  Pcg64(U128 initstate, U128 initseq);

  // Advance one step (state = state*MULT + inc), then emit
  // rotr64(hi ^ lo, state >> 122) from the post-step state - the reference
  // generation order (step first, output second).
  std::uint64_t next();

 private:
  void step();

  U128 state_;
  U128 inc_;
};

// Derive the named-stream generator for (master_seed, stream_name) per the
// Phase 1 contract section 4: FNV-1a keys the name, SplitMix64 expands
// master_seed XOR key into sm1..sm4, and PCG64 is seeded with
// initstate=(sm1<<64)|sm2, initseq=(sm3<<64)|sm4.
Pcg64 make_stream(std::uint64_t master_seed, std::string_view stream_name);

// Map a u64 draw to a double in [0, 1): (x >> 11) * 2^-53. The top 53 bits
// fill the binary64 significand exactly, so the mapping is exact and
// platform-independent (same construction as the PCG and NumPy references).
double u64_to_unit(std::uint64_t x);

// Standard normal deviates via the basic trigonometric Box-Muller transform
// (G. E. P. Box and M. E. Muller, "A Note on the Generation of Random Normal
// Deviates", Annals of Mathematical Statistics 29(2), 1958). Chosen over
// std::normal_distribution because the standard leaves distribution
// algorithms implementation-defined (D-9 rationale).
//
// Draw-consumption pattern (normative, Phase 1 contract section 4): each PAIR
// of normals consumes exactly two u64 draws from the underlying generator.
//   draw x1 -> u1 = 1.0 - u64_to_unit(x1)   (in (0,1]; avoids log(0))
//   draw x2 -> u2 = u64_to_unit(x2)         (in [0,1))
//   z0 = sqrt(-2 ln u1) * cos(2 pi u2)   returned first
//   z1 = sqrt(-2 ln u1) * sin(2 pi u2)   cached, returned by the next call
// An odd number of requested normals therefore consumes the same two draws as
// the even count above it.
class NormalSampler {
 public:
  explicit NormalSampler(Pcg64 generator) : generator_(generator) {}
  double next();

 private:
  Pcg64 generator_;
  double cached_ = 0.0;
  bool has_cached_ = false;
};

}  // namespace rng
}  // namespace star

#endif  // STAR_RNG_HPP
