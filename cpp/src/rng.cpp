// Implementation of the D-9 RNG stack. See star/rng.hpp for the algorithm
// citations; comments here explain choices the header does not cover.
#include "star/rng.hpp"

#include <cmath>

#include "star/constants.hpp"

namespace star {
namespace rng {

namespace {

// 64x64 -> 128-bit multiply from 32-bit partial products. Written out instead
// of using _umul128/__uint128_t so the same code path runs on MSVC, GCC, and
// Clang - one implementation to validate, no per-compiler intrinsic branches.
void mul64wide(std::uint64_t a, std::uint64_t b, std::uint64_t& hi,
               std::uint64_t& lo) {
  const std::uint64_t a_lo = a & 0xFFFFFFFFULL;
  const std::uint64_t a_hi = a >> 32;
  const std::uint64_t b_lo = b & 0xFFFFFFFFULL;
  const std::uint64_t b_hi = b >> 32;
  const std::uint64_t p0 = a_lo * b_lo;
  const std::uint64_t p1 = a_lo * b_hi;
  const std::uint64_t p2 = a_hi * b_lo;
  const std::uint64_t p3 = a_hi * b_hi;
  // mid cannot overflow: (2^32-1)^2 + 2*(2^32-1) < 2^64.
  const std::uint64_t mid = p1 + (p0 >> 32) + (p2 & 0xFFFFFFFFULL);
  lo = (mid << 32) | (p0 & 0xFFFFFFFFULL);
  hi = p3 + (p2 >> 32) + (mid >> 32);
}

// Rotate right; `(64 - r) & 63` keeps the left shift in [0, 63] so r == 0
// never produces the undefined `x << 64`.
std::uint64_t rotr64(std::uint64_t x, unsigned r) {
  return (x >> r) | (x << ((64u - r) & 63u));
}

// PCG64 default multiplier, pcg_setseq_128 (O'Neill reference implementation,
// pcg_variants.h: PCG_DEFAULT_MULTIPLIER_128 = 0x2360ED051FC65DA44385DF649FCCF645).
constexpr U128 kPcgMultiplier = {0x2360ED051FC65DA4ULL, 0x4385DF649FCCF645ULL};

}  // namespace

U128 mul_u128(U128 a, U128 b) {
  // (a.hi*2^64 + a.lo)(b.hi*2^64 + b.lo) mod 2^128: the a.hi*b.hi term shifts
  // past bit 127 and vanishes; cross terms land in the high half.
  U128 r;
  mul64wide(a.lo, b.lo, r.hi, r.lo);
  r.hi += a.hi * b.lo + a.lo * b.hi;
  return r;
}

U128 add_u128(U128 a, U128 b) {
  U128 r;
  r.lo = a.lo + b.lo;
  r.hi = a.hi + b.hi + (r.lo < a.lo ? 1u : 0u);
  return r;
}

std::uint64_t fnv1a64(std::string_view data) {
  // Parameters per IETF draft-eastlake-fnv (64-bit FNV-1a): xor the byte in
  // first, then multiply by the prime.
  std::uint64_t h = 14695981039346656037ULL;  // offset basis 0xCBF29CE484222325
  for (unsigned char c : data) {
    h ^= static_cast<std::uint64_t>(c);
    h *= 1099511628211ULL;  // FNV prime 0x100000001B3
  }
  return h;
}

std::uint64_t SplitMix64::next() {
  // Vigna's reference splitmix64.c, verbatim constants: the increment is the
  // 64-bit golden ratio; the finalizer is Stafford's variant 13 mix.
  state_ += 0x9E3779B97F4A7C15ULL;
  std::uint64_t z = state_;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

Pcg64::Pcg64(U128 initstate, U128 initseq) {
  // Reference pcg_setseq_128_srandom_r: forcing inc odd makes the LCG
  // full-period regardless of the caller's initseq.
  state_ = U128{0, 0};
  inc_.hi = (initseq.hi << 1) | (initseq.lo >> 63);
  inc_.lo = (initseq.lo << 1) | 1ULL;
  step();
  state_ = add_u128(state_, initstate);
  step();
}

void Pcg64::step() { state_ = add_u128(mul_u128(state_, kPcgMultiplier), inc_); }

std::uint64_t Pcg64::next() {
  // Reference pcg_setseq_128_xsl_rr_64_random_r: step first, then apply the
  // XSL-RR output function to the post-step state. `state >> 122` is the top
  // six bits, i.e. hi >> 58.
  step();
  const unsigned rot = static_cast<unsigned>(state_.hi >> 58);
  return rotr64(state_.hi ^ state_.lo, rot);
}

Pcg64 make_stream(std::uint64_t master_seed, std::string_view stream_name) {
  // XOR (not concatenation/hash-of-both) is the contract-specified combiner:
  // it keeps stream derivation a pure 64-bit operation with no allocation.
  SplitMix64 sm(master_seed ^ fnv1a64(stream_name));
  const std::uint64_t sm1 = sm.next();
  const std::uint64_t sm2 = sm.next();
  const std::uint64_t sm3 = sm.next();
  const std::uint64_t sm4 = sm.next();
  // initstate = (sm1 << 64) | sm2, initseq = (sm3 << 64) | sm4.
  return Pcg64(U128{sm1, sm2}, U128{sm3, sm4});
}

double u64_to_unit(std::uint64_t x) {
  // 2^-53 written as a hex literal so the constant is exact by construction.
  return static_cast<double>(x >> 11) * 0x1.0p-53;
}

double NormalSampler::next() {
  if (has_cached_) {
    has_cached_ = false;
    return cached_;
  }
  // Consumption order is normative (see header): x1 feeds the radius, x2 the
  // angle. u1 = 1 - u maps [0,1) to (0,1] so log(u1) is always finite.
  const double u1 = 1.0 - u64_to_unit(generator_.next());
  const double u2 = u64_to_unit(generator_.next());
  const double radius = std::sqrt(-2.0 * std::log(u1));
  const double angle = constants::TWO_PI * u2;
  cached_ = radius * std::sin(angle);
  has_cached_ = true;
  return radius * std::cos(angle);
}

}  // namespace rng
}  // namespace star
