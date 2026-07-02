// RNG golden-vector tests (FR-22 layer 1). The reference values come from
// tests/golden/rng/ - provenance and tolerances in that directory's
// manifest.toml. Test IDs are cited by the math-library validation tables;
// do not rename them.
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "golden_io.hpp"
#include "star/rng.hpp"
#include "vendor/doctest.h"

namespace {

// STAR_GOLDEN_DIR is injected by CMake and points at <repo>/tests/golden.
std::string golden_path(const char* file) {
  return std::string(STAR_GOLDEN_DIR) + "/rng/" + file;
}

}  // namespace

TEST_CASE("rng_splitmix64_golden") {
  const auto cases = star_tests::load_golden_cases(golden_path("splitmix64.toml"));
  REQUIRE(cases.size() == 4);
  for (const auto& c : cases) {
    CAPTURE(c.scalar("name"));
    star::rng::SplitMix64 sm(star_tests::parse_hex_u64(c.scalar("seed")));
    for (const std::string& expected : c.array("values")) {
      CHECK(sm.next() == star_tests::parse_hex_u64(expected));
    }
  }

  // FNV-1a feeds SplitMix64 in the stream-derivation path, so its goldens are
  // checked here alongside the expander they seed.
  const auto fnv_cases = star_tests::load_golden_cases(golden_path("fnv1a.toml"));
  REQUIRE(fnv_cases.size() == 6);
  for (const auto& c : fnv_cases) {
    CAPTURE(c.scalar("name"));
    CHECK(star::rng::fnv1a64(c.scalar("input")) ==
          star_tests::parse_hex_u64(c.scalar("hash")));
  }
}

TEST_CASE("rng_pcg64_golden") {
  const auto cases = star_tests::load_golden_cases(golden_path("pcg64.toml"));
  REQUIRE(cases.size() == 6);
  for (const auto& c : cases) {
    CAPTURE(c.scalar("name"));
    const std::string kind = c.scalar("kind");
    star::rng::Pcg64 gen =
        (kind == "raw")
            ? star::rng::Pcg64(
                  star::rng::U128{
                      star_tests::parse_hex_u64(c.scalar("initstate_hi")),
                      star_tests::parse_hex_u64(c.scalar("initstate_lo"))},
                  star::rng::U128{
                      star_tests::parse_hex_u64(c.scalar("initseq_hi")),
                      star_tests::parse_hex_u64(c.scalar("initseq_lo"))})
            : star::rng::make_stream(
                  star_tests::parse_hex_u64(c.scalar("master_seed")),
                  c.scalar("stream"));
    for (const std::string& expected : c.array("values")) {
      CHECK(gen.next() == star_tests::parse_hex_u64(expected));
    }
  }

  // Distinct-stream property (D-9): the same master seed must yield different
  // sequences for different stream names. Checked on the first golden draws
  // rather than fresh draws so the assertion is itself pinned to the goldens.
  const auto& imu = cases[2];
  const auto& mass = cases[3];
  REQUIRE(imu.scalar("name") == "stream_seed42_sensors_imu");
  REQUIRE(mass.scalar("name") == "stream_seed42_dispersions_mass");
  CHECK(imu.array("values")[0] != mass.array("values")[0]);
}

TEST_CASE("rng_box_muller_golden") {
  const auto cases = star_tests::load_golden_cases(golden_path("box_muller.toml"));
  REQUIRE(cases.size() == 3);
  for (const auto& c : cases) {
    CAPTURE(c.scalar("name"));
    star::rng::NormalSampler sampler(star::rng::make_stream(
        star_tests::parse_hex_u64(c.scalar("master_seed")), c.scalar("stream")));
    for (const std::string& expected_hex : c.array("values")) {
      const double expected = star_tests::parse_hex_double(expected_hex);
      const double got = sampler.next();
      // Tolerance from tests/golden/rng/manifest.toml: abs-or-rel 1e-13,
      // covering ulp-level libm (log/cos/sin/sqrt) spread across C runtimes
      // while still failing on any algorithmic deviation.
      CHECK(std::fabs(got - expected) <=
            1e-13 * std::max(1.0, std::fabs(expected)));
    }
  }
}
