// Ephemeris evaluator tests (FR-4, FR-22 layer 1): bit-level agreement with
// the committed golden vectors, Chebyshev record-boundary continuity, and
// the error paths (unknown body, out-of-span epoch, malformed file). Test
// IDs are cited by the math-library validation table; do not rename them.
//
// The golden values in tests/golden/ephemeris/state_bitlevel.toml were
// produced by the Python reference evaluator over the SAME committed excerpt
// file this test loads, and cross-checked against jplephem at generation
// time (provenance: tests/golden/ephemeris/manifest.toml). The C++ evaluator
// implements the identical operation sequence under the D-10 flags (no FMA
// contraction, no fast-math), so agreement is required at bit level - any
// difference is an implementation divergence, not roundoff to tolerate.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "golden_io.hpp"
#include "star/ephemeris.hpp"
#include "vendor/doctest.h"

namespace {

const std::string kGoldenDir = STAR_GOLDEN_DIR;
const std::string kExcerpt = kGoldenDir + "/ephemeris/excerpt_de440s.sreph";
const std::string kBitlevel = kGoldenDir + "/ephemeris/state_bitlevel.toml";

bool same_bits(double a, double b) {
  return std::memcmp(&a, &b, sizeof a) == 0;
}

// Pull the (value, rate) triplets a golden case describes: body state in
// m and m/s for the state kinds, libration angles in rad and rad/s for the
// librations kinds. The golden file stores both under r_m / v_mps.
void evaluate_case(const star::Ephemeris& eph, const star_tests::GoldenCase& c,
                   Eigen::Vector3d* value, Eigen::Vector3d* rate) {
  const std::string kind = c.scalar("kind");
  const double tdb_s = star_tests::parse_hex_double(c.scalar("tdb_s"));
  if (kind == "librations" || kind == "librations_boundary") {
    const star::LibrationAngles a = eph.lunar_librations(tdb_s);
    *value = a.angles_rad;
    *rate = a.rates_radps;
  } else {
    const star::EphemerisState s = eph.state(c.scalar("body"), tdb_s);
    *value = s.r_m;
    *rate = s.v_mps;
  }
}

}  // namespace

TEST_CASE("ephemeris_bitlevel_golden") {
  // Every golden case (span endpoints, one mid-record epoch, and two record
  // boundaries per stored segment) must reproduce the Python reference
  // evaluator's output bit for bit. This pins the loader (verbatim
  // coefficient bytes), the record-selection rule, the recurrence, the
  // summation order, and the km->m scaling in one check.
  const star::Ephemeris eph = star::Ephemeris::load_file(kExcerpt);
  const auto cases = star_tests::load_golden_cases(kBitlevel);
  REQUIRE(cases.size() >= 40);
  for (const auto& c : cases) {
    const std::string body = c.scalar("body");
    const std::string epoch = c.scalar("epoch_iso");
    CAPTURE(body);
    CAPTURE(epoch);
    Eigen::Vector3d value;
    Eigen::Vector3d rate;
    evaluate_case(eph, c, &value, &rate);
    const auto& golden_value = c.array("r_m");
    const auto& golden_rate = c.array("v_mps");
    REQUIRE(golden_value.size() == 3);
    REQUIRE(golden_rate.size() == 3);
    for (int i = 0; i < 3; ++i) {
      const double gv = star_tests::parse_hex_double(golden_value[i]);
      const double gr = star_tests::parse_hex_double(golden_rate[i]);
      CAPTURE(i);
      CAPTURE(value[i]);
      CAPTURE(gv);
      CHECK(same_bits(value[i], gv));
      CAPTURE(rate[i]);
      CAPTURE(gr);
      CHECK(same_bits(rate[i], gr));
    }
  }
}

TEST_CASE("ephemeris_segment_boundary_continuity") {
  // The *_boundary golden epochs lie exactly on interior Chebyshev record
  // boundaries. The selection rule assigns a boundary epoch to the record
  // that begins there; nextafter(t, -inf) falls in the record ending there.
  // DE Type 2 fits are constrained to be position- and velocity-continuous
  // at record boundaries (Newhall 1989), so the two evaluations may differ
  // only by the true motion across one epoch ulp plus fit-level mismatch.
  //
  // Bound derivation (recorded per FR-22): one ulp of t near the span end
  // (t ~ 1.9e9 s) is ~2.4e-7 s; the fastest stored quantity (venus_bary,
  // ~35 km/s about the SSB) moves ~8.4e-3 m in that interval, which matched
  // the measured worst case exactly (8.375e-3 m, generation log 2026-07-02;
  // velocity 2.7e-9 m/s ~ SSB-frame acceleration over the same interval;
  // librations 4.5e-12 rad / 9.1e-22 rad/s). Bounds sit roughly a decade
  // above the measurement: a wrong record selection displaces the evaluation
  // by whole-record distances (>= kilometers / >= 1e-6 rad) and fails
  // decisively.
  const star::Ephemeris eph = star::Ephemeris::load_file(kExcerpt);
  const auto cases = star_tests::load_golden_cases(kBitlevel);
  int boundary_cases = 0;
  for (const auto& c : cases) {
    const std::string kind = c.scalar("kind");
    if (kind != "state_boundary" && kind != "librations_boundary") {
      continue;
    }
    ++boundary_cases;
    const std::string body = c.scalar("body");
    const std::string epoch = c.scalar("epoch_iso");
    CAPTURE(body);
    CAPTURE(epoch);
    const double tb = star_tests::parse_hex_double(c.scalar("tdb_s"));
    const double tb_minus = std::nextafter(tb, -1.0);
    REQUIRE(tb_minus < tb);
    Eigen::Vector3d v1;
    Eigen::Vector3d r1;
    evaluate_case(eph, c, &v1, &r1);
    Eigen::Vector3d v0;
    Eigen::Vector3d r0;
    if (kind == "librations_boundary") {
      const star::LibrationAngles a = eph.lunar_librations(tb_minus);
      v0 = a.angles_rad;
      r0 = a.rates_radps;
      CHECK((v1 - v0).norm() < 1e-9);   // rad; measured 4.5e-12
      CHECK((r1 - r0).norm() < 1e-19);  // rad/s; measured 9.1e-22
    } else {
      const star::EphemerisState s = eph.state(body, tb_minus);
      v0 = s.r_m;
      r0 = s.v_mps;
      CHECK((v1 - v0).norm() < 5e-2);  // m; measured 8.4e-3
      CHECK((r1 - r0).norm() < 5e-8);  // m/s; measured 2.7e-9
    }
  }
  // Two boundaries per stored segment (7 SPK bodies + librations).
  CHECK(boundary_cases == 16);
}

TEST_CASE("ephemeris_error_paths") {
  const star::Ephemeris eph = star::Ephemeris::load_file(kExcerpt);

  // Probe epochs chosen from the repack constants, not the file header: the
  // committed excerpt stores non-contiguous record runs, so its header span
  // (an intersection) is deliberately not used for anything here. 6.0e8 s
  // TDB (2019-01) precedes every stored record; 1.9e9 s TDB (2060-03) is
  // past every trimmed segment end (latest: mars_bary, 1 895 918 400 s).
  const double kBeforeAll = 6.0e8;
  const double kAfterAll = 1.9e9;

  SUBCASE("unknown body throws invalid_argument naming the body") {
    const double t = 631108800.0;  // 2020-01-01 TDB, inside every segment
    CHECK_THROWS_AS(eph.state("phobos", t), std::invalid_argument);
    try {
      eph.state("phobos", t);
      FAIL("expected std::invalid_argument");
    } catch (const std::invalid_argument& exc) {
      const std::string msg = exc.what();
      CHECK(msg.find("phobos") != std::string::npos);
      CHECK(msg.find("moon") != std::string::npos);  // lists available bodies
    }
  }

  SUBCASE("out-of-span epochs throw out_of_range, never extrapolate") {
    CHECK_THROWS_AS(eph.state("sun", kBeforeAll), std::out_of_range);
    CHECK_THROWS_AS(eph.state("sun", kAfterAll), std::out_of_range);
    CHECK_THROWS_AS(eph.moon_geocentric(kAfterAll), std::out_of_range);
    CHECK_THROWS_AS(eph.lunar_librations(kBeforeAll), std::out_of_range);
    // The excerpt stores non-contiguous record runs, so an epoch in a gap
    // between two stored runs of the same body must be refused rather than
    // served from a neighboring run: 2021-01-01 TDB lies between the runs
    // around the 2019-12-31 span start and the first interior test epoch
    // (2022-03-21) for every body.
    const double kInGap = 662644800.0;  // 2021-01-01T00:00:00 TDB
    CHECK_THROWS_AS(eph.state("sun", kInGap), std::out_of_range);
    CHECK_THROWS_AS(eph.state("moon", kInGap), std::out_of_range);
    try {
      eph.state("sun", kAfterAll);
      FAIL("expected std::out_of_range");
    } catch (const std::out_of_range& exc) {
      const std::string msg = exc.what();
      CHECK(msg.find("sun") != std::string::npos);
      CHECK(msg.find("extrapolate") != std::string::npos);
    }
  }

  SUBCASE("malformed files are rejected with specific errors") {
    std::ifstream in(kExcerpt, std::ios::binary);
    REQUIRE(in.good());
    std::vector<char> bytes((std::istreambuf_iterator<char>(in)),
                            std::istreambuf_iterator<char>());
    REQUIRE(bytes.size() > 200);

    auto write_temp = [](const std::string& name, const std::vector<char>& data) {
      // The test binary's working directory (the build tree) is writable;
      // unique names keep parallel ctest invocations from colliding.
      std::ofstream out(name, std::ios::binary | std::ios::trunc);
      out.write(data.data(), static_cast<std::streamsize>(data.size()));
      return name;
    };

    std::vector<char> bad_magic = bytes;
    bad_magic[0] ^= 0x40;
    const std::string p1 = write_temp("ephemeris_test_bad_magic.sreph", bad_magic);
    CHECK_THROWS_AS(star::Ephemeris::load_file(p1), std::runtime_error);

    std::vector<char> truncated_dir(bytes.begin(), bytes.begin() + 100);
    const std::string p2 = write_temp("ephemeris_test_trunc_dir.sreph", truncated_dir);
    CHECK_THROWS_AS(star::Ephemeris::load_file(p2), std::runtime_error);

    std::vector<char> truncated_coeffs(bytes.begin(),
                                       bytes.begin() + static_cast<long>(bytes.size() / 2));
    const std::string p3 = write_temp("ephemeris_test_trunc_coeffs.sreph", truncated_coeffs);
    CHECK_THROWS_AS(star::Ephemeris::load_file(p3), std::runtime_error);

    CHECK_THROWS_AS(star::Ephemeris::load_file("ephemeris_test_does_not_exist.sreph"),
                    std::runtime_error);

    std::remove(p1.c_str());
    std::remove(p2.c_str());
    std::remove(p3.c_str());
  }
}
