// Time-system golden-vector and property tests (FR-2, D-6; FR-22 layers 1
// and 2). Reference values come from tests/golden/time/ - provenance and
// tolerances in that directory's manifest.toml. Test IDs are cited by the
// math-library validation table (ch:time); do not rename them.
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "golden_io.hpp"
#include "star/time.hpp"
#include "vendor/doctest.h"

namespace {

// STAR_GOLDEN_DIR is injected by CMake and points at <repo>/tests/golden.
std::string golden_path(const char* file) {
  return std::string(STAR_GOLDEN_DIR) + "/time/" + file;
}

std::int64_t parse_int(const std::string& s) { return std::stoll(s); }

// (a1+a2) - (b1+b2) in seconds without collapsing the two-part JDs first:
// the big parts cancel exactly (they differ by a small integer number of
// days), so the comparison keeps sub-nanosecond resolution that a single
// double JD (~4e-5 s quantum) could not represent.
double twopart_delta_s(double a1, double a2, double b1, double b2) {
  return ((a1 - b1) + (a2 - b2)) * 86400.0;
}

star::time::UtcTime utc_of(const star_tests::GoldenCase& c) {
  star::time::UtcTime utc{};
  utc.year = static_cast<std::int32_t>(parse_int(c.scalar("year")));
  utc.month = static_cast<std::int32_t>(parse_int(c.scalar("month")));
  utc.day = static_cast<std::int32_t>(parse_int(c.scalar("day")));
  utc.hour = static_cast<std::int32_t>(parse_int(c.scalar("hour")));
  utc.minute = static_cast<std::int32_t>(parse_int(c.scalar("minute")));
  utc.second = star_tests::parse_hex_double(c.scalar("second"));
  return utc;
}

}  // namespace

TEST_CASE("time_utc_tai_tt_golden") {
  const auto cases =
      star_tests::load_golden_cases(golden_path("utc_tai_tt.toml"));
  REQUIRE(cases.size() == 18);
  for (const auto& c : cases) {
    CAPTURE(c.scalar("name"));
    const star::time::UtcTime utc = utc_of(c);

    CHECK(star::time::tai_minus_utc_s(utc.year, utc.month, utc.day) ==
          static_cast<int>(parse_int(c.scalar("dat"))));

    // The reference implementation performs the identical IEEE-754
    // operation sequence, so the epoch and the two-part JDs must match
    // bit for bit (manifest tolerance: binary64 bit equality).
    const star::time::TaiEpoch tai = star::time::tai_from_utc(utc);
    CHECK(tai.day == parse_int(c.scalar("tai_day")));
    CHECK(tai.sec == star_tests::parse_hex_double(c.scalar("tai_sec")));

    const star::time::TwoPartJd jt = star::time::tai_jd(tai);
    CHECK(jt.jd1 == star_tests::parse_hex_double(c.scalar("tai_jd1")));
    CHECK(jt.jd2 == star_tests::parse_hex_double(c.scalar("tai_jd2")));
    const star::time::TwoPartJd tt = star::time::tt_jd(tai);
    CHECK(tt.jd1 == star_tests::parse_hex_double(c.scalar("tt_jd1")));
    CHECK(tt.jd2 == star_tests::parse_hex_double(c.scalar("tt_jd2")));

    // Phase 2 exit criterion 1 (time part): agreement with ERFA
    // (dtf2d/utctai/taitt) to 1e-9 s, compared two-part.
    CHECK(std::fabs(twopart_delta_s(
              jt.jd1, jt.jd2,
              star_tests::parse_hex_double(c.scalar("erfa_tai_jd1")),
              star_tests::parse_hex_double(c.scalar("erfa_tai_jd2")))) <=
          1e-9);
    CHECK(std::fabs(twopart_delta_s(
              tt.jd1, tt.jd2,
              star_tests::parse_hex_double(c.scalar("erfa_tt_jd1")),
              star_tests::parse_hex_double(c.scalar("erfa_tt_jd2")))) <=
          1e-9);

    // Inverse mapping: every golden input has a dyadic fractional second,
    // so the round trip must reproduce the calendar fields bit-exactly,
    // including second 60.x inside the inserted leap seconds.
    const star::time::UtcTime back = star::time::utc_from_tai(tai);
    CHECK(back.year == utc.year);
    CHECK(back.month == utc.month);
    CHECK(back.day == utc.day);
    CHECK(back.hour == utc.hour);
    CHECK(back.minute == utc.minute);
    CHECK(back.second == utc.second);
  }
}

TEST_CASE("time_tdb_series_golden") {
  const auto cases = star_tests::load_golden_cases(golden_path("tdb.toml"));
  REQUIRE(cases.size() == 18);
  for (const auto& c : cases) {
    CAPTURE(c.scalar("name"));
    const star::time::TaiEpoch tai{
        parse_int(c.scalar("tai_day")),
        star_tests::parse_hex_double(c.scalar("tai_sec"))};

    // Identical operation sequence to the golden reference: bit equality.
    CHECK(star::time::tt_julian_centuries(tai) ==
          star_tests::parse_hex_double(c.scalar("tt_centuries")));

    const double got = star::time::tdb_minus_tt_s(tai);
    // Tolerance from tests/golden/time/manifest.toml: abs 1e-13 s covers
    // libm sin() ulp spread across C runtimes while failing on any term or
    // coefficient deviation.
    CHECK(std::fabs(got - star_tests::parse_hex_double(
                              c.scalar("tdb_minus_tt"))) <= 1e-13);
    // D-6 truncation budget vs the full Fairhead-Bretagnon model (ERFA
    // eraDtdb): 30 us.
    CHECK(std::fabs(got - star_tests::parse_hex_double(
                              c.scalar("erfa_dtdb"))) <= 30e-6);

    // tdb_jd is TT plus the series folded into the fraction; collapsing
    // the parts here only needs ~1e-10 s resolution, well inside double.
    const star::time::TwoPartJd tt = star::time::tt_jd(tai);
    const star::time::TwoPartJd tdb = star::time::tdb_jd(tai);
    CHECK(tdb.jd2 >= 0.0);
    CHECK(tdb.jd2 < 1.0);
    CHECK(std::fabs(twopart_delta_s(tdb.jd1, tdb.jd2, tt.jd1, tt.jd2) - got) <=
          1e-9);
  }
}

TEST_CASE("time_leap_table_golden") {
  const auto cases =
      star_tests::load_golden_cases(golden_path("leap_history.toml"));
  const star::time::LeapTableInfo info = star::time::leap_table_info();
  REQUIRE(cases.size() == static_cast<std::size_t>(info.entries));
  for (const auto& c : cases) {
    CAPTURE(c.scalar("name"));
    const auto y = static_cast<std::int32_t>(parse_int(c.scalar("year")));
    const auto m = static_cast<std::int32_t>(parse_int(c.scalar("month")));
    const int dat = static_cast<int>(parse_int(c.scalar("dat")));
    // The step is effective from year-month-01 and holds mid-span too.
    CHECK(star::time::tai_minus_utc_s(y, m, 1) == dat);
    CHECK(star::time::tai_minus_utc_s(y, m, 15) == dat);
    const std::string prev_dat = c.scalar("prev_dat");
    const auto py =
        static_cast<std::int32_t>(parse_int(c.scalar("prev_year")));
    const auto pm =
        static_cast<std::int32_t>(parse_int(c.scalar("prev_month")));
    const auto pd = static_cast<std::int32_t>(parse_int(c.scalar("prev_day")));
    if (prev_dat == "out_of_domain") {
      // 1971-12-31 precedes the constant-offset UTC era: hard domain error,
      // never a silently wrong extrapolation.
      CHECK_THROWS_AS(star::time::tai_minus_utc_s(py, pm, pd),
                      std::domain_error);
    } else {
      CHECK(star::time::tai_minus_utc_s(py, pm, pd) ==
            static_cast<int>(parse_int(prev_dat)));
    }
  }
  // Post-expiry dates keep the last tabulated offset (documented
  // out-of-domain behavior; the Python layer owns the warning).
  CHECK(star::time::tai_minus_utc_s(2060, 12, 31) == 37);
  CHECK(info.expiry_year == 2026);
  CHECK(info.expiry_month == 7);
  CHECK(info.expiry_day == 1);
  CHECK(std::string(info.version).find("Bulletin C") != std::string::npos);
}

TEST_CASE("time_leap_second_roundtrip") {
  const auto cases =
      star_tests::load_golden_cases(golden_path("leap_history.toml"));
  for (const auto& c : cases) {
    if (c.scalar("prev_dat") == "out_of_domain") {
      continue;  // no in-domain day precedes the first entry
    }
    CAPTURE(c.scalar("name"));
    const auto py =
        static_cast<std::int32_t>(parse_int(c.scalar("prev_year")));
    const auto pm =
        static_cast<std::int32_t>(parse_int(c.scalar("prev_month")));
    const auto pd = static_cast<std::int32_t>(parse_int(c.scalar("prev_day")));
    const auto y = static_cast<std::int32_t>(parse_int(c.scalar("year")));
    const auto m = static_cast<std::int32_t>(parse_int(c.scalar("month")));

    // Every table step since 1972 is an insertion, so the day before each
    // boundary ends with second 60. Round-trip instants before, inside,
    // and after the inserted second; dyadic fractions make the trips
    // bit-exact.
    const double seconds[] = {59.0, 59.5, 60.0, 60.5, 60.984375};
    for (const double s : seconds) {
      const star::time::UtcTime utc{py, pm, pd, 23, 59, s};
      const star::time::TaiEpoch tai = star::time::tai_from_utc(utc);
      const star::time::UtcTime back = star::time::utc_from_tai(tai);
      CAPTURE(s);
      CHECK(back.year == py);
      CHECK(back.month == pm);
      CHECK(back.day == pd);
      CHECK(back.hour == 23);
      CHECK(back.minute == 59);
      CHECK(back.second == s);
    }
    const star::time::UtcTime after{y, m, 1, 0, 0, 0.0};
    const star::time::UtcTime backa =
        star::time::utc_from_tai(star::time::tai_from_utc(after));
    CHECK(backa.year == y);
    CHECK(backa.month == m);
    CHECK(backa.day == 1);
    CHECK(backa.second == 0.0);

    // Elapsed TAI across the boundary: 23:59:59 -> 00:00:00 spans the
    // inserted second, so exactly 2 SI seconds elapse; 23:59:60 -> 00:00:00
    // is exactly 1 (all whole-second cases, so the differences are exact).
    const star::time::TaiEpoch t59 =
        star::time::tai_from_utc({py, pm, pd, 23, 59, 59.0});
    const star::time::TaiEpoch t60 =
        star::time::tai_from_utc({py, pm, pd, 23, 59, 60.0});
    const star::time::TaiEpoch t00 = star::time::tai_from_utc(after);
    CHECK(static_cast<double>(t00.day - t59.day) * 86400.0 +
              (t00.sec - t59.sec) ==
          2.0);
    CHECK(static_cast<double>(t00.day - t60.day) * 86400.0 +
              (t00.sec - t60.sec) ==
          1.0);
  }
}

TEST_CASE("time_two_part_precision") {
  // The design point of D-6: at 2060 an epoch is ~1.9e9 s past J2000, where
  // a single double of seconds has a quantum of ~2.4e-7 s and cannot
  // resolve a 1 ns step - the two-part epoch must.
  const star::time::TaiEpoch e =
      star::time::tai_from_utc({2060, 1, 1, 0, 0, 0.0});
  const double single_double_s =
      static_cast<double>(e.day) * 86400.0 + (e.sec - star::time::J2000_TAI_SOD_S);
  CHECK(single_double_s > 1.8e9);
  CHECK(single_double_s + 1e-9 == single_double_s);  // single double: lost

  const star::time::TaiEpoch e2 = star::time::tai_add_seconds(e, 1e-9);
  const double delta =
      static_cast<double>(e2.day - e.day) * 86400.0 + (e2.sec - e.sec);
  // Two-part: the step survives to the seconds-of-day quantum
  // (ulp(86400 s)/2 ~ 7.3e-12 s; here sec ~ 37 s so the error is ~4e-15 s).
  CHECK(std::fabs(delta - 1e-9) <= 1e-14);

  // Day-boundary normalization: the invariant 0 <= sec < 86400 holds after
  // arithmetic in both directions, and dyadic offsets round-trip exactly.
  const star::time::TaiEpoch fwd = star::time::tai_add_seconds(e, 2.5 * 86400.0);
  CHECK(fwd.day == e.day + 2);
  CHECK(fwd.sec == e.sec + 43200.0);
  const star::time::TaiEpoch bwd =
      star::time::tai_add_seconds(fwd, -2.5 * 86400.0);
  CHECK(bwd.day == e.day);
  CHECK(bwd.sec == e.sec);
  const star::time::TaiEpoch cross =
      star::time::tai_add_seconds(e, -(e.sec + 0.25));
  CHECK(cross.day == e.day - 1);
  CHECK(cross.sec == 86399.75);

  // Representation quantum across the 2020-2060 span: consecutive
  // representable instants near end-of-day are ~1.5e-11 s apart, two
  // orders below the 1e-9 s acceptance and three below the ~1e-8 s UT1
  // requirement that motivated D-6.
  const double quantum = 86400.0 - std::nexttoward(86400.0, 0.0L);
  CHECK(quantum > 0.0);
  CHECK(quantum < 2e-11);
}

TEST_CASE("time_domain_errors") {
  using star::time::tai_from_utc;
  using star::time::UtcTime;
  // Pre-1972: the rubber-second era is out of the table domain.
  CHECK_THROWS_AS(tai_from_utc({1971, 12, 31, 0, 0, 0.0}), std::domain_error);
  // Field ranges.
  CHECK_THROWS_AS(tai_from_utc({2026, 0, 1, 0, 0, 0.0}), std::domain_error);
  CHECK_THROWS_AS(tai_from_utc({2026, 13, 1, 0, 0, 0.0}), std::domain_error);
  CHECK_THROWS_AS(tai_from_utc({2026, 2, 29, 0, 0, 0.0}), std::domain_error);
  CHECK_THROWS_AS(tai_from_utc({2026, 1, 1, 24, 0, 0.0}), std::domain_error);
  CHECK_THROWS_AS(tai_from_utc({2026, 1, 1, 0, 60, 0.0}), std::domain_error);
  CHECK_THROWS_AS(tai_from_utc({2026, 1, 1, 0, 0, 61.0}), std::domain_error);
  const double nan = std::nan("");
  CHECK_THROWS_AS(tai_from_utc({2026, 1, 1, 0, 0, nan}), std::domain_error);
  // Second 60 exists only inside an inserted leap second: not at the end
  // of an ordinary day, and not outside the final minute of a leap day.
  CHECK_THROWS_AS(tai_from_utc({2020, 6, 30, 23, 59, 60.0}),
                  std::domain_error);
  CHECK_THROWS_AS(tai_from_utc({2016, 12, 31, 22, 59, 60.0}),
                  std::domain_error);
  // Leap year handled: 2024-02-29 is valid, and the century rule holds
  // (2100 is not a leap year under the Gregorian rule).
  CHECK(tai_from_utc({2024, 2, 29, 0, 0, 0.0}).sec == 37.0);
  CHECK_THROWS_AS(tai_from_utc({2100, 2, 29, 0, 0, 0.0}), std::domain_error);
}
