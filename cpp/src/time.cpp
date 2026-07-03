// Time-system conversions (FR-2, D-6). Derivations and the precision
// analysis live in the math library chapter ch:time; the equation labels
// echoed in comments below (eq:time:...) trace each code block to its
// defining equation (FR-29 traceability). The whole file is straight-line
// integer and IEEE-754 double arithmetic - no platform-dependent calls -
// so the same inputs produce bit-identical outputs on every conforming
// platform (D-10).
#include "star/time.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace star {
namespace time {

namespace {

// One TAI-UTC step: TAI - UTC equals `dat` seconds from
// year-month-01T00:00:00 UTC until the next entry takes effect.
struct LeapEntry {
  std::int32_t year;
  std::int32_t month;
  std::int32_t dat;
};

// TAI - UTC step history since 1972 (the constant-offset UTC era; the
// pre-1972 "rubber second" era is out of domain). Source: IERS Earth
// Orientation Centre, Bulletin C series (leap-second announcements),
// verified through Bulletin C 71 (January 2026), which announced no leap
// second at the end of June 2026 and TAI-UTC = 37 s until further
// notice. Golden cross-check against ERFA eraDat:
// tests/golden/time/leap_history.toml.
constexpr LeapEntry kLeapTable[] = {
    {1972, 1, 10}, {1972, 7, 11}, {1973, 1, 12}, {1974, 1, 13},
    {1975, 1, 14}, {1976, 1, 15}, {1977, 1, 16}, {1978, 1, 17},
    {1979, 1, 18}, {1980, 1, 19}, {1981, 7, 20}, {1982, 7, 21},
    {1983, 7, 22}, {1985, 7, 23}, {1988, 1, 24}, {1990, 1, 25},
    {1991, 1, 26}, {1992, 7, 27}, {1993, 7, 28}, {1994, 7, 29},
    {1996, 1, 30}, {1997, 7, 31}, {1999, 1, 32}, {2006, 1, 33},
    {2009, 1, 34}, {2012, 7, 35}, {2015, 7, 36}, {2017, 1, 37},
};
constexpr std::int32_t kLeapEntries =
    static_cast<std::int32_t>(sizeof(kLeapTable) / sizeof(kLeapTable[0]));

// First UTC calendar date at which a leap second not present in the table
// could take effect: Bulletin C 71 rules out the end-of-June-2026
// opportunity, so the next unruled insertion point is the end of
// December 2026.
constexpr std::int32_t kExpiryYear = 2027;
constexpr std::int32_t kExpiryMonth = 1;
constexpr std::int32_t kExpiryDay = 1;

constexpr const char* kLeapTableVersion =
    "IERS Bulletin C history, verified through Bulletin C 71 (January 2026): "
    "TAI-UTC = 37 s, no leap second at the end of June 2026";

// Gregorian date -> whole days since 2000-01-01, via the integer Julian Day
// Number algorithm of Fliegel & Van Flandern (Commun. ACM 11(10), 1968)
// (eq:time:fvf). C++ integer division truncates toward zero, matching the
// FORTRAN semantics the formula was written for. JDN(2000-01-01) = 2451545.
std::int64_t days_from_civil(std::int64_t y, std::int64_t m, std::int64_t d) {
  const std::int64_t a = (m - 14) / 12;
  const std::int64_t jdn = d - 32075 + 1461 * (y + 4800 + a) / 4 +
                           367 * (m - 2 - a * 12) / 12 -
                           3 * ((y + 4900 + a) / 100) / 4;
  return jdn - 2451545;
}

// Inverse of days_from_civil (Fliegel & Van Flandern 1968) (eq:time:fvf).
void civil_from_days(std::int64_t days, std::int32_t* year,
                     std::int32_t* month, std::int32_t* day) {
  std::int64_t l = days + 2451545 + 68569;
  const std::int64_t n = 4 * l / 146097;
  l = l - (146097 * n + 3) / 4;
  const std::int64_t i = 4000 * (l + 1) / 1461001;
  l = l - 1461 * i / 4 + 31;
  const std::int64_t j = 80 * l / 2447;
  *day = static_cast<std::int32_t>(l - 2447 * j / 80);
  const std::int64_t k = j / 11;
  *month = static_cast<std::int32_t>(j + 2 - 12 * k);
  *year = static_cast<std::int32_t>(100 * (n - 49) + i + k);
}

bool is_gregorian_leap_year(std::int32_t y) {
  return y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
}

std::int32_t days_in_month(std::int32_t y, std::int32_t m) {
  constexpr std::int32_t kDays[12] = {31, 28, 31, 30, 31, 30,
                                      31, 31, 30, 31, 30, 31};
  if (m == 2 && is_gregorian_leap_year(y)) {
    return 29;
  }
  return kDays[m - 1];
}

// True when the UTC day (year, month, day) ends with an inserted (positive)
// leap second, i.e. the table steps up at the start of the next civil day.
// The table holds only positive steps (every leap second since 1972 has
// been an insertion), so a step of +1 is the only case.
bool day_ends_with_leap_second(std::int32_t year, std::int32_t month,
                               std::int32_t day) {
  std::int32_t ny = 0;
  std::int32_t nm = 0;
  std::int32_t nd = 0;
  civil_from_days(days_from_civil(year, month, day) + 1, &ny, &nm, &nd);
  return tai_minus_utc_s(ny, nm, nd) == tai_minus_utc_s(year, month, day) + 1;
}

[[noreturn]] void domain_fail(const std::string& what) {
  throw std::domain_error("star::time: " + what);
}

}  // namespace

LeapTableInfo leap_table_info() {
  return {kLeapTableVersion, kExpiryYear, kExpiryMonth, kExpiryDay,
          kLeapEntries};
}

int tai_minus_utc_s(std::int32_t year, std::int32_t month, std::int32_t day) {
  // Encode the calendar date as y*10000 + m*100 + d for ordering; entries
  // take effect on the first of their month (eq:time:utc2tai).
  const std::int64_t key = static_cast<std::int64_t>(year) * 10000 +
                           month * 100 + day;
  if (key < 1972'01'01) {
    domain_fail("date " + std::to_string(year) + "-" + std::to_string(month) +
                "-" + std::to_string(day) +
                " precedes the leap-second table domain (1972-01-01)");
  }
  // Fixed forward scan: deterministic and branch-stable; 28 entries make
  // search structure irrelevant.
  std::int32_t dat = kLeapTable[0].dat;
  for (std::int32_t k = 0; k < kLeapEntries; ++k) {
    const std::int64_t effective = static_cast<std::int64_t>(kLeapTable[k].year) * 10000 +
                                   kLeapTable[k].month * 100 + 1;
    if (key >= effective) {
      dat = kLeapTable[k].dat;
    }
  }
  return dat;
}

TaiEpoch tai_from_utc(const UtcTime& utc) {
  if (utc.month < 1 || utc.month > 12) {
    domain_fail("month " + std::to_string(utc.month) + " outside [1, 12]");
  }
  if (utc.day < 1 || utc.day > days_in_month(utc.year, utc.month)) {
    domain_fail("day " + std::to_string(utc.day) + " invalid for " +
                std::to_string(utc.year) + "-" + std::to_string(utc.month));
  }
  if (utc.hour < 0 || utc.hour > 23) {
    domain_fail("hour " + std::to_string(utc.hour) + " outside [0, 23]");
  }
  if (utc.minute < 0 || utc.minute > 59) {
    domain_fail("minute " + std::to_string(utc.minute) + " outside [0, 59]");
  }
  if (!std::isfinite(utc.second) || utc.second < 0.0 || utc.second >= 61.0) {
    domain_fail("second must be finite and within [0, 61)");
  }
  if (utc.second >= 60.0 &&
      !(utc.hour == 23 && utc.minute == 59 &&
        day_ends_with_leap_second(utc.year, utc.month, utc.day))) {
    // Second 60 exists only inside an inserted leap second (the final
    // minute of a day the table steps after); anywhere else it is not a
    // real UTC instant, and silently normalizing it would shift the epoch
    // by one second (abort-on-invalid, never a wrong default).
    domain_fail("second >= 60 outside an inserted leap second");
  }

  // TAI = UTC + dAT (eq:time:utc2tai). All whole-second bookkeeping is
  // exact integer arithmetic; the fractional second rides through
  // untouched, so the one rounding step is the final double addition
  // (chapter precision analysis: error <= 2^-37 s ~ 7.3e-12 s).
  const std::int64_t civil = days_from_civil(utc.year, utc.month, utc.day);
  const int dat = tai_minus_utc_s(utc.year, utc.month, utc.day);
  const std::int64_t si = static_cast<std::int64_t>(std::floor(utc.second));
  const double frac = utc.second - static_cast<double>(si);  // exact split
  const std::int64_t isod_tai =
      static_cast<std::int64_t>(utc.hour) * 3600 + utc.minute * 60 + si + dat;
  // isod_tai >= 0 always (dat >= 10), so truncating division is floor here.
  const std::int64_t day = civil + isod_tai / 86400;
  const std::int64_t isod = isod_tai % 86400;
  return {day, static_cast<double>(isod) + frac};
}

UtcTime utc_from_tai(const TaiEpoch& tai) {
  // Exact split of the seconds-of-day into whole seconds and fraction; the
  // fractional bits of a double are exactly representable, so `frac` is the
  // exact remainder and every subsequent whole-second step is integer
  // arithmetic (eq:time:tai2utc).
  const std::int64_t isod = static_cast<std::int64_t>(std::floor(tai.sec));
  const double frac = tai.sec - static_cast<double>(isod);

  // A UTC civil day starting at date d begins at TAI seconds-of-2000
  // d*86400 + dAT(d); the instant belongs to the previous UTC day when its
  // TAI seconds-of-day is still below today's offset (eq:time:tai2utc).
  std::int32_t y = 0;
  std::int32_t m = 0;
  std::int32_t d = 0;
  civil_from_days(tai.day, &y, &m, &d);
  const int dat_today = tai_minus_utc_s(y, m, d);

  std::int64_t sod_utc = 0;
  UtcTime utc{};
  if (isod >= dat_today) {
    sod_utc = isod - dat_today;
    utc.year = y;
    utc.month = m;
    utc.day = d;
  } else {
    civil_from_days(tai.day - 1, &y, &m, &d);
    const int dat_prev = tai_minus_utc_s(y, m, d);
    // sod_utc lands in [86400, 86401) exactly when the previous day ends
    // with an inserted leap second (dat_today = dat_prev + 1): that is the
    // leap second itself, rendered below as second 60.
    sod_utc = isod + 86400 - dat_prev;
    utc.year = y;
    utc.month = m;
    utc.day = d;
  }

  std::int64_t s_int = 0;
  if (sod_utc >= 86400) {
    utc.hour = 23;
    utc.minute = 59;
    s_int = sod_utc - 86340;  // 23:59:60 within the inserted second
  } else {
    utc.hour = static_cast<std::int32_t>(sod_utc / 3600);
    utc.minute = static_cast<std::int32_t>((sod_utc % 3600) / 60);
    s_int = sod_utc % 60;
  }
  utc.second = static_cast<double>(s_int) + frac;
  return utc;
}

TwoPartJd tai_jd(const TaiEpoch& tai) {
  // JD(2000-01-01T00:00:00 TAI) = 2451544.5; the half-integer jd1 is exact
  // in binary64 for the whole table span, jd2 carries the sub-day part.
  return {2451544.5 + static_cast<double>(tai.day), tai.sec / 86400.0};
}

TwoPartJd tt_jd(const TaiEpoch& tai) {
  // TT = TAI + 32.184 s exactly (eq:time:tt); the day carry keeps jd2 in
  // [0, 1) so the two-part precision contract holds for downstream users.
  const double total = tai.sec + TT_MINUS_TAI_S;
  if (total >= 86400.0) {
    return {2451544.5 + static_cast<double>(tai.day + 1),
            (total - 86400.0) / 86400.0};
  }
  return {2451544.5 + static_cast<double>(tai.day), total / 86400.0};
}

double tt_julian_centuries(const TaiEpoch& tai) {
  // Elapsed TT equals elapsed TAI (constant offset), measured from the
  // J2000 instant TaiEpoch{0, 43167.816} (eq:time:j2000, eq:time:ttcent).
  // The single-double collapse is safe for this consumer: ~1e-16 relative
  // rounding moves the TDB series by ~1e-16 s and the Phase 2 frame
  // polynomials by amounts far below their model error.
  const double elapsed_s =
      static_cast<double>(tai.day) * 86400.0 + (tai.sec - J2000_TAI_SOD_S);
  return elapsed_s / (86400.0 * 36525.0);
}

double tdb_minus_tt_s(const TaiEpoch& tai) {
  // Seven-term truncation of the Fairhead & Bretagnon (1990) series as
  // given by Kaplan, USNO Circular 179 (2005), eq. 2.6 (eq:time:tdb).
  // Coefficients in seconds; arguments in radians with t in TT Julian
  // centuries since J2000 (the formal TDB/TT argument distinction is below
  // 1e-12 s of effect). Term order matches the golden reference
  // (tests/golden/time/generate.py) so the evaluation is bit-comparable.
  const double t = tt_julian_centuries(tai);
  return 0.001657 * std::sin(628.3076 * t + 6.2401) +
         0.000022 * std::sin(575.3385 * t + 4.2970) +
         0.000014 * std::sin(1256.6152 * t + 6.1969) +
         0.000005 * std::sin(606.9777 * t + 4.0212) +
         0.000005 * std::sin(52.9691 * t + 0.4444) +
         0.000002 * std::sin(21.3299 * t + 5.5431) +
         0.000010 * t * std::sin(628.3076 * t + 4.2490);
}

TwoPartJd tdb_jd(const TaiEpoch& tai) {
  // TDB = TT + (TDB - TT); the series stays below 2 ms so the fraction
  // adjustment can only cross a day boundary immediately at midnight.
  const TwoPartJd tt = tt_jd(tai);
  const double jd2 = tt.jd2 + tdb_minus_tt_s(tai) / 86400.0;
  if (jd2 >= 1.0) {
    return {tt.jd1 + 1.0, jd2 - 1.0};
  }
  if (jd2 < 0.0) {
    return {tt.jd1 - 1.0, jd2 + 1.0};
  }
  return {tt.jd1, jd2};
}

TaiEpoch tai_add_seconds(const TaiEpoch& tai, double delta_s) {
  if (!std::isfinite(delta_s)) {
    domain_fail("tai_add_seconds requires a finite offset");
  }
  double sec = tai.sec + delta_s;
  std::int64_t day = tai.day;
  // Whole-day reduction first (k*86400 is an exact integer product), then
  // at most one fix-up per side: floor(sec / 86400) can be off by one ulp
  // at the boundary, and the invariant 0 <= sec < 86400 must hold exactly.
  const double k = std::floor(sec / 86400.0);
  day += static_cast<std::int64_t>(k);
  sec -= k * 86400.0;
  if (sec < 0.0) {
    sec += 86400.0;
    --day;
  } else if (sec >= 86400.0) {
    sec -= 86400.0;
    ++day;
  }
  return {day, sec};
}

}  // namespace time
}  // namespace star
