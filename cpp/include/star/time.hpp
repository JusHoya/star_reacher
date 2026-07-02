// Time systems for the star:: core (FR-2, decision D-6): two-part TAI epoch,
// UTC via a bundled versioned leap-second table, TT by exact offset, TDB by
// truncated periodic series. Derivations, domain bounds, and validation
// evidence are in the math library chapter ch:time
// (docs/mathlib/chapters/time.tex).
//
// Epoch convention (eq:time:j2000). The internal timescale is TAI. An epoch
// is stored in two parts - whole TAI days and TAI seconds of day - counted
// from 2000-01-01T00:00:00.0 TAI. The J2000 epoch is the instant
// 2000-01-01T12:00:00.0 TT = JD 2451545.0 TT, which on the TAI clock reads
// 2000-01-01T11:59:27.816 TAI (TT = TAI + 32.184 s exactly), i.e.
// TaiEpoch{0, 43167.816}. The split representation exists because a single
// binary64 of seconds since J2000 has a quantum of ~2.4e-7 s by 2060
// (ulp(1.9e9 s)), which cannot support the ~1e-8 s UT1 precision the
// Earth-rotation-angle computation needs; the two-part form keeps the
// sub-day quantum at ulp(86400 s) ~ 1.5e-11 s across 2020-2060.
//
// The core never parses text (D-2): calendar input is numeric fields only;
// ISO-8601 parsing lives in the Python frontend.
#ifndef STAR_TIME_HPP
#define STAR_TIME_HPP

#include <cstdint>

namespace star {
namespace time {

// TT - TAI in SI seconds, exact by definition: TT is realized as
// TT(TAI) = TAI + 32.184 s (IAU 1991 Resolution A4; stated in Kaplan,
// USNO Circular 179, 2005, and IERS Conventions 2010, TN No. 36).
inline constexpr double TT_MINUS_TAI_S = 32.184;

// TAI seconds of day of the J2000 instant on 2000-01-01 TAI
// (11:59:27.816 TAI = 43167.816 s; see eq:time:j2000).
inline constexpr double J2000_TAI_SOD_S = 43167.816;

// Two-part TAI epoch (D-6): `day` whole TAI days since
// 2000-01-01T00:00:00.0 TAI (negative before it), `sec` TAI seconds of that
// day with the invariant 0 <= sec < 86400. TAI has no leap seconds, so every
// TAI day is exactly 86400 SI seconds and the pair maps affinely to elapsed
// time.
struct TaiEpoch {
  std::int64_t day;
  double sec;
};

// Numeric UTC calendar fields (proleptic Gregorian). `second` covers
// [0, 60) normally and [60, 61) only during an inserted (positive) leap
// second, i.e. in the final minute of a day after which the leap table
// steps up.
struct UtcTime {
  std::int32_t year;
  std::int32_t month;
  std::int32_t day;
  std::int32_t hour;
  std::int32_t minute;
  double second;
};

// Two-part Julian Date: jd1 a half-integer day count, jd2 the fraction of
// day in [0, 1). Kept split for the same precision reason as TaiEpoch.
struct TwoPartJd {
  double jd1;
  double jd2;
};

// Bundled leap-second table metadata. `version` cites the IERS Bulletin C
// state the table was verified against. `expiry_*` is the first UTC calendar
// date at which a leap second not present in this table could take effect
// (the first insertion opportunity the cited bulletin does not rule out);
// the core exposes it so the Python layer can warn on post-expiry epochs -
// the core itself never reads the clock (D-2), so the warning decision
// cannot live here.
struct LeapTableInfo {
  const char* version;
  std::int32_t expiry_year;
  std::int32_t expiry_month;
  std::int32_t expiry_day;
  std::int32_t entries;
};

LeapTableInfo leap_table_info();

// TAI - UTC in whole seconds for a UTC calendar date (eq:time:utc2tai).
// Domain: 1972-01-01 onward (the constant-offset UTC era); earlier dates
// throw std::domain_error. Dates past the table expiry return the last
// tabulated value (37 s) - the documented post-expiry assumption; the
// Python layer warns.
int tai_minus_utc_s(std::int32_t year, std::int32_t month, std::int32_t day);

// UTC calendar fields -> two-part TAI epoch (eq:time:utc2tai). Whole-second
// bookkeeping is exact integer arithmetic; the fractional second passes
// through to the one rounding step (see the chapter's precision analysis).
// Throws std::domain_error for dates before 1972-01-01, out-of-range or
// non-finite fields, and second >= 60 outside an inserted leap second.
TaiEpoch tai_from_utc(const UtcTime& utc);

// Two-part TAI epoch -> UTC calendar fields, the exact inverse of
// tai_from_utc over the table span; instants inside an inserted leap second
// come back with second in [60, 61). Throws std::domain_error for epochs
// before 1972-01-01 UTC.
UtcTime utc_from_tai(const TaiEpoch& tai);

// Two-part Julian Dates on the TAI, TT, and TDB scales for the given epoch.
// TT applies eq:time:tt; TDB adds the eq:time:tdb series to TT.
TwoPartJd tai_jd(const TaiEpoch& tai);
TwoPartJd tt_jd(const TaiEpoch& tai);
TwoPartJd tdb_jd(const TaiEpoch& tai);

// TT Julian centuries since J2000 (eq:time:ttcent) - the argument of the
// TDB series and of the Phase 2 frame models. Collapsing to one double is
// acceptable here: the consumers' sensitivity to its ~1e-16 relative
// rounding is far below their own model error (chapter, implementation
// notes).
double tt_julian_centuries(const TaiEpoch& tai);

// TDB - TT in seconds (eq:time:tdb): the seven-term truncation of the
// Fairhead & Bretagnon (1990) series as given by Kaplan, USNO Circular 179
// (2005), eq. 2.6; |error| vs the full model is within the 30 us D-6 budget
// (~10 us over 1600-2200 per Kaplan; 5.6e-6 s observed over the golden
// epochs).
double tdb_minus_tt_s(const TaiEpoch& tai);

// Epoch arithmetic: tai + delta_s with the day/sec invariant restored.
// Exact to ulp(max(|delta_s|, 86400)); intended for within-mission offsets
// where delta_s stays far below the ~4.5e15 s at which whole seconds would
// alias (chapter, implementation notes).
TaiEpoch tai_add_seconds(const TaiEpoch& tai, double delta_s);

}  // namespace time
}  // namespace star

#endif  // STAR_TIME_HPP
