"""Regenerate the time-system golden-vector files in this directory.

The values are produced by an independent pure-Python reference
implementation of the conversions the C++ core implements (FR-2, D-6),
cross-checked case by case against ERFA (pyerfa), the reference
implementation of the IAU SOFA algorithms:

- Calendar day counts: Fliegel and Van Flandern, "A Machine Algorithm for
  Processing Calendar Dates", Communications of the ACM 11(10), 1968
  (integer-arithmetic Julian Day Number and its inverse). Cross-checked
  against datetime.date.toordinal over every day 1972-01-01..2068-12-31.
- Leap seconds: the published TAI-UTC step history since 1972 (IERS Earth
  Orientation Centre, Bulletin C series; also tabulated in the IERS
  EOP tai-utc data and in ERFA eraDat). Every entry is cross-checked
  against erfa.dat at its effective date and the day before.
- TT = TAI + 32.184 s exactly (IAU 1991 Resolution A4; Kaplan, USNO
  Circular 179, 2005).
- TDB - TT: the truncated periodic series of Kaplan, USNO Circular 179
  (2005), eq. 2.6 (the leading terms of Fairhead and Bretagnon, A&A 229,
  240, 1990), evaluated operation-for-operation as cpp/src/time.cpp
  evaluates it, and cross-checked against the full erfa.dtdb model.
- Published anchor: the worked example in the IAU SOFA cookbook "SOFA
  Tools for Earth Attitude" (UTC 2007-04-05 12:00:00.0 gives
  TAI 12:00:33.000 and TT 12:01:05.184) is asserted at generation time.

Running this script rewrites the three .toml golden files
byte-identically; any diff after regeneration means either the script or
the goldens were edited by hand, which the FR-22 golden-update discipline
forbids.
"""

from __future__ import annotations

import datetime
import math
import pathlib
import warnings

import erfa

HERE = pathlib.Path(__file__).resolve().parent

# Seconds from TAI midnight 2000-01-01 to the J2000 epoch instant
# (2000-01-01T12:00:00.0 TT = 2000-01-01T11:59:27.816 TAI). Mirrors
# star::time (cpp/src/time.cpp, eq:time:j2000).
J2000_TAI_SOD = 43167.816

# TT - TAI in seconds, exact by definition (IAU 1991 Resolution A4).
TT_MINUS_TAI = 32.184

# TAI - UTC steps since 1972 (year, month of effectivity at
# year-month-01T00:00:00 UTC, TAI-UTC seconds). Published leap-second
# history, IERS Bulletin C; cross-checked against erfa.dat below.
LEAP_TABLE = [
    (1972, 1, 10), (1972, 7, 11), (1973, 1, 12), (1974, 1, 13),
    (1975, 1, 14), (1976, 1, 15), (1977, 1, 16), (1978, 1, 17),
    (1979, 1, 18), (1980, 1, 19), (1981, 7, 20), (1982, 7, 21),
    (1983, 7, 22), (1985, 7, 23), (1988, 1, 24), (1990, 1, 25),
    (1991, 1, 26), (1992, 7, 27), (1993, 7, 28), (1994, 7, 29),
    (1996, 1, 30), (1997, 7, 31), (1999, 1, 32), (2006, 1, 33),
    (2009, 1, 34), (2012, 7, 35), (2015, 7, 36), (2017, 1, 37),
]

# --------------------------------------------------------------------------
# Reference implementations (mirror cpp/src/time.cpp operation for operation)
# --------------------------------------------------------------------------


def cdiv(a: int, b: int) -> int:
    """C++ integer division (truncation toward zero), for the F&VF formulas."""
    q, r = divmod(a, b)
    if r != 0 and (a < 0) != (b < 0):
        q += 1
    return q


def days_from_civil(year: int, month: int, day: int) -> int:
    """Gregorian date -> days since 2000-01-01 (Fliegel & Van Flandern 1968)."""
    jdn = (
        day
        - 32075
        + cdiv(1461 * (year + 4800 + cdiv(month - 14, 12)), 4)
        + cdiv(367 * (month - 2 - cdiv(month - 14, 12) * 12), 12)
        - cdiv(3 * cdiv(year + 4900 + cdiv(month - 14, 12), 100), 4)
    )
    return jdn - 2451545  # JDN of 2000-01-01


def civil_from_days(days: int) -> tuple[int, int, int]:
    """Inverse of days_from_civil (Fliegel & Van Flandern 1968)."""
    l = days + 2451545 + 68569
    n = cdiv(4 * l, 146097)
    l = l - cdiv(146097 * n + 3, 4)
    i = cdiv(4000 * (l + 1), 1461001)
    l = l - cdiv(1461 * i, 4) + 31
    j = cdiv(80 * l, 2447)
    day = l - cdiv(2447 * j, 80)
    l = cdiv(j, 11)
    month = j + 2 - 12 * l
    year = 100 * (n - 49) + i + l
    return year, month, day


def tai_minus_utc(year: int, month: int, day: int) -> int:
    """Table lookup of TAI-UTC for a UTC calendar date (1972 onward)."""
    if (year, month, day) < (1972, 1, 1):
        raise ValueError("leap-second table starts 1972-01-01")
    dat = LEAP_TABLE[0][2]
    for ey, em, ed in LEAP_TABLE:
        if (year, month, day) >= (ey, em, 1):
            dat = ed
    return dat


def tai_epoch_from_utc(
    year: int, month: int, day: int, hour: int, minute: int, second: float
) -> tuple[int, float]:
    """UTC calendar fields -> two-part TAI epoch (day, sec).

    day: whole TAI days since 2000-01-01T00:00:00.0 TAI; sec: TAI seconds
    of that day in [0, 86400). Mirrors star::time::tai_from_utc exactly:
    all whole-second bookkeeping is integer arithmetic; the fractional
    second rides through untouched, so the single rounding step is the
    final float(isod) + frac addition.
    """
    civil = days_from_civil(year, month, day)
    dat = tai_minus_utc(year, month, day)
    si = int(math.floor(second))
    frac = second - si  # exact: the fractional bits of one double
    isod_tai = hour * 3600 + minute * 60 + si + dat
    epoch_day = civil + isod_tai // 86400
    isod = isod_tai % 86400
    return epoch_day, float(isod) + frac


def tai_jd(day: int, sec: float) -> tuple[float, float]:
    """Two-part TAI Julian Date: jd1 half-integer day, jd2 fraction of day."""
    return 2451544.5 + day, sec / 86400.0


def tt_jd(day: int, sec: float) -> tuple[float, float]:
    """Two-part TT Julian Date (TT = TAI + 32.184 s, day carry explicit)."""
    total = sec + TT_MINUS_TAI
    if total >= 86400.0:
        return 2451544.5 + (day + 1), (total - 86400.0) / 86400.0
    return 2451544.5 + day, total / 86400.0


def tt_julian_centuries(day: int, sec: float) -> float:
    """TT Julian centuries since J2000 (elapsed TT == elapsed TAI)."""
    elapsed_s = float(day) * 86400.0 + (sec - J2000_TAI_SOD)
    return elapsed_s / (86400.0 * 36525.0)


def tdb_minus_tt(day: int, sec: float) -> float:
    """TDB - TT [s]: Kaplan (2005), USNO Circular 179, eq. 2.6.

    The seven leading terms of Fairhead & Bretagnon (1990); the argument
    is TT centuries since J2000 (the TDB/TT distinction in the argument is
    below 1e-12 s of effect). Term order mirrors cpp/src/time.cpp.
    """
    t = tt_julian_centuries(day, sec)
    return (
        0.001657 * math.sin(628.3076 * t + 6.2401)
        + 0.000022 * math.sin(575.3385 * t + 4.2970)
        + 0.000014 * math.sin(1256.6152 * t + 6.1969)
        + 0.000005 * math.sin(606.9777 * t + 4.0212)
        + 0.000005 * math.sin(52.9691 * t + 0.4444)
        + 0.000002 * math.sin(21.3299 * t + 5.5431)
        + 0.000010 * t * math.sin(628.3076 * t + 4.2490)
    )


# --------------------------------------------------------------------------
# Golden epochs
# --------------------------------------------------------------------------

# name, (y, mo, d, h, mi, s). Fractional seconds are dyadic (exactly
# representable in binary64) so the committed inputs are exact and the
# UTC -> TAI -> UTC round trip is bit-exact by construction. Coverage:
# the published SOFA cookbook anchor; the 2015 and 2016 leap seconds
# (before, during, mid, and after the inserted second); >= 10 epochs
# spanning 2020-2060 including the leap-table expiry boundary
# (2027-01-01, first instant not covered by IERS Bulletin C 71).
EPOCHS = [
    ("sofa_cookbook_2007_04_05", (2007, 4, 5, 12, 0, 0.0)),
    ("leap_2015_in_leap", (2015, 6, 30, 23, 59, 60.5)),
    ("leap_2016_last_normal_second", (2016, 12, 31, 23, 59, 59.0)),
    ("leap_2016_leap_start", (2016, 12, 31, 23, 59, 60.0)),
    ("leap_2016_mid_leap", (2016, 12, 31, 23, 59, 60.75)),
    ("leap_2017_first_second_after", (2017, 1, 1, 0, 0, 0.0)),
    ("epoch_2020_01_01", (2020, 1, 1, 0, 0, 0.0)),
    ("epoch_2023_06_15", (2023, 6, 15, 12, 34, 56.25)),
    ("expiry_2026_12_31_last_covered", (2026, 12, 31, 23, 59, 59.0)),
    ("expiry_2027_01_01_first_uncovered", (2027, 1, 1, 0, 0, 0.0)),
    ("epoch_2030_01_01", (2030, 1, 1, 0, 0, 0.0)),
    ("epoch_2035_03_20", (2035, 3, 20, 6, 30, 0.5)),
    ("epoch_2040_07_04", (2040, 7, 4, 18, 0, 0.0)),
    ("epoch_2045_12_25", (2045, 12, 25, 0, 0, 1.5)),
    ("epoch_2050_06_01", (2050, 6, 1, 2, 3, 4.125)),
    ("epoch_2055_11_11", (2055, 11, 11, 11, 11, 11.0)),
    ("epoch_2060_01_01", (2060, 1, 1, 0, 0, 0.0)),
    ("epoch_2060_12_31", (2060, 12, 31, 23, 59, 59.0)),
]


# --------------------------------------------------------------------------
# Cross-checks against independent references
# --------------------------------------------------------------------------


def _erfa_chain(y, mo, d, h, mi, s):
    """UTC calendar -> ERFA two-part TAI/TT JDs and full-model TDB-TT."""
    with warnings.catch_warnings():
        # Epochs past ERFA's built-in leap table draw "dubious year"
        # ErfaWarning; ERFA then assumes TAI-UTC stays 37 s, which is
        # exactly this table's post-expiry assumption, so the values
        # remain the correct cross-check.
        warnings.simplefilter("ignore", erfa.core.ErfaWarning)
        d1, d2 = erfa.dtf2d("UTC", y, mo, d, h, mi, s)
        tai1, tai2 = erfa.utctai(d1, d2)
        tt1, tt2 = erfa.taitt(tai1, tai2)
        dat = erfa.dat(y, mo, d, 0.0)
        dtdb = erfa.dtdb(tt1, tt2, 0.0, 0.0, 0.0, 0.0)
    return float(tai1), float(tai2), float(tt1), float(tt2), int(dat), float(dtdb)


def _twopart_delta_s(a1, a2, b1, b2):
    """(a1+a2) - (b1+b2) in seconds without collapsing the two parts."""
    return ((a1 - b1) + (a2 - b2)) * 86400.0


def crosscheck() -> dict:
    # Calendar algorithms against CPython's independent proleptic-Gregorian
    # implementation, every day of the mission-relevant span.
    ord2000 = datetime.date(2000, 1, 1).toordinal()
    d = datetime.date(1972, 1, 1)
    while d <= datetime.date(2068, 12, 31):
        days = d.toordinal() - ord2000
        assert days_from_civil(d.year, d.month, d.day) == days
        assert civil_from_days(days) == (d.year, d.month, d.day)
        d += datetime.timedelta(days=1)

    # Leap table against erfa.dat at every boundary and the day before.
    for k, (ey, em, dat) in enumerate(LEAP_TABLE):
        assert int(erfa.dat(ey, em, 1, 0.0)) == dat
        assert tai_minus_utc(ey, em, 1) == dat
        prev = datetime.date(ey, em, 1) - datetime.timedelta(days=1)
        if k > 0:
            assert int(erfa.dat(prev.year, prev.month, prev.day, 0.0)) == LEAP_TABLE[k - 1][2]
            assert tai_minus_utc(prev.year, prev.month, prev.day) == LEAP_TABLE[k - 1][2]

    # Published SOFA cookbook anchor ("SOFA Tools for Earth Attitude",
    # example date 2007 April 5): UTC 12:00:00.0 -> TAI 12:00:33.000,
    # TT 12:01:05.184.
    day, sec = tai_epoch_from_utc(2007, 4, 5, 12, 0, 0.0)
    assert sec == 43233.0  # 12h00m33s exactly
    assert abs((sec + TT_MINUS_TAI) - 43265.184) <= 1e-9  # 12h01m05.184s

    # Every golden epoch against the ERFA chain.
    max_tai = max_tt = max_dtdb = 0.0
    for _, (y, mo, dd, h, mi, s) in EPOCHS:
        eday, esec = tai_epoch_from_utc(y, mo, dd, h, mi, s)
        j1, j2 = tai_jd(eday, esec)
        t1, t2 = tt_jd(eday, esec)
        e_tai1, e_tai2, e_tt1, e_tt2, e_dat, e_dtdb = _erfa_chain(y, mo, dd, h, mi, s)
        assert tai_minus_utc(y, mo, dd) == e_dat
        dt_tai = abs(_twopart_delta_s(j1, j2, e_tai1, e_tai2))
        dt_tt = abs(_twopart_delta_s(t1, t2, e_tt1, e_tt2))
        d_dtdb = abs(tdb_minus_tt(eday, esec) - e_dtdb)
        assert dt_tai <= 1e-9, (y, mo, dd, dt_tai)
        assert dt_tt <= 1e-9, (y, mo, dd, dt_tt)
        # D-6 budget for the truncated TDB series is 30 us; Kaplan (2005)
        # quotes ~10 us for 1600-2200 against the full model.
        assert d_dtdb <= 30e-6, (y, mo, dd, d_dtdb)
        max_tai = max(max_tai, dt_tai)
        max_tt = max(max_tt, dt_tt)
        max_dtdb = max(max_dtdb, d_dtdb)
    return {"max_tai_s": max_tai, "max_tt_s": max_tt, "max_dtdb_s": max_dtdb}


# --------------------------------------------------------------------------
# TOML emission (restricted subset read by cpp/tests/golden_io.hpp)
# --------------------------------------------------------------------------


def emit(path: pathlib.Path, header: str, cases: list[dict]) -> None:
    lines = [f"# {line}" for line in header.strip().splitlines()]
    for case in cases:
        lines.append("")
        lines.append("[[case]]")
        for key, value in case.items():
            if isinstance(value, list):
                lines.append(f"{key} = [")
                lines.extend(f'  "{item}",' for item in value)
                lines.append("]")
            else:
                lines.append(f'{key} = "{value}"')
    path.write_text("\n".join(lines) + "\n", newline="\n", encoding="utf-8")


def main() -> None:
    maxima = crosscheck()

    utc_cases = []
    tdb_cases = []
    for name, (y, mo, d, h, mi, s) in EPOCHS:
        eday, esec = tai_epoch_from_utc(y, mo, d, h, mi, s)
        j1, j2 = tai_jd(eday, esec)
        t1, t2 = tt_jd(eday, esec)
        e_tai1, e_tai2, e_tt1, e_tt2, e_dat, e_dtdb = _erfa_chain(y, mo, d, h, mi, s)
        utc_cases.append(
            {
                "name": name,
                "year": str(y),
                "month": str(mo),
                "day": str(d),
                "hour": str(h),
                "minute": str(mi),
                "second": float(s).hex(),
                "second_dec": repr(float(s)),
                "dat": str(tai_minus_utc(y, mo, d)),
                "tai_day": str(eday),
                "tai_sec": esec.hex(),
                "tai_jd1": j1.hex(),
                "tai_jd2": j2.hex(),
                "tt_jd1": t1.hex(),
                "tt_jd2": t2.hex(),
                "erfa_tai_jd1": e_tai1.hex(),
                "erfa_tai_jd2": e_tai2.hex(),
                "erfa_tt_jd1": e_tt1.hex(),
                "erfa_tt_jd2": e_tt2.hex(),
            }
        )
        tdb_cases.append(
            {
                "name": name,
                "tai_day": str(eday),
                "tai_sec": esec.hex(),
                "tt_centuries": tt_julian_centuries(eday, esec).hex(),
                "tdb_minus_tt": tdb_minus_tt(eday, esec).hex(),
                "erfa_dtdb": e_dtdb.hex(),
            }
        )

    leap_cases = []
    for k, (ey, em, dat) in enumerate(LEAP_TABLE):
        prev = datetime.date(ey, em, 1) - datetime.timedelta(days=1)
        leap_cases.append(
            {
                "name": f"dat_{ey}_{em:02d}",
                "year": str(ey),
                "month": str(em),
                "dat": str(dat),
                "prev_year": str(prev.year),
                "prev_month": str(prev.month),
                "prev_day": str(prev.day),
                # The day before the first entry is outside the table domain;
                # the consuming test checks the domain error instead.
                "prev_dat": str(LEAP_TABLE[k - 1][2]) if k > 0 else "out_of_domain",
            }
        )

    emit(
        HERE / "utc_tai_tt.toml",
        "UTC -> TAI -> TT golden vectors (FR-2, D-6).\n"
        "Inputs are numeric UTC calendar fields with dyadic fractional\n"
        "seconds (exact binary64). tai_day/tai_sec is the two-part TAI\n"
        "epoch (days since 2000-01-01T00:00:00.0 TAI, seconds of day);\n"
        "*_jd1/_jd2 are two-part Julian Dates. Values without the erfa_\n"
        "prefix come from the generate.py reference implementation that\n"
        "mirrors cpp/src/time.cpp; erfa_* values come from ERFA\n"
        "(dtf2d/utctai/taitt). Provenance and tolerances in manifest.toml.\n"
        "Regenerated by generate.py.",
        utc_cases,
    )
    emit(
        HERE / "tdb.toml",
        "TDB - TT golden vectors (FR-2, D-6).\n"
        "tdb_minus_tt is the Kaplan (2005) USNO Circular 179 eq. 2.6\n"
        "seven-term series evaluated as cpp/src/time.cpp evaluates it;\n"
        "erfa_dtdb is the full Fairhead-Bretagnon model via erfa.dtdb\n"
        "(geocentric: ut=elong=u=v=0). Provenance and tolerances in\n"
        "manifest.toml. Regenerated by generate.py.",
        tdb_cases,
    )
    emit(
        HERE / "leap_history.toml",
        "TAI - UTC leap-second step history since 1972 (IERS Bulletin C).\n"
        "Each case: the step effective at year-month-01T00:00:00 UTC and\n"
        "the expected value on the previous calendar day. Cross-checked\n"
        "against erfa.dat at generation time. Regenerated by generate.py.",
        leap_cases,
    )

    print("golden files regenerated and cross-checked")
    print(f"pyerfa {erfa.__version__} (ERFA {erfa.version.erfa_version})")
    print(
        "observed maxima vs ERFA: "
        f"TAI {maxima['max_tai_s']:.3e} s, TT {maxima['max_tt_s']:.3e} s, "
        f"TDB series vs full model {maxima['max_dtdb_s']:.3e} s"
    )


if __name__ == "__main__":
    main()
