"""Time-system binding tests (FR-2, D-6) against tests/golden/time/.

These drive the same conversions the doctest suite covers, but through the
pybind11 surface the Python frontend uses, against the identical golden
vectors (provenance and tolerances in tests/golden/time/manifest.toml).
Core-requiring tests fail cleanly, never skip, when the compiled core is
absent (see test_integration_core.py for the rationale).
"""

import math
import tomllib
import warnings
from pathlib import Path

import pytest

from star_reacher.mission import validate_mission_file

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = REPO_ROOT / "tests" / "golden" / "time"

_CORE_MISSING_MESSAGE = (
    "star_reacher._core is not built in this environment. These time-system "
    "tests require the compiled core: build and install it with 'pip install .' "
    "from the repository root. This failure is expected on a core-less checkout "
    "and must be green at integration/CI."
)


def _core_or_fail():
    try:
        from star_reacher import _core
    except ImportError:
        pytest.fail(_CORE_MISSING_MESSAGE)
    return _core


def _load_cases(name: str) -> list[dict]:
    with open(GOLDEN_DIR / name, "rb") as fh:
        return tomllib.load(fh)["case"]


def _twopart_delta_s(a1, a2, b1, b2):
    # Difference of two two-part JDs in seconds, big parts cancelled first
    # so the comparison keeps sub-nanosecond resolution.
    return ((a1 - b1) + (a2 - b2)) * 86400.0


def test_time_bindings_match_goldens():
    core = _core_or_fail()
    cases = _load_cases("utc_tai_tt.toml")
    assert len(cases) == 18
    for c in cases:
        utc_fields = (
            int(c["year"]),
            int(c["month"]),
            int(c["day"]),
            int(c["hour"]),
            int(c["minute"]),
            float.fromhex(c["second"]),
        )
        assert core.tai_minus_utc(*utc_fields[:3]) == int(c["dat"]), c["name"]

        day, sec = core.utc_to_tai(*utc_fields)
        # Bit equality: the binding forwards to the same C++ that mirrors
        # the golden reference operation for operation.
        assert day == int(c["tai_day"]), c["name"]
        assert sec == float.fromhex(c["tai_sec"]), c["name"]

        jd1, jd2 = core.tai_to_jd(day, sec)
        assert (jd1, jd2) == (
            float.fromhex(c["tai_jd1"]),
            float.fromhex(c["tai_jd2"]),
        ), c["name"]
        t1, t2 = core.tt_jd(day, sec)
        assert (t1, t2) == (
            float.fromhex(c["tt_jd1"]),
            float.fromhex(c["tt_jd2"]),
        ), c["name"]

        # Phase 2 exit criterion 1 (time part): <= 1e-9 s vs ERFA.
        assert (
            abs(
                _twopart_delta_s(
                    jd1,
                    jd2,
                    float.fromhex(c["erfa_tai_jd1"]),
                    float.fromhex(c["erfa_tai_jd2"]),
                )
            )
            <= 1e-9
        ), c["name"]
        assert (
            abs(
                _twopart_delta_s(
                    t1,
                    t2,
                    float.fromhex(c["erfa_tt_jd1"]),
                    float.fromhex(c["erfa_tt_jd2"]),
                )
            )
            <= 1e-9
        ), c["name"]

        # Round trip, bit-exact for the dyadic golden inputs (leap-second
        # instants come back with second in [60, 61)).
        assert core.tai_to_utc(day, sec) == utc_fields, c["name"]


def test_time_tdb_bindings_golden():
    core = _core_or_fail()
    cases = _load_cases("tdb.toml")
    assert len(cases) == 18
    for c in cases:
        day = int(c["tai_day"])
        sec = float.fromhex(c["tai_sec"])
        assert core.tt_julian_centuries(day, sec) == float.fromhex(
            c["tt_centuries"]
        ), c["name"]
        got = core.tdb_minus_tt(day, sec)
        # Tolerances per tests/golden/time/manifest.toml: 1e-13 s vs the
        # bit-comparable series reference, 30 us (D-6 truncation budget)
        # vs the full Fairhead-Bretagnon model (erfa.dtdb).
        assert abs(got - float.fromhex(c["tdb_minus_tt"])) <= 1e-13, c["name"]
        assert abs(got - float.fromhex(c["erfa_dtdb"])) <= 30e-6, c["name"]
        # TDB two-part JD is TT plus the series folded into the fraction.
        t1, t2 = core.tt_jd(day, sec)
        d1, d2 = core.tdb_jd(day, sec)
        assert 0.0 <= d2 < 1.0
        assert abs(_twopart_delta_s(d1, d2, t1, t2) - got) <= 1e-9, c["name"]


def test_time_leap_roundtrip_bindings():
    core = _core_or_fail()
    info = core.leap_table_info()
    assert info["entries"] == 28
    assert tuple(info["expiry_utc"]) == (2027, 1, 1)
    assert "Bulletin C" in info["version"]

    for c in _load_cases("leap_history.toml"):
        y, m, dat = int(c["year"]), int(c["month"]), int(c["dat"])
        assert core.tai_minus_utc(y, m, 1) == dat, c["name"]
        if c["prev_dat"] == "out_of_domain":
            # 1971-12-31 precedes the constant-offset UTC era; the core's
            # std::domain_error crosses the binding as ValueError.
            with pytest.raises(ValueError):
                core.tai_minus_utc(
                    int(c["prev_year"]), int(c["prev_month"]), int(c["prev_day"])
                )
            continue
        assert core.tai_minus_utc(
            int(c["prev_year"]), int(c["prev_month"]), int(c["prev_day"])
        ) == int(c["prev_dat"]), c["name"]
        # Round trip through the inserted leap second at this boundary.
        py, pm, pd = int(c["prev_year"]), int(c["prev_month"]), int(c["prev_day"])
        for s in (59.0, 60.0, 60.5):
            day, sec = core.utc_to_tai(py, pm, pd, 23, 59, s)
            assert core.tai_to_utc(day, sec) == (py, pm, pd, 23, 59, s)

    # Second 60 outside an inserted leap second is rejected, not normalized.
    with pytest.raises(ValueError):
        core.utc_to_tai(2020, 6, 30, 23, 59, 60.0)


def test_time_two_part_arithmetic_precision():
    core = _core_or_fail()
    day, sec = core.utc_to_tai(2060, 1, 1, 0, 0, 0.0)
    # A single double of seconds since J2000 cannot resolve 1 ns out at
    # 2060 (~1.9e9 s, quantum ~2.4e-7 s); the two-part epoch must (D-6).
    single = day * 86400.0 + sec
    assert single + 1e-9 == single
    day2, sec2 = core.tai_add_seconds(day, sec, 1e-9)
    assert abs(((day2 - day) * 86400.0 + (sec2 - sec)) - 1e-9) <= 1e-14
    # Day-boundary normalization keeps the invariant and round-trips
    # dyadic offsets exactly.
    fday, fsec = core.tai_add_seconds(day, sec, 2.5 * 86400.0)
    assert (fday, fsec) == (day + 2, sec + 43200.0)
    assert core.tai_add_seconds(fday, fsec, -2.5 * 86400.0) == (day, sec)
    assert 0.0 <= fsec < 86400.0


def _mission_text(epoch: str) -> str:
    return f"""\
schema_version = 1

[mission]
name = "time-warning-test"
epoch_utc = "{epoch}"
duration_s = 600.0

[run]
seed = 7

[integrator]
type = "rk4"
dt_s = 0.1

[initial_state.cartesian]
r_m = [6778137.0, 0.0, 0.0]
v_mps = [0.0, 7668.6, 0.0]
frame = "GCRF"

[environment]
central_body = "earth"
"""


def test_time_epoch_expiry_warning(tmp_path):
    _core_or_fail()  # the warning needs the core's leap_table_info
    past_expiry = tmp_path / "past_expiry.toml"
    past_expiry.write_text(_mission_text("2030-01-01T00:00:00Z"), encoding="utf-8")
    with pytest.warns(UserWarning, match="leap-second table"):
        resolved, errors = validate_mission_file(past_expiry)
    assert errors == []
    assert resolved is not None

    # The comparison is by UTC calendar date: an offset epoch that lands on
    # the expiry date in UTC must warn even though its local date is earlier.
    offset_epoch = tmp_path / "offset.toml"
    offset_epoch.write_text(
        _mission_text("2026-12-31T22:30:00-05:00"), encoding="utf-8"
    )
    with pytest.warns(UserWarning, match="leap-second table"):
        validate_mission_file(offset_epoch)


def test_time_epoch_before_expiry_no_warning(tmp_path):
    _core_or_fail()
    ok = tmp_path / "ok.toml"
    ok.write_text(_mission_text("2026-01-01T00:00:00Z"), encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        resolved, errors = validate_mission_file(ok)
    assert errors == []
    assert resolved is not None
    # A post-expiry epoch remains valid input: warning, never an error
    # (the run must not be blocked by an advisory staleness notice).
    assert math.isfinite(resolved["mission"]["duration_s"])
