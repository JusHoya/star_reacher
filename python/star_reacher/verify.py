"""Self-contained acceptance runner behind ``star verify`` (DX-5, FR-22 subset).

Deliberately not pytest: pytest is a dev-only dependency, and verification is
the documented first command on a bare wheel install, so it can depend on
nothing beyond the package itself. Fixture bytes are synthesized in memory
via ``star_reacher._fixtures`` (binary fixtures are never committed), and
reference values are inlined with citations rather than read from
``tests/golden/`` because an installed wheel carries no source tree.

Checks that need the compiled core import it lazily and FAIL, never skip,
when it is missing: a verification pass must mean the whole surface works.
V001-V008 cover the Phase 1 surface (determinism, SRLOG contract, RNG);
V009-V013 cover the Phase 2 math kernel (time, rotations, frames,
ephemeris evaluator, integrators/events); V014-V018 cover the Phase 3
environment models (gravity tiers, third body, shadow/SRP, atmospheres)
and the composed perturbed-run path, as quick variants of the full golden
suites in ``tests/``; V019 covers the Phase 5 exporters (NPZ bit-exact
round trip always, Parquet read-back when the optional pyarrow extra is
installed), V020 the Phase 5 viewer generator (self-containment, exact
scrub-extreme epochs, the decimation bound, and byte-identical
regeneration), and V021 the Phase 5 plot pipeline (element array
preparation against a closed-form circular orbit, a headless Agg PNG
render, and byte-identical re-rendering), as a quick variant of the full
golden suite in ``tests/python/test_plot_golden.py``.
"""

from __future__ import annotations

import hashlib
import math
import random
import struct
import tempfile
from pathlib import Path

import numpy as np

from star_reacher import _fixtures
from star_reacher._corelink import import_core
from star_reacher.export import export_csv
from star_reacher.runner import run_mission
from star_reacher.srlog import SrlogCorruptError, SrlogVersionError, load


class CheckFailure(Exception):
    """A named verify check failed; the message is the observed evidence."""


# 600 s at dt 0.1 s keeps the double-run check well inside the --quick wall
# budget while still writing 6001 truth records through the real run path.
_V001_MISSION = """\
schema_version = 1

[mission]
name = "verify-v001"
epoch_utc = "2026-01-01T00:00:00Z"
duration_s = 600.0

[run]
seed = 24601

[integrator]
type = "rk4"
dt_s = 0.1

[initial_state.cartesian]
r_m = [6778137.0, 0.0, 0.0]
v_mps = [0.0, 7668.6, 0.0]
frame = "GCRF"

[environment]
central_body = "earth"

[logging]
truth_rate_hz = 10
"""
_V001_TRUTH_RECORDS = 6001  # duration * rate + the record at t = 0

# Values whose shortest repr exercises the round-trip corners: subnormal
# minimum, negative zero, non-terminating binary fractions, and large
# magnitudes with full 17-significant-digit reprs.
_TRICKY_FLOATS = [
    0.1,
    1.0 / 3.0,
    -0.0,
    5e-324,
    math.pi,
    6.02214076e23,
    -7668.6,
    1.0000000000000002,
    2.0**-1022,
]


def _write_temp_srlog(tmpdir: Path, name: str, data: bytes) -> Path:
    path = tmpdir / name
    path.write_bytes(data)
    return path


def _check_v001(ctx: dict) -> None:
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        mission = tdp / "verify_v001.toml"
        mission.write_text(_V001_MISSION, encoding="utf-8")
        r1 = run_mission(mission, tdp / "run1")
        r2 = run_mission(mission, tdp / "run2")
        if r1.srlog_sha256 != r2.srlog_sha256:
            raise CheckFailure(
                f"double-run SHA-256 mismatch: {r1.srlog_sha256} != {r2.srlog_sha256}"
            )
        # V007 inspects this output after the temp dir is gone; keep bytes.
        ctx["v001_srlog_bytes"] = r1.srlog_path.read_bytes()
        ctx["v001_config_sha256"] = r1.config_sha256


def _check_v002(ctx: dict) -> None:
    header = _fixtures.contract_header(
        minor=999,
        extra_truth_channels=[{"name": "extra_ch", "dtype": "f64", "units": "1", "frame": ""}],
    )
    # Additive minor-version evolution may introduce header keys this reader
    # has never seen; they must be ignored, not fatal.
    header["future_key"] = {"introduced_in": "1.999"}
    records = [
        (0, (0.0, (1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1.5, 42.0)),
        (0, (0.1, (1.1, 2.1, 3.1), (4.1, 5.1, 6.1), (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1.5, 43.5)),
        _fixtures.event_record(0.0, 1, "run_start"),
    ]
    data = _fixtures.build_srlog(header, records)
    with tempfile.TemporaryDirectory() as td:
        run = load(_write_temp_srlog(Path(td), "v1_999.srlog", data))
    if run.header["format"]["minor"] != 999:
        raise CheckFailure(f"header minor version not surfaced: {run.header['format']}")
    truth = run.groups["truth"]
    if "extra_ch" not in truth.dtype.names:
        raise CheckFailure(f"added channel missing from truth dtype: {truth.dtype.names}")
    if list(truth["extra_ch"]) != [42.0, 43.5]:
        raise CheckFailure(f"added channel values wrong: {list(truth['extra_ch'])}")
    if truth["r_m"][1].tolist() != [1.1, 2.1, 3.1]:
        raise CheckFailure(f"known channel values wrong alongside added channel: {truth['r_m'][1]}")


def _check_v003(ctx: dict) -> None:
    header = _fixtures.contract_header(major=2, minor=0)
    data = _fixtures.build_srlog(header, [_fixtures.event_record(0.0, 1, "run_start")])
    with tempfile.TemporaryDirectory() as td:
        path = _write_temp_srlog(Path(td), "v2_0.srlog", data)
        try:
            load(path)
        except SrlogVersionError as exc:
            text = str(exc)
            if "2" not in text or "1" not in text:
                raise CheckFailure(f"version error does not name both versions: {text}")
            return
        except Exception as exc:  # noqa: BLE001 - evidence gathering
            raise CheckFailure(f"expected SrlogVersionError, got {type(exc).__name__}: {exc}")
    raise CheckFailure("a v2.0 file was loaded without error")


def _check_v004(ctx: dict) -> None:
    header = _fixtures.contract_header()
    good = _fixtures.build_srlog(header, [_fixtures.event_record(0.0, 1, "run_start")])
    bad_magic = bytearray(good)
    bad_magic[0] ^= 0xFF
    truncated_json = good[: 16 + 10]  # header_json_len now points past EOF
    cases = [("bad magic", bytes(bad_magic)), ("truncated header JSON", truncated_json)]
    with tempfile.TemporaryDirectory() as td:
        for label, data in cases:
            path = _write_temp_srlog(Path(td), "corrupt.srlog", data)
            try:
                load(path)
            except SrlogCorruptError:
                continue
            except Exception as exc:  # noqa: BLE001 - evidence gathering
                raise CheckFailure(
                    f"{label}: expected SrlogCorruptError, got {type(exc).__name__}: {exc}"
                )
            raise CheckFailure(f"{label}: file was loaded without error")


def _check_v005(ctx: dict) -> None:
    header = _fixtures.contract_header()
    f = _TRICKY_FLOATS
    m = len(f)
    records = []
    # Rotate the tricky values through every truth column so each float
    # exercises every position in the CSV row at least once.
    for i in range(m):
        records.append(
            (
                0,
                (
                    f[i],
                    (f[(i + 1) % m], f[(i + 2) % m], f[(i + 3) % m]),
                    (f[(i + 4) % m], f[(i + 5) % m], f[(i + 6) % m]),
                    (1.0, 0.0, 0.0, 0.0),
                    (f[(i + 7) % m], f[(i + 8) % m], f[i]),
                    f[(i + 4) % m],
                ),
            )
        )
    records.append(_fixtures.event_record(0.0, 1, 'comma, "quoted" detail'))
    records.append(_fixtures.event_record(600.0, 2, "run_end"))
    data = _fixtures.build_srlog(header, records)
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        path = _write_temp_srlog(tdp, "roundtrip.srlog", data)
        run = load(path)
        export_csv(path, tdp)
        import csv as _csv

        for gname, arr in run.groups.items():
            with open(tdp / f"{gname}.csv", newline="", encoding="utf-8") as fh:
                rows = list(_csv.reader(fh))
            if len(rows) - 1 != len(arr):
                raise CheckFailure(f"{gname}.csv row count {len(rows) - 1} != {len(arr)}")
            for ri, rec in enumerate(arr):
                cells = rows[ri + 1]
                ci = 0
                for fname in arr.dtype.names:
                    fdt = arr.dtype[fname]
                    if fdt.shape:
                        for x in rec[fname]:
                            # Bit-exactness, not closeness: compare the IEEE-754
                            # bytes of the re-parsed value with the original.
                            if struct.pack("<d", float(cells[ci])) != struct.pack("<d", float(x)):
                                raise CheckFailure(
                                    f"{gname}.csv row {ri} col {ci}: {cells[ci]!r} != {float(x)!r}"
                                )
                            ci += 1
                    elif fdt.kind == "f":
                        if struct.pack("<d", float(cells[ci])) != struct.pack(
                            "<d", float(rec[fname])
                        ):
                            raise CheckFailure(
                                f"{gname}.csv row {ri} col {ci}: {cells[ci]!r} != {float(rec[fname])!r}"
                            )
                        ci += 1
                    elif fdt.kind in ("u", "i"):
                        if int(cells[ci]) != int(rec[fname]):
                            raise CheckFailure(
                                f"{gname}.csv row {ri} col {ci}: {cells[ci]!r} != {int(rec[fname])}"
                            )
                        ci += 1
                    else:
                        if cells[ci] != str(rec[fname]):
                            raise CheckFailure(
                                f"{gname}.csv row {ri} col {ci}: {cells[ci]!r} != {rec[fname]!r}"
                            )
                        ci += 1


def _check_v006(ctx: dict) -> None:
    core = import_core()
    a = core.rng_stream_u64(42, "sensors.imu", 16)
    b = core.rng_stream_u64(42, "sensors.imu", 16)
    c = core.rng_stream_u64(42, "dispersions.mass", 16)
    d = core.rng_stream_u64(43, "sensors.imu", 16)
    if list(a) != list(b):
        raise CheckFailure("same seed and stream produced different draws")
    if list(a) == list(c):
        raise CheckFailure("different stream names produced identical draws")
    if list(a) == list(d):
        raise CheckFailure("different master seeds produced identical draws")
    if not all(0 <= int(x) <= 2**64 - 1 for x in a):
        raise CheckFailure("draws outside the u64 range")


def _check_v007(ctx: dict) -> None:
    if "v001_srlog_bytes" not in ctx:
        raise CheckFailure(
            "requires the run.srlog produced by V001, which did not complete (see the V001 result)"
        )
    with tempfile.TemporaryDirectory() as td:
        run = load(_write_temp_srlog(Path(td), "v001.srlog", ctx["v001_srlog_bytes"]))
    for key in (
        "format",
        "producer",
        "config_sha256",
        "master_seed",
        "oracle",
        "epoch_utc",
        "central_body",
        "groups",
    ):
        if key not in run.header:
            raise CheckFailure(f"header field {key!r} missing")
    if run.header["config_sha256"] != ctx["v001_config_sha256"]:
        raise CheckFailure(
            f"header config_sha256 {run.header['config_sha256']} != resolved-config hash "
            f"{ctx['v001_config_sha256']}"
        )
    if run.header["master_seed"] != "24601":
        raise CheckFailure(f"header master_seed {run.header['master_seed']!r} != '24601'")
    truth = run.groups["truth"]
    n = len(truth)
    if n != _V001_TRUTH_RECORDS:
        raise CheckFailure(f"truth record count {n} != {_V001_TRUTH_RECORDS}")
    expected_shapes = {"t_s": (n,), "r_m": (n, 3), "v_mps": (n, 3), "q_i2b": (n, 4), "w_b_radps": (n, 3), "mass_kg": (n,)}
    for fname, shape in expected_shapes.items():
        if fname not in truth.dtype.names:
            raise CheckFailure(f"truth channel {fname!r} missing: {truth.dtype.names}")
        if truth[fname].shape != shape:
            raise CheckFailure(f"truth[{fname!r}].shape {truth[fname].shape} != {shape}")
        if truth[fname].dtype != np.float64:
            raise CheckFailure(f"truth[{fname!r}].dtype {truth[fname].dtype} != float64")
    t_s = truth["t_s"]
    if not np.all(np.diff(t_s) > 0):
        raise CheckFailure("truth t_s is not strictly increasing")
    events = run.events
    codes = [int(c) for c in events["code"]]
    details = [str(d) for d in events["detail"]]
    if 1 not in codes or 2 not in codes or "run_start" not in details or "run_end" not in details:
        raise CheckFailure(f"expected run_start/run_end events, got codes {codes} details {details}")


def _check_v008(ctx: dict) -> None:
    header = _fixtures.contract_header()
    records = [
        (0, (0.0, (1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1.5)),
        (0, (0.1, (1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1.5)),
    ]
    good = _fixtures.build_srlog(header, records)
    cases = [
        ("payload cut mid-record", good[:-3]),
        ("dangling group index byte", good[: len(good) - (2 + 120) + 1]),
    ]
    with tempfile.TemporaryDirectory() as td:
        for label, data in cases:
            path = _write_temp_srlog(Path(td), "truncated.srlog", data)
            try:
                load(path)
            except SrlogCorruptError:
                continue
            except Exception as exc:  # noqa: BLE001 - evidence gathering
                raise CheckFailure(
                    f"{label}: expected SrlogCorruptError, got {type(exc).__name__}: {exc}"
                )
            raise CheckFailure(f"{label}: truncated file was loaded without error")


# --------------------------------------------------------------------------
# Phase 2 checks (V009-V013): quick, self-contained variants of the golden
# acceptance suites, exercised through the installed core bindings.
# --------------------------------------------------------------------------

# UTC -> TAI -> TT golden epochs, transcribed from the committed golden file
# tests/golden/time/utc_tai_tt.toml (provenance and tolerances in
# tests/golden/time/manifest.toml). Citations: leap-second history per the
# IERS Bulletin C series, verified through Bulletin C 71 (January 2026:
# no leap second at the end of June 2026, TAI-UTC = 37 s); TT = TAI +
# 32.184 s exactly per IAU 1991 Resolution A4 (Kaplan, USNO Circular 179,
# 2005); the 2007-04-05 anchor is the published worked example in the IAU
# SOFA cookbook "SOFA Tools for Earth Attitude". Comparisons are binary64
# bit equality, the same discipline the golden manifest records: the
# compiled core performs the identical IEEE-754 operation sequence under
# the D-10 strict-FP build flags, so any difference is an algorithmic
# deviation, not roundoff.
_V009_EPOCHS = [
    # (name, (y, mo, d, h, mi, s), tai_day, tai_sec_hex, tt_jd1_hex, tt_jd2_hex)
    (
        "sofa_cookbook_2007_04_05",
        (2007, 4, 5, 12, 0, 0.0),
        2651,
        "0x1.51c2000000000p+15",  # 43233.0 s = 12:00:33 TAI (cookbook: TAI 12:00:33.000)
        "0x1.2b959c0000000p+21",
        "0x1.0062e2f46e5b0p-1",  # TT 12:01:05.184 (cookbook)
    ),
    (
        "leap_2016_leap_start",
        (2016, 12, 31, 23, 59, 60.0),  # inside the inserted leap second
        6210,
        "0x1.2000000000000p+5",  # 36.0 s: TAI already on the next day
        "0x1.2c04d40000000p+21",
        "0x1.9dc02832069bbp-11",
    ),
    (
        "epoch_2030_01_01",
        (2030, 1, 1, 0, 0, 0.0),  # post-expiry: last tabulated offset applies
        10958,
        "0x1.2800000000000p+5",  # 37.0 s
        "0x1.2c99340000000p+21",
        "0x1.a3d19a5a3a300p-11",
    ),
]


def _check_v009(ctx: dict) -> None:
    core = import_core()
    for name, utc, tai_day, tai_sec_hex, tt_jd1_hex, tt_jd2_hex in _V009_EPOCHS:
        day, sec = core.utc_to_tai(*utc)
        if (day, sec) != (tai_day, float.fromhex(tai_sec_hex)):
            raise CheckFailure(
                f"{name}: utc_to_tai gave (day={day}, sec={sec!r}), expected "
                f"(day={tai_day}, sec={float.fromhex(tai_sec_hex)!r})"
            )
        t1, t2 = core.tt_jd(day, sec)
        if (t1, t2) != (float.fromhex(tt_jd1_hex), float.fromhex(tt_jd2_hex)):
            raise CheckFailure(
                f"{name}: tt_jd gave ({t1!r}, {t2!r}), expected "
                f"({float.fromhex(tt_jd1_hex)!r}, {float.fromhex(tt_jd2_hex)!r})"
            )
        # The inverse must restore the exact calendar fields, including the
        # second-60 rendering inside an inserted leap second.
        back = core.tai_to_utc(day, sec)
        if tuple(back) != utc:
            raise CheckFailure(f"{name}: tai_to_utc round trip gave {back}, expected {utc}")


def _quat_angle_rad(core, p: tuple, q: tuple) -> float:
    """Rotation angle between two unit quaternions.

    Angle of the relative rotation q (x) p^-1 via atan2 of the vector-part
    norm: 2*acos(|p . q|) cannot resolve angles near 1e-13 rad (the cosine
    is within one ulp of 1), while the vector norm is ~angle/2 and keeps
    full relative precision.
    """
    dw, dx, dy, dz = core.quat_multiply(*core.quat_conjugate(*p), *q)
    return 2.0 * math.atan2(math.hypot(dx, dy, dz), abs(dw))


def _check_v010(ctx: dict) -> None:
    core = import_core()
    # Seeded stdlib Mersenne Twister: bit-identical draws on every platform,
    # so the attitude set is the same everywhere. 100 attitudes is the quick
    # tier of the 1,000-attitude Phase 2 exit-criterion-6 sweep in
    # tests/python/test_frames.py; the 1e-13 rad tolerance is the criterion's.
    rng = random.Random(20260702)
    worst = 0.0
    worst_case = ""
    for i in range(100):
        raw = [rng.uniform(-1.0, 1.0) for _ in range(4)]
        if math.hypot(*raw) < 0.1:
            continue  # degenerate draw (probability ~1e-5); density is ample
        q = core.quat_normalize(*raw)
        dcm = core.quat_to_dcm(*q)
        q_back = core.dcm_to_quat(dcm)
        for label, chain in (
            ("quat->dcm->quat", q_back),
            (
                "quat->dcm->euler321->dcm->quat",
                core.dcm_to_quat(core.dcm_from_euler321(*core.euler321_from_dcm(dcm))),
            ),
            (
                "quat->dcm->euler313->dcm->quat",
                core.dcm_to_quat(core.dcm_from_euler313(*core.euler313_from_dcm(dcm))),
            ),
        ):
            err = _quat_angle_rad(core, q, chain)
            if err > worst:
                worst, worst_case = err, f"attitude {i} {label}"
    if worst > 1e-13:
        raise CheckFailure(
            f"worst rotation round-trip error {worst:.3e} rad at {worst_case} "
            f"(tolerance 1e-13 rad, Phase 2 exit criterion 6)"
        )


# GCRF -> ITRF matrix at 2020-01-01T00:00:00 UTC (two-part TAI epoch day
# 7305, sec 37.0; dUT1 = 0), transcribed from the committed golden file
# tests/golden/frames/earth_chain.toml (case epoch_2020_01_01). The elements
# were generated with ERFA (pyerfa 2.0.1.5), the reference implementation of
# the IAU SOFA algorithms, composing exactly the IAU 2006/2000B CIO-based
# chain the core implements (polar motion neglected; provenance and the
# 1e-11 tolerance derivation in tests/golden/frames/manifest.toml).
_V011_TAI = (7305, 37.0)
_V011_DCM_HEX = [
    "-0x1.5ee5e02671bfap-3", "0x1.f86dc3de34908p-1", "0x1.644972ba8ccacp-12",
    "-0x1.f86d87a2491b2p-1", "-0x1.5ee60d4f0364dp-3", "0x1.ed08472a19ae9p-10",
    "0x1.f500bda826a03p-10", "-0x1.a3d0085360000p-17", "0x1.ffffc2b791d32p-1",
]


def _check_v011(ctx: dict) -> None:
    core = import_core()
    got = core.gcrf_to_itrf(*_V011_TAI, 0.0)
    expected = [float.fromhex(h) for h in _V011_DCM_HEX]
    # Phase 2 exit criterion 1: rotation-matrix elements to 1e-11 vs ERFA.
    for k, (g, e) in enumerate(zip(got, expected)):
        if abs(g - e) > 1e-11:
            raise CheckFailure(
                f"gcrf_to_itrf element [{k // 3}][{k % 3}] = {g!r} differs from "
                f"ERFA {e!r} by {abs(g - e):.3e} (tolerance 1e-11)"
            )
    # Orthonormality: C C^T = I and det C = +1. The chain is a product of
    # exact rotations, so residuals are pure roundoff (~1e-16 observed);
    # 1e-14 keeps margin while failing on any non-orthonormal construction.
    c = [got[0:3], got[3:6], got[6:9]]
    for i in range(3):
        for j in range(3):
            dot = sum(c[i][k] * c[j][k] for k in range(3))
            target = 1.0 if i == j else 0.0
            if abs(dot - target) > 1e-14:
                raise CheckFailure(
                    f"C C^T [{i}][{j}] = {dot!r} deviates from {target} by "
                    f"{abs(dot - target):.3e} (tolerance 1e-14)"
                )
    det = (
        c[0][0] * (c[1][1] * c[2][2] - c[1][2] * c[2][1])
        - c[0][1] * (c[1][0] * c[2][2] - c[1][2] * c[2][0])
        + c[0][2] * (c[1][0] * c[2][1] - c[1][1] * c[2][0])
    )
    if abs(det - 1.0) > 1e-14:
        raise CheckFailure(f"det C = {det!r} deviates from +1 by {abs(det - 1.0):.3e}")


# Synthetic SREPH design for V012. All quantities are dyadic (exactly
# representable in binary64) and the record interval is 2^16 s, so every
# arithmetic step of the documented evaluation (scaled time x, Chebyshev
# recurrence, coefficient accumulation, the 2/intlen derivative scale, and
# the km -> m conversion by 1000) is exact: the check may assert bit
# equality against an independently computed closed-form polynomial value.
_V012_INTLEN = 65536.0  # 2^16 s, so 2/intlen = 2^-15 is exact
_V012_RECORDS = [
    # record 0: [x, y, z] component coefficient triples (c0, c1, c2) [km]
    [[3.0, 0.5, 0.25], [-2.0, 1.5, -0.5], [0.125, -0.25, 1.0]],
    # record 1
    [[10.0, 1.0, 0.5], [20.0, -1.0, 0.25], [-5.0, 2.0, -0.125]],
]


def _v012_expected(record: int, x: float) -> tuple[list[float], list[float]]:
    """Closed-form Chebyshev value/rate in m and m/s (T2 = 2x^2 - 1)."""
    r = []
    v = []
    for c0, c1, c2 in _V012_RECORDS[record]:
        r.append((c0 + c1 * x + c2 * (2.0 * x * x - 1.0)) * 1000.0)
        v.append((c1 + c2 * (4.0 * x)) * (2.0 / _V012_INTLEN) * 1000.0)
    return r, v


def _check_v012(ctx: dict) -> None:
    core = import_core()
    data = _fixtures.build_sreph(
        [
            {
                "name": "testbody",
                "target": 999,
                "center": 0,
                "kind": 0,
                "init_tdb_s": 0.0,
                "intlen_s": _V012_INTLEN,
                "records": _V012_RECORDS,
            }
        ]
    )
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "v012.sreph"
        path.write_bytes(data)
        eph = core.Ephemeris.load(str(path))

        if eph.bodies() != ["testbody"]:
            raise CheckFailure(f"bodies() = {eph.bodies()}, expected ['testbody']")
        span = (eph.span_start_tdb_s(), eph.span_end_tdb_s())
        if span != (0.0, 2.0 * _V012_INTLEN):
            raise CheckFailure(f"span = {span}, expected (0.0, {2.0 * _V012_INTLEN})")

        # (epoch, record, scaled time x). The shared boundary epoch must
        # evaluate in the record that begins there, and the final epoch of
        # the segment evaluates in the last record at x = +1 exactly
        # (docs/formats/sreph_v1.md section 5).
        cases = [
            (16384.0, 0, -0.5),
            (65536.0, 1, -1.0),
            (114688.0, 1, 0.5),
            (131072.0, 1, 1.0),
        ]
        for tdb_s, record, x in cases:
            r_got, v_got = eph.state("testbody", tdb_s)
            r_exp, v_exp = _v012_expected(record, x)
            if list(r_got) != r_exp or list(v_got) != v_exp:
                raise CheckFailure(
                    f"state at t={tdb_s}: got r={list(r_got)}, v={list(v_got)}; "
                    f"expected r={r_exp}, v={v_exp} (record {record}, x={x})"
                )

        # Out-of-span epochs are refused, never extrapolated
        # (std::out_of_range crosses the binding as IndexError).
        for bad_t in (-2.0, 131073.0):
            try:
                eph.state("testbody", bad_t)
            except IndexError:
                pass
            except Exception as exc:  # noqa: BLE001 - evidence gathering
                raise CheckFailure(
                    f"out-of-span t={bad_t}: expected IndexError, got "
                    f"{type(exc).__name__}: {exc}"
                )
            else:
                raise CheckFailure(f"out-of-span t={bad_t} was evaluated without error")

        try:
            eph.state("nosuchbody", 0.0)
        except ValueError:
            pass
        except Exception as exc:  # noqa: BLE001 - evidence gathering
            raise CheckFailure(
                f"unknown body: expected ValueError, got {type(exc).__name__}: {exc}"
            )
        else:
            raise CheckFailure("unknown body was evaluated without error")

        # Loader contract: a truncated file is rejected loudly.
        truncated = Path(td) / "truncated.sreph"
        truncated.write_bytes(data[:100])
        try:
            core.Ephemeris.load(str(truncated))
        except RuntimeError:
            pass
        except Exception as exc:  # noqa: BLE001 - evidence gathering
            raise CheckFailure(
                f"truncated file: expected RuntimeError, got {type(exc).__name__}: {exc}"
            )
        else:
            raise CheckFailure("truncated SREPH file was loaded without error")


# Reference eccentric orbit (a = 8000 km, e = 0.15), transcribed from the
# committed golden tests/golden/integrators/kepler_orbit.toml (case
# "definition"; provenance in tests/golden/integrators/manifest.toml).
# mu per IERS Conventions 2010, TN No. 36.
_V013_MU = float.fromhex("0x1.6a8665bda5400p+48")
_V013_R0 = [
    float.fromhex("-0x1.8b4654850dd51p+22"),
    float.fromhex("-0x1.4bacc4ca4ecb4p+22"),
    float.fromhex("0x1.8000000000000p-30"),
]
_V013_V0 = [
    float.fromhex("0x1.72b947a180246p+11"),
    float.fromhex("-0x1.371562d91d310p+12"),
    float.fromhex("-0x1.9cc00b2dd40c1p+11"),
]
_V013_PERIOD_S = float.fromhex("0x1.bd114e244a5b2p+12")  # 7121.081577578023 s
# First apoapsis passage t = (pi - M0)/n, from the committed analytic golden
# tests/golden/integrators/apsis_times.toml (case apsis_0).
_V013_FIRST_APO_S = float.fromhex("0x1.7675ed69bfca6p+10")  # 1497.842615544601 s


def _check_v013(ctx: dict) -> None:
    core = import_core()
    # Quick variant of Phase 2 exit criterion 4 (which gates < 1e-10 drift
    # over 10 orbits at rtol 1e-12 in the pytest/doctest suites): 2 orbits
    # at rtol 1e-11. Invariant drift under adaptive RKF7(8) scales roughly
    # with the local tolerance, so the one-order-looser rtol carries a
    # one-order-looser bound; 1e-9 keeps an order of margin over that and
    # still fails on any conservation-violating integrator defect.
    drift = core.twobody_drift(
        _V013_MU, _V013_R0, _V013_V0, 2.0, 1e-11, 1e-6, 1e-9, 10.0, 600.0
    )
    if drift["max_energy_rel"] >= 1e-9 or drift["max_hmag_rel"] >= 1e-9:
        raise CheckFailure(
            f"two-body invariant drift over 2 orbits at rtol 1e-11: "
            f"energy {drift['max_energy_rel']:.3e}, |h| {drift['max_hmag_rel']:.3e} "
            f"(bound 1e-9)"
        )
    if drift["steps_accepted"] <= 20:
        raise CheckFailure(f"implausibly few accepted steps: {drift}")

    # Apsis events over (0, 1.75 T]: analytic passages at t = (k pi - M0)/n,
    # i.e. 0.2104 T (apo), 0.7104 T (peri), 1.2104 T (apo), 1.7104 T (peri)
    # for this orbit's M0 - exactly four, alternating, spaced T/2, with the
    # nearest passage ~280 s clear of the span end so the count is robust.
    # rtol 1e-11 matches the drift half above: located event times deviate
    # from analytic in proportion to the trajectory tolerance (1.7e-3 s
    # measured at rtol 1e-9, so ~2e-5 s expected here), which gives the
    # 1e-3 s bounds below about two orders of margin while still failing
    # on a misordered or misconverged event root. The full suites gate
    # < 1 us at rtol 1e-12 (exit criterion 5).
    t_end = 1.75 * _V013_PERIOD_S
    hits = core.apsis_events(
        _V013_MU, _V013_R0, _V013_V0, t_end, 1e-11, 1e-6, 1e-9, 10.0, 600.0, 1e-6
    )
    kinds = [h["kind"] for h in hits]
    times = [h["t_s"] for h in hits]
    if kinds != ["apoapsis", "periapsis", "apoapsis", "periapsis"]:
        raise CheckFailure(f"expected apo/peri/apo/peri, got {kinds} at {times}")
    if any(t2 <= t1 for t1, t2 in zip(times, times[1:])):
        raise CheckFailure(f"event times not strictly increasing: {times}")
    if abs(times[0] - _V013_FIRST_APO_S) > 1e-3:
        raise CheckFailure(
            f"first apoapsis at {times[0]!r} s, analytic {_V013_FIRST_APO_S!r} s"
        )
    half_t = 0.5 * _V013_PERIOD_S
    for t1, t2 in zip(times, times[1:]):
        if abs((t2 - t1) - half_t) > 1e-3:
            raise CheckFailure(
                f"apsis spacing {t2 - t1!r} s deviates from T/2 = {half_t!r} s"
            )


# --------------------------------------------------------------------------
# Phase 3 checks (V014-V018): environment force models and the composed
# perturbed-run path. Self-contained on a bare wheel: fields and ephemerides
# are synthesized in memory (star_reacher.data_fetch.write_srgrav,
# star_reacher._fixtures.build_sreph); reference values are inlined with
# citations, never read from tests/golden/ or data/.
# --------------------------------------------------------------------------


def _synthetic_j2_field(tmpdir: Path) -> tuple[Path, float, float, float]:
    """Write a J2-only SRGRAV field; returns (path, gm, ref_radius, c20bar).

    C-bar(2,0) = -4.84165143790815e-4 is the EGM2008 fully normalized zonal
    (Pavlis et al. 2012, as distributed by ICGEM); GM and R are the IERS
    TN36 GM and the EGM2008 reference radius. The field is synthesized here
    (never read from the source tree) - the check compares the compiled
    Pines evaluation against the independent closed-form J2 acceleration.
    """
    import numpy as _np

    from star_reacher import data_fetch as df

    gm = 3.986004418e14
    radius = 6378136.3
    c20 = -4.84165143790815e-4
    n_max = 2
    cbar = _np.zeros((n_max + 1, n_max + 1))
    sbar = _np.zeros((n_max + 1, n_max + 1))
    cbar[0, 0] = 1.0
    cbar[2, 0] = c20
    field = df.GravityCoefficients(
        name="V014J2",
        gm_m3ps2=gm,
        ref_radius_m=radius,
        n_max=n_max,
        m_max=n_max,
        tide_system="tide_free",
        cbar=cbar,
        sbar=sbar,
        source_sha256="00" * 32,
    )
    path = tmpdir / "v014_j2.srgrav"
    df.write_srgrav(path, field)
    return path, gm, radius, c20


def _check_v014(ctx: dict) -> None:
    core = import_core()
    with tempfile.TemporaryDirectory() as td:
        path, gm, radius, c20 = _synthetic_j2_field(Path(td))
        # J2 = -sqrt(5) * C-bar(2,0): degree-2 zonal denormalization
        # (4-pi geodesy normalization N(2,0) = sqrt(5); ch:gravity).
        j2 = -math.sqrt(5.0) * c20
        for r_bf in ((7000.0e3, 0.0, 0.0), (4000.0e3, 3000.0e3, 5000.0e3)):
            got = core.gravity_accel(str(path), "j2", -1, -1, r_bf)
            # Closed-form point-mass + J2 acceleration in the body-fixed
            # frame (Vallado, Fundamentals of Astrodynamics and
            # Applications, 4th ed., Sect. 8.7.1 / eq:gravity:potential):
            #   a = -GM/r^3 * r + a_J2,
            #   a_J2 = -(3/2) J2 (GM/r^2)(R/r)^2 *
            #          [(1-5(z/r)^2) x/r, (1-5(z/r)^2) y/r, (3-5(z/r)^2) z/r]
            x, y, z = r_bf
            r = math.sqrt(x * x + y * y + z * z)
            k = -1.5 * j2 * (gm / r**2) * (radius / r) ** 2
            zr2 = (z / r) ** 2
            expected = [
                -gm / r**3 * x + k * (1.0 - 5.0 * zr2) * (x / r),
                -gm / r**3 * y + k * (1.0 - 5.0 * zr2) * (y / r),
                -gm / r**3 * z + k * (3.0 - 5.0 * zr2) * (z / r),
            ]
            norm = math.sqrt(sum(e * e for e in expected))
            err = math.sqrt(sum((g - e) ** 2 for g, e in zip(got, expected)))
            # An independent formulation agrees to roundoff; a Pines
            # recursion or tier-wiring defect shows at O(J2) ~ 1e-3.
            if err > 1e-12 * norm:
                raise CheckFailure(
                    f"J2-tier acceleration at {r_bf} differs from the closed "
                    f"form by {err / norm:.3e} relative (gate 1e-12)"
                )
        # Point-mass tier: exactly the -GM/r^3 law to a few ulp.
        r_bf = (7000.0e3, 0.0, 0.0)
        got = core.gravity_accel(str(path), "pointmass", -1, -1, r_bf)
        expected0 = -gm / (7000.0e3) ** 2
        if abs(got[0] - expected0) > 1e-13 * abs(expected0) or got[1] != 0.0 or got[2] != 0.0:
            raise CheckFailure(
                f"point-mass tier at {r_bf} gave {list(got)}, expected "
                f"[{expected0!r}, 0.0, 0.0]"
            )


# Battin f(q) third-body reference states, transcribed from the committed
# golden tests/golden/thirdbody/states.toml (cases sun_leo_align and
# sun_leo_perpendicular): the naive two-vector-difference acceleration
# evaluated by mpmath at 60 significant digits from the exact binary64
# inputs, rounded once to binary64 (provenance and the 1e-12 norm-relative
# gate in tests/golden/thirdbody/manifest.toml; formulation per Battin 1999).
_V015_CASES = [
    (
        "sun_leo_align",
        "0x1.cc6546bb37958p+66",
        ("0x1.9db4640000000p+22", "0x0.0p+0", "0x0.0p+0"),
        ("0x1.16a5d2d360000p+37", "0x0.0p+0", "0x0.0p+0"),
        ("0x1.207e0fa5e17c6p-21", "0x0.0p+0", "0x0.0p+0"),
    ),
    (
        "sun_leo_perpendicular",
        "0x1.cc6546bb37958p+66",
        ("0x0.0p+0", "0x1.9db4640000000p+22", "0x0.0p+0"),
        ("0x1.16a5d2d360000p+37", "0x0.0p+0", "0x0.0p+0"),
        ("-0x1.413806bfc9877p-36", "-0x1.20790aa2ffad1p-22", "0x0.0p+0"),
    ),
]


def _check_v015(ctx: dict) -> None:
    core = import_core()
    for name, gm_hex, r_sc_hex, r_third_hex, a_ref_hex in _V015_CASES:
        gm = float.fromhex(gm_hex)
        r_sc = [float.fromhex(h) for h in r_sc_hex]
        r_third = [float.fromhex(h) for h in r_third_hex]
        a_ref = [float.fromhex(h) for h in a_ref_hex]
        got = core.thirdbody_accel(gm, r_sc, r_third)
        norm = math.sqrt(sum(a * a for a in a_ref))
        err = math.sqrt(sum((g - a) ** 2 for g, a in zip(got, a_ref)))
        if err > 1e-12 * norm:
            raise CheckFailure(
                f"{name}: Battin acceleration differs from the extended-"
                f"precision reference by {err / norm:.3e} relative (gate 1e-12, "
                f"Phase 3 exit criterion 7)"
            )


# Conical-shadow reference geometries, transcribed from the committed golden
# tests/golden/srp/shadow_fraction.toml (cases full_sun_subsolar,
# umbra_anti_sun, penumbra_mid; mpmath-generated, provenance and tolerance
# derivations in tests/golden/srp/manifest.toml). Sun and occulter share the
# LEO Earth-shadow geometry: |r_sun| ~ 1 au, R_sun the IAU 2015 nominal
# solar radius, R_occ the WGS84 equatorial radius.
_V016_SUN = ("0x1.16a5d2d360000p+37", "0x0.0p+0", "0x0.0p+0")
_V016_RSUN = "0x1.4bbc510000000p+29"
_V016_ROCC = "0x1.854a640000000p+22"
_V016_CASES = [
    # (name, r_sc hex triple, expected nu hex, exact)
    ("full_sun_subsolar", ("0x1.9db4640000000p+22", "0x0.0p+0", "0x0.0p+0"), "0x1.0p+0", True),
    ("umbra_anti_sun", ("-0x1.9db4640000000p+22", "0x0.0p+0", "0x0.0p+0"), "0x0.0p+0", True),
    (
        "penumbra_mid",
        ("-0x1.17ff80d40d10ap+21", "0x1.854beb345949dp+22", "0x0.0p+0"),
        "0x1.0034c7cbe1cadp-1",
        False,
    ),
]


def _check_v016(ctx: dict) -> None:
    core = import_core()
    r_sun = [float.fromhex(h) for h in _V016_SUN]
    radius_sun = float.fromhex(_V016_RSUN)
    radius_occ = float.fromhex(_V016_ROCC)
    origin = [0.0, 0.0, 0.0]
    for name, r_sc_hex, nu_hex, exact in _V016_CASES:
        r_sc = [float.fromhex(h) for h in r_sc_hex]
        nu_ref = float.fromhex(nu_hex)
        nu = core.shadow_fraction(r_sc, r_sun, radius_sun, origin, radius_occ)
        if exact:
            # The piecewise model returns the constants exactly outside the
            # penumbra (determinism demands bit-exact 0/1; srp manifest).
            if nu != nu_ref:
                raise CheckFailure(f"{name}: nu = {nu!r}, expected exactly {nu_ref!r}")
        elif abs(nu - nu_ref) > 1e-8:
            # 1e-8 abs: the srp manifest's cross-platform libm bound for
            # large-occulter penumbra geometry (observed worst 1.0e-9).
            raise CheckFailure(
                f"{name}: nu = {nu!r} differs from {nu_ref!r} by "
                f"{abs(nu - nu_ref):.3e} (gate 1e-8 abs)"
            )
    # In total umbra the cannonball SRP acceleration is exactly zero.
    a_umbra = core.srp_accel(0.02, 0.0, [float.fromhex(h) for h in _V016_CASES[1][1]], r_sun)
    if list(a_umbra) != [0.0, 0.0, 0.0]:
        raise CheckFailure(f"srp_accel with nu = 0 gave {list(a_umbra)}, expected exact zeros")


def _check_v017(ctx: dict) -> None:
    core = import_core()
    # USSA76 sea level: rho = 1.2250 kg/m^3 (U.S. Standard Atmosphere 1976,
    # Table I, z = 0), print precision (4 significant figures, Phase 3 exit
    # criterion 4).
    rho0 = core.ussa76_density(0.0)
    if abs(rho0 - 1.2250) > 0.5e-4:
        raise CheckFailure(f"USSA76 sea-level density {rho0!r}, published 1.2250 kg/m^3")
    # Harris-Priester at the 500 km node pinned to the bulge minimum
    # (cos_psi = -1) and maximum (cos_psi = +1): the committed
    # Montenbruck & Gill Sect. 3.5.2 table values 3.916e-13 / 2.042e-12
    # kg/m^3, exact at a node by construction (atmosphere manifest,
    # ATM-HP-NODES discipline).
    rho_min = core.hp_density(500000.0, -1.0, 4.0)
    rho_max = core.hp_density(500000.0, 1.0, 4.0)
    if rho_min != 3.916e-13:
        raise CheckFailure(f"HP rho_min(500 km) = {rho_min!r}, table 3.916e-13")
    if rho_max != 2.042e-12:
        raise CheckFailure(f"HP rho_max(500 km) = {rho_max!r}, table 2.042e-12")
    # Above the 1000 km table ceiling HP is exactly zero (Orekit-compatible).
    if core.hp_density(1100000.0, 0.5, 4.0) != 0.0:
        raise CheckFailure("HP density above 1000 km is not exactly zero")
    # Mars piecewise-exponential 40 km node: committed node value (NASA
    # Glenn curve-fit derivation, PRD A-3 confidence low; bit-exact at a
    # node per tests/golden/atmosphere/manifest.toml, ATM-MARS-NODES).
    rho_mars = core.mars_density(40000.0)
    ref_mars = float.fromhex("0x1.43f814fd34db9p-11")
    if rho_mars != ref_mars:
        raise CheckFailure(f"Mars density at 40 km node = {rho_mars!r}, committed {ref_mars!r}")


_V018_MISSION = """\
schema_version = 1

[mission]
name = "verify-v018-{tag}"
epoch_utc = "2026-01-01T00:00:00Z"
duration_s = {duration}

[run]
seed = 20260118

[integrator]
{integrator}

[spacecraft]
mass_kg = 500.0
cd_a_over_m_m2pkg = 0.0044
cr_a_over_m_m2pkg = 0.02

[initial_state.cartesian]
r_m = [6878000.0, 0.0, 0.0]
v_mps = [0.0, 7350.0, 2000.0]
frame = "GCRF"

[environment]
central_body = "earth"
ephemeris = "{ephemeris}"
third_bodies = ["sun", "moon"]

[environment.gravity]
model = "harmonic"
field = "{field}"
degree = 2
order = 0

[environment.srp]

[environment.drag]
atmosphere = "harris_priester"
hp_exponent_n = 4.0

[logging]
truth_rate_hz = 1
"""

_V018_RKF78 = """type = "rkf78"
rtol = 1e-9
atol_pos_m = 1e-4
atol_vel_mps = 1e-7
h_init_s = 30.0
h_max_s = 30.0"""

_V018_RK4 = """type = "rk4"
dt_s = 1.0"""


def _v018_synthetic_ephemeris(tmpdir: Path) -> Path:
    """A short constant-position Chebyshev ephemeris covering the epoch.

    Segment layout matches the DE440 repack (sun/emb SSB-centered, earth/
    moon EMB-centered, kind 0 in km); the constant positions are
    representative magnitudes (Sun at the SSB, EMB at ~1 au, real-scale
    EMB offsets) so the third-body, SRP, and bulge geometry are physically
    sensible. The check gates DETERMINISM of the composed run, not
    astronomy - the real-ephemeris physics is gated by the golden suites.
    """
    core = import_core()
    day, sec = core.utc_to_tai(2026, 1, 1, 0, 0, 0.0)
    jd1, jd2 = core.tdb_jd(day, sec)
    tdb_s = ((jd1 - 2451545.0) + jd2) * 86400.0
    init = tdb_s - 86400.0
    intlen = 3.0 * 86400.0

    def const_record(x_km: float, y_km: float, z_km: float) -> list:
        return [[[x_km, 0.0, 0.0], [y_km, 0.0, 0.0], [z_km, 0.0, 0.0]]]

    segments = [
        {"name": "sun", "target": 10, "center": 0, "kind": 0,
         "init_tdb_s": init, "intlen_s": intlen,
         "records": const_record(0.0, 0.0, 0.0)},
        {"name": "emb", "target": 3, "center": 0, "kind": 0,
         "init_tdb_s": init, "intlen_s": intlen,
         "records": const_record(-1.4959787e8, 0.0, 0.0)},
        {"name": "earth", "target": 399, "center": 3, "kind": 0,
         "init_tdb_s": init, "intlen_s": intlen,
         "records": const_record(4671.0, 0.0, 0.0)},
        {"name": "moon", "target": 301, "center": 3, "kind": 0,
         "init_tdb_s": init, "intlen_s": intlen,
         "records": const_record(-379700.0, 0.0, 0.0)},
    ]
    path = tmpdir / "v018.sreph"
    path.write_bytes(_fixtures.build_sreph(segments))
    return path


def _check_v018(ctx: dict) -> None:
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        field_path, _gm, _radius, _c20 = _synthetic_j2_field(tdp)
        eph_path = _v018_synthetic_ephemeris(tdp)
        for tag, integ, duration, records in (
            ("rkf78", _V018_RKF78, "300.0", 301),
            ("rk4", _V018_RK4, "120.0", 121),
        ):
            mission = tdp / f"v018_{tag}.toml"
            mission.write_text(
                _V018_MISSION.format(
                    tag=tag,
                    duration=duration,
                    integrator=integ,
                    ephemeris=eph_path.as_posix(),
                    field=field_path.as_posix(),
                ),
                encoding="utf-8",
            )
            r1 = run_mission(mission, tdp / f"{tag}_run1")
            r2 = run_mission(mission, tdp / f"{tag}_run2")
            if r1.srlog_sha256 != r2.srlog_sha256:
                raise CheckFailure(
                    f"{tag}: perturbed double-run SHA-256 mismatch: "
                    f"{r1.srlog_sha256} != {r2.srlog_sha256}"
                )
            if r1.summary["truth_records"] != records:
                raise CheckFailure(
                    f"{tag}: truth record count {r1.summary['truth_records']} "
                    f"!= {records}"
                )


# --------------------------------------------------------------------------
# Phase 5 check (V019): exporter fidelity. Pure Python on a synthesized log,
# so it passes on a core-less install like the other format checks; the
# Parquet half runs only when the optional pyarrow extra is importable,
# because verify must hold on a bare wheel where extras are absent.
# --------------------------------------------------------------------------


def _check_v019(ctx: dict) -> None:
    from star_reacher.export import export_npz, export_parquet, load_npz

    header = _fixtures.contract_header()
    f = _TRICKY_FLOATS
    m = len(f)
    records = []
    # The same tricky-float rotation as V005: the NPZ round trip must
    # preserve subnormals, negative zero, and full-precision reprs.
    for i in range(m):
        records.append(
            (
                0,
                (
                    float(i),
                    (f[(i + 1) % m], f[(i + 2) % m], f[(i + 3) % m]),
                    (f[(i + 4) % m], f[(i + 5) % m], f[(i + 6) % m]),
                    (1.0, -0.0, 0.0, 0.0),
                    (f[(i + 7) % m], f[(i + 8) % m], f[i]),
                    f[(i + 4) % m],
                ),
            )
        )
    records.append(_fixtures.event_record(0.0, 1, "run_start"))
    records.append(_fixtures.event_record(1.0, 7, 'comma, "quote", newline\n'))
    records.append(_fixtures.event_record(600.0, 2, "run_end"))
    data = _fixtures.build_srlog(header, records)
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        path = _write_temp_srlog(tdp, "v019.srlog", data)
        run = load(path)

        npz_path = export_npz(path, tdp)
        back = load_npz(npz_path)
        if back.header != run.header:
            raise CheckFailure("NPZ round trip changed the header dict")
        if set(back.groups) != set(run.groups):
            raise CheckFailure(
                f"NPZ round trip changed the group set: {sorted(back.groups)} "
                f"!= {sorted(run.groups)}"
            )
        for gname, arr in run.groups.items():
            got = back.groups[gname]
            if got.dtype != arr.dtype or len(got) != len(arr):
                raise CheckFailure(f"NPZ group '{gname}': dtype or length changed")
            for fname in arr.dtype.names:
                if arr.dtype[fname].base.kind == "O":
                    if list(got[fname]) != list(arr[fname]):
                        raise CheckFailure(
                            f"NPZ group '{gname}' channel '{fname}': string "
                            f"values changed"
                        )
                elif got[fname].tobytes() != arr[fname].tobytes():
                    # Bit-exactness, not closeness (Phase 5 exit criterion 3).
                    raise CheckFailure(
                        f"NPZ group '{gname}' channel '{fname}': bytes changed"
                    )

        try:
            import pyarrow.parquet as pq  # noqa: F401 - availability probe
        except ImportError:
            return  # bare-wheel path: the pyarrow extra is not installed
        written = export_parquet(path, tdp)
        for parquet_path in written:
            gname = parquet_path.stem
            arr = run.groups[gname]
            table = pq.read_table(parquet_path)
            expected_columns = []
            for fname in arr.dtype.names:
                shape = arr.dtype[fname].shape
                if shape:
                    expected_columns.extend(f"{fname}_{i}" for i in range(shape[0]))
                else:
                    expected_columns.append(fname)
            if table.num_rows != len(arr):
                raise CheckFailure(
                    f"{gname}.parquet: {table.num_rows} rows != {len(arr)}"
                )
            if table.column_names != expected_columns:
                raise CheckFailure(
                    f"{gname}.parquet: columns {table.column_names} != "
                    f"{expected_columns}"
                )


# --------------------------------------------------------------------------
# Phase 5 check (V020): the FR-19 viewer generator, as a quick variant of
# the full suite in tests/python/test_viewer.py.
# --------------------------------------------------------------------------


def _check_v020(ctx: dict) -> None:
    from datetime import datetime, timedelta, timezone

    from star_reacher.viewer import (
        extract_view_data,
        generate_view,
        scan_external_references,
    )

    if "v001_srlog_bytes" not in ctx:
        raise CheckFailure(
            "requires the run.srlog produced by V001, which did not complete (see the V001 result)"
        )
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        srlog = _write_temp_srlog(tdp, "v019.srlog", ctx["v001_srlog_bytes"])
        result = generate_view(srlog, tdp / "view.html")
        html = (tdp / "view.html").read_text(encoding="utf-8")

        findings = scan_external_references(html)
        if findings:
            raise CheckFailure(f"external references in the emitted HTML: {findings}")

        # The decimation claim: the measured error is a direct measurement
        # over every dropped truth sample, and must sit within the bound.
        if result.measured_max_error_m > result.bound_m:
            raise CheckFailure(
                f"measured decimation error {result.measured_max_error_m!r} m "
                f"exceeds the bound {result.bound_m!r} m"
            )

        run = load(srlog)
        data = extract_view_data(html)
        if data["epoch"]["utc_first"] != run.header["epoch_utc"]:
            raise CheckFailure(
                f"embedded first epoch {data['epoch']['utc_first']!r} != header "
                f"epoch_utc {run.header['epoch_utc']!r}"
            )
        epoch = datetime.fromisoformat(
            run.header["epoch_utc"].replace("Z", "+00:00")
        ).astimezone(timezone.utc)
        t_last = float(run.groups["truth"]["t_s"][-1])
        last = epoch + timedelta(seconds=t_last)
        expected_last = last.strftime("%Y-%m-%dT%H:%M:%S")
        if last.microsecond:
            expected_last += ("." + f"{last.microsecond:06d}").rstrip("0")
        expected_last += "Z"
        if data["epoch"]["utc_last"] != expected_last:
            raise CheckFailure(
                f"embedded last epoch {data['epoch']['utc_last']!r} != header-"
                f"derived {expected_last!r} (epoch_utc + final truth t_s)"
            )

        # Byte-identical regeneration: the viewer is a derived artifact and
        # must be a pure function of the log bytes (FR-21 discipline).
        generate_view(srlog, tdp / "view2.html")
        if (tdp / "view.html").read_bytes() != (tdp / "view2.html").read_bytes():
            raise CheckFailure("regenerating the viewer produced different bytes")


# --------------------------------------------------------------------------
# Phase 5 check (V021): the FR-18 plot pipeline, as a quick variant of the
# golden suite in tests/python/test_plot_golden.py. Pure Python on a
# synthesized log (no compiled core, no source tree): the element
# preparation is checked against the closed-form elements of a circular
# orbit, then one PNG is rendered headless through the forced Agg backend.
# --------------------------------------------------------------------------

# Circular-orbit design for V021: r0 = 7,000 km on +X, v = sqrt(mu/r0) on
# +Y, mu per IERS Conventions (2010), TN No. 36, Table 1.1 (the loader GM,
# star_reacher.derived.GM_M3_PER_S2["earth"]). For this geometry the
# osculating elements are closed-form: a = r0 (to the rounding of v0^2),
# e ~ 0, i = 0 exactly (h along +Z), and with the derived-elements
# circular-equatorial convention RAAN = argp = 0 exactly and the true-
# longitude slot advances from 0.
_V021_MU = 3.986004418e14
_V021_R0_M = 7.0e6


def _check_v021(ctx: dict) -> None:
    from star_reacher.plotting import PLOT_NAMES, prep_elements, render_plots

    header = _fixtures.contract_header()
    v0 = math.sqrt(_V021_MU / _V021_R0_M)
    n_samples = 32
    records = []
    for k in range(n_samples):
        # Uniform sweep of true longitude; sin/cos parameterization keeps
        # every sample exactly on the circular orbit.
        ang = 2.0 * math.pi * k / n_samples * 0.9
        records.append(
            _fixtures.truth_record(
                60.0 * k,
                r_m=(_V021_R0_M * math.cos(ang), _V021_R0_M * math.sin(ang), 0.0),
                v_mps=(-v0 * math.sin(ang), v0 * math.cos(ang), 0.0),
                mass_kg=150.0,
            )
        )
    records.append(_fixtures.event_record(0.0, 1, "run_start"))
    records.append(_fixtures.event_record(60.0 * (n_samples - 1), 2, "run_end"))
    data = _fixtures.build_srlog(header, records)
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        run = load(_write_temp_srlog(tdp, "v021.srlog", data))

        prep = prep_elements(run)
        if prep.arrays is None:
            raise CheckFailure(f"element preparation produced no arrays: {prep.note}")
        a_km = prep.arrays["a_km"]
        e = prep.arrays["e"]
        i_deg = prep.arrays["i_deg"]
        raan_deg = prep.arrays["raan_deg"]
        if len(a_km) != n_samples:
            raise CheckFailure(f"expected {n_samples} element samples, got {len(a_km)}")
        # v0 = sqrt(mu/r0) rounds once in binary64, so a and e sit within a
        # few ulp of the closed form; 1e-6 relative (a) and 1e-9 absolute
        # (e) give ~9 orders of margin while failing on any element-chain
        # defect (wrong GM, wrong units, wrong convention).
        worst_a = float(np.max(np.abs(a_km * 1000.0 - _V021_R0_M)))
        if worst_a > 1e-6 * _V021_R0_M:
            raise CheckFailure(
                f"circular-orbit semi-major axis off by {worst_a:.3e} m "
                f"(gate {1e-6 * _V021_R0_M:.1e} m)"
            )
        if float(np.max(e)) > 1e-9:
            raise CheckFailure(f"circular-orbit eccentricity {float(np.max(e)):.3e} > 1e-9")
        # Equatorial geometry: i and the RAAN convention value are EXACT
        # zeros per docs/formats/derived_elements.md section 4.
        if float(np.max(np.abs(i_deg))) != 0.0 or float(np.max(np.abs(raan_deg))) != 0.0:
            raise CheckFailure(
                f"equatorial convention violated: max|i| = "
                f"{float(np.max(np.abs(i_deg)))!r} deg, max|RAAN| = "
                f"{float(np.max(np.abs(raan_deg)))!r} deg (both must be exactly 0)"
            )

        if "elements" not in PLOT_NAMES:
            raise CheckFailure(f"'elements' missing from PLOT_NAMES: {PLOT_NAMES}")
        report = render_plots([run], tdp / "plots", plots=["elements"])
        png = tdp / "plots" / "elements.png"
        if [Path(p) for p in report.written] != [png]:
            raise CheckFailure(f"expected [{png}], wrote {report.written}")
        head = png.read_bytes()
        if head[:8] != b"\x89PNG\r\n\x1a\n":
            raise CheckFailure(f"{png.name} does not start with the PNG signature")
        if len(head) < 1024:
            raise CheckFailure(f"{png.name} is implausibly small ({len(head)} bytes)")
        # Deterministic rendering: a second render of the same log must be
        # byte-identical (fixed figure geometry, fixed metadata, no
        # timestamps) - the FR-21 discipline applied to a derived artifact.
        render_plots([run], tdp / "plots2", plots=["elements"])
        if (tdp / "plots2" / "elements.png").read_bytes() != head:
            raise CheckFailure("re-rendering the same log produced different PNG bytes")


# --------------------------------------------------------------------------
# Phase 6 checks (V022-V024): the exit-criterion battery.
#
# Only criteria whose gates were shown able to fail are wired here. The Phase
# 6 evidence audit (docs/audit/phase6_evidence_audit.md) found several of the
# phase's gates unsound - criterion 2's golden scenario leaves three of its
# five equations unexercised, criterion 3's driver gates on a rule its own
# consistency module documents as a coin flip, criterion 9's tolerance sits
# about seven orders above the residual it measures, and criterion 7's
# intrinsics clause has no channel to gate at all. Wiring those into the
# acceptance suite would launder a known-weak check into a green line, so
# they are deliberately absent and are listed as gaps in the Phase 6
# roadmap entry instead. What lands here is criteria 4 and 5, which the
# audit found solid, plus the version coherence that the 0.6.0 bump makes
# checkable from a bare wheel.
#
# The wheel carries no missions/ or vehicles/ tree, so the GNC fixture is
# synthesized in a temp directory like every other fixture in this module.
# Vehicle references resolve against the process working directory, so the
# checks below chdir into that temp directory and restore unconditionally.

_P6_VEHICLE = """\
schema_version = 1
provenance = "representative"

[vehicle]
name = "verify-bus-150"
description = "150 kg smallsat bus, inline fixture for the acceptance suite"

[[stage]]
name = "bus"
dry_mass_kg = 150.0
dry_cg_m = [0.3, 0.0, 0.0]
dry_inertia_kgm2 = [[9.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 11.0]]

[[stage.sensor]]
name = "imu0"
preset = "presets/imu_tactical.toml"
position_m = [0.3, 0.0, 0.0]
axis = [1.0, 0.0, 0.0]
"""

# The closed-loop attitude-acquisition scenario of missions/leo_attitude_gnc
# .toml, shortened to 20 s: q0 is the exact initial attitude the Phase 4
# rule assigns to this state ([0, sqrt(1/2), sqrt(1/2), 0]) and q_cmd is that
# attitude rotated 10 degrees about body +Z, so the run opens with a pure
# 10-degree tracking error and commands a non-trivial torque from cycle one.
_P6_GNC_MISSION = """\
schema_version = 1
vehicle = "verify_vehicle.toml"

[mission]
name = "verify-p6-gnc"
epoch_utc = "2026-01-01T00:00:00Z"
duration_s = 20.0

[run]
seed = 20260601

[integrator]
type = "rk4"
dt_s = 0.1

[environment]
central_body = "earth"

[logging]
truth_rate_hz = 10

[initial_state.cartesian]
r_m = [7.0e6, 0.0, 0.0]
v_mps = [0.0, 7546.0, 0.0]
frame = "GCRF"

[gnc]
control_rate_hz = 10
latency_cycles = 0
{oracle}
[gnc.nav]
component = "dead_reckoning"
q0 = [0.0, 0.7071067811865476, 0.7071067811865476, 0.0]

[gnc.guidance]
component = "attitude_hold"
q_cmd = [0.0, 0.7660444431189781, 0.6427876096865393, 0.0]

[gnc.control]
component = "pd_attitude"
kp_nm_per_rad = [0.4, 0.4, 0.4]
kd_nm_per_radps = [3.6, 3.6, 3.6]
tau_max_nm = [0.05, 0.05, 0.05]

[sensors.imu]
sample_rate_hz = 10
"""


def _p6_fixture(tdp: Path, oracle: bool = False) -> Path:
    """Write the inline vehicle + GNC mission and return the mission path."""
    (tdp / "verify_vehicle.toml").write_text(_P6_VEHICLE, encoding="utf-8")
    name = "gnc_oracle.toml" if oracle else "gnc.toml"
    mission = tdp / name
    mission.write_text(
        _P6_GNC_MISSION.format(oracle="oracle = true\n" if oracle else ""),
        encoding="utf-8",
    )
    return mission


def _check_v022(ctx: dict) -> None:
    """Phase 6 exit criterion 4: stepping and batch agree, observe() is pure."""
    import os

    from star_reacher.sim import Sim

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        mission = _p6_fixture(tdp)
        cwd = os.getcwd()
        sim = None
        os.chdir(tdp)
        try:
            batch = run_mission(mission, tdp / "batch")
            sim = Sim(str(mission), str(tdp / "stepped"))
            sim.reset()
            # observe() must be idempotent: no component runs, no random draw,
            # no sensor sample is consumed. Checked at several depths so a
            # first-cycle special case cannot hide.
            checkpoints = {0, 1, 5, 17}
            steps = 0
            while not sim.done():
                if steps in checkpoints:
                    first = sim.observe()
                    if sim.observe() != first or sim.observe() != first:
                        raise CheckFailure(
                            f"observe() differed across calls without step() "
                            f"at cycle {steps}"
                        )
                sim.step()
                steps += 1
            summary = sim.summary()
        finally:
            # Drop the Sim before the temp tree is removed. A run abandoned
            # part-way still holds its log open, and on Windows that turns
            # any failure here into a PermissionError from the directory
            # cleanup, hiding the evidence this check exists to report.
            sim = None
            os.chdir(cwd)

        if steps <= 0:
            raise CheckFailure("the stepped run advanced zero cycles")
        stepped_sha = hashlib.sha256(
            (tdp / "stepped" / "run.srlog").read_bytes()
        ).hexdigest()
        if stepped_sha != batch.srlog_sha256:
            raise CheckFailure(
                f"stepped and batch logs differ: {stepped_sha} != "
                f"{batch.srlog_sha256} over {steps} cycles"
            )
        if summary["steps"] != batch.summary["steps"]:
            raise CheckFailure(
                f"step tally {summary['steps']} != batch "
                f"{batch.summary['steps']}"
            )


def _check_v023(ctx: dict) -> None:
    """Phase 6 exit criterion 5: major unchanged by v1.2 channels; oracle
    identifiable from the log header alone."""
    import os

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        plain = _p6_fixture(tdp, oracle=False)
        oracle = _p6_fixture(tdp, oracle=True)
        cwd = os.getcwd()
        os.chdir(tdp)
        try:
            a = run_mission(plain, tdp / "plain")
            b = run_mission(oracle, tdp / "oracle")
        finally:
            os.chdir(cwd)

        run = load(a.srlog_path)
        # The new channels are the point of the criterion: if they are absent
        # the major-version claim is trivially true and proves nothing.
        added = ("gnc.cmd", "nav.est", "sensors.imu")
        missing = [g for g in added if g not in run.groups]
        if missing:
            raise CheckFailure(
                f"the Phase 6 channels are absent, so the schema claim is "
                f"vacuous: missing {missing}"
            )
        fmt = run.header["format"]
        if fmt != {"name": "SRLOG", "major": 1, "minor": 2}:
            raise CheckFailure(
                f"expected SRLOG v1.2 with major unchanged at 1, got {fmt}"
            )
        # Identifiable from the HEADER ALONE: read the flag off both headers
        # without touching a single record.
        if run.header.get("oracle") is not False:
            raise CheckFailure(
                f"a non-oracle run reports oracle="
                f"{run.header.get('oracle')!r} in its header"
            )
        if load(b.srlog_path).header.get("oracle") is not True:
            raise CheckFailure(
                "an oracle run is not identifiable from its header alone"
            )


def _check_v024(ctx: dict) -> None:
    """The package and the compiled core report the same version.

    These are separate sources - pyproject.toml and the CMake project()
    VERSION - kept in sync by hand, and every SRLOG header stamps the core's
    value into producer.core_version. When they drifted at the Phase 6 close
    a v1.2 log self-reported a 0.5.0 producer.
    """
    import star_reacher

    core = import_core()
    pkg_version = star_reacher.__version__
    core_version = core.core_version()
    if pkg_version != core_version:
        raise CheckFailure(
            f"package __version__ {pkg_version!r} != compiled "
            f"core_version() {core_version!r}"
        )
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        mission = tdp / "v024.toml"
        mission.write_text(_V001_MISSION, encoding="utf-8")
        result = run_mission(mission, tdp / "run")
        stamped = load(result.srlog_path).header["producer"]["core_version"]
    if stamped != core_version:
        raise CheckFailure(
            f"log header stamped producer.core_version {stamped!r}, core "
            f"reports {core_version!r}"
        )


_CHECKS = [
    ("V001", "two-body double-run SHA-256 bit-identity", _check_v001),
    ("V002", "minor-version-forward read (v1.999 file with one added channel)", _check_v002),
    ("V003", "major-version mismatch rejected (v2.0 file)", _check_v003),
    ("V004", "corrupted header rejected (bad magic; truncated JSON)", _check_v004),
    ("V005", "CSV export round-trips every value bit-exactly", _check_v005),
    ("V006", "RNG stream reproducibility", _check_v006),
    ("V007", "load() smoke: shapes, dtypes, monotonic t_s, header fields", _check_v007),
    ("V008", "truncated trailing record rejected", _check_v008),
    ("V009", "UTC->TAI->TT golden epochs bit-exact with round trip", _check_v009),
    ("V010", "quat<->DCM<->Euler round trips over 100 seeded attitudes", _check_v010),
    ("V011", "GCRF->ITRF at golden epoch vs ERFA elements + orthonormality", _check_v011),
    ("V012", "SREPH loader + Chebyshev evaluator on synthesized file", _check_v012),
    ("V013", "two-body invariant drift and apsis events (quick tier)", _check_v013),
    ("V014", "gravity tiers vs closed-form J2 on a synthesized field", _check_v014),
    ("V015", "Battin third body vs extended-precision references", _check_v015),
    ("V016", "conical shadow exact 0/1, penumbra value, umbra SRP zero", _check_v016),
    ("V017", "USSA76/Harris-Priester/Mars density spot values", _check_v017),
    ("V018", "perturbed-run double-run SHA-256 bit-identity (rkf78 + rk4)", _check_v018),
    ("V019", "NPZ export round-trips bit-exactly (+ Parquet when available)", _check_v019),
    ("V020", "viewer HTML self-contained, epochs exact, decimation bound held", _check_v020),
    ("V021", "plot arrays match closed-form elements; headless PNG render", _check_v021),
    ("V022", "P6 EC-4: stepped and batch log hashes identical; observe() pure", _check_v022),
    ("V023", "P6 EC-5: v1.2 channels leave major at 1; oracle read from header", _check_v023),
    ("V024", "package, compiled core, and log header report one version", _check_v024),
]


def run_checks(quick: bool = False) -> int:
    """Run every acceptance check, print one line each, return the exit code.

    ``quick`` is accepted for CLI symmetry: through Phase 3 the quick tier
    and the full tier run the identical check set (every check is budgeted
    for the < 60 s quick gate); the split becomes meaningful when later
    phases add long-running checks.
    """
    ctx: dict = {}
    results: list[tuple[str, bool]] = []
    for check_id, title, fn in _CHECKS:
        try:
            fn(ctx)
        except CheckFailure as exc:
            ok, note = False, str(exc)
        except Exception as exc:  # noqa: BLE001 - a crash is a failure, not an abort
            ok, note = False, f"{type(exc).__name__}: {exc}"
        else:
            ok, note = True, ""
        line = f"{check_id} {'PASS' if ok else 'FAIL'} {title}"
        if not ok:
            line += f": {note}"
        print(line, flush=True)
        results.append((check_id, ok))
    total = len(results)
    passed = sum(1 for _, ok in results if ok)
    if passed == total:
        print(f"VERIFY: PASS ({passed}/{total})")
        return 0
    failing = ", ".join(check_id for check_id, ok in results if not ok)
    print(f"VERIFY: FAIL ({passed}/{total}) failing: {failing}")
    return 1
