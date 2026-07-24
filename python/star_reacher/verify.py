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
golden suite in ``tests/python/test_plot_golden.py``. V022-V027 cover the
Phase 6 exit-criterion battery: the stepped/batch agreement of criterion 4,
the v1.2 schema and oracle header of criterion 5, version coherence, the PD
reimplementation contract of criterion 2, the aberration recomputation of
criterion 9, and the EKF ensemble consistency of criterion 3. V028-V029 cover
the Phase 7 Monte Carlo layer: the seeded-sweep reproducibility of exit
criterion 1 (a manifest entry re-executed through the star run API reproduces
its logged hash) and the ensemble-statistics regression of exit criterion 2
(chi-square and Anderson-Darling 99 % gates against a frozen golden), both on
temp-directory fixtures with V029's golden inlined for the bare wheel.

TIERS. Every registered check runs in both tiers, criterion 3's ensemble
(V027) included, at the criterion's own R = 100. Criterion 3 was once split
into a full-strength and a reduced-strength variant on the premise that the
100-run ensemble was too costly for ``--quick``; measured, it costs 7.4 s
against a 60 s budget, so the premise was wrong by an order of magnitude and
the reduced variant only bought a check whose green meant less than its name
suggested. One gate, one meaning. The tier machinery below is retained
because a genuinely expensive check will eventually need it, and whichever
tier runs, the runner announces on its first line whether any registered
check is being left out: a quick tier that covered part of a criterion
without saying so is how the criterion-4 blind spot survived a whole phase.
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
# Phase 6 checks (V022-V024): the exit-criterion battery, first tranche.
#
# Only criteria whose gates were shown able to fail are wired here. The Phase
# 6 evidence audit (docs/audit/phase6_evidence_audit.md) found several of the
# phase's gates unsound, and wiring one of those into the acceptance suite
# would launder a known-weak check into a green line. What lands in this
# tranche is criteria 4 and 5, which the audit found solid, plus the version
# coherence that the 0.6.0 bump makes checkable from a bare wheel. Criteria
# 2, 3 and 9 were remediated afterwards and are wired below as V025-V027,
# each demonstrated able to fail under mutation before it was registered.
# Criterion 7's intrinsics clause still has no channel to gate at all and
# remains listed as a gap in the Phase 6 roadmap entry.
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
# .toml, shortened to 20 s, flying the built-in error-state EKF against FOUR
# aiding sensors. q_cmd is the vehicle's exact initial attitude
# ([0, sqrt(1/2), sqrt(1/2), 0]) rotated 10 degrees about body +Z, so the run
# opens with a pure 10-degree tracking error and commands a non-trivial
# torque from cycle one.
#
# MULTI-SENSOR ON PURPOSE. This fixture was IMU-only through the phase, and
# that is exactly why criterion 4 could not see the defect it exists to
# catch: with one sensor the canonical FR-23 order and the alphabetical order
# a sort_keys round trip produces are the same list, so the batch and stepped
# paths agreed by coincidence rather than by construction. The four kinds
# below order canonically as (imu, startracker, navfix, altimeter) and
# alphabetically as (altimeter, imu, navfix, startracker) - two different
# lists - so any regression that lets the configured order follow the input
# dict changes the log header's declared sensor array, the sensor_id every
# nav.innov record is labelled with, and (through the EKF's sequential
# aiding updates, whose order cpp/src/gnc/ekf.cpp declares normative) the
# state trajectory itself. All three land in the compared bytes.
#
# The estimator is the error-state EKF rather than dead reckoning so the
# trajectory consequence is exercised and not only the labelling one: dead
# reckoning folds no measurements, so reordering its sensors would move the
# header bytes while leaving the numbers alone. The filter's initial belief
# is the true initial state perturbed by a draw from P0, carried over from
# missions/leo_ekf_consistency.toml, which keeps the run well-conditioned;
# criterion 4 gates byte equality, not statistical consistency, so nothing
# here is a consistency claim.
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
component = "error_state_ekf"
q0 = [-0.00044626342559570399, 0.70688223269726402, 0.70733106309014382, -0.00027772946007248347]
v0_mps = [0.060921416592906778, 7545.8006329551154, -0.41793559576109252]
p0_m = [6999986.8998344773, -13.811525573585669, -86.675748333963583]
bg0_radps = [0.0, 0.0, 0.0]
ba0_mps2 = [0.0, 0.0, 0.0]
p0_sigma_att_rad = [1.0e-3, 1.0e-3, 1.0e-3]
p0_sigma_vel_mps = [0.5, 0.5, 0.5]
p0_sigma_pos_m = [50.0, 50.0, 50.0]
p0_sigma_bg_radps = [1.0759973046695306e-7, 1.0759973046695306e-7, 1.0759973046695306e-7]
p0_sigma_ba_mps2 = [1.0759973046695306e-5, 1.0759973046695306e-5, 1.0759973046695306e-5]

[gnc.guidance]
component = "attitude_hold"
q_cmd = [0.0, 0.7660444431189781, 0.6427876096865393, 0.0]

[gnc.control]
component = "pd_attitude"
kp_nm_per_rad = [0.4, 0.4, 0.4]
kd_nm_per_radps = [3.6, 3.6, 3.6]
tau_max_nm = [0.05, 0.05, 0.05]

# Written in a deliberately NON-canonical and non-alphabetical TOML order, so
# neither the canonical list nor the sort_keys list can be reproduced by
# accidentally preserving the order the file was parsed in.
[sensors.altimeter]
sample_rate_hz = 1
sigma_noise_m = 20.0
sigma_bias_m = 0.0
h_min_m = 0.0
h_max_m = 0.0

[sensors.imu]
sample_rate_hz = 10
gyro_arw_rad_per_sqrt_s = 1.0e-5
gyro_bias_instability_radps = 1.0e-7
gyro_bias_tau_s = 100.0
accel_vrw_mps_per_sqrt_s = 1.0e-4
accel_bias_instability_mps2 = 1.0e-5
accel_bias_tau_s = 100.0

[sensors.startracker]
sample_rate_hz = 1
boresight_b = [0.0, 0.0, 1.0]
sigma_rad = [1.0e-5, 1.0e-5, 5.0e-5]

[sensors.navfix]
sample_rate_hz = 1
sigma_r_m = [10.0, 10.0, 10.0]
sigma_v_mps = [0.1, 0.1, 0.1]
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

        # The fixture must be able to SEE an ordering divergence before its
        # agreement means anything. A single-sensor mission orders the same
        # canonically and alphabetically, so the two entry points would agree
        # no matter how either built its sensor list - the degeneracy that
        # kept this criterion green while `star run` and `Sim` genuinely
        # disagreed on missions/leo_ekf_consistency.toml. Asserting the
        # non-degeneracy here means a future edit that shrinks the fixture
        # fails loudly instead of quietly restoring the blind spot.
        declared = load(batch.srlog_path).header.get("gnc", {}).get("sensors", [])
        if len(declared) < 2:
            raise CheckFailure(
                f"the fixture declares {len(declared)} sensor(s), so the "
                f"canonical and alphabetical orders coincide and this check "
                f"cannot observe a sensor-ordering divergence: {declared}"
            )
        if declared == sorted(declared):
            raise CheckFailure(
                f"the fixture's canonical sensor order {declared} is already "
                f"alphabetical, so a builder that inherited its input's order "
                f"would still agree and this check cannot observe it"
            )

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
        if fmt != {"name": "SRLOG", "major": 1, "minor": 3}:
            raise CheckFailure(
                f"expected SRLOG v1.3 with major unchanged at 1, got {fmt}"
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


# --------------------------------------------------------------------------
# Phase 6 checks (V025-V027): the three remediated exit criteria.
#
# Criteria 2, 3 and 9 were each found by the Phase 6 evidence audit
# (docs/audit/phase6_evidence_audit.md) to have a gate that passed while
# proving little, and each was remediated in the pytest suite before being
# wired here. What the audit found, and what closed it:
#
# * criterion 2's golden scenario was degenerate on three of the PD law's
#   five equations, so a reference that dropped any of them still reproduced
#   the logged torques exactly. The scenario below is the non-degenerate one
#   of tests/python/test_gnc_missions.py, and V025 re-asserts the
#   non-degeneracy alongside the residual rather than trusting it;
# * criterion 3's driver added an ``inside >= 0.95`` coverage rule that
#   star_reacher.consistency documents as invalid - under the consistency
#   hypothesis the count inside a two-sided 95 % interval is
#   Binomial(T, 0.95), so the rule tests the count against its own mean. It
#   now routes through consistency.ensemble_gate, the FR-26 instrument;
# * criterion 9's gate sat about seven orders above the residual it measures
#   and its fixture held q_w == 0 throughout, under which the DCM of
#   eq:notation:quat2dcm is exactly symmetric and a transposed attitude
#   convention is invisible to an angular separation. V026 gates at 1e-5 mas
#   on an off-axis fixture and asserts the asymmetry it needs.
#
# These are the pytest gates re-expressed for a bare wheel: no missions/,
# vehicles/, tests/refs/ or tests/golden/ tree is available here, so the
# fixtures are synthesized in a temp directory, the reference implementations
# are written out inline from the same chapter equations, and the reference
# values are measured quantities carried in the docstrings with their
# provenance. Vehicle and ephemeris references resolve against the process
# working directory, so each check chdirs into its temp directory and
# restores unconditionally.
# --------------------------------------------------------------------------


def _p6_quat_mul(p, q):
    """Hamilton product, scalar-first (D-7), matching star::rotation."""
    pw, pv = p[0], np.asarray(p[1:], dtype=np.float64)
    qw, qv = q[0], np.asarray(q[1:], dtype=np.float64)
    out = np.empty(4)
    out[0] = pw * qw - float(np.dot(pv, qv))
    out[1:] = pw * qv + qw * pv + np.cross(pv, qv)
    return out


def _p6_quat_exp(phi):
    """Exact exponential map of a rotation vector to a unit quaternion.

    ``q = [cos(|phi|/2), sin(|phi|/2) phi/|phi|]``. The zero-rotation limit
    returns the identity rather than dividing by zero.
    """
    angle = float(np.linalg.norm(phi))
    if angle == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    out = np.empty(4)
    out[0] = math.cos(0.5 * angle)
    out[1:] = math.sin(0.5 * angle) * np.asarray(phi, dtype=np.float64) / angle
    return out


def _p6_quat_to_dcm(q):
    """Frame-transformation DCM C_I2B of ``eq:notation:quat2dcm``.

    Built from the outer-product-plus-skew form, which is the TRANSPOSE of
    the active rotation matrix Eigen returns for the same quaternion - the
    convention trap this reference exists to avoid falling into. Used by V026
    in place of the core's own quat_to_dcm: an angular separation is
    invariant under a rotation applied to both of its arguments, so taking
    the DCM from the code under test would cancel exactly and no attitude
    convention error could ever be detected.
    """
    qw, qv = q[0], np.asarray(q[1:], dtype=np.float64)
    skew = np.array(
        [
            [0.0, -qv[2], qv[1]],
            [qv[2], 0.0, -qv[0]],
            [-qv[1], qv[0], 0.0],
        ]
    )
    return (
        (qw * qw - float(np.dot(qv, qv))) * np.eye(3)
        + 2.0 * np.outer(qv, qv)
        - 2.0 * qw * skew
    )


# --------------------------------------------------------------------------
# V025: Phase 6 exit criterion 2.
# --------------------------------------------------------------------------

# The PD attitude law of Chapter ch:gnc-builtin, restated as the
# cross-workstream contract in cpp/include/star/gnc/builtin.hpp:
#
#     dq    = q_cmd^* (x) q_est                       (eq:gnc:deltaq)
#     s     = (dq_0 >= 0) ? +1 : -1                   (eq:gnc:sign)
#     w_err = w_est - C(dq) w_cmd                     (eq:gnc:werr)
#     tau_i = -kp_i s dq_vec_i - kd_i w_err_i         (eq:gnc:pd)
#     tau_i = clamp(tau_i, -tau_max_i, +tau_max_i)    (eq:gnc:sat)
#
# with NO renormalization of dq - inputs are used as received. This is the
# same law tests/refs/pd_attitude.py carries for the pytest suite, written
# out again here because a wheel install has no tests/ tree; the arithmetic
# is written from the equations, not transcribed from the C++ function body.


def _p6_error_quaternion(q_cmd, q_est):
    """``dq = q_cmd^* (x) q_est`` (eq:gnc:deltaq), on (N, 4) stacks."""
    cmd = np.atleast_2d(np.asarray(q_cmd, dtype=np.float64))
    est = np.atleast_2d(np.asarray(q_est, dtype=np.float64))
    pw, px, py, pz = cmd[:, 0], -cmd[:, 1], -cmd[:, 2], -cmd[:, 3]
    qw, qx, qy, qz = est[:, 0], est[:, 1], est[:, 2], est[:, 3]
    dq = np.empty((cmd.shape[0], 4))
    dq[:, 0] = pw * qw - px * qx - py * qy - pz * qz
    dq[:, 1] = pw * qx + px * qw + py * qz - pz * qy
    dq[:, 2] = pw * qy - px * qz + py * qw + pz * qx
    dq[:, 3] = pw * qz + px * qy - py * qx + pz * qw
    return dq


def _p6_error_dcm(dq):
    """``C(dq)`` per eq:notation:quat2dcm, element by element on a stack.

    Written out componentwise rather than through _p6_quat_to_dcm's outer
    product so the two constructions reach the same matrix by visibly
    different arithmetic, as the pytest references do.
    """
    w, x, y, z = dq[:, 0], dq[:, 1], dq[:, 2], dq[:, 3]
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    c = np.empty((dq.shape[0], 3, 3))
    c[:, 0, 0] = ww + xx - yy - zz
    c[:, 0, 1] = 2.0 * (x * y + w * z)
    c[:, 0, 2] = 2.0 * (x * z - w * y)
    c[:, 1, 0] = 2.0 * (x * y - w * z)
    c[:, 1, 1] = ww - xx + yy - zz
    c[:, 1, 2] = 2.0 * (y * z + w * x)
    c[:, 2, 0] = 2.0 * (x * z + w * y)
    c[:, 2, 1] = 2.0 * (y * z - w * x)
    c[:, 2, 2] = ww - xx - yy + zz
    return c


def _p6_pd_torque(q_cmd, q_est, w_cmd, w_est, kp, kd, tau_max):
    """Commanded body torque, eq:gnc:deltaq through eq:gnc:sat.

    ``tau_max=None`` returns the UNSATURATED torque of eq:gnc:pd alone, which
    is how V025 counts the cycles on which the clamp of eq:gnc:sat actually
    caught: a gate that never separates the two is not testing the clamp.
    """
    dq = _p6_error_quaternion(q_cmd, q_est)
    s = np.where(dq[:, 0] >= 0.0, 1.0, -1.0)  # sign(0) = +1
    w_cmd_b = np.einsum(
        "kij,kj->ki", _p6_error_dcm(dq), np.atleast_2d(np.asarray(w_cmd, np.float64))
    )
    rate = np.atleast_2d(np.asarray(w_est, dtype=np.float64))
    tau = -np.asarray(kp) * s[:, None] * dq[:, 1:] - np.asarray(kd) * (rate - w_cmd_b)
    if tau_max is not None:
        tau = np.clip(tau, -np.asarray(tau_max), np.asarray(tau_max))
    return tau


# The reference attitude mission of missions/leo_attitude_gnc.toml with the
# guidance slot driven through the FR-24 external seam, so the commanded
# attitude and rate are the driver's to choose while the compiled built-in
# pd_attitude component still computes every torque. Dead reckoning rather
# than the EKF keeps nav.est at the 7-wide (quaternion, rate) layout the
# controller's inputs are read from.
_V025_MISSION = """\
schema_version = 1
vehicle = "verify_vehicle.toml"

[mission]
name = "verify-p6-pd"
epoch_utc = "2026-01-01T00:00:00Z"
duration_s = 60.0

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

[gnc.nav]
component = "dead_reckoning"
q0 = [0.0, 0.7071067811865476, 0.7071067811865476, 0.0]

[gnc.guidance]
component = "external"

[gnc.control]
component = "pd_attitude"
kp_nm_per_rad = [0.4, 0.4, 0.4]
kd_nm_per_radps = [3.6, 3.6, 3.6]
tau_max_nm = [0.05, 0.05, 0.05]

[sensors.imu]
sample_rate_hz = 10
"""

_V025_KP = np.array([0.4, 0.4, 0.4])
_V025_KD = np.array([3.6, 3.6, 3.6])
_V025_TAU_MAX = np.array([0.05, 0.05, 0.05])
_V025_TOL_NM = 1e-9  # the exit criterion's own figure

# The scenario's shape, and why each element is there. The committed
# reference mission holds a fixed attitude, and on it three of the five
# equations are multiplied by zero: attitude_hold commands zero body rate so
# eq:gnc:werr has nothing to rotate, the tracking error stays well inside a
# half turn so eq:gnc:sign never takes its short path, and the 10-degree
# transient never reaches tau_max so eq:gnc:sat never clamps. Each element
# below gives one of them a measurable effect.
_V025_CYCLE_S = 0.1
# A 60-degree opening offset against 0.4 N*m/rad gains and 0.05 N*m of
# authority saturates every axis through the transient (eq:gnc:sat).
_V025_OFFSET_AXIS = np.array([1.0, 2.0, -2.0]) / 3.0
_V025_OFFSET_RAD = math.radians(60.0)
# A commanded rate about a DIFFERENT axis than the offset, so the error
# quaternion is not parallel to it: parallel axes leave C(dq) w_cmd == w_cmd
# and make both the rotation of eq:gnc:werr and its transpose invisible.
_V025_RATE_AXIS = np.array([2.0, -1.0, -2.0]) / 3.0
_V025_RATE_RADPS = 0.02
# The commanded quaternion is expressed ANTIPODALLY before this cycle. The
# attitude is physically identical either way, so a controller honouring
# eq:gnc:sign produces the same torque from both representations while one
# omitting the branch reverses it. Placed mid-run so both branches cover a
# comparable number of cycles, with the negative branch holding the
# saturated transient.
_V025_SIGN_FLIP_CYCLE = 300
_V025_Q_START = np.array([0.0, 0.7071067811865476, 0.7071067811865476, 0.0])


def _v025_command(cycle: int):
    """Commanded attitude and body rate at one control cycle.

    A pure function of the cycle index, so the run is reproducible without
    the driver reading any observation back out of the simulation.
    """
    q = _p6_quat_mul(
        _V025_Q_START, _p6_quat_exp(_V025_OFFSET_RAD * _V025_OFFSET_AXIS)
    )
    q = _p6_quat_mul(
        q,
        _p6_quat_exp(
            _V025_RATE_RADPS * cycle * _V025_CYCLE_S * _V025_RATE_AXIS
        ),
    )
    if cycle < _V025_SIGN_FLIP_CYCLE:
        q = -q
    return q, _V025_RATE_RADPS * _V025_RATE_AXIS


def _check_v025(ctx: dict) -> None:
    """Phase 6 exit criterion 2: the Python PD law reproduces the compiled
    controller's commanded torques to < 1e-9 N*m on a non-degenerate scenario.

    latency_cycles is 0, so the applied command logged in gnc.cmd IS the
    cycle's chain output and the comparison needs no shift. Measured worst
    residual on this fixture: 1.39e-17 N*m, about eight orders inside the
    criterion's gate.

    The four reference mutations the audit used to establish that this
    scenario is no longer degenerate, each measured against the same logged
    torques: dropping the eq:gnc:sign branch moves the residual to
    1.0000e-01 N*m, dropping the C(dq) rotation of eq:gnc:werr to
    4.7340e-02 N*m, transposing that C(dq) to 5.1548e-02 N*m, and dropping
    the eq:gnc:sat clamp to 1.5264e-01 N*m. On the previous attitude-hold
    fixture all four moved it by exactly 0.0 N*m.

    The non-degeneracy assertions below are part of the gate, not commentary:
    without them a future edit that shrank the scenario would restore the
    blind spot silently, which is how this criterion stayed green for a whole
    phase while unable to see the defects it names.
    """
    import os

    from star_reacher.sim import Sim

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        (tdp / "verify_vehicle.toml").write_text(_P6_VEHICLE, encoding="utf-8")
        mission = tdp / "pd.toml"
        mission.write_text(_V025_MISSION, encoding="utf-8")
        cwd = os.getcwd()
        sim = None
        os.chdir(tdp)
        try:
            sim = Sim(str(mission), str(tdp / "run"))
            sim.reset()
            cycle = 0
            while not sim.done():
                q_cmd, w_cmd = _v025_command(cycle)
                sim.step(
                    {
                        "q_i2b": list(q_cmd),
                        "omega_b_radps": list(w_cmd),
                        "valid": True,
                    }
                )
                cycle += 1
        finally:
            # Drop the Sim before the temp tree is removed: an abandoned run
            # still holds its log open, and on Windows that turns any failure
            # here into a PermissionError from the cleanup, hiding the
            # evidence this check exists to report.
            sim = None
            os.chdir(cwd)
        run = load(tdp / "run" / "run.srlog")
        est = run.groups["nav.est"]["x_hat"]
        cmd = run.groups["gnc.cmd"]
        q_est = est[:, :4]
        w_est = est[:, 4:]
        q_cmd_log = cmd["q_cmd_i2b"]
        w_cmd_log = cmd["w_cmd_b_radps"]
        logged_tau = cmd["tau_b_nm"]

    if len(logged_tau) < 2:
        raise CheckFailure(f"the scenario logged {len(logged_tau)} command(s)")

    tau = _p6_pd_torque(
        q_cmd_log, q_est, w_cmd_log, w_est, _V025_KP, _V025_KD, _V025_TAU_MAX
    )
    worst = float(np.max(np.abs(tau - logged_tau)))
    if worst >= _V025_TOL_NM:
        raise CheckFailure(
            f"worst Python-versus-core commanded-torque residual {worst:.6e} "
            f"N*m exceeds the exit-criterion-2 gate of {_V025_TOL_NM} N*m"
        )

    # Non-degeneracy, one assertion per equation the residual claims to
    # cover. Thresholds sit well inside the measured values, so a legitimate
    # model change does not trip them while any return to a degenerate
    # fixture does.
    dq = _p6_error_quaternion(q_cmd_log, q_est)
    dq0 = dq[:, 0]
    cycles = len(dq0)
    c = _p6_error_dcm(dq)

    rate_scale = float(np.abs(w_cmd_log).max())
    if rate_scale <= 1e-3:
        raise CheckFailure(
            f"commanded body rate peaks at {rate_scale:.3e} rad/s, so "
            f"eq:gnc:werr has nothing to rotate and the residual cannot see "
            f"that term"
        )
    # eq:gnc:werr is measured on the term the torque actually consumes,
    # C(dq) w_cmd, rather than on how far C(dq) sits from the identity or
    # from symmetric. Those two proxies are what the pytest fixture asserts,
    # and they are not equivalent: an error rotation about an axis PARALLEL
    # to w_cmd leaves C(dq) w_cmd == w_cmd however far C(dq) is from either,
    # so the proxies stay comfortably large while the term is inert. Measured
    # on the shipped scenario the rotation moves the commanded rate by 125 %
    # of its own peak and its transpose differs by 173 %; on a
    # parallel-axis variant the same two quantities collapse to 5.6 % and
    # 13.1 % while the proxies barely move (0.70 and 1.19 against 0.71 and
    # 1.24). The 50 % gate below separates them with better than a factor of
    # two either way.
    rotated = np.einsum("kij,kj->ki", c, w_cmd_log)
    transposed = np.einsum("kji,kj->ki", c, w_cmd_log)
    moved = float(np.abs(rotated - w_cmd_log).max())
    handed = float(np.abs(rotated - transposed).max())
    if moved <= 0.5 * rate_scale or handed <= 0.5 * rate_scale:
        raise CheckFailure(
            f"C(dq) moves the commanded rate by {moved:.3e} rad/s and differs "
            f"from its transpose by {handed:.3e} rad/s, against a commanded "
            f"rate of {rate_scale:.3e} rad/s; eq:gnc:werr's rotation, or its "
            f"handedness, would be invisible to the residual"
        )
    negative = int(np.sum(dq0 < 0.0))
    if negative <= cycles // 8 or cycles - negative <= cycles // 8:
        raise CheckFailure(
            f"eq:gnc:sign takes its short-path branch on {negative} of "
            f"{cycles} cycles; both branches must carry a substantial run"
        )
    unclamped = _p6_pd_torque(
        q_cmd_log, q_est, w_cmd_log, w_est, _V025_KP, _V025_KD, None
    )
    clamped = int(np.any(np.abs(unclamped) > _V025_TAU_MAX, axis=1).sum())
    if clamped <= cycles // 20:
        raise CheckFailure(
            f"eq:gnc:sat clamps on {clamped} of {cycles} cycles; the "
            f"saturated law would be indistinguishable from the unsaturated one"
        )
    per_axis = np.abs(logged_tau).max(axis=0)
    if not np.all(per_axis > 1e-3):
        raise CheckFailure(
            f"per-axis peak torque {per_axis} leaves an eq:gnc:pd gain path "
            f"multiplied by zero"
        )


# --------------------------------------------------------------------------
# V026: Phase 6 exit criterion 9.
# --------------------------------------------------------------------------

# Speed of light in vacuum: exact by the SI definition of the metre (BIPM SI
# Brochure), so it carries no uncertainty.
_P6_C_MPS = 299792458.0
_P6_MAS_PER_RAD = 1000.0 * 648000.0 / math.pi
_P6_J2000_JD = 2451545.0

# The gate is 1e-5 mas, not the criterion's 1 mas, and the two are asserted
# separately so the suite states both the requirement it meets and the
# tighter bound it is held to. Gating at the requirement leaves five orders
# of slack in which an algebraically wrong formula sits comfortably: the
# drop-the-transverse-projection mutation measures 0.5401 mas on this
# fixture and would pass 1 mas untouched. The measured worst residual
# against the reference here is 3.46e-08 mas, so 1e-5 keeps about 290x
# headroom over the observed rounding-order residual while rejecting that
# mutation by about 5.4e+04. This is the same constant, and the same
# reasoning, as ABERRATION_TOL_MAS in tests/python/test_p6_optical_gates.py.
_V026_TOL_MAS = 1e-5
_V026_REQUIREMENT_MAS = 1.0


def _p6_aberrate_first_order(u_i, beta):
    """Apparent direction by eq:optical:aberration (the normative formula).

    ``u' = normalize(u + beta - (u . beta) u)``: the geometric direction plus
    the component of beta transverse to it, renormalized. ``u`` points FROM
    the observer TO the source. The comparison this serves is first-order
    against first-order - ch:sensors-optical declares this equation THE
    formula and specifies that criterion 9 recomputes IT, so gating against
    the exact relativistic form would measure a deliberate modelling choice
    rather than an implementation error.
    """
    u = np.asarray(u_i, dtype=np.float64)
    u = u / np.linalg.norm(u)
    b = np.asarray(beta, dtype=np.float64)
    shifted = u + b - float(np.dot(u, b)) * u
    return shifted / np.linalg.norm(shifted)


def _p6_separation_angle(a, b) -> float:
    """Angle between two vectors [rad], by the atan2 form of eq:optical:gating.

    atan2(|a x b|, a . b) rather than arccos: an aberration residual lives in
    the nearly-parallel regime, where arccos loses half its significant digits.
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    return math.atan2(float(np.linalg.norm(np.cross(x, y))), float(np.dot(x, y)))


# A synthesized heliocentric ephemeris covering the mission epoch. The
# segment layout matches the DE440 repack (sun and emb SSB-centered, earth
# and moon EMB-centered, kind 0 in km), and each record carries a linear
# Chebyshev term so the EMB has a real barycentric VELOCITY: aberration is
# dominated by that annual term, and a constant-position fixture would leave
# beta at the vehicle's LEO speed alone and shrink the signal the gate
# resolves by a factor of four.
#
# The EMB sits one astronomical unit along -Y with its velocity along -X, so
# its position and velocity are orthogonal (circular heliocentric motion) and
# the Sun lies along +Y as seen from Earth. That places the vehicle's own +Y
# LEO velocity ALONG the line of sight, which is what keeps the angle between
# the line of sight and beta away from 90 degrees. It matters: the difference
# between the normative formula and the dropped-transverse-projection
# mutation is (beta^2 / 2) sin(2 theta), so at theta = 90 degrees the
# mutation would be invisible however tight the tolerance. Measured here:
# theta ~ 76 degrees, |beta| = 1.04e-04, and the mutation measures
# 0.5401 mas.
_V026_AU_KM = 1.4959787e8
_V026_V_EMB_KMPS = 29.78  # Earth's mean heliocentric orbital speed
_V026_INTLEN_S = 3.0 * 86400.0
_V026_EPOCH = (2026, 1, 1, 0, 0, 0.0)

_V026_MISSION = """\
schema_version = 1
vehicle = "verify_vehicle.toml"

[mission]
name = "verify-p6-aberration"
epoch_utc = "2026-01-01T00:00:00Z"
duration_s = 60.0

[run]
seed = 20260601

[integrator]
type = "rk4"
dt_s = 0.1

[environment]
central_body = "earth"
# The Sun and Moon third bodies are enabled so the mission validator accepts
# the ephemeris: its consumer list counts force models only, and the optical
# sensors that actually require the Sun direction are not among them.
third_bodies = ["sun", "moon"]
ephemeris = "{ephemeris}"

[logging]
truth_rate_hz = 10

[initial_state.cartesian]
r_m = [7.0e6, 0.0, 0.0]
v_mps = [0.0, 7546.0, 0.0]
frame = "GCRF"

[gnc]
control_rate_hz = 10
latency_cycles = 0

[gnc.nav]
component = "dead_reckoning"
q0 = [0.0, 0.7071067811865476, 0.7071067811865476, 0.0]

[gnc.guidance]
component = "attitude_hold"
# A 10-degree slew about the OFF-AXIS body direction [1, 2, -2]/3 rather than
# about body +Z. A +Z slew from this initial attitude leaves the attitude in
# the q_w == 0 plane for the whole run, and at q_w == 0 the DCM of
# eq:notation:quat2dcm is exactly symmetric - C - C^T = -4 q_w [q_v x]
# vanishes identically. Criterion 9's residual is an angular separation, so
# on such a fixture a TRANSPOSED attitude convention is undetectable by
# geometry no matter which implementation supplies the DCM. This axis carries
# |q_w| up to 0.060 and |C - C^T| up to 0.18.
q_cmd = [-0.061628416716219346, 0.66333041525861247, 0.74550163754690491, 0.020542805572073115]

[gnc.control]
component = "pd_attitude"
kp_nm_per_rad = [0.4, 0.4, 0.4]
kd_nm_per_radps = [3.6, 3.6, 3.6]
tau_max_nm = [0.05, 0.05, 0.05]

[sensors.imu]
sample_rate_hz = 10

[sensors.sunsensor]
sample_rate_hz = 5
boresight_b = [1.0, 0.0, 0.0]
# Full sphere, so the field-of-view gate never masks a sample, and noise-free
# so the logged channel carries the aberration transformation alone.
fov_half_angle_rad = 3.141592653589793
sigma_rad = 0.0
"""


def _v026_ephemeris(tmpdir: Path) -> Path:
    core = import_core()
    day, sec = core.utc_to_tai(*_V026_EPOCH)
    jd1, jd2 = core.tdb_jd(day, sec)
    tdb_s = ((jd1 - _P6_J2000_JD) + jd2) * 86400.0
    init = tdb_s - 86400.0
    # position(x) = c0 + c1 x with x = 2 (t - t_mid) / intlen, so a constant
    # velocity v is the single linear coefficient c1 = v intlen / 2.
    c1 = _V026_V_EMB_KMPS * _V026_INTLEN_S / 2.0

    def const_record(x_km: float, y_km: float, z_km: float) -> list:
        return [[[x_km, 0.0], [y_km, 0.0], [z_km, 0.0]]]

    segments = [
        {"name": "sun", "target": 10, "center": 0, "kind": 0,
         "init_tdb_s": init, "intlen_s": _V026_INTLEN_S,
         "records": const_record(0.0, 0.0, 0.0)},
        {"name": "emb", "target": 3, "center": 0, "kind": 0,
         "init_tdb_s": init, "intlen_s": _V026_INTLEN_S,
         "records": [[[0.0, -c1], [-_V026_AU_KM, 0.0], [0.0, 0.0]]]},
        {"name": "earth", "target": 399, "center": 3, "kind": 0,
         "init_tdb_s": init, "intlen_s": _V026_INTLEN_S,
         "records": const_record(4671.0, 0.0, 0.0)},
        {"name": "moon", "target": 301, "center": 3, "kind": 0,
         "init_tdb_s": init, "intlen_s": _V026_INTLEN_S,
         "records": const_record(-379700.0, 0.0, 0.0)},
    ]
    path = tmpdir / "v026.sreph"
    path.write_bytes(_fixtures.build_sreph(segments))
    return path


def _check_v026(ctx: dict) -> None:
    """Phase 6 exit criterion 9: the logged apparent Sun direction matches an
    independent recomputation of eq:optical:aberration.

    Both sides of the comparison are free of the code under test. The
    ephemeris is evaluated through star_reacher.data_fetch's pure-Python
    SREPH reader, not the C++ loader, so the source direction and the
    observer velocity are not taken from the implementation being gated; the
    apparent direction is rotated into body axes by _p6_quat_to_dcm rather
    than by the core's quat_to_dcm, because an angular separation is
    invariant under a rotation applied to both arguments and the core's own
    DCM would cancel exactly.

    Measured on this fixture: worst residual 3.46e-08 mas against the
    1e-5 mas gate, aberration deflection 20.44 to 20.78 arcsec (the
    eq:optical:abmag scale of beta sin(theta) at Earth's barycentric speed),
    worst |C - C^T| = 0.1796.

    Both mutations the audit used are rejected: replacing the reference with
    ``normalize(u + beta)`` - dropping the transverse projection - measures
    0.5401 mas, which is 5.4e+04 times the gate and yet would pass the
    criterion's own 1 mas figure untouched; transposing the reference DCM
    measures 3.34e+07 mas.
    """
    import os

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        (tdp / "verify_vehicle.toml").write_text(_P6_VEHICLE, encoding="utf-8")
        eph_path = _v026_ephemeris(tdp)
        mission = tdp / "aberration.toml"
        mission.write_text(
            _V026_MISSION.format(ephemeris=eph_path.as_posix()), encoding="utf-8"
        )
        cwd = os.getcwd()
        os.chdir(tdp)
        try:
            result = run_mission(mission, tdp / "run")
        finally:
            os.chdir(cwd)
        run = load(result.srlog_path)

        from star_reacher.data_fetch import evaluate_segment, read_sreph

        eph = read_sreph(eph_path)

        def state_m(name: str, tdb_s: float):
            seg = eph.segment_for(name, tdb_s)
            position_km, rate_km_s = evaluate_segment(seg, tdb_s)
            return np.array(position_km) * 1000.0, np.array(rate_km_s) * 1000.0

        core = import_core()
        epoch_tai = core.utc_to_tai(*_V026_EPOCH)
        truth = run.groups["truth"]
        sun = run.groups["sensors.sunsensor"]
        if len(sun) < 2:
            raise CheckFailure(f"the sun sensor logged {len(sun)} sample(s)")
        # Both grids are exact multiples of the control cycle, so the integer
        # cycle index is an exact key rather than a nearest-neighbour search.
        index = {int(round(t / 0.1)): i for i, t in enumerate(truth["t_s"])}
        rows = [index[int(round(float(t) / 0.1))] for t in sun["t_s"]]

        worst = 0.0
        worst_deflection = 0.0
        least_deflection = math.inf
        for j, t_s in enumerate(sun["t_s"]):
            row = rows[j]
            tai = core.tai_add_seconds(epoch_tai[0], epoch_tai[1], float(t_s))
            jd1, jd2 = core.tdb_jd(tai[0], tai[1])
            tdb_s = ((jd1 - _P6_J2000_JD) + jd2) * 86400.0
            # DE440 stores the EMB against the SSB and the Earth against the
            # EMB, so the barycentric Earth state is the sum of two segments.
            r_emb, v_emb = state_m("emb", tdb_s)
            r_earth_emb, v_earth_emb = state_m("earth", tdb_s)
            r_earth, v_earth = r_emb + r_earth_emb, v_emb + v_earth_emb
            r_sun, _ = state_m("sun", tdb_s)
            # eq:optical:beta: the observer's barycentric velocity is the
            # vehicle velocity relative to the central body plus the central
            # body's own velocity relative to the SSB.
            beta = (truth["v_mps"][row] + v_earth) / _P6_C_MPS
            geometric = r_sun - r_earth - truth["r_m"][row]
            geometric = geometric / np.linalg.norm(geometric)
            apparent = _p6_aberrate_first_order(geometric, beta)
            c_i2b = _p6_quat_to_dcm(truth["q_i2b"][row])
            residual = _p6_separation_angle(c_i2b @ apparent, sun["sun_b"][j])
            worst = max(worst, residual * _P6_MAS_PER_RAD)
            deflection = (
                _p6_separation_angle(geometric, apparent) * _P6_MAS_PER_RAD
            )
            worst_deflection = max(worst_deflection, deflection)
            least_deflection = min(least_deflection, deflection)

    if worst >= _V026_TOL_MAS:
        raise CheckFailure(
            f"worst logged-versus-reference Sun direction residual "
            f"{worst:.6e} mas exceeds the exit-criterion-9 gate of "
            f"{_V026_TOL_MAS} mas"
        )
    if worst >= _V026_REQUIREMENT_MAS:
        raise CheckFailure(
            f"worst residual {worst:.6e} mas exceeds the criterion's own "
            f"requirement of {_V026_REQUIREMENT_MAS} mas"
        )

    # The gate is not vacuous only if the aberration is really present: an
    # implementation that skipped the correction entirely would miss the
    # logged direction by the whole deflection, so pinning the deflection
    # pins the signal the residual is resolving.
    if not 20_000.0 < least_deflection <= worst_deflection < 21_000.0:
        raise CheckFailure(
            f"aberration deflection spans {least_deflection:.1f} to "
            f"{worst_deflection:.1f} mas, outside the eq:optical:abmag scale "
            f"of ~20.5 arcsec at Earth's barycentric speed; the residual is "
            f"not resolving the correction this criterion is about"
        )

    # Substituting the independent DCM is necessary but not sufficient. At
    # q_w == 0 the DCM of eq:notation:quat2dcm is exactly symmetric, so on a
    # fixture whose attitude stays in that plane a transposed convention
    # changes nothing at all and the substitution reads as convention-aware
    # while remaining blind.
    asymmetry = max(
        float(np.abs(_p6_quat_to_dcm(q) - _p6_quat_to_dcm(q).T).max())
        for q in truth["q_i2b"]
    )
    if asymmetry <= 0.05:
        raise CheckFailure(
            f"the fixture's attitude keeps C_I2B within {asymmetry:.3e} of "
            f"symmetric, so a transposed attitude convention would be "
            f"invisible to this residual"
        )


# --------------------------------------------------------------------------
# V027: Phase 6 exit criterion 3.
#
# The criterion is a conjunction: an R-run seeded ensemble of the reference
# EKF mission passes ensemble NEES and per-sensor NIS against the
# eq:ekf:ensemble chi-square bounds, AND re-executing the ensemble reproduces
# every run's SRLOG SHA-256 bit for bit. R = 100 is the criterion's own
# ensemble size and the only size this check runs at - see _V027_RUNS for why
# that number, and not a cheaper one, is what the gate needs.
#
# The verdict is taken through star_reacher.consistency.ensemble_gate, the
# FR-26 acceptance instrument, rather than re-derived here. NEES is gated on
# the headline alone: the state error is a smooth trajectory whose epochs are
# strongly correlated, and a binomial coverage threshold applied to them
# rejects a correct filter about half the time. NIS is gated on the headline
# AND on the binomial coverage threshold, because a consistent filter's
# innovations are white and successive NIS epochs are therefore independent.
# Both declarations are structural properties of the statistics, fixed before
# any data is seen, never estimated from the run being judged.
# --------------------------------------------------------------------------

# The scenario's pinned truth. This is missions/leo_ekf_consistency.toml: the
# truth environment is the central-body POINT MASS, which is the same gravity
# model the navigator carries internally (ch:ekf assumption 2), and the IMU
# carries only the two error terms the filter models. The scenario tests the
# ESTIMATOR, not a model-error budget; running it against a richer truth
# environment is a legitimate diagnostic but is not this gate.
_P6_EKF_A = math.sqrt(0.5)
_P6_EKF_Q_TRUE = np.array([0.0, _P6_EKF_A, _P6_EKF_A, 0.0])
_P6_EKF_R_TRUE_M = np.array([7.0e6, 0.0, 0.0])
_P6_EKF_V_TRUE_MPS = np.array([0.0, 7546.0, 0.0])
_P6_EKF_SIGMA_ATT_RAD = 1.0e-3
_P6_EKF_SIGMA_VEL_MPS = 0.5
_P6_EKF_SIGMA_POS_M = 50.0
_P6_EKF_BASE_RUN_SEED = 20260701
# The draw stream is separate from the core's run seed so the initial-error
# draws and the sensor noise cannot alias onto one another.
_P6_EKF_DRAW_SEED = 90210
_P6_NEES_DIM = 15
# Sensor ids as the canonical kind order assigns them (imu, startracker,
# navfix, altimeter), mapped to each aiding sensor's innovation dimension.
_P6_NIS_DIM_BY_SENSOR = {1: 3, 2: 6, 3: 1}

_P6_EKF_MISSION = """\
schema_version = 1
vehicle = "verify_vehicle.toml"

[mission]
name = "verify-p6-ekf"
epoch_utc = "2026-01-01T00:00:00Z"
duration_s = 60.0

[run]
seed = {seed}

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

[gnc.nav]
component = "error_state_ekf"
q0 = {q0}
v0_mps = {v0}
p0_m = {p0}
bg0_radps = [0.0, 0.0, 0.0]
ba0_mps2 = [0.0, 0.0, 0.0]
p0_sigma_att_rad = [1.0e-3, 1.0e-3, 1.0e-3]
p0_sigma_vel_mps = [0.5, 0.5, 0.5]
p0_sigma_pos_m = [50.0, 50.0, 50.0]
p0_sigma_bg_radps = [1.0759973046695306e-7, 1.0759973046695306e-7, 1.0759973046695306e-7]
p0_sigma_ba_mps2 = [1.0759973046695306e-5, 1.0759973046695306e-5, 1.0759973046695306e-5]

[gnc.guidance]
component = "attitude_hold"
q_cmd = [0.0, 0.7071067811865476, 0.7071067811865476, 0.0]

[gnc.control]
component = "pd_attitude"
kp_nm_per_rad = [0.4, 0.4, 0.4]
kd_nm_per_radps = [3.6, 3.6, 3.6]
tau_max_nm = [0.05, 0.05, 0.05]

[sensors.imu]
sample_rate_hz = 10
gyro_arw_rad_per_sqrt_s = 1.0e-5
gyro_bias_instability_radps = 1.0e-7
gyro_bias_tau_s = 100.0
accel_vrw_mps_per_sqrt_s = 1.0e-4
accel_bias_instability_mps2 = 1.0e-5
accel_bias_tau_s = 100.0

[sensors.navfix]
sample_rate_hz = 1
sigma_r_m = [10.0, 10.0, 10.0]
sigma_v_mps = [0.1, 0.1, 0.1]

[sensors.startracker]
sample_rate_hz = 1
boresight_b = [0.0, 0.0, 1.0]
sigma_rad = [1.0e-5, 1.0e-5, 5.0e-5]

[sensors.altimeter]
sample_rate_hz = 1
sigma_noise_m = 20.0
sigma_bias_m = 0.0
h_min_m = 0.0
h_max_m = 0.0
"""


def _p6_initial_estimate(run_index: int):
    """The filter's initial estimate for one ensemble run.

    The error is drawn from N(0, P0) and SUBTRACTED from truth, so the
    realized initial error is distributed exactly as P0 claims - the
    precondition for NEES consistency from the first cycle. The attitude uses
    the multiplicative convention of eq:ekf:qerr: q_true = q_hat (x)
    dq(dtheta) gives q_hat = q_true (x) dq(-dtheta).

    The bias estimates are deliberately NOT perturbed. They start at zero and
    P0's bias blocks carry the instruments' stationary Gauss-Markov sigmas;
    because the IMU initializes its in-run bias from exactly that stationary
    distribution, the initial bias error already has the distribution the
    filter believes, without this driver needing to know the sensor's private
    draw.
    """
    rng = np.random.default_rng(_P6_EKF_DRAW_SEED + run_index)
    dtheta = _P6_EKF_SIGMA_ATT_RAD * rng.standard_normal(3)
    dv = _P6_EKF_SIGMA_VEL_MPS * rng.standard_normal(3)
    dp = _P6_EKF_SIGMA_POS_M * rng.standard_normal(3)
    return (
        _p6_quat_mul(_P6_EKF_Q_TRUE, _p6_quat_exp(-dtheta)),
        _P6_EKF_V_TRUE_MPS - dv,
        _P6_EKF_R_TRUE_M - dp,
    )


def _p6_format_vector(values) -> str:
    # Full round-trip precision: a truncated initial estimate would make this
    # ensemble member a different run from the one the draw defines.
    return "[" + ", ".join("%.17g" % float(v) for v in values) + "]"


def _p6_reduce_error(e):
    """Reduce the logged 16-vector nav.err to the 15 dimensions P describes.

    The leading four components are the sign-canonicalized multiplicative
    error quaternion of eq:ekf:qerr; the small-angle extraction
    dtheta = 2 sgn(dq_w) dq_v is the same reduction ``star consistency``
    applies.
    """
    sign = np.where(e[:, 0] >= 0.0, 1.0, -1.0)[:, np.newaxis]
    return np.concatenate([2.0 * sign * e[:, 1:4], e[:, 4:]], axis=1)


def _p6_per_sensor_innovations(innov):
    """Split zero-padded nav.innov records into per-sensor (y, S) arrays."""
    from star_reacher.consistency import pack_symmetric, unpack_symmetric

    m_max = innov["y"].shape[-1]
    out = {}
    for sensor_id in sorted({int(s) for s in innov["sensor_id"]}):
        sel = innov["sensor_id"] == sensor_id
        m = int(innov["m"][sel][0])
        y = innov["y"][sel][:, :m]
        s = innov["S"][sel]
        if m < m_max:
            s = pack_symmetric(unpack_symmetric(s)[:, :m, :m])
        out[sensor_id] = (y, s)
    return out


def _p6_execute_ensemble(outroot: Path, runs: int):
    """Run ``runs`` ensemble members; return (sha list, NEES array, NIS dict)."""
    import os

    from star_reacher.consistency import nees, nis

    outroot.mkdir(parents=True, exist_ok=True)
    (outroot / "verify_vehicle.toml").write_text(_P6_VEHICLE, encoding="utf-8")
    shas: list[str] = []
    nees_runs: list = []
    nis_runs: dict = {}
    cwd = os.getcwd()
    # The mission's vehicle reference resolves against the working directory.
    os.chdir(outroot)
    try:
        for i in range(runs):
            q0, v0, p0 = _p6_initial_estimate(i)
            mission = outroot / ("run%03d.toml" % i)
            mission.write_text(
                _P6_EKF_MISSION.format(
                    seed=_P6_EKF_BASE_RUN_SEED + i,
                    q0=_p6_format_vector(q0),
                    v0=_p6_format_vector(v0),
                    p0=_p6_format_vector(p0),
                ),
                encoding="utf-8",
            )
            result = run_mission(mission, outroot / ("run%03d" % i), force=True)
            shas.append(result.srlog_sha256)
            run = load(result.srlog_path)
            nees_runs.append(
                nees(
                    _p6_reduce_error(run.groups["nav.err"]["e"]),
                    run.groups["nav.est"]["P"],
                )
            )
            for sensor_id, (y, s) in _p6_per_sensor_innovations(
                run.groups["nav.innov"]
            ).items():
                nis_runs.setdefault(sensor_id, []).append(nis(y, s))
    finally:
        os.chdir(cwd)
    return (
        shas,
        np.stack(nees_runs),
        {k: np.stack(v) for k, v in nis_runs.items()},
    )


def _p6_criterion3(runs: int) -> None:
    """Execute and gate the criterion-3 ensemble at ``runs`` members."""
    from star_reacher.consistency import ensemble_gate

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        shas, nees_eps, nis_eps = _p6_execute_ensemble(tdp / "ensemble", runs)

        if nees_eps.shape[0] != runs:
            raise CheckFailure(
                f"the ensemble produced {nees_eps.shape[0]} NEES runs, not {runs}"
            )
        # A filter that silently skipped a sensor would otherwise pass the
        # gates it did run and look healthy.
        if sorted(nis_eps) != sorted(_P6_NIS_DIM_BY_SENSOR):
            raise CheckFailure(
                f"innovations arrived from sensors {sorted(nis_eps)}, expected "
                f"{sorted(_P6_NIS_DIM_BY_SENSOR)}; an aiding sensor contributed "
                f"nothing and its NIS gate would be vacuous"
            )

        gate = ensemble_gate(nees_eps, _P6_NEES_DIM, epochs_independent=False)
        if gate.coverage_gated:
            raise CheckFailure(
                "NEES coverage must not gate: the state error is a correlated "
                "trajectory and the binomial premise does not hold for it"
            )
        if not gate.passed:
            raise CheckFailure(
                f"ensemble NEES headline {gate.headline.mean:.4f} outside "
                f"[{gate.lower:.4f}, {gate.upper:.4f}] (chi2 95 %, dof "
                f"{gate.dof}, R = {runs}); coverage "
                f"{gate.inside_count}/{len(gate.epoch_mean)} epochs inside "
                f"(diagnostic only)"
            )

        for sensor_id, dim in sorted(_P6_NIS_DIM_BY_SENSOR.items()):
            nis_gate = ensemble_gate(
                nis_eps[sensor_id], dim, epochs_independent=True
            )
            if not nis_gate.coverage_gated:
                raise CheckFailure(
                    f"sensor {sensor_id} NIS coverage is not gating; a "
                    f"consistent filter's innovations are white and the "
                    f"binomial threshold is an exact false-failure bound"
                )
            if not nis_gate.passed:
                raise CheckFailure(
                    f"sensor {sensor_id} ensemble NIS headline "
                    f"{nis_gate.headline.mean:.4f} against "
                    f"[{nis_gate.lower:.4f}, {nis_gate.upper:.4f}] (chi2 95 %, "
                    f"dof {nis_gate.dof}, R = {runs}); headline "
                    f"{'passed' if nis_gate.headline.passed else 'FAILED'}, "
                    f"coverage {nis_gate.inside_count}/"
                    f"{len(nis_gate.epoch_mean)} epochs inside against a "
                    f"threshold of {nis_gate.min_inside}"
                )

        rerun_shas, _, _ = _p6_execute_ensemble(tdp / "rerun", runs)

    if rerun_shas != shas:
        differing = [
            i for i, (a, b) in enumerate(zip(shas, rerun_shas)) if a != b
        ]
        raise CheckFailure(
            f"re-executing the ensemble changed the SRLOG SHA-256 of "
            f"{len(differing)} of {runs} runs (first at index "
            f"{differing[0]}: {shas[differing[0]]} != "
            f"{rerun_shas[differing[0]]})"
        )


# WHY R = 100 AND NOT A CHEAPER ENSEMBLE. R is the criterion's own figure,
# but it is also the figure the detection power argues for, and that argument
# is recorded here so a future cost pressure meets a number rather than a
# preference.
#
# A filter reporting a covariance mis-scaled by a factor f has a per-epoch
# statistic (1/f) times the consistent one, so the ensemble NEES headline
# sits at n/f and the gate fires when n/f leaves the eq:ekf:ensemble interval
# [chi2_0.025(Rn)/R, chi2_0.975(Rn)/R]. That crossing is the 50 %-power
# point. How much further a mis-scale has to go for reliable detection is set
# by the headline's own sampling spread: the per-run time-averaged NEES has a
# measured standard deviation of 3.838 across the 100 runs - against 0.223 if
# the 601 epochs within a run were independent, so a run carries only about
# 2.0 effectively independent epochs - which gives the headline a standard
# deviation of 3.838/sqrt(R).
#
# Both the interval half-width and that standard deviation scale as
# 1/sqrt(R). Two consequences follow, and they are the whole case for the
# ensemble size. First, the false-failure rate is 0.5 % at EVERY R, so
# shrinking the ensemble buys no robustness - it costs power and nothing
# else. Second, the detection thresholds, computed from the project's own
# exact chi-square evaluator:
#
#                      50 % power            90 % power
#     R = 100     +7.6 % / -6.8 %     +11.1 % / -9.8 %
#     R =  28    +15.0 % / -12.3 %    +22.2 % / -17.7 %
#
# A quarter-sized ensemble is roughly half as sensitive, and a defect that
# mis-scales the reported covariance by 10 % - well inside the range a real
# tuning or linearization error produces - is resolved at R = 100 and missed
# at R = 28. That is the resolution the criterion is worth running at.
#
# The three NIS statistics are weaker even here, and gate a different failure
# (an innovation covariance that does not match the residuals) rather than
# adding power to the NEES statement: their 50 %-power thresholds at R = 100
# are +12.4 % (navfix, dim 6), +18.2 % (startracker, dim 3) and +34.7 %
# (altimeter, dim 1).
#
# The gate is not fragile at this size. Every prefix R = 2..100 of this
# ensemble passes all four gates, and the R = 100 NEES headline sits 43.4 %
# of the interval width above the lower bound.
_V027_RUNS = 100  # the criterion's own ensemble size


def _check_v027(ctx: dict) -> None:
    """Phase 6 exit criterion 3: the criterion's own 100-run EKF ensemble.

    Measured on this fixture at R = 100: ensemble NEES headline 14.8776
    inside [13.9456, 16.0923] with 599 of 601 epochs inside (diagnostic,
    not gated); per-sensor NIS headlines 2.9895 (startracker, dim 3, 56 of
    60 epochs inside), 6.0306 (navfix, dim 6, 56 of 60) and 0.9816
    (altimeter, dim 1, 57 of 60) against a coverage threshold of 51; every
    SRLOG SHA-256 reproduced on the rerun. These are the same numbers
    tests/python/test_ekf_consistency.py measures on the committed mission,
    which is what establishes that the inline fixture here is the same
    scenario.

    The gate was shown able to fail rather than assumed able to: scaling the
    reported covariance flips this ensemble red at f = 1.0668 and at
    f = 0.9245. Those realized flip points are seed artifacts of where this
    ensemble's headline happens to sit inside its band; the population
    thresholds derived at _V027_RUNS - 90 % power against +11.1 % or -9.8 %,
    99 % power against +14.0 % or -12.3 %, at a 0.5 % false-failure rate -
    are the figures to design against.
    """
    _p6_criterion3(_V027_RUNS)


# --------------------------------------------------------------------------
# Phase 7 checks (V028-V029): the Monte Carlo layer's two exit criteria.
#
# V028 is criterion 1 (the seeded sweep is bit-reproducible: expanding a spec,
# deriving a per-run seed, running, and hashing, then re-executing one manifest
# entry through the star run API reproduces its logged hash), and V029 is
# criterion 2 (an ensemble's statistics match frozen goldens within
# chi-square/Anderson-Darling 99 % bounds). Both are self-contained on a bare
# wheel like every check above: the sweep mission is synthesized in a temp
# directory (V029's harmonic gravity field via _synthetic_j2_field, the same
# helper V014 uses), and V029's frozen golden statistics are inlined with their
# provenance rather than read from tests/golden/, because an installed wheel
# carries no source tree.
#
# WORKER COUNT. Both ensembles run at workers=1 (in-process). The result is
# worker-count-independent by construction (D-10: no mutable state is shared,
# so a run is bit-identical whether pooled or standalone), and that
# independence is gated in the pytest suite by
# test_mc.py::test_worker_count_does_not_change_the_result. Housing the
# parallel-pool exercise there rather than here keeps star verify -- the
# documented first command on a fresh install -- off the process pool, whose
# abrupt-worker-death failures are an environment property (observed on some
# Windows hosts) and not a determinism defect the acceptance gate should
# translate into a red.


_V028_MISSION = """\
schema_version = 1

[mission]
name = "verify-v028"
epoch_utc = "2026-01-01T00:00:00Z"
duration_s = 600.0

[run]
seed = 24601

[integrator]
type = "rk4"
dt_s = 1.0

[initial_state.cartesian]
r_m = [6778137.0, 0.0, 0.0]
v_mps = [0.0, 7668.6, 0.0]
frame = "GCRF"

[environment]
central_body = "earth"

[logging]
truth_rate_hz = 1
"""

# A two-dimensional Latin hypercube over the two-body mission: a whole-second
# propagation window and the in-plane velocity component (an indexed vector
# override, so the sweep exercises the dotted-path override vocabulary). N is
# 32, which expands, runs, hashes, and reproduces in well under a second at one
# worker.
_V028_SWEEP = """\
schema_version = 1

[sweep]
mission = "v028_mission.toml"
master_seed = 20260723
method = "lhs"
n_runs = 32

[[sweep.parameter]]
path = "mission.duration_s"
min = 600.0
max = 900.0
integer = true

[[sweep.parameter]]
path = "initial_state.cartesian.v_mps.1"
min = 7600.0
max = 7700.0
"""
_V028_N_RUNS = 32
_V028_REEXEC_INDEX = 7  # any interior entry; re-executed to reproduce its hash


def _check_v028(ctx: dict) -> None:
    """Phase 7 exit criterion 1: a seeded sweep is individually reproducible.

    Runs a self-contained 32-run LHS sweep of a two-body mission through the
    star mc engine (star_reacher.mc.run_sweep), asserts every run succeeds and
    every logged hash is distinct, then re-executes one manifest entry through
    run_mission -- the star run API -- with only its recorded seed and
    overrides and asserts it reproduces both the entry's log_sha256 and its
    config_sha256. That reproduction is the criterion.

    The gate is shown able to fail, not assumed to be: re-executing the same
    entry with the seed flipped by one bit must NOT reproduce the hash, so a
    reproduction that ignored the seed would be caught here.
    """
    import os

    from star_reacher.mc import run_sweep

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        (tdp / "v028_mission.toml").write_text(_V028_MISSION, encoding="utf-8")
        (tdp / "sweep.toml").write_text(_V028_SWEEP, encoding="utf-8")
        cwd = os.getcwd()
        os.chdir(tdp)
        try:
            manifest = run_sweep(
                tdp / "sweep.toml", workers=1, outdir=str(tdp / "out"), force=True
            )
            runs = manifest["runs"]
            failed = [r for r in runs if r["status"] != "success"]
            if failed:
                raise CheckFailure(
                    f"{len(failed)} of {len(runs)} sweep runs failed (first: "
                    f"{failed[0].get('error')})"
                )
            if len(runs) != _V028_N_RUNS:
                raise CheckFailure(
                    f"the sweep produced {len(runs)} runs, expected {_V028_N_RUNS}"
                )
            # Distinct hashes: a sweep that silently ran the same case N times
            # would pass a per-entry reproduction while testing nothing.
            distinct = len({r["log_sha256"] for r in runs})
            if distinct != len(runs):
                raise CheckFailure(
                    f"only {distinct} of {len(runs)} logged hashes are distinct; "
                    f"the sweep is not dispersing its runs"
                )

            entry = runs[_V028_REEXEC_INDEX]
            reexec = run_mission(
                tdp / "v028_mission.toml",
                outdir=str(tdp / "reexec"),
                force=True,
                seed=entry["seed"],
                overrides=entry["overrides"],
            )
            if reexec.srlog_sha256 != entry["log_sha256"]:
                raise CheckFailure(
                    f"re-executing manifest entry {_V028_REEXEC_INDEX} with its "
                    f"recorded seed and overrides gave log SHA-256 "
                    f"{reexec.srlog_sha256}, not the manifest's "
                    f"{entry['log_sha256']}"
                )
            if reexec.config_sha256 != entry["config_sha256"]:
                raise CheckFailure(
                    f"re-executed config SHA-256 {reexec.config_sha256} != "
                    f"manifest {entry['config_sha256']}"
                )

            # Able to fail: the same overrides under a one-bit-different seed
            # must produce a different log, or the reproduction proved nothing.
            wrong = run_mission(
                tdp / "v028_mission.toml",
                outdir=str(tdp / "wrong"),
                force=True,
                seed=entry["seed"] ^ 1,
                overrides=entry["overrides"],
            )
            if wrong.srlog_sha256 == entry["log_sha256"]:
                raise CheckFailure(
                    "a one-bit-different seed reproduced the same log hash; the "
                    "reproduction is not seed-sensitive and the gate is vacuous"
                )
        finally:
            os.chdir(cwd)


# V029 fixture: a J2-only harmonic mission, dispersed over the same
# initial-velocity coordinate the shipped missions/mc_regression_sweep.toml sweep
# uses, but self-contained (synthesized field, inline golden) for the bare
# wheel. The outcome metric is the final osculating specific mechanical energy,
# via star_reacher.mc_regression.
_V029_MISSION = """\
schema_version = 1

[mission]
name = "verify-v029"
epoch_utc = "2026-01-01T00:00:00Z"
duration_s = 2700.0

[run]
seed = 20260723

[integrator]
type = "rk4"
dt_s = 1.0

[spacecraft]
mass_kg = 500.0

[initial_state.cartesian]
r_m = [7000000.0, 0.0, 0.0]
v_mps = [0.0, 6900.0, 3000.0]
frame = "GCRF"

[environment]
central_body = "earth"

[environment.gravity]
model = "harmonic"
field = "{field}"
degree = 2
order = 0

[logging]
truth_rate_hz = 1
"""

_V029_SWEEP = """\
schema_version = 1

[sweep]
mission = "v029_mission.toml"
master_seed = 20260723
method = "lhs"
n_runs = 64

[[sweep.parameter]]
path = "mission.duration_s"
min = 2700.0
max = 2701.0
integer = true

[[sweep.parameter]]
path = "initial_state.cartesian.v_mps.1"
min = 6850.0
max = 6950.0
"""

# The frozen golden statistics of the V029 ensemble, measured from this exact
# synthesized fixture at master_seed 20260723 and inlined (a bare wheel has no
# tests/golden/ tree). The ensemble is bit-reproducible, so these are exact
# binary64 values, carried as float.hex() literals; the gate reproduces the
# statistics below to the bit. The committed, full-field analogue is
# tests/golden/mc_regression/energy_stats.toml, frozen from
# missions/mc_regression_sweep.toml through scripts/golden_update.py --apply and
# tested in tests/python/test_mc_regression.py.
_V029_N = 64
_V029_MEAN = float.fromhex("-0x1.b4f37e59ac1adp+24")  # metric mean [m^2/s^2]
_V029_STD = float.fromhex("0x1.84dc5495b5d03p+17")  # sample std (ddof=1)


def _check_v029(ctx: dict) -> None:
    """Phase 7 exit criterion 2: ensemble statistics match a frozen golden.

    Runs a self-contained 64-run LHS sweep of a J2-only harmonic mission, takes
    each run's final osculating specific mechanical energy as the outcome
    metric (star_reacher.mc_regression), and gates the ensemble against the
    inlined golden (mean, std) at 99 % with BOTH a chi-square two-sided
    interval on the standardized sum of squares and an Anderson-Darling test of
    the standardized metric against N(mean, std).

    Measured at the pass point on this fixture: chi-square S = 63.000 inside
    [38.610, 96.878] (dof 64), and Anderson-Darling A2 = 0.710 with
    p = 0.551. Both are a fixed function of the frozen master seed.

    The gate was shown able to fail, not assumed to be. Against this same
    golden:

    * a +0.5-sigma shift of the metric leaves the chi-square statistic inside
      its interval (a pure location shift barely moves the sum of squares) but
      drives the Anderson-Darling p-value to 1.0e-4 -- red -- which is exactly
      why both tests gate: A-D catches a shift chi-square tolerates;
    * a 1.4x scale inflation of the metric drives BOTH red -- chi-square to
      S = 123.5 (above the 96.878 upper bound) and Anderson-Darling to
      A2 = 5.478, p = 1.7e-3.

    Both mutations are re-measured here on the live ensemble and asserted to
    flip the respective gate, so a future change that made either gate
    insensitive fails this check rather than passing it silently.
    """
    import os

    from star_reacher.mc import run_sweep
    from star_reacher.mc_regression import (
        GoldenStats,
        ensemble_metric,
        regression_gate,
    )

    golden = GoldenStats(
        n=_V029_N,
        mean=_V029_MEAN,
        std=_V029_STD,
        metric="energy_m2ps2",
        mission="verify-v029-j2",
    )

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        field_path, _gm, _radius, _c20 = _synthetic_j2_field(tdp)
        (tdp / "v029_mission.toml").write_text(
            _V029_MISSION.format(field=field_path.as_posix()), encoding="utf-8"
        )
        (tdp / "sweep.toml").write_text(_V029_SWEEP, encoding="utf-8")
        cwd = os.getcwd()
        os.chdir(tdp)
        try:
            manifest = run_sweep(
                tdp / "sweep.toml", workers=1, outdir=str(tdp / "out"), force=True
            )
            failed = [r for r in manifest["runs"] if r["status"] != "success"]
            if failed:
                raise CheckFailure(
                    f"{len(failed)} of {len(manifest['runs'])} V029 runs failed "
                    f"(first: {failed[0].get('error')})"
                )
            metric = ensemble_metric(manifest, tdp / "out")
        finally:
            os.chdir(cwd)

    if metric.shape != (_V029_N,):
        raise CheckFailure(
            f"the V029 ensemble metric has shape {metric.shape}, expected "
            f"({_V029_N},)"
        )

    gate = regression_gate(metric, golden)
    if not gate.chi2_passed:
        raise CheckFailure(
            f"MC-regression chi-square statistic {gate.chi2_stat:.4f} outside "
            f"the 99 % interval [{gate.chi2_lower:.4f}, {gate.chi2_upper:.4f}] "
            f"(dof {gate.dof}) against the frozen golden"
        )
    if not gate.ad_passed:
        raise CheckFailure(
            f"MC-regression Anderson-Darling A2 {gate.ad_stat:.4f} has p-value "
            f"{gate.ad_pvalue:.6f} below the 99 % threshold of 0.01 against the "
            f"frozen golden N(mean, std)"
        )

    # The pass-point statistics are pinned: a drift in either is a visible
    # failure rather than a silently moved gate.
    if abs(gate.chi2_stat - 63.0) > 1e-6:
        raise CheckFailure(
            f"chi-square statistic {gate.chi2_stat:.6f} drifted from the frozen "
            f"63.000; the ensemble is no longer the one the golden was frozen on"
        )
    if abs(gate.ad_stat - 0.710) > 1e-2:
        raise CheckFailure(
            f"Anderson-Darling A2 {gate.ad_stat:.6f} drifted from the frozen 0.710"
        )

    # Able to fail, demonstrated on the live ensemble. A pure shift must trip
    # A-D (the location test) while chi-square tolerates it; a scale inflation
    # must trip BOTH. If either mutation stopped flipping its gate, the gate
    # would have gone insensitive and this check must catch that.
    shifted = regression_gate(metric + 0.5 * golden.std, golden)
    if shifted.ad_passed:
        raise CheckFailure(
            f"a +0.5-sigma shift did not fail the Anderson-Darling gate "
            f"(p={shifted.ad_pvalue:.6f}); the location test has gone "
            f"insensitive and the gate is not meaningful"
        )
    inflated = regression_gate(
        golden.mean + (metric - golden.mean) * 1.4, golden
    )
    if inflated.chi2_passed or inflated.ad_passed:
        raise CheckFailure(
            f"a 1.4x scale inflation did not fail both gates (chi-square "
            f"passed={inflated.chi2_passed}, A-D passed={inflated.ad_passed}); "
            f"the regression gate cannot see a distribution this different"
        )


# Tier membership. BOTH runs in the full tier and in --quick; FULL and QUICK
# restrict a check to one tier. No check currently uses them: every gate,
# criterion 3's 100-run ensemble included, fits inside the < 60 s quick
# budget, so both tiers run the same set and a green means the same thing
# either way. The mechanism is kept for the check that eventually does not
# fit, and run_checks states on its first line which case holds.
TIER_BOTH = "both"
TIER_FULL = "full"
TIER_QUICK = "quick"

_CHECKS = [
    ("V001", TIER_BOTH, "two-body double-run SHA-256 bit-identity", _check_v001),
    ("V002", TIER_BOTH, "minor-version-forward read (v1.999 file with one added channel)", _check_v002),
    ("V003", TIER_BOTH, "major-version mismatch rejected (v2.0 file)", _check_v003),
    ("V004", TIER_BOTH, "corrupted header rejected (bad magic; truncated JSON)", _check_v004),
    ("V005", TIER_BOTH, "CSV export round-trips every value bit-exactly", _check_v005),
    ("V006", TIER_BOTH, "RNG stream reproducibility", _check_v006),
    ("V007", TIER_BOTH, "load() smoke: shapes, dtypes, monotonic t_s, header fields", _check_v007),
    ("V008", TIER_BOTH, "truncated trailing record rejected", _check_v008),
    ("V009", TIER_BOTH, "UTC->TAI->TT golden epochs bit-exact with round trip", _check_v009),
    ("V010", TIER_BOTH, "quat<->DCM<->Euler round trips over 100 seeded attitudes", _check_v010),
    ("V011", TIER_BOTH, "GCRF->ITRF at golden epoch vs ERFA elements + orthonormality", _check_v011),
    ("V012", TIER_BOTH, "SREPH loader + Chebyshev evaluator on synthesized file", _check_v012),
    ("V013", TIER_BOTH, "two-body invariant drift and apsis events (quick tier)", _check_v013),
    ("V014", TIER_BOTH, "gravity tiers vs closed-form J2 on a synthesized field", _check_v014),
    ("V015", TIER_BOTH, "Battin third body vs extended-precision references", _check_v015),
    ("V016", TIER_BOTH, "conical shadow exact 0/1, penumbra value, umbra SRP zero", _check_v016),
    ("V017", TIER_BOTH, "USSA76/Harris-Priester/Mars density spot values", _check_v017),
    ("V018", TIER_BOTH, "perturbed-run double-run SHA-256 bit-identity (rkf78 + rk4)", _check_v018),
    ("V019", TIER_BOTH, "NPZ export round-trips bit-exactly (+ Parquet when available)", _check_v019),
    ("V020", TIER_BOTH, "viewer HTML self-contained, epochs exact, decimation bound held", _check_v020),
    ("V021", TIER_BOTH, "plot arrays match closed-form elements; headless PNG render", _check_v021),
    ("V022", TIER_BOTH, "P6 EC-4: stepped and batch log hashes identical; observe() pure", _check_v022),
    ("V023", TIER_BOTH, "P6 EC-5: v1.2 channels leave major at 1; oracle read from header", _check_v023),
    ("V024", TIER_BOTH, "package, compiled core, and log header report one version", _check_v024),
    ("V025", TIER_BOTH, "P6 EC-2: Python PD law reproduces core torques < 1e-9 N*m, scenario non-degenerate", _check_v025),
    ("V026", TIER_BOTH, "P6 EC-9: logged Sun direction vs independent aberration + DCM < 1e-5 mas", _check_v026),
    ("V027", TIER_BOTH, "P6 EC-3 (R=100): ensemble NEES/NIS gates + bit-identical rerun", _check_v027),
    ("V028", TIER_BOTH, "P7 EC-1: seeded MC sweep reproduces each entry's log hash via star run", _check_v028),
    ("V029", TIER_BOTH, "P7 EC-2: MC ensemble stats match frozen golden (chi-square + A-D 99 %)", _check_v029),
]


def checks_for_tier(quick: bool):
    """The (id, title, fn) triples that run in the requested tier.

    Kept separate from ``run_checks`` so a caller - the CLI contract test
    among them - can ask what a tier contains without executing it.
    """
    wanted = TIER_QUICK if quick else TIER_FULL
    return [
        (check_id, title, fn)
        for check_id, tier, title, fn in _CHECKS
        if tier in (TIER_BOTH, wanted)
    ]


def run_checks(quick: bool = False) -> int:
    """Run the requested tier's checks, print one line each, return the exit code.

    Both tiers currently run every registered check, criterion 3's 100-run
    ensemble included, because every gate fits inside the < 60 s quick
    budget.

    Whether that holds is stated rather than left to be inferred. A quick
    tier that silently covered part of a criterion is how criterion 4's
    blind spot survived a whole phase, so the first line always reports the
    tier and either names the registered checks this tier is leaving out or
    says plainly that it leaves out none.
    """
    selected = checks_for_tier(quick)
    tier_name = TIER_QUICK if quick else TIER_FULL
    ran = {check_id for check_id, _title, _fn in selected}
    skipped = sorted(
        check_id for check_id, _tier, _title, _fn in _CHECKS if check_id not in ran
    )
    coverage = (
        f"not run in this tier: {', '.join(skipped)}"
        if skipped
        else "every registered check runs in this tier"
    )
    print(
        f"VERIFY: tier {tier_name} ({len(selected)} checks; {coverage})",
        flush=True,
    )
    ctx: dict = {}
    results: list[tuple[str, bool]] = []
    for check_id, title, fn in selected:
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
