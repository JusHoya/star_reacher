"""Self-contained acceptance runner behind ``star verify`` (DX-5, FR-22 subset).

Deliberately not pytest: pytest is a dev-only dependency, and verification is
the documented first command on a bare wheel install, so it can depend on
nothing beyond the package itself. Fixture bytes are synthesized in memory
via ``star_reacher._fixtures`` (binary fixtures are never committed).

Checks that need the compiled core import it lazily and FAIL, never skip,
when it is missing: a verification pass must mean the whole surface works.
"""

from __future__ import annotations

import math
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


_CHECKS = [
    ("V001", "two-body double-run SHA-256 bit-identity", _check_v001),
    ("V002", "minor-version-forward read (v1.999 file with one added channel)", _check_v002),
    ("V003", "major-version mismatch rejected (v2.0 file)", _check_v003),
    ("V004", "corrupted header rejected (bad magic; truncated JSON)", _check_v004),
    ("V005", "CSV export round-trips every value bit-exactly", _check_v005),
    ("V006", "RNG stream reproducibility", _check_v006),
    ("V007", "load() smoke: shapes, dtypes, monotonic t_s, header fields", _check_v007),
    ("V008", "truncated trailing record rejected", _check_v008),
]


def run_checks(quick: bool = False) -> int:
    """Run every acceptance check, print one line each, return the exit code.

    ``quick`` is accepted for CLI symmetry: in Phase 1 the quick tier and the
    full tier run the identical check set; the split becomes meaningful when
    later phases add long-running checks.
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
