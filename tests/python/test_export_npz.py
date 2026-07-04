"""NPZ exporter bit-exact round-trip tests (FR-17, D-13, Phase 5 exit
criterion 3: "NPZ round-trips exactly").

Bit-exact means: the header dict is reproduced exactly, every numeric channel
reproduces its IEEE-754 bytes (compared via ``tobytes``), and every string
channel reproduces its decoded values. The synthetic-log tests need no
compiled core; the real-log test requires it and fails (never skips) when it
is absent, matching the test_integration_core.py convention.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from star_reacher import _fixtures, load
from star_reacher.export import NpzFormatError, export_npz, load_npz, write_npz

_CORE_MISSING_MESSAGE = (
    "star_reacher._core is not built in this environment. The real-log NPZ "
    "round-trip test requires the compiled core: build and install it with "
    "'pip install .' from the repository root. This failure is expected on a "
    "core-less checkout and must be green at integration/CI."
)

# Shortest-repr corner cases shared with the CSV round-trip suite: subnormal
# minimum, negative zero, non-terminating binary fractions, and large
# magnitudes with full 17-significant-digit reprs.
TRICKY = [
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


def _tricky_srlog(tmp_path: Path) -> Path:
    """A v1.1-shaped log: truth + events + forces/mass/env vehicle groups."""
    header = _fixtures.contract_header(
        minor=1,
        force_sources=["gravity", "srp"],
        forces_rate_hz=1,
        mass_rate_hz=1,
        env_rate_hz=1,
    )
    n = len(TRICKY)
    records = []
    for i in range(n):
        records.append(
            _fixtures.truth_record(
                float(i),
                (TRICKY[(i + 1) % n], TRICKY[(i + 2) % n], TRICKY[(i + 3) % n]),
                (TRICKY[(i + 4) % n], TRICKY[(i + 5) % n], TRICKY[(i + 6) % n]),
                (1.0, -0.0, 0.0, 0.0),
                (TRICKY[(i + 7) % n], TRICKY[(i + 8) % n], TRICKY[i]),
                TRICKY[(i + 2) % n],
            )
        )
    forces_gi = _fixtures.group_index(header, "forces")
    mass_gi = _fixtures.group_index(header, "mass")
    env_gi = _fixtures.group_index(header, "env")
    records.append(
        (forces_gi, (0.0, (1.0, -2.0, 3.0), (0.5, 0.25, -0.125), (-0.0, 0.1, 5e-324), (0.0, 0.0, 0.0)))
    )
    records.append((mass_gi, (0.0, 1.5, (0.01, -0.02, 0.03), (1.0, 0.0, 0.0, 2.0, 0.0, 3.0))))
    records.append((env_gi, (0.0, 400e3, 0.0, 0.0, 1e-12, 0.5)))
    records.append(_fixtures.event_record(0.0, 1, "run_start"))
    records.append(_fixtures.event_record(0.5, 7, 'comma, "quote" and\nnewline'))
    records.append(_fixtures.event_record(600.0, 2, "run_end"))
    path = tmp_path / "run.srlog"
    path.write_bytes(_fixtures.build_srlog(header, records))
    return path


def assert_runs_equal(got, want):
    assert got.header == want.header
    assert list(got.groups) == list(want.groups)
    for gname, arr in want.groups.items():
        back = got.groups[gname]
        assert back.dtype == arr.dtype, gname
        assert len(back) == len(arr), gname
        for fname in arr.dtype.names:
            if arr.dtype[fname].base.kind == "O":
                assert list(back[fname]) == list(arr[fname]), f"{gname}.{fname}"
            else:
                # Bit-exactness, not closeness: -0.0 and subnormals must
                # survive with their exact IEEE-754 bytes.
                assert back[fname].tobytes() == arr[fname].tobytes(), f"{gname}.{fname}"
    assert got.events.dtype == want.events.dtype
    assert len(got.events) == len(want.events)


def test_npz_round_trip_is_bit_exact(tmp_path):
    path = _tricky_srlog(tmp_path)
    run = load(path)
    npz_path = export_npz(path, tmp_path)
    assert npz_path == tmp_path / "run.npz"
    assert_runs_equal(load_npz(npz_path), run)


def test_npz_contains_no_pickle(tmp_path):
    # np.load defaults to allow_pickle=False and raises on any pickled
    # member; touching every array under the default proves the archive is
    # loadable by any NumPy without code-execution risk (npz_v1.md).
    path = _tricky_srlog(tmp_path)
    npz_path = export_npz(path, tmp_path)
    with np.load(npz_path) as npz:
        for key in npz.files:
            npz[key]


def test_npz_default_outdir_is_input_parent(tmp_path):
    path = _tricky_srlog(tmp_path)
    npz_path = export_npz(path)
    assert npz_path == tmp_path / "run.npz"
    assert npz_path.exists()


def test_npz_events_only_and_empty_groups(tmp_path):
    # A log with zero truth records and only events exercises the
    # empty-fixed-array and string-decomposition paths together.
    header = _fixtures.contract_header()
    records = [_fixtures.event_record(0.0, 1, "run_start")]
    path = tmp_path / "run.srlog"
    path.write_bytes(_fixtures.build_srlog(header, records))
    run = load(path)
    assert len(run.groups["truth"]) == 0
    back = load_npz(export_npz(path, tmp_path))
    assert_runs_equal(back, run)
    assert list(back.events["detail"]) == ["run_start"]


def test_npz_declared_but_empty_events_group(tmp_path):
    # An events group declared in the header with zero records exercises the
    # zero-row string-channel encoding (empty offsets/utf8 arrays).
    header = _fixtures.contract_header()
    path = tmp_path / "run.srlog"
    path.write_bytes(_fixtures.build_srlog(header, [_fixtures.truth_record(0.0)]))
    run = load(path)
    assert len(run.groups["events"]) == 0
    back = load_npz(export_npz(path, tmp_path))
    assert_runs_equal(back, run)


def test_npz_run_without_events_group(tmp_path):
    # write_npz accepts any Run; a groups dict with no events group must
    # come back with the standard empty events array (mirrors load()).
    run = load(_tricky_srlog(tmp_path))
    from star_reacher.srlog import Run

    partial = Run(header=run.header, groups={"truth": run.groups["truth"]}, events=run.events)
    npz_path = write_npz(partial, tmp_path / "partial.npz")
    back = load_npz(npz_path)
    assert list(back.groups) == ["truth"]
    assert len(back.events) == 0
    assert set(back.events.dtype.names) == {"t_s", "code", "detail"}


def test_npz_rejects_foreign_archive(tmp_path):
    foreign = tmp_path / "foreign.npz"
    np.savez(foreign, some_array=np.arange(3))
    with pytest.raises(NpzFormatError, match="not a star_reacher NPZ archive"):
        load_npz(foreign)


def test_npz_rejects_unknown_layout_version(tmp_path):
    bogus = tmp_path / "bogus.npz"
    np.savez(bogus, srnpz_layout=np.array("999"), srlog_header_json=np.array("{}"))
    with pytest.raises(NpzFormatError, match="layout version '999'"):
        load_npz(bogus)


def test_npz_refuses_non_string_object_values(tmp_path):
    run = load(_tricky_srlog(tmp_path))
    run.events["detail"][0] = 42  # simulate a corrupted object column
    with pytest.raises(TypeError, match="not str"):
        write_npz(run, tmp_path / "bad.npz")


def test_npz_round_trip_on_real_generated_log(tmp_path):
    # Phase 5 exit criterion 3 on the real production path: a log written by
    # the compiled core round-trips bit-exactly through NPZ.
    try:
        import star_reacher._core  # noqa: F401
    except ImportError:
        pytest.fail(_CORE_MISSING_MESSAGE)
    from star_reacher.runner import run_mission

    mission = tmp_path / "mission.toml"
    # 60 s two-body LEO at 10 Hz: 601 truth records, small and fast, same
    # shape as the verify V001 mission.
    mission.write_text(
        """\
schema_version = 1

[mission]
name = "npz-roundtrip"
epoch_utc = "2026-01-01T00:00:00Z"
duration_s = 60.0

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
""",
        encoding="utf-8",
    )
    result = run_mission(mission, tmp_path / "out")
    run = load(result.srlog_path)
    back = load_npz(export_npz(result.srlog_path, tmp_path))
    assert_runs_equal(back, run)
    assert len(back.groups["truth"]) == 601
    # The derived-element path works identically on the NPZ-loaded Run.
    np.testing.assert_array_equal(back.elements()["a_m"], run.elements()["a_m"])
