"""Parquet exporter tests (FR-17, D-13, Phase 5 exit criterion 3: "exported
Parquet loads in pandas with matching row counts/columns").

pyarrow and pandas are documented optional extras, so these tests
``importorskip`` them; CI installs the extras on at least one leg (wired by
the Phase 5 perf/CI workstream), which is where this gate is binding. The
missing-dependency error path is tested without uninstalling anything by
poisoning ``sys.modules``.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from star_reacher import _fixtures, load
from star_reacher.export import export_parquet

pa = pytest.importorskip("pyarrow")
pd = pytest.importorskip("pandas")

_CORE_MISSING_MESSAGE = (
    "star_reacher._core is not built in this environment. The real-log "
    "Parquet test requires the compiled core: build and install it with "
    "'pip install .' from the repository root. This failure is expected on a "
    "core-less checkout and must be green at integration/CI."
)


def _synthetic_srlog(tmp_path: Path) -> Path:
    header = _fixtures.contract_header(minor=1, force_sources=["gravity"], forces_rate_hz=1)
    records = [
        _fixtures.truth_record(0.0),
        _fixtures.truth_record(0.1, (6778000.0, 100.0, -50.0), (1.0, 7668.0, 2.0)),
        (_fixtures.group_index(header, "forces"), (0.0, (1.0, -2.0, 3.0), (0.5, 0.25, -0.125))),
        _fixtures.event_record(0.0, 1, "run_start"),
        _fixtures.event_record(0.5, 7, 'comma, "quote" and\nnewline'),
    ]
    path = tmp_path / "run.srlog"
    path.write_bytes(_fixtures.build_srlog(header, records))
    return path


def _expected_columns(arr: np.ndarray) -> list[str]:
    columns = []
    for fname in arr.dtype.names:
        shape = arr.dtype[fname].shape
        if shape:
            columns.extend(f"{fname}_{i}" for i in range(shape[0]))
        else:
            columns.append(fname)
    return columns


def test_parquet_loads_in_pandas_with_matching_rows_and_columns(tmp_path):
    path = _synthetic_srlog(tmp_path)
    run = load(path)
    written = export_parquet(path, tmp_path)
    assert sorted(p.name for p in written) == ["events.parquet", "forces.parquet", "truth.parquet"]
    for parquet_path in written:
        arr = run.groups[parquet_path.stem]
        df = pd.read_parquet(parquet_path)
        assert len(df) == len(arr)
        assert list(df.columns) == _expected_columns(arr)


def test_parquet_float_columns_are_bit_exact_and_types_survive(tmp_path):
    path = _synthetic_srlog(tmp_path)
    run = load(path)
    export_parquet(path, tmp_path)
    truth = run.groups["truth"]
    df = pd.read_parquet(tmp_path / "truth.parquet")
    assert df["t_s"].to_numpy().tobytes() == truth["t_s"].tobytes()
    for i in range(3):
        col = df[f"r_m_{i}"].to_numpy()
        assert col.dtype == np.float64
        assert col.tobytes() == np.ascontiguousarray(truth["r_m"][:, i]).tobytes()
    events = run.groups["events"]
    edf = pd.read_parquet(tmp_path / "events.parquet")
    # u32 codes stay unsigned integers and str16 details stay exact strings,
    # including embedded quotes and newlines.
    assert edf["code"].to_numpy().dtype == np.uint32
    assert list(edf["code"]) == [int(c) for c in events["code"]]
    assert list(edf["detail"]) == [str(d) for d in events["detail"]]


def test_parquet_default_outdir_is_input_parent(tmp_path):
    path = _synthetic_srlog(tmp_path)
    written = export_parquet(path)
    assert all(p.parent == tmp_path for p in written)


def test_parquet_missing_pyarrow_raises_actionable_importerror(tmp_path, monkeypatch):
    path = _synthetic_srlog(tmp_path)
    # A None entry makes ``import pyarrow`` raise ImportError without
    # touching the installed environment.
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    with pytest.raises(ImportError, match=r"star-reacher\[parquet\]"):
        export_parquet(path, tmp_path)


def test_parquet_real_generated_log(tmp_path):
    # The exit-criterion wording on the production path: a core-written log
    # exports to Parquet that pandas loads with matching rows and columns.
    try:
        import star_reacher._core  # noqa: F401
    except ImportError:
        pytest.fail(_CORE_MISSING_MESSAGE)
    from star_reacher.runner import run_mission

    mission = tmp_path / "mission.toml"
    mission.write_text(
        """\
schema_version = 1

[mission]
name = "parquet-export"
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
    written = export_parquet(result.srlog_path, tmp_path)
    for parquet_path in written:
        arr = run.groups[parquet_path.stem]
        df = pd.read_parquet(parquet_path)
        assert len(df) == len(arr)
        assert list(df.columns) == _expected_columns(arr)
    tdf = pd.read_parquet(tmp_path / "truth.parquet")
    assert len(tdf) == 601
