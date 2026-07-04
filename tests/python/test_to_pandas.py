"""Run.to_pandas() smoke tests (FR-17, D-12: loader returns NumPy, pandas is
an optional extra behind a documented, actionable ImportError).

pandas-dependent tests ``importorskip``; CI installs the pandas extra on at
least one leg (wired by the Phase 5 perf/CI workstream). The
missing-dependency path is tested by poisoning ``sys.modules`` so it runs
everywhere, pandas installed or not.
"""

import sys

import numpy as np
import pytest

from star_reacher import _fixtures, load


def _synthetic_run(tmp_path):
    header = _fixtures.contract_header()
    records = [
        _fixtures.truth_record(0.0),
        _fixtures.truth_record(0.1, (6778000.0, 100.0, -50.0), (1.0, 7668.0, 2.0)),
        _fixtures.event_record(0.0, 1, "run_start"),
        _fixtures.event_record(0.5, 2, "run_end"),
    ]
    path = tmp_path / "run.srlog"
    path.write_bytes(_fixtures.build_srlog(header, records))
    return load(path)


def test_to_pandas_frames_match_arrays(tmp_path):
    pd = pytest.importorskip("pandas")
    run = _synthetic_run(tmp_path)
    frames = run.to_pandas()
    assert set(frames) == set(run.groups)
    truth = run.groups["truth"]
    df = frames["truth"]
    assert isinstance(df, pd.DataFrame)
    # Column names follow the CSV flattening convention exactly.
    assert list(df.columns) == [
        "t_s",
        "r_m_0",
        "r_m_1",
        "r_m_2",
        "v_mps_0",
        "v_mps_1",
        "v_mps_2",
        "q_i2b_0",
        "q_i2b_1",
        "q_i2b_2",
        "q_i2b_3",
        "w_b_radps_0",
        "w_b_radps_1",
        "w_b_radps_2",
        "mass_kg",
    ]
    assert len(df) == len(truth)
    assert df["t_s"].to_numpy().tobytes() == truth["t_s"].tobytes()
    for i in range(3):
        assert (
            df[f"v_mps_{i}"].to_numpy().tobytes()
            == np.ascontiguousarray(truth["v_mps"][:, i]).tobytes()
        )
    edf = frames["events"]
    assert list(edf.columns) == ["t_s", "code", "detail"]
    assert list(edf["detail"]) == ["run_start", "run_end"]
    assert edf["code"].to_numpy().dtype == np.uint32


def test_to_pandas_missing_pandas_raises_actionable_importerror(tmp_path, monkeypatch):
    run = _synthetic_run(tmp_path)
    # A None entry makes ``import pandas`` raise ImportError without
    # touching the installed environment.
    monkeypatch.setitem(sys.modules, "pandas", None)
    with pytest.raises(ImportError, match=r"star-reacher\[pandas\]"):
        run.to_pandas()
