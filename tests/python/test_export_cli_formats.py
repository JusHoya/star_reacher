"""``star export`` multi-format flag semantics (D-4, DX-2, Phase 5).

These call ``star_reacher.cli.main`` in-process (rather than a subprocess as
test_cli.py does) so the missing-pyarrow path can be simulated by poisoning
``sys.modules``; exit codes and stderr messages are asserted against the
DX-2 contract: 2 for usage errors, 1 for runtime/environment errors, 0 with
one "wrote <path>" line per written file on success.
"""

import sys

import pytest

from star_reacher import _fixtures
from star_reacher.cli import main


@pytest.fixture()
def srlog_path(tmp_path):
    records = [
        _fixtures.truth_record(0.0),
        _fixtures.truth_record(0.1),
        _fixtures.event_record(0.0, 1, "run_start"),
    ]
    path = tmp_path / "run.srlog"
    path.write_bytes(_fixtures.build_srlog(_fixtures.contract_header(), records))
    return path


def test_export_requires_at_least_one_format_flag(srlog_path, capsys):
    assert main(["export", str(srlog_path)]) == 2
    err = capsys.readouterr().err
    for flag in ("--csv", "--npz", "--parquet"):
        assert flag in err


def test_export_npz_flag_writes_archive(srlog_path, tmp_path, capsys):
    outdir = tmp_path / "npz_out"
    assert main(["export", "--npz", str(srlog_path), "-o", str(outdir)]) == 0
    assert (outdir / "run.npz").exists()
    assert f"wrote {outdir / 'run.npz'}" in capsys.readouterr().out


def test_export_combined_formats_in_one_invocation(srlog_path, tmp_path, capsys):
    pytest.importorskip("pyarrow")
    outdir = tmp_path / "all_out"
    code = main(
        ["export", "--csv", "--npz", "--parquet", str(srlog_path), "-o", str(outdir)]
    )
    assert code == 0
    for name in (
        "truth.csv",
        "events.csv",
        "run.npz",
        "truth.parquet",
        "events.parquet",
    ):
        assert (outdir / name).exists(), name
    out = capsys.readouterr().out
    assert out.count("wrote ") == 5


def test_export_parquet_without_pyarrow_exits_1_with_extra_hint(
    srlog_path, tmp_path, capsys, monkeypatch
):
    # A None entry makes ``import pyarrow`` raise ImportError without
    # touching the installed environment (same trick as the exporter tests).
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    code = main(["export", "--parquet", str(srlog_path), "-o", str(tmp_path / "p")])
    assert code == 1
    err = capsys.readouterr().err
    assert "star export:" in err
    assert "star-reacher[parquet]" in err


def test_export_npz_missing_file_exits_1(tmp_path, capsys):
    assert main(["export", "--npz", str(tmp_path / "absent.srlog")]) == 1
    assert "no such file" in capsys.readouterr().err


def test_export_npz_corrupt_file_exits_1(tmp_path, capsys):
    log = tmp_path / "run.srlog"
    log.write_bytes(b"NOTSRLOG" + b"\x00" * 32)
    assert main(["export", "--npz", str(log)]) == 1
    assert "magic" in capsys.readouterr().err
