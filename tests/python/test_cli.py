"""CLI exit-code and output-contract tests via ``python -m star_reacher``
subprocesses (D-4, DX-1, DX-2).

Every test here runs with or without the compiled core: tests whose expected
behavior legitimately differs (``star run`` on a valid mission) assert the
correct branch for whichever environment they find, so the same suite is
green on a core-less checkout and at orchestrator integration.
"""

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import star_reacher
from star_reacher import _fixtures

REPO_ROOT = Path(__file__).resolve().parents[2]


def _core_available() -> bool:
    return importlib.util.find_spec("star_reacher._core") is not None


def _run_cli(*args, cwd=None):
    env = os.environ.copy()
    # Point the subprocess at the same package this test process imported
    # (source tree or installed wheel), so both see identical code.
    pkg_parent = Path(star_reacher.__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(pkg_parent) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "star_reacher", *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd,
    )


BROKEN_MISSION = """\
schema_version = 1

[mission]
name = "broken"
epoch_utc = "2026-01-01T00:00:00Z"
duration_s = 600.0
bogus_key = 1

[run]
seed = 1

[integrator]
type = "rk4"

[initial_state.cartesian]
r_m = [6778137.0, 0.0, 0.0]
v_mps = [0.0, 7668.6, 0.0]
frame = "GCRF"

[initial_state.keplerian]
sma_m = 6778137.0
ecc = 0.0
inc_deg = 0.0
raan_deg = 0.0
argp_deg = 0.0
ta_deg = 0.0

[environment]
central_body = "earth"
"""


def test_no_arguments_exits_2():
    proc = _run_cli()
    assert proc.returncode == 2


def test_unknown_subcommand_exits_2():
    proc = _run_cli("orbitize")
    assert proc.returncode == 2


def test_run_broken_mission_reports_all_errors_and_exits_2(tmp_path):
    mission = tmp_path / "broken.toml"
    mission.write_text(BROKEN_MISSION, encoding="utf-8")
    proc = _run_cli("run", str(mission))
    assert proc.returncode == 2
    # All three defect classes surface together in one pass (DX-2).
    assert "bogus_key" in proc.stderr
    assert "dt_s" in proc.stderr and "missing required" in proc.stderr
    assert "exactly one initial-state form" in proc.stderr
    assert "No default applied; run aborted." in proc.stderr


def test_run_valid_mission(tmp_path):
    outdir = tmp_path / "out"
    proc = _run_cli(
        "run",
        str(REPO_ROOT / "missions" / "twobody_leo.toml"),
        "-o",
        str(outdir),
        cwd=str(REPO_ROOT),
    )
    if _core_available():
        assert proc.returncode == 0, proc.stderr
        assert (outdir / "run.srlog").exists()
        assert (outdir / "resolved_config.json").exists()
        assert (outdir / "meta.json").exists()
        assert re.search(r"run\.srlog sha256: [0-9a-f]{64}", proc.stdout)
        # Refuse-overwrite contract: a second run without --force must fail.
        again = _run_cli(
            "run",
            str(REPO_ROOT / "missions" / "twobody_leo.toml"),
            "-o",
            str(outdir),
            cwd=str(REPO_ROOT),
        )
        assert again.returncode == 1
        assert "--force" in again.stderr
        forced = _run_cli(
            "run",
            str(REPO_ROOT / "missions" / "twobody_leo.toml"),
            "-o",
            str(outdir),
            "--force",
            cwd=str(REPO_ROOT),
        )
        assert forced.returncode == 0, forced.stderr
    else:
        # Without the compiled core the mission must still validate first,
        # then fail at the lazy import with the actionable build hint.
        assert proc.returncode == 1
        assert "pip install ." in proc.stderr
        assert "star_reacher._core" in proc.stderr


def test_run_refuses_overwrite_without_force(tmp_path):
    # The overwrite check precedes the lazy core import, so this contract
    # holds (and is testable) with or without the compiled core.
    outdir = tmp_path / "out"
    outdir.mkdir()
    (outdir / "run.srlog").write_bytes(b"existing")
    proc = _run_cli(
        "run",
        str(REPO_ROOT / "missions" / "twobody_leo.toml"),
        "-o",
        str(outdir),
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 1
    assert "--force" in proc.stderr
    # The pre-existing file was not touched.
    assert (outdir / "run.srlog").read_bytes() == b"existing"


def test_export_without_csv_flag_exits_2(tmp_path):
    log = tmp_path / "run.srlog"
    log.write_bytes(_fixtures.build_srlog(_fixtures.contract_header(), []))
    proc = _run_cli("export", str(log))
    assert proc.returncode == 2
    assert "--csv" in proc.stderr


def test_export_missing_file_exits_1(tmp_path):
    proc = _run_cli("export", "--csv", str(tmp_path / "absent.srlog"))
    assert proc.returncode == 1
    assert "no such file" in proc.stderr


def test_export_corrupt_file_exits_1(tmp_path):
    log = tmp_path / "run.srlog"
    log.write_bytes(b"NOTSRLOG" + b"\x00" * 32)
    proc = _run_cli("export", "--csv", str(log))
    assert proc.returncode == 1
    assert "magic" in proc.stderr


def test_export_major_version_mismatch_exits_1(tmp_path):
    log = tmp_path / "run.srlog"
    log.write_bytes(_fixtures.build_srlog(_fixtures.contract_header(major=2), []))
    proc = _run_cli("export", "--csv", str(log))
    assert proc.returncode == 1
    assert "major version" in proc.stderr


def test_export_synthesized_log_exits_0_and_writes_csvs(tmp_path):
    log = tmp_path / "run.srlog"
    records = [
        (0, (0.0, (1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1.5)),
        _fixtures.event_record(0.0, 1, "run_start"),
    ]
    log.write_bytes(_fixtures.build_srlog(_fixtures.contract_header(), records))
    outdir = tmp_path / "csv"
    proc = _run_cli("export", "--csv", str(log), "-o", str(outdir))
    assert proc.returncode == 0, proc.stderr
    assert (outdir / "truth.csv").exists()
    assert (outdir / "events.csv").exists()


def test_verify_output_contract(tmp_path):
    proc = _run_cli("verify", "--quick", cwd=str(tmp_path))
    lines = proc.stdout.strip().splitlines()
    # One line per check, each starting with its ID and PASS/FAIL.
    for n in range(1, 14):
        check_id = f"V{n:03d}"
        assert any(re.match(rf"{check_id} (PASS|FAIL) ", ln) for ln in lines), (
            f"missing line for {check_id}:\n{proc.stdout}"
        )
    final = lines[-1]
    assert re.fullmatch(
        r"VERIFY: PASS \(13/13\)|VERIFY: FAIL \((\d|1[0-2])/13\) failing: V\d{3}(, V\d{3})*",
        final,
    ), final
    # Exit code agrees with the verdict line.
    if final.startswith("VERIFY: PASS"):
        assert proc.returncode == 0
    else:
        assert proc.returncode != 0
        # Core-dependent checks FAIL (never skip) with the actionable hint.
        if not _core_available():
            assert "pip install ." in proc.stdout


def test_verify_format_checks_pass_without_core(tmp_path):
    # V002-V005 and V008 are pure format-conformance checks; they must pass
    # in every environment, compiled core or not.
    proc = _run_cli("verify", cwd=str(tmp_path))
    for check_id in ("V002", "V003", "V004", "V005", "V008"):
        assert re.search(rf"^{check_id} PASS ", proc.stdout, re.MULTILINE), (
            f"{check_id} did not pass:\n{proc.stdout}"
        )


def test_docs_mutually_exclusive_flags_exit_2():
    proc = _run_cli("docs", "--mathlib-only", "--report-only")
    assert proc.returncode == 2


def test_verify_help_documents_quick_equals_full():
    proc = _run_cli("verify", "--help")
    assert proc.returncode == 0
    assert "identical" in proc.stdout
