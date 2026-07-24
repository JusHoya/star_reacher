"""Run a Phase 8 GMAT cross-tool case and freeze its truth CSV (D-15).

Maintainer-side only: drives the portable GMAT R2026a console install at
C:/Users/hoyer/WorkSpace/tools/gmat/ against a committed Phase 8 script under
``tests/golden/crosstool/`` and converts the 16-significant-digit report into
the committed frozen truth CSV. CI never runs this; CI consumes only the
committed CSVs, and until they exist the five Phase 8 gates skip (deferred to
docs/release_checklist.md item 10). This is the Phase 8 sibling of
run_gmat_case1.py, parameterized over the five cases.

Cases (``--case``):

  molniya        gmat_molniya.script        -> truth_gmat_molniya.csv        (7-day RMS)
  lunar_orbiter  gmat_lunar_orbiter.script  -> truth_gmat_lunar_orbiter.csv  (7-day RMS)
  mars_orbiter   gmat_mars_orbiter.script   -> truth_gmat_mars_orbiter.csv   (7-day RMS)
  translunar     gmat_translunar.script     -> truth_gmat_translunar.csv     (arrival point)
  mars_cruise    gmat_mars_cruise.script    -> truth_gmat_mars_cruise.csv    (7-day arc)

The 7-day cases emit the full 10081-row 60 s grid exactly as case 1 does; the
trans-lunar case emits a single arrival row at the simulator's SOI-transition
epoch (passed as ``--arrival-s``, the elapsed-from-MECO span 455402 s by
default - the elapsed seconds of the tli SOI event minus the 353 s MECO time).

Zero-EOP: the Earth-regime cases (molniya, translunar) reuse the committed
gmat_startup_zeroeop.txt override (the controlled-comparison configuration
matching the simulator's no-EOP convention), regenerated here from the stock
startup exactly as run_gmat_case1.py does. The Moon-, Mars-, and Sun-centred
cases use body-centred ICRF coordinate systems whose orientation the Earth EOP
does not touch, so they run against the stock startup.

Run from the repo root, e.g.:  python scripts/crosstool/run_gmat_phase8.py --case molniya
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CROSSTOOL = REPO_ROOT / "tests" / "golden" / "crosstool"

GMAT_BIN = Path(r"C:\Users\hoyer\WorkSpace\tools\gmat\bin")
RUN_DIR = Path(r"C:\Users\hoyer\WorkSpace\tools\crosstool-runs")
STARTUP_ZEROEOP = CROSSTOOL / "gmat_startup_zeroeop.txt"

STEP_S = 60.0
DURATION_S = 604800.0
N_ROWS = int(DURATION_S / STEP_S) + 1  # 10081, t = 0 included
GRID_TOL_S = 5e-6  # as run_gmat_case1.py: ~cm-level along-track, negligible

# Per case: (script, truth CSV, zero-EOP?, single arrival row?). The arrival
# case's default stop span is the tli SOI-transition elapsed-from-MECO time.
CASES = {
    "molniya": ("gmat_molniya.script", "truth_gmat_molniya.csv", True, False),
    "lunar_orbiter": (
        "gmat_lunar_orbiter.script",
        "truth_gmat_lunar_orbiter.csv",
        False,
        False,
    ),
    "mars_orbiter": (
        "gmat_mars_orbiter.script",
        "truth_gmat_mars_orbiter.csv",
        False,
        False,
    ),
    "translunar": (
        "gmat_translunar.script",
        "truth_gmat_translunar.csv",
        True,
        True,
    ),
    "mars_cruise": (
        "gmat_mars_cruise.script",
        "truth_gmat_mars_cruise.csv",
        False,
        False,
    ),
}

# Default trans-lunar arrival span (elapsed from the MECO epoch the script
# starts at): the tli SOI-transition event at 455755 s minus the 353 s MECO
# time. The maintainer overrides it with --arrival-s if the frozen simulator
# run locates a different arrival epoch.
DEFAULT_ARRIVAL_S = 455402.0


def write_startup() -> None:
    """Regenerate the zero-EOP startup override (identical to run_gmat_case1)."""
    stock = GMAT_BIN / "gmat_startup_file.txt"
    text = stock.read_text(encoding="ascii", errors="replace")
    eop = str(CROSSTOOL / "eopc04_zero.txt").replace("\\", "/")
    out_lines = []
    replaced = 0
    for line in text.splitlines():
        if line.strip().startswith("EOP_FILE"):
            out_lines.append(f"EOP_FILE               = {eop}")
            replaced += 1
        else:
            out_lines.append(line)
    assert replaced == 1, f"expected exactly one EOP_FILE line, found {replaced}"
    STARTUP_ZEROEOP.write_text(
        "\n".join(out_lines) + "\n", newline="\n", encoding="ascii"
    )
    print(f"startup override: {STARTUP_ZEROEOP.name}")


def run_gmat(script: Path, log: Path, zero_eop: bool) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [str(GMAT_BIN / "GmatConsole.exe"), "--run", str(script)]
    if zero_eop:
        cmd += ["--startup_file", str(STARTUP_ZEROEOP)]
    cmd += ["--logfile", str(log)]
    print("command:", " ".join(cmd))
    proc = subprocess.run(
        cmd, cwd=GMAT_BIN, capture_output=True, text=True, timeout=3600
    )
    print(proc.stdout[-2000:])
    assert proc.returncode == 0, (
        f"GmatConsole exit code {proc.returncode}: {proc.stderr[-2000:]}"
    )
    assert "successful" in proc.stdout, "GMAT success banner missing"


def _parse_report(report: Path) -> list[tuple[float, ...]]:
    lines = report.read_text(encoding="ascii").splitlines()
    assert lines[0].lstrip().startswith("sat.ElapsedSecs"), "unexpected report header"
    rows: list[tuple[float, ...]] = []
    for line in lines[1:]:
        vals = tuple(float(tok) for tok in line.split())
        assert len(vals) == 8, f"unexpected column count: {line!r}"
        if rows and vals[0] <= rows[-1][0]:
            assert abs(vals[0] - rows[-1][0]) < GRID_TOL_S, "non-monotonic report epochs"
            continue
        rows.append(vals)
    return rows


def convert_grid(report: Path, truth_out: Path, case: str) -> None:
    rows = _parse_report(report)
    assert len(rows) == N_ROWS, f"expected {N_ROWS} rows, parsed {len(rows)}"
    worst = 0.0
    out = [
        f"# Frozen GMAT R2026a truth for the Phase 8 {case} cross-tool case",
        "# (exit criterion 1, D-15). Body-centred ICRF Cartesian state on the",
        "# exact 60 s grid; provenance, tool versions, and configuration in",
        "# manifest.toml.",
        "t_s,x_m,y_m,z_m,vx_mps,vy_mps,vz_mps",
    ]
    for k, vals in enumerate(rows):
        t_nominal = STEP_S * k
        worst = max(worst, abs(vals[0] - t_nominal))
        xyz = [v * 1000.0 for v in vals[2:8]]  # km -> m, exact scaling
        out.append("%r,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e" % (t_nominal, *xyz))
    assert worst < GRID_TOL_S, f"report epochs off the 60 s grid by {worst} s"
    truth_out.write_text("\n".join(out) + "\n", newline="\n", encoding="ascii")
    print(f"grid check: max |ElapsedSecs - 60k| = {worst:.3e} s over {len(rows)} rows")
    _report_hashes(truth_out, report)


def convert_arrival(report: Path, truth_out: Path, arrival_s: float) -> None:
    """Emit the single arrival-epoch row for the trans-lunar point gate."""
    rows = _parse_report(report)
    # The arrival row is the last one; assert the script propagated to the
    # requested arrival span within the grid tolerance.
    last = rows[-1]
    assert abs(last[0] - arrival_s) < STEP_S, (
        f"report ends at {last[0]} s, expected the arrival span {arrival_s} s"
    )
    xyz = [v * 1000.0 for v in last[2:8]]
    out = [
        "# Frozen GMAT R2026a truth for the Phase 8 trans-lunar cross-tool case",
        "# (exit criterion 1, D-15): the single arrival-epoch EarthICRF Cartesian",
        f"# state at elapsed-from-MECO {arrival_s} s (lunar-SOI arrival). The gate",
        "# is the position difference vs the tli truth log at this epoch < 1 km.",
        "t_s,x_m,y_m,z_m,vx_mps,vy_mps,vz_mps",
        "%r,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e" % (arrival_s, *xyz),
    ]
    truth_out.write_text("\n".join(out) + "\n", newline="\n", encoding="ascii")
    print(f"arrival row at {arrival_s} s written")
    _report_hashes(truth_out, report)


def _report_hashes(truth_out: Path, report: Path) -> None:
    print(f"{truth_out.name}: {truth_out.stat().st_size} bytes")
    print(f"  sha256 {hashlib.sha256(truth_out.read_bytes()).hexdigest()}")
    print(f"  report sha256 {hashlib.sha256(report.read_bytes()).hexdigest()}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", required=True, choices=sorted(CASES))
    ap.add_argument(
        "--arrival-s",
        type=float,
        default=DEFAULT_ARRIVAL_S,
        help="trans-lunar arrival span (elapsed from MECO); ignored for the "
        "grid cases",
    )
    args = ap.parse_args()

    script_name, truth_name, zero_eop, single_row = CASES[args.case]
    script = CROSSTOOL / script_name
    truth_out = CROSSTOOL / truth_name
    report = RUN_DIR / f"gmat_{args.case}_report.txt"
    log = RUN_DIR / f"gmat_{args.case}_log.txt"
    print(f"script sha256 {hashlib.sha256(script.read_bytes()).hexdigest()}")

    if zero_eop:
        write_startup()
    run_gmat(script, log, zero_eop)
    if single_row:
        convert_arrival(report, truth_out, args.arrival_s)
    else:
        convert_grid(report, truth_out, args.case)


if __name__ == "__main__":
    main()
