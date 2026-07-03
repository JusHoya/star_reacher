"""Run GMAT case 1 (leo-gravity-8x8) and freeze its truth CSV.

Maintainer-side only (D-15): drives the portable GMAT R2026a console install
at C:/Users/hoyer/WorkSpace/tools/gmat/ against the committed script
``tests/golden/crosstool/gmat_leo_gravity_8x8.script`` and converts the
16-significant-digit report into the committed frozen truth
``tests/golden/crosstool/truth_gmat_leo_gravity_8x8.csv``. CI never runs
this; CI consumes only the committed CSV.

Steps:

1. Write the startup-file override ``gmat_startup_zeroeop.txt`` next to the
   committed script: byte-identical to the stock ``bin/gmat_startup_file.txt``
   except EOP_FILE, which points at the committed zeroed-EOP file (the
   controlled-comparison configuration; see gen_zero_eop.py).
2. Invoke GmatConsole.exe with CWD = bin (its relative-path convention),
   asserting the console success banner and exit code 0.
3. Parse the report, require one row per 60 s grid point (10081 rows,
   duplicate epochs at the stop boundary deduplicated), require the reported
   ElapsedSecs to sit on the exact grid within 5e-6 s (fixed-step rows;
   the measured deviation is printed and recorded in the manifest), convert
   km -> m, and write the truth CSV with exact grid times.

Run from the repo root:  python scripts/crosstool/run_gmat_case1.py
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CROSSTOOL = REPO_ROOT / "tests" / "golden" / "crosstool"

GMAT_BIN = Path(r"C:\Users\hoyer\WorkSpace\tools\gmat\bin")
RUN_DIR = Path(r"C:\Users\hoyer\WorkSpace\tools\crosstool-runs")

SCRIPT = CROSSTOOL / "gmat_leo_gravity_8x8.script"
STARTUP_OUT = CROSSTOOL / "gmat_startup_zeroeop.txt"
REPORT = RUN_DIR / "gmat_leo_gravity_8x8_report.txt"
LOG = RUN_DIR / "gmat_leo_gravity_8x8_log.txt"
TRUTH_OUT = CROSSTOOL / "truth_gmat_leo_gravity_8x8.csv"

STEP_S = 60.0
DURATION_S = 604800.0
N_ROWS = int(DURATION_S / STEP_S) + 1  # 10081, t = 0 included
GRID_TOL_S = 5e-6  # 5 us * 7.5 km/s ~ 4 cm, negligible against the 10 m gate


def write_startup() -> None:
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
    STARTUP_OUT.write_text("\n".join(out_lines) + "\n", newline="\n", encoding="ascii")
    print(f"startup override: {STARTUP_OUT.name}")
    print(f"  stock startup sha256 {hashlib.sha256(stock.read_bytes()).hexdigest()}")
    print(f"  override sha256 {hashlib.sha256(STARTUP_OUT.read_bytes()).hexdigest()}")


def run_gmat() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(GMAT_BIN / "GmatConsole.exe"),
        "--run",
        str(SCRIPT),
        "--startup_file",
        str(STARTUP_OUT),
        "--logfile",
        str(LOG),
    ]
    print("command:", " ".join(cmd))
    proc = subprocess.run(
        cmd, cwd=GMAT_BIN, capture_output=True, text=True, timeout=1800
    )
    print(proc.stdout[-2000:])
    assert proc.returncode == 0, f"GmatConsole exit code {proc.returncode}: {proc.stderr[-2000:]}"
    assert "successful" in proc.stdout, "GMAT success banner missing"


def convert_report() -> None:
    lines = REPORT.read_text(encoding="ascii").splitlines()
    assert lines[0].lstrip().startswith("sat.ElapsedSecs"), "unexpected report header"
    rows: list[tuple[float, ...]] = []
    for line in lines[1:]:
        vals = tuple(float(tok) for tok in line.split())
        assert len(vals) == 8, f"unexpected column count: {line!r}"
        # The stop-condition refinement can re-emit the final epoch; keep the
        # first occurrence of each epoch only.
        if rows and vals[0] <= rows[-1][0]:
            assert abs(vals[0] - rows[-1][0]) < GRID_TOL_S, "non-monotonic report epochs"
            continue
        rows.append(vals)
    assert len(rows) == N_ROWS, f"expected {N_ROWS} rows, parsed {len(rows)}"

    worst = 0.0
    out = ["# Frozen GMAT R2026a truth for missions/leo_gravity_8x8.toml"]
    out.append("# (cross-tool case 1, Phase 3 exit criterion 5, D-15). GCRF/EarthICRF")
    out.append("# Cartesian state on the exact 60 s grid; provenance, tool versions,")
    out.append("# command line, and configuration hashes in manifest.toml.")
    out.append("t_s,x_m,y_m,z_m,vx_mps,vy_mps,vz_mps")
    for k, vals in enumerate(rows):
        t_nominal = STEP_S * k
        worst = max(worst, abs(vals[0] - t_nominal))
        xyz = [v * 1000.0 for v in vals[2:8]]  # km -> m, exact scaling
        out.append(
            "%r,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e" % (t_nominal, *xyz)
        )
    assert worst < GRID_TOL_S, f"report epochs off the 60 s grid by {worst} s"
    TRUTH_OUT.write_text("\n".join(out) + "\n", newline="\n", encoding="ascii")
    print(f"grid check: max |ElapsedSecs - 60k| = {worst:.3e} s over {len(rows)} rows")
    print(f"first row UTCModJulian = {rows[0][1]!r} (expect 31041.5 = 2026-01-01T00:00:00 UTC)")
    print(f"{TRUTH_OUT.name}: {TRUTH_OUT.stat().st_size} bytes")
    print(f"  sha256 {hashlib.sha256(TRUTH_OUT.read_bytes()).hexdigest()}")
    print(f"  report sha256 {hashlib.sha256(REPORT.read_bytes()).hexdigest()}")
    print(f"  script sha256 {hashlib.sha256(SCRIPT.read_bytes()).hexdigest()}")


def main() -> None:
    write_startup()
    run_gmat()
    convert_report()


if __name__ == "__main__":
    main()
