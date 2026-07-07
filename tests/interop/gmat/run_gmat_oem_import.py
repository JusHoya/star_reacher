"""Scripted GMAT ephemeris import (Phase 5 exit criterion 3, D-15).

Maintainer-side only: CI never runs this. End-to-end pipeline, one command:

1. Run missions/twobody_leo.toml with the installed package (byte-
   deterministic run.srlog; the srlog and canonical-config SHA-256s from
   meta.json are printed so the transcript pins the exact simulator state).
2. Downsample the 10 Hz truth group to the 60 s grid (decimation only,
   never interpolation, mirroring FR-16's own logging rule) and write it as
   a CCSDS-OEM ephemeris, ``twobody_leo.oem``, committed next to this
   script. The 60 s stamps are asserted exactly representable: the logger
   writes t = i * dt with one rounding, and at dt = 0.1 the relative error
   of fl(0.1) is 2^-54, at most half an ULP of the integer 60k, so every
   60 s sample time is the exact integer double.
3. Write the GMAT script ``twobody_leo_oem_import.script`` next to this
   driver (committed as-run, like the crosstool startup override: the
   absolute paths GMAT requires are derived from this file's location so
   the committed script always matches the run that produced the committed
   transcript) and drive GmatConsole.exe (stock startup file -- no EOP
   enters an inertial-frame ephemeris read-back) to import the OEM through
   GMAT's CCSDS-OEM ephemeris propagator and report the state back on the
   same 60 s grid in EarthMJ2000Eq, the axes GMAT assigns the OEM's
   EME2000 REF_FRAME label (its reader rejects ICRF for Earth; see the
   comment in ``write_oem``), so the read-back involves no frame
   conversion at all.
4. Compare GMAT's reported state at every shared epoch against the OEM
   records it was fed (both sides parsed from the same text, so the
   comparison isolates GMAT's import/interpolation round trip). Lagrange
   interpolation is exact at the record epochs, so the residual measures
   only GMAT's internal epoch quantization (~1e-7 s grid deviation was
   measured for the same console build in tests/golden/crosstool), i.e.
   v * dt ~ 1e-3 m. Gates: max |dr| < 5e-3 m, max |dv| < 1e-4 m/s.
5. Print SHA-256 of every input and output so the committed transcript and
   manifest.toml pin the whole chain.

Run from the repo root with the project venv active (or via its python):

    python tests/interop/gmat/run_gmat_oem_import.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from star_reacher.srlog import load

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

GMAT_BIN = Path(r"C:\Users\hoyer\WorkSpace\tools\gmat\bin")
RUN_DIR = Path(r"C:\Users\hoyer\WorkSpace\tools\interop-runs")

MISSION = REPO_ROOT / "missions" / "twobody_leo.toml"
OUT_DIR = RUN_DIR / "twobody-leo"
OEM = HERE / "twobody_leo.oem"
SCRIPT = HERE / "twobody_leo_oem_import.script"
REPORT = RUN_DIR / "twobody_leo_oem_import_report.txt"
LOG = RUN_DIR / "twobody_leo_oem_import_log.txt"

STEP_S = 60.0
DURATION_S = 5400.0
N_RECORDS = int(DURATION_S / STEP_S) + 1  # 91, t = 0 included
# GMAT's report epochs sit on the 60 s grid to ~1e-7 s (measured for this
# console build in tests/golden/crosstool/manifest.toml item 4); at LEO
# speed that is ~1e-3 m of along-track displacement, and Lagrange
# interpolation is exact at the record epochs, so anything beyond a few
# millimeters would mean the import itself (frame, units, or epoch mapping)
# is wrong. A wrong frame or unit shows up at km scale.
GATE_DR_M = 5e-3
GATE_DV_MPS = 1e-4


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_mission() -> None:
    cmd = [
        sys.executable,
        "-m",
        "star_reacher",
        "run",
        str(MISSION),
        "--outdir",
        str(OUT_DIR),
        "--force",
    ]
    print("command:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    print(proc.stdout.strip())
    assert proc.returncode == 0, f"star run exit {proc.returncode}: {proc.stderr[-2000:]}"
    meta = json.loads((OUT_DIR / "meta.json").read_text(encoding="utf-8"))
    print(f"srlog sha256  {meta['srlog_sha256']}")
    print(f"config sha256 {meta['config_sha256']}")
    # Pin the exact simulator build in the committed transcript, not just
    # the manifest: core_git_hash is baked into the compiled core at build.
    print("versions:", json.dumps(meta["versions"], sort_keys=True))


def write_oem() -> None:
    run = load(OUT_DIR / "run.srlog")
    truth = run.groups["truth"]
    epoch_str = run.header["epoch_utc"]
    epoch = datetime.fromisoformat(epoch_str.replace("Z", "+00:00"))
    assert epoch.tzinfo is not None
    epoch = epoch.astimezone(timezone.utc)

    t = truth["t_s"]
    rate_hz = 10  # missions/twobody_leo.toml truth_rate_hz
    stride = int(round(STEP_S * rate_hz))
    idx = np.arange(0, len(t), stride)
    assert len(idx) == N_RECORDS, f"expected {N_RECORDS} samples, got {len(idx)}"
    # Exactness, not closeness: the 60 s stamps must be the exact integer
    # doubles (see module docstring), otherwise the OEM epoch strings below
    # would misstate the sample times.
    for k, i in enumerate(idx):
        assert t[i] == STEP_S * k, f"t[{i}] = {t[i]!r} is not exactly {STEP_S * k}"

    lines = [
        "CCSDS_OEM_VERS = 1.0",
        f"CREATION_DATE  = {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')}",
        "ORIGINATOR     = star_reacher",
        "",
        "META_START",
        "OBJECT_NAME          = twobody-leo",
        "OBJECT_ID            = STAR-REACHER-INTEROP",
        "CENTER_NAME          = Earth",
        # GMAT's CCSDS-OEM reader rejects REF_FRAME = ICRF for Earth
        # ("not supported for the central body"); EME2000 is the frame its
        # own shipped sample OEM uses. The mission truth frame is GCRF; the
        # GCRF-to-EME2000 frame bias (~23 mas, sub-meter at LEO radius) is
        # a labeling convention of this interop artifact, not an error term
        # in the check: GMAT reports back in the same axes it read
        # (EarthMJ2000Eq), so no frame conversion enters the round trip.
        "REF_FRAME            = EME2000",
        "TIME_SYSTEM          = UTC",
    ]

    def stamp(seconds: float) -> str:
        dt = epoch + timedelta(seconds=float(seconds))
        return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")

    lines += [
        f"START_TIME           = {stamp(0.0)}",
        f"USEABLE_START_TIME   = {stamp(0.0)}",
        f"USEABLE_STOP_TIME    = {stamp(DURATION_S)}",
        f"STOP_TIME            = {stamp(DURATION_S)}",
        "INTERPOLATION        = Lagrange",
        "INTERPOLATION_DEGREE = 7",
        "META_STOP",
        "",
    ]
    for i in idx:
        r_km = [float(x) / 1000.0 for x in truth["r_m"][i]]
        v_kmps = [float(x) / 1000.0 for x in truth["v_mps"][i]]
        vals = "  ".join("%.16e" % x for x in (*r_km, *v_kmps))
        lines.append(f"{stamp(t[i])}  {vals}")
    OEM.write_text("\n".join(lines) + "\n", newline="\n", encoding="ascii")
    print(f"{OEM.name}: {N_RECORDS} records, {OEM.stat().st_size} bytes")
    print(f"  sha256 {sha256(OEM)}")


def write_gmat_script() -> None:
    oem = str(OEM).replace("\\", "/")
    report = str(REPORT).replace("\\", "/")
    text = f"""%------------------------------------------------------------------------------
% twobody_leo_oem_import.script -- scripted GMAT ephemeris import (Phase 5
% exit criterion 3, D-15): the CCSDS-OEM ephemeris exported from the
% missions/twobody_leo.toml truth log (twobody_leo.oem) is imported through
% GMAT's CCSDS-OEM ephemeris propagator and the state is reported back on
% the shared 60 s grid in EarthMJ2000Eq -- the axes GMAT assigns the OEM's
% EME2000 REF_FRAME label -- so the read-back involves no frame conversion.
% The consuming comparison, gates, and provenance: run_gmat_oem_import.py
% and manifest.toml in this directory.
%
% Generated as-run by run_gmat_oem_import.py: it contains maintainer-machine
% absolute paths by design (D-15 runs external tools offline on the
% maintainer machine only; CI never runs this).
%------------------------------------------------------------------------------

Create Spacecraft EphSat;
GMAT EphSat.EphemerisName = '{oem}';

Create Propagator EphProp;
GMAT EphProp.Type = 'CCSDS-OEM';
GMAT EphProp.StepSize = 60;

Create ReportFile rpt;
GMAT rpt.Filename = '{report}';
GMAT rpt.Precision = 16;
GMAT rpt.WriteHeaders = true;
GMAT rpt.LeftJustify = On;
GMAT rpt.FixedWidth = true;
GMAT rpt.WriteReport = true;
GMAT rpt.Add = {{EphSat.ElapsedSecs, EphSat.UTCModJulian, EphSat.EarthMJ2000Eq.X, EphSat.EarthMJ2000Eq.Y, EphSat.EarthMJ2000Eq.Z, EphSat.EarthMJ2000Eq.VX, EphSat.EarthMJ2000Eq.VY, EphSat.EarthMJ2000Eq.VZ}};

BeginMissionSequence;

Propagate EphProp(EphSat) {{EphSat.ElapsedSecs = {DURATION_S}}};
"""
    SCRIPT.write_text(text, newline="\n", encoding="ascii")
    print(f"{SCRIPT.name}: generated as-run")
    print(f"  sha256 {sha256(SCRIPT)}")


def run_gmat() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(GMAT_BIN / "GmatConsole.exe"),
        "--run",
        str(SCRIPT),
        "--logfile",
        str(LOG),
    ]
    print("command:", " ".join(cmd))
    proc = subprocess.run(
        cmd, cwd=GMAT_BIN, capture_output=True, text=True, timeout=600
    )
    print(proc.stdout[-2000:])
    assert proc.returncode == 0, f"GmatConsole exit code {proc.returncode}: {proc.stderr[-2000:]}"
    assert "successful" in proc.stdout, "GMAT success banner missing"
    stock_startup = GMAT_BIN / "gmat_startup_file.txt"
    print(f"stock startup sha256 {sha256(stock_startup)}")


def parse_oem_records() -> list[tuple[float, ...]]:
    records = []
    in_data = False
    for line in OEM.read_text(encoding="ascii").splitlines():
        if line.strip() == "META_STOP":
            in_data = True
            continue
        if in_data and line.strip():
            toks = line.split()
            records.append(tuple(float(x) for x in toks[1:7]))
    assert len(records) == N_RECORDS
    return records


def compare_report() -> None:
    lines = REPORT.read_text(encoding="ascii").splitlines()
    assert lines[0].lstrip().startswith("EphSat.ElapsedSecs"), "unexpected report header"
    rows: list[tuple[float, ...]] = []
    for line in lines[1:]:
        vals = tuple(float(tok) for tok in line.split())
        assert len(vals) == 8, f"unexpected column count: {line!r}"
        # The final epoch can be re-emitted at the stop boundary; keep the
        # first occurrence of each epoch only (crosstool report semantics).
        if rows and vals[0] <= rows[-1][0]:
            assert abs(vals[0] - rows[-1][0]) < 1e-6, "non-monotonic report epochs"
            continue
        rows.append(vals)
    assert len(rows) == N_RECORDS, f"expected {N_RECORDS} report rows, parsed {len(rows)}"

    # ElapsedSecs for an ephemeris-propagated spacecraft counts from the
    # Spacecraft object's default (year-2000) epoch, not from the OEM start,
    # so the grid check uses elapsed time relative to the first row. The
    # absolute epoch is pinned separately through UTCModJulian: GMAT's
    # ModJulian offset is JD 2430000.0, so 2026-01-01T00:00:00 UTC =
    # JD 2461041.5 = 31041.5 (the same first-row value the crosstool
    # baseline run proved for this console build).
    utc_mjd0 = rows[0][1]
    assert abs(utc_mjd0 - 31041.5) < 5e-6 / 86400.0, (
        f"first report row UTCModJulian {utc_mjd0!r} is not the OEM start "
        f"epoch 31041.5 (2026-01-01T00:00:00 UTC)"
    )
    oem = parse_oem_records()
    elapsed0 = rows[0][0]
    worst_grid = 0.0
    worst_dr = 0.0
    worst_dv = 0.0
    sum_dr2 = 0.0
    for k, vals in enumerate(rows):
        worst_grid = max(worst_grid, abs((vals[0] - elapsed0) - STEP_S * k))
        dr = 1000.0 * float(
            np.linalg.norm(np.array(vals[2:5]) - np.array(oem[k][0:3]))
        )
        dv = 1000.0 * float(
            np.linalg.norm(np.array(vals[5:8]) - np.array(oem[k][3:6]))
        )
        worst_dr = max(worst_dr, dr)
        worst_dv = max(worst_dv, dv)
        sum_dr2 += dr * dr
    rms_dr = (sum_dr2 / len(rows)) ** 0.5
    print(f"first row UTCModJulian = {utc_mjd0!r} (expect 31041.5 = 2026-01-01T00:00:00 UTC)")
    print(f"grid check: max |elapsed - 60k| = {worst_grid:.3e} s over {len(rows)} rows")
    assert worst_grid < 5e-6, f"report epochs off the 60 s grid by {worst_grid} s"
    print(f"round trip: max |dr| = {worst_dr:.3e} m, rms |dr| = {rms_dr:.3e} m, "
          f"max |dv| = {worst_dv:.3e} m/s")
    assert worst_dr < GATE_DR_M, f"max |dr| {worst_dr} m exceeds gate {GATE_DR_M} m"
    assert worst_dv < GATE_DV_MPS, f"max |dv| {worst_dv} m/s exceeds gate {GATE_DV_MPS} m/s"
    print(f"GMAT-OEM-IMPORT: PASS (gates |dr| < {GATE_DR_M} m, |dv| < {GATE_DV_MPS} m/s)")


def print_hashes() -> None:
    for path in (OEM, SCRIPT, REPORT, LOG, OUT_DIR / "run.srlog"):
        print(f"sha256 {sha256(path)}  {path}")


def main() -> None:
    run_mission()
    write_oem()
    write_gmat_script()
    run_gmat()
    compare_report()
    print_hashes()


if __name__ == "__main__":
    main()
