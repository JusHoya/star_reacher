"""Generate zeroed Earth-orientation-parameter files for the cross-tool runs.

The simulator deliberately neglects polar motion and sets dUT1 = 0 (PRD
non-goal: EOP ingestion; tests/golden/crosstool/README.md, "Earth
orientation"). For the Phase 3 exit-criterion-5 controlled comparison the
external tools must be configured to the closest legitimate equivalent, so
this script derives EOP files whose measured columns are all zero from the
real files each tool ships, preserving every byte of layout the tools'
fixed-column parsers depend on:

- ``tests/golden/crosstool/eopc04_zero.txt`` -- IERS EOP C04 (old format),
  derived from GMAT R2026a's ``data/planetary_coeff/eopc04_08.62-now``
  (header kept verbatim, data rows date-windowed and zeroed). Wired into a
  GMAT run via a startup-file override of ``EOP_FILE`` (see
  scripts/crosstool/run_gmat_case1.py).
- ``tests/golden/crosstool/finals2000A_zero.all`` -- IERS finals2000A
  fixed-width format, derived from the pinned orekit-data snapshot's
  ``Earth-Orientation-Parameters/IAU-2000/finals2000A.all`` (rows
  date-windowed, measured values zeroed, I/P flags and layout preserved).
  Placed as the ONLY EOP file in the curated Orekit data directory (see
  scripts/crosstool/build_orekit_zeroeop_data.py).

Zeroed columns: polar motion x/y, UT1-UTC, LOD, and the nutation/CIP
corrections (dPsi/dEps in the C04 file, dX/dY in finals2000A, both the rapid
and Bulletin B sections). Formal-error columns are left untouched (neither
tool's frame chain consumes them).

The window (2025-06-01 .. 2026-12-31, MJD 60827..61405) covers the 7-day
missions plus wide margin so neither tool interpolates near a table edge.

Deterministic: output bytes are a pure function of the two source files,
which are pinned by the toolchain provenance recorded in
tests/golden/crosstool/manifest.toml (GMAT archive SHA-256 and orekit-data
snapshot SHA-256). Run from the repo root on the maintainer machine:

    python scripts/crosstool/gen_zero_eop.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "tests" / "golden" / "crosstool"

GMAT_EOP = Path(r"C:\Users\hoyer\WorkSpace\tools\gmat\data\planetary_coeff\eopc04_08.62-now")
OREKIT_FINALS = Path(
    r"C:\Users\hoyer\WorkSpace\tools\orekit-data\Earth-Orientation-Parameters"
    r"\IAU-2000\finals2000A.all"
)

MJD_MIN = 60827  # 2025-06-01
MJD_MAX = 61405  # 2026-12-31


def zero_c04(src: Path) -> str:
    """Zero the C04 'old format' data columns, keeping the header verbatim.

    Row layout per the file's own FORMAT line:
    3(I4) date, I7 MJD, then x y (2F11.6), UT1-UTC LOD (2F12.7), dPsi dEps
    (2F11.6), followed by the six formal-error columns (kept as-is).
    """
    out_lines = []
    for line in src.read_text(encoding="ascii", errors="strict").splitlines():
        fields = line.split()
        is_data = (
            len(fields) >= 10
            and len(line) >= 87
            and fields[0].isdigit()
            and len(fields[0]) == 4
        )
        if not is_data:
            out_lines.append(line)
            continue
        mjd = int(fields[3])
        if not (MJD_MIN <= mjd <= MJD_MAX):
            continue
        # Date + MJD occupy [0:19] (3(I4), I7); the six measured columns
        # x, y, UT1-UTC, LOD, dPsi, dEps occupy [19:87]; formal errors follow.
        zeroed = (
            line[:19]
            + "%11.6f%11.6f%12.7f%12.7f%11.6f%11.6f" % (0, 0, 0, 0, 0, 0)
            + line[87:]
        )
        assert len(zeroed) == len(line), f"layout drift at MJD {mjd}"
        out_lines.append(zeroed)
    return "\n".join(out_lines) + "\n"


# finals2000A fixed-column value fields to zero: (start, end, replacement)
# per the IERS readme.finals2000A layout (1-based columns converted to
# 0-based slices). Bulletin B fields (from column 135) are zeroed only when
# the row carries them.
_FINALS_FIELDS = [
    (18, 27, "%9.6f" % 0.0),  # PM-x [arcsec]
    (37, 46, "%9.6f" % 0.0),  # PM-y [arcsec]
    (58, 68, "%10.7f" % 0.0),  # UT1-UTC [s]
    (79, 86, "%7.4f" % 0.0),  # LOD [ms]
    (97, 106, "%9.3f" % 0.0),  # dX [mas]
    (116, 125, "%9.3f" % 0.0),  # dY [mas]
    (134, 144, "%10.6f" % 0.0),  # Bulletin B PM-x
    (144, 154, "%10.6f" % 0.0),  # Bulletin B PM-y
    (154, 165, "%11.7f" % 0.0),  # Bulletin B UT1-UTC
    (165, 175, "%10.3f" % 0.0),  # Bulletin B dX
    (175, 185, "%10.3f" % 0.0),  # Bulletin B dY
]


def zero_finals(src: Path) -> str:
    out_lines = []
    for line in src.read_text(encoding="ascii", errors="strict").splitlines():
        mjd_text = line[7:15].strip()
        try:
            mjd = int(float(mjd_text))
        except ValueError:
            continue
        if not (MJD_MIN <= mjd <= MJD_MAX):
            continue
        chars = list(line)
        for start, end, repl in _FINALS_FIELDS:
            if len(chars) < end:
                break  # predicted rows carry no Bulletin B section
            if not line[start:end].strip():
                continue  # field empty in the source row; leave it empty
            assert len(repl) == end - start
            chars[start:end] = repl
        zeroed = "".join(chars)
        assert len(zeroed) == len(line), f"layout drift at MJD {mjd}"
        out_lines.append(zeroed)
    return "\n".join(out_lines) + "\n"


def main() -> None:
    for src, name, fn in (
        (GMAT_EOP, "eopc04_zero.txt", zero_c04),
        (OREKIT_FINALS, "finals2000A_zero.all", zero_finals),
    ):
        src_sha = hashlib.sha256(src.read_bytes()).hexdigest()
        text = fn(src)
        out = OUT_DIR / name
        out.write_text(text, newline="\n", encoding="ascii")
        print(f"{name}: {len(text)} bytes, sha256 {hashlib.sha256(out.read_bytes()).hexdigest()}")
        print(f"  source {src} sha256 {src_sha}")


if __name__ == "__main__":
    main()
