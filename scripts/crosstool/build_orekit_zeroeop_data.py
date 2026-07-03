"""Assemble the curated zero-EOP Orekit data directory for the cross-tool runs.

Maintainer-side only (D-15). Orekit loads physical data by crawling a data
directory, so the controlled-comparison configuration is expressed as a
minimal curated directory at C:/Users/hoyer/WorkSpace/tools/orekit-data-zeroeop
containing exactly:

- ``tai-utc.dat``            leap seconds (copied from the pinned orekit-data
                             snapshot; time scales are NOT zeroed, TAI-UTC=37 s)
- ``itrf-versions.conf``     EOP-file-to-ITRF-version map (copied; inert with
                             zeroed pole data, required by the frames factory)
- ``Earth-Orientation-Parameters/IAU-2000/finals2000A.all``
                             the committed zeroed-EOP file (gen_zero_eop.py):
                             polar motion, UT1-UTC, LOD, dX/dY all zero, so
                             the IERS-2010 frame chain reduces to precession-
                             nutation + ERA with UT1 = UTC and no polar motion
                             -- the closest legitimate equivalent of the
                             simulator's no-EOP convention
- ``DE-440-ephemerides/lnxp1990.440``
                             JPL DE440 (copied; the Sun position source for
                             the Harris-Priester bulge, same DE440 family as
                             the mission's committed excerpt)
- ``Potential/earth_egm2008_8x8.gfc``
                             the committed field generated from the same
                             coefficient excerpt the missions load
                             (gen_field_files.py); the ONLY potential file, so
                             GravityFieldFactory cannot pick up eigen-6s.gfc

Everything else in the orekit-data snapshot (real EOP, MSAFE, CSSI space
weather, ocean tides) is deliberately absent: none of it belongs to the
controlled comparison, and its absence makes accidental use impossible
(loading it would raise, not silently substitute).

Deterministic: the output tree is a pure function of the pinned orekit-data
snapshot (SHA-256 in the manifest) and two committed files. Prints the
SHA-256 of every file placed. Run from the repo root:

    python scripts/crosstool/build_orekit_zeroeop_data.py
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CROSSTOOL = REPO_ROOT / "tests" / "golden" / "crosstool"
OREKIT_DATA = Path(r"C:\Users\hoyer\WorkSpace\tools\orekit-data")
OUT = Path(r"C:\Users\hoyer\WorkSpace\tools\orekit-data-zeroeop")

COPIES = [
    (OREKIT_DATA / "tai-utc.dat", OUT / "tai-utc.dat"),
    (OREKIT_DATA / "itrf-versions.conf", OUT / "itrf-versions.conf"),
    (
        OREKIT_DATA / "DE-440-ephemerides" / "lnxp1990.440",
        OUT / "DE-440-ephemerides" / "lnxp1990.440",
    ),
    (
        CROSSTOOL / "finals2000A_zero.all",
        OUT / "Earth-Orientation-Parameters" / "IAU-2000" / "finals2000A.all",
    ),
    (
        CROSSTOOL / "earth_egm2008_8x8.gfc",
        OUT / "Potential" / "earth_egm2008_8x8.gfc",
    ),
]


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    for src, dst in COPIES:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        digest = hashlib.sha256(dst.read_bytes()).hexdigest()
        print(f"{dst.relative_to(OUT)}: {dst.stat().st_size} bytes, sha256 {digest}")
    print(f"curated Orekit data directory ready: {OUT}")


if __name__ == "__main__":
    main()
