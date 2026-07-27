"""Generate the Phase 8 lunar and Mars GMAT gravity-field inputs.

Phase 8 exit criterion 1 (D-15) requires GMAT to propagate the lunar-orbiter
and Mars-orbiter cross-tool cases with the *identical* (GM, R, coefficients)
triple the missions load from the committed SRGRAV fields. This script derives
the GMAT COF form of each from the committed full-precision CSV form of that
field (the CSV and SRGRAV forms are byte-locked by
``tests/python/test_gravity_data.py``), truncated to the degree/order the
missions use:

- ``tests/golden/crosstool/moon_grgm1200a_50x50.cof`` from
  ``tests/golden/gravity/moon_grgm1200a_n50.csv`` (GRGM1200A, degree/order 50,
  the mission's field for missions/lunar_orbiter.toml and lro_illustrative.toml)
- ``tests/golden/crosstool/mars_mro120f_20x20.cof`` from
  ``tests/golden/gravity/mars_mro120f_n20.csv`` (MRO120F, degree/order 20, the
  mission's field for missions/mars_orbiter.toml)

This is the Phase 8 sibling of scripts/crosstool/gen_field_files.py (which
generates the Phase 3 Earth COF/GFC). Only the COF form is produced because
the Phase 8 lunar and Mars cases are GMAT-only (no Orekit tie-breaker case
uses these fields). Output bytes are a pure function of the committed CSVs, so
regeneration is deterministic and needs no external tool. Run from the repo
root:

    python scripts/crosstool/gen_field_files_phase8.py

Format notes match gen_field_files.py: COF numeric fields are fixed-width
``%.14E`` (15 significant digits), so each COF coefficient can differ from the
committed binary64 by at most ~1 part in 1e15; the worst relative deviation
over each slice is printed and recorded in the manifest. GM and R print
through ``%.14E`` from the field's own header constants, so GMAT evaluates
each field with its own self-consistent GM/R/coefficient triple - the same
per-field-constants rule the missions and the crosstool README document.
Degrees 0..1 are omitted from the COF exactly as in the Earth case: all three
fields carry exactly-zero degree-1 coefficients (centre-of-mass frames), so
GMAT's implied C00 = 1 and zero degree-1 reproduce the loaded field.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from star_reacher.data_fetch import read_coeffs_csv  # noqa: E402

GRAVITY_DIR = REPO_ROOT / "tests" / "golden" / "gravity"
OUT_DIR = REPO_ROOT / "tests" / "golden" / "crosstool"

# (out_name, source_csv, degree, body_label, source_model) per case.
CASES = (
    (
        "moon_grgm1200a_50x50.cof",
        "moon_grgm1200a_n50.csv",
        50,
        "GRGM1200A truncated to 50x50",
        "GRGM1200A (Lemoine et al. 2014)",
    ),
    (
        "mars_mro120f_20x20.cof",
        "mars_mro120f_n20.csv",
        20,
        "MRO120F truncated to 20x20",
        "MRO120F (Genova et al. 2016)",
    ),
)


def cof_text(field, degree: int, out_name: str, src_csv: str) -> str:
    """GMAT COF form, byte-layout-identical to gen_field_files.py's output.

    ``POTFIELD`` + degree(%3d) + order(%3d) + ``  1`` + GM, R, scale as
    ``%21.14E``; ``RECOEF`` + n(%5d) + m(%3d) + C(%24.14E) [+ S(%21.14E)];
    zonal rows omit the S field and degrees 0..1 are omitted entirely (C00 = 1
    is implied by GMAT and both fields' degree-1 coefficients are exactly zero).
    """
    lines = [
        "COMMENT   5",
        "CCCCC  ------------------------------------------------------------------  CCCCC",
        "CCCCC  %-58s  CCCCC" % (f"{out_name} : generated from"),
        "CCCCC  %-58s  CCCCC" % (f"tests/golden/gravity/{src_csv} by"),
        "CCCCC  %-58s  CCCCC" % "scripts/crosstool/gen_field_files_phase8.py (star_reacher, D-15).",
        "CCCCC  ------------------------------------------------------------------  CCCCC",
        "POTFIELD%3d%3d  1%21.14E%21.14E%21.14E"
        % (degree, degree, field.gm_m3ps2, field.ref_radius_m, 1.0),
    ]
    for n in range(2, degree + 1):
        for m in range(0, n + 1):
            row = "RECOEF%5d%3d%24.14E" % (n, m, field.cbar[n, m])
            if m > 0:
                row += "%21.14E" % field.sbar[n, m]
            lines.append(row)
    lines.append("END ")
    return "\n".join(lines) + "\n"


def main() -> None:
    for out_name, src_csv, degree, _label, _model in CASES:
        field = read_coeffs_csv(GRAVITY_DIR / src_csv).truncated(degree, degree)

        # Guard the COF layout's assumption: degrees 0..1 are omitted, which is
        # only correct when the degree-1 terms are exactly zero (a CoM frame).
        for m in range(0, 2):
            assert field.cbar[1, m] == 0.0 and field.sbar[1, m] == 0.0, (
                f"{src_csv}: nonzero degree-1 coefficient; the COF layout that "
                f"omits degree 1 would drop it"
            )

        # Quantify the COF 15-significant-digit rounding against the committed
        # binary64 values.
        worst = 0.0
        for n in range(2, degree + 1):
            for m in range(0, n + 1):
                for v in (field.cbar[n, m], field.sbar[n, m]):
                    if v != 0.0:
                        rt = float("%.14E" % v)
                        worst = max(worst, abs(rt - v) / abs(v))

        path = OUT_DIR / out_name
        text = cof_text(field, degree, out_name, src_csv)
        path.write_text(text, newline="\n", encoding="ascii")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{out_name}: {len(text)} bytes, sha256 {digest}")
        print(
            f"  source {src_csv}, GM {field.gm_m3ps2!r}, R {field.ref_radius_m!r}, "
            f"COF 15-digit worst relative deviation {worst:.3e}"
        )


if __name__ == "__main__":
    main()
