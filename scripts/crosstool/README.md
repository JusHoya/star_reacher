# Cross-tool freeze scripts (Phase 3 exit criterion 5, D-15)

Maintainer-run scripts that generate the frozen external-truth baselines
committed under `tests/golden/crosstool/`. **CI never runs anything in this
directory**: per D-15 the external tools (GMAT, Orekit) run offline on the
maintainer machine only, and CI consumes only the committed artifacts through
`tests/python/test_crosstool_frozen_truth.py`.

They depend on the portable toolchain documented (with download URLs, SHA-256
pins, and proof runs) in `C:/Users/hoyer/WorkSpace/tools/bootstrap-notes/README.md`:
GMAT R2026a at `tools/gmat/`, Temurin 21 JDK at `tools/jdk/`, the
`tools/orekit-venv` Python 3.12 environment (orekit_jpype 13.1.5.0 = Orekit
13.1.5), and the pinned orekit-data snapshot at `tools/orekit-data/`.

Regeneration order (repo root; one external propagation at a time):

1. `python scripts/crosstool/gen_field_files.py` — derive the GMAT `.cof` and
   Orekit `.gfc` gravity fields from the committed EGM2008 excerpt (identical
   GM/R/coefficients to what the missions load).
2. `python scripts/crosstool/gen_zero_eop.py` — derive the zeroed-EOP files
   (controlled comparison: the simulator's no-polar-motion, dUT1 = 0
   convention) from each tool's shipped/pinned EOP product.
3. `python scripts/crosstool/build_orekit_zeroeop_data.py` — assemble the
   curated zero-EOP Orekit data directory at `tools/orekit-data-zeroeop/`.
4. `python scripts/crosstool/run_gmat_case1.py` — run GMAT on the committed
   script with the zero-EOP startup override and freeze
   `truth_gmat_leo_gravity_8x8.csv`.
5. `<orekit-venv>/Scripts/python.exe scripts/crosstool/run_orekit.py --case drag`
   — freeze `truth_orekit_leo_drag_hp.csv`.
6. `<orekit-venv>/Scripts/python.exe scripts/crosstool/run_orekit.py --case grav`
   — the informational Orekit corroboration of the GMAT baseline (output stays
   in `tools/crosstool-runs/`; only its RMS numbers are recorded in the
   manifest).
7. `python scripts/crosstool/compare_rms.py <a> <b>` — report the RMS numbers
   recorded in the manifest (accepts truth CSVs and mission `run.srlog` files).

Every frozen artifact, configuration choice, command line, hash, and measured
number is recorded in `tests/golden/crosstool/manifest.toml`; a refreeze must
update that manifest in the same commit (tests/golden/README.md update
policy).
