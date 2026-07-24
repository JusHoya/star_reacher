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

## Phase 8 cross-tool cases (exit criterion 1, D-15)

Five additional GMAT cases: Molniya, lunar orbiter, Mars orbiter, trans-lunar
coast, and Earth-Mars cruise (the Molniya, lunar-orbiter, and Mars-orbiter
gates are 7-day position RMS < 100 m; trans-lunar is < 1 km at lunar arrival;
Mars cruise is < 100 km at arrival). Their missions, GMAT `.script` files, and
gravity-field COF inputs are committed; the frozen truth is generated offline
on the maintainer GMAT machine (`docs/release_checklist.md` item 10) because
GMAT was not installed on the Phase 8 execution host. Status 2026-07-24: four
of the five truth CSVs are frozen and their gates measure (values in
`tests/golden/crosstool/manifest.toml`); the trans-lunar CSV is pending a
re-freeze (its first freeze started from the mid-burn t = 353 s state and was
invalidated -- the manifest's deferral note has the history). A gate in
`tests/python/test_crosstool_frozen_truth.py` whose truth CSV is absent skips
with an explicit deferral message until the CSV lands.

Phase 8 regeneration order (repo root; maintainer with GMAT R2026a):

1. `python scripts/crosstool/gen_field_files_phase8.py` — derive the GMAT COF
   fields `moon_grgm1200a_50x50.cof` and `mars_mro120f_20x20.cof` from the
   committed GRGM1200A/MRO120F excerpts (needs no external tool; already
   committed, re-run only if the source excerpts change).
2. `python tests/golden/ephemeris/generate_lunar.py` — cut the lunar DE440
   excerpt carrying the `moon_librations` segment (needs the fetched DE440
   kernels), required by the lunar-orbiter and LRO missions.
3. `python scripts/crosstool/run_gmat_phase8.py --case molniya` (then
   `lunar_orbiter`, `mars_orbiter`, `translunar`, `mars_cruise`) — run GMAT on
   each committed script and freeze the truth CSV. For `translunar`, pass
   `--arrival-s <span>` if the frozen simulator run locates a different lunar
   arrival epoch than the default 455401 s (elapsed from the t = 354 s
   ballistic burnout the script's coast starts at).
4. `python scripts/crosstool/compare_rms.py <mission run.srlog> <truth.csv>` —
   report the 7-day RMS numbers for the manifest (the trans-lunar and
   Mars-cruise arrival-point numbers are reported by their gates directly).

CI never runs any of this; it consumes only the committed CSVs, and until they
exist the gates skip. The lunar cases additionally need step 2's excerpt.
