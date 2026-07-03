# Golden vectors

Committed reference values for the FR-22 layer-1 test discipline: every
module-level unit test that checks numerical output does so against values in
this tree, and every value here carries provenance. An uncited golden is a
lint failure; the discipline is established in Phase 1 so it never has to be
retrofitted.

## Layout

```
tests/golden/
  README.md            this file
  <topic>/             one directory per golden topic (Phase 1: rng/)
    manifest.toml      provenance manifest (required, schema below)
    *.toml             value files (TOML, D-3: one config language everywhere)
    generate.py        regeneration script, when values are tool-generated
```

Value files are TOML. Unsigned 64-bit integers are recorded as `"0x..."` hex
strings because TOML integers are signed 64-bit; floating-point values that
must be exact are recorded as binary64 hex literals (Python `float.hex()`
form). Binary fixtures are not committed when tests can synthesize the bytes
in test code (the Phase 1 rule, contract section 11). Phase 2 adds one
deliberate exception: real DE440 Chebyshev coefficients cannot be
synthesized, so `ephemeris/` commits small binary SREPH excerpts of the
repacked kernel data (D-8: "tiny kernel excerpts needed by golden-vector
tests are committed with provenance"), marked `binary` in `.gitattributes`
and covered by the directory's provenance manifest like every other golden.

## manifest.toml schema

Every golden directory carries a `manifest.toml`:

| Key | Level | Meaning |
|---|---|---|
| `schema_version` | top | Manifest schema version; currently `1`. |
| `[golden] directory` | table | Directory name, for self-identification. |
| `[golden] date` | table | ISO-8601 date the current values were generated. |
| `[golden] generation` | table | How the directory's values are produced end to end, including the cross-checks that anchor them to independent references. |
| `[[file]] name` | array of tables | Value file the entry describes. |
| `[[file]] source` | array of tables | Where the numbers come from (tool, script, publication). |
| `[[file]] citation` | array of tables | The verifiable reference for the algorithm or data. Never invented; "reference TBD" is not permitted in a manifest — values without a real citation do not land. |
| `[[file]] generation` | array of tables | The exact procedure or script entry point that produced the file. |
| `[[file]] date` | array of tables | ISO-8601 generation date for the file. |
| `[[file]] tolerance` | array of tables | Comparison rule the consuming test applies, with its justification (e.g. `exact (u64 bit equality)`, or an abs/rel bound with the derivation of the bound). |

## Update policy

Goldens change only by rerunning the recorded generation procedure and
committing the resulting diff together with a manifest update (new `date`,
and new `generation`/`tolerance` text if the procedure changed). Hand-editing
value files is forbidden. The full two-key update tooling arrives with the
Monte Carlo regression layer (FR-22 layer 6); until then, review enforces
this policy.

## Phase 1 contents

- `rng/` — FNV-1a, SplitMix64, PCG64 (raw and named-stream), and Box-Muller
  golden vectors consumed by the doctest cases `rng_splitmix64_golden`,
  `rng_pcg64_golden`, and `rng_box_muller_golden` (and by the Python
  cross-validation tests). See `rng/manifest.toml` for provenance and
  tolerances.

## Phase 2 contents

- `time/` — UTC→TAI→TT conversion, TDB−TT series, and leap-second history
  golden vectors (FR-2, D-6), cross-checked against ERFA and the published
  SOFA cookbook anchor; consumed by the doctest cases
  `time_utc_tai_tt_golden`, `time_tdb_series_golden`, and
  `time_leap_table_golden` (and by the Python binding tests). See
  `time/manifest.toml` for provenance and tolerances.
- `ephemeris/` — DE440s repack validation set (FR-4, D-8): a committed
  binary SREPH excerpt of the repacked Chebyshev records, bit-level
  evaluator goldens, JPL Horizons geometric state vectors with raw query
  transcripts (`horizons/`), jplephem-evaluated DE440 lunar states and
  libration angles, and the committed full-span validation summary
  (`full_span_validation.md`). Consumed by the doctest cases
  `ephemeris_bitlevel_golden`, `ephemeris_segment_boundary_continuity`, and
  `ephemeris_error_paths`, and by `tests/python/test_ephemeris_horizons.py`.
  See `ephemeris/manifest.toml` for provenance and tolerances.
- `integrators/` — reference eccentric-orbit definition, analytic Kepler
  checkpoint states, and analytic apsis passage times consumed by the
  integrator/event acceptance suites (doctest `test_integrate.cpp` and
  `test_events.cpp`, pytest `test_integrators.py`). See
  `integrators/manifest.toml` for provenance and tolerances.
- `rotations/` — quaternion→DCM and Euler-sequence (3-2-1, 3-1-3) golden
  vectors (FR-3, D-7), each matrix produced by two independent
  constructions (ERFA rotation primitives and NumPy closed forms) that must
  agree at generation time; consumed by the doctest cases
  `rotation_quat_dcm_golden` and `rotation_euler_golden` (and by the Python
  binding tests). See `rotations/manifest.toml` for provenance and
  tolerances.
- `frames/` — the composed IAU 2006/2000B GCRF→ITRF chain at 14 epochs
  spanning 2020–2060 (ERFA-generated), the published SOFA cookbook
  earth-attitude worked example, Moon principal-axis and Mars IAU 2015
  frame constructions, and the transcribed nutation/s(X,Y) series tables
  (FR-3); consumed by the doctest cases `frames_erfa_chain_golden`,
  `frames_sofa_cookbook_crosscheck`, `frames_moon_pa_golden`,
  `frames_mars_iau_golden`, and `frames_series_transcription` (and by the
  Python binding tests). See `frames/manifest.toml` for provenance and
  tolerances.

## Phase 3 contents

- `gravity/` — FR-5 spherical-harmonic gravity golden set: committed
  coefficient excerpts of Earth EGM2008 (20×20; also serves the 8×8
  cross-tool case by runtime truncation), Moon GRGM1200A (50×50), and Mars
  MRO120F (20×20), each in a full-precision CSV form and a binary SRGRAV v1
  form (the committed-binary exception documented above, marked `binary` in
  `.gitattributes`), plus independently synthesized pyshtools point
  accelerations at the 20 Phase 3 exit-criterion-1 states. Consumed by the
  doctest cases `GRAV-XTOOL-20`, `GRAV-J2-SECULAR`,
  `gravity_pointmass_tier`, `gravity_j2_tier_closed_form`,
  `gravity_pole_regularity`, `gravity_truncation_consistency`,
  `gravity_srgrav_error_paths`, and by
  `tests/python/test_gravity_data.py`. See `gravity/manifest.toml` for
  provenance and tolerances; the manifest doubles as the committed fetch
  record (source URLs, SHA-256 pins, and full-degree repack hashes) for the
  git-ignored `data/` gravity files.
- `thirdbody/` — Battin f(q) third-body reference accelerations (FR-6,
  Phase 3 exit criterion 7): 10 committed states with the naive
  two-vector-difference acceleration evaluated in extended precision
  (mpmath, 60 digits), including the flagged near-alignment state where the
  naive double-precision evaluation loses ≥ 6 significant digits; consumed
  by the doctest cases `thirdbody_battin_extended_reference_golden` and
  `thirdbody_naive_cancellation_digit_loss`. See `thirdbody/manifest.toml`
  for provenance and tolerances.
- `srp/` — cannonball SRP accelerations and conical-shadow illumination
  fractions (FR-7): every branch of the apparent-disk overlap model (full
  sun, umbra, three penumbra depths, annular, off-axis annular, partial
  lunar occultation, Moon-occulter umbra) plus SRP acceleration states
  with the exact-zero umbra case and a Mars-distance inverse-square check,
  all mpmath-generated (60 digits); consumed by the doctest cases
  `srp_shadow_fraction_golden` and `srp_cannonball_accel_golden`. See
  `srp/manifest.toml` for provenance and tolerances.
- `atmosphere/` — atmosphere and orbital-drag golden vectors (FR-8, FR-9):
  USSA76 published table rows and 86–1000 km density nodes transcribed
  from the official 1976 document with per-row page provenance, the
  Harris–Priester min/max density coefficient table (Montenbruck & Gill;
  transcription cross-checked against the Orekit reference implementation),
  Mars piecewise-exponential nodes (flagged `confidence: low` per PRD A-3),
  and mpmath-computed off-node and cannonball-drag reference values.
  Consumed by the doctest cases `ATM-USSA76-ROWS`,
  `ATM-USSA76-UPPER-NODES`, `ATM-HP-NODES`, `ATM-HP-OFFNODE`,
  `ATM-MARS-NODES`, `ATM-MARS-CONT`, and `DRAG-CANNONBALL-GOLDEN`. See
  `atmosphere/manifest.toml` for provenance and tolerances.
