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
