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
form). Binary fixtures are never committed (contract section 11): tests that
need byte streams synthesize them in test code.

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

- `integrators/` — reference eccentric-orbit definition, analytic Kepler
  checkpoint states, and analytic apsis passage times consumed by the
  integrator/event acceptance suites (doctest `test_integrate.cpp` and
  `test_events.cpp`, pytest `test_integrators.py`). See
  `integrators/manifest.toml` for provenance and tolerances.
