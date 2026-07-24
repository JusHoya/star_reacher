# Monte Carlo manifest format (`manifest.json` v1)

`star mc <sweep.toml>` (FR-27) expands a sweep spec into N independent
single-run processes and writes one `manifest.json` into the output directory.
The manifest is the reproducibility record for the whole sweep: it pins the
binary that produced the runs, the sweep that defined them, and, per run, the
seed, overrides, and hashes that make any single run individually
reproducible. Re-executing a run via

```
star run <mission> --seed <run.seed> --set <path>=<value> ...
```

reproduces that run's `log_sha256` and `config_sha256` bit-for-bit; this is
Phase 7 exit criterion 1.

The file is written with `json.dumps(..., indent=2, sort_keys=True)` and a
trailing newline (the same style as a run's `meta.json`), so it is a stable,
diffable artifact.

## Schema

```json
{
  "schema_version": 1,
  "sweep": {
    "mission": "<base mission path, as written in the sweep spec>",
    "master_seed": <u64>,
    "method": "grid" | "list" | "lhs",
    "n_runs": <int>,
    "parameters": [
      {"path": "mission.duration_s", "min": 3600.0, "max": 7200.0},
      {"path": "spacecraft.mass_kg", "values": [400.0, 500.0, 600.0]},
      ...
    ],
    "sweep_spec_sha256": "<sha256 of the sweep spec file's raw bytes>"
  },
  "binary": {
    "core_version": "<core.core_version()>",
    "core_git_hash": "<core.git_hash()>",
    "binary_sha256": "<sha256 of the compiled _core extension file on disk>"
  },
  "runs": [
    {
      "index": 0,
      "seed": <u64>,
      "overrides": {"mission.duration_s": 4467.0, "spacecraft.mass_kg": 449.24},
      "config_sha256": "<hex>",
      "log_sha256": "<hex>",
      "status": "success",
      "outdir": "run_0000"
    },
    ...
  ]
}
```

## Field reference

### top level

| Field | Meaning |
|---|---|
| `schema_version` | Manifest schema version; currently `1`. |
| `sweep` | The sweep that produced these runs (see below). |
| `binary` | The compiled core that produced these runs (see below). |
| `runs` | One entry per run, sorted by `index` ascending. |

### `sweep`

| Field | Meaning |
|---|---|
| `mission` | The base mission path exactly as written in the sweep spec's `[sweep] mission`. |
| `master_seed` | The u64 master seed; every per-run seed derives from it. |
| `method` | The expansion method: `grid` (Cartesian product), `list` (zipped values), or `lhs` (Latin hypercube). |
| `n_runs` | The number of runs, i.e. `len(runs)`. |
| `parameters` | The swept parameters, in spec order. Each carries `path` (the dotted override path) plus either `values` (grid/list) or `min`/`max` (lhs), and an optional `integer: true`. The per-run sampled values live in each run's `overrides`, not here. |
| `sweep_spec_sha256` | SHA-256 of the sweep spec file's raw bytes, so a reader can confirm a manifest was produced from the spec they hold. |

### `binary`

| Field | Meaning |
|---|---|
| `core_version` | `core.core_version()`, the compiled core's semantic version. |
| `core_git_hash` | `core.git_hash()`, the commit the core was built from. |
| `binary_sha256` | SHA-256 of the compiled `_core` extension file on disk (located via its `__file__`). Two runs are only reproducible on the same binary, so the manifest pins which one produced them. |

### `runs[i]`

| Field | Meaning |
|---|---|
| `index` | The 0-based run index. `runs` is sorted by this. |
| `seed` | The per-run master seed: `SplitMix64(master_seed)[index]`, element `index` of the SplitMix64 stream the core owns (D-9). This is the exact value to pass to `star run --seed`. |
| `overrides` | The `{dotted.path: value}` override dict applied to the base mission, in the FR-24/FR-27 override vocabulary. Each entry maps directly to a `star run --set path=value`. |
| `config_sha256` | SHA-256 of the run's resolved, overridden config (the reproducibility anchor). `null` for a failed run. |
| `log_sha256` | SHA-256 of the run's `run.srlog`. `null` for a failed run. Re-executing the run reproduces this hash. |
| `status` | `"success"` or `"failed"`. |
| `outdir` | The run's output subdirectory name, relative to the manifest (e.g. `run_0000`), holding its `run.srlog`, `resolved_config.json`, and `meta.json`. |
| `error` | Present only on a failed run: `"<ExceptionType>: <message>"`. |

## Per-run seed derivation

The per-run seed for run `index` is `SplitMix64(master_seed)[index]`: seed a
SplitMix64 with `master_seed`, and take the `index`-th output (0-based; draw
`index + 1` values and take the last). The single implementation is the core's
`splitmix64_stream(seed, n)` binding (D-9: the core owns the PRNG); a tested
pure-Python mirror in `star_reacher.mc` is used only as a build-free fallback
and cross-check.

## Reproducing one run

```
star run <sweep.mission> \
  --seed <runs[i].seed> \
  --set <path0>=<value0> --set <path1>=<value1> ... \
  -o <somewhere> --force
```

with each `<pathK>=<valueK>` taken from `runs[i].overrides`. The resulting
`run.srlog` hashes to `runs[i].log_sha256` and its resolved config hashes to
`runs[i].config_sha256`.

## Training-data export

`star mc` writes no bespoke dataset format. The NPZ export path
(`star export --npz <run.srlog>`) is the training-data pipeline for a completed
sweep's runs.
