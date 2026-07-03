# Vehicle TOML v1 configuration format

Normative specification of the vehicle definition file (PRD FR-13, DX-3) as
validated by the Python layer (`star_reacher.vehicle`, per D-2: the C++ core
never parses text; a typed, validated struct crosses the binding). The module
docstring carries the same schema for interactive reference; this document is
the format authority. A change here is a format change: `schema_version`
increments on any break, and the validator refuses versions it does not
implement.

The schema is deliberately small ("KSP-lite"): approximately 35 curated
physical parameters, every one justified by an in-file comment in the shipped
starter fleet (DX-3). Required parameters abort validation when missing and
are never silently defaulted (FR-13); the only optional keys are
`[vehicle] description` and `[[aero]] cmq_per_rad`, plus the block arrays
themselves.

## 1. Conventions

- **Structural frame.** One frame per vehicle: +X forward (toward the nose),
  +Y/+Z completing a right-handed triad, origin at the aft plane of the
  assembled stack. Every position, CG, axis, inertia orientation, and aero
  center-of-pressure station uses this frame.
- **Units.** SI, suffixed in key names (`thrust_vac_N`, `isp_vac_s`) so a
  wrong unit is visible at the key (DX-2/DX-3). Dimensionless keys carry no
  suffix.
- **Inertia tensors** are 3x3, symmetric as written, about the owning block's
  own CG, axis-aligned with the structural frame. The core composes the stack
  by parallel-axis transport.
- **Provenance** is mandatory at top level. The starter fleet uses
  `"representative"`: round, class-representative values chosen by the
  author, not published data of any real vehicle.

## 2. Schema

Top level: `schema_version = 1` (integer), `provenance` (non-empty string),
`[vehicle]` (name required, description optional), `[[stage]]` (ordered array,
bottom stage first, at least one), `[[aero]]` (optional array, one block per
stack configuration).

| Table | Key | Units | Requirement |
|---|---|---|---|
| `[[stage]]` | `name` | — | unique across stages |
| | `dry_mass_kg` | kg | > 0 |
| | `dry_cg_m` | m | vec3 |
| | `dry_inertia_kgm2` | kg·m² | 3x3 SPD, principal moments obey I1+I2 ≥ I3 |
| `[[stage.tank]]` | `name` | — | unique in stage |
| | `radius_m`, `length_m` | m | > 0; settled cylinder, axis +X (A-2) |
| | `position_m` | m | vec3, cylinder center |
| | `propellant_mass_kg` | kg | > 0, must fit ρ·π·r²·L |
| | `density_kgpm3` | kg/m³ | > 0, bulk density of the load |
| `[[stage.engine]]` | `name` | — | unique in stage |
| | `feeds_tank` | — | names a tank in the same stage |
| | `thrust_vac_N` | N | > 0; F = F_vac − p_amb·Ae (FR-10) |
| | `isp_vac_s` | s | > 0; warning outside 200–500 |
| | `exit_area_m2` | m² | > 0 |
| | `position_m` | m | vec3 mount point |
| | `axis` | — | unit vec3, nominal thrust force direction |
| | `gimbal_max_deg` | deg | [0, 45]; 0 = fixed engine |
| | `gimbal_rate_dps` | deg/s | ≥ 0 |
| | `throttle_min`, `throttle_max` | — | (0, 1], min ≤ max |
| | `spool_time_s` | s | ≥ 0, linear ramp |
| | `ignitions` | — | integer ≥ 1 |
| `[[stage.rcs]]` | `name` | — | unique in stage |
| | `thrust_N` | N | > 0, per thruster |
| | `min_impulse_bit_Ns` | N·s | > 0 |
| | `thruster_positions_m` | m | array of ≥ 1 vec3 |
| | `thruster_directions` | — | index-matched unit vec3s (force on vehicle) |
| `[[stage.wheel]]` | `name` | — | unique in stage |
| | `axis` | — | unit vec3 spin axis |
| | `max_torque_Nm` | N·m | > 0 |
| | `max_momentum_Nms` | N·m·s | > 0 |
| `[[stage.sensor]]` | `name` | — | unique in stage |
| | `preset` | — | non-empty reference string; dereferenced in Phase 6 |
| | `position_m` | m | vec3 |
| | `axis` | — | unit vec3 boresight/sensitive axis |
| `[[stage.jettison]]` | `name` | — | unique in stage |
| | `mass_kg` | kg | > 0; rides with the stack until jettisoned |
| | `cg_m` | m | vec3 |
| | `inertia_kgm2` | kg·m² | same rules as stage inertia |
| `[[aero]]` | `config` | — | unique stack-configuration name |
| | `ref_area_m2` | m² | > 0 |
| | `ref_diameter_m` | m | > 0 |
| | `mach_table_csv` | — | path, resolved against the working directory |
| | `cmq_per_rad` | 1/rad | optional, ≤ 0 (constant pitch damping) |

## 3. Aero Mach-table CSV

The first non-comment line must be exactly the header
`mach,ca,cnalpha_per_rad,xcp_m`; units are declared by the column names (Mach
and CA dimensionless, CNα per radian, xcp meters in the structural frame).
Lines starting with `#` are comments. At least two rows; the `mach` column
strictly increasing from ≥ 0. The axisymmetric aero model (FR-9) interpolates
CA(M), CNα(M), xcp(M) between breakpoints; its domain of validity is
continuum ascent flight only.

## 4. Validation behavior (FR-15)

Four passes, all errors accumulated (DX-2), nonzero exit on any error:

1. **Parse/schema** — TOML parse, required keys/tables, unknown-key rejection
   at every level (typos are errors).
2. **Field ranges** — positivity, finiteness, and the interval bounds above.
3. **Cross-field physics** — inertia symmetry/SPD/triangle inequality,
   engine-feed tank resolution, unit-norm axes, index-matched RCS arrays,
   propellant-fits-tank, name uniqueness, Mach-table structure.
4. **Vehicle-level sanity** — positive wet mass (hard error), plus the
   warning tier: liftoff thrust-to-weight below 1.2 on an aero-configured
   vehicle, non-positive sea-level thrust on a first-stage engine, Isp
   outside the typical chemical range, stage propellant mass fraction above
   0.95. `--strict` promotes warnings to errors.

Error lines follow DX-2: file, exact table/key path (array indices 1-based,
e.g. `[stage.2.engine.1]`), message, units and typical range where
meaningful, closing with `No default applied; run aborted.`

## 5. Resolved-config echo and hash

On success the validator emits the resolved configuration: canonical key
order, numeric values coerced to canonical types (floats everywhere except
`schema_version` and `ignitions`), optional keys present only when given.
`canonical_vehicle_toml` serializes it as canonical TOML; re-validating the
echo reproduces byte-identical output (Phase 4 exit criterion 1). The
SHA-256 over the canonical JSON bytes (`star_reacher.mission.config_sha256`,
the same recipe as the mission hash) is the vehicle's reproducibility hash;
mission validation embeds it in the mission's resolved config, so the run's
config hash — the reproducibility anchor written to `run.srlog` and
`meta.json` — covers the vehicle definition (FR-15).
