# SRLOG v1 binary log format

Normative specification of the SRLOG on-disk format (PRD D-11, FR-16) as
written by the C++ core (`cpp/src/srlog_writer.cpp`) and read by the Python
loader (`star_reacher.load`). Writer and reader are both implemented against
this document; a change here is a format change and follows the versioning
rules in section 6. The current version is **1.1** (version history in
section 6.1).

SRLOG exists because the determinism contract (D-10/FR-21) is enforced at the
whole-file level: the same inputs on the same binary must produce a
byte-identical file, so the format admits no nondeterministic layout, no
embedded timestamps, and no host-dependent content anywhere.

## 1. Overview

- Byte order: **little-endian throughout**. Every supported platform
  (Windows x64, Linux x86-64, Linux aarch64, macOS arm64) is little-endian;
  the writer refuses to run on a big-endian host.
- File = HEADER, then RECORD STREAM. **No footer.** The file is append-only
  while being written; a truncated trailing record therefore means corruption
  (or an interrupted run) and is rejected by the reader.
- Floating-point values are IEEE-754 binary64, stored little-endian.

## 2. Header

| Bytes | Type | Content |
|---|---|---|
| 0-7 | bytes | Magic: `53 52 4C 4F 47 00 0D 0A` (ASCII `SRLOG`, NUL, CR, LF) |
| 8-9 | u16 | `version_major` = 1 |
| 10-11 | u16 | `version_minor` = 1 (the version this document specifies; readers accept any minor, section 6) |
| 12-15 | u32 | `header_json_len` = L |
| 16..16+L | bytes | UTF-8 JSON, no BOM (section 3) |

The magic is a deliberate tripwire: the NUL terminates C-string readers early,
and the CR/LF pair is altered by any text-mode transfer, so a mangled file
fails the magic check immediately instead of misparsing downstream.

## 3. Header JSON

The header JSON is produced by the core's hand-rolled serializer with
**compact separators** (no whitespace) and a **fixed key order** - the exact
order shown below. It contains **integers, booleans, and strings only; floats
never appear in the header** (float formatting is a portability hazard the
determinism contract does not accept). `master_seed` rides as a decimal
string because JSON numbers cannot carry all u64 values without precision
loss in common readers.

Shown pretty-printed for readability; the file contains one compact line:

```json
{"format":{"name":"SRLOG","major":1,"minor":1},
 "producer":{"core_version":"0.1.0","git_hash":"<40-hex or 'unknown'>"},
 "config_sha256":"<64-hex>",
 "master_seed":"<decimal u64 as string>",
 "oracle":false,
 "epoch_utc":"<ISO-8601 string verbatim from mission file>",
 "central_body":"earth",
 "groups":[
   {"name":"truth","rate_hz":10,"channels":[
     {"name":"t_s","dtype":"f64","units":"s","frame":""},
     {"name":"r_m","dtype":"f64[3]","units":"m","frame":"GCRF"},
     {"name":"v_mps","dtype":"f64[3]","units":"m/s","frame":"GCRF"},
     {"name":"q_i2b","dtype":"f64[4]","units":"1","frame":"GCRF->body Hamilton scalar-first"},
     {"name":"w_b_radps","dtype":"f64[3]","units":"rad/s","frame":"body"},
     {"name":"mass_kg","dtype":"f64","units":"kg","frame":""}]},
   {"name":"events","rate_hz":0,"channels":[
     {"name":"t_s","dtype":"f64","units":"s","frame":""},
     {"name":"code","dtype":"u32","units":"1","frame":""},
     {"name":"detail","dtype":"str16","units":"","frame":""}]}
 ]}
```

Field semantics:

- `format` - format identity; `major`/`minor` duplicate the binary version
  words so the JSON is self-describing when extracted from the file.
- `producer.core_version` - semantic version of the core that wrote the file.
- `producer.git_hash` - 40-hex commit hash embedded at build configure time,
  or `"unknown"` when the build had no git available (e.g. sdist builds).
- `config_sha256` - SHA-256 of the canonical resolved mission config
  (FR-15), computed by the Python validator and passed through the core
  untouched. This is the reproducibility anchor binding the log to its exact
  inputs.
- `master_seed` - the D-9 master seed, decimal u64 as a string.
- `oracle` - the FR-25 oracle debug flag; an oracle run is identifiable from
  the header alone.
- `epoch_utc` - the mission epoch, carried **verbatim** from the mission
  file. The core performs no time parsing or conversion (D-2 boundary rule).
- `central_body` - `"earth"` in v1.0 files written by the Phase 1 core.
- `groups` - the channel dictionary that makes the file self-describing.
  Readers derive record layout entirely from this array; nothing about group
  or channel structure is hard-coded in a conforming reader. `rate_hz` is an
  integer; `rate_hz: 0` marks an aperiodic (event) stream. For every
  fixed-rate group, `rate_hz` MUST be an exact integer divisor of the `truth`
  group's rate: group records are decimated from the truth grid, never
  interpolated (FR-16).

Unknown JSON keys MUST be ignored by readers - that is how additive
minor-version evolution works (section 6).

### 3.1 Vehicle channel groups (v1.1)

Version 1.1 adds three **optional** fixed-rate groups (FR-16). A file
contains a group only when the run's configuration enabled it at
header-write time; when present, enabled groups appear in the `groups` array
in the fixed order `forces`, `mass`, `env`, after `truth` and `events`, so
the header bytes are a pure function of the configuration. Default rates per
FR-16: `truth` 10 Hz, `forces`/`mass`/`env` 1 Hz.

#### `forces` - per-source force/torque (the model-scrutiny channel)

The channel set is derived from the run-configurable **source subset**,
chosen at header-write time from the canonical source vocabulary below. For
each enabled source `<src>`, in declaration order, the group carries the
channel pair:

| Channel | dtype | units | frame | Content |
|---|---|---|---|---|
| `t_s` | `f64` | `s` | | record time |
| `f_<src>_b_n` | `f64[3]` | `N` | `body` | body-frame force from `<src>` |
| `tq_<src>_b_nm` | `f64[3]` | `N*m` | `body` | body-frame torque from `<src>`, about the composite CG |

Every source carries both channels so the record stays fixed-stride and the
dictionary uniform; a source that physically produces no force (e.g.
`wheel`) or no torque about the CG (e.g. point-mass `gravity`) logs zeros
there.

**Canonical source vocabulary** - the only names a conforming writer may
declare, and their canonical order. The enabled subset MUST be declared in
this order, without duplicates, so identical configurations yield identical
headers:

| Source | Meaning |
|---|---|
| `gravity` | central-body gravity, all configured harmonic tiers |
| `thirdbody` | summed third-body perturbations |
| `srp` | solar radiation pressure |
| `drag` | cannonball orbital drag |
| `aero` | vehicle aerodynamics (Mach-table force and moment) |
| `thrust` | main-engine thrust, including TVC deflection |
| `rcs` | reaction-control thrusters |
| `gravgrad` | gravity-gradient torque |
| `wheel` | reaction-wheel reaction torque |

Extending this vocabulary is a minor version bump: readers key on channel
names from the dictionary and MUST NOT assume the list above is exhaustive.

#### `mass` - composite mass properties

Logged at the forces rate by default so depletion CG travel and staging
jumps are inspectable in post-processing (FR-16).

| Channel | dtype | units | frame | Content |
|---|---|---|---|---|
| `t_s` | `f64` | `s` | | record time |
| `mass_kg` | `f64` | `kg` | | composite vehicle mass |
| `cg_b_m` | `f64[3]` | `m` | `body` | CG position in the body (structural) frame |
| `inertia_b_kgm2` | `f64[6]` | `kg*m^2` | `body` | composite inertia tensor about the CG, body axes |

`inertia_b_kgm2` packs the six unique elements of the symmetric tensor in
**row-major upper-triangle order**: `[Ixx, Ixy, Ixz, Iyy, Iyz, Izz]` - the
same packed-upper-triangle convention FR-26 fixes for `nav.est.P`.

#### `env` - flight-environment scalars

| Channel | dtype | units | frame | Content |
|---|---|---|---|---|
| `t_s` | `f64` | `s` | | record time |
| `alt_m` | `f64` | `m` | | geodetic altitude above the central body's reference ellipsoid |
| `mach` | `f64` | `1` | | Mach number of the atmosphere-relative velocity |
| `q_pa` | `f64` | `Pa` | | dynamic pressure of the atmosphere-relative velocity |
| `rho_kgpm3` | `f64` | `kg/m^3` | | local atmospheric density |
| `fpa_rad` | `f64` | `rad` | | flight-path angle of the velocity vector above the local horizontal |

This document is normative for channel names, dtypes, units, and ordering;
the physics definitions behind the logged values (which velocity convention
feeds `mach`/`q_pa`/`fpa_rad`, which atmosphere model feeds `rho_kgpm3`) are
normative in the producing models' math-library chapters.

## 4. Data types

| dtype string | Payload size | Encoding |
|---|---|---|
| `f64` | 8 B | IEEE-754 binary64, little-endian |
| `f64[N]` | 8·N B | N consecutive `f64`; N is a positive decimal with no leading zero |
| `u32` | 4 B | unsigned 32-bit, little-endian |
| `u64` | 8 B | unsigned 64-bit, little-endian |
| `str16` | 2 + n B | u16 byte length n, then n UTF-8 bytes |

The `f64[N]` grammar is size-self-describing, so readers parse the payload
size from the dtype string itself. v1.0 files use only `f64[3]` and
`f64[4]`; v1.1 adds `f64[6]` (the `mass` group's packed inertia tensor,
section 3.1). The reference reader has parsed the general `f64[N]` family
since v1.0, so this is an additive clarification, not a layout change.

`str16` is variable-length and is **allowed only in the `events` group**;
fixed-rate groups must remain fixed-stride so readers can preallocate.
A reader encountering an unknown dtype string must reject the file as
corrupt (it cannot know the payload size, so it cannot skip the channel).

## 5. Record stream

Repeated records, each:

- u16 `group_index` - index into the header's `groups` array;
- payload - the group's channels, in declared order, encoded per section 4.

Records appear in the order the core emitted them. Within each group, `t_s`
is strictly increasing. No count field exists anywhere: readers consume
records until EOF, and a trailing partial record is corruption.

### Phase 1 (two-body placeholder) semantics

- `truth` records are written at `truth_rate_hz`: one record at t = 0 and one
  every 1/`truth_rate_hz` thereafter, **decimated from integrator steps,
  never interpolated** (the Python validator guarantees
  `1/(dt_s * truth_rate_hz)` is an exact positive integer).
- Exactly two `events` records: code 1, detail `"run_start"`, at t = 0; and
  code 2, detail `"run_end"`, at t = duration.
- **Placeholder attitude channels.** The Phase 1 core propagates no attitude
  dynamics; `q_i2b` is written as the identity quaternion `[1,0,0,0]`
  (Hamilton, **scalar-first**, per D-7), `w_b_radps` as zeros, and `mass_kg`
  as the constant configured mass. These channels exist now so the `truth`
  group schema is stable from the first release: when attitude dynamics land
  (Phase 4), the channel dictionary does not change and old readers keep
  working. Consumers must not interpret Phase 1 attitude values as physics.

### Vehicle group semantics (v1.1)

- `forces`, `mass`, and `env` records follow the `truth` decimation rule:
  one record at t = 0 and one every 1/`rate_hz` thereafter, decimated from
  integrator steps, never interpolated (FR-16). Each group's `rate_hz` is an
  exact integer divisor of the `truth` rate, enforced by the writer at
  header-write time.
- Within each group, `t_s` is strictly increasing (the section 5 rule);
  records of different groups may interleave freely in emission order.
- A `forces` record carries one `(f_<src>_b_n, tq_<src>_b_nm)` pair per
  declared source, in declaration order - exactly the channels the header
  dictionary lists, like any other group.

## 6. Versioning

- **Minor version bump** = additive change only: new channels or new groups,
  new header JSON keys. Layout of existing groups never changes within a
  major version. Readers MUST read files whose minor version is newer than
  they know, driving layout purely from the channel dictionary and ignoring
  unknown JSON keys.
- **Major version bump** = layout break. Readers MUST refuse a major version
  they do not implement, loudly, naming both versions; the CLI exits nonzero.

### 6.1 Version history

- **1.0** (Phase 1) - initial format: header, `truth` + `events` groups,
  dtypes `f64`, `f64[3]`, `f64[4]`, `u32`, `u64`, `str16`.
- **1.1** (Phase 4) - additive: the optional `forces`, `mass`, and `env`
  vehicle channel groups (section 3.1), the canonical force-source
  vocabulary, and the `f64[6]` dtype (section 4). No existing group's layout
  changed; a v1.1 file that enables none of the new groups differs from a
  v1.0 file only in the version words. v1.0 readers read v1.1 files by the
  dictionary-driven rule above.

### Reserved group names

The following group names are **reserved** for the sensors/GNC phases and are
absent from v1.0 files: `nav.est`, `nav.err`, `nav.innov`, and the `sensors.*`
namespace (every name beginning `sensors.`). Adding them is a minor version
bump, not a format break (FR-16); the FR-26 consistency channels land inside
these groups by name. Third-party writers must not use these names for
anything else.

## 7. What is never in the file

Per D-11, the file contains **no wall-clock time, no hostname, no username,
no filesystem paths, no environment data** - nothing that varies between two
runs with identical inputs and binary. Run metadata of that kind lives in the
`meta.json` sidecar written by the Python CLI, never by the core. This is
what makes the double-run SHA-256 gate (FR-21) a meaningful whole-file check.

## 8. Reader error behavior (normative for `star_reacher.load`)

| Condition | Behavior |
|---|---|
| `version_major` != 1 | `SrlogVersionError` naming both versions; CLI exits nonzero |
| `version_minor` newer than reader | read normally (layout is dictionary-driven) |
| Bad magic | `SrlogCorruptError`; CLI exits nonzero |
| Truncated or undecodable header JSON | `SrlogCorruptError` |
| `header_json_len` beyond EOF | `SrlogCorruptError` |
| Unknown dtype string | `SrlogCorruptError` naming the dtype |
| Trailing partial record | `SrlogCorruptError` (append-only truncation is corruption) |
| Unknown header JSON keys | ignored (additive evolution) |
