# SRLOG v1.0 binary log format

Normative specification of the SRLOG on-disk format (PRD D-11, FR-16) as
written by the C++ core (`cpp/src/srlog_writer.cpp`) and read by the Python
loader (`star_reacher.load`). Writer and reader are both implemented against
this document; a change here is a format change and follows the versioning
rules in section 6.

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
| 10-11 | u16 | `version_minor` = 0 |
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
{"format":{"name":"SRLOG","major":1,"minor":0},
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
  integer; `rate_hz: 0` marks an aperiodic (event) stream.

Unknown JSON keys MUST be ignored by readers - that is how additive
minor-version evolution works (section 6).

## 4. Data types

| dtype string | Payload size | Encoding |
|---|---|---|
| `f64` | 8 B | IEEE-754 binary64, little-endian |
| `f64[3]` | 24 B | three consecutive `f64` |
| `f64[4]` | 32 B | four consecutive `f64` |
| `u32` | 4 B | unsigned 32-bit, little-endian |
| `u64` | 8 B | unsigned 64-bit, little-endian |
| `str16` | 2 + n B | u16 byte length n, then n UTF-8 bytes |

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

## 6. Versioning

- **Minor version bump** = additive change only: new channels or new groups,
  new header JSON keys. Layout of existing groups never changes within a
  major version. Readers MUST read files whose minor version is newer than
  they know, driving layout purely from the channel dictionary and ignoring
  unknown JSON keys.
- **Major version bump** = layout break. Readers MUST refuse a major version
  they do not implement, loudly, naming both versions; the CLI exits nonzero.

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
