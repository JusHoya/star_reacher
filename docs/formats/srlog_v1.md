# SRLOG v1 binary log format

Normative specification of the SRLOG on-disk format (PRD D-11, FR-16) as
written by the C++ core (`cpp/src/srlog_writer.cpp`) and read by the Python
loader (`star_reacher.load`). Writer and reader are both implemented against
this document; a change here is a format change and follows the versioning
rules in section 6. The current version is **1.2** (version history in
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
| 10-11 | u16 | `version_minor` = 3 (the version this document specifies; readers accept any minor, section 6) |
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
{"format":{"name":"SRLOG","major":1,"minor":3},
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
- `gnc` *(v1.2, optional)* - present exactly when the run declares any GNC
  channel group (section 3.2). Fixed key order:
  `{"cycle_rate_hz":<int>,"latency_cycles":<int>,"sensors":[<kind>...]}`.
  `cycle_rate_hz` is the control-cycle rate that anchors every periodic
  v1.2 group rate; `latency_cycles` is the FR-25 command-application delay
  in control cycles, echoed here so a latency study is identifiable from
  the header alone; `sensors` lists the declared sensor kinds in canonical
  order and is the identity table `nav.innov.sensor_id` indexes. In v1.3
  the object carries a fourth key, `camera`, after `sensors` and present
  exactly when a `sensors.camera` group is declared (section 3.2).
- `gnc.camera` *(v1.3, optional)* - the camera intrinsics and extrinsics,
  echoed once so a consumer composes `eq:camera:pose` and `eq:camera:K`
  from the log alone. Fixed key order:
  `{"float_encoding":"ieee754-binary64-hex","width_px":<int>,"height_px":<int>,"fx_px":<hex>,"fy_px":<hex>,"cx_px":<hex>,"cy_px":<hex>,"q_b2c":[<hex>x4],"r_cam_b_m":[<hex>x3]}`.
  `q_b2c` is Hamilton scalar-first (D-7), body to camera; `r_cam_b_m` is
  the mount station relative to the composite CG in body axes, metres.

  **The `ieee754-binary64-hex` encoding.** Floats never appear in this
  header (see above), so each double rides as its IEEE-754 binary64
  interchange encoding rendered as **16 lowercase hex digits, most
  significant nibble first** - the integer value of the encoding, so the
  rendering is endianness-free. The writer's encoder is pure integer
  shifting and a nibble lookup: no float formatter, rounding mode, or
  locale can perturb the header bytes, which is what keeps the FR-21
  whole-file determinism gate meaningful. The encoding is also exact,
  which a shortest-round-trip decimal is only when the reader's parser is
  correctly rounded. `star_reacher.decode_f64_hex` decodes one value or a
  list of them; `star_reacher.camera_echo` returns the whole object with
  its floats already decoded. `float_encoding` is emitted first and names
  the encoding, so a consumer never infers it from a format version it may
  not know; a reader MUST reject an encoding string it does not implement
  rather than guess.

  Landmark **positions** are not echoed. The landmark count L already
  rides in the `sensors.camera` record dtype `f64[2L]`, and a
  fixture-scale landmark table would add kilobytes to every camera header
  to restate configuration; landmark positions remain resolved-config
  data.
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

### 3.2 GNC channel groups (v1.2)

Version 1.2 populates the group names reserved since v1.0 (section 6) with
ten **optional** groups: the six `sensors.<kind>` groups, `nav.est`,
`nav.err`, `nav.innov`, and `gnc.cmd`. A file contains a group only when the
run's configuration declared it at header-write time; declared groups appear
in the `groups` array after the v1.1 vehicle groups, in the fixed order

```
sensors.imu, sensors.startracker, sensors.sunsensor, sensors.navfix,
sensors.altimeter, sensors.camera, nav.est, nav.err, nav.innov, gnc.cmd
```

so the header bytes remain a pure function of the configuration. Whenever
any of these groups is declared, the header carries the top-level `gnc`
object (section 3) with the control-cycle rate, the latency setting, and
the declared sensor list.

**Rate rule.** v1.2 records are produced on the **control-cycle grid** (the
D-5 major cycle), not the truth grid: every periodic v1.2 group's `rate_hz`
MUST be an exact integer divisor of `gnc.cycle_rate_hz` (decimation only,
never interpolation — the same discipline the v1.1 groups follow against
the truth rate). `nav.innov` is aperiodic (`rate_hz: 0`), one record per
aiding update. No divisor relation between `cycle_rate_hz` and the `truth`
rate is imposed by the format; both grids share t = 0.

**Sensor kinds and instances.** The canonical sensor-kind vocabulary, in
canonical order, is `imu`, `startracker`, `sunsensor`, `navfix`,
`altimeter`, `camera` (FR-23). A v1.2 file declares at most one group per
kind; supporting multiple instances of one kind is a future minor bump.
Each sensor group's rate is that sensor's sample rate, which the producing
core additionally constrains to divide the control rate (a sensor is
sampled every `cycle_rate_hz / rate_hz` cycles, on the cycle grid).

**Record-start semantics.** Sensor groups emit their first record at
t = 1/`rate_hz` after GNC activation (an accumulated increment over an
empty interval does not exist), then one record per sample instant.
`nav.est`, `nav.err`, and `gnc.cmd` emit one record per control cycle from
GNC activation onward. GNC activation is t = 0 for missions that start
free-flying and the pad-release instant for geodetic launch missions (the
vehicle is structurally constrained before release; no command is applied,
so none is logged). Within each group `t_s` is strictly increasing
(section 5); a group whose records begin after t = 0 is well-formed.

#### `sensors.imu` - accumulated inertial increments

| Channel | dtype | units | frame | Content |
|---|---|---|---|---|
| `t_s` | `f64` | `s` | | sample time (end of the accumulation interval) |
| `dtheta_b_rad` | `f64[3]` | `rad` | `body` | integral of the true body rate over the sample interval |
| `dv_b_mps` | `f64[3]` | `m/s` | `body` | integral of the true specific force (body frame) over the sample interval |

The v1 IMU emits **one increment pair per control cycle**: its declared
rate equals the header's `cycle_rate_hz` (faster- or slower-than-cycle IMU
output is out of scope for v1; the sensor chapter's assumption 1). The
v1.2 reference implementation is the **ideal IMU**: zero errors, with the
truth integrals evaluated by **trapezoidal accumulation over the accepted
integrator steps** tiling each sample interval (the steps terminate on
cycle boundaries by construction, so the endpoint pairs are well defined;
eq:imu:quadrature in the sensor chapter). Specific force is the total
**non-gravitational** acceleration of the center of mass resolved in body
axes — thrust, vehicle aerodynamics, SRP, and drag are sensed; gravitation
(central body plus configured third bodies) is not (eq:imu:specificforce:
an accelerometer in free fall reads zero). The full FR-23 error model
(bias, scale factor, misalignment, ARW/VRW, quantization) lands in a later
minor revision of the producing core without changing this record layout.

#### `sensors.startracker`, `sensors.sunsensor`, `sensors.navfix`, `sensors.altimeter`

| Group | Channels after `t_s` |
|---|---|
| `sensors.startracker` | `q_meas_i2b` `f64[4]` (Hamilton scalar-first, GCRF->body), `valid` `u32` (exclusion/slew gating; 1 = valid) |
| `sensors.sunsensor` | `sun_b` `f64[3]` (measured Sun unit vector, body frame), `valid` `u32` (field-of-view gating) |
| `sensors.navfix` | `r_meas_m` `f64[3]` (GCRF), `v_meas_mps` `f64[3]` (GCRF) |
| `sensors.altimeter` | `alt_meas_m` `f64` (m) |

These layouts are normative now so parallel consumers can build against
them; their producing sensor models land in a later workstream against the
same `ISensor` interface.

#### `sensors.camera` - geometric-truth camera hook

| Channel | dtype | units | frame | Content |
|---|---|---|---|---|
| `t_s` | `f64` | `s` | | sample time |
| `r_m` | `f64[3]` | `m` | `GCRF` | camera (vehicle) position |
| `q_i2b` | `f64[4]` | `1` | GCRF->body Hamilton scalar-first | camera (vehicle) attitude |
| `px_uv` | `f64[2L]` | `px` | `image` | *(only when the configuration declares L > 0 landmarks)* interleaved pixel pairs `u0, v0, u1, v1, ...` in the configured landmark order |

The landmark count L is fixed at header-write time. Camera intrinsics and
extrinsics are not record channels — they are constants, so repeating them
per sample would be waste — but as of v1.3 they are **echoed once in the
header** at `gnc.camera` (section 3), which is what lets a consumer form
`eq:camera:pose` and `eq:camera:K` without the mission file. That matters
because FR-23 scopes this hook to geometric truth for *offline external
rendering*: the pose channels carry the vehicle state rather than the
camera's, so before the echo a renderer could not place the camera or
build its projection matrix from the file at all. Landmark positions
remain resolved-config data (section 3). Per FR-23 the camera hook emits
geometric truth only — no in-core rendering.

#### `nav.est` - estimator state and covariance (FR-26)

| Channel | dtype | units | frame | Content |
|---|---|---|---|---|
| `t_s` | `f64` | `s` | | control-cycle time |
| `x_hat` | `f64[n]` | | | estimator state vector |
| `P` | `f64[m(m+1)/2]` | | | covariance, packed **row-major upper triangle** (the same packing as the `mass` group's inertia tensor) |

The state dimension n and the covariance dimension m are declared at
header-write time and are estimator-defined; the estimator's math-library
chapter is normative for the meaning and units of the components. m
defaults to n and differs only for estimators whose covariance lives in a
different parameterization than the state: the reference error-state EKF
(a later workstream) declares n = 16 (`x_hat` = attitude quaternion,
Hamilton scalar-first, then velocity, position, gyro bias, accelerometer
bias) with m = 15 (the 3-component attitude error replaces the
4-component quaternion), so its `P` carries 120 doubles. The reference
dead-reckoning navigator logs n = m = 7:
`x_hat = [q_w, q_x, q_y, q_z, w_x, w_y, w_z]` (attitude quaternion,
Hamilton scalar-first, then body rate in rad/s) with P identically zero
(dead reckoning carries no covariance).

#### `nav.err` - truth-minus-estimate error state

| Channel | dtype | units | frame | Content |
|---|---|---|---|---|
| `t_s` | `f64` | `s` | | control-cycle time |
| `e` | `f64[n]` | | | truth minus estimate, in the estimator's own state convention |

`nav.err` exists only alongside `nav.est`, **at the same rate and with the
state dimension n** — record counts match by construction. The converse does
not hold: `nav.est` may be declared without `nav.err`. The error state is
computed by the loop from the layout the estimator declares for its state
vector (`gnc::ErrorBlock`, so that the estimator is never handed the true
state — FR-24), and an estimator that declares no layout is written no
`nav.err` group rather than a group of zeros, which would be
indistinguishable from a perfect estimate. A reader must therefore test for
the group's presence rather than infer it from `nav.est`.

The `nav.err` contract is with the `star consistency` tooling, which computes NEES from
`nav.err.e` and `nav.est.P` directly when m = n (per-epoch NEES ~
chi-square(n); ensemble mean over R runs gated against two-sided 95 %
chi-square(Rn)/R bounds); when the declared covariance dimension m differs
from n, the estimator's chapter defines the m-dimensional error the NEES
uses. One such reduction is defined and implemented: for a **quaternion-led
error state** (m = n - 1, the leading four components of `e` being an error
quaternion in the estimator's multiplicative convention, scalar-first and
canonicalized to the +w hemisphere), the evaluator collapses those four to
three by

    dtheta = 2 * sign(dq_w) * dq_v

and passes the remaining n - 4 components through unchanged. The reference
error-state EKF is exactly this case, n = 16 against m = 15. Any other
pairing of n and m is reported as a mismatch rather than guessed at. No
state-to-truth mapping is defined in the file. The error is
computed in-core from truth for analysis only; truth never enters the GNC
components' inputs unless the scenario sets the `oracle` flag (FR-25).
For quaternion-bearing states the producing estimator aligns the truth
quaternion's sign to the estimate (q and -q encode the same attitude)
before differencing, so `e` is continuous.

#### `nav.innov` - per-update innovations (FR-26)

Aperiodic (`rate_hz: 0`): one record per aiding-sensor update, at the cycle
time the update was applied.

| Channel | dtype | units | frame | Content |
|---|---|---|---|---|
| `t_s` | `f64` | `s` | | update time |
| `sensor_id` | `u32` | `1` | | index into the header `gnc.sensors` array naming the aiding sensor |
| `m` | `u32` | `1` | | valid innovation dimension of this update (1 <= m <= m_max) |
| `y` | `f64[m_max]` | | | innovation vector; entries beyond `m` are zero |
| `S` | `f64[m_max(m_max+1)/2]` | | | innovation covariance, packed row-major upper triangle; entries whose row or column is beyond `m` are zero |

`m_max`, the maximum innovation dimension across the configured aiding
sensors, is declared at header-write time so the record stays fixed-stride
while updates of different dimensions share one group.

#### `gnc.cmd` - commands as applied (FR-25)

One record per control cycle: the command **as applied** that cycle —
after the `latency_cycles` FIFO and after actuator-limit saturation. This
group is the instrument for the latency exit criterion: `latency_cycles = k`
must visibly shift command application here by exactly k cycles.

| Channel | dtype | units | frame | Content |
|---|---|---|---|---|
| `t_s` | `f64` | `s` | | cycle time at application |
| `tau_b_nm` | `f64[3]` | `N*m` | `body` | applied commanded body torque |
| `q_cmd_i2b` | `f64[4]` | `1` | GCRF->body Hamilton scalar-first | applied commanded attitude |
| `w_cmd_b_radps` | `f64[3]` | `rad/s` | `body` | applied commanded body rate |
| `valid` | `u32` | `1` | | 1 = a fresh GNC output was applied; 0 = hold (the pre-fill of the latency FIFO, or a component that declared its output invalid); the held values are repeated in the record |

**Authority scoping.** Phase 6 GNC commands attitude torque only: the
applied `tau_b_nm` drives the vehicle's torque-driven attitude dynamics
(Euler's equations with the composite stack inertia, integrated per control
cycle under zero-order hold, D-5). Sequence-driven propulsion — ignition,
throttle, staging, jettison — keeps its Phase 4 authority and is not
commandable through the GNC chain. Environmental torques (aero,
gravity-gradient) are logged in the `forces` group but are not coupled into
the v1.2 attitude integration, matching the Phase 4 fidelity level where
attitude was fully kinematic; the coupling is a later, documented change to
the producing core, not to this format.

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
section 3.1); v1.2 uses the general family with run-dependent N (the
`nav.*` state/covariance channels and the camera landmark channel,
section 3.2). The reference reader has parsed the general `f64[N]` family
since v1.0, so none of this is a layout change.

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

### GNC group semantics (v1.2)

- Periodic v1.2 groups (`sensors.*`, `nav.est`, `nav.err`, `gnc.cmd`) emit
  on the control-cycle grid at their declared rates, with the record-start
  semantics of section 3.2 (sensor groups begin at their first sample
  instant; `nav.*`/`gnc.cmd` begin at GNC activation).
- `nav.innov` is aperiodic: records appear only when an aiding update was
  applied, in application order; `t_s` is non-decreasing and may repeat
  across records when two sensors update on the same cycle (each record is
  distinguished by `sensor_id`). This is the one deliberate exception to
  the strictly-increasing `t_s` rule of section 5, mirroring the `events`
  stream's aperiodic character.
- `nav.est` and `nav.err` carry the same number of records, at identical
  `t_s` values, by construction (section 3.2).

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
- **1.2** (Phase 6) - additive: the optional GNC channel groups
  `sensors.imu`, `sensors.startracker`, `sensors.sunsensor`,
  `sensors.navfix`, `sensors.altimeter`, `sensors.camera`, `nav.est`,
  `nav.err`, `nav.innov`, and `gnc.cmd` (section 3.2), the canonical
  sensor-kind vocabulary, and the optional top-level header key `gnc`
  (section 3). These populate the names reserved since v1.0, so the schema
  major is unchanged by design (the reservation worked). No existing
  group's layout changed; a v1.2 file that declares none of the new groups
  differs from a v1.1 file only in the version words. Older readers read
  v1.2 files by the dictionary-driven rule above.
- **1.3** (Phase 6) - additive: the optional header key `gnc.camera`
  (section 3), echoing the camera intrinsics and extrinsics so a camera log
  is self-contained, together with the `ieee754-binary64-hex` value
  encoding it introduces. No group's layout changed and no channel was
  added or removed: a v1.3 file that declares no camera sensor is
  byte-identical to the v1.2 file the same configuration produced apart
  from the version words. This is a minor bump because a new header JSON
  key is one by the rule above, not because anything a v1.2 reader
  consumes moved — such a reader ignores the new key and reads the file
  unchanged.

### Reserved group names

The group names `nav.est`, `nav.err`, `nav.innov`, and the `sensors.*`
namespace (every name beginning `sensors.`) were **reserved** from v1.0 for
the sensors/GNC phases; v1.2 defines them (section 3.2). Within the
`sensors.*` namespace, names beyond the six canonical kinds remain reserved
for future sensor instances. Adding reserved groups is a minor version
bump, not a format break (FR-16); the FR-26 consistency channels
(`nav.est.x_hat`, `nav.est.P`, `nav.innov.y`, `nav.innov.S`) live inside
these groups by exactly these names. Third-party writers must not use these
names for anything else.

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
