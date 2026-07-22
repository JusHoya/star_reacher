# Phase 6 independent code review

Scope: the `main...phase-6-gnc-sensors` diff, reviewed as code. This is a
defect review — correctness, determinism, and latent traps. It is not an
evidence or acceptance-gate audit; that is covered separately by
`docs/audit/phase6_evidence_audit.md`.

Review base: `phase-6-gnc-sensors` at `ea0e5e9`, worktree branch
`ws-p6-review`. Nothing was compiled during this review — another agent held
the sole compiler slot throughout. Every finding below is nonetheless marked
CONFIRMED, because each rests on an airtight static read or on an executed
pure-Python demonstration; none required a build to establish. The closing
method note states what a build would still be worth running, and records the
one hypothesis that traced out weaker than first assumed.

## Headline result: the `auto` + Eigen hunt found nothing

The standing project lesson is that an Eigen expression template bound to
`auto` outlives its temporary operands and reads freed memory — benign on one
compiler, garbage on another, and visible only under a cross-compiler
`-Werror` build. Every line of this phase has been compiled by MSVC release
only; the Linux `-Werror` doctest leg has never run on it. That made this the
highest-value thing to look for.

Every `auto` in the phase's new and changed C++ was enumerated and classified:

| File | `auto` uses | Classification |
| --- | --- | --- |
| `cpp/src/gnc/ekf.cpp` | 2 | `std::map` iterator, range-for over `std::map` |
| `cpp/src/gnc/builtin.cpp` | 4 | map iterators and range-for |
| `cpp/src/gnc/component.cpp` | 4 | map/registry iterators and range-for |
| `cpp/src/vehicle_cycle.cpp` | 4 | map iterator, two RHS lambdas, range-for over `std::unique_ptr` |
| `cpp/src/sensors/imu.cpp` | 5 | map iterators, range-for, one setup lambda |
| `cpp/src/sensors/optical.cpp` | 7 | map iterators and range-for |
| `cpp/src/sensors/radio.cpp` | 6 | map iterators and range-for |
| `cpp/src/sensors/camera.cpp` | 6 | map iterators and range-for |
| `cpp/src/models/environment.cpp` | 0 | — |
| `cpp/src/srlog_writer.cpp` | 0 | — |

**Not one `auto` is bound to an Eigen expression.** Every Eigen intermediate
in `ekf.cpp` — the dense, Eigen-heavy file that was the primary concern — is
bound to a named concrete type (`Eigen::Vector3d`, `Eigen::Matrix3d`,
`Matrix15d`, `Eigen::Matrix<double, M, kM>`, …). This is disciplined code and
the discipline is uniform, not accidental.

Two adjacent shapes were checked and are also clean:

- **Aliasing.** `symmetrize()` (`cpp/src/gnc/ekf.cpp:90`) writes
  `p = 0.5 * (p + p.transpose()).eval()`. The `.eval()` materialises the
  transposed sum into a temporary before the assignment reads it, so the
  transpose aliasing hazard is correctly defused. `joseph_update()`
  (`cpp/src/gnc/ekf.cpp:377-397`) assigns through the named intermediate
  `p_post`, never into `p_` from an expression containing `p_`.
- **References outliving temporaries.** The only two `const Eigen::…&`
  bindings in the phase — `cpp/src/gnc/component.cpp:230` and
  `cpp/src/sensors/camera.cpp:121` — bind to members of a live object, not to
  temporaries.

The pybind11 trampoline deserves specific credit here. Both interface methods
that return a reference to a container — `innovations()` and
`error_layout()` — are the textbook pybind11 dangling-reference footgun, and
both are correctly cached in `mutable` members (`bindings/module.cpp:518-526`,
`548-556`, members at `569-570`) rather than returning a reference to the
caster's temporary.

The `auto` + Eigen hunt is a genuine negative result across every file listed
above.

## Findings, ranked

| # | Finding | Severity | Confidence |
| --- | --- | --- | --- |
| 1 | Batch and stepped runs configure sensors in different orders | HIGH | CONFIRMED |
| 2 | Unvalidated innovation payload overflows fixed log buffers | HIGH | CONFIRMED |
| 3 | `copy_fixed` validates against a Python-controlled dimension, not the buffer size | HIGH | CONFIRMED |
| 4 | `star consistency` reports PASS when the innovation channel is empty | HIGH | CONFIRMED |
| 5 | `~SrlogWriter()` calls a throwing `close()` -> `std::terminate` | MEDIUM-HIGH | CONFIRMED |
| 6 | `Sim` cannot release its log, and a second `reset()` fails by default | MEDIUM | CONFIRMED |
| 7 | The plugin module cache is never invalidated while `meta.json` re-hashes | MEDIUM | CONFIRMED |
| 8 | The nav fix reports `valid = true` before it has ever been sampled | MEDIUM | CONFIRMED |
| 9 | The altimeter update is silently discarded without a body-fixed frame | MEDIUM | CONFIRMED |
| 10 | `LDLT::info()` is never checked on the innovation covariance | MEDIUM | CONFIRMED |
| 11 | An invalid IMU sample freezes the covariance while time advances | MEDIUM | CONFIRMED |
| 12 | The ensemble chi-square dimension is taken from run 0 only | MEDIUM | CONFIRMED |
| 13 | `_reduce_error` guesses the quaternion collapse whenever `n == m + 1` | MEDIUM | CONFIRMED |
| 14 | `write_nav_innov` does not bound-check `sensor_id` | MEDIUM | CONFIRMED |
| 15 | The ellipsoid is defined twice with incompatible sphere sentinels | MEDIUM (latent) | CONFIRMED |
| 16 | `_apply_override` silently truncates a float to an integer leaf | LOW | CONFIRMED |
| 17 | `_load_module` swallows `KeyboardInterrupt` and `SystemExit` | LOW | CONFIRMED |
| 18 | `sensor_index_` brace initialiser breaks silently on a 7th sensor kind | LOW (latent) | CONFIRMED |
| 19 | Three `SrlogHeaderFields` members have no default initialiser | LOW (latent) | CONFIRMED |
| 20 | Unguarded division by `r^3` in the filter's gravity model | LOW | CONFIRMED |
| 21 | Sun sensor emits a normalised pure-noise direction with no ephemeris | LOW | CONFIRMED |
| 22 | Minor and dead logic | informational | CONFIRMED |

Findings 5, 19, and the latent half of 18 are **pre-existing code, not
introduced by Phase 6** - verified absent from the phase diff hunks. They are
reported because Phase 6 newly exposes them: the stepping API makes the
destructor path (5) a routine occurrence rather than an error path, and the
FR-23 sensor vocabulary is what 18 indexes.

**Finding 1 is the one to fix first.** It is the only finding that misbehaves
on a correct, shipped mission today, with no plugin and no unusual
configuration: `missions/leo_ekf_consistency.toml` produces different log
bytes, different sensor labels, and a genuinely different filter trajectory
depending on whether it is run through `star run` or through the stepping API.

Findings 2 and 3 are the ones to fix before the Linux `-Werror` doctest leg
first runs. Both are memory-safety defects on the FR-25 plugin path, both are
invisible to MSVC release, and an ASan build will find them immediately.

### 1. Batch and stepped runs configure sensors in different orders (CONFIRMED, HIGH)

**Where:** `python/star_reacher/runner.py:383`, reached with differently-ordered
input from `python/star_reacher/runner.py:450` (batch) and
`python/star_reacher/sim.py:419` (stepped).

`build_run_config` builds the core's sensor list by iterating a dict:

```
for kind, spec in resolved["sensors"].items():
```

The two entry points hand it that dict in **different orders**:

- `run_mission` passes the validator's dict directly, whose `sensors` sub-dict
  was built by walking `_SENSOR_KINDS` in the canonical FR-23 vocabulary order
  (`python/star_reacher/mission.py:328-335`, `2156-2157`).
- `Sim.reset()` passes the output of `_deep_copy_resolved`
  (`python/star_reacher/sim.py:419`), which round-trips through
  `canonical_bytes`. That function uses `json.dumps(..., sort_keys=True)`
  (`python/star_reacher/mission.py:2860-2862`), so the reconstructed dict is in
  **alphabetical** key order.

For the phase's flagship EKF mission, `missions/leo_ekf_consistency.toml`,
which configures `imu`, `navfix`, `startracker`, and `altimeter`:

```
run_mission (validator canonical): ['imu', 'startracker', 'navfix', 'altimeter']
Sim.reset  (sort_keys round trip): ['altimeter', 'imu', 'navfix', 'startracker']
IDENTICAL? False
```

The core is order-sensitive in three separate places:

- `cpp/src/vehicle_cycle.cpp:821-826` — `sensor_list` construction order, and
  `id = sensor_list.size() - 1` assigns `navfix_id`, `startracker_id`, and
  `altimeter_id` **by index**. For this mission the star tracker is
  `sensor_id = 1` under `star run` and `sensor_id = 3` under `Sim`; the
  altimeter is `3` and `0` respectively.
- `cpp/src/vehicle_cycle.cpp:506-517` — `fields.sensors.push_back(decl)` sets
  the declared sensor array in the SRLOG header, so **the log bytes differ**
  between the two entry points for the same mission and seed.
- `cpp/src/vehicle_cycle.cpp:989` — the sampling and offer order, so the EKF's
  sequential aiding updates are applied in a different order within a cycle.
  `cpp/src/gnc/ekf.cpp:179-183` states explicitly that this order is normative
  because "the ensemble gate is only reproducible if the order is pinned".

**Failure scenario.** Run `missions/leo_ekf_consistency.toml` through
`star run` and through the stepping API with identical seeds. The two runs
produce different `run.srlog` bytes, different `sensor_id` labels on every
`nav.innov` record, and — because the EKF folds its three aiding updates in
list order against the running post-update covariance — genuinely different
state trajectories. `star consistency` then labels its NIS gates
`NIS[sensor k]` from the raw id, so the same physical instrument is reported
under a different label depending only on which entry point produced the log.

Noise realisations are **not** affected: `rng::make_stream` keys each
substream on the sensor kind name rather than the index
(`cpp/src/sensors/imu.cpp:270-291`), so the draws themselves are stable. The
divergence is in ordering and identity, not in the random stream.

**Why nothing catches it.** The acceptance gate for exit criterion 4 is
structurally blind: `verify.py`'s `_check_v022` uses an inline
`_P6_GNC_MISSION` fixture configuring only `[sensors.imu]`, and
`tests/python/test_sim.py:30` uses `missions/leo_attitude_gnc.toml`, also
IMU-only. With a single sensor the canonical and alphabetical orders coincide,
so no existing test can distinguish them. The claim in
`python/star_reacher/sim.py:3-8` that byte-identity is "a property of the
factoring rather than of a comparison test: there is only one implementation
to disagree with" is not correct as written — there is one *builder*, but two
callers feed it differently-ordered input.

**Remedy.** Make the order canonical at the one place it matters: in
`build_run_config`, iterate `_SENSOR_KINDS` rather than
`resolved["sensors"].items()`, so both entry points emit the same list
regardless of dict order. Then extend the V022 fixture to a multi-sensor
mission so the gate can actually fail.

### 2. Unvalidated innovation payload overflows fixed log buffers (CONFIRMED, HIGH)

**Where:** `cpp/src/vehicle_cycle.cpp:1110-1138` (writes), sized at
`cpp/src/vehicle_cycle.cpp:812-820`.

The nav-innovation log buffers are sized once, at GNC activation, from the
component's declared maximum:

```
innov_mm = nav->innov_max_dim();
innov_y_buf.assign(innov_mm, 0.0);
innov_s_buf.assign(innov_mm * (innov_mm + 1) / 2, 0.0);
```

Each cycle, every `InnovationSample` the component returns is copied into
those buffers with **no check that it fits**:

- `cpp/src/vehicle_cycle.cpp:1117` —
  `std::copy(s.y.begin(), s.y.end(), innov_y_buf.begin())` copies `s.y.size()`
  doubles into a buffer of `innov_mm`.
- `cpp/src/vehicle_cycle.cpp:1131` —
  `innov_s_buf[row0 + (j - i)] = s.s_upper[src++]` indexes the destination by
  `m = s.y.size()` and reads the source for `m(m+1)/2` entries.

`m` is taken from `s.y.size()` at `cpp/src/vehicle_cycle.cpp:1116` and is never
compared against `innov_mm`; `s.s_upper.size()` is never compared against
`m(m+1)/2`.

**Reachability.** This is reachable from pure Python with no unsafe API.
`InnovationSample` is exposed with a default constructor and read-write access
to both vectors (`bindings/module.cpp:1248-1257`), and a Python nav component
supplies both `innov_max_dim()` and `innovations()` through the trampoline
(`bindings/module.cpp:512-526`).

**Failure scenario.** A Python nav component that declares
`innov_max_dim() -> 1` but returns an `InnovationSample` with `y` of length 6
and `s_upper` of length 21 causes a write of 6 doubles into a 1-element vector
(40 bytes past the end) and a write into `innov_s_buf` at indices up to 20 in a
1-element vector (160 bytes past the end) — heap corruption on the first cycle
that applies an aiding update. The mirror case, declaring
`innov_max_dim() -> 6` and returning a short `s_upper`, is an out-of-bounds
**read** that silently writes uninitialised heap into the `nav.innov` channel.

**Why this reads as an oversight rather than a design choice.** Every other
variable-length quantity a Python component returns *is* length-checked:
`state()` and `covariance_upper()` go through `copy_fixed()`, which refuses a
wrong length by name (`bindings/module.cpp:454-468`), and `error_layout()` is
validated by `validate_error_layout()`, which requires the declared blocks to
tile `[0, state_dim)` exactly (`cpp/src/gnc/component.cpp:136-183`).
`innovations()` is the one gap in an otherwise complete perimeter.

The built-in `error_state_ekf` is safe — it declares `innov_max_dim() == 6`
(`cpp/src/gnc/ekf.cpp:210`) and its widest update is the 6-dimensional nav fix
— so no shipped configuration triggers this. It is a plugin-boundary defect.

**Remedy.** Validate in the loop, before either copy, that
`s.y.size() <= innov_mm` and `s.s_upper.size() == s.y.size() * (s.y.size() + 1) / 2`,
and throw a named `std::length_error` in the same style as `copy_fixed()`.
Validating in the loop rather than in the trampoline also covers a
hypothetical third-party C++ component registered through the same registry.

### 3. `copy_fixed` validates against a Python-controlled dimension, not the caller's buffer size (CONFIRMED, HIGH)

**Where:** `bindings/module.cpp:454-468`, called at `bindings/module.cpp:532`
and `540`; buffers sized at `cpp/src/vehicle_cycle.cpp:811-816`.

```
copy_fixed(ov(), "state", state_dim(), x_hat);        // module.cpp:532
copy_fixed(ov(), "covariance_upper", m*(m+1)/2, p);   // module.cpp:540
```

`copy_fixed()` checks the length of the list Python returned against
`state_dim()` / `cov_dim()`. Both of those **dispatch back into Python on every
call** — `int_override()` invokes the override afresh each time
(`bindings/module.cpp:561-567`), with no caching. But `x_hat` and `p` point at
buffers sized **once, at GNC activation**, from whatever those methods returned
*then* (`cpp/src/vehicle_cycle.cpp:811-816`).

So the validation compares a Python-supplied length against a Python-supplied
dimension. Both can change together, and the check passes while the
destination buffer stays the size it was at construction.

**Failure scenario.** A Python estimator whose state grows during the run —
`def state_dim(self): return len(self.x)` on an augmented or adaptive filter,
which is exactly the kind of component FR-25 exists to allow. At construction
`state_dim()` returns 6, so `x_hat_buf` is 6 doubles
(`cpp/src/vehicle_cycle.cpp:813`). On a later cycle the filter augments its
state: `state_dim()` now returns 12, `state()` returns 12 floats,
`copy_fixed()` compares 12 against 12 and **passes**, and `std::copy` then
writes 12 doubles into a 6-double heap buffer. `write_nav_est` does not catch
it either, because it re-reads `x_hat_buf.size()`
(`cpp/src/vehicle_cycle.cpp:1074`), which has not changed. Silent heap
corruption with no diagnostic.

The same shape applies to `covariance_upper()` via `cov_dim()`. Additionally,
`m * (m + 1) / 2` at `bindings/module.cpp:540` is `int` arithmetic: a
`cov_dim()` returning a large value overflows, and one returning a negative
value makes `copy_fixed()` demand a length of 0 or less and succeed vacuously.

Nothing in the interface documentation
(`cpp/include/star/gnc/component.hpp:352-373`) states that these dimensions
must be constant across a run. `error_layout()` does carry that clause; the
three integer dimensions do not.

**Remedy.** Cache `state_dim()`, `cov_dim()`, and `innov_max_dim()` in
`PyGncComponent` on first call and make any later divergence a hard error —
that enforces the constancy contract instead of assuming it — or pass the
caller's actual buffer length down rather than re-querying. Document the
constancy requirement in `component.hpp` either way.

### 4. `star consistency` reports PASS when the innovation channel is empty (CONFIRMED, HIGH)

**Where:** `python/star_reacher/consistency_cli.py:203` and `:407`.

`_group_innovations` builds its per-sensor groups from
`sorted({int(s) for s in innov["sensor_id"]})`. A `nav.innov` group that is
present but holds **zero records** yields an empty group dict and an empty
problem list — the emptiness is never flagged. `cmd_consistency` then sets
`sensor_ids = []`, never builds the NIS half of the series, and reports only
the NEES headline.

Driven end to end against a synthetic run with well-formed `nav.err` and
`nav.est` and a present-but-empty `nav.innov`:

```
CONSISTENCY: PASS (1/1 gates)
EXIT CODE: 0
```

**Failure scenario.** An estimator whose measurement update never fires —
because a sensor is never sampled, because every update is gated out (see
finding 9), or because of a wiring defect that leaves `innovations()` empty —
passes the FR-26 acceptance gate with exit code 0 and no warning. The report
prints one gate instead of four, and nothing in the output tells the reader
that NIS was skipped rather than passed. This is the archetypal silent-failure
shape: the gate that was supposed to detect a broken filter reports success
*because* the filter is broken.

**Remedy.** Treat an empty `nav.innov` group, and an empty per-sensor group,
as a problem in `_extract_arrays`. At minimum `cmd_consistency` must refuse to
report PASS when `sensor_ids` is empty.

### 5. `~SrlogWriter()` calls a throwing `close()`, so a failed flush terminates the process (CONFIRMED, MEDIUM-HIGH — pre-existing)

**Where:** `cpp/src/srlog_writer.cpp:531` and `533-544`; declaration at
`cpp/include/star/srlog_writer.hpp:247`.

```
SrlogWriter::~SrlogWriter() { close(); }

void SrlogWriter::close() {
  if (out_.is_open()) {
    out_.flush();
    if (!out_) { out_.close(); throw std::runtime_error("SRLOG writer: flush failed on close"); }
    out_.close();
  }
}
```

The destructor is declared without `noexcept(false)` and no base or member has
a throwing destructor, so it is implicitly `noexcept(true)`. A throw escaping
it calls `std::terminate()` — immediate process abort, no unwinding, no Python
traceback.

**Failure scenario.** The log is on a full disk or a network share that
drops; the final buffered block fails to flush; the writer is destroyed
through `~VehicleCycle`. The process aborts with a bare terminate message
instead of raising a Python exception that names the file and the cause. This
is precisely the I/O-error condition where a diagnostic matters most.

The inline comment at `cpp/src/srlog_writer.cpp:537-539` shows the destructor
path was considered — "Failing loudly here (not silently in the destructor
path) is why close() should be called explicitly on the success path" — but
the conclusion is inverted: in a destructor the throw is not "loud", it is
fatal.

`close()` is correctly idempotent (guarded by `is_open()`), so double-close is
safe. That part is fine.

**Remedy.** Catch and record in the destructor
(`try { close(); } catch (...) { }`), and surface the error from an explicit
close on the normal path — which finding 4 requires anyway.

### 6. `Sim` cannot release its log, and a second `reset()` fails by default (CONFIRMED, MEDIUM)

**Where:** the `Sim` binding at `bindings/module.cpp:1358-1421`;
`python/star_reacher/sim.py:228-231`.

The bound `Sim` surface is `__init__`, `step`, `observe`, `truth`, `time`,
`cycle`, `done`, `summary`, and `has_external_command`. There is **no
`close()` and no `__enter__`/`__exit__`**, and `VehicleCycle` exposes no public
close to bind (`cpp/include/star/vehicle_cycle.hpp:94-156`) — `writer.close()`
is reachable only through `finish()`
(`cpp/src/vehicle_cycle.cpp:1546-1547`), which runs only when the run
completes normally. The Python wrapper has no `close`, `__enter__`, `__exit__`,
or `__del__` either.

`python/star_reacher/sim.py:228-231` therefore relies entirely on refcount
timing:

```
# Dropping the previous Sim closes its log before the new one opens the same path.
self._sim = None
self._sim = self._core.Sim(cfg, str(srlog_path))
```

**Failure scenario.** This is the Windows `PermissionError` already seen once
this phase. Any exception escaping `step()` — including the
`std::invalid_argument` from `cpp/src/srlog_writer.cpp:781` when a component
reports `m == 0`, or any exception from a Python component — leaves the C++
`Sim` alive while pytest's `ExceptionInfo` or `sys.last_traceback` pins the
frame holding the Python `Sim`. The `run.srlog` handle stays open, and
`tmp_path` teardown or a `reset()` on the same path fails with
`PermissionError: [WinError 32]`. On Linux the unlink silently succeeds, which
is why this reproduces under MSVC and would not be caught by the Linux leg.

**Remedy.** Add an idempotent `close()` to `VehicleCycle` delegating to
`SrlogWriter::close()`, bind it, and give `star_reacher.sim.Sim` a `close()`
plus `__enter__`/`__exit__`. This removes a class of Windows-only flake and
gives finding 3 a non-destructor path on which to report a flush failure.

**A second, independent defect on the same object: `reset()` twice raises by
default.** `python/star_reacher/sim.py:213` checks
`if srlog_path.exists() and not self._force`. The native `Sim(cfg, path)`
creates and header-writes `run.srlog` at construction
(`python/star_reacher/sim.py:231`), so after the first `reset()` the file
exists. The docstring at `python/star_reacher/sim.py:198` promises that
"calling `reset` again starts a new run over the same output path" - it does
not, for the default `force=False`. This is the Gym-style API of FR-24, where
`for ep in range(N): sim.reset()` is the normal usage pattern. Every test in
`tests/python/test_sim.py` constructs `Sim(..., force=True)`, so the default
constructor followed by a repeated `reset()` is never exercised. Even with
`force=True`, each `reset()` overwrites `out/run.srlog`, so a multi-episode
driver keeps only the last episode's log.

`python/star_reacher/verify.py` already works around the handle problem at one
call site with a manual `sim = None` in a `finally`, whose own comment names
the Windows `PermissionError` - so the hazard is known and patched locally
while the class still exposes no remedy.

### 7. The plugin module cache is never invalidated while `meta.json` re-hashes the file (CONFIRMED, MEDIUM)

**Where:** `python/star_reacher/plugin.py:154-158` and `:182`;
`python/star_reacher/runner.py:424`.

`_load_module` returns a cached module on a path hit with no mtime or content
check, and `_loaded_modules[resolved] = module` is never invalidated. But
`_plugin_provenance` computes
`hashlib.sha256(path.read_bytes()).hexdigest()` fresh at
`python/star_reacher/runner.py:424`.

**Failure scenario.** Within one process: run a mission with
`--gnc-plugin p.py`, edit `p.py`, run again. The second run **flies the cached
old module** while `meta.json` records the **new** file's SHA-256. The recorded
hash is of code that did not execute — the exact inversion of the guarantee
the field exists to provide, which
`python/star_reacher/runner.py:411-416` justifies on the grounds that "two runs
of one mission with two revisions of a plugin are different experiments".

Not reachable through `star run`, which is one process per run. Directly
reachable through `Sim` (`python/star_reacher/sim.py:146` calls
`load_plugins`), through any in-process Monte Carlo or driver loop, and
through a pytest session.

**Remedy.** Key `_loaded_modules` on `(resolved_path, sha256_of_bytes)`, or
record the hash at load time in `plugin.py` and have `_plugin_provenance` read
that recorded value rather than re-reading the file.

### 8. The nav fix reports `valid = true` before it has ever been sampled (CONFIRMED, MEDIUM)

**Where:** `cpp/src/vehicle_cycle.cpp:1009-1015`.

```
in.navfix.valid = true;  // the nav fix carries no gating flag
in.navfix.fresh = navfix_fresh;
in.navfix.r_i_m = navfix->last_position_m();
in.navfix.v_i_mps = navfix->last_velocity_mps();
```

`valid` is hardcoded `true` unconditionally, including on every cycle before
the sensor's first sample instant. The star tracker and altimeter, two lines
below and above, correctly forward the sensor's own flag via `last_valid()`.

`r_meas_` and `v_meas_` are properly zero-initialised
(`cpp/include/star/sensors/radio.hpp:65-66`), so this is not an uninitialised
read — but zero is precisely the "plausible payload" shape: a consumer sees a
nav fix flagged valid that places the vehicle at the centre of the central
body, at rest.

**Failure scenario.** The built-in EKF is protected, because it gates on
`fresh && valid` (`cpp/src/gnc/ekf.cpp:184`) and `fresh` is correctly false
until the first sample. But the same struct is copied verbatim onto the FR-24
observation surface (`cpp/src/vehicle_cycle.cpp:1063`, `obs.navfix = in.navfix`)
and exposed to a stepping driver. A Python nav component or a stepping driver
that checks `valid` — the flag whose entire purpose is to answer "may I trust
this payload" — and not also `fresh` will fold a zero position/velocity fix
into its estimate on every cycle before the first sample. With a 1 Hz nav fix
on a 100 Hz control cycle, that is 100 consecutive spurious updates at the
origin.

**Remedy.** Give `NavFix` a `sampled_` flag set in `sample()` and forward it,
so `valid` means "this payload is real" for all four aiding sensors uniformly.
The inline comment is right that the nav fix has no *gating* flag; it does not
follow that it has no *validity* flag.

### 9. The altimeter update is silently discarded without a body-fixed frame (CONFIRMED, MEDIUM)

**Where:** `cpp/src/gnc/ekf.cpp:496-502`.

```
if (!env.bodyfixed_valid || !(ellipsoid_a_m_ > 0.0)) {
  return;
}
```

The altimeter **sensor** does not skip in this case: without a valid body-fixed
frame it falls back to the closed spherical form `|r| - a` and emits the
measurement flagged valid (`cpp/src/sensors/radio.cpp:189-190`, gate at
`203-205`). The filter takes the opposite branch and drops it.

The early return produces no `InnovationSample`, writes nothing to
`nav.innov`, and sets no flag. The measurement is consumed and discarded with
zero observability.

**Failure scenario.** A run configured with an altimeter on a central body
whose body-fixed frame is unavailable logs a full `sensor.altimeter` channel
with `valid = 1` throughout, and an empty `nav.innov` for that sensor. The
filter behaves as if the altimeter were not configured, and nothing in the log
distinguishes "the altimeter was never fresh" from "every altimeter update was
refused". A NEES gate then fails or passes for a reason the log cannot
explain.

Skipping is defensible — the comment argues it, and folding a measurement in
against the wrong frame would be worse. The defect is that the skip is
invisible.

**Remedy.** Emit an `InnovationSample` with a zero-width or explicitly-flagged
payload for a refused update, or add a counter to the run summary. The
project's own standing lesson on making silent rejections observable applies
directly.

### 10. `LDLT::info()` is never checked on the innovation covariance (CONFIRMED, MEDIUM)

**Where:** `cpp/src/gnc/ekf.cpp:387-388`.

```
const Eigen::LDLT<Eigen::Matrix<double, M, M>> ldlt(s);
const Eigen::Matrix<double, M, kM> kt = ldlt.solve(pht.transpose());
```

If the decomposition fails, `solve()` returns a result that is silently wrong
rather than signalling. The gain `k` then propagates NaN or garbage into `dx`,
into the reset, and into `p_` through the Joseph form — after which every
subsequent cycle is contaminated and the covariance logged to `nav.est.P` is
meaningless.

`S = H P Hᵀ + R` is positive definite whenever `R` is, and `R` is built from
configured sigmas. The exposure is therefore a run whose sigmas are zero: the
sensor parsers accept `sigma_rad`, `sigma_r_m`, and `sigma_v_mps` entries that
are `>= 0` and reject only negatives (`cpp/src/sensors/optical.cpp:87-90`,
`cpp/src/sensors/radio.cpp:63-66`), and `NavSensorModel`'s sigma members
default to zero (`cpp/include/star/gnc/component.hpp:139-140, 146-147`). A
noiseless sensor is a configuration a user can plausibly write when
constructing a controlled test case.

I could not determine without running whether `python/star_reacher/mission.py`
independently forbids a zero sigma; that check is in the Python layer's
section below if it was covered there.

**Failure scenario.** Configure `[sensors.startracker]` with
`sigma_rad = [0.0, 0.0, 0.0]`. `R` is exactly zero, `S = H P Hᵀ`, and once the
attitude block of `P` collapses toward zero after repeated zero-noise updates,
`LDLT` on a numerically singular `S` yields a garbage solve that is never
detected. The run completes and writes a log that looks structurally valid.

**Remedy.** Check `ldlt.info() != Eigen::Success` and throw a named error, and
separately require every configured sigma that reaches `R` to be strictly
positive at parse time — this filter's own `require_sigma3()`
(`cpp/src/gnc/ekf.cpp:55-66`) already argues exactly this case for `P0`, with
the reasoning that a zero variance makes NEES undefined rather than merely
large. The same argument applies to `R`.

### 11. An invalid IMU sample freezes the covariance while time advances (CONFIRMED, MEDIUM)

**Where:** `cpp/src/gnc/ekf.cpp:176-178`.

```
if (input.imu_fresh && input.imu.valid && input.imu.dt_s > 0.0) {
  propagate(input.imu);
}
```

When the guard fails, neither the nominal state nor `P` advances — but the
aiding updates below still run, and truth keeps moving. The filter's stated
uncertainty stops growing while its actual error grows.

**Failure scenario.** Any interval during which the IMU is fresh but flagged
invalid produces a filter that is systematically overconfident: `P` is frozen
at its last propagated value while the true error accumulates at the full
unaided drift rate. A subsequent aiding update then computes a gain from a
covariance that understates the error, under-corrects, and drives NEES up —
with no entry anywhere in the log identifying the skipped propagations as the
cause. This is the same overconfidence failure the freshness comment at
`cpp/src/vehicle_cycle.cpp:993-996` is careful to prevent for reprocessing,
reached from the opposite direction.

I did not find a path in the shipped `Imu` that sets `valid = false` after the
first sample — `cpp/src/sensors/imu.cpp:255` sets it true and nothing clears
it — so this is currently unreachable with the built-in IMU. It becomes
reachable with any sensor plugin or future IMU gating.

**Remedy.** On a skipped propagation, still advance `P` by the process-noise
term for the elapsed control period, or refuse the cycle outright. Silently
holding the covariance is the one option that produces a plausible-looking
wrong answer.

### 12. The ensemble chi-square dimension is taken from run 0 only (CONFIRMED, MEDIUM)

**Where:** `python/star_reacher/consistency_cli.py:460` (NEES) and `:467`
(NIS).

```
("NEES", nees_runs, per_file[0][1]["e"].shape[-1], False)
...
per_file[0][1]["innov"][sid][0].shape[-1],
```

`cmd_consistency` checks that every run exposes the same *sensor id set*
(`:409`) and the same *epoch count* (`:472`), but never that a given sensor
has the same measurement *dimension* across runs. Each run's `nis()` returns a
`(T,)` array regardless of its own `m`, so `np.stack` succeeds and the whole
ensemble is gated at run 0's dimension.

**Failure scenario.** Two runs both logging sensor 1, one at `m = 3` and one at
`m = 6`, have identical sensor id sets and stack without error. The
chi-square interval is then computed at `dof = R·3` for data that is partly
chi-square(6) — roughly a factor-of-two shift in the headline statistic,
producing a confident and wrong verdict in either direction depending on which
run happened to be first.

**Remedy.** Validate `dim` across runs alongside the existing epoch-count
check, and fail with the same wording.

### 13. `_reduce_error` guesses the quaternion collapse whenever `n == m + 1` (CONFIRMED, MEDIUM)

**Where:** `python/star_reacher/consistency_cli.py:170-175`.

```
if n == m + 1 and n >= 4:
```

Both the module docstring (`python/star_reacher/consistency_cli.py:33`) and
`docs/formats/srlog_v1.md:337` state that "any other pairing of n and m is
reported as a mismatch rather than guessed at". `n == m + 1` is never
reported — it is always collapsed, on the assumption that slots 0..3 are a
scalar-first error quaternion. Nothing in the log states the estimator's error
layout, so the CLI cannot verify that assumption.

**Failure scenario.** `docs/gnc_plugins.md:165-173` documents
`ErrorForm.ROTATION_VECTOR_LOCAL` and `_GLOBAL`, whose attitude block is
**three** slots. A plugin estimator that happens to have `n = m + 1` for any
unrelated reason gets `e[0:4]` mangled into `2·sgn(e[0])·e[1:4]`, and the
resulting NEES has the right shape, is positive, and is order-unity — it looks
entirely plausible while being wrong.

> **Since: the two named forms were removed** (every attitude form is now four
> slots), so the specific route above is closed. The attitude-block-not-first
> shape — `[VELOCITY(3), ATTITUDE(4)]` against a 6-dimensional covariance —
> outlived that removal and reached `n = m + 1` without a quaternion in slots
> 0..3. It is now refused at run construction:
> `validate_error_layout` takes the component's `cov_dim()` and rejects a
> declared layout at `n == m + 1` (`n >= 4`) whose offset-0 block is not the
> attitude block. The remedy below is still the one that would close the
> reader; it remains open. See KNOWN-ISSUE-P6-5.

**Remedy.** Carry the estimator's declared error layout, or at minimum a
"quaternion-led attitude block" flag, in the SRLOG header, and require it
before applying the collapse. The layout already exists in the core
(`error_layout()`); it is simply not written to the log.

### 14. `write_nav_innov` does not bound-check `sensor_id` (CONFIRMED, MEDIUM)

**Where:** `cpp/src/srlog_writer.cpp:771-794`.

The writer validates `y_len`, `s_len`, and `m` against the declared `m_max`
(`cpp/src/srlog_writer.cpp:781`) but writes `sensor_id` verbatim at `:790`
with no check. The contract says `sensor_id` indexes the header's
`gnc.sensors` array (`cpp/include/star/srlog_writer.hpp:228-231`,
`docs/formats/srlog_v1.md:353`), and the writer knows that array's size at
construction — but never stores it.

`InnovationSample::sensor_id` is a `std::uint32_t`
(`cpp/include/star/gnc/component.hpp:228`) exposed via `def_readwrite`
(`bindings/module.cpp:1255`), so a Python estimator can set it to any value.

**Failure scenario.** A Python estimator sets `sensor_id = 3` on a mission
declaring two sensors. The file is written, passes every writer check, and a
downstream reader indexing `header["gnc"]["sensors"][3]` raises `IndexError`
at analysis time — or, if a tool clamps instead, silently mislabels which
sensor produced the innovation. That attribution is exactly what the NEES/NIS
work depends on. Every other v1.2 dimension is checked at write time; this is
the gap.

**Remedy.** Store `sensor_count_` in the constructor from
`fields.sensors.size()` and throw `std::invalid_argument` at
`cpp/src/srlog_writer.cpp:781` when `sensor_id >= sensor_count_`.

### 15. The ellipsoid is defined twice with incompatible sphere sentinels (CONFIRMED, MEDIUM)

**Where:** `cpp/src/models/environment.cpp:291-311` versus
`cpp/src/vehicle_cycle.cpp:717-721`.

The same reference ellipsoid reaches the sensor and the filter by two
independent paths that disagree on how a sphere is encoded:

- The **sensor** path reads `SensorCycleTruth::geom.ellipsoid_inv_f`, filled by
  `EnvironmentModel::central_ellipsoid()`, which encodes the Moon as
  `inv_f = 0.0` (`cpp/src/models/environment.cpp:301-304`). The altimeter tests
  `spherical = !(geom.ellipsoid_inv_f > 1.0)`
  (`cpp/src/sensors/radio.cpp:187`) and takes the closed spherical branch.
- The **filter** path reads `GncInitContext::ellipsoid_inv_f`, filled from
  `planet_inv_f` (`cpp/src/vehicle_cycle.cpp:963-964`), which encodes the Moon
  as `inv_f = 1.0e12` (`cpp/src/vehicle_cycle.cpp:721`). The EKF has no
  spherical branch and runs the Bowring conversion
  (`cpp/src/gnc/ekf.cpp:509`).

So for a lunar mission the altimeter measures `|r| - R_moon` while the filter
predicts a Bowring geodetic height at `f = 1e-12`.

**This does not currently misbehave**, and I want to be precise about that: I
initially expected a throw, and checked. `geodetic_lat_lon_alt()` rejects
`inv_f <= 1.0` with `std::domain_error` (`cpp/src/models/atmosphere_hp.cpp:157`),
but the `1.0e12` sentinel clears that guard, and at `f = 1e-12` the Bowring
result differs from `|r| - a` by order `a·e²` ≈ 3 µm — negligible against any
configured altimeter noise. Mars (`MARS_ELLIPSOID_INV_F`) and Earth
(`WGS84_INV_F = 298.257223563`) are both well above 1 on both paths. The Sun
has no body-fixed frame, so the EKF returns early and no vehicle can be
configured there anyway (`python/star_reacher/mission.py:2585`).

The finding is therefore **latent, not active**: two sources of truth for one
physical constant, using mutually incompatible conventions for the same
degenerate case (`0.0` means sphere on one path and would mean "invalid,
throw" on the other), with correctness resting on a magic `1.0e12` whose only
documentation is a nine-word trailing comment. Any future edit that
harmonises one path to the other — the obvious cleanup — converts this into an
immediate `std::domain_error` thrown from inside the deterministic time loop
on every lunar altimeter update.

**Remedy.** Have `GncInitContext` take its ellipsoid from
`EnvironmentModel::central_ellipsoid()`, the same call the sensor geometry
uses, and give the EKF the same explicit spherical branch the altimeter has.
One ellipsoid, one sphere convention, one branch test.

### 16. `_apply_override` silently truncates a float to an integer leaf (CONFIRMED, LOW)

**Where:** `python/star_reacher/sim.py:381`, and the same shape at `:202`.

```
container[key] = int(value) if isinstance(current, int) else float(value)
```

The docstring at `python/star_reacher/sim.py:186-189` states that "an integer
leaf takes an integer, so a control rate cannot silently become fractional".
It does not reject a fractional value — it truncates it.

**Failure scenario.** `sim.reset(overrides={"gnc.latency_cycles": 2.7})` yields
`2` with no diagnostic, and the resulting `config_sha256` records `2`. The run
is perfectly reproducible, and is not the run the driver asked for.
`reset(seed=3.9)` truncates to `3` the same way at `:202`.

**Remedy.** Raise `SimError` when `current` is an `int` and `value` is not
integral, matching the documented contract.

### 17. `_load_module` swallows `KeyboardInterrupt` and `SystemExit` (CONFIRMED, LOW)

**Where:** `python/star_reacher/plugin.py:175`.

`except BaseException as exc:` converts anything raised during plugin import
into a `PluginError`. Ctrl-C during a slow plugin import surfaces as
`"...GNC plugin raised while being imported: KeyboardInterrupt: "` and the
interrupt is consumed rather than propagating.

**Remedy.** Use `except Exception` for the wrap, with a separate bare
`except BaseException:` that pops `sys.modules` and re-raises.

### 18. `sensor_index_` brace initialiser breaks silently on a seventh sensor kind (CONFIRMED, LOW — latent)

**Where:** `cpp/include/star/srlog_writer.hpp:268`.

```
int sensor_index_[kSensorKindCount] = {-1, -1, -1, -1, -1, -1};
```

Correct today, since `kSensorKindCount == 6`
(`cpp/include/star/srlog_writer.hpp:124-127`). But the header itself states
that extending `kSensorKinds` is a sanctioned minor bump (`:121-123`).

**Failure scenario.** A seventh kind is added. The seventh element
value-initialises to `0` rather than `-1`, and `0` is the group index of
**`truth`** (`cpp/src/srlog_writer.cpp:551`). The `idx < 0` guard in every
`write_sensor_*` then passes, and `put_u16(0)` writes a sensor payload tagged
as a truth record. The result is a **silently corrupt log**, not a crash: a
reader parses it as truth with the wrong stride and desynchronises the rest of
the file.

**Remedy.** Use `std::array<int, kSensorKindCount>` filled with `-1` in the
constructor, or put a `static_assert(kSensorKindCount == 6, ...)` next to the
initialiser so the compiler catches the extension.

### 19. Three `SrlogHeaderFields` members have no default initialiser (CONFIRMED, LOW — latent, pre-existing)

**Where:** `cpp/include/star/srlog_writer.hpp:47`, `:48`, `:51`.

```
std::uint64_t master_seed;    // no initialiser
bool oracle;                  // no initialiser
std::uint32_t truth_rate_hz;  // no initialiser
```

Every other member of the struct carries a default. All current call sites set
these three, so nothing is triggered today.

**Failure scenario.** Any new call site writing `SrlogHeaderFields f;` and
forgetting `oracle` reads an indeterminate `bool` — undefined behaviour that
serialises as an arbitrary `"true"` or `"false"` at
`cpp/src/srlog_writer.cpp:270`. That breaks the FR-21 double-run SHA-256
determinism gate, and breaks it *intermittently*, which is the worst possible
failure mode for a determinism gate.

**Remedy.** `= 0`, `= false`, `= 0`.

### 20. Unguarded division by `r³` in the filter's gravity model (CONFIRMED, LOW)

**Where:** `cpp/src/gnc/ekf.cpp:355-369`.

`gravity()` computes `-mu_ * p / (r*r*r)` and `gravity_gradient()` computes
`u = p / r`, both without a guard on `r == 0`. A position *estimate* at the
origin yields inf/NaN that propagates into `v_hat_`, `p_hat_`, and the `F`
matrix, corrupting the run from that cycle on.

`p0_m` is a required parameter with no positivity constraint
(`cpp/src/gnc/ekf.cpp:124` calls `require_vector`, which checks only presence,
length, and finiteness), so `p0_m = [0, 0, 0]` is accepted and produces NaN on
the first propagation. This is a configuration error rather than a realistic
run, and it fails loudly enough in the output to be diagnosed — hence LOW —
but the project's own `require_sigma3()` sets the precedent for rejecting a
degenerate parameter at construction instead.

**Remedy.** Reject a zero `p0_m` in the constructor, and guard `r` in both
gravity routines.

### 21. Sun sensor emits a normalised pure-noise direction when the ephemeris is absent (CONFIRMED, LOW — previously identified, still present)

**Where:** `cpp/src/sensors/optical.cpp:266-283`.

With `geom.ephemeris_valid == false`, `u_b` stays zero, so `sum = eta` is pure
noise and line 283 normalises it to a unit vector. The sample is correctly
flagged `valid = 0`, but the payload is a perfectly plausible unit direction
rather than something a consumer can recognise as meaningless.

This is recorded as already found during the phase. It is confirmed still
present at `ea0e5e9`. Emitting the zero vector, or NaN, when
`geometry == false` would make the invalidity self-evident in the payload as
well as the flag.

### 22. Minor and dead logic (CONFIRMED, informational)

Two items that are not defects in effect but read as checks they are not:

- `python/star_reacher/mission.py:2155`, `2198-2201` — `kinds_ok` is a single
  accumulator across all sensor kinds and is never reset per kind, so
  `if kinds_ok and srate is not None:` stops populating `resolved_kinds` for
  every kind after the first failure. Harmless, because the gate at `:2202`
  discards the dict wholesale and sets `ok = False`, but the inner guard is
  dead and reads as a per-kind check.
- `python/star_reacher/consistency.py:436` — `inside_count_threshold` returns
  `0` for `epochs <= 2`, which makes `coverage_passed = inside_count >= 0`
  unconditionally true. Documented at `:415-416`, but it is a gate that cannot
  fire on very short runs.
- `python/star_reacher/consistency_cli.py:173` — since the producer already
  canonicalises the error quaternion to the `+w` hemisphere
  (`docs/formats/srlog_v1.md:330`), the `sign = np.where(w >= 0.0, 1.0, -1.0)`
  factor is always `+1` for core-produced logs. A defensive branch that cannot
  fire; harmless.

## Areas reviewed and found clean

- **`auto` + Eigen, aliasing, and reference lifetime** across all ten new and
  changed C++ source files — see the table above. Genuine negative result.
- **Quaternion conventions.** Scalar-first construction is correct throughout
  (`cpp/src/gnc/ekf.cpp:122`, `219-220`; `cpp/src/sensors/optical.cpp:107`).
  `quat_normalize()` refuses a zero or non-finite quaternion rather than
  fabricating an attitude (`cpp/src/rotation.cpp:64-74`), which closes the
  degenerate-`q0` hole that would otherwise mirror finding 7.
- **The star tracker forward model and the filter's prediction agree
  exactly.** The sensor composes
  `q_meas = q_ab ⊗ q_true ⊗ dq_n` with `q_ab = quat_exp(-b_I × β)`
  (`cpp/src/sensors/optical.cpp:185-201`); the filter predicts
  `q_pred = q_ab ⊗ q_hat` with the identical construction
  (`cpp/src/gnc/ekf.cpp:464-472`). The innovation
  `dq_y = q_pred⁻¹ ⊗ q_meas` therefore telescopes to `dq_true ⊗ dq_n`
  exactly, leaving only the second-order difference between evaluating the
  boresight at `q_true` versus `q_hat`. This is the single easiest place in
  the phase to invert a sign or a transpose, and it is right.
- **The EKF error-dynamics matrix `F`.** Every block was derived
  independently against the local (body-frame) multiplicative error
  convention `δq = q̂⁻¹ ⊗ q_true` and the truth-minus-estimate sign of the
  additive blocks, and all six populated blocks
  (`cpp/src/gnc/ekf.cpp:307-319`) are correct and mutually consistent with
  `reset()`'s right-multiplication (`cpp/src/gnc/ekf.cpp:405-413`) and with
  the star tracker's `H = [I 0 0 0 0]` (`cpp/src/gnc/ekf.cpp:483`).
- **`nav.innov` structural zero-padding.** The row-by-row embedding at
  `cpp/src/vehicle_cycle.cpp:1126-1134` correctly places an `m×m` packed upper
  triangle into the leading corner of an `m_max×m_max` one; the naive flat
  copy the comment warns against would indeed have scattered it. Verified by
  hand for the offsets. (The bounds defect of finding 1 is orthogonal to
  this — the arithmetic itself is right.)
- **Determinism inside the time loop.** No `unordered_map` or `unordered_set`
  anywhere in the phase; every string-keyed container is `std::map`
  (`cpp/include/star/gnc/config.hpp:31-33, 43-46`), so iteration order is
  deterministic. No clock read, network access, or text parsing inside the
  loop. Seeds are threaded explicitly through `rng::make_stream(master_seed,
  <name>)` with a pure 64-bit derivation and no allocation
  (`cpp/src/rng.cpp:104-114`).
- **RNG draw schedules do not depend on gating.** Both optical sensors draw
  their three normals unconditionally, before and independently of the
  validity gate (`cpp/src/sensors/optical.cpp:190-192, 279-280`), and the IMU's
  initialisation and per-sample schedules are unconditional by construction,
  multiplying by a zero sigma rather than skipping a draw
  (`cpp/src/sensors/imu.cpp:180-198`, `216-241`). The nav fix's optional
  Gauss-Markov draws are conditional on configuration only
  (`cpp/src/sensors/radio.cpp:128-129`), which is constant across a run and
  enters the FR-15 resolved-config hash. This is exactly right and is the
  hazard most often gotten wrong.
- **Per-kind RNG stream names cannot collide.** Each sensor derives its stream
  from its kind string, which would correlate two instances of the same kind —
  but `[sensors.<kind>]` is a TOML table keyed by kind
  (`python/star_reacher/mission.py:2129-2135, 2156-2157`), so a second instance
  of a kind is structurally impossible. Closed by construction.
- **Sensor member initialisation.** Every measurement-holding member in the
  four sensor headers carries an explicit initialiser
  (`cpp/include/star/sensors/radio.hpp:59-66, 99-101`;
  `cpp/include/star/sensors/optical.hpp:96-97, 128-129`), as does every field
  of `SensorCycleTruth` (`cpp/include/star/sensors/sensor.hpp:55-82`) and
  `NavSensorModel` (`cpp/include/star/gnc/component.hpp:129-152`). No
  read-before-assign found.
- **The camera hook's bit-exactness claim holds by construction.** The pose
  channels are copies of the truth doubles, not recomputations
  (`cpp/src/sensors/camera.cpp:121-123`), and `px_` is sized from the landmark
  count at construction (`cpp/src/sensors/camera.cpp:110-111`).
- **The FR-24 truth boundary is structural.** No virtual on `IGncComponent`
  takes a `TruthState`; the loop computes `nav.err` itself from the
  component's declared layout (`cpp/src/vehicle_cycle.cpp:1082-1107`), and
  `GncInput.oracle` is populated only under `cfg.oracle`
  (`cpp/src/vehicle_cycle.cpp:1037-1047`). The guarantee does not rest on a
  rule an implementation is asked to honour.
- **The Python-component registry avoids the static-destruction crash.** The
  name→factory table is a deliberately leaked heap allocation
  (`bindings/module.cpp:577-585`) rather than a static `std::map<std::string,
  py::object>`, which would release Python references after interpreter
  finalisation.
- **`truth_vector()` does not return a reference to a temporary.** Every arm
  of the switch returns a reference to a member of the `truth` parameter, and
  the unmatched case throws rather than returning a default-constructed
  temporary (`cpp/src/gnc/component.cpp:68-85`). The caller binds the result
  to a `const Eigen::Vector3d&` (`cpp/src/gnc/component.cpp:230`), so a
  `return Eigen::Vector3d::Zero()` in the default arm would have been a
  dangling read. It is not there.
- **The two error conventions agree with their declarations.**
  `attitude_error()` computes `q̂⁻¹ ⊗ q_true` for the local forms
  (`cpp/src/gnc/component.cpp:98-100`), exactly the convention
  `cpp/include/star/gnc/ekf.hpp` documents for the EKF, and canonicalises to
  the `+w` hemisphere so the logged error cannot flip between epochs. Both
  declared layouts tile their state vectors exactly — the EKF's 4+3+3+3+3 = 16
  against `state_dim() == 16`, dead reckoning's 4+3 = 7 against
  `state_dim() == 7` — which is what `validate_error_layout()` enforces.
- **Sensor rate divisibility is defensively re-checked in C++.**
  `cpp/src/vehicle_cycle.cpp:434-438` rejects a `sample_rate_hz` below 1 or
  one that does not divide `control_rate_hz`, and the IMU is separately
  required to equal the control rate (`428-432`). Without that check the
  integer division at `cpp/src/vehicle_cycle.cpp:823-824` could yield a zero
  decimation factor and the modulo at `cpp/src/vehicle_cycle.cpp:988` would be
  a division by zero. The guard is present and correct.
- **The latency FIFO is a pure state machine.** Pre-filled with `k` hold
  entries so an output produced on cycle `i` surfaces on cycle `i + k`, with a
  popped hold resolving to the previous applied command with its valid flag
  cleared (`cpp/src/gnc/component.cpp:311-335`). No clock, no allocation after
  construction.
- **`builtin.cpp` is clean.** The PD control law's composition is correct
  under D-7: `dq = q_cmd* ⊗ q_est` is `q_cmd2b`, so `C(dq)` resolves the
  commanded rate into the estimated body frame, and the sign convention drives
  the body toward the command (`cpp/src/gnc/builtin.cpp:296-330`). Every
  built-in refuses unknown parameter keys rather than ignoring them, and the
  hold paths return `valid = false` so the FIFO — not an accidental identity
  quaternion — supplies the applied command.
- **`chi2.py` is clean, and was validated rather than only read.** The closed
  forms are exact to 1e-14 at k = 2 and 1e-13 at k = 1; the
  `chi2_ppf` → `chi2_cdf` round trip has a worst relative error of 9.9e-10
  over k ∈ {1 … 1e6} × p ∈ {1e-12 … 1−1e-12}, matching the docstring's stated
  ~1e-9 CDF rounding at k = 1e6 and sitting about six orders below the
  interval half-width the gate uses. The Lentz continued fraction matches
  Numerical Recipes 6.2 including the `_FPMIN` guards, the `sqrt(a)`-scaled
  iteration cap is sound, and the safeguarded Newton keeps every iterate
  strictly inside a monotonically shrinking bracket so the log-density step
  can neither overflow nor divide by an underflowed pdf. No unguarded
  division.
- **`consistency.py` is clean, and was validated numerically.** Packing was
  checked byte for byte against a manual row-major upper triangle;
  `unpack_symmetric ∘ pack_symmetric` is exactly the identity; `nees` agrees
  with an explicit `inv(P)` quadratic form to 3.6e-16 relative. The
  normalisation is right and is the thing most often gotten wrong here: the
  statistic is *not* divided by dof — `time_average_gate` uses `dof = T·dim`
  scaled by `1/T` and `ensemble_gate` uses `dof = R·dim` scaled by `1/R`, both
  centring at `dim`, confirmed on 100 × 601 synthetic chi-square(15) data
  (headline 14.99 in [13.95, 16.09]). Cholesky is used with a per-epoch rescan
  that names the first non-positive-definite matrix rather than returning
  garbage, and `np.linalg.solve` on `L` avoids forming an explicit inverse.
- **`mission.py` validation is clean on every gap hunted.** `sample_rate_hz`
  divisibility is checked (`python/star_reacher/mission.py:2185-2192`) with the
  IMU pinned to exactly `control_rate_hz`; the EKF's `p0_sigma_*` are required
  strictly positive with the correct stated reason; unknown-key rejection
  reaches every nesting level — root, `[gnc]`, `[gnc.*]`, `[sensors]`, and
  `[sensors.<kind>]`. Two hunted items turned out to be guarded downstream
  rather than defects: quaternions are checked only for non-zero norm, but the
  core normalises through `rotation::quat_normalize`
  (`cpp/src/gnc/builtin.cpp:91`, `cpp/src/gnc/ekf.cpp:121`), and Gauss-Markov
  `*_tau_s` may be zero, but every use in the core is guarded
  (`cpp/src/gnc/ekf.cpp:97`, `314`, `317`, `338`, `343`;
  `cpp/src/sensors/imu.cpp:170`; `cpp/src/sensors/radio.cpp:101`, `108`).
- **The plugin path does not break determinism.** Load order follows the CLI
  argument order, `load_plugins` returns `sorted(set(...))`, duplicate names
  across files are refused rather than shadowed, and plugin exceptions are
  **not** swallowed at the C++ boundary — there is no `catch` in
  `bindings/module.cpp`, `cpp/src/gnc/*.cpp`, or `cpp/src/vehicle_cycle.cpp`.
  `examples/gnc_plugins/pd_attitude.py` reads no clock, opens no file, draws no
  RNG, iterates no set, and holds no cross-cycle state. (Finding 7 is about
  provenance recording, not about determinism within a run.)
- **No determinism hazard in the reviewed Python.** No `os.urandom`, no
  unseeded `default_rng()`, no `hash()`, no time-derived seed, and no set or
  dict iteration feeding a numeric result. The only set/dict iterations are
  `sorted(...)` at `python/star_reacher/consistency_cli.py:407`, `409` and
  `python/star_reacher/plugin.py:250`. Finding 1 is an ordering defect, but it
  is deterministic ordering — reproducible per entry point, just not equal
  across the two.
- **No per-step Python inside the propagation loop** other than the sanctioned
  FR-25 plugin path.
- **pybind11 lifetime is handled correctly throughout.**
  `PythonComponentHandle` declares `obj_` before `impl_`
  (`bindings/module.cpp:620-621`) so the member-init order is valid, and its
  destructor acquires the GIL and nulls `obj_` by move-assign so the later
  member destructor is a no-op on a null pointer. Every hand-rolled trampoline
  override acquires the GIL, reentrant acquisition is safe, and Python
  exceptions propagate as `py::error_already_set` rather than being swallowed.
  `VehicleCycle::Impl` stores `RunConfig` by value
  (`cpp/src/vehicle_cycle.cpp:580`), so `Sim(config, path)` cannot dangle on a
  dropped Python `RunConfig`, and the Python component is kept alive by a
  strong `py::object`, so no `py::keep_alive<>` is needed on either.
- **SRLOG binary framing is clean.** There is no backpatched offset, no
  trailer, and no length field written ahead of its payload — the header length
  is computed from the already-serialised JSON and written immediately before
  it (`cpp/src/srlog_writer.cpp:527`). The endianness probe is correct, the
  `put_*` helpers check `!out_` on every call so no write status is ignored,
  and validation runs before `out_.open()` so a rejected config leaves no
  truncated file. The constructor's group-index walk was checked against
  `header_json`'s emission order sequence by sequence; they agree. Every
  declared sensor group has a writer.
- **The packed-upper-triangle order agrees between writer and reader.** The
  writer is a pass-through of a caller-supplied buffer, the documented order is
  row-major upper triangle (`docs/formats/srlog_v1.md:289`, `356`), and the
  reader implements exactly that via `np.triu_indices`
  (`python/star_reacher/consistency.py:222-256`). No disagreement.
- **No determinism violation anywhere in loop-reachable C++.** A sweep for
  `std::chrono`, `time()`, `clock()`, `rand()`, `random_device`, `getenv`, and
  unordered iteration across `cpp/src/gnc/`, `cpp/src/sensors/`,
  `cpp/src/vehicle_cycle.cpp`, and `cpp/src/models/environment.cpp` returned a
  single hit: the `std::sort` at `cpp/src/gnc/component.cpp:152`. That sort is
  construction-time, not in the loop, and its only unstable case — two blocks
  sharing an offset — is rejected two lines later by the exact-tiling check,
  so the outcome is a throw regardless of how the tie ordered. Clean.

## Not reviewed

Stated explicitly so this review is not read as broader coverage than it has:

- `cpp/tests/*` (the new doctest files) were not reviewed as code.
- `docs/mathlib/chapters/*.tex` derivations were not checked against the
  implementations beyond the specific equation cross-checks noted above.
- `scripts/nees_diag/*` and `tests/refs/*` were not reviewed.
- The mission TOML fixtures under `missions/` were read only for the specific
  questions above.
- In the Python package: `srlog.py`, `plotting.py`, `export.py`, `vehicle.py`,
  `data_fetch.py`, `derived.py`, `viewer.py`, and `_fixtures.py` were not
  reviewed. The eight files listed in the scope section were.
- `tests/python/*` were not reviewed as code, beyond establishing which
  fixtures findings 1 and 4 are invisible to.

## Method note

Nothing was compiled at any point, by design: another agent held the sole
compiler slot for the duration. Every finding above is marked CONFIRMED
because it rests on an airtight static read or on an executed pure-Python
demonstration — findings 1 and 4 were both driven to concrete output, and the
`chi2.py` and `consistency.py` numerics were validated numerically rather than
only read. No finding is marked SUSPECTED; where a compile would have
strengthened a result, the result did not depend on it.

Two experiments are worth running once a build is available, both expected to
reproduce immediately:

- **Findings 2 and 3.** Register a Python nav component with
  `innov_max_dim()` returning 3 and `innovations()` returning an
  `InnovationSample` with `y = [1.0] * 6` and `s_upper = [1.0] * 21`, then run
  one cycle under a Linux ASan build. Expect
  `heap-buffer-overflow WRITE of size 8` at `cpp/src/vehicle_cycle.cpp:1117`.
  Separately, a component whose `state_dim()` grows mid-run reproduces
  finding 3 at `bindings/module.cpp:532`.
- **Finding 1.** Run `missions/leo_ekf_consistency.toml` through `star run`
  and through `Sim` with identical seeds and compare the two `run.srlog` files
  byte for byte. They are expected to differ in the header's declared sensor
  array and in every `nav.innov` record's `sensor_id`.

One correction worth recording, since a wrong finding is worse than no
finding: I initially expected finding 15 to be an active crash — a lunar
altimeter update throwing `std::domain_error` from inside the time loop —
and traced the chain to check. `cpp/src/vehicle_cycle.cpp:721` sets the Moon's
`inv_f` to `1.0e12` rather than `0.0`, which clears the guard at
`cpp/src/models/atmosphere_hp.cpp:157`. The finding is real but latent, and is
reported at that severity rather than the one I first assumed.

## Corrections from the remediation pass (2026-07-20)

The review above was written without a compiler, as its method note states.
Building and running the two experiments it proposed confirmed finding 1's
defect but contradicted two statements about its *consequence*. Both are
recorded here rather than edited into the text above, so the review stays the
artifact it was and the correction is attributable and dated. A wrong finding
is worse than no finding, and the same standard applies to a wrong
consequence.

**Correction 1: the divergence was a hard failure, not silently different
bytes.** The review predicted that `missions/leo_ekf_consistency.toml` would
produce two different `run.srlog` files depending on the entry point. It does
not. `check_sensor_decls()` (`cpp/src/srlog_writer.cpp:152-170`) requires the
declared sensor array to be a subsequence of the canonical vocabulary *in
canonical order*, and rejects anything else. For any sensor set whose
alphabetical order differs from its canonical order, the alphabetical order is
by construction not a canonical subsequence, so the writer refuses it.
Reproduced with the pre-fix expression restored verbatim:

```
batch   run.srlog sha256: 317e43abd6e4cbd3891056ff2ad426ca960906d56a914f7599b2c48c0f0d0962
batch header gnc.sensors : ['imu', 'startracker', 'navfix', 'altimeter']
stepped run RAISED (ValueError): SRLOG writer: sensor kind 'imu' is
  duplicated or out of canonical order; declare a subset of {imu,
  startracker, sunsensor, navfix, altimeter, camera} in that order
```

The real defect is therefore that **the FR-24 stepping API did not work at all
on any multi-sensor mission whose sensor set is not already in canonical
order**, including the phase's flagship EKF mission. That is a worse
availability defect than the review described and a better safety one: the
failure was loud, and no corrupt or mislabelled log was ever produced.

**Correction 2: the EKF's update order does not depend on the sensor list
order.** The review states that the ordering divergence would give "a
genuinely different state trajectory" because the filter "folds its three
aiding updates in list order". It does not. `ErrorStateEkf::update()`
(`cpp/src/gnc/ekf.cpp:174-194`) applies its updates from named members of
`GncInput` in a hardcoded sequence — nav fix, star tracker, altimeter — which
is the order Chapter `ch:ekf` documents as normative and is independent of
`cfg.gnc.sensors`. The normativity comment the review cites is correct about
*why* the order is pinned; it is pinned in the filter, not inherited from the
configuration. The reachable consequences of the ordering divergence were the
header's declared sensor array and the `sensor_id` label, both of which the
writer guard above converts into a refusal.

Confirming this in the other direction: the batch hash is unchanged by the
fix (`317e43ab...` both before and after), because `star run` was already
passing the canonical order. The remedy moved only the stepping path.

**Confirmed as described.** Findings 2 and 3 reproduced exactly as predicted
once built. With the bound checks reverted, a Python component declaring
`innov_max_dim() -> 1` and returning a six-wide innovation, and one whose
`state_dim()` grows mid-run, both abort the process with
`0xC0000374` (`STATUS_HEAP_CORRUPTION`) under MSVC release; a short `s_upper`
is a silent out-of-bounds read that completes the run. Finding 4 reproduced as
described.
