# Phase 6 code-review triage

Disposition of the eleven findings left open by
[`phase6_code_review.md`](phase6_code_review.md) after findings 1-6 were fixed
as priorities and 12, 14, 16, 18, and 19 were fixed incidentally.

Triage base: `ws-p6-triage` off `phase-6-gnc-sensors` at `6e6d427`. The review
was written against `ea0e5e9`, 34 files earlier; line numbers in this document
are this base's and differ from the review's where the intervening fixes moved
code. Every citation here was resolved against the working tree before it was
written down.

## Method, and what "measured" means below

Nothing was compiled during this triage either — the sole compiler slot was
held by another agent throughout. That is the same constraint under which the
original review was written, and the correction section that review carries is
the reason this document separates its evidence into three grades:

- **Measured.** An experiment was executed and its output is quoted. The core
  is deliberately not imported by `star_reacher/__init__.py`, so the mission
  validator, the plugin loader, and the consistency evaluator all run as pure
  Python against the worktree source on `PYTHONPATH` — no wheel, no build, no
  race with the other agent's reinstall.
- **Read.** A claim established by reading source, including third-party
  source. Where a library's behaviour is load-bearing, the vendored copy that
  the build actually consumes was read rather than recalled.
- **Reasoned.** A consequence inferred from code without an executed
  demonstration. Every such statement is labelled inline, because the failure
  the review's correction section records is exactly the failure of not
  labelling them.

Two of the eleven turned out to rest on consequence claims that measurement
contradicts, and one more on a reachability claim that the mission validator
refuses. Those are reported as refutations rather than as fixes.

## Verdicts

| # | Finding | Verdict | Basis |
| --- | --- | --- | --- |
| 7 | Plugin module cache never invalidated | ACCEPT AND DOCUMENT | Measured; provenance inversion unreachable from any shipped entry point |
| 8 | Nav fix reports `valid = true` before first sample | **FIX BEFORE CLOSE** | Read; contract defect on the FR-24 surface this phase ships |
| 9 | Altimeter update silently discarded without a body-fixed frame | NOT A DEFECT | Measured; the branch is unreachable from any accepted mission |
| 10 | `LDLT::info()` never checked on the innovation covariance | **FIX BEFORE CLOSE** (different remedy) | Measured; stated consequence refuted, stated remedy ineffective, underlying gap real |
| 11 | Invalid IMU sample freezes the covariance | ACCEPT AND DOCUMENT | Read; unreachable in Phase 6, no sensor-plugin surface exists |
| 13 | `_reduce_error` guesses the quaternion collapse | ACCEPT AND DOCUMENT | Measured; reproduces exactly, but the full remedy is a format change |
| 15 | Ellipsoid defined twice with incompatible sphere sentinels | ACCEPT AND DOCUMENT | Read; confirmed latent, ~3 µm at worst today |
| 17 | `_load_module` swallows `KeyboardInterrupt`/`SystemExit` | ACCEPT AND DOCUMENT | Measured; negligible consequence, and half of it is desirable |
| 20 | Unguarded division by `r³` in the filter's gravity model | ACCEPT AND DOCUMENT | Measured reachability; consequence is an all-NaN log, loud |
| 21 | Sun sensor emits normalised pure noise with no ephemeris | ALREADY DOCUMENTED | Read; covered verbatim by KNOWN-ISSUE-P6-1 |
| 22 | Minor and dead logic | NOT A DEFECT (informational) | Measured and read; all three sub-items behave as the review describes and none is a defect |

Two findings are in the fix set. Nine are accepted, refuted, or already
documented.

---

## 7. The plugin module cache is never invalidated — ACCEPT AND DOCUMENT

**Measured.** The mechanism reproduces exactly as described. Loading a plugin,
editing it, and loading it again within one process:

```
F7 second load returned the cached module: True
F7 module in force: MARKER = v1 | sha256 recorded would be: dab8a0fa92c8267d (v1 sha: 9637c79a5f4cec26)
F7 recorded hash describes code that did NOT execute: True
```

`_load_module` (`python/star_reacher/plugin.py:153-183`) returns the cached
module on a path hit; `_plugin_provenance`
(`python/star_reacher/runner.py:414-428`) re-reads the file.

**Measured reachability, and where the review overreaches.** The review states
the defect is "directly reachable through `Sim`". It is not — not the part that
matters. `_plugin_provenance` has exactly one call site,
`python/star_reacher/runner.py:517`, inside `run_mission`. `Sim` calls
`load_plugins` (`python/star_reacher/sim.py:168`) but never writes `meta.json`
and never computes a plugin hash. Through `Sim` the consequence is a stale
module flying with no false record written anywhere; the provenance inversion —
a recorded SHA-256 for code that did not execute — requires two `run_mission`
calls in one process with a plugin and an edit between them. No shipped code
path does that: `verify.py`'s repeated `run_mission` calls
(`python/star_reacher/verify.py:122-123`, `:1036-1037`, `:2555`) pass no
plugins, and `star run` is one process per run.

**Why accept.** Reaching it requires a hand-written in-process driver plus a
mid-session source edit. The obvious remedy — keying `_loaded_modules` on
`(path, sha256)` — is also the wrong one: it would let two revisions of one
file register the same component name twice, which the core refuses by design
and which the cache exists to prevent
(`python/star_reacher/plugin.py:127-132`). The correct remedy, if a driver path
ever appears, is the review's second option: record the digest at load time in
`plugin.py` and have `_plugin_provenance` read the recorded value instead of
re-reading the file. That preserves the cache's stated invariant and makes
`meta.json` describe what ran.

## 8. The nav fix reports `valid = true` before it has ever been sampled — FIX BEFORE CLOSE

**Read.** `cpp/src/vehicle_cycle.cpp:1016` sets `in.navfix.valid = true`
unconditionally, on every cycle including those before the sensor's first
sample. The star tracker and altimeter immediately below
(`:1023`, `:1029`) forward the sensor's own `last_valid()`. `r_meas_` and
`v_meas_` are zero-initialised (`cpp/include/star/sensors/radio.hpp:65-66`), so
the payload before the first sample is a fix placing the vehicle at the centre
of the central body, at rest, flagged valid.

**Read.** The built-in EKF is protected: `cpp/src/gnc/ekf.cpp:184` gates on
`fresh && valid`. The exposure is the FR-24 observation surface, where the same
struct is copied out verbatim (`cpp/src/vehicle_cycle.cpp:1069`,
`obs.navfix = in.navfix`) for a stepping driver or a Python nav component.

**Why this one cannot wait.** Not because a shipped run misbehaves — none does.
Because Phase 6 is the phase that ships FR-24, and `valid` is a flag whose
entire purpose is to answer "may I trust this payload". Three of the four
aiding sensors on that surface answer it honestly and the fourth does not. A
driver author who generalises from the other three writes correct-looking code
that folds a zero fix at the origin into its estimate on every cycle before the
first sample — at a 1 Hz fix on a 100 Hz cycle, 100 consecutive spurious
updates. Published API semantics are the class of defect that gets more
expensive after release, not less: once drivers exist that compensate by also
checking `fresh`, fixing `valid` becomes a behaviour change rather than a bug
fix.

**Remedy.** Give `NavFix` a private `sampled_` flag in
`cpp/include/star/sensors/radio.hpp`, initialised `false`, set `true` in
`NavFix::sample()` alongside the assignments to `r_meas_`/`v_meas_`. Expose it
as `last_valid()` for symmetry with `StarTracker` and `Altimeter`, and replace
`cpp/src/vehicle_cycle.cpp:1016` with `in.navfix.valid = navfix->last_valid();`.
The existing inline comment is correct that the nav fix carries no *gating*
flag; it does not follow that it carries no *validity* flag, and the fix should
replace that comment rather than leave it contradicting the new code.

## 9. The altimeter update is silently discarded without a body-fixed frame — NOT A DEFECT

The review describes "a run configured with an altimeter on a central body
whose body-fixed frame is unavailable". No such run can be configured.

**Read.** `cpp/src/models/environment.cpp:319` sets
`g.bodyfixed_valid = central_ != CentralBody::kSun`. The Sun is the only
central body of the four (`cpp/include/star/models/environment.hpp:40`) for
which the flag is false.

**Measured.** A GNC chain requires a vehicle, and a vehicle is refused at the
Sun. Both halves, from the validator itself:

```
central_body = 'sun' with an altimeter -> validator ok: False
    [root] vehicle: a vehicle reference is not accepted with central_body = "sun"
      (the heliocentric regime is point-mass only; vehicle missions require a
      planetary central body).

sun + third bodies + NO vehicle, with [gnc] and [sensors] -> ok: False
    [root] gnc: a [gnc] table requires a vehicle reference (the GNC chain
      commands the 6DOF vehicle path; set vehicle = "vehicles/<file>.toml").
```

**Read.** The guard's second half, `!(ellipsoid_a_m_ > 0.0)`
(`cpp/src/gnc/ekf.cpp:516`), is likewise unreachable: `planet_a_m` is either the
WGS84 default or one of the Mars and Moon radii
(`cpp/src/vehicle_cycle.cpp:721-726`), all strictly positive.

The early return is unreachable defensive code, not a silent discard. The
review's observability argument would be correct if the branch could be
entered, and it is worth one comment recording *why* it cannot — the
unreachability is a property of the validator, three files away, and a future
central body without a body-fixed frame would make the concern live. That is a
comment, not a defect, and not a `KNOWN_ISSUES.md` entry.

## 10. `LDLT::info()` is never checked — FIX BEFORE CLOSE, with a different remedy

The gap this finding points at is real. Its stated consequence is wrong, and
its stated remedy would not fire on its own scenario. All three parts were
measured.

**Measured: the configuration is reachable, and more easily than the review
supposed.** The review could not determine whether `mission.py` independently
forbids a zero sigma. It does not:

```
ZERO startracker sigma_rad -> validated ok: True | errors: []
ZERO navfix sigmas         -> validated ok: True | errors: []
OMITTED startracker sigma_rad -> validated ok: True | errors: []
   resolved startracker: {'boresight_b': [0.0, 0.0, 1.0], 'sample_rate_hz': 1}
```

Omission is accepted too, and `NavSensorModel`'s sigma members default to zero
(`cpp/include/star/gnc/component.hpp:142-152`), so a mission that simply omits
`sigma_rad` gets `R = 0` without ever writing a zero.

**Measured: the project already enforces exactly this rule one table away.**
The same validator refuses a zero `P0` sigma, with precisely the reasoning the
review invokes:

```
p0_sigma_att_rad = [0,0,0] -> validated ok: False
    [gnc.nav] p0_sigma_att_rad: entries must be > 0 (a zero initial variance
      makes P0 singular and NEES undefined), got [0.0, 0.0, 0.0]
```

`_SENSOR_POSITIVE` (`python/star_reacher/mission.py:445-447`) covers only the
camera's `fx_px`, `fy_px`, `width_px`, and `height_px`; every other sensor
parameter is checked for `>= 0` only (`:489-498`, `:524-528`). The asymmetry is
an oversight, not a decision — `R` and `P0` fail for the same reason.

**Measured: the consequence is not NaN.** The review states that a failed
decomposition makes `solve()` return "a result that is silently wrong", that
"the gain `k` then propagates NaN or garbage into `dx`", and that "every
subsequent cycle is contaminated". Eigen 3.4.0 — the version
`CMakeLists.txt:39` pins and the version vendored into the build tree — does
not behave that way. Its `LDLT::_solve_impl_transposed` applies the
*pseudo-inverse* of `D`:

```
RealScalar tolerance = (std::numeric_limits<RealScalar>::min)();
for (Index i = 0; i < vecD.size(); ++i)
{
  if(abs(vecD(i)) > tolerance)
    dst.row(i) /= vecD(i);
  else
    dst.row(i).setZero();
}
```

A zero pivot yields a zeroed row, never a division by zero. Transcribing that
routine and `ldlt_inplace<Lower>::unblocked` from the vendored source and
running the finding's own scenario:

```
S = 0 exactly:  unblocked() returned True -> info() == Success
  solve(S=0, b=[1,2,3]) = [0. 0. 0.]  finite: True
S = diag(1e-40,1,1): info() == Success  solve = [1.e+40 1.e+00 1.e+00]
```

Two consequences follow. First, `info()` returns `Success` on an exactly
singular `S`, so the review's primary remedy — check `ldlt.info()` — would not
fire on the case it was proposed for. Second, the near-singular case that
*does* produce a dangerous gain (`1e40` above) also returns `Success`, so
`info()` is not the instrument that detects it either.

**Measured: what actually happens.** Running the Joseph update from
`cpp/src/gnc/ekf.cpp:380-402` with the flagship mission's `P0` and
`sigma_rad = [0,0,0]`:

```
update 1: S diag = [1.e-06 1.e-06 1.e-06]
          attitude block of P after = [0. 0. 0.]
          eigenvalues of attitude block = [0. 0. 0.]
update 2: S diag = [0. 0. 0.]  -> all pivots exactly zero: True
          K max |.| = 0.0    dx = [0. 0. 0.]
          P unchanged by update 2: True
```

The first zero-noise update snaps the attitude exactly onto the measurement and
drives the attitude block of `P` to exactly zero. Every subsequent star-tracker
update is a **silent no-op**: gain zero, correction zero, covariance untouched.
No NaN, no garbage, no contamination.

**Measured: the outcome is loud at analysis.** A singular `P` reaches
`consistency.py::_cholesky_or_report` (`python/star_reacher/consistency.py:256-273`),
which rescans epoch by epoch and raises
`"nav.est.P at flat epoch index N is not positive definite; a reported
covariance must be a valid covariance matrix"`. The run does not pass a gate
while being wrong.

**Why fix it anyway, and why now.** The reachable consequence is milder than
the review claimed, but it is still a silent no-op inside the deterministic
time loop: an aiding sensor the mission configures, the log declares, and the
`nav.innov` channel records, which after its first update contributes nothing
and says so nowhere. That is the shape this phase already treated seriously
when it fixed finding 4. It also has a genuinely cheap and correct remedy that
needs no compiler, which is the deciding factor at a phase boundary.

**Remedy.** In `python/star_reacher/mission.py`, require the four sigmas that
build `R` to be strictly positive, alongside the existing `_SENSOR_POSITIVE`
mechanism:

- `startracker.sigma_rad` — every entry `> 0`;
- `navfix.sigma_r_m` and `navfix.sigma_v_mps` — every entry `> 0`;
- `altimeter` — `sigma_noise_m**2 + sigma_bias_m**2 > 0`, not
  `sigma_noise_m > 0`. `cpp/src/gnc/ekf.cpp:542-545` builds
  `r(0,0) = sn*sn + sb*sb`, so a zero white noise with a configured turn-on
  bias is a legitimate configuration and must stay accepted.

Make all four required rather than optional, so omission cannot reach `R = 0`
by the back door. Carry the `P0` message's phrasing — a zero measurement
variance makes the update a no-op and NEES undefined — so the two rules read as
one rule. Do **not** add an `ldlt.info()` check: the measurements above show it
returns `Success` in both the singular and the near-singular case, so it would
be a guard that cannot fire, which this project's own finding 22 already
catalogues as a defect shape.

## 11. An invalid IMU sample freezes the covariance — ACCEPT AND DOCUMENT

**Read.** `cpp/src/gnc/ekf.cpp:176` guards propagation on
`input.imu_fresh && input.imu.valid && input.imu.dt_s > 0.0`, and the aiding
updates below run regardless. The review's description of the mechanism is
accurate.

**Read: unreachable in Phase 6.** `cpp/src/sensors/imu.cpp:254`
(`last_.valid = true`) is the only assignment to that flag anywhere in the IMU,
and `ImuSample::valid` defaults to `false`
(`cpp/include/star/gnc/component.hpp:45`). Before the first sample `imu_fresh`
is also false, so the two conditions never disagree. The review says as much,
then adds that it "becomes reachable with any sensor plugin". There is no
sensor plugin surface: `bindings/module.cpp` binds `GncSensorCfg` and
`NavSensorModel` but not `ISensor`, and FR-25 swaps GNC components, not
sensors. The IMU sample always originates in the C++ `Imu`.

**Why accept.** Nothing can enter the state the finding describes. It is a
genuine trap for whoever adds IMU gating or a sensor-plugin surface in a later
phase, and the remedy the review proposes is a design decision — advance `P` by
process noise, or refuse the cycle — that should be made when the gating that
motivates it is designed, not speculatively now. The right record is a comment
at the guard, not a `KNOWN_ISSUES.md` entry for a state the code cannot reach.

## 13. `_reduce_error` guesses the quaternion collapse — ACCEPT AND DOCUMENT

**Measured.** The mangling reproduces exactly. A six-wide error whose leading
three slots are a rotation-vector attitude block, against a five-dimensional
covariance:

```
   input : [1.e-03 2.e-03 3.e-03 1.e+01 2.e+01 3.e+01]
   output: [4.e-03 6.e-03 2.e+01 2.e+01 3.e+01] | problem reported: None
```

The first slot is dropped, the next two are doubled, and the fourth slot — an
unrelated state, here `10.0` — is doubled to `20.0` and carried forward as if
it were an attitude component. No mismatch is reported. The resulting NEES has
the right shape and is order-unity while being wrong.

**Read.** The premise holds: the estimator's error layout does not reach the
log. `cpp/include/star/srlog_writer.hpp` carries no layout field, and
`build_header_fields` receives only `!nav_layout.empty()`
(`cpp/src/vehicle_cycle.cpp:712-716`) — a boolean, not the layout. The CLI
genuinely cannot verify the assumption it makes.

**One correction to the finding's framing.** The review reads the module
docstring and `docs/formats/srlog_v1.md:337` as contradicted by the code,
because "`n == m + 1` is never reported — it is always collapsed". Those
documents say the *one sanctioned reduction* is `m == n - 1` and that **any
other** pairing is reported rather than guessed at. `n == m + 1` is the
sanctioned reduction, not an "other pairing", so the code matches the
documents. The defect is the unverifiable assumption inside the sanctioned
reduction, not a doc/code contradiction.

**Why accept.** The only producer that can reach the bad case is a plugin
estimator declaring a non-quaternion attitude block at `n == m + 1`; the
built-in EKF is quaternion-led by construction (`state_dim` 16, `cov_dim` 15).
The complete remedy — carry the declared layout in the SRLOG header and require
it before collapsing — is a v1.2 format field, and adding a header field at a
phase boundary is a larger change than the exposure justifies. Plugins are
already an explicitly contract-not-enforced surface
(`python/star_reacher/plugin.py:120-125`). The honest disposition is to record
the assumption where a plugin author will read it.

## 15. The ellipsoid is defined twice with incompatible sphere sentinels — ACCEPT AND DOCUMENT

**Read.** Both halves confirmed at this base. `EnvironmentModel::central_ellipsoid`
encodes the Moon as `inv_f = 0.0` (`cpp/src/models/environment.cpp:301-304`) and
the altimeter branches on `spherical = !(geom.ellipsoid_inv_f > 1.0)`
(`cpp/src/sensors/radio.cpp:187`). The filter's context takes
`planet_inv_f = 1.0e12` (`cpp/src/vehicle_cycle.cpp:725`) and the EKF has no
spherical branch (`cpp/src/gnc/ekf.cpp:528`). Two sources of truth for one
constant, disagreeing on the encoding of the degenerate case.

**Why accept.** The reviewer already downgraded this from an active throw to
latent after tracing it, and that downgrade holds: the `1.0e12` sentinel clears
the `inv_f > 1.0` guard in `geodetic_lat_lon_alt`, and the residual at
`f = 1e-12` is order `a·e²` ≈ 3 µm against a metre-class altimeter sigma. There
is no measurable consequence today. The hazard is entirely prospective and is
best carried as a documented duplication rather than fixed under phase-close
pressure, where harmonising two ellipsoid paths and adding a spherical branch to
the filter is a change that wants its own golden-vector evidence.

## 17. `_load_module` swallows `KeyboardInterrupt` and `SystemExit` — ACCEPT AND DOCUMENT

**Measured.** Both reproduce:

```
F17 Ctrl-C during import surfaces as: PluginError
F17 SystemExit during import surfaces as: PluginError
```

`except BaseException` at `python/star_reacher/plugin.py:175` converts both.

**Why accept.** The consequence is a Ctrl-C pressed during a plugin import
being reported as a load failure instead of interrupting. Plugin imports are a
single `exec_module` of a small source file; the window is sub-millisecond, no
automated path sends `SIGINT`, and a second Ctrl-C works. The `SystemExit` half
is arguably desirable rather than defective: a plugin that calls `sys.exit()`
during import should fail its load with a named error, not silently terminate
the run that loaded it. Fixing only the `KeyboardInterrupt` half is four lines
and carries no risk, but it buys nothing that matters at a phase boundary, and
the finding is correctly ranked LOW.

## 20. Unguarded division by `r³` — ACCEPT AND DOCUMENT

**Read.** `gravity()` (`cpp/src/gnc/ekf.cpp:355-360`) computes
`-mu_ * p / (r*r*r)` and `gravity_gradient()` (`:362-370`) computes `p / r`,
neither guarded. `p0_m` is parsed by `require_vector`
(`cpp/src/gnc/ekf.cpp:124`), which checks presence, length, and finiteness only.

**Measured.** The Python validator does not constrain it either:

```
F20: p0_m = [0,0,0] -> validator ok: True | errors: []
F20: p0_m = [1,0,0] (inside the body) -> validator ok: True | errors: []
```

**Reasoned (not measured).** At `p0_m = [0,0,0]` the numerator is also zero, so
the expression is `0.0/0.0` — NaN rather than infinity — from the first
propagation onward. Neither `cpp/src/srlog_writer.cpp` nor
`python/star_reacher/srlog.py` screens non-finite values, so the run completes
and writes an all-NaN `nav.est`. This is a code-reading inference; it was not
executed.

**Why accept.** It requires placing the filter's initial position estimate at
the exact centre of the central body — a configuration error whose output is an
entirely NaN log, unmistakable to any consumer and fatal to the consistency
Cholesky. The review's own LOW ranking is right. Note that if the finding-10
remedy lands, rejecting a zero `p0_m` is roughly two additional lines in the
same validator function, and folding it in there is cheaper than doing it
separately later.

## 21. Sun sensor emits a normalised pure-noise direction — ALREADY DOCUMENTED

**Read.** The behaviour is confirmed present at this base:
`cpp/src/sensors/optical.cpp:266-283`, where `u_b` stays zero without an
ephemeris, `sum = eta`, and `u_meas_ = (sum / n).eval()` normalises pure noise
to a unit vector with `valid = 0`.

**No action.** `docs/KNOWN_ISSUES.md:79-85`, under KNOWN-ISSUE-P6-1, already
records it in substantially the review's own terms: "with no ephemeris the sun
sensor emits `valid = 0` and, because the noise draw is unconditional and the
true direction is the zero vector, its `sun_b` channel carries a normalized
draw of pure noise rather than a direction. That is honest at the flag level
and is what the implementation documents, but a consumer reading `sun_b`
without checking `valid` would see a plausible-looking unit vector." The
finding is already dispositioned; it needs no second entry.

## 22. Minor and dead logic — NOT A DEFECT (informational)

All three sub-items behave as the review describes. None is a defect and none
needs an entry.

**22a, `kinds_ok` (`python/star_reacher/mission.py:2185`, `:2229`).** Read:
confirmed. The accumulator is set once before the loop over `_SENSOR_KINDS` and
never reset per kind, so `if kinds_ok and srate is not None:` stops populating
`resolved_kinds` for every kind after the first failure. Harmless, because
`:2232-2235` discards `resolved_kinds` wholesale and sets `ok = False` when
`kinds_ok` is false. The inner guard is dead and reads as a per-kind check;
deleting `kinds_ok and` from `:2229` would say the same thing more honestly, but
nothing depends on it.

**22b, `inside_count_threshold` (`python/star_reacher/consistency.py:400-436`).**
Measured, at the shipped `DEFAULT_CONFIDENCE = 0.999`:

```
  epochs=   1  threshold=   0   gate is vacuous (t==0): True
  epochs=   2  threshold=   0   gate is vacuous (t==0): True
  epochs=   3  threshold=   1   gate is vacuous (t==0): False
  epochs= 100  threshold=  87   gate is vacuous (t==0): False
```

The review's claim is exactly right: the coverage criterion cannot fire on runs
of one or two epochs. It is also already stated in the function's own docstring
at `:415-416` ("Returns 0 when even an empty count is admissible, which happens
only for very short runs"), which is the correct place for it. A two-epoch
consistency run is not a scenario the phase gates on.

**22c, the `sign` factor (`python/star_reacher/consistency_cli.py:173`).** Read:
confirmed. `attitude_error` (`cpp/src/gnc/component.cpp:101-103`) negates any
error quaternion with `w < 0`, so core-produced logs are always `+w` and the
`np.where(w >= 0.0, 1.0, -1.0)` factor is always `+1`. The branch is defensive
and should stay: `star consistency` reads `.srlog` files, and nothing prevents
a third-party or hand-written log from carrying an uncanonicalised error
quaternion. A branch that cannot fire on *our* producer is not the same as a
branch that cannot fire.

---

## The fix set, ranked by consequence

**1. Finding 10 — sensor noise sigmas that build `R` are unconstrained.**
Highest because it is reachable from a plain mission file with no plugin and no
driver, because omission alone is enough to trigger it, because the outcome is
a silently inert aiding sensor inside the deterministic time loop, and because
the remedy is pure Python in a function that already implements the identical
rule for `P0`. Remedy as specified in the section above: strict positivity for
`startracker.sigma_rad`, `navfix.sigma_r_m`, `navfix.sigma_v_mps`, and
`sigma_noise_m**2 + sigma_bias_m**2` for the altimeter; make all four required;
do not add an `info()` check.

**2. Finding 8 — `NavFix.valid` is hardcoded true.**
Second because no shipped run misbehaves and the built-in EKF is protected, but
it cannot be deferred past the phase that publishes FR-24: it is a semantic
defect in a public flag on a newly-shipped observation surface, and the cost of
changing it rises once drivers exist. Remedy as specified: a `sampled_` flag on
`NavFix`, exposed as `last_valid()`, forwarded at
`cpp/src/vehicle_cycle.cpp:1016`.

Finding 10's remedy needs no compiler. Finding 8's does, and should be
sequenced against whoever next holds the compiler slot.

## Exact documentation text for the accept set

Four new entries for `docs/KNOWN_ISSUES.md`, following that file's existing
heading and "exit-criterion impact" convention, plus two in-source comments.
Finding 21 needs nothing — KNOWN-ISSUE-P6-1 already covers it.

### `docs/KNOWN_ISSUES.md` — new entry, findings 7 and 17

```markdown
## KNOWN-ISSUE-P6-4 — the plugin loader caches by path, and wraps every import failure

`_load_module` (`python/star_reacher/plugin.py:153-183`) caches a loaded plugin
module by resolved path with no content check, deliberately: re-executing a
plugin file would register its component names twice, which the core refuses,
and would leave two class objects answering to one name. The cost is that a
plugin edited between two loads in one process does not take effect, while
`_plugin_provenance` (`python/star_reacher/runner.py:414-428`) re-reads the file
and records the *new* SHA-256. A `meta.json` can therefore name a digest for
code that did not execute.

Reaching that inversion requires two `run_mission` calls in one process, with a
plugin, and a source edit between them. No shipped path does this: `star run` is
one process per run, and `Sim` loads plugins but writes no `meta.json`. The
remedy, if an in-process driver is ever added, is to record the digest at load
time in `plugin.py` and have `_plugin_provenance` read the recorded value rather
than re-reading the file — not to key the cache on the digest, which would
reintroduce the duplicate-registration the cache prevents.

Separately, `python/star_reacher/plugin.py:175` wraps `BaseException`, so a
`KeyboardInterrupt` raised while a plugin is being imported is reported as a
`PluginError` instead of interrupting. The window is one `exec_module` of a
single source file. The `SystemExit` case is intentional: a plugin calling
`sys.exit()` during import should fail its load with a named error rather than
terminate the run that loaded it.

**Exit-criterion impact: none.** No exit criterion runs two plugin-bearing
missions in one process, and none sends an interrupt.
```

### `docs/KNOWN_ISSUES.md` — new entry, finding 13

```markdown
## KNOWN-ISSUE-P6-5 — `star consistency` assumes a quaternion-led attitude block at n = m + 1

When `nav.err` has dimension `n` and `nav.est` reports an `m`-dimensional
covariance with `n == m + 1`, `_reduce_error`
(`python/star_reacher/consistency_cli.py:157-181`) collapses slots 0..3 as a
scalar-first error quaternion, `dtheta = 2 sgn(dq_w) dq_v`. This is the one
sanctioned reduction and is correct for the built-in error-state EKF, whose
state is 16-dimensional and whose covariance is 15-dimensional.

The estimator's declared error layout is not written to the log — the SRLOG
header carries only a boolean for whether a layout is present — so the CLI
cannot verify the assumption. A plugin estimator that reaches `n == m + 1` with
a three-slot attitude block (`ErrorForm.ROTATION_VECTOR_LOCAL` or `_GLOBAL`, see
`docs/gnc_plugins.md:199-204`) has its leading four slots collapsed anyway: the first is
dropped, the next two are doubled, and the fourth — an unrelated state — is
doubled and carried forward as an attitude component. The resulting NEES is
positive, order-unity, and wrong.

Closing this properly means carrying the declared layout in the SRLOG header and
requiring it before the collapse; the layout already exists in the core as
`error_layout()`. That is a format field and is deferred. Until then, a plugin
estimator that is not quaternion-led at `n == m + 1` must not be evaluated with
`star consistency`.

**Exit-criterion impact: none.** Every criterion that computes NEES does so on
the built-in EKF, for which the collapse is the correct reduction.
```

> **Since:** the two `ROTATION_VECTOR_*` forms quoted above were removed, so
> the three-slot route into this misreading no longer exists. The second
> shape — an attitude block that is not first — outlived that removal and has
> since been closed at the producer: `validate_error_layout` now takes the
> component's `cov_dim()` and refuses a declared layout that reaches
> `n == m + 1` with `n >= 4` unless the attitude block holds offset 0. The
> reader-side assumption in `_reduce_error` is unchanged and still deferred
> behind the SRLOG header field. The current wording is in
> `docs/KNOWN_ISSUES.md` under KNOWN-ISSUE-P6-5.

### `docs/KNOWN_ISSUES.md` — new entry, finding 15

```markdown
## KNOWN-ISSUE-P6-6 — the reference ellipsoid reaches the sensors and the filter by two paths

The same central-body ellipsoid is built twice, with incompatible conventions
for a sphere. `EnvironmentModel::central_ellipsoid`
(`cpp/src/models/environment.cpp:291-311`) encodes the Moon as `inv_f = 0.0`,
and the altimeter tests `spherical = !(geom.ellipsoid_inv_f > 1.0)`
(`cpp/src/sensors/radio.cpp:187`) and takes a closed spherical branch.
`GncInitContext` takes `planet_inv_f = 1.0e12`
(`cpp/src/vehicle_cycle.cpp:725`), and the EKF has no spherical branch — it runs
the Bowring conversion at `f = 1e-12` (`cpp/src/gnc/ekf.cpp:529`).

Nothing misbehaves today. The `1.0e12` sentinel clears the `inv_f > 1.0` guard
in `geodetic_lat_lon_alt` (`cpp/src/models/atmosphere_hp.cpp`), and at
`f = 1e-12` the Bowring result differs from `norm(r) - a` by order `a·e²`,
about 3 µm — negligible against any configured altimeter sigma. Earth and Mars
are well above `inv_f = 1` on both paths.

The hazard is prospective: the two paths use `0.0` and `1.0e12` for the same
physical case, and `0.0` on the filter's path would mean "invalid, throw". A
future edit that harmonises one path to the other — the obvious cleanup — turns
every lunar altimeter update into a `std::domain_error` thrown from inside the
deterministic time loop. Harmonising requires taking the filter's ellipsoid from
`EnvironmentModel::central_ellipsoid` and giving the EKF the same explicit
spherical branch the altimeter has, in one change, with golden-vector evidence
for a lunar altimeter run.

**Exit-criterion impact: none.** No Phase 6 criterion runs a lunar altimeter.
```

### `docs/KNOWN_ISSUES.md` — new entry, finding 20

```markdown
## KNOWN-ISSUE-P6-7 — the filter's initial position estimate is not constrained away from the origin

`ErrorStateEkf` parses `p0_m` with `require_vector`
(`cpp/src/gnc/ekf.cpp:124`), which checks presence, length, and finiteness but
not magnitude, and `python/star_reacher/mission.py` does not constrain it
either. `gravity()` and `gravity_gradient()` (`cpp/src/gnc/ekf.cpp:355-370`)
divide by `r³` and `r` with no guard, so `p0_m = [0, 0, 0]` — the filter's
initial position estimate at the exact centre of the central body — yields
`0.0/0.0` and propagates NaN from the first cycle.

Neither the writer nor the reader screens non-finite values, so the run
completes and writes an all-NaN `nav.est`. The failure is unmistakable in the
log and fatal to the consistency evaluator's Cholesky, so it is diagnosed rather
than mistaken for a result. Rejecting a zero `p0_m` at parse time — the
precedent `require_sigma3` sets for `P0`'s sigmas — is the eventual fix.

**Exit-criterion impact: none.** Every criterion's `p0_m` is a real orbital
position.
```

### In-source comments

Two records belong in code rather than in `KNOWN_ISSUES.md`, because they
document why a branch is unreachable rather than a limitation a user can hit.

At `cpp/src/gnc/ekf.cpp:516`, extending the existing comment on the altimeter
early return (finding 9):

```
    // Unreachable as configured: bodyfixed_valid is false only for the Sun
    // (models/environment.cpp), and the validator refuses a vehicle - and
    // therefore a [gnc] chain - with central_body = "sun". Kept because a
    // future central body without a body-fixed frame would make it live, at
    // which point the skip must become observable rather than silent.
```

At `cpp/src/gnc/ekf.cpp:176`, on the propagation guard (finding 11):

```
    // imu.valid is false only before the first sample, when imu_fresh is
    // false too, so the two never disagree with the shipped Imu (it is the
    // sole writer of that flag) and no sensor-plugin surface exists to
    // supply one. If IMU gating is ever added, note that skipping the
    // propagation while the aiding updates below still run freezes P while
    // the true error keeps growing: advance P by process noise or refuse the
    // cycle, but do not hold it.
```

---

## Other inferred-consequence claims in the review

The dated correction section covers finding 1 only. Reviewing the rest of the
report against what is measurable without a compiler, three further claims
assert consequences or reachability that were inferred rather than established,
and one is a citation that no longer resolves.

**Finding 10's consequence and remedy — the most serious.** "The gain `k` then
propagates NaN or garbage into `dx`, into the reset, and into `p_` through the
Joseph form — after which every subsequent cycle is contaminated and the
covariance logged to `nav.est.P` is meaningless." Eigen 3.4.0's LDLT solve uses
the pseudo-inverse of `D` and zeroes rows at zero pivots; the measured result is
`K = 0`, `dx = 0`, `P` untouched — a silent no-op, not contamination. The
proposed remedy, checking `ldlt.info()`, returns `Success` on both the exactly
singular and the near-singular case and would not fire. This is finding 1's
error repeated: a consequence inferred from what a library plausibly does,
inside a finding whose underlying observation is nonetheless correct.

**Finding 9's reachability.** "A run configured with an altimeter on a central
body whose body-fixed frame is unavailable logs a full `sensor.altimeter`
channel with `valid = 1` throughout, and an empty `nav.innov` for that sensor."
No such run exists: the Sun is the only body with no body-fixed frame, and the
validator refuses both a vehicle at the Sun and a `[gnc]` table without a
vehicle. The whole failure scenario describes a configuration the product
rejects.

**Finding 11's escalation.** The review is candid that the state is unreachable
with the built-in IMU, then writes "It becomes reachable with any sensor plugin
or future IMU gating." The "future IMU gating" half is fair. The "any sensor
plugin" half is not: `ISensor` is not bound in `bindings/module.cpp`, and FR-25
swaps GNC components rather than sensors, so no plugin can supply an IMU sample
at all.

**Finding 7's reachability.** "Not reachable through `star run`... Directly
reachable through `Sim`." The stale-module half is reachable through `Sim`; the
provenance inversion that gives the finding its severity is not, because `Sim`
writes no `meta.json` and never calls `_plugin_provenance`. The two halves are
conflated into one reachability statement.

**Finding 15's citation.** The parenthetical "no vehicle can be configured
there anyway (`python/star_reacher/mission.py:2585`)" points at sequence
validation. The constraint is real and lives at `:2610-2615` at this base. This
is a slipped citation rather than an inferred consequence, and the underlying
claim is correct — it is recorded only because a citation that does not resolve
is the same class of problem.

Two claims were checked specifically because they looked inferable and turned
out to be exactly right: finding 22b's "returns 0 for `epochs <= 2`" (measured,
correct at the shipped default) and finding 22c's "always `+1` for
core-produced logs" (confirmed in `attitude_error`). Reporting those is part of
the answer: the review's error rate is not uniform, and the failures cluster in
consequence claims about compiled behaviour, not in claims about Python.

## Not assessable without a compiler

Nothing in the eleven was left undetermined for want of a build. Two items
would be worth confirming against a binary once the compiler slot frees, both
to close the loop on measurements made by transcription rather than execution:

- **Finding 10's numerical outcome.** Run
  `missions/leo_ekf_consistency.toml` with
  `[sensors.startracker] sigma_rad = [0.0, 0.0, 0.0]`. Expect the run to
  complete, `nav.est.P`'s attitude block to be exactly zero from the first
  star-tracker epoch onward, every subsequent `nav.innov` record for that
  sensor to carry an all-zero `S`, and `star consistency` to fail with
  `"nav.est.P at flat epoch index N is not positive definite"`. The prediction
  comes from a NumPy transcription of the Joseph update and of Eigen 3.4.0's
  `_solve_impl_transposed`; the binary is what settles it. If instead the run
  produces NaN, the review's original consequence claim was right and this
  triage is wrong.
- **Finding 20's NaN path.** Run the same mission with
  `p0_m = [0.0, 0.0, 0.0]`. Expect an all-NaN `nav.est` channel and a completed
  run rather than a throw. This one was reasoned from code alone and is the
  weakest-evidenced statement in this document.
