# Known issues

Tracked defects and documented limitations, with the exit-criterion impact of
each stated plainly. Entries are removed when fixed (with the fixing commit
noted in the changelog history, not here).

## KNOWN-ISSUE-P4-1 — intermittent native fault on the high-volume by-source log path (mitigated)

**Symptom (original).** With the FR-16 `forces`/`mass`/`env` channel groups
enabled *and* a very large record count (order 10^5 records, order 100 MB of
log), the run intermittently aborted partway through with a native memory fault
(access violation / `0xC0000005`, occasionally surfacing as
`0xC0000409`), leaving a truncated `run.srlog` and no `meta.json`. Measured on
the build host at roughly 8 faults in 48 runs of the trans-lunar case
(`missions/tli.toml` with all groups at 1 Hz: ~455k records, ~211 MB).

**Investigation.** The propagation is deterministic — every full-groups run
that *completes* produces a bit-identical log — so the fault was confined to the
high-volume by-source write path, not the computed trajectory. The SRLOG writer
streams directly to the file with no unbounded in-memory buffer (peak RSS ~444
MB for a 211 MB log), and every record write is bounds-checked. The per-cycle
logging assembly was examined under four independent memory tools — Linux
AddressSanitizer at the full 211 MB volume, UndefinedBehaviorSanitizer
(including alignment), Valgrind memcheck with `--track-origins`, and an earlier
MSVC AddressSanitizer pass — and **none reported a memory-safety defect**
(Valgrind: zero errors, all ~180k allocations freed). No code-level buffer
overrun, use-after-free, or uninitialized read was found. The fault correlates
with high-frequency heap-allocation churn during the large-volume write; a
code-level root cause could not be isolated, and a contribution from build-host
instability cannot be excluded (this host's compiler intermittently faults with
the same access-violation code during compilation).

**Mitigation.** The per-source forces record now reuses a single buffer across
the whole run instead of allocating and freeing a fresh vector every logged
step (`cpp/src/run.cpp`). This removes ~455k per-cycle allocations on the
trans-lunar case and eliminated the fault across 45 consecutive full-groups runs
(versus ~17 % previously), with **byte-identical** log output (same SHA-256).
The change is a determinism-preserving optimization; it does not alter the
logged bytes.

**Residual caveat.** Because no code defect was isolated, the possibility of an
environmental (build-host) contribution remains. The mitigation removes the
observed symptom but is not proven to address a specific logic defect, since
none was found.

**Exit-criterion impact: none.** No Phase 4 exit criterion depends on the
by-source groups at high volume. EC-6 evaluates `missions/tli.toml` in its
committed configuration (truth records plus the SOI-transition event), which is
reliable and bit-reproducible.

## KNOWN-ISSUE-P4-2 — FR-16 `thirdbody` force channel lumps the environment residual

The vehicle run path's `forces` group emits the sources `gravity`, `thirdbody`,
`aero`, `thrust`, and `gravgrad`. The `thirdbody` channel value is the full
non-central-gravity environment residual (central-body gravity subtracted from
the composed environment acceleration), not strictly the third-body term. For
every shipped mission this residual *equals* the third-body acceleration
(`missions/ascent_leo.toml` enables no third bodies; `missions/tli.toml` enables
Sun and Moon with no SRP or orbital drag), so the logged value is exact for what
ships. A future vehicle mission that enables environment SRP or orbital drag
would fold those into the `thirdbody` channel rather than emitting the separate
`srp`/`drag` sources named in `docs/formats/srlog_v1.md`. Per-source
decomposition of the environment terms in the vehicle path is deferred.

**Exit-criterion impact: none.** No Phase 4 exit criterion tests a per-source
environment force decomposition, and no shipped mission enables SRP or orbital
drag on the vehicle path.

## KNOWN-ISSUE-P6-1 — the mission validator does not count the FR-23 optical sensors as ephemeris consumers

`[environment] ephemeris` is rejected unless a *force* model consumes it: the
check in `python/star_reacher/mission.py` accepts third bodies, SRP,
Harris-Priester drag, or a Moon central body, and nothing else. The FR-23 sun
sensor and star tracker also require an ephemeris — the sun sensor has no Sun
direction to measure without one, and the star tracker's Sun and central-body
exclusion cones silently stop gating — but configuring either does not make the
key acceptable.

The consequence is that a mission whose *only* ephemeris consumer is an optical
sensor cannot be written. The sensor still runs: with no ephemeris the sun
sensor emits `valid = 0` and, because the noise draw is unconditional and the
true direction is the zero vector, its `sun_b` channel carries a normalized
draw of pure noise rather than a direction. That is honest at the flag level
and is what the implementation documents, but a consumer reading `sun_b`
without checking `valid` would see a plausible-looking unit vector.

The workaround, used by `tests/python/test_p6_optical_gates.py`, is to enable
the Sun and Moon third bodies alongside the ephemeris. This is physically
legitimate rather than a stub, but it forces an unrelated force model into any
mission that wants a working sun sensor.

**Exit-criterion impact: none.** Phase 6 exit criteria 6, 7, and 9 are gated on
missions that enable third bodies, so every optical channel they read is
ephemeris-backed.

## KNOWN-ISSUE-P6-2 — exit criterion 9's 1 mas figure is a requirement, not a resolution limit

**Status: remediated 2026-07-19.** The entry is rewritten rather than deleted,
because its earlier framing is what kept a loose gate in place and that is
worth recording.

This entry previously described 1 milliarcsecond as a *budget* that "cannot
resolve second-order aberration algebra". That reasoning conflated two
different quantities. Terms of order `beta**2` are indeed worth up to
`beta**2 / 2 ~ 1.08 mas` at `|beta| ~ 1.02e-4`, but that bounds the difference
between two algebraically distinct *forms* — it says nothing about the
precision at which either form can be checked. Criterion 9 recomputes the
normative first-order equation and compares it against an implementation of
that same equation, and two evaluations of one formula agree to rounding: the
measured worst residual is `4.73e-08 mas`. Gating at 1 mas therefore carried
roughly `2.1e+07x` headroom, and 1 mas was only ever the criterion's
*requirement*.

The cost of the wrong framing was measurable. Dropping the transverse
projection from the reference (`u + beta` in place of
`u + beta - (u . beta) u`) changes the answer by `0.4696 mas` on this fixture,
and **passed** the 1 mas gate.

Two fixes landed in `tests/python/test_p6_optical_gates.py`:

- `ABERRATION_TOL_MAS` is now `1e-5`, which keeps about `210x` headroom over
  the observed residual while rejecting the drop-transverse mutation by about
  `4.7e+04`. The criterion's own 1 mas figure is asserted alongside it, so the
  requirement is still stated in the suite.
- The reference side now rotates through `tests/refs/quaternions.quat_to_dcm`
  instead of `_core.quat_to_dcm`. The core's DCM previously appeared on both
  sides of an angular separation and cancelled exactly, so no attitude-
  convention error could reach the residual. The fixture's commanded attitude
  was also moved off the body +Z axis: the old slew held `q_w == 0` for the
  whole run, where `C - C^T = -4 q_w [q_v x]` vanishes identically, so a
  transposed convention was undetectable by geometry regardless of which
  implementation supplied the DCM. With the off-axis slew that mutation is
  rejected at `4.03e+07 mas`;
  `test_aberration_fixture_can_see_an_attitude_convention_error` pins the
  asymmetry so the fixture cannot drift back.

What remains true, and is a modelling choice rather than a gate weakness: the
first-order versus exact relativistic difference is material at 0.51 mas, which
is why `ch:sensors-optical` declares the first-order equation normative and
criterion 9 recomputes *that* form. `test_first_order_versus_exact_gap_is_recorded`
measures that gap non-normatively; `ch:sensors-optical` assumption 2 bounds it.

**Exit-criterion impact: none.** The criterion was met as written before and is
met now, but it is now gated by an assertion that has been shown to fail
against a wrong formula and a wrong convention.

## KNOWN-ISSUE-P6-3 — resolved: the pitch program's roll at a true vertical

**Status: the guidance singularity is fixed.** This entry is kept because the
fix relocated a discontinuity rather than removing one; the relocation has been
reviewed and accepted, and the reasoning is recorded below.

The pitch table shared by `missions/ascent_leo.toml` and
`missions/ascent_leo_gnc.toml` holds pitch at exactly 90 degrees — the local
vertical — from t = 0 to t = 10 s. The commanded body axis is then local up,
which is also the reference the triad construction of
`eq:vehicle6dof:attitude` projects against, so that construction was
degenerate and fell back to an azimuth-independent inertial axis. The first
cycle that left the vertical re-resolved the roll and stepped the commanded
attitude **89.922 degrees between two consecutive 0.1 s cycles**, against
0.100 degrees for every other cycle, logging a commanded body rate of
809.73 deg/s.

The fix is in the law, not the table: `models::pitch_program_roll_ref`
evaluates the closed form `eq:vehicle6dof:rollref` of the same Gram-Schmidt
where the direct construction is ill conditioned. The commanded azimuth
therefore continues to fix the roll through the vertical. Measured on
`missions/ascent_leo.toml` after the fix, the largest single-cycle attitude
change across the former singularity is **0.100000 degrees**, exactly the
typical cycle, at a commanded rate of 1.0000 deg/s. The closed-loop mission's
commanded attitude is now continuous over its whole run: its largest
single-cycle change is 0.100000 degrees.

The trajectory did not move. Logged `truth.r_m` is bit-identical to the
pre-fix run and `truth.v_mps` differs by at most 2.8e-14 m/s; the insertion
state reduced at the exact perigee crossing is bit-identical, so the EC-11
cross-check and its sanity bands are unchanged rather than merely re-passed.
Only `truth.q_i2b` and `truth.w_b_radps` over t in [2.0, 10.0] s, and the
body-frame resolution of the logged forces over that window, changed.

### Accepted decision — the roll convention at pad release

The pad-hold mode clocks body +Y to local north; the pitch program clocks it
into the pitch plane, which at the vertical is the ground-track direction.
For the reference mission's 90-degree azimuth those differ by 90 degrees, and
they always will: the two modes define roll independently, and no pitch-plane
convention agrees with a north convention at an arbitrary azimuth. That
90 degrees has to appear somewhere in the open-loop command.

Before the fix it was split — 0.083 degrees at release, where the fallback
happened to land near geocentric north, and 89.922 degrees at t = 10 s.
After the fix it appears in one place, as a **90.005-degree single-cycle step
at t = 1.9 -> 2.0 s**, the cycle where attitude authority passes from pad hold
to the pitch program. In the closed-loop mission this is an initial attitude
error at loop closure rather than a mid-flight command step: tracking error
peaks at 90.005 degrees at t = 2.0 s and settles to a 0.0081 degree median,
under 0.033 degrees after t = 140 s.

**The relocation has been reviewed and accepted deliberately.** The reasoning
is that the 90 degrees is not an artifact to be removed but a consequence of
the two modes defining roll independently, so the only open question was where
it should land. A mode boundary — the cycle on which attitude authority changes
hands — is an explicable place for a commanded-attitude discontinuity. Inside a
smooth guidance segment, where the pre-fix step sat, is not: a discontinuity
there has no corresponding event in the mission and reads as a defect in the
guidance law. Relocating it is therefore an improvement in attribution, and it
is the improvement this fix was for.

What the fix delivers is a clean closed-loop **commanded** attitude, which is
what the GNC stack actually consumes. Measured on `missions/ascent_leo_gnc.toml`,
the largest single-cycle change in the logged `gnc.cmd` command is
**0.100000 degrees**, exactly the typical cycle, with no cycle exceeding one
degree anywhere in the run.

**The open-loop reference mission's logged truth still contains the step, and
this entry is not a claim that the fix removed it.** Measured on
`missions/ascent_leo.toml`, `truth.q_i2b` still steps **90.004996 degrees
between t = 1.90 s and t = 2.00 s** — about 900x the 0.100000-degree change of
every other cycle in the pitch program, and the only single-cycle change above
one degree in the whole 400 s flight. A reimplementer inherits it and should
expect to see it.

Removing it entirely was considered and rejected on cost. It would mean
clocking the pad-hold attitude to the flight azimuth, which requires giving pad
hold an azimuth it does not currently have and changes the initial attitude of
every pad mission in the repository, or modelling an explicit rate-limited roll
program after liftoff. Neither is warranted by a discontinuity that is now
confined to one cycle at a mode boundary and absent from the commanded signal.
If a future phase makes the open-loop ascent an attitude-truth benchmark rather
than a trajectory benchmark, that is the point to revisit it.

**Exit-criterion impact: none for criterion 10**, which gates throughput
rather than tracking accuracy, and the closed-loop mission still reaches
orbit insertion (180.7 x 3356.1 km, against the open-loop 181 x 3444 km).
It is recorded because the transient is visible in every plot of the
closed-loop ascent and would otherwise read as a controller defect, and
because a smoothed pitch table is the obvious remediation if a future phase
wants the closed-loop ascent to be a tracking benchmark rather than a
throughput benchmark.

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
reintroduce the duplicate registration the cache prevents.

Separately, `python/star_reacher/plugin.py:175` wraps `BaseException`, so a
`KeyboardInterrupt` raised while a plugin is being imported is reported as a
`PluginError` instead of interrupting. The window is one `exec_module` of a
single source file. The `SystemExit` case is intentional: a plugin calling
`sys.exit()` during import should fail its load with a named error rather than
terminate the run that loaded it.

**Exit-criterion impact: none.** No exit criterion runs two plugin-bearing
missions in one process, and none sends an interrupt.

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
`docs/gnc_plugins.md:199-204`) has its leading four slots collapsed anyway: the
first is dropped, the next two are doubled, and the fourth — an unrelated state
— is doubled and carried forward as an attitude component. The resulting NEES is
positive, order-unity, and wrong.

Closing this properly means carrying the declared layout in the SRLOG header and
requiring it before the collapse; the layout already exists in the core as
`error_layout()`. That is a format field and is deferred. Until then, a plugin
estimator that is not quaternion-led at `n == m + 1` must not be evaluated with
`star consistency`.

**Exit-criterion impact: none.** Every criterion that computes NEES does so on
the built-in EKF, for which the collapse is the correct reduction.

## KNOWN-ISSUE-P6-6 — the reference ellipsoid reaches the sensors and the filter by two paths

The same central-body ellipsoid is built twice, with incompatible conventions
for a sphere. `EnvironmentModel::central_ellipsoid`
(`cpp/src/models/environment.cpp:291-311`) encodes the Moon as `inv_f = 0.0`,
and the altimeter tests `spherical = !(geom.ellipsoid_inv_f > 1.0)`
(`cpp/src/sensors/radio.cpp:187`) and takes a closed spherical branch.
`GncInitContext` takes `planet_inv_f = 1.0e12`
(`cpp/src/vehicle_cycle.cpp:725`), and the EKF has no spherical branch — it runs
the Bowring conversion at `f = 1e-12` (`cpp/src/gnc/ekf.cpp:528`).

Nothing misbehaves today. The `1.0e12` sentinel clears the `inv_f > 1.0` guard
in `geodetic_lat_lon_alt` (`cpp/src/models/atmosphere_hp.cpp:157`), and at
`f = 1e-12` the Bowring result differs from `norm(r) - a` by order `a e^2`,
about 3 um — negligible against any configured altimeter sigma. Earth and Mars
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

## KNOWN-ISSUE-P6-7 — the filter's initial position estimate is not constrained away from the origin

`ErrorStateEkf` parses `p0_m` with `require_vector`
(`cpp/src/gnc/ekf.cpp:124`), which checks presence, length, and finiteness but
not magnitude, and `python/star_reacher/mission.py` does not constrain it
either. `gravity()` and `gravity_gradient()` (`cpp/src/gnc/ekf.cpp:355-370`)
divide by `r^3` and `r` with no guard, so `p0_m = [0, 0, 0]` — the filter's
initial position estimate at the exact centre of the central body — yields
`0.0/0.0` and propagates NaN from the first cycle.

The run does not complete. Measured on `missions/leo_ekf_consistency.toml`
with `p0_m` replaced by `[0, 0, 0]` and nothing else changed, `star run`
**aborts at t = 1.0 s** — the first aiding epoch, where the 1 Hz nav fix, star
tracker, and altimeter all take their first sample — with
`ValueError: quat_normalize: zero or non-finite quaternion` and **exit
code 1**. The NaN reaches the attitude correction on the first measurement
update, and normalizing the corrected quaternion is the guard that stops the
run. The log is left truncated at **10 records** per cycle-rate group against
the **601** the completed 60 s mission writes.

The corruption is partial, not total, which matters when reading such a log:

| group | non-finite | share |
| --- | --- | --- |
| `nav.est` `x_hat` | 54 / 160 | 33.8% |
| `nav.est` `P` | 1002 / 1200 | 83.5% |
| `nav.err` `e` | 54 / 160 | 33.8% |
| `truth` | 0 / 150 | 0% |
| `gnc.cmd` | 0 / 110 | 0% |
| all four `sensors.*` groups | 0 / 84 | 0% |

Epoch 0 is fully finite in both `x_hat` and `P`. From epoch 1 onward exactly
six of the sixteen states — the three position and three velocity components —
are non-finite, and that set does not grow; the four quaternion components and
the six bias states stay finite in every logged epoch. Truth, the commanded
attitude, and every sensor channel are untouched, because none of them is
computed from the filter's estimate.

That behavior is safer than a completed run: the abort is loud, immediate, and
impossible to mistake for a result, and no full-length log of a diverged filter
is produced to be analyzed by accident. It is still worth closing, because the
diagnostic points at the quaternion rather than at `p0_m`, which is where the
error actually is. Rejecting a zero `p0_m` at parse time — the precedent
`require_sigma3` sets for `P0`'s sigmas — remains the fix, and would move the
failure to validation time with a message that names the offending key.

**Exit-criterion impact: none.** Every criterion's `p0_m` is a real orbital
position.
