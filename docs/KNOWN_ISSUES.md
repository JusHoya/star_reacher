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

## KNOWN-ISSUE-P6-3 — the ascent pitch table steps 89.922 degrees at t = 10 s

The pitch table shared by `missions/ascent_leo.toml` and
`missions/ascent_leo_gnc.toml` holds pitch at exactly 90 degrees — the local
vertical — from t = 0 to t = 10 s. The pitch-program axis is degenerate there:
with the commanded body axis along the local vertical, the azimuth cannot be
resolved. The moment pitch leaves 90 degrees the azimuth resolves and the
commanded attitude steps **89.922 degrees between two consecutive 0.1 s
cycles**, against 0.100 degrees for every other cycle in the run.

Open-loop flight never revealed this. The open-loop mission's
`pitch_program` sequence action sets attitude kinematically, so the true
attitude simply teleports through the step and the trajectory is unaffected —
the discontinuity is present in `ascent_leo.toml`'s own logged `truth.q_i2b`
and has been since Phase 4. Closing the loop is what exposes it: a vehicle
driven by torque must physically slew through the step. On the closed-loop
mission the controller saturates briefly and takes roughly 120 s to bleed the
transient out, which is the whole of the atmospheric phase; tracking error
peaks at 89.7 degrees at t = 10.1 s and settles to a 0.0083 degree median
after t = 140 s.

The table is deliberately **not** smoothed. Holding it bit-identical between
the two missions is what makes
`tests/python/test_gnc_missions.py::test_pitch_program_guidance_equals_openloop_command`
meaningful, and changing it would move the Phase 4 ascent goldens and the
EC-11 3DOF cross-check for a reason unrelated to either.

**Exit-criterion impact: none for criterion 10**, which gates throughput
rather than tracking accuracy, and the closed-loop mission still reaches
orbit insertion (180.7 x 3356.1 km, against the open-loop 181 x 3444 km).
It is recorded because the transient is visible in every plot of the
closed-loop ascent and would otherwise read as a controller defect, and
because a smoothed pitch table is the obvious remediation if a future phase
wants the closed-loop ascent to be a tracking benchmark rather than a
throughput benchmark.
