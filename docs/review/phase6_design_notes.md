# Phase 6 design notes

Two Phase 6 questions that are decisions or measurements rather than code
changes: what to do about the rotation-vector attitude block's slot mismatch,
and whether the closed-loop ascent's documented insertion apogee is right.

Both were settled against branch `phase-6-gnc-sensors` at `376e068`. Every
number below was produced by the installed core reporting
`git_hash() == 376e0680eb4c998463c82120d9d9edadf307971f`, so the binary and
the source read here are the same revision. Nothing was rebuilt for this
document; where a conclusion rests on reading rather than on a measurement it
is labelled as such.

## 1. The rotation-vector attitude-block slot mismatch

### 1.1 The inconsistency

`error_block_size` reports **three** slots for `kRotationVectorLocal` and
`kRotationVectorGlobal` (`cpp/src/gnc/component.cpp:116-118`), and that is the
width `validate_error_layout` tiles the *state* vector with
(`component.cpp:158`, `:177`). `compute_error_state` then reads **four**
consecutive slots at the same offset and assembles them into a quaternion
(`component.cpp:194-195`).

A layout that passes validation therefore cannot supply the fourth slot out of
its own block. Two shapes follow, and both are reachable:

- the attitude block is **not** last, and the fourth component of the
  "quaternion" is the first slot of the neighbouring block;
- the attitude block **is** last, and the read is one `double` past
  `x_hat_buf`, which `vehicle_cycle.cpp:817` sizes by
  `assign(state_dim(), 0.0)` on the vector declared at `:686`.

### 1.2 Measured exposure

This was not left to reading. `ErrorForm.ROTATION_VECTOR_LOCAL` and
`ROTATION_VECTOR_GLOBAL` are exposed to Python at
`bindings/module.cpp:1341-1344` and documented at `docs/gnc_plugins.md:204`,
so a nav plugin loaded through the shipped `star run --gnc-plugin` path can
select them. Two probe plugins were written against the installed wheel and
flown on a one-second variant of `missions/leo_attitude_gnc_plugin.toml`, each
declaring `state_dim() == 9`, publishing the sentinel state
`[1, 2, ..., 9]` from `state()`, and differing only in where the attitude
block sits:

| Probe | Layout | Slots read as the quaternion |
|---|---|---|
| `rv_first` | attitude@0, velocity@3, position@6 | `x_hat[0..3]`, in bounds |
| `rv_last` | position@0, velocity@3, attitude@6 | `x_hat[6..9]`, **`x_hat[9]` out of bounds** |

Both **passed `validate_error_layout` and ran to completion.** Neither raised,
neither aborted, and both wrote a well-formed nine-wide `nav.err`.

`rv_first` produced `e[0..2] = [7.071068, -4.242641, 1.414214]` on the first
record. Composing the declared reduction by hand for
`q_est = (1, 2, 3, 4)` — that is, the three attitude slots plus the velocity
block's first slot — against the truth attitude `q0 = [0, a, a, 0]`,
`a = sqrt(1/2)`, gives `2 dq_v = (10a, -6a, 2a)`, which is those three numbers
exactly. The velocity slot is consumed twice: once as the quaternion's `z`
component and once, correctly, as velocity.

`rv_last` produced `e[6..8] = [9.899495, 9.899495, 1.414214]`. Solving the
same closed form backwards for the unknown fourth component recovers the value
the out-of-bounds slot supplied:

    e[6] = 2a(7 + Z),  e[7] = 2a(7 - Z),  e[8] = 2a,  q_est = (7, 8, 9, Z)

    Z from e[6]:  -8.88e-16
    Z from e[7]:   1.78e-15

`x_hat[9]` read as **0.0** — the residuals are the solve's own rounding. The
read therefore produced a finite, plausible, entirely wrong error state with no
diagnostic of any kind. Three consecutive `rv_last` runs were **byte-identical**
(`run.srlog` SHA-256 `11dd4a65b1c2aedb64b9a079ea502b82398a76e447b7ab4d03bd95a4d574d36e`),
so on this host the defect is silent *and* stable, which is the worst of the
available combinations: it looks exactly like working software.

That the adjacent heap word happens to be zero here is a property of the
allocator, not of the code. `docs/review/phase6_coverage_closure.md` records
the measured ASan behaviour of the sibling buffers `innov_y_buf` and
`innov_s_buf`, which use the identical `assign`-on-a-fresh-vector pattern:
an overrun leaves the allocation and is reported as `heap-buffer-overflow`,
"0 bytes after a 24-byte region". The inference that a sanitizer build would
flag `x_hat[9]` likewise is a reading of that record plus the shared
allocation pattern, not a measurement of this buffer; it was not run, because
running it requires a build.

**Verdict on exposure.** No configuration committed to this repository reaches
the forms: the reference EKF declares `kQuatErrorLocal`
(`cpp/src/gnc/ekf.cpp:248`) and `dead_reckoning` declares
`kQuatDifferenceAligned` (`cpp/src/gnc/builtin.cpp:164`). But this is **more
than a latent API trap**. It is reachable today, without recompiling anything,
by a user-authored Python nav plugin using a documented enum value on a
documented command line — which is precisely the audience `docs/gnc_plugins.md`
is written for. What bounds the damage is that the user has to write the
plugin; what makes it serious is that nothing tells them they are wrong.

### 1.3 Why neither obvious reading works

The descriptor conflates two widths that are only accidentally equal. A block
has a **state width** (how many `x_hat` slots it consumes) and an **error
width** (how many `e` slots it produces). `error_block_size` returns one number
and both callers use it: `validate_error_layout` as the state width,
`compute_error_state` as the offset arithmetic for the error width. For every
form except the rotation-vector pair the two are equal, which is why the
conflation survived design, review, and two audits.

For the rotation-vector forms they are genuinely different — state width 4, a
quaternion; error width 3, its small-angle reduction — and no single return
value can be right. Returning 3 breaks the read; returning 4 leaves an `e` slot
nothing writes, and `nav.err` is declared with the state dimension
(`cpp/src/srlog_writer.cpp:434-440`), so that slot logs as zero, which the
design forbids on the grounds that a zero in an error channel reads as "no
error" rather than "not known" (`component.hpp:320-325`).

### 1.4 The capability already exists once

The decisive context is that the reduction these forms name is **already
defined, already implemented, and already applied downstream**.
`docs/formats/srlog_v1.md:322-336` fixes `nav.err` at the state dimension `n`
and gives the consistency evaluator the job of collapsing a quaternion-led
error state to the `m = n - 1` the covariance describes, by

    dtheta = 2 * sign(dq_w) * dq_v

naming the reference EKF (`n = 16`, `m = 15`) as exactly that case. That is the
same arithmetic `compute_error_state` performs in its rotation-vector branch
(`component.cpp:219-222`). The `ROTATION_VECTOR_*` forms are therefore not a
missing capability. They are a second, redundant, and broken implementation of
a convention the pipeline already carries, placed at the one point in the chain
where the state vector's own width contradicts it.

### 1.5 Options

**A — split the state width from the error width in the descriptor.** Add
`state_block_size` beside `error_block_size`; tile `[0, state_dim)` with the
former; accumulate the latter into a new `err_dim`; give `compute_error_state`
both offsets.

*Cost:* `nav.err` stops being `n` wide, so the SRLOG header needs a third
dimension beside `nav_state_dim` and `nav_cov_dim`, the writer's
`nav_err_enabled` flag stops being sufficient (`srlog_writer.hpp:97-103`
pins the width to `n` deliberately, "so a file cannot even express a
mismatched declaration"), `srlog_v1.md:312-313` and `:322-336` both change, the
reader and the consistency evaluator gain an `m == err_dim` case, and the
`ErrorBlock.offset` invariant that the state and error vectors share one
offset (`component.hpp:304-306`) is broken. *Breaks:* the format contract, in a
revision other workstreams are actively building against. *Buys:* a `nav.err`
that is directly NEES-able without the evaluator's reduction — which the
evaluator already does correctly.

**B — require an accompanying quaternion block.** Make a rotation-vector block
legal only when the layout also declares the quaternion it reduces.

*Cost:* the quaternion block would occupy its own state slots and produce its
own error slots, so the attitude error is logged twice in two
parameterizations, and the "declared blocks tile the state exactly" rule has to
grow an exception for a block that is not really a block. This is Option A with
extra steps and a worse log; it is listed because it is the obvious reading and
it does not survive contact with the tiling rule.

**C — reject the rotation-vector forms at validation.** Have
`error_block_size` throw `std::invalid_argument` for both, naming
`kQuatErrorLocal`/`kQuatErrorGlobal` and the evaluator's documented reduction
as the supported route.

*Cost:* two public enum values that exist only to throw. *Breaks:* nothing
shipped. Smallest diff, preserves the binding surface.

**D — remove the two enum values.** Option C's honest form: delete
`kRotationVectorLocal` and `kRotationVectorGlobal` outright.

*Cost:* an enum-value removal from a public binding. *Breaks:* nothing
released — Phase 6 is not on `main` (`main` is at `d08b6a6`, the Phase 5
merge), so `ErrorForm` has never shipped with these values in a tagged
release.

### 1.6 Recommendation

**Option D, with Option C as the fallback if the binding surface must be held
stable.**

The argument is section 1.4. These values do not name a capability the system
lacks; they name one it already has, one layer up, correctly implemented and
specified. Keeping them under Option A means maintaining two implementations of
`dtheta = 2 sgn(dq_w) dq_v` in different modules and paying a format field for
the privilege. Keeping them under Option C means shipping public API that
exists only to raise. Removing them leaves exactly one definition of the
convention, in the place the format spec already points at.

The capability that *is* genuinely missing after removal is an estimator whose
attitude state is a three-parameter set — MRPs, Gibbs vector — rather than a
quaternion. That estimator is unserved today too: `attitude_error` needs a
`q_est` to compose against (`component.cpp:91-105`) and a three-parameter state
does not publish one. Serving it means a new form that reads three slots and
*reconstructs* a quaternion, which is a different feature from what these two
values name. Removal makes that gap visible instead of appearing to fill it.

Note what removal costs the evidence record: `cpp/tests/test_gnc.cpp:420`
(`gnc_rotation_vector_error_forms_match_the_analytic_reduction`) becomes
unbuildable, and with it mutation rows M15 through M18 in
`docs/review/phase6_coverage_closure.md`. That closure document is a dated
record of a measurement and should not be rewritten; the finding it records as
"C-2 closed as coverage, defect open" is resolved by this change, and the
resolution belongs here rather than as an edit there.

### 1.7 Implementation sketch for Option D

1. `cpp/include/star/gnc/component.hpp` — delete `kRotationVectorLocal` and
   `kRotationVectorGlobal` from `ErrorForm` with their comments (`:288-293`);
   correct the `kAttitude` comment at `:261` from "4 or 3 slots depending on
   the form" to four slots.
2. `cpp/src/gnc/component.cpp` — delete the `return 3` arm at `:116-118`; the
   remaining three attitude forms all return 4 and the switch stays
   exhaustive. In `attitude_error`, reduce the `local` test at `:96-97` to
   `form == ErrorForm::kQuatErrorLocal`. In `compute_error_state`, replace the
   `if`/`else` at `:212-223` with the unconditional four-slot write, which
   removes the branch that reads three.
3. `bindings/module.cpp` — delete the two `.value(...)` lines at `:1341-1344`
   and the `ROTATION_VECTOR` sentence at `:1331`.
4. `docs/gnc_plugins.md:204` — drop the `ROTATION_VECTOR_*` clause from the
   `ErrorForm` paragraph and point at `docs/formats/srlog_v1.md:327-333` for
   where the small-angle reduction is applied.
5. `cpp/tests/test_gnc.cpp:420` — the analytic content of this case (the
   closed form `2 sin(theta/2) u`) is still worth keeping, but it now belongs
   to the consistency evaluator's reduction rather than to
   `compute_error_state`. Retarget it there, or replace it with a case
   asserting that every admissible attitude form occupies four slots.

Under Option C instead, only steps 2 (throw rather than delete) and 4 apply,
and the test at step 5 is retargeted to assert the rejection.

Neither option was compiled or tested here; both sketches are from reading.

## 2. The closed-loop ascent insertion apogee

`docs/KNOWN_ISSUES.md:234` records the closed-loop GNC ascent inserting at
"180.7 x 3356.1 km, against the open-loop 181 x 3444 km", and
`missions/ascent_leo_gnc.toml:181-182` repeats the closed-loop figure as
"3,356 km ... (-2.6 %)". Neither number is computed by any test; both are
prose.

### 2.1 The repository has two reduction methods, and they differ by 28 km

There is no shared apsis reduction in the shipped library —
`python/star_reacher/derived.py` returns `a_m` and `e` but no apsis altitude —
so the reduction lives in test code, in two places that agree on the
arithmetic and disagree on the point:

**EC-6 — the final truth record.** `tests/python/test_vehicle_missions.py:41-51`
computes the osculating apsides and `:76-81` applies them to
`truth["r_m"][-1]`. There is no root refinement: the `perigee_above` condition
trips on a 0.1 s cycle boundary (`cpp/src/vehicle_cycle.cpp:1204-1206`),
`terminate` sets `stop` (`:1287`), and `:1374` forces a final truth record on
that cycle. The last record therefore *overshoots* 180.000 km.

**EC-11 — the exact 180.000 km perigee crossing.**
`tests/crosscheck/ascent_3dof.py:621-647` brackets the first upward crossing of
the target perigee and linearly interpolates the state to it;
`tests/python/test_crosscheck_ascent.py:79-96` applies that to the 6DOF log.
The docstring at `test_crosscheck_ascent.py:21-24` states the reason: comparing
both trajectories at the same exact perigee makes the perigee agreement exact
by construction, so the apogee reflects insertion energy rather than the step
at which each run trips the gate.

Both use `mu = 3.986004418e14` and `R = 6378137.0` and were confirmed
numerically identical to the core's `gm("earth")` during this work.

### 2.2 Measurements

Both missions were run once from the worktree against the `376e068` core and
reduced by both methods.

| Run | Method | Perigee (km) | Apogee (km) | Speed (m/s) | t (s) |
|---|---|---:|---:|---:|---:|
| `ascent_leo.toml` | EC-6, final record | 181.0318 | 3444.1456 | 8405.129 | 400.4000 |
| `ascent_leo.toml` | EC-11, exact perigee | 180.0011 | 3416.0804 | 8400.035 | 400.3146 |
| `ascent_leo_gnc.toml` | EC-6, final record | 180.7025 | **3357.9824** | 8399.039 | 400.3000 |
| `ascent_leo_gnc.toml` | EC-11, exact perigee | 180.0021 | 3338.1273 | 8395.400 | 400.2389 |

Two independent checks that the reduction pipeline used here is the
repository's own:

- the EC-11 open-loop row reproduces the committed EC-11 figures at
  `tests/crosscheck/manifest.toml:77-79` — 3416.1 km and 8400.0 m/s — to
  the precision they are recorded at;
- the open-loop `truth.q_i2b` step measured from the same log is
  **90.004996 degrees between t = 1.90 s and t = 2.00 s**, reproducing
  `docs/KNOWN_ISSUES.md` exactly.

### 2.3 The documented figures are the EC-6 reduction, and one of them is stale

The documented pair is unambiguously the **EC-6, final-truth-record**
reduction: the open-loop 181 x 3444 km matches 181.0318 x 3444.1456 exactly,
while EC-11 would have given 3416.1 km. Under that method:

- **open-loop 181 x 3444 km — confirmed.**
- **closed-loop perigee 180.7 km — confirmed** (180.7025).
- **closed-loop apogee 3356.1 km — wrong. The measured value is 3358.0 km.**

The 1.9 km is **not** a reduction-point artifact. The previous agent declined
to edit the number in case its reduction point differed from the documented
method; it did not. Applying the *other* documented method moves the closed-loop
apogee to 3338.1 km, which is 18 km further away, so no reduction defined in
this repository yields 3356.1.

### 2.4 It is a genuine trajectory change, and it is attributable

The ascent really did move. The evidence chain:

1. `502cc4e` (2026-07-19) introduced the figure, adding the line
   "orbit insertion (180.7 x 3356.1 km, against the open-loop 181 x 3444 km)"
   to `docs/KNOWN_ISSUES.md`.
2. `8f09032` ("Resolve the pitch program's roll reference through vertical")
   lands after `502cc4e` and before `376e068` — it is in
   `git log 502cc4e..376e068` — and it changes `cpp/src/gnc/builtin.cpp`,
   which is the pitch-program guidance the closed-loop mission flies.
3. `4f056dd` then rewrote the mission header, and its own diff records the
   before and after: the removed text says the commanded attitude "steps
   89.922 degrees between two consecutive 0.1 s cycles", and the added text
   says "the largest single-cycle change anywhere in this run is 0.100
   degrees".
4. Measured on the cached closed-loop log at `376e068`, the maximum
   single-cycle change in `gnc.cmd.q_cmd_i2b` is **0.100000 degrees**,
   confirming the log reduced above is post-fix.

So the commanded signal the closed loop tracks changed between the commit that
wrote 3356.1 and the commit under review, and the flown trajectory changed with
it. `4f056dd` relocated the `KNOWN-ISSUE-P6-3` prose and carried the
pre-fix apogee forward unchanged. The open-loop mission is unaffected because
it sets attitude kinematically from mission-sequence actions rather than
through the guidance component, which the unchanged 3444 km confirms.

**Nobody noticed a 1.9 km apogee change** because nothing measures it: no test,
script, or fixture computes the closed-loop insertion elements, and
`scripts/perf_gate.py` gates the closed-loop mission on throughput only.

### 2.5 The correction

`docs/KNOWN_ISSUES.md:234` should read `180.7 x 3358.0 km`, and
`missions/ascent_leo_gnc.toml:181-182` should read 3,358 km against the
open-loop 3,444 km, a change of -2.5 % rather than -2.6 %. Both are the EC-6
reduction, and saying which reduction produced them is worth the clause: the
same run reduced at the exact perigee is 3338.1 km, and a reader who assumes
the other method will conclude the figure is wrong by 20 km.

The durable fix for the class of defect is an assertion. A figure quoted in
prose and computed by nothing goes stale silently, which is what happened here;
the closed-loop insertion elements would be cheap to gate in
`tests/python/test_gnc_missions.py` using the EC-6 helper that already exists
in `tests/python/test_vehicle_missions.py:41-51`.
