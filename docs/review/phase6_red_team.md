# Phase 6 red-team — exit-criteria adversarial review

Branch `phase-6-gnc-sensors` at `ebab433`. Read-only adversarial review of the
ten Phase 6 exit criteria (PRD.md line 211) before phase close. Three peer
agents attacked the criteria; each non-SOLID finding was handed to an
independent sceptic instructed to refute it. This document is the synthesis:
it records the surviving verdicts, the confirmed defects ranked by consequence,
the refuted findings, the experiments that still need a compiler, and an honest
account of what this red-team did and did not establish.

**Constraints honored.** Nothing was compiled. No repository file other than
this report was written, and no state-changing git command was run. The
verification below is source reading, `grep`, `git` history reads, and the
peers' measurements (each cross-checked by an independent sceptic); this
synthesis agent ran no `pytest` or `star verify` of its own, to avoid loading a
shared build host, because every load-bearing check here is documentary or
structural and resolves by read alone.

**Input provenance check.** Every gate name, test name, source symbol, PRD line,
and git commit cited by the peers was re-resolved against the tree before being
repeated here. All resolved. The load-bearing chains were re-derived
independently (see Section 5). Where a peer number was corrected by its sceptic,
the corrected figure is used and the correction is recorded.

---

## 1. Per-criterion verdict table

| # | Subject | Final verdict | Attack depth | Residual / defect (one line) |
|---|---------|---------------|--------------|------------------------------|
| 1 | IMU Allan recovery; star-tracker chi-square 1,000 draws; per-sensor bit-identity | **SOLID (bounded residuals)** | Attacked — Python port of the doctest arithmetic, 200-seed power sweep, three mutation topologies; clause C tests run; clause B reasoned + grep | `b_hat` ±10% check has a 4.0% false-failure rate on a pinned seed (flakiness, not a gap); `tests/refs/allan.py` never pointed at core output; PRD.md:222 status bullet stale |
| 2 | Python PD law reproduces C++ torques < 1e-9 N·m | **SOLID** | Attacked — two mutation batteries executed (golden half + in-loop half, 8 and 4 mutations) | None disqualifying; the pre-remediation degenerate fixture (`w_cmd≡0`) has been replaced |
| 3 | Ensemble NEES/NIS 95% gates over R=100; rerun bit-identical | **SOLID (documented limitation)** | Attacked — full 100-run ensemble, epoch-split attack at three amplitudes, constant + bias-block mis-scale sweeps, five block marginals | NEES headline has a disclosed epoch-structure null direction, spec-registered in `ekf.tex`; needs a checklist owner (windowed statistic) |
| 4 | Stepped vs batch identical hashes; `observe()` purity | **SOLID** | Reasoned + live `star verify --quick` (fixture audited parameter-by-parameter; no in-memory divergence perturbation) | Optional hardening: no stepped-vs-batch hash check at `latency_cycles > 0` or with a `python:` component |
| 5 | Schema major unchanged; `oracle: true` identifiable from header alone | **SOLID** | Attacked — live byte-identity of both runs, both-polarity truth-hunter on the installed binary, prologue-version mutation | Coverage-locus note: V023 alone would not catch deletion of the injection block; caught by pytest + the C++ doctest |
| 6 | Nav-fix + altimeter chi-square inside 95% bounds over 1,000 draws | **MODERATE** | Attacked — bit-exact offline RNG replication (6/6 + 3/3 goldens), live mutation battery, independent geodetic-vs-spherical computation | Literal "1,000 draws" met only by the C++ gate whose spherical fixture never exercises the geodetic model; the geodetic model runs only at M=300; ≤0.2 s altimeter timing blind spot; untested `!bodyfixed_valid` fallback (unreachable for a real Earth altimeter) |
| 7 | Camera pose **and intrinsics** bit-exact to `truth`; landmark pixels < 1e-6 px | **WEAK (confirmed)** | Attacked (projection gate, 8 mutations, live core) + documentary defect verified by source read and git history | Criterion text is unsatisfiable as written (no `truth` channel is an intrinsic); the property is silently proxied by the v1.3 camera echo without amending the criterion; PRD.md:221 is factually wrong in both directions |
| 8 | `latency_cycles = k` shifts logged command application by exactly k | **SOLID** | Attacked — live runs at k = 0,1,2,3,5; FIFO and application-site source read | One degenerate assertion (`q_cmd_i2b` constant over the fixture) masks nothing reachable; recommend dropping or re-anchoring it |
| 9 | Optical directions carry velocity aberration, < 1 mas vs independent computation | **SOLID (confined sub-gate defect)** | Attacked — four reference mutations against the real DE440 excerpt; per-segment velocity measured in the V026 fixture | V026's synthesized ephemeris zeroes the Earth-about-EMB velocity, so `star verify` alone is blind to a dropped barycentric term worth 5.5 mas; the pytest gate uses the real ephemeris and catches it; the star-tracker `eq:optical:rho` path is gated only by chi-square, never at mas level |
| 10 | FR-32 ascent ≥ 100× real time with the C++ GNC stack, on `ascent_leo_gnc.toml` | **SOLID (hardening recommended)** | Attacked — substitution attack, `compare_metric` on the rolling gate, live probe of a stripped-`[gnc]` mission | No committed test pins perf_gate's `--ascent-gnc` default, so a deliberate repoint of that default would silently violate the criterion |

**Verdict tally:** SOLID or SOLID-with-bounded-residuals — 8 (criteria 1, 2, 3, 4,
5, 8, 9, 10); MODERATE — 1 (criterion 6); WEAK — 1 (criterion 7).

**Attacked vs reasoned.** Genuinely attacked with measured perturbations: 1, 2,
3, 5, 6, 8, 9, 10 (and criterion 7's projection/pose gates). Structurally
reasoned with a live acceptance run but no perturbation: criterion 4. Criterion
7's decisive finding is documentary and was verified by source read and git
history rather than by perturbation. **Nothing was compiled**, so every
C++-mutation-level confirmation is pending (Section 4); each nevertheless has an
independent non-compiled line of evidence.

---

## 2. Confirmed defects, ranked by consequence

### D1 — Criterion 7: the criterion text is unsatisfiable as written and the PRD status block is factually wrong (documentary; highest consequence)

This is the one finding that survived its sceptic as a defect rather than being
downgraded, and it is the item that gets signed at phase close.

**Verified independently by this synthesis.**

- The criterion (PRD.md:211) reads: "camera-hook pose **and intrinsics** equal
  the `truth` channels bit-exactly, and landmark pixel projections match an
  independent NumPy pinhole-projection script to < 1e-6 px." No `truth` channel
  is an intrinsic — `truth` carries `t_s`, `r_m`, `v_mps`, `q_i2b`,
  `w_b_radps`, `mass_kg` — so the intrinsics half of the conjunction has no
  referent. The test suite concedes this in `test_p6_optical_gates.py`
  (docstring: the intrinsics half is "ill-posed") and substitutes a different,
  well-posed property: that the `gnc.camera` header echo reproduces the logged
  pixels. The criterion text was never amended.
- PRD.md:221 states: "Criterion 7's intrinsics clause has no gate, because no
  intrinsics are logged: the camera record carries only `t_s`, `r_m`, `q_i2b`,
  `px_uv`. … **Open.**" Both assertions are false at HEAD:
  - Intrinsics **are** logged. `python/star_reacher/srlog.py:85` `camera_echo()`
    decodes `header['gnc']['camera']` (fx, fy, cx, cy, width, height, `q_b2c`,
    `r_cam_b_m`), and `cpp/src/srlog_writer.cpp:228-259` emits the echo and
    refuses a camera group without it — with the comment "exit criterion 7's
    intrinsics clause exists to close."
  - A gate **exists**. `test_camera_intrinsics_echo_reproduces_logged_pixels`,
    `test_camera_echo_mutation_is_detected`, and
    `test_camera_echo_fixture_is_not_degenerate`
    (`test_p6_optical_gates.py:595, 619, 571`) gate it with a six-way mutation
    battery plus a non-degeneracy guard.
  - The staleness is provable by timeline, not inference. PRD.md:221 predates
    commit `25b0d94` ("Gate the criterion-7 intrinsics clause and the IMU
    stochastic path," 2026-07-21 13:19), which `git merge-base --is-ancestor`
    confirms is an ancestor of HEAD `ebab433`. The PRD asserts "no gate … Open"
    for a clause a later, merged commit is titled for gating.
- The independent evidence audit already rates this WEAK
  (`docs/audit/phase6_evidence_audit.md:43`: "WEAK; intrinsics clause
  UNCOVERED").

**What is *not* wrong.** The projection and pose gates are strong, not weak. The
peer's live 8-mutation battery on `tests/refs/pinhole.project_landmark` against
the logged pixels rejected every real mutation (reverse `q_b2c` composition
`inf` px; swap fx↔fy 1.188e+02; transpose `C_i2b` 6.655e+01; substitute the
image centre for (cx,cy) 2.550e+01; drop aberration 7.455e-02; drop `r_cam_b`
7.903e-04 px) against a 1e-6 px gate, and the weakest real mutation still clears
the gate by 790×. So PRD.md:213's justification for withholding criterion 7 from
`star verify` — that its gates are "weaker than their wording" — is contradicted
by measurement for the projection and pose clauses.

**Consequence.** Documentary defect in the signed artifact. No code or compile
is required to fix it. This is the primary blocker: a phase cannot close on a
criterion whose text cannot be satisfied and whose recorded status contradicts
the shipped code.

**Remedy.**
1. Amend criterion 7 to the property actually held and gated: "camera-hook pose
   channels equal the `truth` channels bit-exactly, and the logged camera
   intrinsics/extrinsics echo reproduces the logged landmark pixels, and those
   pixels match an independent NumPy pinhole-projection script, to < 1e-6 px."
2. Strike the "Criterion 7's intrinsics clause has no gate … Open" paragraph
   (PRD.md:221) and record the SRLOG v1.3 echo as the closure.
3. Reconsider the `star verify` exclusion, or state the real reason (the mission
   needs the repo ephemeris and vehicle tree that a bare wheel lacks) rather
   than a weakness claim measurement contradicts.

### D2 — Criterion 6: the literal "1,000 draws" clause is met only by a gate that does not exercise the altitude model (MODERATE)

**Verified structurally.** The only 1,000-draw gate is
`cpp/tests/test_sensors.cpp:603` `sensors_navfix_altimeter_chi_square_over_1000_draws`.
Its altimeter fixture sets `ellipsoid_inv_f = 0.0` and
`c_gcrf_to_bodyfixed = Identity`, so `Altimeter::sample` takes the two-line
spherical branch (`|r| − a`) and never executes the Bowring geodetic path or the
body-fixed rotation. `inv_f = 0.0` appears at exactly the two altimeter fixtures
in the file. The only gate exercising the real geodetic model is the pytest
altimeter gate (`test_altimeter_statistic_passes_the_reference_gate`), and it
runs at M=300 (60 s × 5 Hz), not 1,000.

**Measured (peer, reproduced offline by the sceptic to the millimetre).** At the
1,000-draw fixture point (7e6, 1e6, −2e6) the WGS84 geodetic altitude is
971,922.374 m against 970,332.228 m spherical — a 1,590.146 m ≈ 636σ difference
the C++ gate is structurally unable to see. But the altimeter error statistic is
model-invariant (`((h_meas − h_true)/σ)²` with the test subtracting the sensor's
own `h_true`), so the C++ gate's 1,000-draw pass is a valid noise-consistency
result identical to what a geodetic 1,000-draw fixture would produce, and the
headline radius-vs-geodetic risk is caught by the pytest gate at 636σ (a
spherical implementation is rejected live at 2128.84 vs upper bound 1.1662).
Two narrower gaps: an altimeter-only truth misalignment ≤ 0.2 s is invisible
(the channel varies only ~0.31σ/cycle; the co-located nav-fix gate still rejects
gross misalignment at ~14878σ), and the `!bodyfixed_valid` spherical fallback is
untested — though `environment.cpp` sets `bodyfixed_valid = true` for every
non-Sun central body, so it is unreachable for a real Earth altimeter.

**Consequence.** The nav-fix half is fully closed at 1,000 draws; the altimeter
half's dominant risk is caught, but the literal wording ("over 1,000 seeded
draws") is not met on the path that exercises the altitude model. Bounded
materiality — ~0.05 m of resolution on a purely systematic altitude bug — but a
real shortfall against the criterion as written.

**Remedy (config only, no compile).** Raise the pytest statistics mission's
altimeter/nav-fix draw count to ≥ 1,000 (longer duration or higher sample rate)
so the model-exercising gate meets the criterion's own number, and/or add a
geodetic, non-equatorial fixture. Optionally add an eccentric altimeter fixture
so a one-cycle timing shift moves the statistic past its bound.

### D3 — Criterion 9: V026's fixture multiplies one aberration term by zero (confined sub-gate defect; criterion closed by the gate union)

**Verified directly.** In `verify.py:_v026_ephemeris`, the `earth` segment
(Earth about EMB) is built with `const_record(4671.0, 0.0, 0.0)` — a zero linear
Chebyshev coefficient, hence zero velocity — while the `emb` segment carries the
nonzero linear term `c1`. So the second term of `eq:optical:beta`'s barycentric
velocity composition is multiplied by zero in the `star verify` wiring. The peer
measured this term at 13.068431716 m/s in the committed DE440 excerpt and worth
5.5054 mas — 5.5× the criterion's own 1 mas requirement — and V026 alone cannot
see it dropped. The pytest gate
(`test_aberration_matches_independent_reference`) uses the real ephemeris and
resolves the dropped term at 5.5054 mas, going red. Both gates run in CI, so the
criterion is closed by their union; V026 in isolation is not.

The rest of criterion 9 is genuinely strong: baseline worst residual 2.43e-08
mas against a 1e-5 mas gate; the four reference mutations reject at 0.470 /
4.03e+07 / 5.51 / 2186 mas; fixture non-degeneracy holds (|q_w| reaches 0.060 so
`C − C^T` does not vanish; θ ≈ 76°). Note the tightened 1e-5 gate, not the
criterion's own 1 mas wording, is what closes the drop-transverse mutation
(0.4696 mas would pass at 1 mas).

**Scope gap.** Three code paths apply aberration. The sun sensor and camera
bearings go through `sensors::aberrate()` and are covered (the camera
transitively via criterion 7's pinhole reference). The star tracker uses the
structurally different rigid field rotation `ρ = b_I × β` (`eq:optical:rho`,
`cpp/src/sensors/optical.cpp`), which is gated only by a chi-square statistic,
never at mas level against an independent computation.

**Consequence.** Low — both aberration gates are in CI and the criterion is met
— but the fixture degeneracy is exactly the checklist pattern (a term multiplied
to zero) and should be closed cheaply.

**Remedy.** Give V026's `emb` and `earth` segments independent nonzero linear
Chebyshev terms (a two-value edit; the fixture builder already takes per-segment
coefficients) so the composition's second term carries real signal — the 5.5 mas
signal is 5.5e5× the 1e-5 mas gate. Optionally add a direct mas-level comparison
of the logged star-tracker quaternion on the noise-free optical fixture to put
`eq:optical:rho` under the same instrument.

### D4 — Criterion 1: `b_hat` flakiness, an orphaned reference, and a stale PRD bullet (bounded residuals)

The peer's WEAK verdict was refuted (Section 3): its two load-bearing
consequence claims are false, and the sceptic reproduced the corrected numbers.
What survives are three bounded residuals.

- **`b_hat` ±10% check is fragile on a pinned seed.** Measured `b_hat/B − 1`
  scatter is 4.87% std over 200 seeds, with 8/200 = 4.0% outside the ±10% gate.
  On the single pinned doctest seed this is a false-failure/flakiness hazard
  under any last-bit RNG-stream perturbation (a compiler or platform change),
  not a detection gap — the check catches a real ×1.15 conversion defect at
  81% power and a ×1.30 at 100%, because the conversion is fed to the generator
  and `b_hat` tracks the defect 1:1. The test's own comment derives the
  trade-off.
- **`tests/refs/allan.py` is orphaned from the core.** The independently
  validated Allan estimator is never pointed at core IMU output anywhere in the
  repository (only `test_refs_allan.py` imports it, and that file has no core
  import); the gate's `oadev` is a test-local C++ reimplementation with no
  cross-check. This is partly self-protecting — a transcription error in the C++
  `oadev` would shift `n_hat`/`b_hat` and fire the ±10% checks — but the
  cross-check the audit recommended does not exist.
- **PRD.md:222 is stale.** "Criterion 1's bit-identity clause exercises only a
  noise-free IMU, so the stochastic sensor path is uncovered by it. **Open.**"
  Clause C was remediated (the stochastic IMU rerun tests pass) by the same
  commit `25b0d94` whose title names "the IMU stochastic path." Same documentary
  family as D1, milder.

**Remedy.** Strike the stale PRD.md:222 bullet and restate clause A (the Allan
recovery) as the residual half. Reduce the `b_hat` flakiness by averaging the
recovery over the three instrument axes (≈√3 scatter reduction to ~2.8% std,
sub-1% false-failure) or over a seed ensemble. Add the cross-check of the C++
`oadev` against `tests/refs/allan.py` on the same core output. None is
phase-blocking.

### D5 — Criterion 3: a disclosed, spec-registered NEES null direction needs a backlog owner (low)

The peer's WEAK verdict was refuted (Section 3). The residual is that the NEES
headline averages over 601 epochs, so a covariance error whose sign reverses
across the run cancels — a real null direction, but a **disclosed** one.
`docs/mathlib/chapters/ekf.tex:641` states the consequence verbatim ("a
sustained NEES defect that leaves the epoch average inside … does not set the
exit code"), records two measured instances under a heading that says they are
recorded "rather than left to be rediscovered," rejects the peer's proposed
coverage-gate remedy with measurement, and names an out-of-scope successor
(`ekf.tex:747`, "gate on a windowed statistic over a segment short enough to
resolve the transient"). This is the opposite of a gate blind to its target.

**Remedy.** Register the windowed-statistic remedy on the pre-release checklist
alongside KNOWN-ISSUE-P6-5, so the disclosed blind spot has an owner rather than
only a paragraph. Not phase-blocking.

### D6 — Criterion 5, criterion 8, criterion 10, criterion 4: coverage-locus and hardening notes (low to cosmetic)

- **Criterion 5 (coverage-locus).** V023 reads the oracle flag from the header
  in both polarities, but the fuller FR-25 conjunction (truth crossed the
  boundary ⇒ the run is marked) is gated by
  `test_gnc_python_component.py:947` (601/601 vs 0/601 injected-truth cycles,
  header agreeing both ways) and the C++ doctest `gnc_cycle_oracle_gating`, not
  by V023. V023 alone would not catch deletion of the injection block. Record
  the locus in the evidence audit; not an open criterion.
- **Criterion 8 (degenerate assertion).** `q_cmd_i2b` is constant over the
  latency fixture (checklist item 1), so that half of the log-level assertion
  has zero discrimination. It masks nothing reachable — `q_cmd` and torque are
  one `GncOutput` pushed through one FIFO and logged from the same `applied`
  object, and the shift is enforced by the discriminating `first_valid == k`,
  `t_s` offset, valid pattern, and bit-identical torque. Recommend dropping or
  re-anchoring the degenerate assertion, or adding the peer's full-history
  log-level gate on the trajectory-independent ascent mission.
- **Criterion 10 (hardening gap).** No committed test pins perf_gate's
  `--ascent-gnc` default (`scripts/perf_gate.py:574`, correctly
  `ascent_leo_gnc.toml`); the only test touching that argument overrides it. A
  deliberate repoint of the default would silently change what CI measures. The
  criterion is measured correctly today (nightly job uses the default). Close
  with a one-line assertion that the default equals
  `missions/ascent_leo_gnc.toml`, or that the measured GNC log carries a
  `gnc.cmd` group.
- **Criterion 4 (optional hardening).** No stepped-vs-batch hash comparison
  exists at `latency_cycles > 0` or with a `python:` component in the loop; both
  V022 and the tests run `latency_cycles = 0` with built-in C++ components. The
  known defect class here is config-construction divergence, and `latency_cycles`
  is the remaining cross-cycle config field whose byte equality is unproven. Add
  a second V022 parametrization at `latency_cycles = 2`. Not required to close
  the criterion as written.

---

## 3. Refuted findings — real results that belong in the record

Six peer WEAK findings were handed to independent sceptics and did not survive
as WEAK. Each is recorded here because a refuted finding is a result: it tells
the phase where a gate was suspected and why the suspicion was retired.

**Criterion 1 — peer WEAK → refuted (SOLID with bounded residuals).** The peer's
two consequence claims are wrong. (a) "It passes a 15% bias-instability model
error": the sceptic ported the generation model and measured that with the
`gm_sigma` scale defect present in both data and model — the only reachable
topology, since `imu.cpp` feeds the generator from the same conversion call —
`b_hat` tracks the defect 1:1 (+14.18% at ×1.15, gate fires 162/200 = 81%;
+29.19% at ×1.30, 200/200). The peer's +8.54% is a single-seed artifact of a
surrogate RNG stream. (b) The octave-overlay cancellation is real but the
alleged defect is unreachable: `test_sensors.cpp:275`
`sensors_imu_gauss_markov_recursion_and_preset_map` gates the identical
conversion function directly at 1e-4 relative, an untagged `TEST_CASE` in the
same doctest target — a ×1.15 error is 1500× over that tolerance. The peer
looked at neither net. Surviving residuals become D4 above.

**Criterion 3 — peer WEAK → refuted (SOLID with documented limitation).** Every
headline number reproduced, but the finding fails on three counts: the null
direction is documented and measured in `ekf.tex` (not discovered); the peer's
proposed coverage remedy is already analysed and rejected there, and would not
catch the mission-reachable instance (coverage 574/601, above any workable
threshold); and the reachability claim is false — the attack scales P post-hoc
while holding the error trajectory fixed, which no filter defect does, and a
Q-only mis-scale is architecturally excluded because `ekf.cpp` assembles Q from
the same `[sensors.imu]` block that drives truth. The sceptic also caught a
numeric transposition in the peer's bias-block claim ("f = 1.15 green at 13.7138
… f = 1.25 reddens" — measured, 13.7138 is the f = 1.25 headline and is red;
last green is f = 1.15 at 14.1186). Surviving item becomes D5.

**Criterion 5 — peer WEAK → refuted (SOLID).** Measurements reproduced (record
streams byte-identical, headers differ only in `oracle` and `config_sha256`),
but the alleged defect falls: the two halves the peer said are "never conjoined
in one run" are conjoined in `test_gnc_python_component.py:947` (verified live
at both polarities, 601/601 vs 0/601), the peer's grep was scoped to
`cpp/src/gnc/` while the injection is in `cpp/src/vehicle_cycle.cpp:1060`, and
the version is a hard literal in two places so the fixture's channel coverage is
immaterial. Surviving item becomes the D6 coverage-locus note.

**Criterion 6 — peer WEAK → downgraded, not refuted (MODERATE).** All structural
facts reproduced, but the severity overstates the consequence: the nav-fix half
is fully gated at 1,000 draws, the altimeter statistic is model-invariant so the
1,000-draw pass is valid, and the headline altitude-model risk is caught at
636σ by the pytest gate. This is the one refuted finding that lands above SOLID:
the literal-clause shortfall is real (D2).

**Criterion 8 — peer WEAK → refuted (SOLID).** Facts reproduced (`q_cmd_i2b`
constant, no live defect), but the degenerate assertion masks nothing reachable
because `q` and torque are one object through one FIFO, and the shift is enforced
by multiple discriminating assertions plus a verbatim config→FIFO pass-through
and k = 0..4 unit coverage. Surviving item becomes the D6 cleanup note.

**Criterion 10 — peer WEAK → refuted (SOLID with hardening).** The "no gate"
framing is wrong — the nightly job measures the criterion on the named mission
with the correct default — and the "lost `[gnc]` table leaves every gate green"
scenario is empirically false: a stripped-`[gnc]` `ascent_leo_gnc.toml` does not
reach insertion, so `test_closed_loop_ascent_insertion_elements` goes red. The
only genuine residual is the unpinned default (D6).

---

## 4. Experiments that still need a compiler

Nothing in this review was compiled. Each item below has an independent
non-compiled line of evidence, so none is load-bearing for the verdicts; they
are the confirmatory builds to run when a build host is next free. Per the
project's serialize-native-builds rule, run them one at a time.

1. **Criterion 1 — Allan overlay inertness and `b_hat` seed fragility.** Rebuild
   with `star::sensors::gm_sigma_from_bias_instability` scaled by 1.30 and run
   `sensors_imu_allan_recovers_arw_and_bias_instability`: predict the
   octave-overlay CHECK stays green at every cluster time while the `b_hat` CHECK
   fires, and that `sensors_imu_gauss_markov_recursion_and_preset_map`
   (`test_sensors.cpp:283`) also reddens (1500× over its 1e-4 tolerance). Repeat
   at ×1.15. Separately, reseed the doctest IMU to ten other values on an
   unmutated core: predict ≈ 1-in-25 seeds fails the `b_hat` ±10% assertion.
2. **Criterion 3 — faithful Q-only mis-scale.** Multiply the `ng2`/`na2`
   coefficients in the Q assembly (`cpp/src/gnc/ekf.cpp:331-345`) by a factor g,
   rebuild, rerun the ensemble: predict the headline shifts monotonically and
   reddens for |g − 1| beyond ≈ 7%, with no sign reversal across the run.
3. **Criterion 5 — injection-block deletion.** Delete the `if (cfg.oracle)`
   block at `cpp/src/vehicle_cycle.cpp:1060-1069`, rebuild, and confirm
   `gnc_cycle_oracle_gating` fails (oracle_valid_cycles 21 → 0) while V023 stays
   green — the demonstration that V023 alone does not carry the FR-25
   conjunction.
4. **Criterion 6 — geodetic branch in the 1,000-draw gate.** Set
   `ellipsoid_inv_f = WGS84_INV_F` and a non-equatorial `r_true` in
   `sensors_navfix_altimeter_chi_square_over_1000_draws`, computing `h_true`
   through an independent Bowring conversion in the test: predict the gate stays
   green while now covering the branch, and that reverting `Altimeter::sample`'s
   ellipsoid arm to `norm(r) − a` moves `qh/M` from ~1.06 to ~4e5. Separately,
   mutate `models::geodetic_altitude` by a small factor and rerun both the C++
   1,000-draw gate and the pytest altimeter gate: predict the C++ gate does not
   move at all.
5. **Criterion 8 — FIFO depth off-by-one.** Build with the FIFO depth forced to
   k − 1 and confirm the single-point log gate reddens (confirmatory; the live
   k-sweep already shows the valid-flag index assertion would catch it).

Not needed for any verdict: criterion 2's direct C++ golden
(`gnc_pd_attitude_golden`) — V025 already renders it redundant as evidence — and
criterion 9's `eq:optical:rho` mas gate, which is a test addition rather than a
build.

---

## 5. What this red-team did and did not establish

**Established (verified by this synthesis, read-only).**

- Every gate name, test name, source symbol, PRD line, and git commit cited by
  the peers resolves against the tree at `ebab433`. No broken citation was
  found. (The peers' own cross-checks corrected three internal numeric slips —
  criterion 1's +8.54%, criterion 3's bias-block transposition, criterion 6's
  "M=301" typo — all recorded above.)
- The criterion-7 documentary defect (D1) is real and independently confirmed:
  the criterion text is unsatisfiable as written, the property is silently
  proxied by the v1.3 camera echo, and PRD.md:221's "no intrinsics logged / no
  gate / Open" is contradicted by `srlog.py:85`, `srlog_writer.cpp:228-259`, and
  the merged commit `25b0d94`.
- The criterion-9 sub-gate defect (D3) is real: `_v026_ephemeris` zeroes the
  Earth-about-EMB velocity while the pytest gate uses the real ephemeris, so the
  criterion is closed by the gate union but not by `star verify` alone.
- The criterion-1 stale PRD bullet (D4) is real: PRD.md:222 says "Open" for a
  clause `25b0d94` closed.
- Criterion 3's NEES limitation (D5) is a disclosed, spec-registered property of
  the statistic (`ekf.tex:641, 747`), not an undiscovered blind spot.
- Criterion 6's 1,000-draw / geodetic-model split (D2) is a structural fact of
  the fixtures (`inv_f = 0.0` in the only 1,000-draw gate; the geodetic model
  only in the M=300 pytest gate).

**Not established.**

- **No compiled behaviour was verified.** All C++-mutation-level confirmations
  (Section 4) are pending. The peers' perturbation magnitudes were measured
  against Python references, installed binaries built earlier, or offline RNG
  ports — not against freshly mutated cores. Where a claim depends on the
  compiled core, it rests on an independent non-compiled proxy (a sibling gate,
  a gate union, or a live run of the existing binary), which is weaker than a
  rebuild.
- **This synthesis agent ran no `pytest` or `star verify`.** It relied on the
  three peers and their independent sceptics for every executed measurement, and
  on its own source/`git` reads for the documentary chains. The peer and sceptic
  measurements agree with each other where they overlap, which is the basis for
  trusting them, but this document adds no new execution.
- **Criterion 4 was reasoned, not perturbed.** Its SOLID verdict rests on a
  fixture audit, a source trace of `run_vehicle`, and one `star verify --quick`
  pass — no in-memory divergence was injected into the multi-sensor ordering.
  The verdict is sound on the structure (three independent routes carry sensor
  order into the compared bytes) but is the least adversarially exercised of the
  ten.
- **Criteria resting on evidence nobody attacked at depth:** criterion 4 (above);
  and the `star verify`-excluded criteria 1 and 7, whose acceptance-suite
  behaviour is asserted by the PRD but, by design, not exercised by the suite —
  their gates live only in `pytest` and the C++ doctests.

---

## 6. Judgement: should Phase 6 close?

**Not as-is — but the only true blocker is documentation, clearable in one edit
pass without a compiler.**

Nine of ten criteria are substantively closed. Eight are SOLID or SOLID with
bounded residuals; criterion 6 is MODERATE (the dominant altitude-model risk is
caught at 636σ, but the literal "1,000 draws" clause is unmet on the
model-exercising path). The lone WEAK, criterion 7, is not a numeric gate hole —
its projection, pose, and echo gates are strong — but a defect in the signed
artifact: a criterion whose text cannot be satisfied and a status bullet that
contradicts the shipped code. Signing a phase on that text would certify a
property that has no referent.

**Mandatory before close (no compile required):**

1. **Amend criterion 7's text** to the pose + echo + independent-pinhole
   property that is actually gated (D1, remedy 1), and **strike PRD.md:221's "no
   gate … Open" paragraph**, recording the SRLOG v1.3 echo as the closure
   (D1, remedy 2).
2. **Correct PRD.md:222** (criterion 1): clause C (stochastic bit-identity) is
   closed by `25b0d94`; restate clause A (Allan recovery) as the open residual
   (D4).
3. **Resolve criterion 6's literal clause (D2):** either raise the pytest
   statistics mission's altimeter/nav-fix draws to ≥ 1,000 (a config edit, no
   compile) so the model-exercising gate meets the criterion's own number, or
   record an explicit, signed rationale that the C++ 1,000-draw noise gate plus
   the M=300 geodetic gate jointly satisfy the intent. The former is preferred:
   it makes the wording true rather than reinterpreted.
4. **Register the disclosed residuals that currently have no owner** on the
   pre-release checklist / KNOWN_ISSUES, per the project's
   register-deferred-evidence discipline: criterion 3's windowed statistic
   (alongside P6-5), criterion 9's V026 fixture velocity and the `eq:optical:rho`
   mas gate, criterion 1's `allan.py` orphan and `b_hat` flakiness, and
   criterion 10's `--ascent-gnc` default pin.

**Recommended, not blocking:** close criterion 9's V026 fixture degeneracy (the
two-value edit, D3), pin criterion 10's default (D6), drop criterion 8's
degenerate assertion (D6), and add criterion 4's `latency_cycles = 2` V022
parametrization (D6). Run the Section 4 confirmatory builds when a host is free;
each is confirmatory, not gating.

With the four mandatory documentation items landed, Phase 6's exit criteria are
met by gates demonstrated able to fail — and the phase closes honestly, with its
disclosed blind spots registered rather than laundered.
