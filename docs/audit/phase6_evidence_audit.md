# Phase 6 exit-criteria evidence audit

An adversarial audit of the evidence chain behind the ten Phase 6 exit
criteria in [`PRD.md`](../../PRD.md). For each criterion the audit
establishes which committed test gates it, whether that test asserts what
the criterion requires, whether the gate can go red, and whether every
clause of a multi-clause criterion is covered.

The method was to attack gates rather than read them: perturb a reference,
corrupt logged data, mutate a formula, and measure whether the gate
responds. Numbers quoted below are measured on the installed 0.5.0 core in
this worktree unless marked otherwise. No native build was performed, so
gates whose sensitivity can only be shown by rebuilding the core are
marked as such with the experiment specified.

> **Status: this is a dated findings record, not a current description of
> the gates.** Criteria 2, 3, and 9 have since been remediated along the
> lines this audit recommended, so the measurements below describe the
> fixtures as they stood at audit time and no longer match the committed
> tests. The figures are deliberately left as measured rather than
> restated, because the record of what the audit found is the point of the
> document. For the current state see the criterion-2 and criterion-3
> notes in `ch:gnc-builtin` and `ch:ekf`, the criterion-9 tolerance note in
> `ch:sensors-optical`, and `docs/KNOWN_ISSUES.md`. The most consequential
> divergences: criterion 9's gate is now `ABERRATION_TOL_MAS = 1e-5` on an
> off-axis fixture that rotates the reference side through
> `tests/refs/quaternions.quat_to_dcm`, against which the measured worst
> residual is 4.726e-08 mas rather than the 2.576e-08 mas recorded here;
> criterion 2 now runs a non-degenerate scenario gated on both the mpmath
> goldens and the in-loop torques; and criterion 3 now routes both halves
> through `consistency.ensemble_gate`.

## Verdicts

| # | Criterion (abbreviated) | Verdict |
|---|---|---|
| 1 | Allan ±10 %; star-tracker chi-square over 1,000 draws; every sensor bit-identical across two seeded runs | WEAK |
| 2 | Python PD controller reproduces the C++ torques to < 1e-9 N·m on a golden scenario | WEAK (vacuous for three of the five equations it claims) |
| 3 | Reference EKF passes ensemble NEES and NIS 95 % gates over 100 seeded runs; rerun bit-identical | WEAK |
| 4 | Stepped and batch runs produce identical log hashes; `observe()` twice returns identical dictionaries | SOLID |
| 5 | Schema version major unchanged; an `oracle: true` run identifiable from its header alone | SOLID |
| 6 | External-nav-fix and altimeter statistics inside two-sided 95 % bounds over 1,000 seeded draws | SOLID |
| 7 | Camera pose **and intrinsics** bit-exact to `truth`; landmark pixels < 1e-6 px | WEAK; intrinsics clause UNCOVERED |
| 8 | `latency_cycles = k` shifts logged command application by exactly k cycles | WEAK |
| 9 | Optical truth directions carry velocity aberration, matching an independent computation to < 1 mas | WEAK |
| 10 | FR-32 ascent target (≥ 100× real time, Pi 5) holds with the C++ GNC stack in the loop | UNCOVERED |

Four criteria (1, 2, 7, 9) are weakened by fixture geometry that makes part
of the check degenerate — the same failure shape as the two precedents this
phase already found and fixed.

## Criterion 1 — sensor statistics and bit-identity

**Clause A (Allan, ±10 %).** Gated by `sensors_imu_allan_recovers_arw_and_bias_instability`
(`cpp/tests/test_sensors.cpp:393`) on core-produced IMU samples, asserting
`std::fabs(n_hat / c.n_coeff - 1.0) <= 0.10` (`:481`) and the same form for
the bias-instability coefficient (`:493`). The data is genuinely the
shipped IMU's output, and the configured coefficients are inputs rather
than expected outputs, so there is no circularity.

Two gaps. The Allan estimator used by the gate (`oadev`, `:369`) is a
test-local reimplementation; the independently validated Python estimator
in `tests/refs/allan.py` is never pointed at core IMU output anywhere in
the repository, so the two estimators are never cross-checked on the same
data. And the gate preset disables the quantizer, so the `sigma_Q` term of
`eq:imu:recovery` is never exercised — acknowledged at
`docs/mathlib/chapters/sensors_imu.tex:664`.

**Clause B (star tracker, 1,000 draws).** Gated by
`sensors_startracker_chi_square_over_1000_draws`
(`cpp/tests/test_sensors.cpp:511`), `CHECK(q_mean >= 2.850)` / `<= 3.154`
(`:569-570`). The bounds are hard-coded literals but are independently
reproduced from first principles at `tests/python/test_refs_chi2.py:105`,
so they are not self-generated. The gate sets
`truth.v_end_i_mps = Eigen::Vector3d::Zero()` (`:525`), which makes the
aberration quaternion the identity — so the composition order that both
optical chapters flag as the primary transcription hazard is not exercised
by the criterion-1 gate itself.

**Clause C (every sensor bit-identical).** Gated by
`test_full_sensor_suite_reruns_bit_identical`
(`tests/python/test_gnc_missions.py:420`), which compares the SRLOG
SHA-256 of two runs (`:430`) and then every channel of all six sensor
groups (`:441-443`). Noise is enabled for five of the six sensors.

The IMU is not. `missions/leo_attitude_gnc.toml` declares only
`sample_rate_hz = 10` under `[sensors.imu]` (verified: no `random_walk`,
`bias_instability`, `bias_tau_s`, or quantizer keys), and the full-suite
variant appends the other sensors without adding IMU noise. The IMU's
stochastic path — angle-random-walk draws, the Gauss-Markov bias
recursion, and the quantizer carry state, which is the only sensor state
that persists across samples — is therefore covered by no bit-identity
assertion at all.

**Defect (documentation, four chapters).** No `.tex` file cites
`test_full_sensor_suite_reruns_bit_identical`. All four Phase 6 sensor
chapters instead cite `\testid{gnc_cycle_double_run_is_byte_identical}`
(`sensors_imu.tex:575`, `sensors_radio.tex:269`, `sensors_optical.tex:460`,
`camera.tex:316`) and claim it covers "the radio sensor channels", "the
optical sensor channels", and the camera channels respectively. That test
is `cpp/tests/test_gnc_cycle.cpp:344`, whose scenario helper configures
exactly one sensor — an ideal IMU (`kind = "imu"`, `sample_rate_hz = 10`,
no other fields). Those three coverage claims are false: the scenario
contains no navfix, no altimeter, no star tracker, no sun sensor, and no
camera.

**Remedy.** Add IMU noise coefficients to the full-suite bit-identity
variant so the stochastic path is covered. Repoint the four chapter
evidence rows at `test_full_sensor_suite_reruns_bit_identical` and
restate the `gnc_cycle_double_run_is_byte_identical` row as whole-file
identity on an ideal-IMU scenario. Add a test that runs
`tests/refs/allan.py` against core IMU output.

## Criterion 2 — Python PD controller versus the C++ controller

Gated by `test_pd_law_python_reimplementation_contract`
(`tests/python/test_gnc_missions.py:175`), asserting
`np.max(np.abs(tau - cmd["tau_b_nm"])) < 1e-9` (`:228`). The Python
transcription is genuinely independent: it imports only `numpy` and
`tomllib` and re-derives the Hamilton product, the sign branch, the error
DCM, and the clamp by hand.

**The fixture is degenerate.** Measured on
`missions/leo_attitude_gnc.toml` over all 601 cycles:

| Quantity | Measured | Consequence |
|---|---|---|
| `max abs(w_cmd_b_radps)` | `0.000e+00` (identically zero) | `eq:gnc:werr` is multiplied by zero |
| `dq0` range | `[0.996195, 0.999998]`, never negative | `eq:gnc:sign` short-path branch never taken |
| Cycles at the saturation clamp | 0 of 601 (0.00 %) | `eq:gnc:sat` never exercised |
| `max abs(tau)` per axis | `[0.0, 0.0, 0.0349]` | x and y gain paths multiplied by zero |

Mutating the reference and re-measuring the worst residual against the
logged torques:

| Mutation of the reference | Worst residual | Gate response |
|---|---|---|
| Drop the short-path sign branch | `0.0000e+00` N·m | **missed** |
| Drop the error-DCM rotation of `w_cmd` | `0.0000e+00` N·m | **missed** |
| Transpose the error DCM | `0.0000e+00` N·m | **missed** |
| Scale `kp` by 1.001 | `3.4862e-05` N·m | detected |

The gate is not vacuous in the absolute sense — a 0.1 % gain error is
caught — but for `eq:gnc:werr`, `eq:gnc:sign`, and `eq:gnc:sat` it is
exactly vacuous: the mutations change the answer by identically zero, so
no tolerance could ever separate them. The actual residual on the
unmutated reference is `0.000000e+00` N·m, so the 1e-9 tolerance is doing
no work either.

**Defect (evidence table).** `docs/mathlib/chapters/gnc_builtin.tex:487`
claims this test exercises "equations~\eqref{eq:gnc:deltaq}--\eqref{eq:gnc:sat}"
— a range that includes `werr`, `sign`, and `sat`, none of which it
exercises.

**Defect (criterion conjunction).** The criterion names a *golden
scenario*. The committed goldens at `tests/golden/gnc/pd_attitude.toml`
have strong provenance — their header records that `expected_tau_nm` is
"the 60-digit mpmath evaluation of the `gnc/builtin.hpp` law", i.e. not
generated by the implementation under test — and they do cover both signs
of `dq_w` and the clamp. But they are consumed only from
`cpp/tests/test_gnc.cpp`; no Python PD implementation is ever evaluated
against them. The criterion's conjunction (Python controller + C++
controller + golden scenario + 1e-9) is never satisfied by a single test.

**Remedy.** Point the Python transcription at
`tests/golden/gnc/pd_attitude.toml`, which already exercises both sign
branches and the clamp, and keep the mission-level comparison as a
supplementary in-loop check. Separately, add a slewing guidance case with
non-zero `w_cmd` and a saturating transient so the mission-level gate
stops being blind to three of the five equations.

## Criterion 3 — ensemble NEES and NIS gates

The driver is `tests/python/test_ekf_consistency.py`. It genuinely
executes 100 runs (`N_RUNS = 100`, `:69`), and
`test_ensemble_rerun_is_bit_identical` (`:280`) executes a second 100 and
compares the ordered SHA-256 list — 200 full 601-cycle EKF missions per
session, with no skip marker, environment gate, or CI deselection.
The bit-identity clause is properly closed.

The headline gates are sound. `test_ensemble_nees_gate_passes` (`:248`)
and `test_ensemble_nis_gate_passes` (`:268`) bound the ensemble average by
two-sided 95 % chi-square quantiles computed from `star_reacher.chi2`.
That quantile evaluator was validated against SciPy across
k ∈ [1, 1e6] and p ∈ [0.005, 0.995]: worst relative error `3.765e-13`,
comfortably inside its documented 1e-10 claim. The bounds themselves are
correct.

**Defect (mis-calibrated statistic).** Alongside the headline bound,
`:261` asserts a coverage rule:

```python
inside = np.mean((epoch_mean >= lower) & (epoch_mean <= upper))
assert inside >= 0.95
```

This is the exact rule the project's own instrument documents as invalid.
`python/star_reacher/consistency.py:90-95` states that "a rule of 'at
least 95 % of epochs inside' therefore tests X against its own mean, and a
correct filter passes it only about half the time — it is a coin flip, not
a gate", and `:129-136` adds that for NEES specifically the count-inside
is over-dispersed.

Measured, 4,000 synthetic ensembles under the exact null (R = 100,
T = 601, dim = 15) with **independent** epochs — the most favourable case,
since real epochs are serially correlated and therefore worse:

| Rule | P(pass) | False-failure rate |
|---|---|---|
| Driver's `inside >= 0.95` | 0.5495 | **45.05 %** |
| Headline (ensemble average in bounds) | 1.0000 | 0.00 % |
| Library's sanctioned threshold (579/601) | 0.0760 | 92.40 % |

The driver's assertion rejects a correct filter 45 % of the time. It is
green today only because the seeds are pinned (`BASE_RUN_SEED = 20260701`,
`DRAW_SEED = 90210`) and the run is bit-deterministic. It is the single
most fragile assertion in the exit-criteria set: any change to the core
that perturbs the last bits — a compiler change, a platform with different
rounding, a legitimate model fix — flips it with probability near one half,
and it will present as an EKF consistency failure when nothing is wrong.
The driver also bypasses the library's `ensemble_gate()` entirely and
re-derives bounds locally, so the sanctioned instrument's coverage logic
never runs on the criterion-3 path.

**Defect (evidence table).** `docs/mathlib/chapters/ekf.tex:869` describes
the test as asserting only that "the ensemble average is inside the
two-sided 95 % bounds"; the coverage assertion is undocumented. And
`ekf.tex:851-853` says the directory CLI form is "the form the
exit-criterion-3 driver will invoke" — the driver computes in-process and
never invokes it.

**Remedy.** Delete the `inside >= 0.95` assertion, or replace it with
`consistency.inside_count_threshold` applied through `ensemble_gate()`
with epochs declared correlated (which is what the library's NEES
configuration does, leaving the verdict resting on the headline). Correct
the two `ekf.tex` statements.

## Criterion 4 — stepped versus batch, and `observe()` idempotence

Both clauses are genuinely tested, not asserted by comment.

`test_stepped_run_hashes_identically_to_batch`
(`tests/python/test_sim.py:99`) actually runs both paths — `_batch_run`
calls `run_mission`, `_stepped_run` drives `Sim.step()` in a loop — and
compares SHA-256 of the two files. The C++ side has the equivalent
byte-vector comparison in `gnc_cycle_batch_wrapper_matches_stepping`
(`cpp/tests/test_gnc_cycle.cpp:367`). The "by construction" comment at
`cpp/src/run.cpp:398-402` sits above the two-line wrapper and is
explanatory; it does not stand in for the test.

`test_observe_is_idempotent_without_step` (`test_sim.py:135`) compares
returned dictionaries by value at four checkpoints, and is reinforced by
`test_observe_returns_copies_not_views` (`:161`), which mutates the
returned dict and re-reads. The C++ comparator `same_observation`
(`test_gnc_cycle.cpp:400`) is explicitly a deep value comparison, with the
rationale recorded that "an idempotence test that compared object identity
would pass trivially".

**Minor gap, not disqualifying.** Neither hash test uses the EKF scenario;
both run `leo_attitude_gnc.toml` with `dead_reckoning`. Step-versus-batch
byte equality is therefore not demonstrated on the Phase 6 headline chain
with its four sensors and aperiodic `nav.innov` writes. Since
`run_vehicle` is a literal loop over the same `VehicleCycle::step()`, the
risk is low, but adding `missions/leo_ekf_consistency.toml` as a second
parametrization would close it cheaply.

## Criterion 5 — schema major and oracle identifiability

Both clauses are covered, each by more than one test, with negative cases.

Schema major: `test_v12_groups_present_and_shaped`
(`tests/python/test_gnc_missions.py:89`) asserts
`header["format"] == {"name": "SRLOG", "major": 1, "minor": 2}` on a real
writer-produced GNC run. That the reservation held is corroborated by the
reader accepting minors 0, 1 and 2 while refusing major 2
(`test_srlog_reader.py:89`, `test_cli.py:180`).

Oracle: `test_oracle_flag_stamped_in_header`
(`test_gnc_missions.py:257`) and
`test_oracle_true_still_injects_truth_and_stamps_the_header`
(`test_gnc_python_component.py:869`) both assert `header["oracle"] is True`
reading the header alone, covering the built-in chain and a plugin
component respectively. The negative direction is pinned in three further
tests.

## Criterion 6 — nav-fix and altimeter statistics

Gated on core data over 1,000 draws by
`sensors_navfix_altimeter_chi_square_over_1000_draws`
(`cpp/tests/test_sensors.cpp:584`), and re-gated end to end through the
blind reference at `tests/python/test_p6_optical_gates.py:647` and `:673`.
The bounds `[2.850, 3.154]` and `[0.914, 1.090]` are hard-coded in C++ but
independently reproduced at `tests/python/test_refs_chi2.py:105`.

This gate was attacked directly by corrupting the logged altimeter stream
and re-running the reference gate. Baseline statistic `0.9936` against
bounds `[0.8464, 1.1662]` over N = 300:

| Injected defect | Statistic | Gate response |
|---|---|---|
| Constant bias 0.10 m (0.20 σ) | 1.0593 | pass |
| Constant bias 0.15 m (0.30 σ) | 1.1221 | pass |
| Constant bias 0.20 m (0.40 σ) | 1.2049 | **detected** |
| Constant bias 1.00 m (2.00 σ) | 5.2498 | **detected** |
| Measurement noise scaled ×0.90 | 0.8049 | **detected** |
| Measurement noise scaled ×1.10 | 1.2023 | **detected** |
| Ellipsoid dropped (h = norm(r) − a) | 2128.09 | **detected** |

The ellipsoid-drop result confirms that the documented inclination fix
works as recorded: on the 45-degree fixture the sphere-versus-ellipsoid
truth difference spans −49.98 m to −0.16 m against a 0.5 m sigma, and the
mutation is rejected by three orders of magnitude. The gate resolves a
0.4 σ systematic and a 10 % noise mis-scale. It is sharp.

**Note, not a defect.** The criterion's "1,000 seeded draws" is met only
by the C++ test; the Python integration re-gate runs ~300 draws with
correspondingly wider bounds derived from the sample size. The clause is
satisfied, but `docs/mathlib/chapters/sensors_radio.tex:194` claims "these
are exactly the statistics the committed pytest driver computes", which
elides the draw-count difference.

## Criterion 7 — camera pose, intrinsics, and pixel projection

**Pixel clause: solid.** `test_landmark_pixels_match_independent_reference`
(`tests/python/test_p6_optical_gates.py:435`) gates logged pixels against
`tests/refs/pinhole.py` at `< 1e-6 px` (`:456`), on a fixture with
`fx != fy`, an off-axis principal point, a non-zero mount station, and a
non-identity mount rotation — geometry chosen so a transposed convention
or a shared scale cannot hide. The reference imports nothing from
`star_reacher`, and `test_landmarks_are_actually_visible` (`:461`) guards
against the gate passing on an empty set.

**Pose clause: satisfied by construction, as the criterion's phrasing
invites.** `test_camera_pose_channels_are_bit_exact_truth` (`:478`)
asserts `np.array_equal(camera["r_m"], truth["r_m"][rows])`. Both arrays
are read from the same log, written by the same run, and the hook assigns
the truth doubles rather than recomputing them
(`cpp/src/sensors/camera.cpp:121-123`). This is a value compared against a
copy of itself. It retains real power to catch two specific defects — a
mis-indexed truth row, and a hook that logged the mount-offset-shifted
station `r_cam_i` instead of `r_i`, which is a live distinction on this
fixture since `r_cam_b_m = [0.5, -0.25, 0.125]`. Judged against what the
criterion asks, "equal bit-exactly" is a statement about a copy, and a
copy is what was built; the clause is adequate but proves less than a
reader of the criterion would assume.

**Intrinsics clause: UNCOVERED.** The criterion says "camera-hook pose
**and intrinsics** equal the `truth` channels bit-exactly." No intrinsics
channel exists. `cpp/include/star/srlog_writer.hpp:209-214` carries
`t_s`, `r_m`, `q_i2b`, and `px_uv` only, and
`docs/formats/srlog_v1.md:278` states that "camera intrinsics are
configuration data (they ride in the resolved config, not the record
stream)". The gate test reads intrinsics from the mission TOML it wrote
itself, so no intrinsic value is ever compared against a core output.

**Defect (documentation, self-contradictory).**
`docs/mathlib/chapters/camera.tex:359-360` states "Exit criterion~7 gates
the pose, intrinsics, and landmark pixels, all of which the current layout
carries." Six lines earlier, `:353-355` states that the record layout
carries "$t_s$, $\vv{r}$, $\quat{q}_{i2b}$, and the $2L$ interleaved pixel
coordinates only." The two statements contradict each other and the second
is the correct one.

**Defect (evidence not claimed).** `camera.tex:325-336` still declares the
independent NumPy pixel recomputation "outstanding". That clause is
implemented and passing at `test_p6_optical_gates.py:435`. The chapter's
evidence table cites no Python test; its caption attributes every row to
`cpp/tests/test_sensors.cpp`.

**Remedy.** Either amend the criterion to drop "and intrinsics", recording
why (intrinsics are configuration, not a logged channel), or add an
intrinsics echo to the log header and gate it. Correct `camera.tex:359-360`
and move the pixel clause from "outstanding" into the evidence table.

## Criterion 8 — latency shifts application by exactly k cycles

Gated by `test_latency_two_cycles_shifts_application`
(`tests/python/test_gnc_missions.py:229`), plus
`gnc_cycle_latency_shifts_application_by_exactly_k`
(`cpp/tests/test_gnc_cycle.cpp:291`) and `gnc_latency_fifo_semantics`
(`cpp/tests/test_gnc.cpp:336`).

**The comparison covers exactly one cycle pair.** The Python test asserts
the k = 2 run's cycle 2 equals the k = 0 run's cycle 0 (`:253-254`), that
the two pre-fill cycles apply zero torque, and that the first valid index
is 2. There is no assertion that cycle 3 equals cycle 1, no loop over the
601 records, and no assertion about the run-end boundary, where the last
k computed commands are never applied.

This is not simply an omission: the loop is closed, so delaying the
applied torque changes the plant trajectory, and from cycle 1 onward the
k = 2 run's *computed* commands genuinely differ from the k = 0 run's.
Only cycle 0 is comparable between the two runs. The tests are correct to
restrict themselves to it — but that means the criterion as written
("shifts logged command application by exactly k control cycles versus the
k = 0 run") is verified at one sample, and the general claim rests on the
FIFO unit test in open loop.

**Defect (documentation overclaim).**
`docs/mathlib/chapters/gnc_builtin.tex:108-113` asserts that "exit
criterion~8 holds by construction: the logged command-application history
of an $L = \ell$ run is the $L = 0$ history delayed by exactly $\ell$
cycles". That is true of the FIFO in isolation and false of a closed-loop
run, which is the only thing `gnc.cmd` ever logs. The evidence-table row
at `:502-506` is, by contrast, accurately scoped. A minor related hole:
`test_gnc_cycle.cpp:333` checks only `k2[1].tau[2] == 0.0`, leaving
`tau[0]` and `tau[1]` of the second pre-fill cycle unasserted.

**Remedy.** Restate the chapter prose to scope the "by construction" claim
to the FIFO. Add an open-loop assertion covering the full history: drive
the cycle with a recorded command sequence and a fixed plant, and assert
`applied[i + k] == computed[i]` for every i, including the drain at run
end. That is a C++ test and requires a rebuild.

## Criterion 9 — velocity aberration to < 1 milliarcsecond

Gated by `test_aberration_matches_independent_reference`
(`tests/python/test_p6_optical_gates.py:319`), asserting
`worst < ABERRATION_TOL_MAS` with `ABERRATION_TOL_MAS = 1.0` (`:67`),
against `tests/refs/aberration.py` — a genuinely independent reference
that imports only NumPy, defines its own constants, and derives the exact
relativistic form in its docstring from the Lorentz boost rather than
transcribing it. `test_aberration_signal_dominates_the_gate` (`:339`)
guards non-vacuity by asserting the aberration deflection is really
present at 20,000–21,000 mas.

**KNOWN-ISSUE-P6-2 is accurately described.** Measured on the fixture
geometry: dropping the transverse projection from the reference
(`u + beta` in place of `u + beta - (u·beta)u`) changes the answer by
**0.4696 mas**, against the documented 0.470 mas. The mutation survives the
1 mas gate. The fixture's line-of-sight sits at 102.93–103.02 degrees from
beta, off the 135-degree peak where the same mutation reaches 1.0177 mas.

**The known issue understates the remedy, and this is the substantive
finding.** It frames the blind spot as a budget limitation — that 1 mas
"cannot resolve second-order algebra". Measured, the actual worst residual
between the logged directions and the independent reference is
**2.576e-08 mas**, so the gate carries **3.88e+07×** headroom. The gate
compares the implementation against the normative first-order formula, and
against that reference the implementation agrees to eight significant
figures. There is no physical reason for the tolerance to sit at 1 mas;
the criterion's 1 mas is the *requirement*, not the achievable agreement.
Tightening the assertion to, say, 1e-5 mas would retain 400× headroom over
the observed residual while rejecting the drop-transverse mutation by a
factor of roughly 47,000.

**Second defect: the gate is blind to the attitude convention.** At
`:311` the reference side computes
`c_i2b = np.array(_core.quat_to_dcm(*truth["q_i2b"][row])).reshape(3, 3)`
and the residual is `separation_angle(c_i2b @ apparent, logged)`. The
logged value was produced by the core using the same rotation. Angular
separation is invariant under a common orthogonal rotation, so the DCM
cancels exactly and no error in `quat_to_dcm` can be detected here. An
independently validated `quat_to_dcm` exists at
`tests/refs/quaternions.py` and is used by the pinhole gate, but not by
this one. The module docstring's otherwise careful shared-inputs list
(`:32-38`) does not mention it.

**Defect (evidence not claimed).**
`docs/mathlib/chapters/sensors_optical.tex:468-483` still records the
independent aberration recomputation as "Outstanding", listing precisely
the work that `tests/refs/aberration.py` and
`test_aberration_matches_independent_reference` now do.

**Remedy.** Tighten `ABERRATION_TOL_MAS` to a value near the achievable
agreement and record the 1 mas criterion separately as the requirement the
tightened gate implies. Substitute `tests/refs/quaternions.quat_to_dcm` at
`:311`. Move the criterion-9 item from "Outstanding" into the evidence
table.

## Criterion 10 — ascent performance with the GNC stack in the loop

**No evidence exists, and the deferral was never registered.**

The only statement is `docs/mathlib/chapters/gnc_builtin.tex:511-517`,
which records the re-gate as "outstanding" and refers to "the Phase~5
deferral provisions" without naming a checklist item, a procedure, or a
recording location.

The Section 9 valve requires the item to be committed fully prepared and
registered on `docs/release_checklist.md`. It is not. That file's only
reference is prospective, at `:12-15`: later phases "add items here at
their phase closes if the hardware is still unavailable." No item was
added. This contrasts with the Phase 5 deferrals, which PRD `:204` records
against a named item and a named procedure.

The item is also not prepared. `docs/perf/pi5_checklist.md` scopes itself
to the Phase 5 clauses (`:8-12`, `:86-87`), and its measurement step
invokes `perf_gate.py measure` without `--ascent`, defaulting to
`missions/ascent_leo.toml` — which has no `[gnc]` table. Verified: the
only committed missions carrying a `[gnc]` table are
`leo_attitude_gnc.toml`, `leo_attitude_gnc_plugin.toml`, and
`leo_ekf_consistency.toml`, none of them an ascent. The GNC-in-loop ascent
variant is synthesized inline into `tmp_path` at
`tests/python/test_gnc_missions.py:295-326` and never persisted. And
`docs/perf/results/` records that no release has been qualified through
the checklist.

`tests/python/test_perf_harness.py` bears on the gate arithmetic only; its
one measurement test substitutes `missions/twobody_leo.toml` and accepts
either verdict (`assert proc.returncode in (0, 1)`).

**Remedy.** Commit the GNC-in-loop ascent mission as a real file, extend
`perf_gate.py` and `docs/perf/pi5_checklist.md` to measure it, and add a
Phase 6 criterion-10 item to `docs/release_checklist.md` naming both. The
criterion cannot be called closed until the item is at least fully
prepared and registered, which is what the valve requires even when the
hardware is absent.

## Evidence-table mis-citations

Collected for convenience. Each was verified against the cited file.

| Location | Claim | Status |
|---|---|---|
| `sensors_radio.tex:269-272` | `gnc_cycle_double_run_is_byte_identical` covers "the radio sensor channels" | False — the scenario has one ideal IMU |
| `sensors_optical.tex:460-463` | same, "the optical sensor channels" | False — same reason |
| `camera.tex:316` | same, camera channels | False — same reason |
| `sensors_imu.tex:575-578` | same, "the sensor channels" | True only for a noise-free IMU |
| `gnc_builtin.tex:487` | Python transcription exercises `eq:gnc:deltaq`–`eq:gnc:sat` | False — `werr`, `sign`, `sat` are all degenerate on the fixture |
| `gnc_builtin.tex:108-113` | criterion 8 "holds by construction" for the logged history | False in closed loop |
| `camera.tex:359-360` | criterion 7 gates intrinsics, "all of which the current layout carries" | False, and contradicts `:353-355` |
| `camera.tex:325-336` | pixel recomputation "outstanding" | Stale — implemented at `test_p6_optical_gates.py:435` |
| `sensors_optical.tex:468-483` | independent aberration recomputation "Outstanding" | Stale — implemented at `test_p6_optical_gates.py:319` |
| `ekf.tex:851-853` | the criterion-3 driver invokes the directory CLI form | False — it computes in-process |
| `ekf.tex:869` | criterion-3 test asserts only the ensemble average in bounds | Incomplete — the undocumented coverage assertion also gates |
| `sensors_radio.tex:194` | "exactly the statistics the committed pytest driver computes" | Elides 1,000 versus ~300 draws |

The `eq:gnc:cmdrate` / `eq:gnc:latency` pair that an earlier workstream
corrected is now clean: both `eq:gnc:cmdrate` citations attach to
commanded-rate code, and the sole `eq:gnc:latency` reference attaches to
the FIFO row.

Two test docstrings mislabel their criterion:
`test_p6_optical_gates.py:567` and `:608` say "Exit criterion 6" for the
star tracker and sun sensor. The star tracker belongs to criterion 1; the
sun sensor is required by neither criterion 1 nor 6 and is supplementary
coverage.

## Not assessable without a rebuild

The audit did not compile. The following need a rebuilt core, with the
experiment specified.

1. **Criterion 1 clause C, IMU stochastic path.** Add
   `random_walk`, `bias_instability`, `bias_tau_s`, and a quantizer step to
   the full-suite bit-identity mission and confirm two seeded runs still
   hash identically. Then perturb the Gauss-Markov state initialization and
   confirm the hash comparison goes red.
2. **Criterion 1 clause B, aberration composition.** Re-run
   `sensors_startracker_chi_square_over_1000_draws` with a non-zero
   `truth.v_end_i_mps`, then reverse the `q_ab` composition order and
   confirm the statistic leaves `[2.850, 3.154]`.
3. **Criterion 2, C++ side.** Confirm `gnc_pd_attitude_golden` rejects a
   `kp`/`kd` swap and a dropped saturation clamp against the mpmath
   goldens, which is the coverage the Python gate cannot supply.
4. **Criterion 8, full-history shift.** Add the open-loop assertion
   described above and confirm it rejects an off-by-one FIFO depth, which
   the single-cycle comparison cannot distinguish from a correct shift at
   cycle 0.
5. **Criterion 10.** The measurement itself, on Pi 5 hardware, against a
   committed GNC-in-loop ascent mission that does not yet exist.

## Addendum: the rebuild-required experiments, run

The five experiments above were specified but not executed, because the
audit could not compile. Four of them have since been run on a rebuilt
core. Every figure below was measured, not predicted; each mutation was
applied to the source, built, run, and reverted.

**Experiment 4, criterion 8 — the FIFO depth. Both gates fire.**
`gnc_latency_fifo_full_history_shift` (`cpp/tests/test_gnc.cpp:339`) was
added as the open-loop assertion this document called for: it drives the
FIFO with a recorded 20-command sequence for k in 0..4 and asserts
`applied[i + k] == produced[i]` on all three axes for every i, plus the
pre-fill holds and the drain at run end.

| Mutation of `cpp/src/gnc/component.cpp:318` | Latency cases | Result |
|---|---|---|
| baseline | 3 of 3 pass, 1328 assertions | — |
| pre-fill depth k -> k+1 | 0 of 3 pass, 285 assertions fail | detected |
| pre-fill depth k -> k-1 | 0 of 3 pass, 222 assertions fail | detected |

Under the k -> k+1 mutation the Python gate
`test_latency_two_cycles_shifts_application` also fails, at
`cmd0["valid"][0] == 1` (measured `0`). Criterion 8's gates are sharp
against an off-by-one in either direction.

**Experiment 1, criterion 1 clause C — and the finding.** The
bit-identity gate's target is reproducibility, so the mutation of its
target is non-determinism rather than a changed value: a deterministic
perturbation of the Gauss-Markov initialization changes the numbers but
leaves two runs of one build identical, which is the correct behaviour of
an identity gate. The IMU stream was therefore seeded
`master_seed ^ std::random_device{}()`.

| Gate | Verdict under a non-deterministic IMU stream |
|---|---|
| `test_full_sensor_suite_reruns_bit_identical` | **PASSED** |
| `test_stochastic_imu_reruns_bit_identical` (new) | failed, SHA-256 mismatch |
| `test_ensemble_rerun_is_bit_identical` (criterion 3) | failed, ordered SHA list mismatch |

**The criterion-1 clause C gate named in this document does not fire when
the IMU's random-number stream is made completely non-deterministic.** Its
mission declares an ideal IMU, so every draw the stream produces is
multiplied by a zero coefficient and never reaches an output. The clause
was not wholly uncovered — criterion 3's ensemble identity test runs
`leo_ekf_consistency.toml`, whose IMU enables the random-walk and
Gauss-Markov terms, and it does fire — but the coverage was incidental to
another criterion and reached only two of the six error terms. The
quantizer's residual carry, the turn-on bias, the scale factor, and the
misalignment matrix were enabled by no committed mission at all.
`test_stochastic_imu_reruns_bit_identical` closes this with the full error
chain.

**Criterion 1 clause A — the Allan gate is sharp and selective.**

| Mutation | `n_hat/N - 1` | `b_hat/B - 1` | Result |
|---|---|---|---|
| baseline | — | — | passes |
| ARW coefficient x 1.2 | **0.190033** | 0.030107 | ARW clause detected |
| Gauss-Markov sigma x 1.2 | 0.004078 | **0.229376** | bias clause detected |

Each mutation is caught by its own clause and by neither the other's, so
the two coefficients are separately resolvable rather than one standing in
for the other.

**Criterion 1 clause B — sharp on magnitude, blind on composition order.**
Scaling the star-tracker error sigma by 1.2 moves the statistic to
`q_mean = 4.33832` against the upper bound `3.154` (the expected
1.2^2 x 3 = 4.32), and the per-axis variance ratios to 1.414, 1.531, and
1.394 against 1.0900. The gate resolves a 20 % noise mis-scale.

Experiment 2 confirms the structural finding by measurement. With the
`q_ab` composition order reversed in `cpp/src/sensors/optical.cpp` —
`q_true (x) q_ab` in place of `q_ab (x) q_true` —
`sensors_startracker_chi_square_over_1000_draws` **passes unchanged**, 8
of 8 assertions. The fixture sets `truth.v_end_i_mps` to zero, which makes
`q_ab` the identity, and the identity commutes.

That order is not uncovered repository-wide: the Python integration
re-gate `test_star_tracker_statistic_passes_the_reference_gate` rejects
the same mutation at `statistic=104.463254` against `bounds=[2.729187,
3.283440]` over 300 draws, because it runs a real mission with non-zero
velocity. The composition hazard is covered; the criterion-1 C++ gate is
simply not what covers it.

**Experiment 5, criterion 10** remains unrun: it needs Pi 5 hardware and a
committed GNC-in-loop ascent mission. **Experiment 3, criterion 2** was
not run in this pass.

## Independence of evidence

The three modules under `tests/refs/` — `aberration.py`, `pinhole.py`,
`quaternions.py` — import nothing from `star_reacher` and re-derive their
formulas and constants. That is genuine independence and it is the
strongest part of the Phase 6 evidence chain. The chi-square bounds are
likewise recomputed from first principles rather than transcribed, and the
PD golden torques come from an mpmath evaluation of the specification
rather than from the implementation.

Where independence is weaker it is because the harness reintroduces the
core on the reference side: `_core.quat_to_dcm` at
`test_p6_optical_gates.py:311` (criterion 9, cancels exactly),
`_core.gcrf_to_itrf` at `:423` (criterion 7, cancels exactly), and
`_core.geodetic_altitude` at `:698` (criterion 6, supplies the truth
altitude). The first is avoidable — an independent implementation is
already available and validated. The second and third are shared *inputs*
to both sides, which the module docstring discloses, and are defensible.

Criteria 7 and 9 were closed by the same workstream that wrote the optical
implementation, on evidence that workstream authored. The references are
blind and well constructed, but no independent party had attacked them
before this audit, which found the tolerance headroom in criterion 9 that
the authoring workstream had recorded as a budget limitation rather than a
loose gate.
