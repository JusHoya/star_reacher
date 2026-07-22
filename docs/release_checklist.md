# Pre-release checklist

The single register of deferred, maintainer-discharged items that qualify a
release. It implements the PRD section 9 valve — an exit-criterion clause
that could not be closed inside its own phase is deferred here, fully
prepared, and a release is qualified against that clause only by discharging
the item below — and additionally carries any release-qualifying
confirmation that can only be produced after a phase merge (item 3). Green
CI is necessary but not sufficient wherever this register applies.

The valve admits two kinds of blocker, and the register carries both.
Items 1 and 2 are the original kind: a required external **tool or hardware**
was unavailable at phase close, and the item is a prepared measurement
waiting for the resource. Item 4 is the second kind, added to section 9 at
the Phase 6 close: a clause whose closure requires a new field in a **frozen
on-disk format**, where what is deferred is a specified change rather than a
measurement. The three obligations are identical for both — committed fully
prepared, registered here, recorded inline beside the criterion — and only
the **Procedure:** line differs in character.

The register covers the phases closed so far (through Phase 6). Later phases
re-gate on Pi 5 hardware — Phase 7 exit criterion 4 and Phase 8 exit
criterion 4 — and add items here at their phase closes if the hardware is
still unavailable. Once Pi 5 hardware is available, attaching it as the
pinned self-hosted runner (PRD section 9) supersedes the manual route for
the performance clauses.

Each item names the clause it carries, its prepared procedure, and where the
result is recorded. When an item is discharged, commit its evidence as its
procedure directs and update its status line here in the same commit, so this
register always states what has and has not been done.

## 1. Raspberry Pi 5 hardware checklist

- **Carries:** the Pi 5 hardware clauses of Phase 5 exit criteria 1, 2, and 4
  (headless quicklook plots on a Pi 5; viewer in Pi 5 Chromium; the three
  single-core performance absolutes on real Pi 5 silicon), and the Pi 5
  hardware clause of **Phase 6 exit criterion 10** (the FR-32 ascent target
  holding with the built-in C++ GNC stack in the loop). Criterion 10 adds no
  new step: it is a fourth metric, `ascent_gnc_rt_factor`, measured by the
  same harness invocation in step 4 and gated at the same >= 100x.
- **Procedure:** [`docs/perf/pi5_checklist.md`](perf/pi5_checklist.md). Steps
  1–3 of that document double as the generic Pi 5 bring-up procedure
  (toolchain, source build into a fresh venv, `star verify --quick`) for any
  downstream Pi 5 deployment of the simulator.
- **Records to:** `docs/perf/results/` (measurement JSONs plus a README
  entry), per that checklist.
- **Status:** pending — no Pi 5 hardware is available to the maintainer.
  Deferred at Phase 5 close (2026-07-07); extended at Phase 6 close
  (2026-07-19) to carry exit criterion 10 on the same provision. The nightly
  `ubuntu-24.04-arm` leg is the interim aarch64 proxy and is never reported
  as a Pi 5 measurement. For the record, the closed-loop GNC ascent measures
  10,096x real time (median of three) on the maintainer's x86-64 Windows
  development host against 14,301x for the open-loop ascent on the same
  host and in the same runs — the GNC chain costs about 1.4x the wall time
  per simulated second. Neither number is a Pi 5 number and neither
  discharges this item.

## 2. MATLAB `parquetread` transcript

- **Carries:** the MATLAB clause of Phase 5 exit criterion 3 (D-15): exported
  Parquet loads in MATLAB with the documented schema and bit-exact values,
  evidenced by a committed console transcript.
- **Procedure:** [`tests/interop/matlab/`](../tests/interop/matlab/README.md)
  — validation script, expected values, and pinned input hashes are committed;
  the run is one scripted command on any MATLAB R2019a+ host.
- **Records to:** `tests/interop/matlab/transcript.txt` plus a manifest entry,
  per that README.
- **Status:** pending — no MATLAB-licensed host is available to the
  maintainer. Deferred at Phase 5 close (2026-07-07).

## 3. Nightly performance history

- **Carries:** Phase 5 exit criterion 5 in its steady state: the rolling
  10-run-median regression gate only accumulates history once
  `.github/workflows/nightly.yml` is on the default branch (GitHub schedules
  cron only there). The gate's compare logic is CI-tested independently of the
  schedule.
- **Procedure:** after the phase merge, confirm at least one green `nightly`
  run before tagging a release — either the scheduled run or a manual
  `workflow_dispatch` from the Actions tab.
- **Records to:** the workflow's run history and its measurement artifacts
  (self-recording).
- **Status:** pending first post-merge run.

## 4. SRLOG error-layout header field (KNOWN-ISSUE-P6-5, reader side)

- **Carries:** the reader-side half of KNOWN-ISSUE-P6-5, recorded inline
  against **Phase 6 exit criterion 3** in `PRD.md`. `_reduce_error`
  (`python/star_reacher/consistency_cli.py`) collapses slots 0..3 of
  `nav.err` as a scalar-first error quaternion whenever `n == m + 1` with
  `n >= 4`, on the strength of the dimensions alone, because the SRLOG
  header records only *whether* an error layout is present and not *what*
  it is. The producing side is already closed and is not part of this item:
  `validate_error_layout` (`cpp/src/gnc/component.cpp`) takes the
  component's `cov_dim()` alongside its `state_dim()` and refuses a layout
  reaching that shape unless the attitude block holds offset 0, so no log
  this simulator produces can reach the consumer mangled. What is deferred
  is the check for logs this simulator did **not** produce — a hand-written
  file, a synthetic fixture, or a log from a future producer whose rule
  differs — which are reduced with the assumption unverified. Criterion 3
  itself is unaffected in substance: it computes NEES on the built-in EKF,
  which is quaternion-led at 16/15, where the collapse is the correct
  reduction.
- **Procedure:** this is a format-field item, so what is prepared is a
  specified change rather than a measurement. Target format version **SRLOG
  1.4**, an additive minor bump under `docs/formats/srlog_v1.md` section 6.
  The new optional header key `gnc.error_layout` is present exactly when the
  navigation component declares a non-empty `error_layout()`, so a run that
  declares none stays byte-identical to its 1.3 predecessor apart from the
  version words. It must carry, per declared block and in ascending offset
  order: the **quantity** (`attitude`, `velocity`, `position`, `gyro_bias`,
  `accel_bias`; `quantity_name()` in `cpp/src/gnc/component.cpp` already
  supplies the canonical strings), the **form**, the **offset**, and the
  **width** in slots — plus the declaring component's **`state_dim`** and
  **`cov_dim`**, because it is their relationship (`n == m + 1`) that
  triggers the collapse and neither is otherwise recoverable from the log.
  Integers and enum strings only: the header carries no floats, so this
  needs none of the `ieee754-binary64-hex` treatment `gnc.camera` required.
  The layout already exists in the core as `error_layout()` and is already
  captured and validated at run construction by `capture_error_layout`
  (`cpp/src/vehicle_cycle.cpp`), which currently discards it after
  validation; the writer change is to thread it into `SrlogHeaderFields`
  instead. On the reading side the collapse must become conditional on the
  field, and **three** sites re-implement that reduction independently and
  must move together or the divergence becomes a new defect:
  `_reduce_error` in `python/star_reacher/consistency_cli.py`,
  `_p6_reduce_error` in `python/star_reacher/verify.py`, and `reduce_error`
  in `tests/python/test_ekf_consistency.py`. A log carrying no layout field
  keeps today's behaviour and must say so where the user can see it, rather
  than silently reducing as if verified.
- **Records to:** `docs/formats/srlog_v1.md` (the section 3 key definition
  and a section 6.1 history entry for 1.4), and a conformance test that
  drives `star consistency` against a log whose declared layout is *not*
  quaternion-led and asserts refusal — the coverage gap that currently makes
  the mangling reproducible only by hand.
- **Status:** pending — deferred at Phase 6 close (2026-07-19). The blocker
  is format stability rather than an unavailable resource: the change moves
  the writer, the reader, three independent reduction sites, the format
  specification and its conformance tests together, and landing it at the
  phase close would re-open the format surface after this phase's logs were
  frozen as goldens and its cross-platform byte determinism measured at
  SRLOG 1.3. It is not waived: KNOWN-ISSUE-P6-5 records the hazard, the
  producer-side refusal is committed and proven at three levels, and the
  field's contents are specified above so whoever implements it is not
  starting from scratch.

## Disclosed Phase 6 residuals (red-team registered)

The Phase 6 exit-criteria red-team (`docs/review/phase6_red_team.md`) found
nine of ten criteria substantively closed and the tenth (criterion 7) closed
on documentation alone. The items below are the bounded residuals it surfaced
behind the closed criteria. They are registered here — per the
register-deferred-evidence discipline — so each has an owner and a closure
step rather than living only in a review paragraph. Unlike items 1–4, these
do **not** block the phase close or gate a release: every criterion they touch
is met by committed gates demonstrated able to fail. They are follow-up
hardening and coverage work, tracked here to completion.

## 5. Criterion 3 — NEES epoch-structure null (windowed statistic)

- **Carries:** the disclosed null direction behind Phase 6 exit criterion 3.
  The headline ensemble NEES averages over 601 epochs, so a covariance defect
  whose sign reverses across the run cancels in the average and does not set
  the exit code even while it is visible in the reported coverage number. The
  limitation is spec-registered, not undiscovered:
  `docs/mathlib/chapters/ekf.tex:641` states the consequence verbatim and
  records measured instances, and `ekf.tex:747` names the out-of-scope
  successor — gate on a windowed statistic over a segment short enough to
  resolve the transient. This is distinct from the criterion-3 reader-side
  format field carried by item 4 (KNOWN-ISSUE-P6-5).
- **Procedure:** implement the windowed NEES statistic named at `ekf.tex:747`
  over a segment short enough to resolve a sustained sign-reversing transient,
  add it to the ensemble gate alongside the whole-run aggregate, and
  demonstrate it rejects one of the measured instances the chapter records.
  Test and chapter work only; no compile and no format change.
- **Records to:** the ekf chapter (extend the consistency/validation sections)
  and the ensemble-gate driver and its tests.
- **Status:** open — registered at Phase 6 close (2026-07-21). Not a
  phase-close or release blocker: criterion 3's committed gates (NEES on the
  headline, NIS on binomial coverage) pass and are proven able to fail; this
  is added detection power for a class the headline aggregate does not gate.

## 6. Criterion 9 — V026 fixture velocity and the star-tracker mas gate

- **Carries:** two confined sub-gate residuals behind Phase 6 exit criterion 9
  (optical velocity aberration). (a) `_v026_ephemeris`
  (`python/star_reacher/verify.py`) builds the Earth-about-EMB segment with
  `const_record(4671.0, 0.0, 0.0)` — a zero linear Chebyshev term, hence zero
  velocity — so the barycentric-velocity composition's second term is
  multiplied by zero and `star verify` alone is blind to a dropped term worth
  ~5.5 mas (5.5× the criterion's own 1 mas). The pytest gate
  `test_aberration_matches_independent_reference` uses the real DE440 excerpt
  and catches it, so criterion 9 is closed by the gate union, not by
  `star verify` in isolation. (b) The star-tracker aberration path
  `eq:optical:rho` (`rho = b_I × beta`, `cpp/src/sensors/optical.cpp`) is gated
  only by a chi-square statistic, never at the mas level against an independent
  computation as the sun-sensor and camera bearings are.
- **Procedure:** (a) give V026's `emb` and `earth` segments independent nonzero
  linear Chebyshev terms (the fixture builder already takes per-segment
  coefficients) so the composition's second term carries real signal and
  `star verify` alone would catch the dropped term; (b) add a direct mas-level
  comparison of the logged star-tracker quaternion on the noise-free optical
  fixture against an independent `eq:optical:rho` computation. Both are
  test/fixture edits; no compile and no format change.
- **Records to:** `python/star_reacher/verify.py` (`_v026_ephemeris`) and the
  optical-gate tests.
- **Status:** open — registered at Phase 6 close (2026-07-21). Not a
  phase-close or release blocker: criterion 9 is met by the gate union (its
  reference mutations are rejected far outside the tightened 1e-5 mas gate, per
  the criterion-9 remediation record in `PRD.md`). These close the fixture
  degeneracy and extend mas-level coverage to the third aberration path.

## 7. Criterion 1 — Allan `b_hat` flakiness and the orphaned `allan.py`

- **Carries:** two bounded residuals behind Phase 6 exit criterion 1 (IMU Allan
  recovery). (a) The `b_hat` ±10 % recovery check in
  `sensors_imu_allan_recovers_arw_and_bias_instability`
  (`cpp/tests/test_sensors.cpp:412`) is fragile on its pinned seed: the
  red-team measured a ~4.0 % false-failure rate (single-axis `b_hat/B − 1`
  scatter ~4.9 % std), so a last-bit RNG-stream change from a compiler or
  platform difference could flip the pinned seed to a false failure. This is
  flakiness to harden, not a detection gap — the check catches a ×1.15
  conversion defect at ~81 % power and ×1.30 at 100 %. (b) `tests/refs/allan.py`,
  the independently validated overlapping-Allan-deviation estimator, is
  imported only by `tests/python/test_refs_allan.py` (which has no core import)
  and is never run against core IMU output; the C++ gate computes its own
  overlapping Allan deviation with no cross-check against it.
- **Procedure:** (a) reduce the `b_hat` flakiness by averaging the recovery
  over the three instrument axes (~√3 scatter reduction toward sub-1 %
  false-failure) or over a seed ensemble; (b) add a cross-check of the C++
  Allan estimate against `tests/refs/allan.py` on the same core IMU output so
  the validated reference gates something. (a) needs a rebuild to re-measure;
  (b) is a test addition.
- **Records to:** `cpp/tests/test_sensors.cpp` (the Allan gate) and a new
  cross-check test wiring `tests/refs/allan.py` to core output.
- **Status:** open — registered at Phase 6 close (2026-07-21). Not a
  phase-close or release blocker: the criterion's bit-identity clause (C) is
  closed and its star-tracker chi-square clause holds at 1,000 draws; this
  hardens a fragile check and retires an orphaned reference.

## 8. Criterion 10 — `perf_gate` `--ascent-gnc` default pin

- **Carries:** the unpinned default behind Phase 6 exit criterion 10 (FR-32
  ascent ≥ 100× real time with the C++ GNC stack). `scripts/perf_gate.py:574`
  sets the `--ascent-gnc` mission default to `missions/ascent_leo_gnc.toml`
  (correct today), but no committed test pins that default, so a deliberate
  repoint would silently change what CI measures. The criterion is measured
  correctly today — the nightly job uses the default.
- **Procedure:** add a one-line assertion that `perf_gate.py`'s `--ascent-gnc`
  default equals `missions/ascent_leo_gnc.toml` (or that the measured GNC log
  carries a `gnc.cmd` group), so a changed default fails a committed test
  rather than passing silently. Pure test addition; no compile.
- **Records to:** a `perf_gate` unit test under `tests/python/`.
- **Status:** open — registered at Phase 6 close (2026-07-21). Not a
  phase-close or release blocker: criterion 10 is measured correctly by the
  nightly job on the named mission; this pins against silent default drift. The
  Pi 5 hardware clause of criterion 10 is carried separately by item 1.
