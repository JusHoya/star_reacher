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
waiting for the resource. Item 10 was that kind too, and is discharged. Item 4
is the second kind, added to section 9 at the Phase 6 close: a clause whose
closure requires a new field in a **frozen on-disk format**, where what is
deferred is a specified change rather than a measurement. The three
obligations are identical for both — committed fully prepared, registered
here, recorded inline beside the criterion — and only the **Procedure:** line
differs in character.

The register covers the phases closed so far (through Phase 8). Phase 7 exit
criterion 4 re-gates on Pi 5 hardware and is registered as item 9 at its phase
close; Phase 8 exit criterion 1 (the five new cross-tool cases vs frozen GMAT
truth) was registered as item 10 at the Phase 8 close, GMAT being unavailable
to the maintainer on the execution host, and is discharged as of 2026-07-25;
Phase 8 exit criterion 4 (the fresh-machine Pi 5 `star verify` walkthrough) is
carried by item 1's Pi 5 hardware register.
Once Pi 5 hardware is available, attaching it as the pinned self-hosted runner
(PRD section 9) supersedes the manual route for the performance clauses.

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
  same harness invocation in step 4 and gated at the same >= 100x. It also
  carries the Pi 5 timing clause of **Phase 8 exit criterion 4** (`star verify`
  prints `VERIFY: PASS` in `< 10 min` on a Pi 5): this is exactly the
  `star verify` run in the bring-up procedure below (step 3), timed on Pi 5
  silicon. The x86-64 measurement — full-tier `star verify` at ~9 s, 65x inside
  the 10-minute budget — is recorded in the README and the fresh-machine
  walkthrough, but an x86-64 number is not a Pi 5 number and does not discharge
  this clause.
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

## Phase 7 deferred items (Pi 5 hardware clause)

## 9. Phase 7 criterion 4 — ONNX loop closure on Pi 5 and the ARM cross-platform final state

- **Carries:** the Pi 5 hardware and cross-platform clauses of **Phase 7 exit
  criterion 4** ("an ONNX MLP exported from an external framework closes the
  loop for a full scenario on x86-64 and Pi 5, with cross-platform final states
  within the published bound"). The x86-64 clause is **met and gated in CI**:
  `tests/python/test_onnx_gnc.py` runs the committed closed-loop scenario
  (`missions/leo_attitude_onnx.toml` with `examples/onnx_gnc_plugin.py` and the
  committed MLP `tests/golden/onnx/pd_mlp.onnx`) on the `ubuntu-24.04` extras
  leg, proving the loop settles (10° → 0.171°, no NaN, `run_end` reached) and is
  bit-deterministic across reruns. What is deferred is the literal **Pi 5**
  half and the **x86-64-versus-aarch64 final-state** comparison, neither
  runnable without aarch64 hardware.
- **Procedure:** on real Raspberry Pi 5 silicon (aarch64): (a) `pip install
  'star_reacher[ml]'` and confirm the onnxruntime aarch64 wheel installs;
  (b) `star run missions/leo_attitude_onnx.toml --gnc-plugin
  examples/onnx_gnc_plugin.py` and confirm it reaches `run_end` with final
  attitude error < 1° and no NaN; (c) `python
  scripts/cross_platform_divergence.py extract` on both the x86-64 and the Pi 5
  `run.srlog` (distinct `--leg` labels), then `measure --bound 1e-9` and `gate
  --bound 1e-9`, confirming the ONNX mission's cross-platform final-state
  divergence is within the D-10 bound. The extract/measure/gate format is
  already exercised by `test_onnx_gnc.py`'s
  `test_final_state_is_capturable_for_cross_platform_comparison`, so the item
  ships fully prepared: the mission, the model, the plugin, and the comparison
  script are all committed, and discharging the clause is running the same
  scripted steps on aarch64. Until Pi 5 hardware is available, extending the
  extras install and this mission run to the `ubuntu-24.04-arm` CI leg is the
  committed aarch64 proxy (it is never reported as a Pi 5 measurement).
- **Records to:** `docs/perf/results/` (the Pi 5 run) and a cross-platform
  measurement JSON alongside the Phase 2 record, per the divergence script.
- **Status:** pending — no Pi 5 or other aarch64 hardware is available to the
  maintainer. Deferred at Phase 7 close (2026-07-23) on the same provision as
  items 1 and 2. Whether onnxruntime CPU inference is bit-reproducible across
  x86-64 and aarch64 within the D-10 bound is the open empirical question this
  item resolves; the x86-64 leg alone cannot answer it.

## Phase 8 deferred items (external-tool clause)

## 10. Phase 8 criterion 1 — frozen GMAT truth for the five new cross-tool cases

- **Carries:** the frozen-GMAT-truth generation for Phase 8 exit criterion 1's
  five new cross-tool cases, recorded inline against **Phase 8 exit criterion 1**
  in `PRD.md`: Molniya (`missions/molniya.toml`), lunar orbiter with the
  degree-matched GRGM field and SRP on (`missions/lunar_orbiter.toml`), and Mars
  orbiter, MRO-class, harmonics + SRP (`missions/mars_orbiter.toml`), each gated
  at position RMS < 100 m over 7 days; trans-lunar (the ballistic coast of
  `missions/tli.toml`) gated at < 1 km at lunar arrival; and Earth-Mars cruise
  (`missions/mars_cruise.toml`) gated at < 100 km at arrival. The two Phase 3
  cross-tool cases (GMAT LEO gravity, Orekit LEO drag) are already frozen and
  gated and are **not** part of this item. The illustrative LRO-ephemeris
  comparison (`missions/lro_illustrative.toml`) is **report-only, not a gate**
  (the real LRO's maneuvers and SRP/attitude history are unmodeled), so it
  carries no tolerance and appears here only as a note. What was deferred is the
  external truth: GMAT was not installed on the Phase 8 execution host, exactly
  the D-15 maintainer-boundary/unavailable-tool blocker the section 9 valve
  covers (the analogue of item 2's MATLAB clause). The DE440 lunar excerpt
  carrying the `moon_librations` segment
  (`tests/golden/ephemeris/excerpt_de440s_lunar.sreph`) that the lunar cases
  need was originally coupled to this item, but the DE440 kernels turned out to
  be fetchable on the Phase 8 host, so the excerpt was generated and **committed**
  there (deterministic, bit-identity-verified against the full repack); the
  lunar missions now run out of the box. Only the **GMAT frozen truth** was
  deferred under this item, for all five cases equally; it is now frozen and
  measured for all five (see **Status**).
- **Procedure:** this was a tool item, so what was prepared is a scripted
  measurement waiting for the resource. It ran as written, on a maintainer host
  with GMAT R2026a (pinned in `tests/golden/crosstool/manifest.toml`):
  1. (Optional) `python tests/golden/ephemeris/generate_lunar.py` — the lunar
     excerpt is already committed; the generator regenerates it byte-identically
     and asserts bit-identity against the full repack, so this only re-confirms
     it. No DE440 fetch is required for GMAT itself (GMAT supplies its own
     ephemeris).
  2. `python scripts/crosstool/gen_field_files_phase8.py` — confirm the
     committed `moon_grgm1200a_50x50.cof` and `mars_mro120f_20x20.cof` (already
     committed and deterministic; re-run only if the source excerpts changed).
  3. For each case, `python scripts/crosstool/run_gmat_phase8.py --case
     <molniya|lunar_orbiter|mars_orbiter|translunar|mars_cruise>` — run GMAT on
     the committed `.script` and freeze the truth CSV. `--arrival-s` is
     available to override the arrival epoch, and was not used for any case:
     `translunar` was frozen on the committed 455401 s default. The gravity
     fields, GMAT scripts, coordinate systems, and exact initial states are
     already committed and documented per case in the manifest; only the truth
     CSV bytes and the measured RMS are produced here.
  4. Confirm the five gates in
     `tests/python/test_crosstool_frozen_truth.py` stop skipping and pass, and
     record each measured RMS in `tests/golden/crosstool/manifest.toml`. All
     five now measure and pass, and the manifest's `date`, `generation`, and
     `tolerance` fields carry the frozen dates and the measured values; no
     `pending` marker remains under this item.
  The comparison machinery those gates use is already proven correct
  independent of GMAT by `test_rms_machinery_is_correct_on_sim_own_states`
  (the sim's own states as pseudo-truth, measured position RMS ~5e-9 m), so the
  external truth was the only missing piece.
- **Records to:** `tests/golden/crosstool/` (the five truth CSVs, with their
  manifest entries carrying the measured values and the SHA-256 pins of the
  frozen CSVs and their source `.script` files). The lunar excerpt and its
  `tests/golden/ephemeris/manifest.toml` entry are likewise committed. No
  `pending` entry remains for this item in either manifest.
- **Status:** discharged 2026-07-25 — the external GMAT truth is **frozen and
  committed** for all five cases, and all five star-vs-truth gates measure and
  pass. GMAT R2026a was unavailable on the Phase 8 execution host at close, but
  the maintainer's portable GMAT install lives on a second host (the laptop
  LAPTOP-HOYA), so the freeze half of this item was executed there in two
  sessions: the four orbit and cruise cases on 2026-07-24, and trans-lunar on
  2026-07-25 after the first trans-lunar freeze was invalidated (recorded
  below). Each session opened with a toolchain canary before any new run — the
  same install regenerated the committed Phase 3 truth
  `truth_gmat_leo_gravity_8x8.csv` **byte-identically** (CSV SHA-256
  181627b0..., GMAT report SHA-256 9909743b...) — so the freeze rests on an
  install first shown to reproduce an independently committed result. Every
  `truth_gmat_*.csv` was generated by `run_gmat_phase8.py` from the committed
  `.script`, and its SHA-256 pin and date are recorded in
  `tests/golden/crosstool/manifest.toml`.

  The measurement half ran through `tests/python/test_crosstool_frozen_truth.py`
  on a 0.8.0 build host, where all 8 tests in the file pass. The five gates
  measure:

  - `XTOOL-MOLNIYA-GMAT` — position RMS 0.160317 m against the < 100 m bound;
  - `XTOOL-LUNAR-GMAT` — position RMS 7.232688 m against < 100 m;
  - `XTOOL-MARS-ORBITER-GMAT` — position RMS 79.478630 m against < 100 m, the
    thinnest margin in the set at ~1.26x;
  - `XTOOL-MARS-CRUISE-GMAT` — arrival |dr| 952.550 m against < 100 km;
  - `XTOOL-TRANSLUNAR-GMAT` — |dr| 18.089824 m at the lunar-SOI arrival epoch
    against < 1 km, a margin of about 55x.

  The first four were measured on the desktop on 2026-07-24; the laptop carried
  only the 0.6.0 wheel and could not measure then. On 2026-07-25 a source build
  of this branch on the laptop (CPython 3.12.10, win_amd64) re-measured all
  four, and they reproduced exactly — an incidental confirmation of cross-host
  bit-determinism. The trans-lunar figure comes from that same run: velocity
  difference |dv| 5.379870e-05 m/s at the same epoch, dr components
  (-3.4141, -14.7677, -9.8742) m, with both tools placing the spacecraft about
  397,665 km from Earth at arrival (simulator 397664.930 km, GMAT
  397664.947 km). The two Phase 3 cross-tool gates (0.015243 m against GMAT,
  3.376229 m against Orekit) and the GMAT-independent machinery proof
  (`test_rms_machinery_is_correct_on_sim_own_states`, position RMS 4.618e-09 m)
  pass in the same run.

  What was **not** done, and why: `gmat_lunar_orbiter.script` and
  `gmat_mars_orbiter.script` could not be re-run from this worktree in the
  2026-07-25 session, because each hard-codes the main checkout's absolute path
  in its `.cof` field under the D-15 as-run convention — a property of an as-run
  artifact rather than a defect. Their truth stands from the 2026-07-24 freeze
  on the same install, and the corroboration available instead is that `molniya`
  and `mars_cruise` were re-run in the 2026-07-25 session and returned
  **byte-identical** CSVs (SHA-256 5341e53e... and 9892f24d...). The DE440 lunar
  excerpt sub-blocker was lifted earlier and stays lifted (the excerpt is
  committed and its manifest entry finalized). The item was never waived: every
  input is committed as-run, and the gates measure real RMS — they do not pass
  blind.

  **Invalidated first trans-lunar freeze (2026-07-24, recorded history).** The
  original deferral carried two residuals for the 0.8.0 build host: (a) confirm
  the trans-lunar arrival span, and (b) record each measured RMS in the manifest
  `tolerance` fields. Residual (b) was discharged for four of the five cases on
  2026-07-24 with the position RMS and arrival |dr| numbers above. Residual (a)
  uncovered more than an arrival-span change: the trans-lunar first freeze was
  **invalidated and removed** because its `.script` initial state was the tli
  t = 353 s truth state, which is mid-burn — the meco cutoff is commanded at
  353 s but the delivered thrust level is zero only from 354 s under the
  per-step spool discipline on the 1 s grid, so the replicated coast was
  ~15.3 m/s low in energy and missed arrival by 75,189 km at matched epochs (the
  simulator's own ballistic coast of the same mid-burn state reproduced the
  removed GMAT arrival state to 0.022 km, proving the tools agree and isolating
  the initial state as the fault). The gate's 75,212 km readout also contained
  ~23 km from a second, independent defect fixed alongside: the superseded test
  sampled the simulator log at the CSV's elapsed stamp without the 353 s MECO
  offset — a comparison-epoch misalignment worth ~45 km on its own, which would
  have failed the < 1 km gate even against a perfect freeze; the corrected test
  maps CSV time to mission time explicitly and asserts the epoch. The corrected
  script (t = 354 s burnout state, epoch 12:05:54Z, arrival span 455401 s) was
  committed, and the 455402 s span quoted while residual (a) was open belonged
  to the invalidated 353 s-based script and does not describe anything
  committed.

  **Residual (a), closed 2026-07-25.** The re-freeze ran exactly `python
  scripts/crosstool/run_gmat_phase8.py --case translunar` on the laptop, with no
  `--arrival-s` override: the committed 455401 s default stands, and the source
  `gmat_translunar.script` is unchanged from the pin the manifest already
  carries (SHA-256 7b2ae9a5...). GMAT's last reported row is
  ElapsedSecs 455401.000000129 s, reproducing the requested span to 1.29e-7 s;
  the frozen `truth_gmat_translunar.csv` (581 bytes, one arrival row, SHA-256
  7a3cca16...) stamps the nominal 455401.0 s. The corresponding arrival mission
  time is 455755.0 s exactly — 455401 s of CSV elapsed time plus the 354 s
  burnout offset — the tli truth log carries a row exactly there, so the gate's
  epoch assertion holds, and `tli.toml`'s `duration_s = 600000.0` covers it. The
  measured 18.089824 m lands where the removed-freeze diagnostic predicted
  (~0.02 km).

- **Ephemeris-source note (recorded 2026-07-25):** the four Phase 8 `.script`
  headers state a DE440 point-mass and lunar-orientation source for GMAT (as did
  `tests/golden/crosstool/README.md`, corrected in the same change). That is not
  what ran, and it is not
  selectable on the pinned install: no Phase 8 script sets
  `SolarSystem.EphemerisSource`, so GMAT used its default, and the run log
  records the planetary source as **DE405**, loading
  `data/planetary_ephem/spk/DE405AllPlanets.bsp`; the install ships only
  `leDE1941.405`, `leDE1900.421`, and `leDE18002100.424`, with no DE440 file
  present to select. The `.script` files are committed as-run artifacts whose
  SHA-256 pins the manifest already carries, and the frozen truth was generated
  from exactly those bytes, so they are not edited; the correction is recorded
  additively, in the manifest prose, the crosstool README, the report, and here. The simulator side reads the
  committed DE440 excerpt, so the planetary-ephemeris difference — a
  DE440-family excerpt against GMAT's DE405 — is an irreducible tool difference
  of exactly the kind exit criterion 1 exists to measure, alongside the
  already-recorded FK5/IAU-76-versus-CIO frame-chain difference. It is bounded
  empirically below every gate by the five measured residuals above, and most
  tightly by the trans-lunar case at 18.089824 m against 1 km, which is the most
  lunar-ephemeris-sensitive case in the set.

## 11. Phase 8 criterion 5 — release-wheel build and smoke on all four platforms

- **Carries:** Phase 8 exit criterion 5 in its release-time form — the
  four-platform wheels install and pass `verify --quick`. Unlike items 1, 2, 9,
  and 10, this was never blocked on an unavailable resource: it is a
  confirmation that can only be produced after the release tag is pushed
  (GitHub runs a tag-triggered workflow only once the tag exists), so it is
  registered here in the same spirit as item 3's first-nightly confirmation.
  The per-push `wheel-budget` and `dep-minimality` jobs already build and
  audit the wheel on all four legs, and the per-push `build-test` job already
  installs from source and runs `star verify --quick` on all four legs, so the
  capability is continuously exercised; what the tag adds is the distributable
  cibuildwheel artifacts and their isolated-venv `verify --quick` smoke.
- **Procedure:** after the phase merge, push the annotated `v0.8.0` tag and
  confirm the `release` job in `.github/workflows/ci.yml` goes green on all
  four legs (ubuntu-24.04, ubuntu-24.04-arm, macos-15, windows-2022): each
  builds its native wheel with cibuildwheel and runs `star verify --quick` in a
  fresh venv containing only the built wheel and its declared runtime
  dependencies. The wheels upload as `wheels-<os>` artifacts for attachment to
  the GitHub release.
- **Records to:** the `release` workflow run history and its uploaded wheel
  artifacts (self-recording).
- **Status:** pending first post-tag run. The release job and its tag trigger
  are committed; the tag is a maintainer action (a public disclosure event,
  D-19) and is not pushed as part of the phase merge.
