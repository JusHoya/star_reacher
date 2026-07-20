# Pre-release checklist

The single register of manual, maintainer-run evidence items that qualify a
release. It implements the PRD section 9 valve — an exit-criterion clause
whose required external tool or hardware was unavailable when its phase
closed is deferred here, fully prepared, and a release is qualified against
that clause only by executing the item below — and additionally carries any
release-qualifying confirmation that can only be produced after a phase
merge (item 3). Green CI is necessary but not sufficient wherever this
register applies.

The register covers the phases closed so far (through Phase 5). Later
phases re-gate on Pi 5 hardware — Phase 6 exit criterion 10, Phase 7 exit
criterion 4, and Phase 8 exit criterion 4 — and add items here at their
phase closes if the hardware is still unavailable. Once Pi 5 hardware is
available, attaching it as the pinned self-hosted runner (PRD section 9)
supersedes the manual route for the performance clauses.

Each item names the clause it carries, its prepared procedure, and where the
result is recorded. When an item is executed, commit its evidence as its
procedure directs and update its status line here in the same commit, so this
register always states what has and has not been run.

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
