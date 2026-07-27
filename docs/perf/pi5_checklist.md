# Raspberry Pi 5 checklist (manual, pre-release)

This checklist implements the PRD section 9 valve for the Pi 5 hardware
clauses: honest gating requires a pinned self-hosted Pi runner, none is
attached to this repository, and the affected exit-criterion clauses
therefore move to this manual pre-release checklist. A maintainer runs it
on real Raspberry Pi 5 hardware; until it has been run, every clause it
carries stays open. Besides the performance gates it carries the two other
Phase 5 exit-criterion clauses that name literal Pi 5 hardware — the
headless quicklook-plot render (exit criterion 1) and the
viewer-in-Chromium check (exit criterion 2) — so every Pi 5 clause in the
phase has exactly one manual home (steps 6 and 7). From the Phase 6 close
it additionally carries that phase's exit criterion 10, the ascent target
re-gated with the built-in C++ GNC stack in the loop, which rides along in
step 4 as a fourth metric. From the Phase 8 close it carries that phase's
exit criterion 4 timing clause — `star verify` printing `VERIFY: PASS` in
`< 10 min` on a Pi 5 — which is the timed full-tier run in step 3, and
from the Phase 7 close it carries that phase's exit criterion 4 (the ONNX
MLP closing the loop on a Pi 5, and the x86-64-versus-aarch64 final-state
comparison), which is steps 8 and 9, so one Pi 5 session discharges both
register items rather than leaving the second silently open. It is
registered as items 1 and 9 of the pre-release checklist
(`docs/release_checklist.md`); both items are scheduled into **Phase 9 —
hardware and external-tool validation** as of 2026-07-26, which
re-schedules the clauses and neither discharges nor waives them — and
neither item gates the `v0.8.0` tag, whose release gate is
`docs/release_checklist.md` items 3 and 11 alone. The "the release does not
ship" conditions below are therefore conditions on a release that is being
qualified against these clauses, not on v0.8.0. Steps 1–3
also serve as the Pi 5 bring-up procedure for any downstream deployment of
the simulator: they take a bare Raspberry Pi OS image to an installed,
verified `star` CLI using only the project's standard from-source install
(toolchain, clone, `pip install .`, `star verify --quick`) — nothing in
that subset is specific to release qualification, and a bring-up that is
not qualifying a release can stop at the health check without the timed
run that follows it.

**Proxy honesty.** The nightly workflow (`.github/workflows/nightly.yml`)
runs the same gates on the GitHub-hosted `ubuntu-24.04-arm` runner class.
That leg shares the Pi 5's aarch64 architecture but runs a Neoverse-class
server core with server-class storage: its numbers are proxy numbers that
catch regressions between releases, and they are never reported as Pi 5
measurements. Only the procedure below produces a Pi 5 number.

## Hardware and OS baseline

Record deviations from this baseline alongside the results; a different
SD card or cooling situation changes the write-throughput and thermal
behavior and must be visible next to the numbers it produced.

- Raspberry Pi 5 (8 GB), official 27 W USB-C power supply, active cooler.
- Raspberry Pi OS (64-bit, Bookworm or later), fully updated.
- Storage: name the actual medium (microSD class/model, or NVMe HAT + SSD
  model) in the results record; the SRLOG write gate is storage-bound.
- No other user workload running (fresh boot, no desktop session needed).

## Procedure

The FR-32 targets are single-core targets, so every FR-32 measurement is
pinned to one core with `taskset`; child processes inherit the affinity
mask. The step 3 `star verify` timing is not an FR-32 target and is not
pinned — see that step.

1. Install the toolchain prerequisites and check out the release under test
   (replace `vX.Y.Z` with the tag being qualified):

   ```sh
   sudo apt-get update
   sudo apt-get install -y git python3-venv python3-dev cmake build-essential
   git clone https://github.com/JusHoya/star_reacher.git
   cd star_reacher
   git checkout vX.Y.Z
   ```

   For a bring-up that is not qualifying a tagged release — for example, a
   downstream Pi 5 deployment before the first tagged release exists —
   check out `main` or the commit under test instead of a tag.

2. Build and install into a fresh venv (the native core builds from source;
   allow several minutes on the Pi):

   ```sh
   python3 -m venv .venv-pi5
   .venv-pi5/bin/pip install .
   ```

3. Confirm the installation is healthy, then time the full acceptance suite:

   ```sh
   .venv-pi5/bin/star verify --quick
   ```

   Proceed only on `VERIFY: PASS`.

   That health check is the smoke tier. **Phase 8 exit criterion 4** names
   the full suite — `star verify` with no `--quick`, printing `VERIFY: PASS`
   in `< 10 min` on a Pi 5 — so run it again on the full tier and keep the
   wall clock:

   ```sh
   mkdir -p perf-results
   start=$(date +%s)
   .venv-pi5/bin/star verify > perf-results/pi5-verify-full.log 2>&1
   rc=$?
   end=$(date +%s)
   tail -n 1 perf-results/pi5-verify-full.log
   echo "verify_full_s = $((end - start)) (budget 600), exit ${rc}"
   ```

   The step passes only when all three hold: exit status 0, a last line of
   `VERIFY: PASS (N/N)`, and `verify_full_s` under the 600 s budget. If any
   of the three misses, the release does not ship until the failure is
   understood and resolved. Record `verify_full_s` in the results entry: it
   is the number that discharges criterion 4's Pi 5 timing clause, and no
   x86-64 or proxy-runner number substitutes for it.

   Unlike the measurements below, this run is deliberately **not** pinned
   with `taskset`. The criterion states the wall time a fresh-machine user
   observes on the whole Pi, not a single-core FR-32 target, so pinning it
   would measure something the criterion does not ask for. The tier line
   `star verify` prints first names any registered check the running tier
   leaves out, so the captured log evidences for itself that the timed run
   was the full suite.

4. Run the performance harness pinned to a single core (core 3 here;
   any single core is equivalent on the Pi 5):

   ```sh
   mkdir -p perf-results
   taskset -c 3 .venv-pi5/bin/python scripts/perf_gate.py measure \
     --json "perf-results/pi5-vX.Y.Z-$(date -u +%Y%m%d).json"
   ```

   The harness prints one line per metric and `PERF: PASS` or `PERF: FAIL`,
   and exits nonzero on any failed gate. Four gates are measured. Three are
   the Phase 5 exit criterion 4 absolutes: Mission A wall < 60 s, ascent
   real-time factor >= 100x, sustained SRLOG write >= 50 MB/s. The fourth,
   `ascent_gnc_rt_factor`, is **Phase 6 exit criterion 10**: the same >= 100x
   ascent target re-measured on `missions/ascent_leo_gnc.toml`, which flies
   the same profile with the built-in C++ GNC chain closing the attitude loop
   instead of an open-loop pitch-program sequence action. It needs no
   separate command — the default metric set already includes it.

   Report both ascent factors in the results entry, not just the closed-loop
   one. Their ratio is the per-cycle cost of the GNC chain on Pi 5 silicon,
   which is the number a reader wants and which no proxy runner can supply.

5. Repeat step 4 twice more (three runs total, sequential, same command).
   Thermal throttling or SD-card garbage collection shows up as run-to-run
   spread; if any run fails a gate, the release does not ship until the
   failure is understood and resolved.

6. Headless quicklook plots on the Pi (Phase 5 exit criterion 1's "on a
   Pi 5" clause). From an SSH session with no display attached:

   ```sh
   .venv-pi5/bin/star run missions/ascent_leo.toml -o perf-results/pi5-ascent
   .venv-pi5/bin/star plot perf-results/pi5-ascent/run.srlog
   ls perf-results/pi5-ascent/plots
   ```

   All seven named PNGs (groundtrack, altitude_speed, elements,
   attitude_rates, mass_thrust_throttle, qbar_mach, forces_by_source) must
   be present and nonzero; record the file listing with the results. The
   data-level golden regression itself is CI-gated on the aarch64 leg and
   is not repeated here.

7. Viewer in Pi 5 Chromium (Phase 5 exit criterion 2's "Pi 5 Chromium"
   clause). Generate the viewer for the same run, disconnect networking
   (`sudo rfkill block all` and unplug Ethernet, or `nmcli networking off`),
   then open the file in the OS-shipped Chromium:

   ```sh
   .venv-pi5/bin/star view perf-results/pi5-ascent/run.srlog
   chromium-browser perf-results/pi5-ascent/run.html
   ```

   Confirm: the scene renders and plays; scrubbing to the extremes shows
   the HUD epochs equal to the log's first/last epochs (printed by
   `star view` at generation); DevTools' Network panel records no request
   beyond the local file and its blob: module URL. Record PASS/FAIL and the
   Chromium version with the results, then re-enable networking.

8. Learned-controller loop closure on the Pi (Phase 7 exit criterion 4's Pi 5
   clause, register item 9). Networking must be back on — step 7 ends by
   re-enabling it — because the extras install fetches a wheel. Install the
   `[ml]` extra, confirm an aarch64 onnxruntime wheel resolves, and fly the
   committed closed-loop ONNX mission:

   ```sh
   .venv-pi5/bin/pip install '.[ml,dev]'
   .venv-pi5/bin/python -c "import onnxruntime as o; print(o.__version__)"
   .venv-pi5/bin/star run missions/leo_attitude_onnx.toml \
     --gnc-plugin examples/onnx_gnc_plugin.py -o perf-results/pi5-onnx
   .venv-pi5/bin/python -m pytest tests/python/test_onnx_gnc.py -q
   ```

   The mission is a 60 s attitude acquisition whose control slot is flown by
   the committed MLP `tests/golden/onnx/pd_mlp.onnx` through onnxruntime
   inside the deterministic time loop; `--gnc-plugin` is what puts it there,
   and the mission errors out naming the control slot if the flag is
   omitted. `tests/python/test_onnx_gnc.py` is the committed gate for the
   clause's substance — it asserts the run reaches `run_end`, that no
   quaternion, body rate, or commanded torque goes non-finite, and that the
   opening 10-degree attitude error settles below 1 degree — so run it here
   rather than re-deriving those checks by hand. The `dev` extra is
   installed only for pytest; the reinstall re-runs the source build, which
   the persistent build directory `pyproject.toml` configures
   (`tool.scikit-build.build-dir`) keeps incremental rather than a fresh
   compile of the core.

   Those tests **skip** when onnxruntime is absent, so a skipped count is a
   failed install rather than a pass: the step is qualified only when the
   module reports every test passed. Record the onnxruntime version and the
   `run.srlog sha256` line `star run` prints with the results; the run
   directory feeds step 9.

9. Cross-platform final state, x86-64 against the Pi (Phase 7 exit criterion
   4's remaining clause, the second half of register item 9). The comparison
   needs the same mission's final state from an x86-64 host at the same
   source state: run step 8's `star run` on the development host at the
   identical tag or commit, extract there with the same command, and copy
   that `finalstate.json` to the Pi (or copy the Pi's to the host — the
   comparison is symmetric). Extract both legs into one directory under
   distinct `--leg` labels, then measure and gate at the D-10 bound:

   ```sh
   .venv-pi5/bin/python scripts/cross_platform_divergence.py extract \
     --srlog perf-results/pi5-onnx/run.srlog \
     --leg pi5-aarch64-onnx \
     --out perf-results/onnx-xplat/pi5/finalstate.json
   # place the x86-64 leg's finalstate.json, produced by the same command on
   # the development host, at perf-results/onnx-xplat/x86_64/finalstate.json
   .venv-pi5/bin/python scripts/cross_platform_divergence.py measure \
     --dir perf-results/onnx-xplat --expect-legs 2 --bound 1e-9 \
     --out perf-results/onnx-xplat/measurement.json
   .venv-pi5/bin/python scripts/cross_platform_divergence.py gate \
     --measurement perf-results/onnx-xplat/measurement.json \
     --record tests/golden/determinism/cross_platform.toml \
     --bound 1e-9
   ```

   `measure` requires the two legs' final epochs to be bit-identical and
   fails otherwise, an epoch mismatch meaning the two runs are not the same
   scenario and the comparison would be meaningless; it prints
   `max_rel=<value>` but does not itself exit nonzero on a breach, so `gate`
   is the command that enforces the 1e-9 bound and the one whose exit status
   is the verdict. `gate` also reads the committed criterion-8 record, whose
   `measured_max_rel` belongs to the two-body reference mission on the four
   CI legs, so the factor-of-10 drift line it may print against that record
   is advisory here and is not a statement about this mission.

   Whether onnxruntime CPU inference is bit-reproducible across x86-64 and
   aarch64 within the D-10 bound is the open empirical question this step
   resolves, so record the measured `max_rel` whichever way it lands. A
   value above the bound is a finding to attribute and report — under exit
   criterion 8, D-10's bound is revised only formally and in the same change
   as its record, never widened to absorb a measurement.

## Recording the result

Commit the three measurement JSONs under `docs/perf/results/` in the release
branch, named as produced by step 4 (with `-run2`/`-run3` suffixes for the
repeats), together with one short entry appended to
`docs/perf/results/README.md` stating: release tag, date, storage medium,
cooling, OS image version, and the three-run PASS/FAIL verdict. The JSON
files already carry the runner identity block (platform, machine, CPU
count), package version, and git SHA, so the numbers stay attributable to
the exact hardware and source state that produced them.

Record the step 6 file listing and the step 7 PASS/FAIL + Chromium version
in the same README entry.

Record the step 3 `verify_full_s` wall time, the step 8 onnxruntime version
and loop verdict, and the step 9 measured `max_rel` in that entry as well —
the harness JSONs carry only the four gated FR-32 metrics and their runner
identity, so these numbers have no home in them. Commit the step 9 Pi 5
`finalstate.json` under `docs/perf/results/` and its `measurement.json`
alongside the committed criterion-8 record
`tests/golden/determinism/cross_platform.toml`, as register item 9's
*Records to* line directs, so the divergence figure is reproducible from
committed artifacts rather than readable only in the entry.

A release is qualified against the Pi 5 clauses of Phase 5 exit criteria
1, 2, and 4, of Phase 6 exit criterion 10, of Phase 7 exit criterion 4, and
of Phase 8 exit criterion 4 only by this checklist; a green nightly proxy
leg is necessary but not sufficient.
