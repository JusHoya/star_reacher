# Raspberry Pi 5 performance checklist (manual, pre-release)

This checklist implements the PRD section 9 provision for the Pi 5
performance gates: *"a pinned self-hosted Pi runner is required for honest
gating (if unavailable, perf gates move to a manual pre-release checklist)."*
No self-hosted Pi 5 runner is attached to this repository, so this document
is that checklist. A maintainer runs it on real Raspberry Pi 5 hardware
before each release.

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

The FR-32 targets are single-core targets, so every measurement is pinned
to one core with `taskset`; child processes inherit the affinity mask.

1. Install the toolchain prerequisites and check out the release under test
   (replace `vX.Y.Z` with the tag being qualified):

   ```sh
   sudo apt-get update
   sudo apt-get install -y git python3-venv python3-dev cmake build-essential
   git clone https://github.com/JusHoya/star_reacher.git
   cd star_reacher
   git checkout vX.Y.Z
   ```

2. Build and install into a fresh venv (the native core builds from source;
   allow several minutes on the Pi):

   ```sh
   python3 -m venv .venv-pi5
   .venv-pi5/bin/pip install .
   ```

3. Confirm the installation is healthy before timing anything:

   ```sh
   .venv-pi5/bin/star verify --quick
   ```

   Proceed only on `VERIFY: PASS`.

4. Run the performance harness pinned to a single core (core 3 here;
   any single core is equivalent on the Pi 5):

   ```sh
   mkdir -p perf-results
   taskset -c 3 .venv-pi5/bin/python scripts/perf_gate.py measure \
     --json "perf-results/pi5-vX.Y.Z-$(date -u +%Y%m%d).json"
   ```

   The harness prints one line per metric and `PERF: PASS` or `PERF: FAIL`,
   and exits nonzero on any failed gate. The three gates are the Phase 5
   exit criterion 4 absolutes: Mission A wall < 60 s, ascent real-time
   factor >= 100x, sustained SRLOG write >= 50 MB/s.

5. Repeat step 4 twice more (three runs total, sequential, same command).
   Thermal throttling or SD-card garbage collection shows up as run-to-run
   spread; if any run fails a gate, the release does not ship until the
   failure is understood and resolved.

## Recording the result

Commit the three measurement JSONs under `docs/perf/results/` in the release
branch, named as produced by step 4 (with `-run2`/`-run3` suffixes for the
repeats), together with one short entry appended to
`docs/perf/results/README.md` stating: release tag, date, storage medium,
cooling, OS image version, and the three-run PASS/FAIL verdict. The JSON
files already carry the runner identity block (platform, machine, CPU
count), package version, and git SHA, so the numbers stay attributable to
the exact hardware and source state that produced them.

A release is qualified against Phase 5 exit criterion 4 only by this
checklist; a green nightly proxy leg is necessary but not sufficient.
