# Fresh-machine walkthrough

This is the path a stranger with a clean machine follows from a bare clone to a
rendered 3D trajectory, using only documented commands. It is the concrete
form of the four-command [Quickstart](../README.md#quickstart) (DX-1), run
verification-first (DX-5), and it adds two things the Quickstart summarizes:
the read-the-log-from-NumPy analysis path (FR-31) and the full-suite
verification step. Every command below was run and confirmed on the
maintainer's x86-64 host; the only platform-specific line is the venv path
(`Scripts/` on Windows, `bin/` on Linux/macOS).

The Phase 8 exit criterion this discharges: *a user on a fresh machine with only
documented prerequisites reaches a rendered trajectory using only these
commands.* The one clause it cannot close on this host — `star verify` in under
10 minutes **on a Pi 5** — is a hardware measurement deferred, fully prepared,
to the [pre-release checklist](release_checklist.md) (item 1, procedure
[`docs/perf/pi5_checklist.md`](perf/pi5_checklist.md)); see
[Verification](#4-verification-first-dx-5) below.

## Prerequisites

Exactly three, all standard:

- **Python ≥ 3.11** (the frontend needs stdlib `tomllib`; A-1). `python --version`.
- **A C++17 compiler** — GCC ≥ 9, Clang ≥ 10, or MSVC (Visual Studio 2019+).
  The native core builds from source during `pip install`.
- **CMake ≥ 3.26** and **git**. `cmake --version`, `git --version`.

No GPU, no HDF5, no SPICE, no MATLAB. The only mandatory runtime dependencies
(`numpy`, `matplotlib`, `jplephem`) install automatically with the wheel.

## 1. Clone

```sh
git clone https://github.com/JusHoya/star_reacher.git
cd star_reacher
```

## 2. Build and install into a fresh virtual environment

The native core builds from source here; allow a minute or two on first build.

```sh
python -m venv .venv
# Linux/macOS:
.venv/bin/pip install .
# Windows (PowerShell/Git Bash):
.venv/Scripts/pip install .
```

`pip install .` compiles the C++17 core (`star._core`), builds the wheel with
scikit-build-core, and installs the `star` console script. Everything after
this uses that installed `star` — activate the venv (`source .venv/bin/activate`
or `.venv\Scripts\activate`) or call `star` through the venv path as shown.

## 3. Run the mission (before touching anything else, verify)

The verification-first rule (DX-5): nobody uses the sim before `star verify`
passes locally. Do it first.

```sh
star verify --quick
```

It prints one line per check and ends in an unambiguous verdict:

```
VERIFY: tier quick (29 checks; every registered check runs in this tier)
V001 PASS two-body double-run SHA-256 bit-identity
...
V029 PASS P7 EC-2: MC ensemble stats match frozen golden (chi-square + A-D 99 %)
VERIFY: PASS (29/29)
```

Proceed only on `VERIFY: PASS`. (`--quick` is the < 60 s smoke tier; it
presently runs the identical check set as the full tier — the tier line says
so. Drop `--quick` to run the full suite; on x86-64 both take about 9 s.)

## 4. Propagate a mission and render its trajectory

Two commands take you from a mission file to a rendered 3D trajectory:

```sh
star run missions/twobody_leo.toml
star view out/twobody-leo/run.srlog
```

- `star run` validates and hashes the mission, propagates it with the compiled
  core, and writes `out/twobody-leo/` containing `run.srlog` (the binary log),
  `resolved_config.json`, and `meta.json`. It prints the resolved-config
  SHA-256 and the log SHA-256 — reruns are bit-identical, so the same inputs
  print the same hashes.
- `star view` reads only the log and writes `out/twobody-leo/run.html`: one
  self-contained WebGL playback file. **Open it in any browser** — it makes
  zero network requests, so it works fully offline. Scrub the timeline, play
  it back at 0.1×–1000×, switch among the four camera modes, toggle the axes
  triad / velocity / trail / groundtrack overlays. That HTML file *is* the
  rendered trajectory.

`star view` prints the decimation summary so the fidelity claim is checkable
without opening the file:

```
view stream: kept 257 of 54001 truth samples
decimation bound: 1917.17 m (= max(100 m, 0.01 % of the 1.91717e+07 m position span))
decimation measured max error: 482.811 m
wrote out/twobody-leo/run.html (829690 bytes)
```

That is the four-command path — `pip install .`, `star verify --quick`,
`star run`, `star view` — from a clean machine to a rendered 3D trajectory.

### Quicklook plots (optional)

`star plot` renders the FR-18 PNG set headless (no display needed). See the
[example gallery](../README.md#example-gallery) for representative output:

```sh
star plot out/twobody-leo/run.srlog          # writes out/twobody-leo/plots/*.png
```

A richer example — the scripted ascent exercises the atmosphere, propulsion,
and staging, so its plots include the force/torque budget and mass/thrust
panels:

```sh
star run missions/ascent_leo.toml
star plot out/ascent-leo/run.srlog
star view out/ascent-leo/run.srlog
```

## 5. Read the log from NumPy — without the simulator installed

The SRLOG loader is **pure NumPy plus the standard library**: it imports and
works with no compiled core, so an analysis or CI machine never needs a
compiler (FR-31 data-I/O guide). On such a machine you install only NumPy and
put `python/star_reacher` on the path (or `pip install .` if you also want the
CLI), then:

```python
from star_reacher import load

run = load("out/twobody-leo/run.srlog")

# The reproducibility anchor: the exact configuration that produced this file.
run.header["config_sha256"]

# Per-group channels are NumPy structured arrays (D-12, NumPy-first):
t = run.groups["truth"]["t_s"]        # (N,)   float64, strictly increasing
r = run.groups["truth"]["r_m"]        # (N, 3) float64, GCRF position [m]
v = run.groups["truth"]["v_mps"]      # (N, 3) float64, GCRF velocity [m/s]

# The event stream is a structured array of (t_s, code, detail):
run.events                            # e.g. run_start, staging, orbit-insertion

import numpy as np
speed = np.linalg.norm(v, axis=1)     # inertial speed per sample [m/s]
```

No pandas, no HDF5, no vendor reader. `run.to_pandas()` returns a DataFrame
when pandas happens to be installed (an optional extra), but the arrays above
are always available. For a machine-portable, pickle-free interchange (the ML
training format), `star export --npz out/twobody-leo/run.srlog` writes one NPZ
holding every group, the events, and the header; `--csv` and `--parquet` (the
latter behind the `pyarrow` extra) are there too. The format contracts live in
[`docs/formats/`](formats/).

## What you have now

- A verified install (`VERIFY: PASS`).
- A propagated mission with a bit-reproducible binary log bound to its
  resolved-config hash.
- A rendered, offline, self-contained 3D trajectory (`run.html`).
- The quicklook PNG set.
- A NumPy handle on every logged channel, with or without the compiler.

The everyday research loop from here is three commands (DX-4): run a baseline,
edit one value in a mission or vehicle TOML, run the variant, and overlay the
plots — `star plot a/run.srlog b/run.srlog` labels every curve with its
resolved-config hash so a plot can never be misattributed to the wrong edit.
