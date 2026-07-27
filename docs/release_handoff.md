# Release handoff — v0.8.0 (Phase 8 close)

**Status: prepared, NOT released.** The Phase 8 work is complete and verified on
the `phase-8-validation-report-release` branch to the full extent of the machine
it was executed on, but two exit-criterion clauses could not be closed there
because they need a resource that host did not have (Raspberry Pi 5 hardware) or
an action that only the maintainer takes (pushing the public release tag, a D-19
disclosure event). Each is shipped fully prepared under the PRD section 9 valve
and registered on [`release_checklist.md`](release_checklist.md).

This document is the task-oriented entry point: if you have the missing hardware
or the maintainer role, find your section below and run it. Handoff A is
discharged and its section is retained as the record of what was run. The
register in `release_checklist.md` is the status-of-record; this file tells you
what to do.

Do **not** tag or announce v0.8.0 as released until the "Definition of done"
below is fully satisfied. Green CI on the branch is necessary but not
sufficient — it cannot exercise the deferred clauses.

## What is already closed

These need nothing from you and are gated in `.github/workflows/ci.yml`:

- Criterion 1 — all five new cross-tool cases are frozen against GMAT and all
  five gates in `tests/python/test_crosstool_frozen_truth.py` pass, each
  measured residual inside its PRD tolerance and recorded in
  `tests/golden/crosstool/manifest.toml`. This was Handoff A, discharged
  2026-07-25; the record and the measured residuals are below. The Earth-Mars
  cruise gate measures at the end of the committed 7-day arc rather than at
  Mars-SOI arrival; that substituted epoch is registered as
  [`release_checklist.md`](release_checklist.md) item 12 and is a residual, not
  a release blocker.
- Criterion 2 — the math-library and report PDFs build warning-clean, the
  chapter-accretion lint is green, every code-cited equation label resolves,
  and every validation-table test ID exists. Warning-clean is enforced by
  `star docs` itself, so it holds on any host and not only in CI: `build_docs`
  scans each document's `.log` and fails on any LaTeX, Package, Class, or Font
  `Warning:` diagnostic, on an undefined reference or citation, and on a
  missing log. Overfull and underfull box reports are outside that definition
  and are counted and printed rather than gated. The `docs` job additionally
  builds both PDFs, fully cleans each document directory, rebuilds, and fails
  on any SHA-256 difference, which is the byte-identity clause;
  `SOURCE_DATE_EPOCH` is pinned to the HEAD commit time by `build_docs`.
- Criterion 3 — the report case-study figures regenerate bit-for-bit
  (`report-figures` job runs `scripts/figures/casestudy_figures.py --verify-repro`;
  `--check-committed` additionally proves a fresh render equals the committed
  PNGs on the pinned matplotlib/FreeType).
- Criterion 4, fresh-machine half — the README quickstart and
  [`walkthrough.md`](walkthrough.md) reach a rendered trajectory using README
  commands only, and `star verify` prints `VERIFY: PASS (29/29)` (measured ~9 s
  on x86-64; the `< 10 min on a Pi 5` half is Handoff B below).
- The two Phase 3 frozen cross-tool cases stay gated and passing (GMAT
  LEO-gravity RMS 0.0152 m, Orekit LEO-drag RMS 3.376 m).

## Definition of done for the v0.8.0 release

Tag and announce only when all of these hold:

- [x] **Handoff A** — the five new cross-tool cases are frozen against GMAT and
      their five gates in `tests/python/test_crosstool_frozen_truth.py` pass
      (no longer skip), each measured residual within its PRD tolerance and
      recorded in `tests/golden/crosstool/manifest.toml`. Discharged: four cases
      frozen and measured 2026-07-24, the trans-lunar case re-frozen and measured
      2026-07-25 at `|dr|` = 18.089824 m against its 1 km gate. See the Handoff A
      record below.
- [ ] **Handoff B** — the Pi 5 checklist is run and `star verify` completes in
      `< 10 min` on real Pi 5 silicon (plus the Phase 5/6 Pi 5 performance
      clauses that share item 1).
- [ ] Branch CI is green on all legs, and the first post-merge `nightly` run is
      green (`release_checklist.md` item 3).
- [ ] **Handoff C** — the `v0.8.0` tag is pushed and the `release` job builds
      the four-platform wheels and passes `verify --quick` on each.

Until Handoff B is discharged, this is a prepared release candidate, not a
release.

---

## Handoff A (discharged) — the GMAT machine (criterion 1, checklist item 10)

**Status: discharged 2026-07-25.** All five cross-tool cases are frozen against
GMAT and all five gates pass. Nothing in this section is outstanding for a
reader; it is kept as the record of what was run and with what result, and as
the procedure for anyone re-running the GMAT half.

**Host and toolchain.** The freeze ran on the maintainer laptop LAPTOP-HOYA with
the portable **GMAT R2026a** install pinned in
`tests/golden/crosstool/manifest.toml`'s toolchain-provenance block, this branch
checked out, and the package built from source. The gates were measured on the
same host as a 0.8.0 build (CPython 3.12.10 win_amd64). Neither step needed a
DE440 fetch or any fixture generation: the DE440 lunar excerpt was fetchable and
was generated and committed on the Phase 8 host, so the lunar cases run out of
the box, and GMAT supplies its own ephemeris.

Everything else was already committed, and committed *as-run*: the five
missions, the hand-authored GMAT `.script` replications, the degree-matched
GRGM/MRO `.cof` field inputs, the committed DE440 lunar excerpt
(`tests/golden/ephemeris/excerpt_de440s_lunar.sreph`, carrying the
`moon_librations` segment the lunar cases need), the deterministic generators,
the freeze driver, and the per-case provenance manifest entries carrying each
PRD tolerance verbatim. The freeze produced only the truth-CSV bytes; the build
host produced the measured residuals.

**What was run, from the repository root:**

1. The toolchain canary first — the same install regenerated the committed
   Phase 3 truth `truth_gmat_leo_gravity_8x8.csv` byte-identically (CSV SHA-256
   `181627b00cb380db958ddafc9423f03063c35fa0e743531335a8041c2dd1d895`, GMAT
   report SHA-256
   `9909743be59de0fb3270dc3b18386727ab36e813934a1510bab182c98a7df812`), which
   establishes that this install reproduces a truth frozen by an earlier one
   before any new truth is frozen with it.

2. The committed inputs regenerate byte-identically and change nothing unless a
   source excerpt changed; a re-runner can confirm that first:

   ```sh
   python scripts/crosstool/gen_field_files_phase8.py   # Moon/Mars .cof fields
   python tests/golden/ephemeris/generate_lunar.py      # the lunar excerpt (already committed)
   ```

3. The trans-lunar re-freeze, the last of the five. Molniya (position RMS
   0.160317 m), lunar orbiter (7.232688 m), Mars orbiter (79.478630 m, the
   thinnest margin at ~1.26x), and Mars cruise (`|dr|` 952.550 m at arc end) were
   closed by the 2026-07-24 laptop freeze plus the build-host measurement. The
   trans-lunar case needed a second freeze because its first one was
   invalidated: the original `.script` started from the tli t = 353 s truth
   state, which is mid-burn (the cutoff is commanded at 353 s, but the delivered
   thrust level is zero only from 354 s under the per-step spool discipline on
   the 1 s grid), so the replicated coast was ~15.3 m/s low in energy and missed
   lunar arrival by 75,189 km at matched epochs (the gate printed 75,212 km; the
   extra ~23 km came from a second defect fixed alongside — the superseded test
   compared at an epoch 353 s off, worth ~45 km by itself). The simulator's own
   ballistic coast of that same mid-burn state reproduced the invalidated GMAT
   arrival state to 0.022 km — the tools agreed; the initial state was the
   fault. The corrected script (t = 354 s burnout state, epoch 12:05:54Z,
   arrival span 455401 s) is committed, and the re-freeze command was exactly:

   ```sh
   python scripts/crosstool/run_gmat_phase8.py --case translunar
   ```

   No `--arrival-s` override was used; the committed 455401.0 s default stands.
   The source script `gmat_translunar.script` SHA-256
   `7b2ae9a5b89746b65eb49d81571c818785fbe3471a26c9055c0ff03680ad376b` is
   unchanged and still matches the pin the manifest already carries. The frozen
   output `truth_gmat_translunar.csv` is 581 bytes, one arrival row, SHA-256
   `7a3cca16b89b7fb48be460321b734a378c6760155ee66eda80cba2d27a718c67`; the GMAT
   report SHA-256 is
   `b74fbdb58b01743c4142cbc77a2702a9e8c51f4030a696e3a6fb406b208550be`. GMAT's
   last reported row is `ElapsedSecs` = 455401.000000129 s, that is, the
   requested 455401 s span reproduced to 1.29e-7 s; the CSV stamps the nominal
   455401.0.

4. In the same session `molniya` and `mars_cruise` were regenerated and came
   back byte-identical to their committed CSVs (`5341e53e…` and `9892f24d…`).
   `gmat_lunar_orbiter.script` and `gmat_mars_orbiter.script` were not re-run,
   for the path reason under "gotchas" below; their committed truth is
   unaffected.

5. The gates, on the 0.8.0 build host:

   ```sh
   python -m pytest tests/python/test_crosstool_frozen_truth.py -q
   ```

   8 passed. The trans-lunar gate flipped from skip to pass, and the
   `truth_gmat_translunar.csv` `[[file]]` entry — CSV SHA-256, date, and the
   measured arrival `|dr|` — now stands in `tests/golden/crosstool/manifest.toml`
   in place of the pending-re-freeze deferral note, which is retained there as
   resolved history.

**Measured results.** All five gates pass with the residual inside the PRD
tolerance:

- `XTOOL-MOLNIYA-GMAT` — position RMS 0.160317 m, tolerance `< 100 m` over 7 days.
- `XTOOL-LUNAR-GMAT` — position RMS 7.232688 m, tolerance `< 100 m` over 7 days.
- `XTOOL-MARS-ORBITER-GMAT` — position RMS 79.478630 m, tolerance `< 100 m` over
  7 days.
- `XTOOL-MARS-CRUISE-GMAT` — `|dr|` 952.550 m at arc end, tolerance `< 100 km` at
  arrival SOI.
- `XTOOL-TRANSLUNAR-GMAT` — `|dr|` 18.089824 m at lunar arrival, tolerance
  `< 1 km`.

For the trans-lunar case the velocity difference at the same epoch is
`|dv|` = 5.379870e-05 m/s and the position difference components are
(-3.4141, -14.7677, -9.8742) m. The comparison epoch is mission time 455755.0 s
exactly (455401 s of CSV elapsed time plus the 354 s burnout offset); the tli
truth log carries a row exactly there, so the gate's epoch assertion holds, and
`tli.toml`'s `duration_s = 600000.0` covers it. Both tools place the spacecraft
~397,665 km from Earth at that epoch — simulator 397664.930 km, GMAT
397664.947 km.

The whole suite re-run on this host also reproduced the four previously
desktop-measured Phase 8 values exactly, which independently confirms cross-host
bit-determinism, together with the two Phase 3 gates (0.015243 m against GMAT,
3.376229 m against Orekit).

**Acceptance: met.** All five gates pass, each residual inside its PRD
tolerance, with the thinnest margin at the Mars orbiter (79.478630 m against
100 m, ~1.26x) and the trans-lunar case at ~55x margin. The comparison machinery
those gates use is proven correct independent of GMAT by
`test_rms_machinery_is_correct_on_sim_own_states` (the sim's own states as
pseudo-truth, position RMS 4.618e-09 m), so these residuals are a real
tool-versus-tool difference, not a harness artifact.

**Planetary-ephemeris provenance — a correction recorded 2026-07-25.** The four
Phase 8 `.script` headers name a DE440 point-mass and lunar-orientation source,
as did `tests/golden/crosstool/README.md` before it was corrected in the same
change. That is not what GMAT used. No Phase 8
script sets `SolarSystem.EphemerisSource`, so the run took the install default,
and the GMAT run log records the planetary source as DE405, loading
`data/planetary_ephem/spk/DE405AllPlanets.bsp`; the pinned install ships only
`leDE1941.405`, `leDE1900.421`, and `leDE18002100.424`, so DE440 is not
selectable on it. The `.script` files are committed as-run artifacts whose
SHA-256 pins the manifest already carries and from whose exact bytes the frozen
truth was generated, so they are not edited; the correction is recorded
additively in the manifest prose, the crosstool README, the report, and the
release checklist. The simulator side reads the committed DE440
excerpt, so the two tools differ in planetary ephemeris — an irreducible tool
difference of exactly the kind exit criterion 1 exists to measure, alongside the
already-recorded FK5/IAU-76-versus-CIO frame-chain difference. It is empirically
bounded below every gate by the five residuals above, and most tightly by the
trans-lunar case (18.089824 m against 1 km), which is the most
lunar-ephemeris-sensitive case in the set.

**Two gotchas if you re-run the GMAT half.** First, `run_gmat_phase8.py`'s
`write_startup()` rewrites the SHA-pinned `gmat_startup_zeroeop.txt` with the
current checkout's absolute EOP path — restore that file after a run. Second,
`gmat_lunar_orbiter.script` and `gmat_mars_orbiter.script` run only from the main
checkout path, because their `.cof` field paths are absolute as-run under the
D-15 convention; that is why they were not regenerated from the worktree used on
2026-07-25, and it is a property of the as-run convention, not a defect.

The illustrative LRO case (`missions/lro_illustrative.toml`) is **report-only,
not a gate** — the real spacecraft's maneuver and SRP history is unmodeled — so
it carries no tolerance and nothing needs freezing for it.

## Handoff B — the Raspberry Pi 5 (criterion 4 timing, checklist item 1)

**You need:** Raspberry Pi 5 (8 GB) hardware per the baseline in
[`perf/pi5_checklist.md`](perf/pi5_checklist.md).

Run that checklist end to end. Its step 3 is the criterion-4 timing clause
(`star verify` under 10 min on Pi 5 silicon); steps 4, 6, and 7 carry the Phase
5/6 Pi 5 performance and viewer/plot clauses that share item 1. Record results
under `docs/perf/results/`. The nightly `ubuntu-24.04-arm` leg is only a proxy
and never counts as a Pi 5 measurement.

## Handoff C — tag and release (criterion 5, checklist item 11)

**Do this last**, after Handoff B is discharged (Handoff A already is) and
branch + nightly CI are green. Pushing the tag is a public disclosure event
(D-19); it is a maintainer action, deliberately not performed by the phase work.

```sh
git tag -a v0.8.0 -m "star_reacher v0.8.0"
git push origin v0.8.0
```

The tag triggers the `release` job in `.github/workflows/ci.yml`: it builds the
four-platform wheels (manylinux x86-64/aarch64, macOS arm64, Windows x64) with
cibuildwheel and runs `star verify --quick` in an isolated venv per wheel,
uploading them as `wheels-<os>` artifacts for attachment to the GitHub release.

**Acceptance:** the `release` job is green on all four legs. Attach the uploaded
wheels to the GitHub release for v0.8.0.
