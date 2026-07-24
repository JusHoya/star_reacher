# Release handoff — v0.8.0 (Phase 8 close)

**Status: prepared, NOT released.** The Phase 8 work is complete and verified on
the `phase-8-validation-report-release` branch to the full extent of the machine
it was executed on, but three exit-criterion clauses could not be closed there
because they need resources that host did not have (GMAT, Raspberry Pi 5
hardware) or an action that only the maintainer takes (pushing the public
release tag, a D-19 disclosure event). Each is shipped fully prepared under the
PRD section 9 valve and registered on [`release_checklist.md`](release_checklist.md).

This document is the task-oriented entry point: if you have one of the missing
resources, find your section below and run it. The register in
`release_checklist.md` is the status-of-record; this file tells you what to do.

Do **not** tag or announce v0.8.0 as released until the "Definition of done"
below is fully satisfied. Green CI on the branch is necessary but not
sufficient — it cannot exercise the deferred clauses.

## What is already closed (on any machine, verified in CI)

These need nothing from you and are gated in `.github/workflows/ci.yml`:

- Criterion 2 — the math-library and report PDFs build warning-clean via
  `star docs` on the CI TeX Live, the chapter-accretion lint is green, every
  code-cited equation label resolves, and every validation-table test ID exists
  (`docs` job; the two consecutive builds are byte-identical under
  `SOURCE_DATE_EPOCH`).
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

- [ ] **Handoff A** — the five new cross-tool cases are frozen against GMAT and
      their five gates in `tests/python/test_crosstool_frozen_truth.py` pass
      (no longer skip), each measured RMS within its PRD tolerance and recorded
      in `tests/golden/crosstool/manifest.toml`.
- [ ] **Handoff B** — the Pi 5 checklist is run and `star verify` completes in
      `< 10 min` on real Pi 5 silicon (plus the Phase 5/6 Pi 5 performance
      clauses that share item 1).
- [ ] Branch CI is green on all legs, and the first post-merge `nightly` run is
      green (`release_checklist.md` item 3).
- [ ] **Handoff C** — the `v0.8.0` tag is pushed and the `release` job builds
      the four-platform wheels and passes `verify --quick` on each.

Until Handoff A and B are discharged, this is a prepared release candidate, not
a release.

---

## Handoff A — the GMAT machine (criterion 1, checklist item 10)

**You need:** a host with **GMAT R2026a** (the exact build pinned in
`tests/golden/crosstool/manifest.toml`'s toolchain-provenance block), this
branch checked out, and the package installed (`pip install .`). You do **not**
need to fetch DE440 or generate any fixture: the DE440 lunar excerpt was
fetchable and was generated and committed on the Phase 8 host, so the lunar
cases run out of the box, and GMAT supplies its own ephemeris.

Everything else is already committed and committed *as-would-be-run*: the five
missions, the hand-authored GMAT `.script` replications, the degree-matched
GRGM/MRO `.cof` field inputs, the committed DE440 lunar excerpt
(`tests/golden/ephemeris/excerpt_de440s_lunar.sreph`, carrying the
`moon_librations` segment the lunar cases need), the deterministic generators,
the freeze driver, and the per-case provenance manifest entries carrying each
PRD tolerance verbatim. You produce only the truth-CSV bytes and the measured
RMS.

**Run, from the repository root:**

1. (Optional) Confirm the committed inputs are intact — both regenerate
   byte-identically and change nothing unless a source excerpt changed:

   ```sh
   python scripts/crosstool/gen_field_files_phase8.py   # Moon/Mars .cof fields
   python tests/golden/ephemeris/generate_lunar.py      # the lunar excerpt (already committed)
   ```

2. For each case, run GMAT on the committed script and freeze its truth CSV:

   ```sh
   python scripts/crosstool/run_gmat_phase8.py --case molniya
   python scripts/crosstool/run_gmat_phase8.py --case lunar_orbiter
   python scripts/crosstool/run_gmat_phase8.py --case mars_orbiter
   python scripts/crosstool/run_gmat_phase8.py --case translunar    # pass --arrival-s if the sim locates a different lunar-arrival epoch than the 455402 s default
   python scripts/crosstool/run_gmat_phase8.py --case mars_cruise
   ```

3. Confirm the five gates flip from skip to pass and record each measured RMS
   in the manifest's `pending` fields:

   ```sh
   python -m pytest tests/python/test_crosstool_frozen_truth.py -q
   ```

   Edit `tests/golden/crosstool/manifest.toml`, replacing each case's
   `date="pending"` / `SHA-256 pending` / measured-RMS placeholders with the
   frozen CSV's real values.

**Acceptance:** all five gates pass with position RMS inside the PRD tolerance —
trans-lunar `< 1 km` at lunar arrival; Mars cruise `< 100 km` at arrival SOI;
Molniya, lunar orbiter, Mars orbiter `< 100 m` over 7 days. The comparison
machinery those gates use is already proven correct independent of GMAT by
`test_rms_machinery_is_correct_on_sim_own_states` (the sim's own states as
pseudo-truth, RMS ≈ 5e-9 m), so a failure here is a real physics or
configuration disagreement, not a harness artifact.

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

**Do this last**, after Handoffs A and B are discharged and branch + nightly CI
are green. Pushing the tag is a public disclosure event (D-19); it is a
maintainer action, deliberately not performed by the phase work.

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
