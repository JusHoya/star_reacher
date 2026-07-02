# star_reacher -- seeded from hypoCamp

Project-local memory for `star_reacher`, seeded from the hypoCamp second brain at
`C:/Users/hoyer/WorkSpace/hypoCamp`. It inherits the brain's constitution and public profile by
reference below, digests the standing conventions, and lists the
cross-project insights matched to this project's domains.

@C:/Users/hoyer/WorkSpace/hypoCamp/CLAUDE.md
@C:/Users/hoyer/WorkSpace/hypoCamp/wiki/user/profile.md

## Project overview

`star_reacher` is a small, deterministic, high-fidelity six-degree-of-freedom
space-mission simulator for launch vehicles, satellites, and lunar/Mars
missions -- a research instrument for mission analysis, GNC algorithm
development, and world-model / AI-ML spacecraft-navigation research. The design
principle is scientific fidelity at minimal size: every physical model carries a
first-principles derivation, explicit domain-of-validity bounds, and validation
evidence (golden vectors, analytic benchmarks, GMAT cross-checks), and the same
inputs on the same binary always produce bit-identical outputs. The authoritative
specification is [`PRD.md`](PRD.md) (32 functional requirements, 19 keyed
decisions D-1..D-19, an 8-phase roadmap); the public front door is
[`README.md`](README.md).

**Status (2026-07-02): specification baseline. No code implemented yet.** The
working tree holds the PRD, README, banner assets, this memory file, and
`.gitattributes` (`* text=auto` LF normalization). Implementation begins at
Phase 1. The license and repository visibility are an open decision (D-19) that
must be made before the first substantive public push -- a public commit is a
legal disclosure event (see [[public-commit-starts-patent-disclosure-clock]]).
Git remote: `https://github.com/JusHoya/star_reacher.git` (branch `main`).

**Stack (D-1).** A pure C++17 core (`star::`) with Eigen as the only mandatory
core dependency; pybind11 bindings; a Python >= 3.11 frontend (`star_reacher`);
CMake presets and scikit-build-core wheels. Boundary rule: everything inside the
deterministic time loop is C++ (it never parses text, touches the network, or
reads the clock); everything before t0 or after tf is Python (TOML
validation/canonicalization, Monte Carlo orchestration, loader/exporters,
plotting, HTML viewer, docs build, Gym/ONNX adapters). See
[[cpp-python-split-stack]] for the general pattern.

**Target commands -- one `star` CLI (D-4/FR-20), not yet implemented.** Recorded
here so the intended interface is stable across sessions; each lands in the phase
that earns it:

- `pip install .` -- build the native core + install the CLI (wheels via cibuildwheel)
- `star verify [--quick]` -- run the acceptance suite (`--quick` is a < 60 s smoke tier); ends in `VERIFY: PASS (N/N)` or `FAIL`
- `star run <mission.toml>` -- propagate a mission, emit `run.srlog` + `meta.json`
- `star plot | view | export | mc | consistency | data fetch | docs` -- quicklook plots, HTML 3D playback, CSV/NPZ/Parquet export, Monte Carlo sweeps, NEES/NIS gates, ephemeris fetch, and the LaTeX docs build

Execution follows the phased-contract method (see [[phased-contract-driven-development]]):
each phase is independently shippable, `/sprint`-executable, and gated on
red-team-checkable exit criteria; a physical model's math-library chapter and its
golden-vector tests land in the same phase as the model.

## Shared conventions

A faithful digest of the imported constitution and working-style rules
(full text is in the two `@import` files above -- these are not
new rules):

- **Neutral register in artifacts.** Every committed artifact -- code,
  docs, commit messages -- is neutral professional prose with zero
  persona tells. Interactive chatter may be informal; committed text is not.
- **Additive by default.** Prefer append or patch over rewrite; do not
  delete or wholesale-rewrite existing files without cause.
- **Cite or flag.** Every asserted fact cites a real source, or is
  explicitly flagged low-confidence. Never invent a citation or a link.
- **No placeholder stubs in committed files.** Do not commit TODO or
  placeholder scaffolding; land complete, working content.
- **WHY-not-WHAT comments.** Comment the reason a line exists, not a
  restatement of what the code already says.
- **Commit-and-push discipline.** Land work as ordinary, reversible
  commits; never run destructive git operations.

## Inherited insights (matched to domains: [gnc, navigation, astrodynamics])

Cross-project lessons carried in from the brain. Each links to the full
insight page; run `/recall <topic>` to pull the body on demand.

- [[unit-test-first-flight-software]] -- Write per-module golden-vector unit tests first for navigation and flight software, then gate acceptance on NEES/NIS consistency and Monte Carlo regression — build confidence bottom-up from local correctness to system evidence.  (matched tags: gnc)
- [[backtest-before-deploy]] -- Prove a predictive signal on held-out history — time-split, walk-forward, no lookahead, minimum trade count, then mandatory paper trading — and auto-pause it live when rolling accuracy degrades.  (general)
- [[beat-the-trivial-baseline]] -- A high validation metric can be a price echo; always benchmark against the trivial baseline feature and never trust a metric computed on corrupted labels.  (general)
- [[branch-per-session-parallel-agent-dev]] -- Run parallel agents on a branch-per-session, merge-to-main workflow so concurrent agents editing a shared repo do not collide.  (general)
- [[cap-agent-fanout-resource-exhaustion]] -- Bound concurrency and matrix size when fanning out agents — an unbounded red-team fan-out exhausted memory and crashed the machine.  (general)

## Relevant brain pages (mapped from the PRD)

Existing hypoCamp pages the PRD's design maps directly onto, beyond the
tag-matched insights above. Pull any with `/recall <topic>`:

- [[cpp-python-split-stack]] -- the C++/Eigen-core-behind-Python-frontend pattern with a versioned log contract and pybind11 bindings (D-1, architecture).
- [[phased-contract-driven-development]] -- numbered phases, each with an acceptance contract and red-team boundary (PRD section 8 roadmap; `/sprint`).
- [[numerical-integrators-rk4-rk78]] -- fixed-step RK4 vs. adaptive RKF7(8) and the accuracy/cost tradeoff (FR-11).
- [[nees-nis-filter-consistency]] -- NEES/NIS Monte Carlo consistency checks (FR-26 `star consistency`, the estimator acceptance instrument).
- [[error-state-ekf]], [[strapdown-ins-mechanization]], [[imu-error-modeling]] -- the navigation/sensor substrate for the Phase 6 GNC hooks (FR-23/FR-25).
- [[public-commit-starts-patent-disclosure-clock]] -- the first public commit is a legal disclosure event (D-19 license/visibility decision).
- [[verify-in-real-app-before-done]] -- drive the real app before calling a task done ("tests passed" is not "it works"; DX-5 verification-first onboarding).

## hypoCamp feedback loop

This project is wired to the hypoCamp second brain at
`C:\Users\hoyer\WorkSpace\hypoCamp`. Knowledge flows both ways — run these
commands from this directory as the work progresses:

- **Pull knowledge in** — `/recall <topic>`. Index-first, grep-second query of
  the brain (public tier by default; `--private` gates private material). Run it
  before solving something the brain may already cover.
- **Bank one lesson** — `/reflect <the lesson>`. Right after a notable success
  or failure, capture the single most valuable reusable lesson as a cited
  insight page in the brain. Lightweight; use it often.
- **Bank many lessons** — `/harvest [scope]`. Sweep this project for reusable
  lessons, preferences, decisions, and gotchas and bank them into the brain.
  Use it at milestones or at session end.
- **Keep the brain record current** — this project's page in the brain is
  `wiki/projects/star-reacher.md`. As the purpose and status firm up, update it (via
  `/harvest` or directly) so the catalog stays honest.

All four are additive and supervised: they propose before writing, run a
secret/OPSEC gate before transcribing anything, never modify or delete existing
`raw/` ground truth, and never run destructive git operations.
