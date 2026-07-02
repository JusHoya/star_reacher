# star_reacher -- seeded from hypoCamp

Project-local memory for `star_reacher`, seeded from the hypoCamp second brain at
`C:/Users/hoyer/WorkSpace/hypoCamp`. It inherits the brain's constitution and public profile by
reference below, digests the standing conventions, and lists the
cross-project insights matched to this project's domains.

@C:/Users/hoyer/WorkSpace/hypoCamp/CLAUDE.md
@C:/Users/hoyer/WorkSpace/hypoCamp/wiki/user/profile.md

## Project overview

_Greenfield project initialized on 2026-07-02._ As of this date the working
tree contains only `.gitattributes` (`* text=auto` LF normalization) and has no
build system, sources, or toolchain yet. The git remote is
`https://github.com/JusHoya/star_reacher.git` (branch `main`). Record the
purpose, stack, and the build / test / run commands here once the toolchain is
chosen.

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
