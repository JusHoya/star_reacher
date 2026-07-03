# Known issues

Tracked defects and documented limitations, with the exit-criterion impact of
each stated plainly. Entries are removed when fixed (with the fixing commit
noted in the changelog history, not here).

## KNOWN-ISSUE-P4-1 — memory corruption in the high-volume by-source log path

**Symptom.** Running a mission with the FR-16 `forces`/`mass`/`env` channel
groups enabled *and* a very large record count (order 10^5 records, order
100 MB of log) intermittently aborts partway through the run with a native
memory fault (observed exit codes `0xC0000409` STATUS_STACK_BUFFER_OVERRUN and
`0xC0000005` access violation), leaving a truncated `run.srlog` and no
`meta.json`. Reproduced on roughly 1 run in 4 for the trans-lunar case
(`missions/tli.toml` with all groups at 1 Hz: ~455k records, ~211 MB).

**Scope and severity.** The propagation itself is deterministic: every
full-groups run that *completes* produces a bit-identical log, so the fault is
confined to the high-volume by-source write path and does not affect the
computed trajectory. The `truth`-only path (no vehicle groups) is unaffected
and reliable at any length, and the shorter ascent mission
(`missions/ascent_leo.toml`, ~390 by-source records) exercises the FR-16 groups
reliably. This is not machine flakiness — the failure correlates specifically
with the large by-source log volume and never occurs on truth-only runs.

**Exit-criterion impact: none.** No Phase 4 exit criterion depends on the
by-source groups at high volume. EC-6 evaluates `missions/tli.toml` in its
committed `truth`-only configuration (truth records plus the SOI-transition
event), which is reliable and bit-reproducible.

**Workaround.** For long missions, leave `forces_rate_hz`/`mass_rate_hz`/
`env_rate_hz` at 0 (the committed `missions/tli.toml` does this), or lower the
group rates so the by-source record count stays modest.

**Status.** Under investigation. The SRLOG writer streams directly to the file
with no unbounded in-memory buffer, so the fault is being traced with a
sanitizer build of the vehicle run path's per-cycle logging assembly.

## KNOWN-ISSUE-P4-2 — FR-16 `thirdbody` force channel lumps the environment residual

The vehicle run path's `forces` group emits the sources `gravity`, `thirdbody`,
`aero`, `thrust`, and `gravgrad`. The `thirdbody` channel value is the full
non-central-gravity environment residual (central-body gravity subtracted from
the composed environment acceleration), not strictly the third-body term. For
every shipped mission this residual *equals* the third-body acceleration
(`missions/ascent_leo.toml` enables no third bodies; `missions/tli.toml` enables
Sun and Moon with no SRP or orbital drag), so the logged value is exact for what
ships. A future vehicle mission that enables environment SRP or orbital drag
would fold those into the `thirdbody` channel rather than emitting the separate
`srp`/`drag` sources named in `docs/formats/srlog_v1.md`. Per-source
decomposition of the environment terms in the vehicle path is deferred.

**Exit-criterion impact: none.** No Phase 4 exit criterion tests a per-source
environment force decomposition, and no shipped mission enables SRP or orbital
drag on the vehicle path.
