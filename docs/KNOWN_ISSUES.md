# Known issues

Tracked defects and documented limitations, with the exit-criterion impact of
each stated plainly. Entries are removed when fixed (with the fixing commit
noted in the changelog history, not here).

## KNOWN-ISSUE-P4-1 — intermittent native fault on the high-volume by-source log path (mitigated)

**Symptom (original).** With the FR-16 `forces`/`mass`/`env` channel groups
enabled *and* a very large record count (order 10^5 records, order 100 MB of
log), the run intermittently aborted partway through with a native memory fault
(access violation / `0xC0000005`, occasionally surfacing as
`0xC0000409`), leaving a truncated `run.srlog` and no `meta.json`. Measured on
the build host at roughly 8 faults in 48 runs of the trans-lunar case
(`missions/tli.toml` with all groups at 1 Hz: ~455k records, ~211 MB).

**Investigation.** The propagation is deterministic — every full-groups run
that *completes* produces a bit-identical log — so the fault was confined to the
high-volume by-source write path, not the computed trajectory. The SRLOG writer
streams directly to the file with no unbounded in-memory buffer (peak RSS ~444
MB for a 211 MB log), and every record write is bounds-checked. The per-cycle
logging assembly was examined under four independent memory tools — Linux
AddressSanitizer at the full 211 MB volume, UndefinedBehaviorSanitizer
(including alignment), Valgrind memcheck with `--track-origins`, and an earlier
MSVC AddressSanitizer pass — and **none reported a memory-safety defect**
(Valgrind: zero errors, all ~180k allocations freed). No code-level buffer
overrun, use-after-free, or uninitialized read was found. The fault correlates
with high-frequency heap-allocation churn during the large-volume write; a
code-level root cause could not be isolated, and a contribution from build-host
instability cannot be excluded (this host's compiler intermittently faults with
the same access-violation code during compilation).

**Mitigation.** The per-source forces record now reuses a single buffer across
the whole run instead of allocating and freeing a fresh vector every logged
step (`cpp/src/run.cpp`). This removes ~455k per-cycle allocations on the
trans-lunar case and eliminated the fault across 45 consecutive full-groups runs
(versus ~17 % previously), with **byte-identical** log output (same SHA-256).
The change is a determinism-preserving optimization; it does not alter the
logged bytes.

**Residual caveat.** Because no code defect was isolated, the possibility of an
environmental (build-host) contribution remains. The mitigation removes the
observed symptom but is not proven to address a specific logic defect, since
none was found.

**Exit-criterion impact: none.** No Phase 4 exit criterion depends on the
by-source groups at high volume. EC-6 evaluates `missions/tli.toml` in its
committed configuration (truth records plus the SOI-transition event), which is
reliable and bit-reproducible.

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
