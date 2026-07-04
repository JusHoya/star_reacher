# Pi 5 performance results record

Measurement JSONs and their context entries produced by the manual
pre-release checklist `docs/perf/pi5_checklist.md`. Only numbers measured on
real Raspberry Pi 5 hardware belong here; GitHub-hosted ARM runner numbers
are proxies and live in the nightly workflow's artifacts instead.

Each qualification appends one entry in this format:

```
## vX.Y.Z — YYYY-MM-DD
- storage: <medium, model>
- cooling: <active cooler / case / ambient notes>
- os: <image name and version>
- runs: <file names of the three JSONs>
- verdict: PASS | FAIL (<failing metric(s), if any>)
```

No entries yet: no release has been qualified through the checklist since it
was introduced (Phase 5).
