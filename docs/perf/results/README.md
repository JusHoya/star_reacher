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
- verify_full_s: <wall seconds> / 600 s budget — PASS | FAIL
- runs: <file names of the three JSONs>
- verdict: PASS | FAIL (<failing metric(s), if any>)
- onnx: onnxruntime <version> — loop PASS | FAIL, run.srlog sha256 <hex>
- onnx_cross_platform_max_rel: <value> / 1e-9 bound — PASS | FAIL
```

The `verify_full_s`, `onnx` and `onnx_cross_platform_max_rel` fields are the
numbers the checklist produces outside the performance harness, and the entry
is their only home: a `perf_gate.py
measure` document carries the four gated FR-32 metrics plus the runner and
version identity and nothing else, so a result recorded only in the JSONs
would leave them in a shell transcript. `verify_full_s` is the timed
full-tier `star verify` of checklist step 3 (Phase 8 exit criterion 4);
`onnx` and `onnx_cross_platform_max_rel` are the closed-loop ONNX run of
step 8 and the x86-64-versus-aarch64 final-state comparison of step 9
(Phase 7 exit criterion 4). Step 9 also commits the Pi 5 leg's
`finalstate.json` here, beside the measurement JSONs, so the recorded
`max_rel` stays recomputable from committed artifacts.

No entries yet: no release has been qualified through the checklist since it
was introduced (Phase 5).
