# Phase 7 red-team — exit-criteria adversarial review

Branch `main` at `09ea479`. Independent, execution-driven adversarial review of
the four Phase 7 exit criteria (PRD.md line 228) before phase close. The
verifier did not implement any of this code. Every criterion was defaulted to
NOT MET and promoted only on evidence the verifier personally ran against a
working editable install of the compiled core (`star_reacher._core`,
`core_version 0.6.0`, `core_git_hash 09ea479`). Unlike the Phase 6 review, this
review compiled nothing new but **executed the shipped binary throughout**: the
`star mc` engine, `star run`, `star verify`, the regression gate, the Gym
adapter under `gymnasium.utils.env_checker.check_env`, and the ONNX controller
in a full closed loop.

**Constraints honored.** No file other than this report was written into the
repository, and no state-changing git command was run. All working outputs were
written under a scratch directory. Every number below was observed, not
recited.

**Environment.** CPython 3.12.1, native Windows x86-64; gymnasium 1.3.0,
onnxruntime 1.27.0, numpy present; `onnx` and `scikit-learn` absent (correct —
they are generation-time only). The full Python suite is **1126 passed, 3
skipped** (the three skips are the pandas/pyarrow optional-export extras, not
Phase 7 surface). `star verify --quick` and full `star verify` both print
`VERIFY: PASS (29/29)` in ~8.5 s with V028 and V029 present.

---

## 1. Per-criterion verdict table

| # | Subject | Verdict | Strongest evidence personally observed |
|---|---------|---------|----------------------------------------|
| 1 | 256-run/8-worker LHS sweep 256/256 success; any entry re-runs to its logged SHA-256 | **MET** | Live sweep 256/256, 256 distinct hashes; index 137 and 42 re-executed via `star run` reproduce their exact `log_sha256`; wrong seed and dropped-override both flip the hash; per-run seeds equal `core.splitmix64_stream(20260723,256)` exactly; 3 invocations bit-identical |
| 2 | MC ensemble stats match frozen golden within chi-square/A-D 99 %; two-key golden path | **MET** | Frozen 128-run ensemble: S=127.000 in [90.543,172.957], A2=1.408 p=0.200, reproduces golden mean/std bit-exactly; +0.5σ shift fails A-D (p=4.7e-6), ×1.3 fails BOTH; byte-flip in a golden fails `check_golden_manifests.py`; A-D anchors match Marsaglia (adinf(1)=0.642714, adinf(2)=0.908164), SciPy-free |
| 3 | `check_env` passes on `SpaceEnv`; Gym seeding is exactly core seeding | **MET** | `check_env(env, skip_render_check=False)` returns without assertion; env-seeded episode log sha256 == bare-Sim episode log (95aded33…) for same seed+command; seed reaches `Sim.reset(seed=S)` unchanged as a `run.seed` override; core has zero gymnasium import; `[rl]` extra not in base deps |
| 4 | ONNX MLP from an external framework closes the loop on x86-64 (Pi 5 + cross-platform deferred) | **MET (x86-64 clause); Pi 5 + cross-platform HONESTLY DEFERRED** | Closed loop settles 10.0°→0.1709°, no NaN, reaches `run_end`, bit-identical reruns; **logged torque equals ONNX inference to exactly 0.0 over all 601 cycles while differing from the PD law by 4.37e-4 N·m**; broken model path and absent onnxruntime both ERROR (no fallback); genuine Gemm→Relu→Gemm→Relu→Gemm graph; deferral fully prepared at release_checklist item 9 and recorded inline in PRD |

**Overall: all four criteria MET. Phase 7 closes honestly.** The only defects
found are cosmetic/documentary (Section 2), none phase-blocking. The Pi 5 and
x86-64-vs-aarch64 cross-platform clauses of criterion 4 are **deferred, not
waived**: the Section 9 valve is satisfied on all three obligations.

---

## 2. Confirmed defects, ranked by consequence

All defects found are low or cosmetic. None blocks phase close.

### D1 — `star verify` CLI help string is stale ("V001-V027") (cosmetic)

`python/star_reacher/cli.py:110` sets the `verify` subcommand help to
`"run the acceptance check suite (V001-V027)"`. The suite actually runs
V001-V029: `star verify` and `star verify --quick` both print
`VERIFY: PASS (29/29)` with `V028 PASS` and `V029 PASS` in the roster. This is a
help-text-only staleness — the runner and the `_CHECKS` table are correct, the
count is right, and both new checks execute and are demonstrated able to fail.
The docstring at `verify.py:24-33` already describes V028-V029 accurately, so
only the one-line `cli.py` help string lags. Remedy: change the help string to
"V001-V029" (one-line edit, no behaviour change).

### D2 — `integer=true` LHS dimension is not strictly one-per-bin after rounding (documented behaviour, not a defect in the sampler)

The reference sweep's `mission.duration_s` parameter carries `integer = true`.
Its underlying Latin hypercube is genuinely one-sample-per-stratum (verified
directly: pre-rounding, both dimensions' 256 stratum bins equal `range(256)`
exactly), but a 3600 s range over 256 strata gives a 14.06 s stratum width, and
rounding each sample to whole seconds moves 6 of 256 samples into an adjacent
bin (6 dup bins, 6 empties). This is the honest and documented consequence of
the `integer` flag (the sweep spec's own comment explains the whole-second
requirement), not a broken sampler: the continuous `spacecraft.mass_kg`
dimension is perfectly one-per-bin. Materiality: none — the sweep is still
deterministic, reproducible, and dispersed; the criterion says "LHS sweep," and
the sampler is a true Latin hypercube whose integer projection is a stated
rounding, not an unstratified draw. Recorded for completeness only.

**No other defects.** In particular, no silent fallback, no vacuous gate, no
overclaim in README or PRD was found. The README's Phase 7 row and the
`Status:` banner correctly scope the ONNX clause to "on x86-64" and defer Pi 5.

---

## 3. Per-criterion evidence

### Criterion 1 — MET

Command: `star mc missions/leo_gravity_8x8_sweep.toml --workers 8 --outdir <s> --force`.

- **256/256 success, 256 distinct log hashes.** Manifest `runs` length 256, all
  `status == "success"`, `len(set(log_sha256)) == 256`. Each run wrote a real
  546 KB `run.srlog` plus `meta.json` and `resolved_config.json`. No
  BrokenProcessPool observed via the CLI path (the pool only breaks under a
  heredoc-`__main__` re-import, which is a test-harness artifact, not an engine
  fault; the CLI and script-file paths run clean).
- **Re-execution reproduces the logged hash.** Non-trivial entries 137
  (`c59d5c8f…`) and 42 (`683e031c…`) re-executed via
  `star run missions/leo_gravity_8x8.toml --seed <s> --set mission.duration_s=<v> --set spacecraft.mass_kg=<v> -o <d> --force`
  using only the manifest's recorded seed and overrides — both matched their
  `log_sha256` exactly.
- **Adversarial (a) wrong seed:** entry 137 with `seed+1` → hash DIFFERS.
- **Adversarial (b) dropped override:** entry 137 omitting either
  `spacecraft.mass_kg` or `mission.duration_s` → hash DIFFERS in both cases.
- **Adversarial (c) SplitMix64:** `core.splitmix64_stream(20260723, 256)` equals
  the manifest's 256 per-run seeds in index order, element-for-element.
- **Adversarial (d) genuine LHS:** `spacecraft.mass_kg` is one-sample-per-bin
  over 256 equal-probability strata; `mission.duration_s` is one-per-stratum
  pre-rounding (see D2).
- **Adversarial (e) FR-27 fields:** manifest carries `schema_version=1`,
  `binary.core_version/core_git_hash/binary_sha256` (64-hex),
  `sweep.master_seed/method/n_runs/parameters/sweep_spec_sha256`, and per-run
  `seed/overrides/config_sha256/log_sha256/status`. `core_git_hash` = `09ea479`
  = HEAD.
- **Determinism:** three full sweep invocations produced bit-identical
  `(index, seed, config_sha256, log_sha256)` tuples.
- **Falsifiability:** an injected bad override (`mass_kg < 0`) yields
  `status: failed` per run and `star mc` exit code 1 — a partial sweep cannot
  masquerade as success.
- **CI logic:** the ci.yml criterion-1 gate step's exact Python (index 42,
  `repr(float)` override literals) was reproduced locally and matched
  (`683e031c…`), so the committed CI step is runnable and correct.

### Criterion 2 — MET

- **Pass point (live).** Regenerated the 128-run `mc_regression_sweep.toml`
  ensemble and gated it: chi-square S = 127.0000 inside [90.5431, 172.9575],
  Anderson-Darling A2 = 1.407935 with p = 0.200072, both green; the fresh
  ensemble reproduces the golden `mean` and `std` bit-exactly (`float.hex()`
  equality).
- **Both gates able to fail (live mutations).** +0.5σ shift → chi-square stays
  green but A-D fails (A2=16.4, p=4.69e-6); ×1.3 scale → BOTH fail (S=214.63 >
  172.96 upper; A-D p=1.8e-4); a wrong golden (mean+1σ) → both fail. This
  confirms the design rationale: A-D catches a location shift chi-square
  tolerates.
- **Anderson-Darling correctness (D-12 SciPy-free).** `adinf(1)=0.642714`,
  `adinf(2)=0.908164` match Marsaglia & Marsaglia (2004) Table 1 to 5e-4;
  `ad_cdf(3.878, n→∞)=0.98999…` reproduces the classic 1 % critical value; the
  gate uses the 99 % bound (reject p<0.01); `scipy` is not in `sys.modules`
  after import.
- **Two-key (a) clean tree:** `scripts/check_golden_manifests.py` → OK, 22
  manifests checked.
- **Two-key (b) byte flip:** flipping one byte of a temp copy of
  `energy_stats.toml` makes the checker FAIL with the on-disk-vs-recorded SHA-256
  mismatch message (exit 1).
- **Two-key (c) diff-summary + apply:** `scripts/golden_update.py` (no `--apply`)
  on the clean tree prints "no change" (exit 0); it emits a stat-level unified
  diff and refuses to write without `--apply`, and `_write_manifest` updates the
  recorded `values_sha256` together with the value. `test_golden_two_key.py`,
  `test_mc_regression.py`, `test_anderson.py`, `test_mc.py`, `test_mc_seed.py`:
  60 passed.
- **V029 re-measures the mutation.** `verify.py::_check_v029` runs a
  self-contained 64-run J2 ensemble, gates it at 99 %, pins the pass-point stats
  (S=63.0, A2=0.710), and **re-runs the +0.5σ and ×1.4 mutations on the live
  ensemble**, raising `CheckFailure` if either gate stops flipping. It is not a
  static assertion.

### Criterion 3 — MET

- **check_env (full battery).** `check_env(env, skip_render_check=False)` on a
  `SpaceEnv` over `missions/leo_attitude_rl.toml` returns with no assertion. The
  one `UserWarning` ("no spec, alternative render modes not tested") is a benign
  Gymnasium note, not a failure; the env declares `render_modes: []` and
  `render_mode=None`, so the render check is a legitimate no-op pass.
- **Seeding identity (byte-exact).** A `SpaceEnv` episode under a fixed
  zero-torque action produces a `run.srlog` whose sha256 (95aded33…) equals a
  bare `Sim` episode of the same mission and seed stepped with the identical
  command `{torque_b_nm:[0,0,0], valid:True}`. Same seed twice → identical;
  different seeds → different. Reading `space_env.py::reset` and
  `sim.py::reset`: `super().reset(seed=seed)` seeds gym's `np_random` (which
  check_env verifies is set) but does not feed the core; the SAME integer is
  passed to `Sim.reset(seed=S)`, which applies it as a `run.seed` override
  through the identical path a batch `star run --seed` uses — not hashed, not
  sub-seeded, not routed through gym's RNG.
- **No hard-coded semantics.** `SpaceEnv.__init__` takes `observation_space`,
  `action_space`, `observation`, `action`, `reward` from the caller; the shipped
  reference defaults live in `star_reacher.gym.defaults`, outside the class. The
  reward callable is handed truth; the observation callable never is.
- **Core is Gym-agnostic.** `grep -niE 'gymnasium'` over `cpp/`, `bindings/`,
  `python/star_reacher/sim.py` → nothing. `[project.optional-dependencies]` has
  `rl = ["gymnasium>=1.0"]`; base `dependencies` is `numpy, jplephem,
  matplotlib` only. `test_space_env.py`: 6 passed.

### Criterion 4 — MET (x86-64); Pi 5 + cross-platform deferred

Command:
`star run missions/leo_attitude_onnx.toml --gnc-plugin examples/onnx_gnc_plugin.py -o <d> --force`.

- **Closes the loop.** 600 steps, 601 truth records, reaches `run_end`. Final
  attitude error **0.1709°** from a 10.0° opening error (matches the PRD's
  claimed 0.171°); no NaN in q, r, or torque; terminal command torque within the
  0.05 N·m saturation. Two runs bit-identical (`5373a5c1…`).
- **ONNX is genuinely in the loop (decisive).** Rebuilding the six-input feature
  at every cycle from the logged `nav.est` state estimate and the commanded
  attitude, and running it through the committed `pd_mlp.onnx` via onnxruntime,
  reproduces the logged `gnc.cmd.tau_b_nm` to **exactly 0.0 over all 601
  cycles**, while the analytic PD law over the same features differs by up to
  **4.37e-4 N·m**. The loop is running the ONNX model, not a coincidentally
  similar PD law.
- **No silent fallback (a).** Pointing the plugin at a nonexistent model file
  makes the run ERROR with `RuntimeError: OnnxGnc could not load the ONNX model
  '…NONEXISTENT.onnx': NoSuchFile`. Blocking the `onnxruntime` import makes
  `_import_onnxruntime()` raise the actionable `ImportError` ("install the [ml]
  extra"). Neither substitutes a different controller.
- **Genuine ONNX MLP (a).** onnxruntime loads input `[None,6]` → output
  `[None,3]`; the graph bytes carry 6 `Gemm` and 4 `Relu` op strings (a
  Gemm→Relu→Gemm→Relu→Gemm perceptron), 0 `MatMul`. Inference is bit-identical
  across 20 repeated calls on a fixed input.
- **Provenance (b).** `tests/golden/onnx/manifest.toml` cites scikit-learn 1.9.0
  `MLPRegressor` fitted to the PD law, standardization folded into the weights,
  exported via onnx.helper (opset 18, IR 9), with pinned framework versions and
  a recorded byte-reproducibility claim; `generate.py` is committed.
- **Determinism single-thread (c).** onnxruntime configured
  `intra_op=inter_op=1`, `ORT_SEQUENTIAL`; verified bit-identical repeated
  inference. `test_onnx_gnc.py`: 6 passed.
- **`[ml]` extra (d).** `ml = ["onnxruntime>=1.18"]` in optional-dependencies,
  not base deps.

**Deferral honesty.** `docs/release_checklist.md` item 9 ("Phase 7 criterion 4
— ONNX loop closure on Pi 5 and the ARM cross-platform final state") names the
exact deferred clause, states the x86-64 clause is met and CI-gated, gives a
concrete procedure (install `[ml]` on aarch64, run the mission, `extract` +
`measure --bound 1e-9` + `gate --bound 1e-9`), records status "pending — no
aarch64 hardware," and notes the open empirical question (whether onnxruntime
CPU inference is bit-reproducible across x86-64/aarch64 within the D-10 bound).
The PRD Phase 7 entry (line 230) records the same deferral inline. The
`scripts/cross_platform_divergence.py extract` step ran successfully on the two
x86-64 ONNX runs and the format round-trips (the pytest test measures
`max_pairwise_divergence == 0.0`). All three Section 9 obligations (fully
prepared, registered on the checklist, recorded inline) are met — this is a
disclosed deferral, not a silent waiver.

---

## 4. Refuted concerns

Concerns raised in the review brief that were investigated and did not hold:

- **"A <1 s sweep is too fast to be real."** Refuted. The 256 runs are short
  RKF78 propagations (a fraction of one LEO orbit) fanned across 8 worker
  processes; the parent `time` shows near-zero user/sys because the CPU work is
  in the children. Real 546 KB `run.srlog` files were written per run, and the
  entries re-execute to their logged hashes — the work is genuine.
- **"BrokenProcessPool means the sweep engine is flaky."** Refuted in isolation.
  The pool break reproduces ONLY when the driver script is fed via a stdin
  heredoc (children cannot re-import `<stdin>` as `__main__`). Via the CLI
  (`star mc`) and via a real `.py` script file the pool runs clean every time,
  across 5+ invocations, on a quiet machine.
- **"The ONNX loop might silently fall back to the built-in PD law."** Refuted
  decisively: logged torque equals ONNX inference to 0.0 and differs from PD by
  4.37e-4 N·m; breaking the model path or the runtime import errors loudly.
- **"Gym seeding might be routed through gym's RNG."** Refuted by source read and
  byte-identity: the integer seed reaches `Sim.reset` unchanged and the two logs
  hash-match.

---

## 5. What this red-team did and did not establish

**Established (by execution).** All four criteria are met on x86-64 by gates
demonstrated able to fail. Every headline number in the PRD Phase 7 entry that
could be checked on this platform was reproduced: 256/256, distinct hashes,
per-run SplitMix64 seeds, S=127.0/A2=1.408 pass point, the mutation reddening,
0.171° ONNX settling, bit-identical reruns, and the two-key rejection. The
deferral is honest.

**Not established (out of platform reach).** The literal Pi 5 run and the
x86-64-vs-aarch64 cross-platform final-state comparison at the D-10 1e-9 bound
cannot be executed without aarch64 hardware and were not. They are correctly
scoped as deferred, fully prepared, on release_checklist item 9; the verifier
confirmed the item carries the mission, model, plugin, and extract/measure/gate
tooling, but cannot confirm the empirical result the item exists to resolve.
No C++ was recompiled; the compiled core used is the committed editable build at
`09ea479`.

## 6. Judgement: should Phase 7 close?

**Yes.** All four exit criteria are met and gated by checks the verifier
personally drove and personally mutated to red. The x86-64 half of criterion 4
is genuinely closed (ONNX bit-exactly in the loop), and its hardware-bound half
is deferred through the Section 9 valve on all three obligations rather than
waived. The two defects found are a stale CLI help string (D1) and a documented
integer-LHS rounding note (D2), neither of which affects any gate. Recommended
before close, non-blocking: fix the `cli.py:110` help string to "V001-V029".
