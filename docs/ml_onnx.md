# Learned models in the loop: `star_reacher.ml.OnnxGnc` (FR-28)

`star_reacher.ml.OnnxGnc` is the one sanctioned path in v1 for running a
trained model inside the GNC control loop. It is an ONNX network executed by
[onnxruntime](https://onnxruntime.ai/) on the CPU, swapped into a GNC chain
slot through the FR-25 plugin mechanism, so a mission flies it with no
recompilation and a checkout without the model runtime is unaffected.

This document is the operator- and author-facing contract. The determinism
rules it depends on live in `star_reacher.sim` (the GNC plugin contract) and
`python/star_reacher/plugin.py` (the loader); this file adds only what is
specific to a learned model.

## Why ONNX and onnxruntime

* **Portable across frameworks.** ONNX is an interchange format, so a model
  trained in any framework that can export ONNX runs here without that
  framework being a dependency of `star_reacher`.
* **Runs on a Pi 5.** onnxruntime ships CPU wheels for aarch64, so the same
  model file flies on x86-64 and on the Raspberry Pi 5 (Phase 7 exit
  criterion 4). No GPU, no accelerator, no platform-specific build.
* **Deterministic when pinned.** Single-threaded, sequential CPU inference on
  a fixed model and a fixed input is bit-reproducible, which is what lets a
  learned controller live inside the D-10 deterministic time loop.

onnxruntime is the *only* runtime dependency the learned-model path adds; it
is gated behind the `[ml]` extra (`pip install star_reacher[ml]`). The `onnx`
package and any training framework are needed only to *produce* a model, never
to fly one.

## Selecting an ONNX controller in a mission

The adapter is a plugin component, so a mission selects it in the `python:`
namespace and the run supplies the plugin file explicitly:

```toml
[gnc.control]
component = "python:onnx_gnc"
tau_max_nm = [0.05, 0.05, 0.05]
```

```
star run missions/leo_attitude_onnx.toml \
    --gnc-plugin examples/onnx_gnc_plugin.py
```

Running without `--gnc-plugin` is an error that names the control slot: the
mission declares its dependency in the component name, exactly as any other
plugin mission does.

## Where the model path comes from

A `GncComponentCfg` carries only a component name and numeric `scalars` and
`vectors` — there is no string field a mission could use to name a model file,
and that is deliberate: a mission TOML must never be able to cause a file to be
loaded. The model path is therefore supplied by the **plugin factory**, in
Python, on the `--gnc-plugin` file the operator named explicitly:

```python
# examples/onnx_gnc_plugin.py
from pathlib import Path
from star_reacher.ml import make_onnx_gnc_factory

MODEL_PATH = Path(__file__).resolve().parents[1] / "tests" / "golden" / "onnx" / "pd_mlp.onnx"
STAR_GNC_COMPONENTS = {"onnx_gnc": make_onnx_gnc_factory(MODEL_PATH)}
```

Resolving the path relative to the plugin file (not the working directory)
makes the run launchable from any cwd.

## The signal contract of the shipped model

The default input builder and output applier match the built-in `pd_attitude`
law's signals, so a learned controller is directly comparable to the analytic
one it approximates:

| | quantity | shape |
|---|---|---|
| **input** | sign-corrected attitude-error quaternion vector part `s·dq[1:4]`, then body-rate error `we` | 6 |
| **output** | body torque (pre-saturation) | 3 |

`s = sign(dq0)` and `dq = q_cmd* ⊗ q_est`, `we = w_est − C(dq)·w_cmd` (the
eq:gnc:werr quantity), computed by the core's own rotation kernel so the
features are identical to the built-in's. After inference the adapter clamps
the torque per axis to the mission's `tau_max_nm` (eq:gnc:sat) and passes the
command attitude and rate through. An invalid estimate or absent command is a
hold, exactly as the built-in does — not an instruction to apply zero torque.

Both hooks are overridable (`make_onnx_gnc_factory(..., input_builder=...,
output_applier=...)`) for a differently shaped model, but the shipped
`tests/golden/onnx/pd_mlp.onnx` uses exactly this contract.

## Determinism inside the time loop

`OnnxGnc` runs inside the deterministic time loop, so the full
`star_reacher.sim` plugin contract applies: no clock, no I/O beyond the one
model load, no unseeded randomness, no iteration over unordered containers, no
mutable global state. Two properties make onnxruntime safe here:

* **The model is loaded once, in `init`, never in `update`.** No file is
  touched inside the per-cycle call.
* **Inference is pinned to a single sequential thread.** The session sets
  `intra_op_num_threads = inter_op_num_threads = 1`,
  `ExecutionMode.ORT_SEQUENTIAL`, and `GraphOptimizationLevel.ORT_ENABLE_BASIC`.
  Multi-threaded reduction order is the usual source of run-to-run float
  variation in an inference engine; a single sequential thread removes it, so a
  fixed model on a fixed input returns bit-identical outputs every call. The
  optimization level is pinned rather than disabled so the same optimized graph
  runs on every host.

`tests/python/test_onnx_gnc.py` asserts the bit-identity this buys, that two
full closed-loop runs are sha256-identical, and that the loop closes (the
opening attitude error settles with no NaN and the terminal event reached).

## Producing a model (`tests/golden/onnx/`)

The committed `pd_mlp.onnx` is a multi-layer perceptron fitted to the
`pd_attitude` law by scikit-learn and exported to ONNX with `onnx.helper`. It
is reproducible from `tests/golden/onnx/generate.py` (a standalone script, run
with a Python that has `onnx` and `scikit-learn`), and its provenance —
framework versions, the training recipe, the fixed seed, and the tolerance the
consuming tests hold — is recorded in `tests/golden/onnx/manifest.toml`. The
generator standardizes inputs and outputs for a well-conditioned fit and folds
the standardization back into the exported weights, so the graph maps raw
features to raw torque and the runtime performs no normalization of its own.

Any model that honours the six-in / three-out signal contract (or a custom
contract via the overridable hooks) can be flown; the shipped model exists so
the exit-criterion scenario has a committed, reproducible artifact to close
the loop with.
