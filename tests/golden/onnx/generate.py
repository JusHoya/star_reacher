#!/usr/bin/env python3
"""Reproducibly fit and export the Phase 7 in-the-loop ONNX MLP (FR-28).

Phase 7 exit criterion 4 requires "an ONNX MLP exported from an external
framework" to close a full attitude-control scenario. This script is the
provenance of that artifact: it fits a multi-layer perceptron with an
external ML framework (scikit-learn's ``MLPRegressor``) to input/output
pairs drawn from the built-in ``pd_attitude`` control law, then writes the
fitted weights into a genuine ONNX graph (Gemm -> ReLU -> Gemm -> ... -> Gemm)
built with ``onnx.helper``. The weights therefore come from a real training
step and the graph is a real MLP that ``onnxruntime`` executes; only the ONNX
export path is hand-built, because ``skl2onnx`` is not a guaranteed dev
dependency and building the graph directly keeps the artifact free of any
converter-version drift.

The learned signal set matches ``examples/gnc_plugins/pd_attitude.py`` and
``star_reacher.ml.onnx_gnc`` exactly: six inputs and three outputs.

  input  x = [ s*dq1, s*dq2, s*dq3, we0, we1, we2 ]
  output tau_preclamp = [ tau0, tau1, tau2 ]

where ``s = sign(dq0)`` is the shortest-rotation sign, ``dq = q_cmd* (x) q_est``
is the attitude-error quaternion, and ``we`` is the body-rate error
``w_est - C(dq) w_cmd`` (the eq:gnc:werr quantity). Against the reference
mission's per-axis PD gains, the pre-saturation control law

  tau_i = -kp_i * (s*dq_(i+1)) - kd_i * we_i                        (eq:gnc:pd)

is the target. The runtime adapter applies the eq:gnc:sat clamp from the
mission's ``tau_max_nm`` after inference. Fitting a stabilizing law rather
than an arbitrary one is what lets the learned loop settle instead of diverge.

Training-distribution choices that make the fit honest
------------------------------------------------------

Two choices keep the exported model accurate across the region the closed
loop actually visits, rather than merely on the points it was shown:

* **Physically valid attitude errors.** The three attitude-error features are
  the vector part of a *unit* error quaternion, so they satisfy
  ``dq1^2 + dq2^2 + dq3^2 <= 1`` and cannot all be large at once. Sampling
  them from real rotations (a random axis and an error angle up to 60 degrees)
  instead of from an independent box keeps the training points on the manifold
  the loop feeds the model; an independent-box sample would place most of its
  mass on impossible corners and leave the reachable region under-fitted.
* **Baked-in standardization.** Inputs and outputs are standardized before the
  fit (adam conditions far better on unit-scaled data), and the standardization
  is then FOLDED INTO the exported weights -- absorbed into the first layer's
  ``W,b`` and the last layer's ``W,b`` -- so the ONNX graph maps RAW features
  to RAW torque with no separate normalization node. The runtime therefore
  never standardizes; it hands the model the same six numbers the built-in law
  uses and reads a torque back.

The recipe reports the worst-case torque error on a held-out sample drawn from
the same reachable region and fails unless it is below ``--max-abs-err``; that
held-out number, not the training error, is what the manifest records, because
generalization on the operating region is the property the closed loop needs.

Determinism. Every stochastic step is seeded from ``SEED`` (the attitude/rate
sampling, the sklearn solver's weight initialization, and its minibatch
shuffling), so a rerun reproduces the same weights and therefore a
byte-identical ``pd_mlp.onnx``. The exported protobuf is serialized with a
pinned IR/opset and a fixed ``doc_string``/producer, and initializer tensors
are written in a fixed order, so the bytes are stable given the same onnx and
numpy versions. Regenerate with::

    python tests/golden/onnx/generate.py

and commit the resulting ``pd_mlp.onnx`` together with a ``manifest.toml``
date/tolerance update if the recipe changed. Byte-stability across onnx
library versions is not claimed (the protobuf field layout is onnx's, not
ours); the manifest records the version the committed bytes were produced
with, and ``verify_close`` below is the semantic gate that a regenerated
model still approximates the PD law tightly enough for the loop to close.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

# The reference mission's per-axis PD gains. These are the gains of
# missions/leo_attitude_gnc.toml; the network learns the unclamped law for
# these gains, and the runtime adapter reads tau_max from the mission so a
# mission that retunes the saturation needs no new model.
KP = np.array([0.4, 0.4, 0.4], dtype=np.float64)
KD = np.array([3.6, 3.6, 3.6], dtype=np.float64)

# The single seed every stochastic step draws from, so the artifact is
# reproducible (D-10 discipline for golden vectors).
SEED = 20260723

# The MLP shape: two hidden layers of sixteen ReLU units. Small enough to keep
# onnxruntime inference cheap on a Pi-5-class CPU, deep enough to be a genuine
# multi-layer perceptron.
HIDDEN = (16, 16)

# Training-region bounds. The attitude error is a rotation of up to
# MAX_ERR_DEG about a random axis (the reference scenario opens at 10 degrees;
# 60 degrees is a comfortable margin over any transient). The body-rate error
# spans a band that brackets the transient (which peaks well inside 0.01 rad/s
# for this scenario, so 0.3 rad/s is generous headroom).
N_TRAIN = 40000
N_HELDOUT = 30000
MAX_ERR_DEG = 60.0
WE_RANGE = 0.30

# Adam with input/output standardization converges reliably on this smooth
# target; the long no-change patience lets it run to the iteration cap rather
# than early-stopping on a plateau.
MAX_ITER = 12000
LEARNING_RATE_INIT = 3.0e-3
TOL = 1.0e-14

# ONNX export pins. IR 9 / opset 18 are widely supported by onnxruntime >= 1.18
# (the [ml] extra floor) including its aarch64 wheels, so the committed model
# loads on the Pi 5 leg as well as on x86-64.
ONNX_IR_VERSION = 9
ONNX_OPSET = 18
PRODUCER = "star_reacher.tests.golden.onnx.generate"


def pd_torque(x: np.ndarray) -> np.ndarray:
    """The PD law tau = -kp*(s*dq) - kd*we, given x = [s*dq, we].

    ``x`` is (N, 6): columns 0..2 are the sign-corrected attitude-error vector
    part, columns 3..5 the body-rate error. Returns (N, 3).
    """
    sdq = x[:, 0:3]
    we = x[:, 3:6]
    return -KP * sdq - KD * we


def _attitude_error_vec(rng: np.random.Generator, n: int) -> np.ndarray:
    """Vector part of a unit error quaternion for a random rotation.

    A random unit axis and an error angle uniform in [0, MAX_ERR_DEG] give
    ``vec = axis * sin(theta/2)``; ``|vec| = sin(theta/2) <= sin(30 deg)``.
    This is the sign-corrected ``s*dq`` the controller sees, so the training
    points lie on exactly the manifold the closed loop feeds the model.
    """
    axis = rng.normal(size=(n, 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    theta = rng.uniform(0.0, np.radians(MAX_ERR_DEG), size=n)
    return axis * np.sin(theta / 2.0)[:, None]


def sample_data(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Draw (x, y) pairs over the controller's reachable operating region."""
    sdq = _attitude_error_vec(rng, n)
    we = rng.uniform(-WE_RANGE, WE_RANGE, size=(n, 3))
    x = np.concatenate([sdq, we], axis=1)
    return x, pd_torque(x)


def fit_mlp(x: np.ndarray, y: np.ndarray):
    """Fit an sklearn MLPRegressor (the external framework) on standardized data.

    Returns ``(model, xm, xs, ym, ys)`` -- the fitted regressor and the input
    and output standardization statistics, which the export folds into the
    graph weights so the runtime works in raw units.
    """
    # Imported here, not at module top, so a checkout without scikit-learn can
    # still read this file's docstring. The runtime never imports this module.
    from sklearn.neural_network import MLPRegressor

    xm = x.mean(axis=0)
    xs = x.std(axis=0)
    ym = y.mean(axis=0)
    ys = y.std(axis=0)
    xn = (x - xm) / xs
    yn = (y - ym) / ys

    model = MLPRegressor(
        hidden_layer_sizes=HIDDEN,
        activation="relu",
        solver="adam",
        alpha=0.0,  # no L2 penalty: the target is smooth and exactly shaped
        max_iter=MAX_ITER,
        tol=TOL,
        n_iter_no_change=MAX_ITER,  # run to the cap rather than early-stopping
        learning_rate_init=LEARNING_RATE_INIT,
        random_state=SEED,  # seeds weight init and minibatch shuffling
    )
    model.fit(xn, yn)
    return model, xm, xs, ym, ys


def folded_layers(model, xm, xs, ym, ys):
    """The MLP's per-layer (W, b) with standardization folded in, raw in/out.

    sklearn's fitted net computes, on standardized data,
    ``yn = f(xn)`` with ``xn = (x - xm)/xs`` and ``y = yn*ys + ym``. Folding
    the input scaling into the first layer and the output scaling into the last
    lets the exported graph consume raw ``x`` and emit raw ``y``:

      first layer:  W0' = W0 / xs[:,None],  b0' = b0 - (xm/xs) @ W0
      last  layer:  Wn' = Wn * ys[None,:],  bn' = bn*ys + ym

    Hidden layers are unchanged. Returns a list of (W, b) float32 pairs.
    """
    coefs = [np.asarray(c, dtype=np.float64) for c in model.coefs_]
    inter = [np.asarray(b, dtype=np.float64) for b in model.intercepts_]

    # Input standardization folded into layer 0.
    coefs[0] = coefs[0] / xs[:, None]
    inter[0] = inter[0] - (xm / xs) @ np.asarray(model.coefs_[0], dtype=np.float64)

    # Output destandardization folded into the last layer.
    coefs[-1] = coefs[-1] * ys[None, :]
    inter[-1] = inter[-1] * ys + ym

    return [(w.astype(np.float32), b.astype(np.float32)) for w, b in zip(coefs, inter)]


def build_onnx(layers) -> "onnx.ModelProto":  # noqa: F821 - onnx imported below
    """Write the folded MLP layers into a Gemm/ReLU ONNX graph.

    Each layer is one ONNX ``Gemm`` (default alpha=beta=1, transA=transB=0, so
    it computes ``X @ W + B`` directly from the (fan_in, fan_out) weight matrix
    with no transpose) followed by ``Relu`` on every layer but the last
    (identity output activation).
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    n_in = layers[0][0].shape[0]
    n_out = layers[-1][0].shape[1]

    nodes = []
    initializers = []
    prev = "input"
    for k, (w, b) in enumerate(layers):
        w_name = f"W{k}"
        b_name = f"B{k}"
        # Fixed (layer-major) order so the serialized bytes do not depend on
        # dict iteration.
        initializers.append(numpy_helper.from_array(w, name=w_name))
        initializers.append(numpy_helper.from_array(b, name=b_name))
        out_name = "output" if k == len(layers) - 1 else f"gemm{k}"
        nodes.append(
            helper.make_node("Gemm", [prev, w_name, b_name], [out_name], name=f"Gemm{k}")
        )
        if k < len(layers) - 1:
            relu_out = f"relu{k}"
            nodes.append(helper.make_node("Relu", [out_name], [relu_out], name=f"Relu{k}"))
            prev = relu_out

    graph = helper.make_graph(
        nodes,
        "pd_mlp",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, int(n_in)])
        ],
        outputs=[
            helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, int(n_out)])
        ],
        initializer=initializers,
    )
    # A fixed doc_string and producer keep the serialized bytes free of any
    # host- or time-dependent field.
    graph.doc_string = (
        "PD attitude control law learned by MLPRegressor with standardization "
        "folded in; inputs [s*dq1, s*dq2, s*dq3, we0, we1, we2] -> torque [3]"
    )
    model_proto = helper.make_model(
        graph,
        producer_name=PRODUCER,
        opset_imports=[helper.make_operatorsetid("", ONNX_OPSET)],
        doc_string="star_reacher Phase 7 FR-28 in-the-loop ONNX MLP",
    )
    model_proto.ir_version = ONNX_IR_VERSION
    onnx.checker.check_model(model_proto)
    return model_proto


def verify_close(model_proto, x: np.ndarray, y: np.ndarray) -> float:
    """Run the exported model through onnxruntime and bound its error.

    Returns the maximum absolute torque error against the PD law over ``x``.
    Passing a HELD-OUT sample makes this a generalization bound, which is what
    the closed loop's stability actually depends on.
    """
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess = ort.InferenceSession(model_proto.SerializeToString(), sess_options=so)
    pred = sess.run(None, {"input": x.astype(np.float32)})[0]
    return float(np.max(np.abs(pred - y)))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=str(Path(__file__).with_name("pd_mlp.onnx")),
        help="output ONNX model path",
    )
    parser.add_argument(
        "--max-abs-err",
        type=float,
        default=1.0e-2,
        help="fail if the model's worst held-out torque error exceeds this (N*m)",
    )
    args = parser.parse_args(argv)

    train_rng = np.random.default_rng(SEED)
    x, y = sample_data(train_rng, N_TRAIN)
    model, xm, xs, ym, ys = fit_mlp(x, y)
    layers = folded_layers(model, xm, xs, ym, ys)
    model_proto = build_onnx(layers)

    # The gate is on a held-out sample from a fixed, independent seed: the
    # reachable operating region, points the model never saw.
    heldout_rng = np.random.default_rng(SEED + 1)
    xh, yh = sample_data(heldout_rng, N_HELDOUT)
    worst = verify_close(model_proto, xh, yh)
    if worst > args.max_abs_err:
        raise SystemExit(
            f"fitted MLP departs from the PD law by {worst:.3e} N*m on held-out "
            f"data, above the {args.max_abs_err:.1e} N*m gate; the closed loop "
            f"may not settle"
        )

    out = Path(args.out)
    out.write_bytes(model_proto.SerializeToString())
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"wrote {out}")
    print(f"  worst held-out |tau_pred - tau_pd| over {len(xh)} samples: {worst:.3e} N*m")
    print(f"  sha256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
