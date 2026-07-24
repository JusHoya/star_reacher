"""A GNC plugin (FR-25/FR-28): the ``OnnxGnc`` control component wired up.

Fly the ONNX attitude mission with this file in the control slot::

    star run missions/leo_attitude_onnx.toml \\
        --gnc-plugin examples/onnx_gnc_plugin.py

The control law is not written here -- it is the ONNX model committed at
``tests/golden/onnx/pd_mlp.onnx``, a multi-layer perceptron fitted to the
built-in ``pd_attitude`` law by an external ML framework (scikit-learn) and
exported to ONNX (provenance in ``tests/golden/onnx/manifest.toml``). This
plugin only names the model file and hands it to
:class:`star_reacher.ml.OnnxGnc`, which runs it through onnxruntime on the CPU
inside the deterministic time loop.

Model path resolution. ``GncComponentCfg`` carries no string field, so the
model path cannot ride in the mission file -- it is supplied here, in the
factory, and resolved relative to *this file* rather than to the working
directory, so ``star run`` finds the model whatever cwd it is launched from.
The path is committed in-tree, so the resolution is stable across checkouts.

Determinism. Every rule of the ``star_reacher.sim`` contract is honoured: the
model is loaded once at run start (never inside the per-cycle update), and
onnxruntime is pinned to a single sequential thread so a fixed model on a
fixed input returns bit-identical torques. This file itself reads no clock,
opens no file beyond the model load ``OnnxGnc`` performs in ``init``, and
draws no random number.
"""

from pathlib import Path

from star_reacher.ml import make_onnx_gnc_factory

# The committed model, resolved relative to this plugin file. The example lives
# at examples/onnx_gnc_plugin.py and the model at tests/golden/onnx/pd_mlp.onnx,
# so the repository root is this file's parent's parent.
_REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = _REPO_ROOT / "tests" / "golden" / "onnx" / "pd_mlp.onnx"

# The name a mission selects as "python:onnx_gnc". The factory closes over the
# resolved model path so the plugin mechanism can call it with the slot's
# GncComponentCfg alone.
STAR_GNC_COMPONENTS = {"onnx_gnc": make_onnx_gnc_factory(MODEL_PATH)}
