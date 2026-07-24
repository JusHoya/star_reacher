"""FR-28 learned-model-in-the-loop path: the sanctioned ONNX GNC adapter.

``star_reacher.ml`` is the one place a trained model enters the deterministic
time loop in v1. Its single public surface is :class:`OnnxGnc` -- an
``IGncComponent`` that runs an ONNX network through onnxruntime on the CPU --
plus :func:`make_onnx_gnc_factory`, which packages it for the FR-25 plugin
mechanism.

Importing this package pulls in no heavy dependency: onnxruntime is imported
lazily inside :class:`OnnxGnc` so a checkout without the ``[ml]`` extra can
still import ``star_reacher.ml`` and read its help; a run that actually flies
an ``OnnxGnc`` gets an actionable error naming the extra if onnxruntime is
absent. See :mod:`star_reacher.ml.onnx_gnc` for the full contract.
"""

from __future__ import annotations

from star_reacher.ml.onnx_gnc import (
    OnnxGnc,
    default_input_builder,
    default_output_applier,
    make_onnx_gnc_factory,
)

__all__ = [
    "OnnxGnc",
    "make_onnx_gnc_factory",
    "default_input_builder",
    "default_output_applier",
]
