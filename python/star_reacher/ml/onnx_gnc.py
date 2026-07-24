"""FR-28 ``OnnxGnc``: run a learned ONNX model in the GNC control loop.

FR-28 makes ``star_reacher.ml.OnnxGnc`` the *only* sanctioned in-the-loop
learned-model path in v1: an ONNX network, executed by onnxruntime on the CPU
(so it runs on a Pi 5), swapped into a GNC chain slot through the FR-25 plugin
mechanism. This module is that adapter. It is an ``IGncComponent`` like any
other, so a mission selects it as ``python:<name>`` and ``star run`` flies it
with no recompilation; what it adds over a hand-written Python control law is
that the control map comes from a trained model file rather than from code.

Where the model path comes from
-------------------------------

A GNC component is configured through ``GncComponentCfg``, which carries only
a ``component`` name plus numeric ``scalars`` and ``vectors`` -- there is no
string field a mission could use to name a model file, and that is deliberate:
a mission TOML must never be able to cause a file to be loaded (the FR-25
trust boundary; loading a plugin is already the one explicit act that runs
code). So the model path is supplied by the *plugin factory*, in Python, on
the ``--gnc-plugin`` file the operator named explicitly::

    from star_reacher.ml import make_onnx_gnc_factory

    STAR_GNC_COMPONENTS = {
        "onnx_gnc": make_onnx_gnc_factory(MODEL_PATH),
    }

The factory closes over a resolved model path (see ``examples/onnx_gnc_plugin.py``
for resolving it relative to the plugin file so it works from any cwd).

Determinism (D-10) inside the time loop
---------------------------------------

This component runs inside the deterministic time loop, so the
``star_reacher.sim`` contract applies to it in full: no clock, no I/O beyond
the one model load, no unseeded randomness, no iteration over unordered
containers, no mutable global state. Two properties make onnxruntime safe to
use here:

* **The model is loaded once, in :meth:`init`, never in :meth:`update`.** No
  file is touched inside the per-cycle call; ``update`` only runs the already
  built session on numbers it was handed.
* **Inference is pinned to a single, sequential thread.** onnxruntime is
  configured with ``intra_op_num_threads = inter_op_num_threads = 1`` and
  ``ExecutionMode.ORT_SEQUENTIAL``, and graph optimization is left at the
  basic level. Multi-threaded reduction order is the usual source of run-to-run
  float variation in an inference engine; forcing a single sequential thread
  removes it, so a fixed model on a fixed input returns bit-identical outputs
  every call. The optimization level is pinned rather than disabled so the
  same optimized graph runs on every host; ``test_onnx_gnc.py`` asserts the
  bit-identity this buys.

onnxruntime is imported lazily so importing this module (and therefore the
``star_reacher.ml`` package) never requires it; a run that actually flies an
``OnnxGnc`` gets an actionable "install the [ml] extra" error if it is absent.

Signal contract of the shipped model
-------------------------------------

The default input builder and output applier below match the signals of the
built-in ``pd_attitude`` law (reimplemented in ``examples/gnc_plugins/pd_attitude.py``):
six inputs -- the sign-corrected attitude-error quaternion vector part and the
body-rate error -- mapping to a three-vector torque, which the applier clamps
to the mission's ``tau_max_nm``. Both hooks are overridable so a differently
shaped model can be flown without subclassing, but the shipped
``tests/golden/onnx/pd_mlp.onnx`` uses exactly this contract.
"""

from __future__ import annotations

from star_reacher._corelink import import_core
from star_reacher.sim import GncOutput, IGncComponent

__all__ = [
    "OnnxGnc",
    "make_onnx_gnc_factory",
    "default_input_builder",
    "default_output_applier",
]


def _import_onnxruntime():
    """Return the onnxruntime module, or raise an actionable ImportError.

    Lazy so ``import star_reacher.ml`` never requires onnxruntime; a mission
    that actually flies an OnnxGnc hits this at construction and is told to
    install the extra rather than failing with a bare ModuleNotFoundError.
    """
    try:
        import onnxruntime  # noqa: PLC0415 - lazy by design (see docstring)
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "star_reacher.ml.OnnxGnc requires the onnxruntime CPU runtime, which "
            "is not installed. Install the ML extra with 'pip install "
            "star_reacher[ml]' (onnxruntime >= 1.18, which ships aarch64/Pi-5 "
            "wheels). onnxruntime is the ONLY runtime dependency the learned-"
            "model path adds; onnx and the training frameworks are needed only "
            "to regenerate the model, never to fly it."
        ) from exc
    return onnxruntime


def default_input_builder(inp, _core):
    """Build the six-input feature vector from a ``GncInput`` (or None to hold).

    Mirrors the ``pd_attitude`` signal set exactly (eq:gnc:deltaq / eq:gnc:sign
    / eq:gnc:werr): the sign-corrected attitude-error quaternion vector part
    and the body-rate error, six numbers, one row. The quaternion product and
    the direction-cosine matrix come from the core's own rotation kernel so the
    features are computed identically to the built-in and the plugin, making a
    learned controller comparable to the analytic one it approximates.

    Returns a ``list[float]`` of length six, or ``None`` when the estimate or
    command is invalid -- the adapter reads that as a hold, exactly as the
    built-in does for an invalid input.
    """
    est = inp.nav_est
    cmd = inp.att_cmd
    if not est.valid or not cmd.valid:
        return None

    qc = cmd.q_i2b
    qe = est.q_i2b
    # dq = q_cmd^* (x) q_est   (eq:gnc:deltaq)
    dq = _core.quat_multiply(
        qc[0], -qc[1], -qc[2], -qc[3], qe[0], qe[1], qe[2], qe[3]
    )
    sign = 1.0 if dq[0] >= 0.0 else -1.0  # eq:gnc:sign, sign(0) = +1
    dcm = _core.quat_to_dcm(dq[0], dq[1], dq[2], dq[3])
    wc = cmd.omega_b_radps
    we = est.omega_b_radps
    # w_err = w_est - C(dq) w_cmd   (eq:gnc:werr)
    rotated = [
        dcm[0] * wc[0] + dcm[1] * wc[1] + dcm[2] * wc[2],
        dcm[3] * wc[0] + dcm[4] * wc[1] + dcm[5] * wc[2],
        dcm[6] * wc[0] + dcm[7] * wc[1] + dcm[8] * wc[2],
    ]
    # The network learned tau against sign-corrected dq (see
    # tests/golden/onnx/generate.py), so the sign is folded into the feature
    # here rather than left for the model to rediscover.
    return [
        sign * dq[1],
        sign * dq[2],
        sign * dq[3],
        we[0] - rotated[0],
        we[1] - rotated[1],
        we[2] - rotated[2],
    ]


def default_output_applier(raw, inp, tau_max):
    """Turn the model's raw output row into a control ``GncOutput``.

    ``raw`` is the model's first output tensor as a flat sequence of at least
    three floats (the body torque). The torque is clamped per axis to
    ``tau_max`` (eq:gnc:sat) -- the same saturation the built-in applies, read
    from the mission so a retune of the clamp needs no new model -- and the
    command attitude and rate are passed through, matching the built-in's
    ``GncOutput``.
    """
    out = GncOutput()
    cmd = inp.att_cmd
    torque = []
    for i in range(3):
        t = float(raw[i])
        # eq:gnc:sat: symmetric per-axis saturation.
        t = max(-tau_max[i], min(tau_max[i], t))
        torque.append(t)
    out.valid = True
    out.q_i2b = cmd.q_i2b
    out.omega_b_radps = cmd.omega_b_radps
    out.torque_b_nm = torque
    return out


class OnnxGnc(IGncComponent):
    """A GNC control component whose control map is an ONNX model (FR-28).

    Construct through :func:`make_onnx_gnc_factory` in a plugin file; the
    factory supplies ``model_path`` (which a ``GncComponentCfg`` cannot carry)
    and any custom hooks. Direct construction is used by the tests.

    Parameters
    ----------
    cfg
        The slot's ``GncComponentCfg``. Its ``vectors["tau_max_nm"]`` sets the
        output saturation if present; absent, the torque is unclamped and the
        applier trusts the model to have learned bounded output.
    model_path
        Path to the ONNX model file. Loaded once in :meth:`init`.
    input_builder
        ``f(GncInput, core) -> list[float] | None`` producing the model's
        input row, or ``None`` to hold. Defaults to :func:`default_input_builder`.
    output_applier
        ``f(raw_row, GncInput, tau_max) -> GncOutput`` turning the model output
        into a command. Defaults to :func:`default_output_applier`.
    """

    def __init__(self, cfg, model_path, *, input_builder=None, output_applier=None):
        # pybind11 trampoline requirement: construct the C++ base before the
        # core can call any override.
        super().__init__()
        self._core = import_core()
        self._model_path = str(model_path)
        self._input_builder = input_builder or default_input_builder
        self._output_applier = output_applier or default_output_applier
        # tau_max is configuration, read once here. A mission that omits it
        # gets an unclamped applier; the reference mission supplies it.
        tau = cfg.vectors.get("tau_max_nm") if hasattr(cfg, "vectors") else None
        self._tau_max = [float(x) for x in tau] if tau else None
        # Built in init(), not here: no session or numpy import at construction
        # keeps the failure modes (missing runtime, missing file) at the run's
        # start where the message can name the mission.
        self._session = None
        self._input_name = None
        self._np = None

    def init(self, ctx):
        # One-time setup at run start (the sole point a file is read). Loading
        # here rather than in update() is what keeps update() free of I/O and
        # therefore deterministic.
        ort = _import_onnxruntime()
        import numpy as np  # noqa: PLC0415 - numpy is a core runtime dependency

        self._np = np
        so = ort.SessionOptions()
        # Single sequential thread: removes multi-threaded reduction-order
        # nondeterminism so a fixed model on a fixed input is bit-reproducible.
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        # Basic optimization is pinned (not disabled): the same optimized graph
        # then runs on every host, so the result does not depend on which
        # fusions a given onnxruntime build would apply at a higher level.
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        try:
            self._session = ort.InferenceSession(
                self._model_path,
                sess_options=so,
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:  # noqa: BLE001 - re-raised with the path named
            raise RuntimeError(
                f"OnnxGnc could not load the ONNX model '{self._model_path}': "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        self._input_name = self._session.get_inputs()[0].name

    def update(self, inp):
        # Called once per control cycle. No file, no clock, no RNG: only a
        # forward pass on the already-loaded session.
        features = self._input_builder(inp, self._core)
        if features is None:
            # Hold: an invalid estimate or absent command is not an
            # instruction to apply zero torque (matches the built-in).
            return GncOutput()
        # float32 is the model's declared input dtype; the row is shaped
        # (1, n) for the single-sample forward pass.
        x = self._np.asarray([features], dtype=self._np.float32)
        raw = self._session.run(None, {self._input_name: x})[0]
        row = raw[0]
        tau_max = self._tau_max
        if tau_max is None:
            # No configured clamp: apply a wide open one so the applier's
            # signature is uniform. The reference mission always configures it.
            row_max = [abs(float(v)) + 1.0 for v in row[:3]]
            return self._output_applier(row, inp, row_max)
        return self._output_applier(row, inp, tau_max)


def make_onnx_gnc_factory(model_path, *, input_builder=None, output_applier=None):
    """Build a ``factory(cfg) -> OnnxGnc`` for a ``STAR_GNC_COMPONENTS`` entry.

    The plugin mechanism calls a factory with the slot's ``GncComponentCfg``
    alone; this closes over the model path (and any custom hooks) so the
    resulting callable has the one-argument shape ``load_plugins`` requires.
    """

    def factory(cfg):
        return OnnxGnc(
            cfg,
            model_path,
            input_builder=input_builder,
            output_applier=output_applier,
        )

    return factory
