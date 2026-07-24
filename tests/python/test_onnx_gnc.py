"""FR-28 / Phase 7 exit criterion 4: an ONNX MLP closes the GNC loop.

Exit criterion 4 requires "an ONNX MLP exported from an external framework
[to close] the loop for a full scenario on x86-64 and Pi 5, with cross-platform
final states within the published bound". This module drives the x86-64 leg
for real: it flies missions/leo_attitude_onnx.toml with the learned controller
(star_reacher.ml.OnnxGnc running tests/golden/onnx/pd_mlp.onnx through
onnxruntime) in the control slot and checks that

* the closed loop RUNS TO COMPLETION and SETTLES -- the 10-degree opening
  attitude error is driven down and the terminal body rate is small, with no
  NaN and the run_end event reached (the loop genuinely closed, not merely
  terminated);
* two identical runs are sha256-identical (D-10), so a learned model in the
  loop does not cost reproducibility;
* onnxruntime single-thread inference on a fixed input is bit-identical
  call-to-call, which is why the loop above is deterministic; and
* the final truth state is captured in the same interchange format
  scripts/cross_platform_divergence.py already gates at <= 1e-9 relative, so
  the ARM/Pi-5 leg can be compared against the x86-64 leg by the existing
  machinery (see the module note on cross-platform wiring below).

The runtime path imports ONLY onnxruntime and numpy -- never onnx or any
training framework, which are generation-time tools (tests/golden/onnx/
generate.py). The model itself is committed, so no fitting happens here.

These tests fail cleanly, never skip, when the compiled core is absent (the
project's agent-honesty gate). When onnxruntime (the [ml] extra) is absent
they DO skip: onnxruntime is an optional runtime dependency, so its absence is
a genuinely unconfigured environment rather than a broken one, and a hard fail
there would red every core-only checkout.

Cross-platform wiring (the deferred ARM/Pi-5 leg)
-------------------------------------------------

``test_final_state_is_capturable_for_cross_platform_comparison`` writes a
finalstate.json with cross_platform_divergence.py's ``extract`` subcommand
(reusing the exact format the criterion-8 gate reads) and re-parses it with
that script's own ``load_finalstate``, proving the ONNX mission's final state
is interchange-ready. What remains -- running the same mission on the
ubuntu-24.04-arm CI leg and feeding both legs' finalstate.json to ``measure``
+ ``gate`` at the <= 1e-9 bound -- is CI wiring on a leg this environment
cannot run; it is described in the workstream report for the orchestrator to
apply, following the Section 9 deferral pattern Phases 5 and 6 used for their
literal-Pi-5 clauses.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import math
import os
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ONNX_MISSION = REPO_ROOT / "missions" / "leo_attitude_onnx.toml"
ONNX_PLUGIN = REPO_ROOT / "examples" / "onnx_gnc_plugin.py"
ONNX_MODEL = REPO_ROOT / "tests" / "golden" / "onnx" / "pd_mlp.onnx"

# The attitude the guidance holds: q0 rotated 10 degrees about body +Z. The
# closed loop must drive the estimated (and true) attitude to this.
Q_CMD = [0.0, 0.7660444431189781, 0.6427876096865393, 0.0]

_CORE_MISSING_MESSAGE = (
    "star_reacher._core is not built in this environment. These integration "
    "tests require the compiled core: build and install it with 'pip install .' "
    "from the repository root. This failure is expected on a core-less checkout "
    "and must be green at integration/CI."
)


def _core_or_fail():
    try:
        from star_reacher import _core  # noqa: F401
    except ImportError:
        pytest.fail(_CORE_MISSING_MESSAGE)
    from star_reacher import _core

    return _core


def _require_onnxruntime():
    if importlib.util.find_spec("onnxruntime") is None:
        pytest.skip(
            "onnxruntime (the [ml] extra) is not installed; the FR-28 learned-"
            "model path is optional runtime surface. Install with "
            "'pip install star_reacher[ml]' to exercise it."
        )


@contextlib.contextmanager
def _in_repo_root():
    # The plugin resolves the model relative to its own file, so cwd does not
    # matter for the model load; the mission's vehicle = "vehicles/..." path,
    # however, is resolved relative to cwd, so the run is launched from the
    # repository root exactly as the CLI would be.
    cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        yield
    finally:
        os.chdir(cwd)


@pytest.fixture(autouse=True)
def _repo_root_cwd():
    with _in_repo_root():
        yield


def _run(outdir):
    """Fly the ONNX mission through the batch runner with the plugin."""
    from star_reacher.runner import run_mission

    return run_mission(
        str(ONNX_MISSION),
        outdir=str(outdir),
        force=True,
        gnc_plugins=[str(ONNX_PLUGIN)],
    )


# ---------------------------------------------------------------------------
# The committed artifacts exist and are what the mission names
# ---------------------------------------------------------------------------


def test_committed_model_and_plugin_exist():
    """The mission's dependencies are in-tree, so the run is self-contained."""
    assert ONNX_MODEL.is_file(), f"the committed ONNX model {ONNX_MODEL} is missing"
    assert ONNX_PLUGIN.is_file()
    assert ONNX_MISSION.is_file()
    # The plugin points at the committed model (path resolved relative to the
    # plugin file, so this is the model a run actually loads).
    import examples.onnx_gnc_plugin as plugin  # noqa: PLC0415

    assert Path(plugin.MODEL_PATH).resolve() == ONNX_MODEL.resolve()
    assert "onnx_gnc" in plugin.STAR_GNC_COMPONENTS


# ---------------------------------------------------------------------------
# The learned model loads and infers deterministically (single-thread)
# ---------------------------------------------------------------------------


def test_onnx_inference_is_deterministic_single_thread():
    """A fixed model on a fixed input returns bit-identical outputs (D-10).

    This is the property that makes the closed loop below reproducible: if
    onnxruntime reordered a reduction between calls, the loop's torque would
    differ run to run. The session is configured exactly as OnnxGnc.init
    configures it.
    """
    _require_onnxruntime()
    import onnxruntime as ort  # noqa: PLC0415

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess = ort.InferenceSession(
        str(ONNX_MODEL), sess_options=so, providers=["CPUExecutionProvider"]
    )
    name = sess.get_inputs()[0].name
    assert sess.get_inputs()[0].shape[1] == 6, "the model takes the six-input feature"
    assert sess.get_outputs()[0].shape[1] == 3, "the model returns a three-vector torque"

    x = np.array([[0.0872, -0.01, 0.02, 0.001, -0.002, 0.003]], dtype=np.float32)
    first = sess.run(None, {name: x})[0]
    for _ in range(20):
        again = sess.run(None, {name: x})[0]
        assert again.tobytes() == first.tobytes(), (
            "onnxruntime single-thread inference is not bit-reproducible; the "
            "closed loop would not be deterministic"
        )


def test_onnx_model_approximates_the_pd_law():
    """The learned torque tracks the PD law over the reachable operating region.

    A model that had drifted from the stabilizing law would not settle the
    loop; this pins the approximation quality the manifest records (worst-case
    well under the mission's 0.05 N*m saturation) independently of the closed
    loop, so a bad model is diagnosed at the model rather than at the mission.

    The sample is drawn from the SAME physically valid distribution the model
    was fitted over: the attitude-error features are the vector part of a unit
    error quaternion (a random axis, an error angle up to 60 degrees), so
    ``dq1^2 + dq2^2 + dq3^2 <= 1`` -- the manifold the closed loop feeds the
    model. Sampling the three components from an independent box instead would
    place most points on impossible corners the loop never reaches and the fit
    never targeted, which is a test of extrapolation rather than of the
    controller.
    """
    _require_onnxruntime()
    import onnxruntime as ort  # noqa: PLC0415

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess = ort.InferenceSession(
        str(ONNX_MODEL), sess_options=so, providers=["CPUExecutionProvider"]
    )
    name = sess.get_inputs()[0].name

    # The reference mission's gains and the PD law the model learned:
    # tau = -kp*(s*dq) - kd*we over x = [s*dq, we].
    kp = np.array([0.4, 0.4, 0.4])
    kd = np.array([3.6, 3.6, 3.6])
    rng = np.random.default_rng(7)
    n = 20000
    axis = rng.normal(size=(n, 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    theta = rng.uniform(0.0, np.radians(60.0), size=n)
    sdq = axis * np.sin(theta / 2.0)[:, None]
    we = rng.uniform(-0.30, 0.30, (n, 3))
    x = np.concatenate([sdq, we], axis=1).astype(np.float32)
    pd = -kp * sdq - kd * we
    pred = sess.run(None, {name: x})[0]
    worst = float(np.max(np.abs(pred - pd)))
    # 1e-2 N*m is the held-out bound the manifest records, ~5x below the
    # mission's 0.05 N*m saturation and small against the opening ~0.035 N*m
    # command, which is why the learned loop settles.
    assert worst < 1.0e-2, (
        f"the ONNX model departs from the PD law by {worst:.3e} N*m over the "
        f"reachable region, above the 1e-2 N*m the manifest records; "
        f"regenerate the model"
    )


# ---------------------------------------------------------------------------
# The closed loop runs to completion and settles on x86-64
# ---------------------------------------------------------------------------


def test_onnx_loop_closes_a_full_scenario(tmp_path):
    """The learned controller flies the full attitude-acquisition scenario."""
    _core_or_fail()
    _require_onnxruntime()
    from star_reacher import load

    res = _run(tmp_path / "onnx_run")
    assert res.summary["steps"] > 0
    assert res.summary["truth_records"] == 601, "60 s at 10 Hz plus the initial state"

    run = load(res.srlog_path)

    # The run reached its terminal event: the log is a complete run, not a
    # prefix left by a crash.
    events = run.groups["events"]
    assert str(events[-1]["detail"]) == "run_end", (
        "the run did not reach run_end; the loop terminated abnormally"
    )

    truth = run.groups["truth"]
    q = truth["q_i2b"]
    w = truth["w_b_radps"]
    tau = run.groups["gnc.cmd"]["tau_b_nm"]

    # Finite throughout: a NaN anywhere is a diverged loop.
    assert np.isfinite(q).all(), "the attitude quaternion went non-finite"
    assert np.isfinite(w).all(), "the body rate went non-finite"
    assert np.isfinite(tau).all(), "the commanded torque went non-finite"

    # The torque was genuinely exercised (the opening 10-degree error is real)
    # and stayed inside the configured saturation (0.05 N*m per axis).
    assert np.abs(tau).max() > 1e-3, "the controller commanded ~zero torque; nothing was tested"
    assert np.abs(tau).max() <= 0.05 + 1e-12, "the torque breached the configured saturation"

    # The loop CLOSED: the opening 10-degree attitude error was driven down to
    # a small residual, and the terminal body rate is small (the transient
    # settled rather than limit-cycling). The learned model carries a ~1e-3 N*m
    # approximation error against the PD law, so the residual is looser than the
    # analytic law's near-zero settle but is unambiguously a closed loop.
    def _err_deg(quat):
        d = min(1.0, abs(float(np.dot(quat, Q_CMD))))
        return math.degrees(2.0 * math.acos(d))

    initial_err = _err_deg(q[0])
    final_err = _err_deg(q[-1])
    assert initial_err == pytest.approx(10.0, abs=0.1), (
        f"the scenario did not open with the 10-degree error it is built around "
        f"(got {initial_err:.3f} deg)"
    )
    assert final_err < 1.0, (
        f"the loop did not close: final attitude error {final_err:.3f} deg is not "
        f"a settled acquisition of the 10-degree opening error"
    )
    final_rate = float(np.linalg.norm(w[-1]))
    assert final_rate < 1e-3, (
        f"the loop did not settle: terminal body-rate magnitude {final_rate:.3e} "
        f"rad/s is not small"
    )
    # The error decreased monotonically enough that the end is far below the
    # start: a controller that merely wandered could end small by luck, so the
    # peak-to-final drop is asserted too.
    assert final_err < initial_err / 5.0


# ---------------------------------------------------------------------------
# Determinism: two identical runs are byte-identical (D-10)
# ---------------------------------------------------------------------------


def test_onnx_loop_is_bit_reproducible(tmp_path):
    """Two identical runs produce sha256-identical logs (D-10)."""
    _core_or_fail()
    _require_onnxruntime()

    a = _run(tmp_path / "a")
    b = _run(tmp_path / "b")
    assert a.srlog_sha256 == b.srlog_sha256, (
        "two identical ONNX-in-the-loop runs diverged; a learned model in the "
        "loop must not cost D-10 reproducibility"
    )
    # And the recorded hash is the hash of the bytes on disk (the RunResult
    # value is not merely an in-memory artifact).
    on_disk = hashlib.sha256(Path(a.srlog_path).read_bytes()).hexdigest()
    assert on_disk == a.srlog_sha256


# ---------------------------------------------------------------------------
# Cross-platform final-state capture (the deferred ARM/Pi-5 comparison)
# ---------------------------------------------------------------------------


def _load_cpd():
    """Import the CI cross-platform script by path (it lives outside the pkg)."""
    spec = importlib.util.spec_from_file_location(
        "cross_platform_divergence",
        REPO_ROOT / "scripts" / "cross_platform_divergence.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_final_state_is_capturable_for_cross_platform_comparison(tmp_path):
    """The ONNX run's final state is interchange-ready for the <=1e-9 gate.

    The criterion-4 cross-platform clause compares the x86-64 and Pi-5 final
    states against the published D-10 bound. This proves the ONNX mission's
    final state serializes into, and round-trips out of, the exact format
    scripts/cross_platform_divergence.py already uses for that comparison, so
    the ARM leg needs only to run the same mission and hand its finalstate.json
    to the existing measure/gate subcommands.
    """
    _core_or_fail()
    _require_onnxruntime()
    cpd = _load_cpd()

    res = _run(tmp_path / "onnx_run")
    out = tmp_path / "finalstate" / "finalstate.json"
    rc = cpd.main(
        [
            "extract",
            "--srlog", str(res.srlog_path),
            "--leg", "windows-2022-onnx",
            "--out", str(out),
        ]
    )
    assert rc == 0
    assert out.is_file()

    # Round-trips through the measure-side reader with full-precision hex
    # authoritative fields, so a later ARM-leg finalstate.json can be diffed
    # against this one bit-for-bit up to the <= 1e-9 bound.
    parsed = cpd.load_finalstate(out)
    assert parsed["leg"] == "windows-2022-onnx"
    # The final state is a real LEO state (the translational orbit propagated
    # under point-mass gravity while the attitude loop ran), not a placeholder.
    r_norm = math.hypot(*parsed["r_m"])
    v_norm = math.hypot(*parsed["v_mps"])
    assert 6.9e6 < r_norm < 7.1e6, f"final radius {r_norm:.3e} m is not the LEO state"
    assert 7.0e3 < v_norm < 8.0e3, f"final speed {v_norm:.3e} m/s is not the LEO state"

    # A run of the same mission compares to itself at exactly zero divergence,
    # which is the identity check the cross-platform gate performs pairwise:
    # two legs producing this state would pass the <= 1e-9 bound trivially, and
    # any real divergence would show as a nonzero max_rel.
    second = _run(tmp_path / "onnx_run2")
    out2 = tmp_path / "finalstate2" / "finalstate.json"
    assert cpd.main(
        ["extract", "--srlog", str(second.srlog_path), "--leg", "linux-arm-onnx", "--out", str(out2)]
    ) == 0
    states = [cpd.load_finalstate(out), cpd.load_finalstate(out2)]
    result = cpd.max_pairwise_divergence(states)
    assert result["max_rel"] == 0.0, (
        "the same mission run twice diverged in its final state; the "
        "cross-platform comparison would be measuring nondeterminism"
    )
