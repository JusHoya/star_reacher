"""Phase 6 exit criterion 3: the reference EKF's ensemble consistency gates.

Executes the 100-run seeded ensemble of ``missions/leo_ekf_consistency.toml``
and gates the reference error-state EKF on ensemble NEES and per-sensor NIS
against two-sided 95 % chi-square bounds (ch:ekf, eq:ekf:ensemble), then
re-executes the ensemble and requires the SRLOG SHA-256 of every run to be
bit-identical.

The gate statistic is the one ch:ekf sec:ekf:consistency specifies: the
N-run ensemble average at each epoch, averaged over epochs for the headline
number, against ``[chi2_0.025(Nn)/N, chi2_0.975(Nn)/N]``. Bounds come from
the project's own exact evaluator (``star_reacher.chi2.chi2_ppf``), never
from the Wilson--Hilferty numbers quoted in the chapter prose -- those are
an approximation stated for the reader's cross-check, and gating on them
would gate on a rounded transcription rather than on the distribution.

Two aggregations documented in the chapter as DIAGNOSTICS are deliberately
not asserted here, because on real runs they are known to be mis-calibrated
and asserting them would gate on a statistical artifact rather than on the
filter:

* the per-run time average, whose chi-square bounds assume epochs within a
  run are independent while a filter's state error is a smooth, strongly
  serially correlated trajectory (the chapter says so in prose;
  ``test_per_run_time_average_is_a_diagnostic_not_a_gate`` measures how far
  off it is);
* the pooled ensemble statistic, whose bounds at ``dof = R*T*n`` assume the
  same independence across epochs.

The ensemble per-epoch average is not affected: at each epoch it averages R
INDEPENDENT runs, which is exactly the assumption its bounds are derived
under.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MISSION = REPO_ROOT / "missions" / "leo_ekf_consistency.toml"

_CORE_MISSING_MESSAGE = (
    "star_reacher._core is not importable: the compiled core is required to "
    "execute the exit-criterion-3 ensemble. Build and install it with "
    "`pip install .` from the repository root, then re-run. This test fails "
    "rather than skipping so a missing core can never be mistaken for a "
    "passing consistency gate."
)

# --- the scenario's pinned truth, mirrored from the mission file ----------
# The mission states the TRUE initial state; the driver perturbs the
# filter's ESTIMATE of it. test_mission_file_matches_driver_run_zero pins
# this mirror against the committed file.
_A = math.sqrt(0.5)
Q_TRUE = np.array([0.0, _A, _A, 0.0])
R_TRUE_M = np.array([7.0e6, 0.0, 0.0])
V_TRUE_MPS = np.array([0.0, 7546.0, 0.0])
# P0's 1-sigma values for the three blocks the driver perturbs.
SIGMA_ATT_RAD = 1.0e-3
SIGMA_VEL_MPS = 0.5
SIGMA_POS_M = 50.0

N_RUNS = 100
BASE_RUN_SEED = 20260701
# The draw stream is separate from the core's run seed so the initial-error
# draws and the sensor noise cannot alias onto one another.
DRAW_SEED = 90210

# Sensor ids as the canonical kind order assigns them (imu, startracker,
# navfix, altimeter), mapped to each aiding sensor's innovation dimension.
NIS_DIM_BY_SENSOR = {1: 3, 2: 6, 3: 1}
NEES_DIM = 15


def _core_or_fail():
    try:
        from star_reacher import _core  # noqa: F401
    except ImportError:
        pytest.fail(_CORE_MISSING_MESSAGE)


def _quat_multiply(p, q):
    """Hamilton product, scalar-first (D-7), matching star::rotation."""
    w1, x1, y1, z1 = p
    w2, x2, y2, z2 = q
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _quat_exp(v):
    """Exact exponential map of a rotation vector (eq:optical:qab)."""
    theta = float(np.linalg.norm(v))
    if theta == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    s = math.sin(0.5 * theta) / theta
    return np.array([math.cos(0.5 * theta), s * v[0], s * v[1], s * v[2]])


def initial_estimate(run_index: int):
    """The filter's initial estimate for one ensemble run.

    The error is drawn from N(0, P0) and SUBTRACTED from truth, so the
    realized initial error is distributed exactly as P0 claims. The
    attitude uses the multiplicative convention of eq:ekf:qerr:
    ``q_true = q_hat (x) dq(dtheta)`` gives ``q_hat = q_true (x) dq(-dtheta)``.

    The bias estimates are deliberately NOT perturbed: they start at zero
    and P0's bias blocks carry the instruments' stationary Gauss-Markov
    sigmas. Because the IMU initializes its in-run bias from exactly that
    stationary distribution, the initial bias error already has the
    distribution the filter believes, without the driver needing to know
    the sensor's private draw.
    """
    rng = np.random.default_rng(DRAW_SEED + run_index)
    dtheta = SIGMA_ATT_RAD * rng.standard_normal(3)
    dv = SIGMA_VEL_MPS * rng.standard_normal(3)
    dp = SIGMA_POS_M * rng.standard_normal(3)
    return (
        _quat_multiply(Q_TRUE, _quat_exp(-dtheta)),
        V_TRUE_MPS - dv,
        R_TRUE_M - dp,
    )


def _format_vector(values) -> str:
    # Full round-trip precision: a truncated initial estimate would make the
    # committed mission and the driver's run 0 different runs.
    return "[" + ", ".join("%.17g" % float(v) for v in values) + "]"


def mission_text_for_run(base_text: str, run_index: int) -> str:
    """Derive one ensemble member from the committed reference mission."""
    q_hat, v_hat, p_hat = initial_estimate(run_index)
    text = re.sub(
        r"(?m)^seed = .*$", "seed = %d" % (BASE_RUN_SEED + run_index), base_text
    )
    # q0 appears once, under [gnc.nav]; the guidance slot uses q_cmd.
    text = re.sub(r"(?m)^q0 = .*$", "q0 = " + _format_vector(q_hat), text, count=1)
    text = re.sub(r"(?m)^v0_mps = .*$", "v0_mps = " + _format_vector(v_hat), text)
    text = re.sub(r"(?m)^p0_m = .*$", "p0_m = " + _format_vector(p_hat), text)
    return text


def reduce_error(e: np.ndarray) -> np.ndarray:
    """Reduce the logged 16-vector nav.err to the 15 dimensions P describes.

    The leading four components are the sign-canonicalized multiplicative
    error quaternion of eq:ekf:qerr; the small-angle extraction
    ``dtheta = 2 sgn(dq_w) dq_v`` is the same reduction
    ``star consistency`` applies (docs/formats/srlog_v1.md, nav.err).
    """
    w = e[:, 0]
    qv = e[:, 1:4]
    sign = np.where(w >= 0.0, 1.0, -1.0)[:, np.newaxis]
    return np.concatenate([2.0 * sign * qv, e[:, 4:]], axis=1)


def _per_sensor_innovations(innov):
    """Split zero-padded nav.innov records into per-sensor (y, S) arrays."""
    from star_reacher.consistency import pack_symmetric, unpack_symmetric

    m_max = innov["y"].shape[-1]
    out = {}
    for sensor_id in sorted({int(s) for s in innov["sensor_id"]}):
        sel = innov["sensor_id"] == sensor_id
        m = int(innov["m"][sel][0])
        y = innov["y"][sel][:, :m]
        s = innov["S"][sel]
        if m < m_max:
            s = pack_symmetric(unpack_symmetric(s)[:, :m, :m])
        out[sensor_id] = (y, s)
    return out


def _execute_ensemble(outroot: Path, n_runs: int = N_RUNS):
    """Run the ensemble; return (sha256 list, NEES array, NIS arrays)."""
    from star_reacher import load
    from star_reacher.consistency import nees, nis
    from star_reacher.runner import run_mission

    base_text = MISSION.read_text()
    outroot = Path(outroot).resolve()
    outroot.mkdir(parents=True, exist_ok=True)
    shas: list[str] = []
    nees_runs: list[np.ndarray] = []
    nis_runs: dict[int, list[np.ndarray]] = {}

    cwd = os.getcwd()
    # The mission's vehicle path is repository-relative.
    os.chdir(REPO_ROOT)
    try:
        for i in range(n_runs):
            mission_path = outroot / ("run%03d.toml" % i)
            mission_path.write_text(mission_text_for_run(base_text, i))
            result = run_mission(mission_path, outroot / ("run%03d" % i), force=True)
            shas.append(result.srlog_sha256)
            run = load(result.srlog_path)
            nees_runs.append(
                nees(reduce_error(run.groups["nav.err"]["e"]), run.groups["nav.est"]["P"])
            )
            for sensor_id, (y, s) in _per_sensor_innovations(
                run.groups["nav.innov"]
            ).items():
                nis_runs.setdefault(sensor_id, []).append(nis(y, s))
    finally:
        os.chdir(cwd)
    return shas, np.stack(nees_runs), {k: np.stack(v) for k, v in nis_runs.items()}


def ensemble_bounds(dim: int, n_runs: int = N_RUNS) -> tuple[float, float]:
    """The eq:ekf:ensemble acceptance interval, from the exact evaluator."""
    from star_reacher.chi2 import chi2_ppf

    dof = n_runs * dim
    return chi2_ppf(0.025, dof) / n_runs, chi2_ppf(0.975, dof) / n_runs


@pytest.fixture(scope="module")
def ensemble(tmp_path_factory):
    _core_or_fail()
    return _execute_ensemble(tmp_path_factory.mktemp("ekf_ensemble"))


def test_ensemble_has_the_expected_shape(ensemble):
    shas, nees_eps, nis_eps = ensemble
    assert len(shas) == N_RUNS
    assert nees_eps.shape[0] == N_RUNS
    # All three aiding sensors must have contributed innovations; a filter
    # that silently skipped a sensor would otherwise pass the gates it did
    # run and look healthy.
    assert sorted(nis_eps) == sorted(NIS_DIM_BY_SENSOR)
    for sensor_id, arr in nis_eps.items():
        assert arr.shape[0] == N_RUNS, sensor_id


def test_ensemble_nees_gate_passes(ensemble):
    """Exit criterion 3, NEES half (eq:ekf:nees, eq:ekf:ensemble)."""
    _, nees_eps, _ = ensemble
    lower, upper = ensemble_bounds(NEES_DIM)
    epoch_mean = nees_eps.mean(axis=0)
    headline = float(epoch_mean.mean())
    assert lower <= headline <= upper, (
        f"ensemble NEES {headline:.4f} outside [{lower:.4f}, {upper:.4f}] "
        f"(chi2 95 %, dof {N_RUNS * NEES_DIM})"
    )
    # The per-epoch form of the same gate: the ensemble average must sit
    # inside the bounds at essentially every epoch, not merely on average.
    inside = np.mean((epoch_mean >= lower) & (epoch_mean <= upper))
    assert inside >= 0.95, (
        f"only {100.0 * inside:.1f} % of epochs inside [{lower:.4f}, "
        f"{upper:.4f}]"
    )


@pytest.mark.parametrize("sensor_id", sorted(NIS_DIM_BY_SENSOR))
def test_ensemble_nis_gate_passes(ensemble, sensor_id):
    """Exit criterion 3, NIS half, per aiding sensor (eq:ekf:nis)."""
    _, _, nis_eps = ensemble
    dim = NIS_DIM_BY_SENSOR[sensor_id]
    lower, upper = ensemble_bounds(dim)
    headline = float(nis_eps[sensor_id].mean(axis=0).mean())
    assert lower <= headline <= upper, (
        f"sensor {sensor_id} ensemble NIS {headline:.4f} outside "
        f"[{lower:.4f}, {upper:.4f}] (chi2 95 %, dof {N_RUNS * dim})"
    )


def test_ensemble_rerun_is_bit_identical(tmp_path, ensemble):
    """Exit criterion 3's second half: the gate rerun is bit-identical."""
    shas, _, _ = ensemble
    rerun_shas, _, _ = _execute_ensemble(tmp_path / "rerun")
    assert rerun_shas == shas


def test_mission_file_matches_driver_run_zero():
    """The committed mission is exactly run 0, so the two cannot drift."""
    text = MISSION.read_text()
    q_hat, v_hat, p_hat = initial_estimate(0)
    for key, expected in (("q0", q_hat), ("v0_mps", v_hat), ("p0_m", p_hat)):
        match = re.search(r"(?m)^%s = \[(.*)\]$" % key, text)
        assert match is not None, key
        committed = [float(x) for x in match.group(1).split(",")]
        np.testing.assert_allclose(committed, expected, rtol=0.0, atol=0.0)
    assert re.search(r"(?m)^seed = %d$" % BASE_RUN_SEED, text) is not None


def test_per_run_time_average_is_a_diagnostic_not_a_gate(ensemble):
    """Measure the chapter's serial-correlation caveat instead of asserting it.

    ch:ekf sec:ekf:consistency states that a per-run time average is a
    diagnostic with indicative bounds, because successive errors within one
    run are serially correlated and the independence assumption behind its
    chi-square bounds does not hold. This test pins the consequence
    quantitatively so it cannot be mistaken for a filter defect: on an
    ensemble whose per-epoch statistic is inside the acceptance bounds at
    every epoch, only a small minority of individual runs pass the
    time-averaged interval, while the mean over runs of those same
    time averages sits at the expected dimension.
    """
    from star_reacher.consistency import time_average_gate

    _, nees_eps, _ = ensemble
    passing = sum(
        1 for i in range(N_RUNS) if time_average_gate(nees_eps[i], NEES_DIM).passed
    )
    # The mean across runs of the per-run time averages is unbiased: serial
    # correlation inflates the spread, not the expectation.
    lower, upper = ensemble_bounds(NEES_DIM)
    assert lower <= float(nees_eps.mean()) <= upper
    # And the per-run interval, applied as if epochs were independent,
    # rejects the large majority of those same consistent runs.
    assert passing < N_RUNS // 2, (
        f"{passing}/{N_RUNS} runs passed the per-run time-averaged interval; "
        f"if this ever approaches the ensemble pass rate the chapter's "
        f"serial-correlation caveat should be revisited"
    )
