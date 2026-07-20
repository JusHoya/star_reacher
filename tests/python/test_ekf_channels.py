"""FR-26 navigation-channel round-trip for the reference error-state EKF.

Drives the committed reference consistency mission and checks that what the
filter reported reaches a reader intact: the pinned ``nav.est``/``nav.err``
dimensions, the structural zero-padding of ``nav.innov``, and the ability of
``star consistency`` to consume the result. These are the channels the
estimator acceptance instrument reads, so a defect here would silently
corrupt every consistency number computed downstream.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MISSION = REPO_ROOT / "missions" / "leo_ekf_consistency.toml"

_CORE_MISSING_MESSAGE = (
    "star_reacher._core is not importable: the compiled core is required to "
    "produce the navigation channels under test. Build and install it with "
    "`pip install .` from the repository root, then re-run. This test fails "
    "rather than skipping so a missing core cannot be mistaken for a pass."
)

# Sensor id -> the innovation dimension that sensor's update produces.
NIS_DIM_BY_SENSOR = {1: 3, 2: 6, 3: 1}
INNOV_MAX_DIM = 6


def _core_or_fail():
    try:
        from star_reacher import _core  # noqa: F401
    except ImportError:
        pytest.fail(_CORE_MISSING_MESSAGE)


@pytest.fixture(scope="module")
def reference_run(tmp_path_factory):
    _core_or_fail()
    from star_reacher import load
    from star_reacher.runner import run_mission

    cwd = os.getcwd()
    # The mission's vehicle path is repository-relative.
    os.chdir(REPO_ROOT)
    try:
        out = tmp_path_factory.mktemp("ekf_channels")
        result = run_mission(MISSION, out / "run")
        return result, load(result.srlog_path)
    finally:
        os.chdir(cwd)


def test_nav_channels_have_the_pinned_dimensions(reference_run):
    """n = 16 and m = 15, so P carries 120 doubles (format doc section 3.2)."""
    _, run = reference_run
    est = run.groups["nav.est"]
    err = run.groups["nav.err"]
    assert est["x_hat"].shape[1] == 16
    assert est["P"].shape[1] == 120
    assert err["e"].shape[1] == 16
    # nav.est and nav.err are logged per estimation cycle at identical
    # timestamps, which is what lets the evaluator pair them by index.
    assert len(est) == len(err)
    np.testing.assert_array_equal(est["t_s"], err["t_s"])
    # The state's leading four entries are a unit quaternion.
    norms = np.linalg.norm(est["x_hat"][:, :4], axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-12)


def test_nav_innov_fires_for_every_aiding_sensor(reference_run):
    """Each configured aiding sensor contributes updates at its own m."""
    _, run = reference_run
    innov = run.groups["nav.innov"]
    assert len(innov) > 0, "no aiding update was ever applied"
    assert innov["y"].shape[1] == INNOV_MAX_DIM
    assert innov["S"].shape[1] == INNOV_MAX_DIM * (INNOV_MAX_DIM + 1) // 2
    seen = {int(s) for s in innov["sensor_id"]}
    assert seen == set(NIS_DIM_BY_SENSOR)
    for sensor_id, dim in NIS_DIM_BY_SENSOR.items():
        sel = innov["sensor_id"] == sensor_id
        assert {int(v) for v in innov["m"][sel]} == {dim}


def test_nav_innov_padding_is_structural(reference_run):
    """Padding embeds the m-by-m block in the leading corner, not the row.

    An m-by-m packed upper triangle and an m_max-wide one have different
    row strides, so a flat copy of the short triangle into the front of the
    long buffer would scatter the block across the first row and leave a
    singular S. This pins the corrected layout: entries whose row or column
    exceeds m are exactly zero, and the leading m-by-m block is a valid
    symmetric positive-definite covariance.
    """
    from star_reacher.consistency import unpack_symmetric

    _, run = reference_run
    innov = run.groups["nav.innov"]
    for sensor_id, dim in NIS_DIM_BY_SENSOR.items():
        sel = innov["sensor_id"] == sensor_id
        y = innov["y"][sel]
        s_full = unpack_symmetric(innov["S"][sel])
        # Everything outside the valid dimension is exactly zero.
        np.testing.assert_array_equal(y[:, dim:], 0.0)
        np.testing.assert_array_equal(s_full[:, dim:, :], 0.0)
        np.testing.assert_array_equal(s_full[:, :, dim:], 0.0)
        block = s_full[:, :dim, :dim]
        # A reported covariance must be a valid covariance: symmetric, and
        # positive definite at every record (Cholesky is the check the
        # consistency engine itself applies).
        np.testing.assert_allclose(block, np.swapaxes(block, 1, 2), atol=0.0)
        np.linalg.cholesky(block)


def test_star_consistency_consumes_a_real_ekf_log(reference_run):
    """FR-26 at the command level, on a real error-state run.

    The command must apply the documented quaternion-led n -> m reduction
    (16 -> 15) and group innovations per sensor at each sensor's own
    dimension. The per-run time-averaged numbers are checked for shape and
    labelling only: ch:ekf sec:ekf:consistency calls that aggregation a
    diagnostic with indicative bounds, because its chi-square interval
    assumes epochs within a run are independent while a filter's state
    error is a smooth serially correlated trajectory.
    ``test_per_run_time_average_is_a_diagnostic_not_a_gate`` measures the
    size of that effect over the full ensemble. What is asserted to pass
    is the acceptance instrument itself: the per-sensor NIS gates of
    eq:ekf:ensemble, evaluated here at R = 1.
    """
    result, _ = reference_run
    proc = subprocess.run(
        [sys.executable, "-m", "star_reacher", "consistency", str(result.srlog_path)],
        capture_output=True,
        text=True,
    )
    assert "Traceback" not in proc.stderr, proc.stderr
    # The reduction worked: a 16-dimensional error was gated against a
    # 15-dimensional covariance instead of being rejected as mismatched.
    assert "n=15" in proc.stdout, proc.stdout + proc.stderr
    # Per-sensor grouping worked: one diagnostic line and one pair of
    # ensemble gate lines per aiding sensor, each at its own dimension.
    for sensor_id, dim in sorted(NIS_DIM_BY_SENSOR.items()):
        line = [
            ln
            for ln in proc.stdout.splitlines()
            if f"NIS[sensor {sensor_id}] time-averaged" in ln
        ]
        assert len(line) == 1, proc.stdout
        assert f"m={dim}" in line[0], line[0]
        # A diagnostic never carries a verdict word, whichever side of its
        # indicative interval it lands on.
        assert line[0].rstrip().endswith("[diagnostic, not gated]"), line[0]
        for criterion in ("headline", "coverage"):
            gate = [
                ln
                for ln in proc.stdout.splitlines()
                if f"NIS[sensor {sensor_id}] ensemble {criterion}" in ln
            ]
            assert len(gate) == 1, proc.stdout
            assert gate[0].rstrip().endswith("PASS"), gate[0]
