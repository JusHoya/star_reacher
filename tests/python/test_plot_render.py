"""Rendering and CLI tests for ``star plot`` (FR-18).

Covers: headless PNG production for both reference missions, the multi-run
overlay mode with resolved-config-hash labels, graceful degradation on logs
missing optional groups, the --plots subset, and the CLI error contract.
PNG assertions are production-level (signature, plausible size, expected
file set); the data-level regression lives in test_plot_golden.py.

Mission-driven tests require the compiled core and fail (never skip)
without it; the synthesized-log tests exercise only core-free plots so the
degradation contract stays testable on a core-less checkout.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

import star_reacher
from star_reacher import _fixtures

REPO_ROOT = Path(__file__).resolve().parents[2]

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

# Expected PNG sets per reference mission (documented in docs/formats/
# plots.md): the two-body log has no forces/env groups, so those plots are
# skipped by design.
FULL_SET = {
    "groundtrack.png",
    "altitude_speed.png",
    "elements.png",
    "attitude_rates.png",
    "mass_thrust_throttle.png",
    "qbar_mach.png",
    "forces_by_source.png",
}
TWOBODY_SET = FULL_SET - {"qbar_mach.png", "forces_by_source.png"}

_CORE_MISSING_MESSAGE = (
    "star_reacher._core is not built in this environment. The plot render "
    "tests over the reference missions require the compiled core: build and "
    "install it with 'pip install .' from the repository root."
)


def _core_or_fail():
    try:
        from star_reacher import _core
    except ImportError:
        pytest.fail(_CORE_MISSING_MESSAGE)
    return _core


def _core_available() -> bool:
    return importlib.util.find_spec("star_reacher._core") is not None


def _run_cli(*args, cwd=None):
    env = os.environ.copy()
    pkg_parent = Path(star_reacher.__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(pkg_parent) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "star_reacher", *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd,
    )


def _assert_valid_png(path: Path) -> None:
    assert path.is_file(), f"{path} was not written"
    data = path.read_bytes()
    assert data[:8] == PNG_MAGIC, f"{path.name}: not a PNG (bad signature)"
    assert len(data) > 1024, f"{path.name}: implausibly small ({len(data)} bytes)"


@pytest.fixture(scope="module")
def mission_logs(tmp_path_factory):
    """Both reference runs, once per module: mission name -> srlog path."""
    _core_or_fail()
    from star_reacher.runner import run_mission

    base = tmp_path_factory.mktemp("plot_render_runs")
    logs = {}
    for mission in ("twobody_leo", "ascent_leo"):
        result = run_mission(REPO_ROOT / "missions" / f"{mission}.toml", base / mission)
        logs[mission] = result.srlog_path
    return logs


# ---------------------------------------------------------------------------
# PNG production smoke over the real reference missions (headless CLI path)
# ---------------------------------------------------------------------------


def test_cli_renders_full_set_for_ascent(mission_logs, tmp_path):
    proc = _run_cli("plot", str(mission_logs["ascent_leo"]), "-o", str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    written = {p.name for p in tmp_path.glob("*.png")}
    assert written == FULL_SET
    for name in FULL_SET:
        _assert_valid_png(tmp_path / name)
    # The only ascent degradation is the throttle panel (not a log channel).
    assert "throttle" in proc.stdout


def test_cli_renders_orbit_set_for_twobody_with_notes(mission_logs, tmp_path):
    proc = _run_cli("plot", str(mission_logs["twobody_leo"]), "-o", str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    written = {p.name for p in tmp_path.glob("*.png")}
    assert written == TWOBODY_SET
    for name in TWOBODY_SET:
        _assert_valid_png(tmp_path / name)
    # FR-18 graceful degradation: skipped plots are announced, exit stays 0.
    assert "qbar_mach" in proc.stdout
    assert "forces_by_source" in proc.stdout


def test_cli_default_outdir_is_plots_beside_log(mission_logs, tmp_path):
    # Copy the log so the default-outdir write lands in this test's sandbox.
    log = tmp_path / "run.srlog"
    log.write_bytes(mission_logs["twobody_leo"].read_bytes())
    proc = _run_cli("plot", str(log), "--plots", "elements")
    assert proc.returncode == 0, proc.stderr
    _assert_valid_png(tmp_path / "plots" / "elements.png")


# ---------------------------------------------------------------------------
# Multi-run overlays
# ---------------------------------------------------------------------------


def test_cli_overlay_produces_shared_set(mission_logs, tmp_path):
    proc = _run_cli(
        "plot",
        str(mission_logs["twobody_leo"]),
        str(mission_logs["ascent_leo"]),
        "-o",
        str(tmp_path),
    )
    assert proc.returncode == 0, proc.stderr
    # Union behavior: a plot renders when ANY given run can feed it, so the
    # overlay set equals the ascent (full) set.
    written = {p.name for p in tmp_path.glob("*.png")}
    assert written == FULL_SET
    for name in FULL_SET:
        _assert_valid_png(tmp_path / name)


def test_overlay_labels_are_short_config_hashes(mission_logs):
    """FR-18 labeling contract, checked at the array-prep layer."""
    from star_reacher.plotting import LABEL_HEX_DIGITS, run_label
    from star_reacher.srlog import load

    labels = {}
    for mission, srlog in mission_logs.items():
        run = load(srlog)
        label = run_label(run)
        assert label == run.header["config_sha256"][:LABEL_HEX_DIGITS]
        assert len(label) == LABEL_HEX_DIGITS
        labels[mission] = label
    # Distinct configurations must yield distinct overlay labels.
    assert labels["twobody_leo"] != labels["ascent_leo"]


# ---------------------------------------------------------------------------
# Graceful degradation and the CLI error contract (core-free)
# ---------------------------------------------------------------------------

_CORE_FREE_PLOTS = "elements,attitude_rates,mass_thrust_throttle,qbar_mach,forces_by_source"


def _synthesized_log(tmp_path, **header_kwargs) -> Path:
    header = _fixtures.contract_header(**header_kwargs)
    records = [
        _fixtures.truth_record(
            float(t), r_m=(7.0e6, 1000.0 * t, 0.0), v_mps=(0.0, 7500.0, 100.0)
        )
        for t in range(12)
    ]
    records.append(_fixtures.event_record(0.0, 1, "run_start"))
    records.append(_fixtures.event_record(11.0, 2, "run_end"))
    log = tmp_path / "synth.srlog"
    log.write_bytes(_fixtures.build_srlog(header, records))
    return log


def test_missing_groups_skip_with_note_and_exit_zero(tmp_path):
    log = _synthesized_log(tmp_path)  # truth + events only
    outdir = tmp_path / "png"
    proc = _run_cli("plot", str(log), "-o", str(outdir), "--plots", _CORE_FREE_PLOTS)
    assert proc.returncode == 0, proc.stderr
    written = {p.name for p in outdir.glob("*.png")}
    assert written == {
        "elements.png",
        "attitude_rates.png",
        "mass_thrust_throttle.png",
    }
    assert "no env group" in proc.stdout
    assert "no forces group" in proc.stdout


def test_prep_layer_reports_reasons_for_missing_groups(tmp_path):
    from star_reacher.plotting import prep_forces_by_source, prep_qbar_mach
    from star_reacher.srlog import load

    run = load(_synthesized_log(tmp_path))
    for prep in (prep_qbar_mach(run), prep_forces_by_source(run)):
        assert prep.arrays is None
        assert prep.note  # a skip always carries its reason


def test_unknown_plot_name_exits_2(tmp_path):
    log = _synthesized_log(tmp_path)
    proc = _run_cli("plot", str(log), "--plots", "elements,orbitograph")
    assert proc.returncode == 2
    assert "orbitograph" in proc.stderr
    # The message teaches the valid vocabulary (DX-2 actionable errors).
    assert "groundtrack" in proc.stderr


def test_missing_file_exits_1(tmp_path):
    proc = _run_cli("plot", str(tmp_path / "absent.srlog"))
    assert proc.returncode == 1
    assert "no such file" in proc.stderr


def test_corrupt_file_exits_1(tmp_path):
    log = tmp_path / "run.srlog"
    log.write_bytes(b"NOTSRLOG" + b"\x00" * 32)
    proc = _run_cli("plot", str(log))
    assert proc.returncode == 1
    assert "magic" in proc.stderr


def test_render_is_headless_and_deterministic(tmp_path):
    """Agg is forced inside the render path; same log -> same PNG bytes."""
    from star_reacher.plotting import render_plots
    from star_reacher.srlog import load

    run = load(_synthesized_log(tmp_path))
    render_plots([run], tmp_path / "a", plots=["elements"])
    render_plots([run], tmp_path / "b", plots=["elements"])
    a = (tmp_path / "a" / "elements.png").read_bytes()
    b = (tmp_path / "b" / "elements.png").read_bytes()
    assert a[:8] == PNG_MAGIC
    assert a == b, "re-rendering the same log produced different PNG bytes"
    import matplotlib

    # The render path must have pinned the non-interactive backend.
    assert matplotlib.get_backend().lower() == "agg"
