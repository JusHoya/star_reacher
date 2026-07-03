"""Cross-tool frozen-truth gates (Phase 3 exit criterion 5, D-15).

Each test re-propagates a committed mission with the installed package and
compares the resulting truth log against the frozen external baseline
committed under ``tests/golden/crosstool/``:

- XTOOL-LEO-GRAV-GMAT   missions/leo_gravity_8x8.toml vs GMAT R2026a,
                        position RMS < 10 m over 7 days;
- XTOOL-LEO-DRAG-OREKIT missions/leo_drag_hp.toml vs Orekit 13.1.5,
                        position RMS < 100 m over 7 days.

The baselines were generated offline on the maintainer machine (D-15: CI
never installs GMAT or Orekit) with the external tools configured as a
controlled comparison -- the identical gravity field generated from the
committed coefficient excerpt, and zeroed Earth-orientation parameters
matching the simulator's no-EOP convention. Full provenance, tool versions,
command lines, and the measured baseline RMS values are recorded in
``tests/golden/crosstool/manifest.toml``.

Both trajectories are compared on the shared exact 60 s grid: the frozen
CSVs carry one row per 60 s (the external runs were sampled at that fixed
cadence, never at adaptive integrator steps) and the mission truth log is
1 Hz, so rows align bit-exactly in time and no interpolation enters the
comparison.

Like test_crosstool_missions.py, these tests REQUIRE the compiled core and
fail, never skip, without it; they run from the repo root (mission data
paths are CWD-relative) and consume only committed artifacts.
"""

from pathlib import Path

import numpy as np

from star_reacher.runner import run_mission

import star_reacher

REPO_ROOT = Path(__file__).resolve().parents[2]
CROSSTOOL = REPO_ROOT / "tests" / "golden" / "crosstool"

STEP_S = 60.0
N_ROWS = 10081  # 7 days at 60 s plus t = 0
CSV_HEADER = "t_s,x_m,y_m,z_m,vx_mps,vy_mps,vz_mps"


def _load_truth_csv(name: str) -> np.ndarray:
    """Frozen-truth rows as a (10081, 7) array on the exact 60 s grid."""
    lines = [
        line
        for line in (CROSSTOOL / name).read_text(encoding="ascii").splitlines()
        if line and not line.startswith("#")
    ]
    assert lines[0] == CSV_HEADER, f"{name}: unexpected column header {lines[0]!r}"
    data = np.array([[float(v) for v in line.split(",")] for line in lines[1:]])
    assert data.shape == (N_ROWS, 7), f"{name}: expected {N_ROWS} rows, got {data.shape}"
    # The freeze scripts write exact grid times; bit-equality keeps the
    # comparison free of any hidden time interpolation.
    assert np.array_equal(data[:, 0], np.arange(N_ROWS) * STEP_S), (
        f"{name}: truth epochs are not the exact 60 s grid"
    )
    return data


def _run_and_sample(mission_name: str, tmp_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Re-run the mission; return (r_m, v_mps) sampled at the 60 s epochs."""
    result = run_mission(REPO_ROOT / "missions" / mission_name, tmp_path / "run")
    truth = star_reacher.load(result.srlog_path).groups["truth"]
    idx = np.arange(N_ROWS) * int(STEP_S)
    # The 1 Hz truth grid is exact integer seconds (FR-16); the frozen 60 s
    # epochs are a bit-exact subset, so sampling is pure row selection.
    assert np.array_equal(truth["t_s"][idx], np.arange(N_ROWS) * STEP_S)
    return truth["r_m"][idx], truth["v_mps"][idx]


def _position_rms_m(r_sim: np.ndarray, truth: np.ndarray) -> float:
    dr = r_sim - truth[:, 1:4]
    return float(np.sqrt(np.mean(np.sum(dr * dr, axis=1))))


def test_xtool_leo_grav_gmat(tmp_path):
    """XTOOL-LEO-GRAV-GMAT: position RMS < 10 m vs frozen GMAT truth."""
    truth = _load_truth_csv("truth_gmat_leo_gravity_8x8.csv")
    r_sim, v_sim = _run_and_sample("leo_gravity_8x8.toml", tmp_path)
    rms = _position_rms_m(r_sim, truth)
    dv = v_sim - truth[:, 4:7]
    vrms = float(np.sqrt(np.mean(np.sum(dv * dv, axis=1))))
    print(f"XTOOL-LEO-GRAV-GMAT: position RMS {rms:.6f} m, velocity RMS {vrms:.3e} m/s")
    assert rms < 10.0, (
        f"XTOOL-LEO-GRAV-GMAT: position RMS vs frozen GMAT truth is {rms:.6f} m "
        f"(gate: < 10 m over 7 days)"
    )


def test_xtool_leo_drag_orekit(tmp_path):
    """XTOOL-LEO-DRAG-OREKIT: position RMS < 100 m vs frozen Orekit truth."""
    truth = _load_truth_csv("truth_orekit_leo_drag_hp.csv")
    r_sim, v_sim = _run_and_sample("leo_drag_hp.toml", tmp_path)
    rms = _position_rms_m(r_sim, truth)
    dv = v_sim - truth[:, 4:7]
    vrms = float(np.sqrt(np.mean(np.sum(dv * dv, axis=1))))
    print(f"XTOOL-LEO-DRAG-OREKIT: position RMS {rms:.6f} m, velocity RMS {vrms:.3e} m/s")
    assert rms < 100.0, (
        f"XTOOL-LEO-DRAG-OREKIT: position RMS vs frozen Orekit truth is {rms:.6f} m "
        f"(gate: < 100 m over 7 days)"
    )
