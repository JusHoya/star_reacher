"""End-to-end tests of the committed cross-tool missions (Phase 3, D-15).

Both missions run from their committed definitions on a clean clone: the
gravity field is the committed EGM2008 excerpt and the drag case's ephemeris
is the committed continuous DE440 excerpt, so no fetched data is required.
The frozen GMAT/Orekit comparison itself is maintainer-side (workstream E;
tests/golden/crosstool/README.md); these tests gate what CI can gate:
bit-identical double runs (FR-21/D-10), the truth-log contract, and sane
physics (bounded orbit; drag-only secular energy loss).

Like test_integration_core.py, these REQUIRE the compiled core and fail,
never skip, without it.
"""

import math
from pathlib import Path

import numpy as np
import pytest

import star_reacher
from star_reacher.data_fetch import read_srgrav
from star_reacher.runner import run_mission

REPO_ROOT = Path(__file__).resolve().parents[2]
FIELD_PATH = REPO_ROOT / "tests" / "golden" / "gravity" / "earth_egm2008_n20.srgrav"

# Central-body attraction in these missions comes from the EGM2008 field, so
# the energy diagnostics must use the field's own GM/R/C20 triple (never the
# IERS GM used by the two-body model; the split is deliberate, see
# cpp/include/star/constants.hpp).
_FIELD = read_srgrav(FIELD_PATH)
_GM = _FIELD.gm_m3ps2
_REF_RADIUS = _FIELD.ref_radius_m
_J2 = -math.sqrt(5.0) * _FIELD.cbar[2, 0]


def _run_twice(mission_name: str, tmp_path: Path):
    mission = REPO_ROOT / "missions" / mission_name
    r1 = run_mission(mission, tmp_path / "run1")
    r2 = run_mission(mission, tmp_path / "run2")
    return r1, r2


def _j2_energy(truth) -> np.ndarray:
    """Specific orbital energy including the J2 potential term [J/kg].

    eps = v^2/2 - (GM/r) * (1 - J2 (R/r)^2 P2(z/r)), P2(s) = (3 s^2 - 1)/2.
    With the J2 term included, the only secular trend left in an 8x8-gravity
    trajectory is drag (the omitted J3..J8 terms contribute a zero-mean
    oscillation ~2.5e-6 of |eps|), which is what the drag test measures.
    """
    r = truth["r_m"]
    v = truth["v_mps"]
    rn = np.linalg.norm(r, axis=1)
    s = r[:, 2] / rn
    p2 = 0.5 * (3.0 * s * s - 1.0)
    potential = -(_GM / rn) * (1.0 - _J2 * (_REF_RADIUS / rn) ** 2 * p2)
    return 0.5 * np.sum(v * v, axis=1) + potential


def _day_mean(t: np.ndarray, eps: np.ndarray, day_start_s: float) -> float:
    mask = (t >= day_start_s) & (t < day_start_s + 86400.0)
    return float(np.mean(eps[mask]))


def test_leo_gravity_8x8_runs_and_is_deterministic(tmp_path):
    r1, r2 = _run_twice("leo_gravity_8x8.toml", tmp_path)
    # FR-21/D-10: the whole 7-day perturbed run is bit-identical.
    assert r1.srlog_sha256 == r2.srlog_sha256
    assert r1.summary["truth_records"] == 604801  # 7 d at 1 Hz plus t = 0

    run = star_reacher.load(r1.srlog_path)
    truth = run.groups["truth"]
    t = truth["t_s"]
    assert len(truth) == 604801
    assert np.all(np.diff(t) > 0)

    # Bounded orbit: the initial conditions put perigee ~6919 km and apogee
    # 7000 km; J2-class perturbations move the osculating radius by ~10 km.
    rn = np.linalg.norm(truth["r_m"], axis=1)
    assert rn.min() > 6.85e6
    assert rn.max() < 7.05e6

    # Gravity only: no secular energy trend. The windowed-mean diagnostic
    # carries a residual from non-integer-orbit windows over the omitted
    # J3..J8 oscillations (measured -45 J/kg, 1.6e-6 of |eps|); 150 J/kg
    # bounds that with margin while the drag case's measured -481 J/kg
    # signal would still fail it decisively.
    eps = _j2_energy(truth)
    delta = _day_mean(t, eps, 604800.0 - 86400.0) - _day_mean(t, eps, 0.0)
    assert abs(delta) < 150.0, f"gravity-only energy drift {delta} J/kg"


def test_leo_drag_hp_runs_and_loses_energy(tmp_path):
    r1, r2 = _run_twice("leo_drag_hp.toml", tmp_path)
    assert r1.srlog_sha256 == r2.srlog_sha256
    assert r1.summary["truth_records"] == 604801

    run = star_reacher.load(r1.srlog_path)
    truth = run.groups["truth"]
    t = truth["t_s"]

    # Bounded orbit around the ~6878 x 6895 km initial geometry.
    rn = np.linalg.norm(truth["r_m"], axis=1)
    assert rn.min() > 6.82e6
    assert rn.max() < 6.95e6

    # Drag is the only dissipative term: the J2-inclusive energy must
    # decrease secularly. Measured 7-day loss with the Harris-Priester
    # density at this geometry: -481 J/kg; the diagnostic's gravity-only
    # residual is -45 J/kg (see the sibling test), so -200 J/kg separates
    # a live drag path from windowing noise with >2x margin on both sides.
    eps = _j2_energy(truth)
    delta = _day_mean(t, eps, 604800.0 - 86400.0) - _day_mean(t, eps, 0.0)
    assert delta < -200.0, f"drag energy change {delta} J/kg (expected < -200)"


def test_drag_mission_uses_committed_excerpt():
    # The mission must stay pinned to the committed continuous excerpt so it
    # runs in CI and on clean clones (and so the config hash is stable).
    text = (REPO_ROOT / "missions" / "leo_drag_hp.toml").read_text(encoding="utf-8")
    assert 'ephemeris = "tests/golden/ephemeris/excerpt_de440s_crosstool.sreph"' in text
    assert (
        REPO_ROOT / "tests" / "golden" / "ephemeris" / "excerpt_de440s_crosstool.sreph"
    ).is_file()


def test_missions_reject_when_run_from_elsewhere(tmp_path, monkeypatch):
    # The mission's relative field/ephemeris paths resolve against the
    # working directory; from a different cwd validation must abort with the
    # DX-2 file-not-found errors, never fall back to a default (FR-15).
    from star_reacher.mission import validate_mission_file

    monkeypatch.chdir(tmp_path)
    resolved, errors = validate_mission_file(
        REPO_ROOT / "missions" / "leo_drag_hp.toml"
    )
    assert resolved is None
    assert any("[environment.gravity] field:" in e for e in errors), errors
    assert any("No default applied; run aborted." in e for e in errors)
