"""System-level Phase 4 vehicle tests that REQUIRE the compiled core.

These drive the real run_vehicle path (via the CLI runner and via a directly
constructed config) rather than a parallel one, so a green suite means the
whole 6DOF contract holds. They fail cleanly, never skip, when the core is
absent (the project's agent-honesty gate): expected to fail on a core-less
checkout and to pass once ``pip install .`` has built the wheel.

Covers Phase 4 exit criteria 6 (both scripted missions reach their targets and
rerun SHA-256-identical) and 3 (a +10 s upper-stage Isp edit moves the burnout
velocity by the rocket-equation-predicted amount) at the system level; the
closed-form staging/pad/actuator criteria (2, 5, 7, 10) are the C++ doctest
``test_vehicle6dof.cpp``.
"""

import hashlib
import math
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

_CORE_MISSING_MESSAGE = (
    "star_reacher._core is not built in this environment. These integration "
    "tests require the compiled core: build and install it with 'pip install .' "
    "from the repository root. This failure is expected on a core-less checkout "
    "and must be green at integration/CI."
)


def _core_or_fail():
    try:
        from star_reacher import _core
    except ImportError:
        pytest.fail(_CORE_MISSING_MESSAGE)
    return _core


def _osculating_apsides(core, r, v, radius_m):
    """Perigee and apogee altitude [m] above ``radius_m`` from a state."""
    mu = core.gm("earth")
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    rn = np.linalg.norm(r)
    energy = 0.5 * np.dot(v, v) - mu / rn
    a = -mu / (2.0 * energy)
    h = np.cross(r, v)
    e = np.linalg.norm(np.cross(v, h) / mu - r / rn)
    return a * (1.0 - e) - radius_m, a * (1.0 + e) - radius_m


def test_ascent_leo_reaches_target_and_reruns_identical(tmp_path):
    core = _core_or_fail()
    from star_reacher import load
    from star_reacher.mission import validate_mission_file
    from star_reacher.runner import run_mission

    mission = REPO_ROOT / "missions" / "ascent_leo.toml"
    resolved, errors = validate_mission_file(mission)
    assert not errors, errors
    target = resolved["mission"]["target_apoapsis_alt_m"]

    r1 = run_mission(mission, tmp_path / "a1")
    r2 = run_mission(mission, tmp_path / "a2")
    # FR-21 determinism: byte-identical reruns on the same binary.
    assert r1.srlog_sha256 == r2.srlog_sha256

    run = load(r1.srlog_path)
    # EC-6: terminates on the orbit-insertion event.
    details = [str(d) for d in run.events["detail"]]
    assert any("insertion" in d for d in details)

    truth = run.groups["truth"]
    perigee, apogee = _osculating_apsides(
        core, truth["r_m"][-1], truth["v_mps"][-1], 6378137.0
    )
    # EC-6: final osculating perigee >= 180 km, apoapsis within 5 % of target.
    assert perigee >= 180_000.0, perigee
    assert abs(apogee - target) <= 0.05 * target, (apogee, target)

    # The v1.1 vehicle channel groups are present and populated.
    for group in ("forces", "mass", "env"):
        assert group in run.groups and len(run.groups[group]) > 0
    # resolved_vehicle.toml is written beside resolved_config.json.
    assert (r1.outdir / "resolved_vehicle.toml").is_file()


def _tdb_s_of_utc(core, iso):
    from datetime import datetime, timezone

    m = datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(timezone.utc)
    day, sec = core.utc_to_tai(
        m.year, m.month, m.day, m.hour, m.minute, m.second + m.microsecond * 1e-6
    )
    jd1, jd2 = core.tdb_jd(day, sec)
    return ((jd1 - 2451545.0) + jd2) * 86400.0


def test_tli_reaches_lunar_soi_and_reruns_identical(tmp_path):
    core = _core_or_fail()
    from star_reacher import load
    from star_reacher.mission import validate_mission_file
    from star_reacher.runner import run_mission

    mission = REPO_ROOT / "missions" / "tli.toml"
    resolved, errors = validate_mission_file(mission)
    assert not errors, errors
    target = resolved["mission"]["target_perilune_alt_m"]
    eph_path = resolved["environment"]["ephemeris"]

    r1 = run_mission(mission, tmp_path / "t1")
    r2 = run_mission(mission, tmp_path / "t2")
    assert r1.srlog_sha256 == r2.srlog_sha256  # FR-21 determinism

    run = load(r1.srlog_path)
    # EC-6: the Earth->Moon SOI-transition event is logged (and terminal).
    details = [str(d) for d in run.events["detail"]]
    assert any("soi_transition" in d for d in details), details

    # EC-6: perilune (osculating periapsis about the Moon of the SOI-entry
    # state, patched-conic) within 10 % of the mission-file target.
    truth = run.groups["truth"]
    eph = core.Ephemeris.load(eph_path)
    t0 = _tdb_s_of_utc(core, resolved["mission"]["epoch_utc"])
    r_sc = np.asarray(truth["r_m"][-1])
    v_sc = np.asarray(truth["v_mps"][-1])
    r_m, v_m = eph.moon_geocentric(t0 + float(truth["t_s"][-1]))
    rr = r_sc - np.asarray(r_m)
    vv = v_sc - np.asarray(v_m)
    mu_moon = 4.902800118e12
    rn = np.linalg.norm(rr)
    a = -mu_moon / (2.0 * (0.5 * np.dot(vv, vv) - mu_moon / rn))
    e = np.linalg.norm(np.cross(vv, np.cross(rr, vv)) / mu_moon - rr / rn)
    perilune = a * (1.0 - e) - 1737400.0
    assert abs(perilune - target) <= 0.10 * target, (perilune, target)


def _minimal_vacuum_vehicle(core, isp_s):
    """A single-stage vacuum test vehicle differing only in engine Isp."""
    tank = core.TankCfg()
    tank.radius_m = 0.5
    tank.length_m = 2.0
    tank.position_m = (1.0, 0.0, 0.0)
    tank.propellant_mass_kg = 1000.0
    tank.density_kgpm3 = 1000.0

    engine = core.EngineCfg()
    engine.name = "main"
    engine.feeds_tank_index = 0
    engine.thrust_vac_N = 20000.0
    engine.isp_vac_s = isp_s
    engine.exit_area_m2 = 0.1
    engine.position_m = (0.0, 0.0, 0.0)
    engine.axis = (1.0, 0.0, 0.0)
    engine.gimbal_max_deg = 0.0
    engine.gimbal_rate_dps = 0.0
    engine.throttle_min = 1.0
    engine.throttle_max = 1.0
    engine.spool_time_s = 0.0
    engine.ignitions = 1

    stage = core.StageCfg()
    stage.name = "s"
    stage.dry_mass_kg = 200.0
    stage.dry_cg_m = (1.0, 0.0, 0.0)
    stage.dry_inertia_kgm2 = [200.0, 0.0, 0.0, 0.0, 400.0, 0.0, 0.0, 0.0, 400.0]
    stage.tanks = [tank]
    stage.engines = [engine]

    vc = core.VehicleConfig()
    vc.stages = [stage]
    return vc


def _burnout_speed(core, tmp_path, isp_s, name):
    """Fly a straight vacuum burn far from Earth and return the speed gain."""
    ignite = core.SequenceEntry()
    ignite.name = "ig"
    ignite.trigger = "elapsed"
    ignite.t_s = 0.0
    ignite.action = "ignite_engine"
    ignite.stage = "s"
    ignite.engine = "main"

    cfg = core.RunConfig()
    cfg.epoch_utc = "2026-01-01T00:00:00Z"
    cfg.duration_s = 400.0
    cfg.integrator = "rk4"
    cfg.central_body = "earth"
    cfg.dt_s = 0.05
    cfg.truth_rate_hz = 10
    cfg.master_seed = 1
    cfg.config_sha256 = "0" * 64
    cfg.initial_form = "cartesian"
    # Far from Earth so point-mass gravity is negligible over the burn
    # (mu/r^2 ~ 2e-3 m/s^2 at 4e8 m); the initial speed sets the prograde axis.
    cfg.r0_m = (4.0e8, 0.0, 0.0)
    cfg.v0_mps = (0.0, 100.0, 0.0)
    cfg.vehicle = _minimal_vacuum_vehicle(core, isp_s)
    cfg.sequence = [ignite]

    from star_reacher import load

    out = tmp_path / name
    out.mkdir(parents=True, exist_ok=True)
    core.run_vehicle(cfg, str(out / "run.srlog"))
    truth = load(out / "run.srlog").groups["truth"]
    v0 = np.linalg.norm(truth["v_mps"][0])
    vf = np.linalg.norm(truth["v_mps"][-1])
    return vf - v0


def test_isp_edit_moves_burnout_velocity_by_rocket_equation(tmp_path):
    core = _core_or_fail()
    g0 = 9.80665
    m0 = 200.0 + 1000.0  # dry + propellant
    mf = 200.0
    dv_base = _burnout_speed(core, tmp_path, 320.0, "isp_base")
    dv_edit = _burnout_speed(core, tmp_path, 330.0, "isp_edit")

    # EC-3: the burnout velocity matches Tsiolkovsky, and a +10 s Isp edit
    # moves it by the rocket-equation-predicted amount, both within 1 %.
    tsiol_base = 320.0 * g0 * math.log(m0 / mf)
    predicted_change = 10.0 * g0 * math.log(m0 / mf)
    assert abs(dv_base - tsiol_base) / tsiol_base < 0.01
    assert abs((dv_edit - dv_base) - predicted_change) / predicted_change < 0.01
