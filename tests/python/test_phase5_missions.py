"""End-to-end tests of the committed Phase 5 example missions (FR-32).

Both missions run from their committed definitions on a clean clone: the
gravity field and both ephemeris excerpts are committed goldens, so no fetched
data is required. Mission A (missions/mission_a_cislunar.toml) is the FR-32
performance-gate cislunar transfer; the Mars cruise
(missions/mars_cruise.toml) is the first heliocentric (sun-central) mission.
These tests gate what CI can gate: bit-identical double runs (FR-21/D-10),
correct termination and record accounting, and physics sanity (a real lunar
flyby inside the 4.5-day window; a heliocentric transfer ellipse whose
aphelion sits in the Mars band). The Pi 5 wall-clock gate itself runs on the
pinned performance runner, not here.

Like test_crosstool_missions.py, these REQUIRE the compiled core and fail,
never skip, without it.
"""

import math
from datetime import datetime, timezone
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

# DE440 GM values (cpp/include/star/constants.hpp cites Park et al. 2021); the
# binding agreement with these is asserted in test_sun_gm_single_home below.
_GM_SUN = 1.32712440041279419e20
_AU_M = 149597870700.0
_R_MOON_M = 1737400.0


def _core_or_fail():
    try:
        from star_reacher import _core
    except ImportError:
        pytest.fail(_CORE_MISSING_MESSAGE)
    return _core


def _run_twice(mission_name, tmp_path):
    from star_reacher.runner import run_mission

    mission = REPO_ROOT / "missions" / mission_name
    r1 = run_mission(mission, tmp_path / "run1")
    r2 = run_mission(mission, tmp_path / "run2")
    return r1, r2


def _t0_tdb_s(core, epoch_utc):
    """TDB seconds since J2000 of the mission epoch (via the bound leap table)."""
    m = datetime.fromisoformat(epoch_utc.replace("Z", "+00:00")).astimezone(timezone.utc)
    day, sec = core.utc_to_tai(
        m.year, m.month, m.day, m.hour, m.minute, m.second + m.microsecond * 1e-6
    )
    jd1, jd2 = core.tdb_jd(day, sec)
    return ((jd1 - 2451545.0) + jd2) * 86400.0


def test_sun_gm_single_home():
    # The Sun GM crosses the binding from its single home in constants.hpp
    # (DE440 header value): a drift here would silently rescale every
    # heliocentric mission.
    core = _core_or_fail()
    assert core.gm("sun") == _GM_SUN


def test_mission_a_flyby_and_determinism(tmp_path):
    core = _core_or_fail()
    import star_reacher
    from star_reacher.mission import validate_mission_file

    resolved, errors = validate_mission_file(
        REPO_ROOT / "missions" / "mission_a_cislunar.toml"
    )
    assert not errors, errors

    r1, r2 = _run_twice("mission_a_cislunar.toml", tmp_path)
    # FR-21/D-10: the whole 4.5-day perturbed run is bit-identical.
    assert r1.srlog_sha256 == r2.srlog_sha256
    # 4.5 d at 1 Hz plus t = 0 (duration termination, no event sequence).
    assert r1.summary["truth_records"] == 388801

    run = star_reacher.load(r1.srlog_path)
    truth = run.groups["truth"]
    t = np.asarray(truth["t_s"])
    assert len(truth) == 388801
    assert np.all(np.diff(t) > 0)

    # Physics gate: the coast is a real lunar flyby INSIDE the 4.5-day
    # window. The mission header documents the targeting: closest approach
    # ~1.94e7 m from the Moon's center at t ~ 3.71 d. Gates:
    #   - the minimum lunar center distance lies below the ~6.6e7 m mean
    #     Earth-Moon patched-conic SOI radius (a_moon (m_moon/m_earth)^0.4,
    #     the FR-12 SOI convention) - the spacecraft demonstrably enters the
    #     Moon's dynamical neighborhood, not merely its general direction;
    #   - it stays above the 1.7374e6 m lunar radius plus margin (no impact);
    #   - the minimum falls strictly inside the log (at least an hour from
    #     both ends), i.e. the perilune passage itself is captured, which is
    #     what makes this a 4.5-day TRANSFER rather than an outbound coast.
    eph = core.Ephemeris.load(resolved["environment"]["ephemeris"])
    t0 = _t0_tdb_s(core, resolved["mission"]["epoch_utc"])
    r_sc = np.asarray(truth["r_m"])
    # Coarse 30 s scan brackets the minimum; a fine pass pins it to 1 s.
    sel = np.arange(0, len(t), 30)
    d_coarse = np.empty(len(sel))
    for j, k in enumerate(sel):
        r_moon, _ = eph.moon_geocentric(t0 + float(t[k]))
        d_coarse[j] = np.linalg.norm(r_sc[k] - np.asarray(r_moon))
    j_min = int(np.argmin(d_coarse))
    lo = max(0, int(sel[j_min]) - 60)
    hi = min(len(t), int(sel[j_min]) + 60)
    d_min, t_min = math.inf, 0.0
    for k in range(lo, hi):
        r_moon, _ = eph.moon_geocentric(t0 + float(t[k]))
        d = float(np.linalg.norm(r_sc[k] - np.asarray(r_moon)))
        if d < d_min:
            d_min, t_min = d, float(t[k])
    soi_radius_m = 3.84e8 * (4.902800118e12 / 3.986004418e14) ** 0.4
    assert d_min < soi_radius_m, (d_min, soi_radius_m)
    assert d_min > _R_MOON_M + 1.0e5, d_min
    assert 3600.0 < t_min < 388800.0 - 3600.0, t_min


def test_mars_cruise_transfer_and_determinism(tmp_path):
    core = _core_or_fail()
    import star_reacher
    from star_reacher.mission import validate_mission_file

    resolved, errors = validate_mission_file(REPO_ROOT / "missions" / "mars_cruise.toml")
    assert not errors, errors
    assert resolved["environment"]["central_body"] == "sun"

    r1, r2 = _run_twice("mars_cruise.toml", tmp_path)
    assert r1.srlog_sha256 == r2.srlog_sha256  # FR-21 determinism
    assert r1.summary["truth_records"] == 604801  # 7 d at 1 Hz plus t = 0

    run = star_reacher.load(r1.srlog_path)
    assert run.header["central_body"] == "sun"
    truth = run.groups["truth"]
    t = np.asarray(truth["t_s"])
    assert len(truth) == 604801
    assert np.all(np.diff(t) > 0)

    r = np.asarray(truth["r_m"])
    v = np.asarray(truth["v_mps"])
    rn0 = float(np.linalg.norm(r[0]))
    rnf = float(np.linalg.norm(r[-1]))

    # Physics gates on the final state's osculating heliocentric ellipse
    # (two-body about the Sun; the mission header documents the measured
    # values: aphelion 1.552 au, perihelion 0.986 au after 7 days):
    #   - aphelion in [1.30, 1.75] au: brackets Mars' heliocentric radial
    #     range (perihelion 1.381 au to aphelion 1.666 au for a = 1.5237 au,
    #     e = 0.0934) plus the residual escape excess the handoff carries -
    #     a dropped Sun GM or a wrong central body moves this out of band
    #     immediately;
    #   - perihelion in [0.90, 1.05] au: the departure end of a Hohmann-class
    #     ellipse stays at Earth's orbital radius;
    #   - heliocentric distance receding (outbound leg of the transfer);
    #   - Earth-relative distance grows beyond 2e9 m: v_inf ~ 2.9 km/s over
    #     7 days (~1.8e9 m) plus the 1e9 m handoff offset, i.e. the probe
    #     really escapes the departure planet rather than orbiting it.
    energy = 0.5 * float(np.dot(v[-1], v[-1])) - _GM_SUN / rnf
    a = -_GM_SUN / (2.0 * energy)
    h = np.cross(r[-1], v[-1])
    e = float(np.linalg.norm(np.cross(v[-1], h) / _GM_SUN - r[-1] / rnf))
    aphelion_au = a * (1.0 + e) / _AU_M
    perihelion_au = a * (1.0 - e) / _AU_M
    assert 1.30 < aphelion_au < 1.75, aphelion_au
    assert 0.90 < perihelion_au < 1.05, perihelion_au
    assert rnf > rn0, (rn0, rnf)

    eph = core.Ephemeris.load(resolved["environment"]["ephemeris"])
    t0 = _t0_tdb_s(core, resolved["mission"]["epoch_utc"])

    def earth_helio(tdb_s):
        r_emb, _ = eph.state("emb", tdb_s)
        r_e, _ = eph.state("earth", tdb_s)
        r_sun, _ = eph.state("sun", tdb_s)
        return np.asarray(r_emb) + np.asarray(r_e) - np.asarray(r_sun)

    d_earth_f = float(np.linalg.norm(r[-1] - earth_helio(t0 + float(t[-1]))))
    assert d_earth_f > 2.0e9, d_earth_f


def test_missions_use_committed_excerpts():
    # Both missions must stay pinned to committed excerpts so they run in CI
    # and on clean clones (and so the config hashes are stable).
    text_a = (REPO_ROOT / "missions" / "mission_a_cislunar.toml").read_text(encoding="utf-8")
    assert 'ephemeris = "tests/golden/ephemeris/excerpt_de440s_crosstool.sreph"' in text_a
    assert 'field = "tests/golden/gravity/earth_egm2008_n20.srgrav"' in text_a
    text_m = (REPO_ROOT / "missions" / "mars_cruise.toml").read_text(encoding="utf-8")
    assert 'ephemeris = "tests/golden/ephemeris/excerpt_de440s_mars_cruise.sreph"' in text_m
    for rel in (
        "tests/golden/ephemeris/excerpt_de440s_crosstool.sreph",
        "tests/golden/ephemeris/excerpt_de440s_mars_cruise.sreph",
        "tests/golden/gravity/earth_egm2008_n20.srgrav",
    ):
        assert (REPO_ROOT / rel).is_file(), rel
