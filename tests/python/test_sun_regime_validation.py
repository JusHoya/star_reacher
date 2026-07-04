"""Sun-regime (heliocentric) mission validator tests (Phase 5, FR-15).

Mirrors test_mission_validation.py's conventions: build one valid heliocentric
mission text, then mutate it one defect at a time and assert every defect is
named with its table path and rejected (DX-2 accumulate-all, no defaults). No
compiled core is required by any test in this file.

The referenced ephemeris excerpt and gravity field are committed files, so the
path-existence checks pass on a clean clone.
"""

from pathlib import Path

from star_reacher.mission import validate_mission_file

REPO_ROOT = Path(__file__).resolve().parents[2]

# Committed fixtures (existence is all validation checks here).
_EPH = "tests/golden/ephemeris/excerpt_de440s_mars_cruise.sreph"
_FIELD = "tests/golden/gravity/earth_egm2008_n20.srgrav"

# One tagged line per schema entry, so each test deletes or replaces exactly
# one surface (the test_mission_validation.py pattern).
_VALID_LINES = [
    ("schema_version", "schema_version = 1"),
    ("mission", "[mission]"),
    ("mission.name", 'name = "sun-regime-unit-test"'),
    ("mission.epoch_utc", 'epoch_utc = "2026-12-05T00:00:00Z"'),
    ("mission.duration_s", "duration_s = 600.0"),
    ("run", "[run]"),
    ("run.seed", "seed = 7"),
    ("integrator", "[integrator]"),
    ("integrator.type", 'type = "rk4"'),
    # dt = 0.1 s keeps 1/(dt_s * truth_rate_hz) integral at the default
    # 10 Hz truth rate (the Phase 1 decimation rule).
    ("integrator.dt_s", "dt_s = 0.1"),
    ("spacecraft", "[spacecraft]"),
    ("spacecraft.mass_kg", "mass_kg = 500.0"),
    ("spacecraft.cr", "cr_a_over_m_m2pkg = 0.013"),
    ("initial_state.cartesian", "[initial_state.cartesian]"),
    ("initial_state.cartesian.r_m", "r_m = [43441832987.8, 129269774755.0, 56035879787.2]"),
    ("initial_state.cartesian.v_mps", "v_mps = [-31803.3, 8954.5, 3882.3]"),
    ("initial_state.cartesian.frame", 'frame = "GCRF"'),
    ("environment", "[environment]"),
    ("environment.central_body", 'central_body = "sun"'),
    ("environment.third_bodies", 'third_bodies = ["earth", "mars"]'),
    ("environment.ephemeris", f'ephemeris = "{_EPH}"'),
    ("environment.srp", "[environment.srp]"),
]


def _mission_text(exclude=(), extra_lines=(), replace=None, prelude=()):
    replace = replace or {}
    lines = list(prelude)
    for tag, line in _VALID_LINES:
        if any(tag == ex or tag.startswith(ex + ".") for ex in exclude):
            continue
        lines.append(replace.get(tag, line))
    lines.extend(extra_lines)
    return "\n".join(lines) + "\n"


def _validate_text(tmp_path, text):
    # Relative fixture paths resolve against the working directory, so the
    # validator must run from the repository root (the documented rule).
    path = tmp_path / "mission.toml"
    path.write_text(text, encoding="utf-8")
    import os

    cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        return validate_mission_file(path)
    finally:
        os.chdir(cwd)


def test_valid_sun_mission_accepted_and_resolved(tmp_path):
    resolved, errors = _validate_text(tmp_path, _mission_text())
    assert errors == []
    env = resolved["environment"]
    assert env["central_body"] == "sun"
    # Canonical third-body order (D-10), independent of file order.
    assert env["third_bodies"] == ["earth", "mars"]
    # The sun regime records SRP with an explicit empty occulter set.
    assert env["srp"] == {"occulters": []}
    assert env["ephemeris"] == _EPH


def test_committed_mars_cruise_mission_validates_cleanly():
    import os

    cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        resolved, errors = validate_mission_file(REPO_ROOT / "missions" / "mars_cruise.toml")
    finally:
        os.chdir(cwd)
    assert errors == []
    env = resolved["environment"]
    assert env["central_body"] == "sun"
    assert env["third_bodies"] == ["earth", "moon", "venus", "mars", "jupiter"]
    assert env["srp"] == {"occulters": []}


def test_sun_rejects_drag(tmp_path):
    text = _mission_text(
        extra_lines=("[environment.drag]", 'atmosphere = "ussa76"'),
        replace={"spacecraft.cr": "cr_a_over_m_m2pkg = 0.013\ncd_a_over_m_m2pkg = 0.004"},
    )
    resolved, errors = _validate_text(tmp_path, text)
    assert resolved is None
    assert any(
        "[environment.drag] atmosphere:" in e and "heliocentric" in e for e in errors
    ), errors


def test_sun_rejects_unknown_keys(tmp_path):
    # Unknown-key rejection (FR-15) holds unchanged in the sun regime, and
    # every error is accumulated in one report (DX-2).
    text = _mission_text(
        prelude=("unknown_top = 1",),
        extra_lines=("[environment.drag]", 'atmosphere = "ussa76"'),
        replace={
            "spacecraft.cr": "cr_a_over_m_m2pkg = 0.013\ncd_a_over_m_m2pkg = 0.004"
        },
    )
    resolved, errors = _validate_text(tmp_path, text)
    assert resolved is None
    joined = "\n".join(errors)
    assert "[root] unknown_top: unknown key" in joined
    assert "[environment.drag] atmosphere:" in joined
    assert all("No default applied; run aborted." in e for e in errors)


def test_sun_requires_explicit_third_bodies(tmp_path):
    # Absent key: heliocentric two-body motion is never a silent default.
    resolved, errors = _validate_text(
        tmp_path, _mission_text(exclude=("environment.third_bodies",))
    )
    assert resolved is None
    assert any(
        "[environment] third_bodies:" in e and "explicit non-empty" in e for e in errors
    ), errors

    # Empty list: an explicit empty set is equally rejected.
    resolved, errors = _validate_text(
        tmp_path,
        _mission_text(replace={"environment.third_bodies": "third_bodies = []"}),
    )
    assert resolved is None
    assert any(
        "[environment] third_bodies:" in e and "explicit non-empty" in e for e in errors
    ), errors


def test_sun_cannot_be_its_own_third_body(tmp_path):
    resolved, errors = _validate_text(
        tmp_path,
        _mission_text(
            replace={"environment.third_bodies": 'third_bodies = ["sun", "earth"]'}
        ),
    )
    assert resolved is None
    assert any("cannot also be a third body" in e for e in errors), errors


def test_sun_rejects_srp_occulters_key(tmp_path):
    resolved, errors = _validate_text(
        tmp_path,
        _mission_text(extra_lines=('occulters = ["earth"]',)),  # inside [environment.srp]
    )
    assert resolved is None
    assert any(
        "[environment.srp] occulters:" in e and 'not accepted for central_body = "sun"' in e
        for e in errors
    ), errors


def test_sun_rejects_harmonic_gravity(tmp_path):
    for model_lines in (
        ("[environment.gravity]", 'model = "j2"', f'field = "{_FIELD}"'),
        ("[environment.gravity]", 'model = "harmonic"', f'field = "{_FIELD}"', "degree = 8", "order = 8"),
    ):
        resolved, errors = _validate_text(tmp_path, _mission_text(extra_lines=model_lines))
        assert resolved is None
        assert any(
            "[environment.gravity] model:" in e and "point-mass only" in e for e in errors
        ), errors

    # Explicit pointmass stays accepted (it is the sun regime's only tier).
    resolved, errors = _validate_text(
        tmp_path,
        _mission_text(extra_lines=("[environment.gravity]", 'model = "pointmass"')),
    )
    assert errors == []
    assert resolved["environment"]["gravity"] == {"model": "pointmass"}


def test_sun_rejects_vehicle_and_sequence(tmp_path):
    text = _mission_text(
        prelude=('vehicle = "vehicles/probe.toml"',),
        extra_lines=(
            "[[sequence]]",
            'name = "stop"',
            'trigger = "elapsed"',
            "t_s = 100.0",
            'action = "terminate"',
        ),
    )
    resolved, errors = _validate_text(tmp_path, text)
    assert resolved is None
    joined = "\n".join(errors)
    assert "[root] vehicle:" in joined and 'central_body = "sun"' in joined
    assert "[root] sequence:" in joined


def test_sun_rejects_geodetic_launch_form(tmp_path):
    text = _mission_text(
        exclude=("initial_state.cartesian",),
        extra_lines=(
            "[initial_state.geodetic]",
            "lat_deg = 28.5",
            "lon_deg = -80.6",
            "alt_m = 10.0",
        ),
    )
    resolved, errors = _validate_text(tmp_path, text)
    assert resolved is None
    # The existing FR-14 rule already pins the launch form to Earth.
    assert any('requires central_body = "earth"' in e for e in errors), errors


def test_soi_transition_body_sun_rejected(tmp_path):
    # The FR-12 SOI event vocabulary stays planetary even though "sun" is now
    # a central body: entering "the Sun's SOI" has no patched-conic meaning.
    # Exercised on an Earth-central mission because the sun regime rejects
    # sequences outright.
    text = _mission_text(
        prelude=('vehicle = "vehicles/probe.toml"',),
        replace={
            "environment.central_body": 'central_body = "earth"',
            "environment.third_bodies": 'third_bodies = ["sun", "moon"]',
        },
        extra_lines=(
            "[[sequence]]",
            'name = "soi"',
            'trigger = "condition"',
            'condition = "soi_transition"',
            'body = "sun"',
            'action = "terminate"',
        ),
    )
    resolved, errors = _validate_text(tmp_path, text)
    assert resolved is None
    assert any(
        'must be one of "earth", "moon", "mars"' in e and "sun" in e for e in errors
    ), errors
