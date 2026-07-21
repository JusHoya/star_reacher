"""Mission-schema validation for the Phase 6 [gnc] and [sensors] tables.

Pure-Python validation tests (core-less, like the rest of the mission
validation suite), plus one vocabulary-drift check that requires the
compiled core: mission.py's static component vocabulary must equal the core
registry's built-in set, so a component added on one side cannot silently
diverge from the other.
"""

from pathlib import Path

import pytest

from star_reacher.mission import validate_mission_file

REPO_ROOT = Path(__file__).resolve().parents[2]

_CORE_MISSING_MESSAGE = (
    "star_reacher._core is not built in this environment. This registry "
    "cross-check requires the compiled core: build and install it with "
    "'pip install .' from the repository root."
)

GOLDEN_MISSION = """
schema_version = 1
vehicle = "vehicles/smallsat.toml"

[mission]
name = "gnc-validation-case"
epoch_utc = "2026-01-01T00:00:00Z"
duration_s = 10.0

[run]
seed = 1

[integrator]
type = "rk4"
dt_s = 0.1

[environment]
central_body = "earth"

[initial_state.cartesian]
r_m = [7.0e6, 0.0, 0.0]
v_mps = [0.0, 7546.0, 0.0]
frame = "GCRF"

[gnc]
control_rate_hz = 10
latency_cycles = 2
oracle = true

[gnc.nav]
component = "dead_reckoning"
q0 = [0.0, 0.7071067811865476, 0.7071067811865476, 0.0]

[gnc.guidance]
component = "attitude_hold"
q_cmd = [0.0, 0.7660444431189781, 0.6427876096865393, 0.0]

[gnc.control]
component = "pd_attitude"
kp_nm_per_rad = [0.4, 0.4, 0.4]
kd_nm_per_radps = [3.6, 3.6, 3.6]
tau_max_nm = [0.05, 0.05, 0.05]

[sensors.imu]
sample_rate_hz = 10
"""


def _write(tmp_path, text):
    p = tmp_path / "mission.toml"
    p.write_text(text, encoding="utf-8")
    return p


def _validate_text(tmp_path, text, monkeypatch):
    # Vehicle paths resolve against the working directory; validation of the
    # referenced starter vehicle needs the repository root.
    monkeypatch.chdir(REPO_ROOT)
    return validate_mission_file(_write(tmp_path, text))


def test_gnc_mission_validates_and_resolves(tmp_path, monkeypatch):
    resolved, errors = _validate_text(tmp_path, GOLDEN_MISSION, monkeypatch)
    assert not errors, errors
    gnc = resolved["gnc"]
    assert gnc["control_rate_hz"] == 10
    assert gnc["latency_cycles"] == 2
    assert gnc["oracle"] is True
    assert gnc["nav"] == {
        "component": "dead_reckoning",
        "q0": [0.0, 0.7071067811865476, 0.7071067811865476, 0.0],
    }
    assert gnc["guidance"]["component"] == "attitude_hold"
    assert len(gnc["guidance"]["q_cmd"]) == 4
    assert gnc["control"]["component"] == "pd_attitude"
    assert resolved["sensors"] == {"imu": {"sample_rate_hz": 10}}


def test_gnc_defaults_are_recorded(tmp_path, monkeypatch):
    text = GOLDEN_MISSION.replace("latency_cycles = 2\n", "").replace(
        "oracle = true\n", ""
    )
    resolved, errors = _validate_text(tmp_path, text, monkeypatch)
    assert not errors, errors
    # Defaults are applied here and recorded in the resolved config, never
    # silently (D-2).
    assert resolved["gnc"]["latency_cycles"] == 0
    assert resolved["gnc"]["oracle"] is False


def test_committed_golden_gnc_mission_validates(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    resolved, errors = validate_mission_file(
        REPO_ROOT / "missions" / "leo_attitude_gnc.toml"
    )
    assert not errors, errors
    assert resolved["gnc"]["control_rate_hz"] == 10
    assert resolved["sensors"]["imu"]["sample_rate_hz"] == 10


@pytest.mark.parametrize(
    ("mutation", "expected_fragment"),
    [
        # Missing required keys name the exact key (DX-2).
        (lambda t: t.replace("control_rate_hz = 10\n", ""), "control_rate_hz"),
        (
            lambda t: t.replace(
                '[gnc.nav]\ncomponent = "dead_reckoning"\n'
                "q0 = [0.0, 0.7071067811865476, 0.7071067811865476, 0.0]\n",
                "",
            ),
            "[gnc] nav",
        ),
        # The dead reckoner's initial estimate is explicit configuration
        # (no implicit truth access): a missing q0 is an error.
        (
            lambda t: t.replace(
                "q0 = [0.0, 0.7071067811865476, 0.7071067811865476, 0.0]\n", ""
            ),
            "q0",
        ),
        # The control cycle is the integrator step: rate must equal 1/dt_s.
        (
            lambda t: t.replace("control_rate_hz = 10", "control_rate_hz = 20"),
            "must equal 1/dt_s",
        ),
        # Unknown keys and unknown components are errors, not warnings.
        (lambda t: t.replace("[gnc]\n", "[gnc]\nwarp_factor = 9\n"), "warp_factor"),
        (
            lambda t: t.replace(
                'component = "dead_reckoning"', 'component = "kalman_9000"'
            ),
            "kalman_9000",
        ),
        # Latency must be a bounded non-negative integer.
        (
            lambda t: t.replace("latency_cycles = 2", "latency_cycles = -1"),
            "latency_cycles",
        ),
        # Oracle is a boolean, not a truthy string.
        (lambda t: t.replace("oracle = true", 'oracle = "yes"'), "oracle"),
        # Gain arrays are per-axis triples.
        (
            lambda t: t.replace(
                "kp_nm_per_rad = [0.4, 0.4, 0.4]", "kp_nm_per_rad = [0.4, 0.4]"
            ),
            "kp_nm_per_rad",
        ),
        # Saturation limits must be strictly positive.
        (
            lambda t: t.replace(
                "tau_max_nm = [0.05, 0.05, 0.05]", "tau_max_nm = [0.05, 0.0, 0.05]"
            ),
            "tau_max_nm",
        ),
        # The v1 IMU emits one increment pair per control cycle: its rate
        # must EQUAL the control rate - even an exact divisor is rejected.
        (
            lambda t: t.replace("sample_rate_hz = 10", "sample_rate_hz = 5"),
            "must equal",
        ),
        # Unknown sensor kinds are rejected by name.
        (
            lambda t: t.replace("[sensors.imu]", "[sensors.lidar]"),
            "lidar",
        ),
        # [gnc] without [sensors.imu] cannot navigate.
        (
            lambda t: t.replace("[sensors.imu]\nsample_rate_hz = 10\n", ""),
            "sensors",
        ),
        # A GNC mission needs the vehicle path.
        (
            lambda t: t.replace('vehicle = "vehicles/smallsat.toml"\n', ""),
            "vehicle",
        ),
    ],
)
def test_gnc_validation_rejections(tmp_path, monkeypatch, mutation, expected_fragment):
    resolved, errors = _validate_text(
        tmp_path, mutation(GOLDEN_MISSION), monkeypatch
    )
    assert resolved is None
    assert any(expected_fragment in e for e in errors), (expected_fragment, errors)


# The three aiding sensors whose sigmas build the EKF's measurement-noise
# matrix R, appended to GOLDEN_MISSION by the R-singularity cases below. Every
# sigma here is nonzero, so the block is accepted as written and each case
# below isolates exactly one way of driving a sigma to zero.
AIDING_SENSORS = """
[sensors.startracker]
sample_rate_hz = 1
boresight_b = [0.0, 0.0, 1.0]
sigma_rad = [1.0e-5, 1.0e-5, 5.0e-5]

[sensors.navfix]
sample_rate_hz = 1
sigma_r_m = [10.0, 10.0, 10.0]
sigma_v_mps = [0.1, 0.1, 0.1]

[sensors.altimeter]
sample_rate_hz = 1
sigma_noise_m = 20.0
sigma_bias_m = 0.0
h_min_m = 0.0
h_max_m = 0.0
"""


@pytest.mark.parametrize(
    ("mutation", "expected_fragment"),
    [
        # A zero sigma is refused for the reason a zero p0_sigma already is:
        # it makes the matrix it builds singular. Writing the zero and
        # omitting the key are separate routes to the same R, because the
        # core-side NavSensorModel members default to zero, so both are
        # covered for every key.
        (
            lambda t: t.replace(
                "sigma_rad = [1.0e-5, 1.0e-5, 5.0e-5]",
                "sigma_rad = [0.0, 0.0, 0.0]",
            ),
            "[sensors.startracker] sigma_rad",
        ),
        (
            lambda t: t.replace("sigma_rad = [1.0e-5, 1.0e-5, 5.0e-5]\n", ""),
            "[sensors.startracker] sigma_rad: missing required key",
        ),
        (
            lambda t: t.replace(
                "sigma_r_m = [10.0, 10.0, 10.0]", "sigma_r_m = [0.0, 0.0, 0.0]"
            ),
            "[sensors.navfix] sigma_r_m",
        ),
        (
            lambda t: t.replace("sigma_r_m = [10.0, 10.0, 10.0]\n", ""),
            "[sensors.navfix] sigma_r_m: missing required key",
        ),
        (
            lambda t: t.replace(
                "sigma_v_mps = [0.1, 0.1, 0.1]", "sigma_v_mps = [0.0, 0.0, 0.0]"
            ),
            "[sensors.navfix] sigma_v_mps",
        ),
        (
            lambda t: t.replace("sigma_v_mps = [0.1, 0.1, 0.1]\n", ""),
            "[sensors.navfix] sigma_v_mps: missing required key",
        ),
        # A single zeroed axis is enough: R is singular if any diagonal entry
        # vanishes, not only if all three do.
        (
            lambda t: t.replace(
                "sigma_rad = [1.0e-5, 1.0e-5, 5.0e-5]",
                "sigma_rad = [1.0e-5, 0.0, 5.0e-5]",
            ),
            "[sensors.startracker] sigma_rad",
        ),
        # The altimeter's rule is the quadrature sum, so it fires only when
        # both terms vanish.
        (
            lambda t: t.replace("sigma_noise_m = 20.0", "sigma_noise_m = 0.0"),
            "[sensors.altimeter] sigma_noise_m: sigma_noise_m**2 + "
            "sigma_bias_m**2 must be > 0",
        ),
        (
            lambda t: t.replace("sigma_noise_m = 20.0\n", ""),
            "[sensors.altimeter] sigma_noise_m: missing required key",
        ),
        (
            lambda t: t.replace("sigma_bias_m = 0.0\n", ""),
            "[sensors.altimeter] sigma_bias_m: missing required key",
        ),
    ],
)
def test_singular_r_sigmas_rejected(
    tmp_path, monkeypatch, mutation, expected_fragment
):
    """Every route to a singular R is refused, naming the offending key."""
    text = mutation(GOLDEN_MISSION + AIDING_SENSORS)
    resolved, errors = _validate_text(tmp_path, text, monkeypatch)
    assert resolved is None
    assert any(expected_fragment in e for e in errors), (
        expected_fragment, errors
    )


def test_aiding_sensor_block_is_accepted_unmutated(tmp_path, monkeypatch):
    """The control for the rejection cases above.

    Without it, a typo in AIDING_SENSORS would make every case in that
    parametrization pass for the wrong reason.
    """
    resolved, errors = _validate_text(
        tmp_path, GOLDEN_MISSION + AIDING_SENSORS, monkeypatch
    )
    assert not errors, errors
    assert resolved["sensors"]["altimeter"]["sigma_noise_m"] == 20.0


@pytest.mark.parametrize(
    ("sigma_noise_m", "sigma_bias_m"),
    [
        (0.0, 5.0),   # bias-only: white-noise-free, turn-on bias configured
        (20.0, 0.0),  # noise-only: the form the shipped missions use
        (20.0, 5.0),  # both terms present
    ],
)
def test_altimeter_accepts_either_sigma_alone(
    tmp_path, monkeypatch, sigma_noise_m, sigma_bias_m
):
    """A bias-only altimeter stays legal.

    cpp/src/gnc/ekf.cpp builds the altimeter's R entry as
    r = sigma_noise_m**2 + sigma_bias_m**2, so an instrument with no white
    noise but a configured turn-on bias has a perfectly well-conditioned R.
    Gating on sigma_noise_m alone - the obvious reading of "sigmas must be
    positive" - would reject it, so this is the case that pins the rule to
    the quadrature sum.
    """
    text = (GOLDEN_MISSION + AIDING_SENSORS).replace(
        "sigma_noise_m = 20.0", f"sigma_noise_m = {sigma_noise_m}"
    ).replace("sigma_bias_m = 0.0", f"sigma_bias_m = {sigma_bias_m}")
    resolved, errors = _validate_text(tmp_path, text, monkeypatch)
    assert not errors, errors
    altimeter = resolved["sensors"]["altimeter"]
    assert altimeter["sigma_noise_m"] == sigma_noise_m
    assert altimeter["sigma_bias_m"] == sigma_bias_m


def test_every_shipped_mission_still_validates(monkeypatch):
    """The over-rejection gate for the R-singularity rules.

    Making four sensor keys required can only be right if no mission the
    repository ships relied on omitting them; this proves it against the
    whole catalogue rather than the one mission that carries aiding sensors
    today, so a future mission cannot be added in the rejected shape.
    """
    monkeypatch.chdir(REPO_ROOT)
    missions = sorted((REPO_ROOT / "missions").glob("*.toml"))
    assert missions, "no shipped missions found"
    failures = {}
    for mission in missions:
        resolved, errors = validate_mission_file(mission)
        if resolved is None:
            failures[mission.name] = errors
    assert not failures, failures


def test_sensors_without_gnc_rejected(tmp_path, monkeypatch):
    text = GOLDEN_MISSION
    # Strip every [gnc*] table but keep [sensors.imu].
    start = text.index("[gnc]")
    end = text.index("[sensors.imu]")
    text = text[:start] + text[end:]
    resolved, errors = _validate_text(tmp_path, text, monkeypatch)
    assert resolved is None
    assert any("requires a [gnc] table" in e for e in errors), errors


def test_gnc_rejects_openloop_attitude_actions(tmp_path, monkeypatch):
    text = GOLDEN_MISSION + (
        "\n[[sequence]]\n"
        'name = "hold"\n'
        'trigger = "elapsed"\n'
        "t_s = 1.0\n"
        'action = "attitude_hold"\n'
    )
    resolved, errors = _validate_text(tmp_path, text, monkeypatch)
    assert resolved is None
    assert any("cannot be combined with [gnc]" in e for e in errors), errors


def test_pitch_program_guidance_requires_geodetic(tmp_path, monkeypatch):
    text = GOLDEN_MISSION.replace(
        'component = "attitude_hold"\n'
        "q_cmd = [0.0, 0.7660444431189781, 0.6427876096865393, 0.0]",
        'component = "pitch_program"\n'
        "azimuth_deg = 90.0\n"
        "pitch_t_s = [0.0, 10.0]\n"
        "pitch_deg = [90.0, 80.0]",
    )
    resolved, errors = _validate_text(tmp_path, text, monkeypatch)
    assert resolved is None
    assert any("geodetic" in e for e in errors), errors


def test_component_vocabulary_matches_core_registry():
    """mission.py's core-less vocabulary must equal the core registry."""
    try:
        from star_reacher import _core
    except ImportError:
        pytest.fail(_CORE_MISSING_MESSAGE)
    from star_reacher import mission

    python_side = sorted(
        set(mission._GNC_NAV_COMPONENTS)
        | set(mission._GNC_GUIDANCE_COMPONENTS)
        | set(mission._GNC_CONTROL_COMPONENTS)
    )
    core_side = sorted(
        name for name in _core.gnc_component_names()
        if not name.startswith("test_")  # doctest probe registrations
        # FR-25 plugin components live in a namespace disjoint from the
        # built-in vocabulary by design: mission.py validates them by grammar
        # because it cannot see a plugin, and the loader proves them real
        # against the file. Including them here would assert that the static
        # mirror knows names that only exist once a --gnc-plugin was passed.
        and not name.startswith(mission._GNC_PLUGIN_PREFIX)
    )
    assert python_side == core_side
