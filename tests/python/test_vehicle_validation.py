"""Vehicle TOML validator tests (FR-13/FR-15, DX-2, Phase 4 exit criterion 1).

Covers the four validation passes on a synthesized minimal vehicle plus the
committed malformed-fixture corpus, the warning tier with --strict promotion,
and the resolved-config echo byte-identity. The fleet-level mutation gate
(every required key of every starter vehicle) lives in test_vehicle_fleet.py.
No compiled core is required by any test in this file.
"""

import math
from pathlib import Path

import pytest

from star_reacher.mission import config_sha256
from star_reacher.vehicle import (
    REQUIRED_KEYS,
    canonical_vehicle_toml,
    validate_vehicle_file,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "vehicles"

# One tagged line per schema entry so mutation tests can delete exactly one
# key (or one whole block with its children) and re-serialize, mirroring the
# mission validator's test convention.
_VALID_LINES = [
    ("schema_version", "schema_version = 1"),
    ("provenance", 'provenance = "representative"'),
    ("vehicle", "[vehicle]"),
    ("vehicle.name", 'name = "unit-test-vehicle"'),
    ("stage.1", "[[stage]]"),
    ("stage.1.name", 'name = "stage1"'),
    ("stage.1.dry_mass_kg", "dry_mass_kg = 500.0"),
    ("stage.1.dry_cg_m", "dry_cg_m = [2.0, 0.0, 0.0]"),
    (
        "stage.1.dry_inertia_kgm2",
        "dry_inertia_kgm2 = [[150.0, 0.0, 0.0], [0.0, 1200.0, 0.0], [0.0, 0.0, 1200.0]]",
    ),
    ("stage.1.tank.1", "[[stage.tank]]"),
    ("stage.1.tank.1.name", 'name = "main_tank"'),
    ("stage.1.tank.1.radius_m", "radius_m = 0.5"),
    ("stage.1.tank.1.length_m", "length_m = 3.0"),
    ("stage.1.tank.1.position_m", "position_m = [2.0, 0.0, 0.0]"),
    ("stage.1.tank.1.propellant_mass_kg", "propellant_mass_kg = 2000.0"),
    ("stage.1.tank.1.density_kgpm3", "density_kgpm3 = 1030.0"),
    ("stage.1.engine.1", "[[stage.engine]]"),
    ("stage.1.engine.1.name", 'name = "main_engine"'),
    ("stage.1.engine.1.feeds_tank", 'feeds_tank = "main_tank"'),
    ("stage.1.engine.1.thrust_vac_N", "thrust_vac_N = 60000.0"),
    ("stage.1.engine.1.isp_vac_s", "isp_vac_s = 320.0"),
    ("stage.1.engine.1.exit_area_m2", "exit_area_m2 = 0.08"),
    ("stage.1.engine.1.position_m", "position_m = [0.1, 0.0, 0.0]"),
    ("stage.1.engine.1.axis", "axis = [1.0, 0.0, 0.0]"),
    ("stage.1.engine.1.gimbal_max_deg", "gimbal_max_deg = 5.0"),
    ("stage.1.engine.1.gimbal_rate_dps", "gimbal_rate_dps = 10.0"),
    ("stage.1.engine.1.throttle_min", "throttle_min = 0.6"),
    ("stage.1.engine.1.throttle_max", "throttle_max = 1.0"),
    ("stage.1.engine.1.spool_time_s", "spool_time_s = 0.5"),
    ("stage.1.engine.1.ignitions", "ignitions = 1"),
]


def _vehicle_text(exclude=(), extra_lines=(), replace=None, prelude=()):
    replace = replace or {}
    lines = list(prelude)
    for tag, line in _VALID_LINES:
        if any(tag == ex or tag.startswith(ex + ".") for ex in exclude):
            continue
        lines.append(replace.get(tag, line))
    lines.extend(extra_lines)
    return "\n".join(lines) + "\n"


def _validate_text(tmp_path, text, strict=False):
    path = tmp_path / "vehicle.toml"
    path.write_text(text, encoding="utf-8")
    return validate_vehicle_file(path, strict=strict)


def test_minimal_vehicle_validates_cleanly(tmp_path):
    resolved, errors, warns = _validate_text(tmp_path, _vehicle_text())
    assert errors == []
    assert warns == []
    assert resolved["vehicle"]["name"] == "unit-test-vehicle"
    assert resolved["provenance"] == "representative"
    assert resolved["stage"][0]["engine"][0]["ignitions"] == 1
    # Numeric parameters canonicalize to float regardless of TOML literal type.
    assert isinstance(resolved["stage"][0]["dry_mass_kg"], float)


def test_integer_literals_resolve_to_floats(tmp_path):
    # A curated file may write thrust_vac_N = 60000; the resolved config and
    # therefore the hash must not depend on the literal's TOML type.
    a, _, _ = _validate_text(tmp_path, _vehicle_text())
    b, _, _ = _validate_text(
        tmp_path, _vehicle_text(replace={"stage.1.engine.1.thrust_vac_N": "thrust_vac_N = 60000"})
    )
    assert config_sha256(a) == config_sha256(b)


@pytest.mark.parametrize(
    ("table", "key"),
    [("root", key) for key in REQUIRED_KEYS["root"]]
    + [("vehicle", key) for key in REQUIRED_KEYS["vehicle"]]
    + [("stage.1", key) for key in REQUIRED_KEYS["stage"]]
    + [("stage.1.tank.1", key) for key in REQUIRED_KEYS["stage.tank"]]
    + [("stage.1.engine.1", key) for key in REQUIRED_KEYS["stage.engine"]],
)
def test_deleting_each_required_key_names_that_key(tmp_path, table, key):
    tag = key if table == "root" else f"{table}.{key}"
    resolved, errors, _ = _validate_text(tmp_path, _vehicle_text(exclude=(tag,)))
    assert resolved is None
    assert any(f"[{table}] {key}:" in e and "missing required" in e for e in errors), errors


def test_deleting_feeds_tank_is_not_masked_by_the_dangling_check(tmp_path):
    resolved, errors, _ = _validate_text(
        tmp_path, _vehicle_text(exclude=("stage.1.engine.1.feeds_tank",))
    )
    assert resolved is None
    assert any("[stage.1.engine.1] feeds_tank: missing required key" in e for e in errors), errors


def test_deleting_the_stage_block_names_it(tmp_path):
    resolved, errors, _ = _validate_text(tmp_path, _vehicle_text(exclude=("stage.1",)))
    assert resolved is None
    assert any("[root] stage: missing required table array" in e for e in errors), errors


def test_deleting_the_vehicle_table_names_it(tmp_path):
    resolved, errors, _ = _validate_text(tmp_path, _vehicle_text(exclude=("vehicle",)))
    assert resolved is None
    assert any("[root] vehicle: missing required table" in e for e in errors), errors


def test_unknown_key_rejected_at_every_level(tmp_path):
    text = _vehicle_text(
        prelude=("mystery_root = 1",),
        replace={
            "vehicle.name": 'name = "x"\npaint_scheme = "black"',
            "stage.1.tank.1.radius_m": "radius_m = 0.5\nullage_frac = 0.05",
        },
    )
    resolved, errors, _ = _validate_text(tmp_path, text)
    assert resolved is None
    joined = "\n".join(errors)
    assert "[root] mystery_root: unknown key" in joined
    assert "[vehicle] paint_scheme: unknown key" in joined
    assert "[stage.1.tank.1] ullage_frac: unknown key" in joined


def test_all_errors_accumulate_in_one_report(tmp_path):
    # One broken file with a missing key, a range error, and an unknown key:
    # every defect surfaces in a single pass (DX-2, never fail-first).
    text = _vehicle_text(
        exclude=("stage.1.engine.1.isp_vac_s",),
        replace={"stage.1.tank.1.radius_m": "radius_m = -0.5\nbogus = 1"},
    )
    resolved, errors, _ = _validate_text(tmp_path, text)
    assert resolved is None
    joined = "\n".join(errors)
    assert "[stage.1.engine.1] isp_vac_s: missing required key" in joined
    assert "[stage.1.tank.1] radius_m:" in joined
    assert "[stage.1.tank.1] bogus: unknown key" in joined
    assert len(errors) >= 3


def test_error_lines_follow_dx2_format(tmp_path):
    path = tmp_path / "vehicle.toml"
    path.write_text(_vehicle_text(exclude=("stage.1.engine.1.isp_vac_s",)), encoding="utf-8")
    resolved, errors, _ = validate_vehicle_file(path)
    assert resolved is None
    for line in errors:
        assert line.startswith(f"{path}: ["), line
        assert line.endswith("No default applied; run aborted."), line
    # The DX-2 exemplar shape: table path, key, units, typical range.
    assert any(
        "[stage.1.engine.1] isp_vac_s: missing required key "
        "(units: s; typical range typical chemical range 200-465)" in e
        for e in errors
    ), errors


@pytest.mark.parametrize(
    ("replace", "needle"),
    [
        ({"schema_version": "schema_version = 2"}, "[root] schema_version:"),
        ({"provenance": 'provenance = ""'}, "[root] provenance:"),
        ({"stage.1.dry_mass_kg": "dry_mass_kg = 0.0"}, "[stage.1] dry_mass_kg:"),
        ({"stage.1.dry_cg_m": "dry_cg_m = [2.0, 0.0]"}, "[stage.1] dry_cg_m:"),
        ({"stage.1.tank.1.length_m": "length_m = -3.0"}, "[stage.1.tank.1] length_m:"),
        ({"stage.1.engine.1.thrust_vac_N": "thrust_vac_N = 0.0"}, "[stage.1.engine.1] thrust_vac_N:"),
        ({"stage.1.engine.1.gimbal_max_deg": "gimbal_max_deg = 60.0"}, "[stage.1.engine.1] gimbal_max_deg:"),
        ({"stage.1.engine.1.gimbal_rate_dps": "gimbal_rate_dps = -1.0"}, "[stage.1.engine.1] gimbal_rate_dps:"),
        ({"stage.1.engine.1.throttle_min": "throttle_min = 0.0"}, "[stage.1.engine.1] throttle_min:"),
        ({"stage.1.engine.1.throttle_max": "throttle_max = 1.5"}, "[stage.1.engine.1] throttle_max:"),
        ({"stage.1.engine.1.spool_time_s": "spool_time_s = -0.5"}, "[stage.1.engine.1] spool_time_s:"),
    ],
)
def test_field_range_errors(tmp_path, replace, needle):
    resolved, errors, _ = _validate_text(tmp_path, _vehicle_text(replace=replace))
    assert resolved is None
    assert any(needle in e for e in errors), errors


@pytest.mark.parametrize("literal", ["0", "-1", "1.5", "true"])
def test_ignitions_must_be_positive_integer(tmp_path, literal):
    resolved, errors, _ = _validate_text(
        tmp_path, _vehicle_text(replace={"stage.1.engine.1.ignitions": f"ignitions = {literal}"})
    )
    assert resolved is None
    assert any("[stage.1.engine.1] ignitions:" in e for e in errors), errors


def test_axis_written_at_repr_precision_is_accepted(tmp_path):
    c = 1.0 / math.sqrt(2.0)
    resolved, errors, _ = _validate_text(
        tmp_path,
        _vehicle_text(replace={"stage.1.engine.1.axis": f"axis = [{c!r}, {c!r}, 0.0]"}),
    )
    assert errors == []
    assert resolved is not None


# ---------------------------------------------------------------------------
# Committed malformed-fixture corpus: one seeded defect per file, one file
# per validation pass surface.
# ---------------------------------------------------------------------------

_CORPUS = [
    ("bad_range.toml", "[stage.1] dry_mass_kg:"),
    ("non_spd_inertia.toml", "positive definite"),
    ("triangle_inequality.toml", "triangle inequality"),
    ("asymmetric_inertia.toml", "must be symmetric"),
    ("dangling_feed.toml", "[stage.1.engine.1] feeds_tank: unknown tank"),
    ("non_unit_axis.toml", "[stage.1.engine.1] axis: must be unit-norm"),
    ("unknown_key.toml", "[stage.1.engine.1] thrust_sl_N: unknown key"),
    ("missing_provenance.toml", "[root] provenance: missing required key"),
    ("tank_overfill.toml", "[stage.1.tank.1] propellant_mass_kg: exceeds the tank capacity"),
    ("throttle_inverted.toml", "[stage.1.engine.1] throttle_min: must be <= throttle_max"),
    ("duplicate_stage_names.toml", "[stage.2] name: duplicate stage name"),
    ("bad_aero_header.toml", "[aero.1] mach_table_csv:"),
]


@pytest.mark.parametrize(("fixture", "needle"), _CORPUS, ids=[c[0] for c in _CORPUS])
def test_malformed_fixture_corpus(fixture, needle, monkeypatch):
    # Fixture aero paths are repo-root relative (the same working-directory
    # rule as mission [environment] paths).
    monkeypatch.chdir(REPO_ROOT)
    resolved, errors, _ = validate_vehicle_file(FIXTURES / fixture)
    assert resolved is None
    assert any(needle in e for e in errors), errors


def test_warning_fixture_is_valid_but_warns(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    resolved, errors, warns = validate_vehicle_file(FIXTURES / "warning_low_liftoff_tw.toml")
    assert errors == []
    assert resolved is not None
    assert any("thrust-to-weight" in w for w in warns), warns
    assert all(w.endswith("--strict promotes warnings to errors.") for w in warns), warns


def test_strict_promotes_warnings_to_errors(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    resolved, errors, warns = validate_vehicle_file(
        FIXTURES / "warning_low_liftoff_tw.toml", strict=True
    )
    assert resolved is None
    assert any("thrust-to-weight" in e and "Promoted to an error by --strict" in e for e in errors)
    # The advisory list is still reported alongside the promotion.
    assert any("thrust-to-weight" in w for w in warns)


def test_isp_outside_chemical_range_warns(tmp_path):
    resolved, errors, warns = _validate_text(
        tmp_path, _vehicle_text(replace={"stage.1.engine.1.isp_vac_s": "isp_vac_s = 900.0"})
    )
    assert errors == []
    assert any("outside the typical chemical range" in w for w in warns), warns
    resolved, errors, _ = _validate_text(
        tmp_path,
        _vehicle_text(replace={"stage.1.engine.1.isp_vac_s": "isp_vac_s = 900.0"}),
        strict=True,
    )
    assert resolved is None


def test_propellant_fraction_above_095_warns(tmp_path):
    # 2000 kg propellant on a 100 kg structure: fraction 0.952. The tank
    # still fits (capacity 2427 kg), so only the warning tier fires.
    resolved, errors, warns = _validate_text(
        tmp_path, _vehicle_text(replace={"stage.1.dry_mass_kg": "dry_mass_kg = 100.0"})
    )
    assert errors == []
    assert any("propellant mass fraction" in w for w in warns), warns


def test_toml_parse_error_reported(tmp_path):
    path = tmp_path / "vehicle.toml"
    path.write_text("this is [not valid toml\n", encoding="utf-8")
    resolved, errors, warns = validate_vehicle_file(path)
    assert resolved is None
    assert len(errors) == 1
    assert "TOML parse error" in errors[0]
    assert warns == []


def test_missing_file_reported(tmp_path):
    resolved, errors, _ = validate_vehicle_file(tmp_path / "absent.toml")
    assert resolved is None
    assert len(errors) == 1
    assert "cannot read vehicle file" in errors[0]


# ---------------------------------------------------------------------------
# Optional blocks: RCS, wheels, sensors, jettison, aero.
# ---------------------------------------------------------------------------

_RCS_BLOCK = (
    "[[stage.rcs]]",
    'name = "rcs_a"',
    "thrust_N = 20.0",
    "min_impulse_bit_Ns = 0.04",
    "thruster_positions_m = [[2.5, 0.5, 0.0], [2.5, -0.5, 0.0]]",
    "thruster_directions = [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]",
)

_WHEEL_BLOCK = (
    "[[stage.wheel]]",
    'name = "rw_x"',
    "axis = [1.0, 0.0, 0.0]",
    "max_torque_Nm = 0.02",
    "max_momentum_Nms = 0.4",
)

_SENSOR_BLOCK = (
    "[[stage.sensor]]",
    'name = "imu0"',
    'preset = "presets/imu_tactical.toml"',
    "position_m = [1.0, 0.0, 0.0]",
    "axis = [1.0, 0.0, 0.0]",
)

_JETTISON_BLOCK = (
    "[[stage.jettison]]",
    'name = "fairing"',
    "mass_kg = 40.0",
    "cg_m = [4.0, 0.0, 0.0]",
    "inertia_kgm2 = [[15.0, 0.0, 0.0], [0.0, 30.0, 0.0], [0.0, 0.0, 30.0]]",
)


def test_optional_blocks_validate_and_resolve(tmp_path):
    text = _vehicle_text(
        extra_lines=_RCS_BLOCK + _WHEEL_BLOCK + _SENSOR_BLOCK + _JETTISON_BLOCK
    )
    resolved, errors, warns = _validate_text(tmp_path, text)
    assert errors == []
    stage = resolved["stage"][0]
    assert stage["rcs"][0]["thrust_N"] == 20.0
    assert stage["wheel"][0]["max_momentum_Nms"] == 0.4
    assert stage["sensor"][0]["preset"] == "presets/imu_tactical.toml"
    assert stage["jettison"][0]["mass_kg"] == 40.0
    # Absent optional block arrays are omitted from the resolved config, so
    # pre-existing vehicles keep byte-identical resolutions as blocks land.
    assert "rcs" not in resolved["stage"][0] or stage["rcs"]


@pytest.mark.parametrize(
    ("swap", "needle"),
    [
        (
            (
                "thruster_directions = [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]",
                "thruster_directions = [[0.0, 1.0, 0.0]]",
            ),
            "one entry per thruster position",
        ),
        (
            (
                "thruster_directions = [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]",
                "thruster_directions = [[0.0, 1.0, 0.0], [0.0, -2.0, 0.0]]",
            ),
            "unit-norm",
        ),
        (("min_impulse_bit_Ns = 0.04", "min_impulse_bit_Ns = 0.0"), "min_impulse_bit_Ns"),
    ],
)
def test_rcs_cross_field_errors(tmp_path, swap, needle):
    lines = tuple(ln.replace(swap[0], swap[1]) for ln in _RCS_BLOCK)
    resolved, errors, _ = _validate_text(tmp_path, _vehicle_text(extra_lines=lines))
    assert resolved is None
    assert any(needle in e for e in errors), errors


def test_jettison_inertia_rules_apply(tmp_path):
    lines = tuple(
        ln.replace(
            "inertia_kgm2 = [[15.0, 0.0, 0.0], [0.0, 30.0, 0.0], [0.0, 0.0, 30.0]]",
            "inertia_kgm2 = [[15.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 30.0]]",
        )
        for ln in _JETTISON_BLOCK
    )
    resolved, errors, _ = _validate_text(tmp_path, _vehicle_text(extra_lines=lines))
    assert resolved is None
    assert any("[stage.1.jettison.1] inertia_kgm2:" in e and "triangle" in e for e in errors)


def test_duplicate_names_within_stage_rejected(tmp_path):
    text = _vehicle_text(extra_lines=_WHEEL_BLOCK + _WHEEL_BLOCK)
    resolved, errors, _ = _validate_text(tmp_path, text)
    assert resolved is None
    assert any("[stage.1.wheel.2] name: duplicate" in e for e in errors), errors


def test_aero_block_with_good_table_resolves(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    text = _vehicle_text(
        replace={"stage.1.engine.1.thrust_vac_N": "thrust_vac_N = 60000.0"},
        extra_lines=(
            "[[aero]]",
            'config = "full_stack"',
            "ref_area_m2 = 1.13",
            "ref_diameter_m = 1.2",
            'mach_table_csv = "tests/fixtures/vehicles/slender_aero.csv"',
            "cmq_per_rad = -0.4",
        ),
    )
    resolved, errors, warns = _validate_text(tmp_path, text)
    assert errors == []
    assert resolved["aero"][0]["cmq_per_rad"] == -0.4
    # 60 kN over ~2.5 t wet: comfortably above the T/W warning threshold.
    assert warns == []


def test_aero_positive_cmq_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    text = _vehicle_text(
        extra_lines=(
            "[[aero]]",
            'config = "full_stack"',
            "ref_area_m2 = 1.13",
            "ref_diameter_m = 1.2",
            'mach_table_csv = "tests/fixtures/vehicles/slender_aero.csv"',
            "cmq_per_rad = 0.4",
        ),
    )
    resolved, errors, _ = _validate_text(tmp_path, text)
    assert resolved is None
    assert any("[aero.1] cmq_per_rad:" in e for e in errors), errors


def test_missing_aero_csv_aborts(tmp_path):
    text = _vehicle_text(
        extra_lines=(
            "[[aero]]",
            'config = "full_stack"',
            "ref_area_m2 = 1.13",
            "ref_diameter_m = 1.2",
            'mach_table_csv = "no/such/table.csv"',
        ),
    )
    resolved, errors, _ = _validate_text(tmp_path, text)
    assert resolved is None
    assert any(
        "[aero.1] mach_table_csv: cannot read aero table" in e for e in errors
    ), errors


# ---------------------------------------------------------------------------
# Resolved-config echo and hash (FR-15; Phase 4 exit criterion 1).
# ---------------------------------------------------------------------------


def _echo_roundtrip(tmp_path, resolved):
    echo1 = canonical_vehicle_toml(resolved)
    echo_path = tmp_path / "echo.toml"
    echo_path.write_text(echo1, encoding="utf-8", newline="")
    resolved2, errors2, _ = validate_vehicle_file(echo_path)
    assert errors2 == [], errors2
    return echo1, canonical_vehicle_toml(resolved2), resolved2


def test_echo_revalidates_byte_identically(tmp_path):
    text = _vehicle_text(extra_lines=_RCS_BLOCK + _WHEEL_BLOCK + _SENSOR_BLOCK + _JETTISON_BLOCK)
    resolved, errors, _ = _validate_text(tmp_path, text)
    assert errors == []
    echo1, echo2, resolved2 = _echo_roundtrip(tmp_path, resolved)
    assert echo1 == echo2
    assert config_sha256(resolved) == config_sha256(resolved2)


def test_echo_is_independent_of_authoring_order(tmp_path):
    # The same document with scalar keys shuffled within each table must
    # resolve, echo, and hash identically (canonical key order).
    base, errors, _ = _validate_text(tmp_path, _vehicle_text())
    assert errors == []
    # Move ignitions from the last engine line to the first: same document,
    # different authoring order.
    shuffled = _vehicle_text(
        exclude=("stage.1.engine.1.ignitions",),
        replace={"stage.1.engine.1.name": 'ignitions = 1\nname = "main_engine"'},
    )
    resolved, errors, _ = _validate_text(tmp_path, shuffled)
    assert errors == []
    assert canonical_vehicle_toml(base) == canonical_vehicle_toml(resolved)
    assert config_sha256(base) == config_sha256(resolved)


def test_changed_value_changes_hash(tmp_path):
    a, _, _ = _validate_text(tmp_path, _vehicle_text())
    b, _, _ = _validate_text(
        tmp_path, _vehicle_text(replace={"stage.1.engine.1.isp_vac_s": "isp_vac_s = 321.0"})
    )
    assert config_sha256(a) != config_sha256(b)
