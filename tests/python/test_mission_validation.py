"""Mission TOML validator tests (contract section 6, DX-2, FR-14/FR-15 lite).

Covers the mutation direction demanded by the Phase 1 contract: deleting each
required key from a valid mission must produce an error naming exactly that
key and its table path, and a single unknown key anywhere must be rejected.
No compiled core is required by any test in this file.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from star_reacher.mission import (
    keplerian_to_cartesian,
    validate_mission_file,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# Mirrors _core.gm("earth") (IERS Conventions 2010, TN No. 36) so the
# conversion tests run without the compiled core; the end-to-end agreement
# with the core's value is asserted in test_integration_core.py.
GM_EARTH = 3.986004418e14

# One tagged line per schema entry so mutation tests can delete exactly one
# key (or one whole table with its children) and re-serialize without a TOML
# writer dependency.
_VALID_LINES = [
    ("schema_version", "schema_version = 1"),
    ("mission", "[mission]"),
    ("mission.name", 'name = "unit-test"'),
    ("mission.epoch_utc", 'epoch_utc = "2026-01-01T00:00:00Z"'),
    ("mission.duration_s", "duration_s = 600.0"),
    ("run", "[run]"),
    ("run.seed", "seed = 7"),
    ("integrator", "[integrator]"),
    ("integrator.type", 'type = "rk4"'),
    ("integrator.dt_s", "dt_s = 0.1"),
    ("initial_state.cartesian", "[initial_state.cartesian]"),
    ("initial_state.cartesian.r_m", "r_m = [6778137.0, 0.0, 0.0]"),
    ("initial_state.cartesian.v_mps", "v_mps = [0.0, 7668.6, 0.0]"),
    ("initial_state.cartesian.frame", 'frame = "GCRF"'),
    ("environment", "[environment]"),
    ("environment.central_body", 'central_body = "earth"'),
]


def _mission_text(exclude=(), extra_lines=(), replace=None, prelude=()):
    # prelude lines land before any table header: in TOML a bare key written
    # after a [table] header belongs to that table, so root-level additions
    # must come first.
    replace = replace or {}
    lines = list(prelude)
    for tag, line in _VALID_LINES:
        # Deleting a table tag removes the header and every child line so the
        # resulting document is still well-formed TOML.
        if any(tag == ex or tag.startswith(ex + ".") for ex in exclude):
            continue
        lines.append(replace.get(tag, line))
    lines.extend(extra_lines)
    return "\n".join(lines) + "\n"


def _validate_text(tmp_path, text):
    path = tmp_path / "mission.toml"
    path.write_text(text, encoding="utf-8")
    return validate_mission_file(path)


def test_reference_mission_validates_cleanly():
    resolved, errors = validate_mission_file(REPO_ROOT / "missions" / "twobody_leo.toml")
    assert errors == []
    assert resolved["mission"]["name"] == "twobody-leo"
    assert resolved["spacecraft"]["mass_kg"] == 150.0
    assert resolved["logging"]["truth_rate_hz"] == 10
    assert resolved["initial_state"]["cartesian"]["frame"] == "GCRF"


def test_valid_minimal_mission_records_defaults(tmp_path):
    resolved, errors = _validate_text(tmp_path, _mission_text())
    assert errors == []
    # Defaults are applied here and recorded, never silent (D-2/FR-15).
    assert resolved["spacecraft"]["mass_kg"] == 1.0
    assert resolved["logging"]["truth_rate_hz"] == 10
    assert resolved["run"]["seed"] == 7
    assert resolved["integrator"]["dt_s"] == 0.1


@pytest.mark.parametrize(
    ("table", "key"),
    [
        ("root", "schema_version"),
        ("mission", "name"),
        ("mission", "epoch_utc"),
        ("mission", "duration_s"),
        ("run", "seed"),
        ("integrator", "type"),
        ("integrator", "dt_s"),
        ("initial_state.cartesian", "r_m"),
        ("initial_state.cartesian", "v_mps"),
        ("initial_state.cartesian", "frame"),
        ("environment", "central_body"),
    ],
)
def test_deleting_each_required_key_names_that_key(tmp_path, table, key):
    tag = key if table == "root" else f"{table}.{key}"
    resolved, errors = _validate_text(tmp_path, _mission_text(exclude=(tag,)))
    assert resolved is None
    assert any(f"[{table}] {key}:" in e and "missing required" in e for e in errors), errors


@pytest.mark.parametrize("table", ["mission", "run", "integrator", "environment"])
def test_deleting_each_required_table_names_that_table(tmp_path, table):
    resolved, errors = _validate_text(tmp_path, _mission_text(exclude=(table,)))
    assert resolved is None
    assert any(f"[root] {table}: missing required table" in e for e in errors), errors


def test_deleting_initial_state_entirely_names_the_table(tmp_path):
    # Removing [initial_state.cartesian] removes the only mention of the
    # parent table, so the whole initial_state table is gone.
    resolved, errors = _validate_text(tmp_path, _mission_text(exclude=("initial_state.cartesian",)))
    assert resolved is None
    assert any("[root] initial_state: missing required table" in e for e in errors), errors


def test_empty_initial_state_table_requires_exactly_one_form(tmp_path):
    text = _mission_text(exclude=("initial_state.cartesian",), extra_lines=("[initial_state]",))
    resolved, errors = _validate_text(tmp_path, text)
    assert resolved is None
    assert any("exactly one initial-state form is required, found none" in e for e in errors), errors


def test_unknown_top_level_bare_key_rejected(tmp_path):
    resolved, errors = _validate_text(tmp_path, _mission_text(prelude=("unknown_top = 1",)))
    assert resolved is None
    assert any("[root] unknown_top: unknown key" in e for e in errors), errors


def test_unknown_top_level_table_rejected(tmp_path):
    resolved, errors = _validate_text(tmp_path, _mission_text(extra_lines=("[mission_extras]",)))
    assert resolved is None
    assert any("[root] mission_extras: unknown key" in e for e in errors), errors


def test_unknown_nested_key_names_exact_table_path(tmp_path):
    # A typo inside a nested table must be named with its full path (DX-2).
    text = _mission_text(replace={"initial_state.cartesian.frame": 'frame = "GCRF"\nfrmae = "GCRF"'})
    resolved, errors = _validate_text(tmp_path, text)
    assert resolved is None
    assert any("[initial_state.cartesian] frmae: unknown key" in e for e in errors), errors


def test_unknown_key_in_mission_table(tmp_path):
    text = _mission_text(replace={"mission.name": 'name = "x"\nattitude = "nadir"'})
    resolved, errors = _validate_text(tmp_path, text)
    assert resolved is None
    assert any("[mission] attitude: unknown key" in e for e in errors), errors


def test_two_initial_state_forms_rejected(tmp_path):
    extra = (
        "[initial_state.keplerian]",
        "sma_m = 6778137.0",
        "ecc = 0.0",
        "inc_deg = 0.0",
        "raan_deg = 0.0",
        "argp_deg = 0.0",
        "ta_deg = 0.0",
    )
    resolved, errors = _validate_text(tmp_path, _mission_text(extra_lines=extra))
    assert resolved is None
    assert any("exactly one initial-state form is required, found 2" in e for e in errors), errors


def test_geodetic_recognized_but_rejected_until_phase_4(tmp_path):
    extra = ("[initial_state.geodetic]", "lat_deg = 28.5", "lon_deg = -80.6", "alt_m = 10.0")
    text = _mission_text(exclude=("initial_state.cartesian",), extra_lines=extra)
    resolved, errors = _validate_text(tmp_path, text)
    assert resolved is None
    assert any("supported from Phase 4" in e for e in errors), errors


@pytest.mark.parametrize(
    ("replace", "needle"),
    [
        ({"schema_version": "schema_version = 2"}, "[root] schema_version:"),
        ({"mission.duration_s": "duration_s = -1.0"}, "[mission] duration_s:"),
        ({"mission.duration_s": "duration_s = 0.0"}, "[mission] duration_s:"),
        ({"mission.epoch_utc": 'epoch_utc = "2026-01-01"'}, "[mission] epoch_utc:"),
        ({"mission.epoch_utc": 'epoch_utc = "not-a-date"'}, "[mission] epoch_utc:"),
        ({"mission.name": 'name = ""'}, "[mission] name:"),
        ({"run.seed": "seed = -1"}, "[run] seed:"),
        ({"run.seed": "seed = 1.5"}, "[run] seed:"),
        ({"integrator.type": 'type = "rk45"'}, "[integrator] type:"),
        ({"integrator.dt_s": "dt_s = 0.0"}, "[integrator] dt_s:"),
        ({"integrator.dt_s": "dt_s = -0.1"}, "[integrator] dt_s:"),
        ({"initial_state.cartesian.frame": 'frame = "J2000"'}, "[initial_state.cartesian] frame:"),
        ({"initial_state.cartesian.r_m": "r_m = [1.0, 2.0]"}, "[initial_state.cartesian] r_m:"),
        ({"initial_state.cartesian.r_m": "r_m = [0.0, 0.0, 0.0]"}, "[initial_state.cartesian] r_m:"),
        ({"initial_state.cartesian.v_mps": "v_mps = [0.0, nan, 0.0]"}, "[initial_state.cartesian] v_mps:"),
        ({"environment.central_body": 'central_body = "moon"'}, "[environment] central_body:"),
    ],
)
def test_field_range_and_type_errors(tmp_path, replace, needle):
    resolved, errors = _validate_text(tmp_path, _mission_text(replace=replace))
    assert resolved is None
    assert any(needle in e for e in errors), errors


def test_seed_u64_bounds(tmp_path):
    # 0 and 2^63-1 (the TOML signed-integer ceiling) are valid u64 seeds.
    for literal in ("0", str(2**63 - 1)):
        resolved, errors = _validate_text(
            tmp_path, _mission_text(replace={"run.seed": f"seed = {literal}"})
        )
        assert errors == [], errors
    # tomllib parses arbitrary-precision integers, so the u64 ceiling must be
    # enforced by the validator, not assumed from the TOML grammar.
    resolved, errors = _validate_text(
        tmp_path, _mission_text(replace={"run.seed": f"seed = {2**64}"})
    )
    assert resolved is None
    assert any("[run] seed:" in e for e in errors), errors


@pytest.mark.parametrize(
    ("extra", "needle"),
    [
        (("[spacecraft]", "mass_kg = -5.0"), "[spacecraft] mass_kg:"),
        (("[spacecraft]", "mass_kg = 0.0"), "[spacecraft] mass_kg:"),
        (("[logging]", "truth_rate_hz = 0"), "[logging] truth_rate_hz:"),
        (("[logging]", "truth_rate_hz = 2.5"), "[logging] truth_rate_hz:"),
    ],
)
def test_optional_table_field_errors(tmp_path, extra, needle):
    resolved, errors = _validate_text(tmp_path, _mission_text(extra_lines=extra))
    assert resolved is None
    assert any(needle in e for e in errors), errors


def test_duration_not_multiple_of_dt_rejected(tmp_path):
    text = _mission_text(replace={"mission.duration_s": "duration_s = 600.05"})
    resolved, errors = _validate_text(tmp_path, text)
    assert resolved is None
    assert any("integer multiple" in e and "[mission] duration_s:" in e for e in errors), errors


def test_decimation_must_be_exact_positive_integer(tmp_path):
    # 1/(0.3 * 10) is not an integer: the truth rate cannot divide the step
    # rate unevenly (decimation only, never interpolation).
    text = _mission_text(
        replace={"integrator.dt_s": "dt_s = 0.3", "mission.duration_s": "duration_s = 600.0"},
        extra_lines=("[logging]", "truth_rate_hz = 10"),
    )
    resolved, errors = _validate_text(tmp_path, text)
    assert resolved is None
    assert any("exact positive integer" in e for e in errors), errors


def test_truth_rate_faster_than_step_rate_rejected(tmp_path):
    # rate 100 Hz at dt 0.1 s would need interpolation between steps.
    text = _mission_text(extra_lines=("[logging]", "truth_rate_hz = 100"))
    resolved, errors = _validate_text(tmp_path, text)
    assert resolved is None
    assert any("[logging] truth_rate_hz:" in e for e in errors), errors


def test_all_errors_accumulate_in_one_report(tmp_path):
    # One broken file with an unknown key, a missing required key, and two
    # initial-state forms: every error must surface in a single validation
    # pass (DX-2: accumulated, never fail-first).
    text = _mission_text(
        exclude=("integrator.dt_s",),
        prelude=("bogus_key = 1",),
        extra_lines=(
            "[initial_state.keplerian]",
            "sma_m = 6778137.0",
            "ecc = 0.0",
            "inc_deg = 0.0",
            "raan_deg = 0.0",
            "argp_deg = 0.0",
            "ta_deg = 0.0",
        ),
    )
    resolved, errors = _validate_text(tmp_path, text)
    assert resolved is None
    joined = "\n".join(errors)
    assert "bogus_key" in joined
    assert "[integrator] dt_s: missing required" in joined
    assert "exactly one initial-state form" in joined
    assert len(errors) >= 3


def test_error_lines_follow_dx2_format(tmp_path):
    path = tmp_path / "mission.toml"
    path.write_text(_mission_text(exclude=("integrator.dt_s", "run.seed")), encoding="utf-8")
    resolved, errors = validate_mission_file(path)
    assert resolved is None
    for line in errors:
        assert line.startswith(f"{path}: ["), line
        assert line.endswith("No default applied; run aborted."), line
    # Numeric fields carry units and a typical range in the message body.
    assert any("(units: s;" in e for e in errors), errors


def test_epoch_with_numeric_offset_accepted(tmp_path):
    text = _mission_text(replace={"mission.epoch_utc": 'epoch_utc = "2026-01-01T00:00:00+00:00"'})
    resolved, errors = _validate_text(tmp_path, text)
    assert errors == []
    # Carried verbatim: canonicalization must not rewrite the epoch string.
    assert resolved["mission"]["epoch_utc"] == "2026-01-01T00:00:00+00:00"


def test_toml_parse_error_reported_with_exit_semantics(tmp_path):
    path = tmp_path / "mission.toml"
    path.write_text("this is [not valid toml\n", encoding="utf-8")
    resolved, errors = validate_mission_file(path)
    assert resolved is None
    assert len(errors) == 1
    assert "TOML parse error" in errors[0]


def test_keplerian_mission_validates(tmp_path):
    text = _mission_text(
        exclude=("initial_state.cartesian",),
        extra_lines=(
            "[initial_state.keplerian]",
            "sma_m = 7000000.0",
            "ecc = 0.001",
            "inc_deg = 51.6",
            "raan_deg = 30.0",
            "argp_deg = 60.0",
            "ta_deg = 100.0",
        ),
    )
    resolved, errors = _validate_text(tmp_path, text)
    assert errors == []
    assert resolved["initial_state"]["keplerian"]["sma_m"] == 7000000.0


@pytest.mark.parametrize(
    ("replace_line", "needle"),
    [
        ("ecc = 1.0", "[initial_state.keplerian] ecc:"),
        ("ecc = -0.1", "[initial_state.keplerian] ecc:"),
        ("sma_m = -7000000.0", "[initial_state.keplerian] sma_m:"),
        ("inc_deg = 190.0", "[initial_state.keplerian] inc_deg:"),
    ],
)
def test_keplerian_range_errors(tmp_path, replace_line, needle):
    base = {
        "sma_m": "sma_m = 7000000.0",
        "ecc": "ecc = 0.001",
        "inc_deg": "inc_deg = 51.6",
        "raan_deg": "raan_deg = 30.0",
        "argp_deg": "argp_deg = 60.0",
        "ta_deg": "ta_deg = 100.0",
    }
    key = replace_line.split(" ")[0]
    base[key] = replace_line
    text = _mission_text(
        exclude=("initial_state.cartesian",),
        extra_lines=("[initial_state.keplerian]", *base.values()),
    )
    resolved, errors = _validate_text(tmp_path, text)
    assert resolved is None
    assert any(needle in e for e in errors), errors


def test_keplerian_to_cartesian_circular_equatorial():
    a = 7000e3
    r, v = keplerian_to_cartesian(
        {"sma_m": a, "ecc": 0.0, "inc_deg": 0.0, "raan_deg": 0.0, "argp_deg": 0.0, "ta_deg": 0.0},
        GM_EARTH,
    )
    assert np.allclose(r, [a, 0.0, 0.0], rtol=1e-12, atol=1e-6)
    assert np.allclose(v, [0.0, math.sqrt(GM_EARTH / a), 0.0], rtol=1e-12, atol=1e-9)


def test_keplerian_to_cartesian_invariants_general_case():
    # Conic and rotation invariants (radius equation, vis-viva, angular
    # momentum magnitude and inclination) pin the conversion without an
    # external golden vector; each follows from the cited Vallado algorithm.
    elems = {
        "sma_m": 8000e3,
        "ecc": 0.1,
        "inc_deg": 45.0,
        "raan_deg": 30.0,
        "argp_deg": 60.0,
        "ta_deg": 100.0,
    }
    r, v = keplerian_to_cartesian(elems, GM_EARTH)
    a, e = elems["sma_m"], elems["ecc"]
    p = a * (1 - e * e)
    nu = math.radians(elems["ta_deg"])
    r_expected = p / (1 + e * math.cos(nu))
    assert math.isclose(np.linalg.norm(r), r_expected, rel_tol=1e-12)
    v_sq_expected = GM_EARTH * (2.0 / np.linalg.norm(r) - 1.0 / a)
    assert math.isclose(float(v @ v), v_sq_expected, rel_tol=1e-12)
    h = np.cross(r, v)
    assert math.isclose(np.linalg.norm(h), math.sqrt(GM_EARTH * p), rel_tol=1e-12)
    inc = math.degrees(math.acos(h[2] / np.linalg.norm(h)))
    assert math.isclose(inc, elems["inc_deg"], rel_tol=1e-12)
