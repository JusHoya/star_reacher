"""Mission-schema Phase 4 extension tests (FR-14): vehicle reference, event
[[sequence]], geodetic launch state, and the vehicle hash chain (FR-15).

Includes the subprocess-level proof of Phase 4 exit criterion 1's "nonzero
exit" clause: deleting a required key from a starter vehicle makes `star run`
of a mission referencing it exit 2, naming that exact key in the DX-2 format.
Subprocess fan-out is deliberately small and serial. The validation tests
need no compiled core; the two CLI propagation tests exit 0 with the compiled
core (run_vehicle) and exit 1 with the core-missing hint without it.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import star_reacher
from star_reacher.mission import config_sha256, validate_mission_file
from star_reacher.vehicle import validate_vehicle_file

REPO_ROOT = Path(__file__).resolve().parents[2]
FLEET_DIR = REPO_ROOT / "vehicles"

_SMALLSAT = (FLEET_DIR / "smallsat.toml").as_posix()

# Same tagged-line convention as test_mission_validation.py; the root-level
# vehicle key must precede any table header (TOML assigns bare keys after a
# [table] header to that table).
_VALID_LINES = [
    ("schema_version", "schema_version = 1"),
    ("vehicle", f'vehicle = "{_SMALLSAT}"'),
    ("mission", "[mission]"),
    ("mission.name", 'name = "sequence-test"'),
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


def _mission_text(exclude=(), extra_lines=(), replace=None):
    replace = replace or {}
    lines = []
    for tag, line in _VALID_LINES:
        if any(tag == ex or tag.startswith(ex + ".") for ex in exclude):
            continue
        lines.append(replace.get(tag, line))
    lines.extend(extra_lines)
    return "\n".join(lines) + "\n"


def _validate_text(tmp_path, text, strict=False, name="mission.toml"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return validate_mission_file(path, strict=strict)


# ---------------------------------------------------------------------------
# Vehicle reference and the FR-15 hash chain.
# ---------------------------------------------------------------------------


def test_vehicle_reference_resolves_and_chains_the_hash(tmp_path):
    resolved, errors = _validate_text(tmp_path, _mission_text())
    assert errors == []
    vres, verrs, _ = validate_vehicle_file(_SMALLSAT)
    assert verrs == []
    # The mission's resolved config embeds the vehicle's config hash, so the
    # mission hash (the run's reproducibility anchor) covers the vehicle.
    assert resolved["vehicle"] == {
        "path": _SMALLSAT,
        "config_sha256": config_sha256(vres),
    }


def test_mission_hash_changes_when_the_vehicle_changes(tmp_path):
    edited_vehicle = tmp_path / "smallsat_edited.toml"
    edited_vehicle.write_text(
        (FLEET_DIR / "smallsat.toml")
        .read_text(encoding="utf-8")
        .replace("dry_mass_kg = 150.0", "dry_mass_kg = 151.0"),
        encoding="utf-8",
    )
    base, errors = _validate_text(tmp_path, _mission_text(), name="a.toml")
    assert errors == []
    edited, errors = _validate_text(
        tmp_path,
        _mission_text(replace={"vehicle": f'vehicle = "{edited_vehicle.as_posix()}"'}),
        name="b.toml",
    )
    assert errors == []
    assert (
        base["vehicle"]["config_sha256"] != edited["vehicle"]["config_sha256"]
    )
    assert config_sha256(base) != config_sha256(edited)


def test_missions_without_phase4_keys_resolve_unchanged(tmp_path):
    # No vehicle, no sequence, no geodetic: the resolved config must carry
    # none of the Phase 4 keys, preserving pre-Phase-4 hashes byte for byte
    # (the committed golden hash in test_config_hash.py pins the value).
    resolved, errors = _validate_text(tmp_path, _mission_text(exclude=("vehicle",)))
    assert errors == []
    assert "vehicle" not in resolved
    assert "sequence" not in resolved


def test_missing_vehicle_file_aborts(tmp_path):
    resolved, errors = _validate_text(
        tmp_path, _mission_text(replace={"vehicle": 'vehicle = "no/such/vehicle.toml"'})
    )
    assert resolved is None
    assert any("cannot read vehicle file" in e for e in errors), errors


def test_vehicle_errors_accumulate_with_mission_errors(tmp_path):
    broken_vehicle = tmp_path / "broken_vehicle.toml"
    broken_vehicle.write_text(
        (FLEET_DIR / "smallsat.toml")
        .read_text(encoding="utf-8")
        .replace("dry_mass_kg = 150.0", ""),
        encoding="utf-8",
    )
    text = _mission_text(
        exclude=("integrator.dt_s",),
        replace={"vehicle": f'vehicle = "{broken_vehicle.as_posix()}"'},
    )
    resolved, errors = _validate_text(tmp_path, text)
    assert resolved is None
    joined = "\n".join(errors)
    # Both files' defects in one report (DX-2), each line naming its source.
    assert "[stage.1] dry_mass_kg: missing required key" in joined
    assert "[integrator] dt_s: missing required key" in joined
    # Vehicle-file lines carry the vehicle path (as written in the mission
    # file) as their source, so the two files' defects are distinguishable.
    assert broken_vehicle.as_posix() in joined


def test_vehicle_warnings_warn_by_default_and_promote_under_strict(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    text = _mission_text(
        replace={"vehicle": 'vehicle = "tests/fixtures/vehicles/warning_low_liftoff_tw.toml"'}
    )
    with pytest.warns(UserWarning, match="thrust-to-weight"):
        resolved, errors = _validate_text(tmp_path, text)
    assert errors == []
    assert resolved is not None

    resolved, errors = _validate_text(tmp_path, text, strict=True)
    assert resolved is None
    assert any(
        "thrust-to-weight" in e and "Promoted to an error by --strict" in e for e in errors
    ), errors


# ---------------------------------------------------------------------------
# Geodetic launch-site form (FR-14).
# ---------------------------------------------------------------------------

_GEODETIC = (
    "[initial_state.geodetic]",
    "lat_deg = -39.0",
    "lon_deg = 177.9",
    "alt_m = 10.0",
)

_ASCENT_SEQUENCE = (
    "[[sequence]]",
    'name = "ignite_s1"',
    'trigger = "elapsed"',
    "t_s = 0.0",
    'action = "ignite_engine"',
    'stage = "stage1"',
    'engine = "s1_cluster"',
    "",
    "[[sequence]]",
    'name = "release"',
    'trigger = "after_event"',
    'event = "ignite_s1"',
    "offset_s = 1.0",
    'action = "pad_release"',
    "",
    "[[sequence]]",
    'name = "pitchover"',
    'trigger = "elapsed"',
    "t_s = 10.0",
    'action = "pitch_program"',
    "azimuth_deg = 90.0",
    "pitch_t_s = [10.0, 25.0, 60.0, 121.0, 200.0, 300.0]",
    "pitch_deg = [90.0, 75.0, 40.0, 30.0, 6.0, 0.0]",
    "",
    "[[sequence]]",
    'name = "meco"',
    'trigger = "elapsed"',
    "t_s = 121.0",
    'action = "cutoff_engine"',
    'stage = "stage1"',
    'engine = "s1_cluster"',
    "",
    "[[sequence]]",
    'name = "stage_sep"',
    'trigger = "after_event"',
    'event = "meco"',
    "offset_s = 2.0",
    'action = "separate_stage"',
    'stage = "stage1"',
    "",
    "[[sequence]]",
    'name = "ignite_s2"',
    'trigger = "after_event"',
    'event = "stage_sep"',
    "offset_s = 2.0",
    'action = "ignite_engine"',
    'stage = "stage2"',
    'engine = "s2_vacuum"',
    "",
    "[[sequence]]",
    'name = "fairing_jettison"',
    'trigger = "condition"',
    'condition = "altitude_above"',
    "altitude_m = 120000.0",
    'action = "jettison"',
    'stage = "stage2"',
    'item = "fairing"',
    "",
    "[[sequence]]",
    'name = "orbit_insertion"',
    'trigger = "condition"',
    'condition = "perigee_above"',
    "perigee_alt_m = 195000.0",
    'action = "terminate"',
)


def _ascent_text(**kwargs):
    """A full scripted pad-to-LEO ascent mission against the starter LV."""
    replace = {
        "vehicle": 'vehicle = "vehicles/electron_class.toml"',
        "mission.duration_s": "duration_s = 600.0",
    }
    replace.update(kwargs.pop("replace", {}))
    return _mission_text(
        exclude=("initial_state.cartesian",) + tuple(kwargs.pop("exclude", ())),
        extra_lines=_GEODETIC + ("",) + _ASCENT_SEQUENCE + tuple(kwargs.pop("extra_lines", ())),
        replace=replace,
    )


def test_scripted_ascent_mission_validates(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    resolved, errors = _validate_text(tmp_path, _ascent_text())
    assert errors == []
    geo = resolved["initial_state"]["geodetic"]
    assert geo == {"lat_deg": -39.0, "lon_deg": 177.9, "alt_m": 10.0}
    # File order is preserved: the sequence is ordered, not sorted.
    assert [e["name"] for e in resolved["sequence"]] == [
        "ignite_s1",
        "release",
        "pitchover",
        "meco",
        "stage_sep",
        "ignite_s2",
        "fairing_jettison",
        "orbit_insertion",
    ]
    assert resolved["sequence"][2]["pitch_t_s"][0] == 10.0
    assert resolved["vehicle"]["path"] == "vehicles/electron_class.toml"


@pytest.mark.parametrize(
    ("swap", "needle"),
    [
        (("lat_deg = -39.0", "lat_deg = 100.0"), "[initial_state.geodetic] lat_deg:"),
        (("lon_deg = 177.9", "lon_deg = 200.0"), "[initial_state.geodetic] lon_deg:"),
        (("alt_m = 10.0", "alt_m = 20000.0"), "[initial_state.geodetic] alt_m:"),
        (("alt_m = 10.0", "alt_m = 10.0\nramp_len_m = 3.0"), "[initial_state.geodetic] ramp_len_m: unknown key"),
    ],
)
def test_geodetic_field_errors(tmp_path, monkeypatch, swap, needle):
    monkeypatch.chdir(REPO_ROOT)
    text = _ascent_text().replace(swap[0], swap[1])
    resolved, errors = _validate_text(tmp_path, text)
    assert resolved is None
    assert any(needle in e for e in errors), errors


def test_geodetic_requires_earth_central_body(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    text = _ascent_text(replace={"environment.central_body": 'central_body = "mars"'})
    resolved, errors = _validate_text(tmp_path, text)
    assert resolved is None
    assert any('requires central_body = "earth"' in e for e in errors), errors


def test_pad_release_requires_geodetic_form(tmp_path):
    text = _mission_text(
        extra_lines=(
            "[[sequence]]",
            'name = "release"',
            'trigger = "elapsed"',
            "t_s = 0.0",
            'action = "pad_release"',
        )
    )
    resolved, errors = _validate_text(tmp_path, text)
    assert resolved is None
    assert any(
        "[sequence.1] action:" in e and "geodetic" in e for e in errors
    ), errors


def test_duplicate_pad_release_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    extra = (
        "",
        "[[sequence]]",
        'name = "release_again"',
        'trigger = "elapsed"',
        "t_s = 5.0",
        'action = "pad_release"',
    )
    resolved, errors = _validate_text(tmp_path, _ascent_text(extra_lines=extra))
    assert resolved is None
    assert any("duplicate pad_release" in e for e in errors), errors


# ---------------------------------------------------------------------------
# Sequence triggers and actions.
# ---------------------------------------------------------------------------


def _sequence_entry(*lines):
    return ("[[sequence]]",) + lines


def test_tli_style_mission_validates(tmp_path):
    # Keplerian LEO start, kick-stage vehicle, timed TLI burn, SOI-transition
    # terminal event: the second target profile of Phase 4 (no GNC in the
    # loop). The kick stage carries no aero block, so no chdir is needed.
    kick = (FLEET_DIR / "kick_stage.toml").as_posix()
    text = _mission_text(
        exclude=("initial_state.cartesian",),
        replace={
            "vehicle": f'vehicle = "{kick}"',
            "mission.duration_s": "duration_s = 432000.0",
        },
        extra_lines=(
            "[initial_state.keplerian]",
            "sma_m = 6578137.0",
            "ecc = 0.0",
            "inc_deg = 28.5",
            "raan_deg = 0.0",
            "argp_deg = 0.0",
            "ta_deg = 0.0",
            "",
            *_sequence_entry(
                'name = "tli_ignite"',
                'trigger = "elapsed"',
                "t_s = 3600.0",
                'action = "ignite_engine"',
                'stage = "kick"',
                'engine = "main"',
            ),
            "",
            *_sequence_entry(
                'name = "tli_cutoff"',
                'trigger = "after_event"',
                'event = "tli_ignite"',
                "offset_s = 361.0",
                'action = "cutoff_engine"',
                'stage = "kick"',
                'engine = "main"',
            ),
            "",
            *_sequence_entry(
                'name = "moon_soi"',
                'trigger = "condition"',
                'condition = "soi_transition"',
                'body = "moon"',
                'action = "terminate"',
            ),
        ),
    )
    resolved, errors = _validate_text(tmp_path, text)
    assert errors == []
    assert resolved["sequence"][2]["body"] == "moon"


@pytest.mark.parametrize(
    ("entry", "needle"),
    [
        # Trigger surface.
        (
            ('name = "e"', 'trigger = "at_time"', "t_s = 1.0", 'action = "terminate"'),
            "[sequence.1] trigger:",
        ),
        (
            ('name = "e"', 'trigger = "elapsed"', "t_s = -1.0", 'action = "terminate"'),
            "[sequence.1] t_s: must be >= 0",
        ),
        (
            ('name = "e"', 'trigger = "elapsed"', "t_s = 601.0", 'action = "terminate"'),
            "can never fire",
        ),
        (
            ('name = "e"', 'trigger = "elapsed"', 'action = "terminate"'),
            "[sequence.1] t_s: missing required key",
        ),
        (
            ('name = "e"', 'trigger = "after_event"', 'event = "ghost"', "offset_s = 1.0", 'action = "terminate"'),
            "not an earlier sequence entry",
        ),
        (
            ('name = "e"', 'trigger = "after_event"', 'event = "e"', "offset_s = 1.0", 'action = "terminate"'),
            "not an earlier sequence entry",
        ),
        (
            ('name = "e"', 'trigger = "after_event"', 'event = "ghost"', 'action = "terminate"'),
            "[sequence.1] offset_s: missing required key",
        ),
        (
            ('name = "e"', 'trigger = "condition"', 'condition = "full_moon"', 'action = "terminate"'),
            "[sequence.1] condition: unknown condition",
        ),
        (
            ('name = "e"', 'trigger = "condition"', 'condition = "altitude_above"', "altitude_m = -5.0", 'action = "terminate"'),
            "[sequence.1] altitude_m:",
        ),
        (
            ('name = "e"', 'trigger = "condition"', 'condition = "soi_transition"', 'body = "earth"', 'action = "terminate"'),
            "already starts inside the SOI",
        ),
        (
            ('name = "e"', 'trigger = "condition"', 'condition = "soi_transition"', 'body = "pluto"', 'action = "terminate"'),
            "[sequence.1] body:",
        ),
        # Action surface.
        (
            ('name = "e"', 'trigger = "elapsed"', "t_s = 1.0", 'action = "warp"'),
            "[sequence.1] action: unknown action",
        ),
        (
            ('name = "e"', 'trigger = "elapsed"', "t_s = 1.0", 'action = "terminate"', "thrust_frac = 1.0"),
            "[sequence.1] thrust_frac: unknown key",
        ),
        (
            ('name = "e"', 'trigger = "elapsed"', "t_s = 1.0", 'action = "attitude_hold"'),
            "requires a vehicle reference",
        ),
    ],
)
def test_sequence_entry_errors_without_vehicle(tmp_path, entry, needle):
    # These defects are all independent of any vehicle: the mission omits the
    # vehicle key so the vehicle-free surface is exercised directly.
    text = _mission_text(exclude=("vehicle",), extra_lines=_sequence_entry(*entry))
    resolved, errors = _validate_text(tmp_path, text)
    assert resolved is None
    assert any(needle in e for e in errors), errors


@pytest.mark.parametrize(
    ("entry", "needle"),
    [
        (
            ('name = "e"', 'trigger = "elapsed"', "t_s = 1.0", 'action = "ignite_engine"', 'stage = "kick"', 'engine = "ghost"'),
            "unknown engine 'ghost'",
        ),
        (
            ('name = "e"', 'trigger = "elapsed"', "t_s = 1.0", 'action = "separate_stage"', 'stage = "ghost"'),
            "unknown stage 'ghost'",
        ),
        (
            ('name = "e"', 'trigger = "elapsed"', "t_s = 1.0", 'action = "jettison"', 'stage = "kick"', 'item = "ghost"'),
            "unknown jettison item 'ghost'",
        ),
        (
            ('name = "e"', 'trigger = "elapsed"', "t_s = 1.0", 'action = "rate_command"', 'frame = "lvlh"', "omega_dps = [0.1, 0.0, 0.0]"),
            "[sequence.1] frame:",
        ),
    ],
)
def test_sequence_vehicle_reference_errors(tmp_path, entry, needle):
    kick = (FLEET_DIR / "kick_stage.toml").as_posix()
    text = _mission_text(
        replace={"vehicle": f'vehicle = "{kick}"'},
        extra_lines=_sequence_entry(*entry),
    )
    resolved, errors = _validate_text(tmp_path, text)
    assert resolved is None
    assert any(needle in e for e in errors), errors


def test_pitch_program_table_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    # Index mismatch.
    text = _ascent_text().replace(
        "pitch_deg = [90.0, 75.0, 40.0, 30.0, 6.0, 0.0]",
        "pitch_deg = [90.0, 75.0, 40.0]",
    )
    resolved, errors = _validate_text(tmp_path, text)
    assert resolved is None
    assert any("one entry per pitch_t_s breakpoint" in e for e in errors), errors
    # Non-increasing time table.
    text = _ascent_text().replace(
        "pitch_t_s = [10.0, 25.0, 60.0, 121.0, 200.0, 300.0]",
        "pitch_t_s = [10.0, 25.0, 25.0, 121.0, 200.0, 300.0]",
    )
    resolved, errors = _validate_text(tmp_path, text)
    assert resolved is None
    assert any("strictly increasing" in e for e in errors), errors
    # Pitch outside [-90, 90].
    text = _ascent_text().replace(
        "pitch_deg = [90.0, 75.0, 40.0, 30.0, 6.0, 0.0]",
        "pitch_deg = [95.0, 75.0, 40.0, 30.0, 6.0, 0.0]",
    )
    resolved, errors = _validate_text(tmp_path, text)
    assert resolved is None
    assert any("[-90, 90]" in e for e in errors), errors


def test_duplicate_sequence_names_rejected(tmp_path):
    text = _mission_text(
        exclude=("vehicle",),
        extra_lines=(
            *_sequence_entry('name = "end"', 'trigger = "elapsed"', "t_s = 1.0", 'action = "terminate"'),
            "",
            *_sequence_entry('name = "end"', 'trigger = "elapsed"', "t_s = 2.0", 'action = "terminate"'),
        ),
    )
    resolved, errors = _validate_text(tmp_path, text)
    assert resolved is None
    assert any("[sequence.2] name: duplicate event name" in e for e in errors), errors


def test_sequence_errors_accumulate(tmp_path):
    text = _mission_text(
        exclude=("vehicle", "integrator.dt_s"),
        extra_lines=(
            *_sequence_entry('name = "a"', 'trigger = "warp"', 'action = "terminate"'),
            "",
            *_sequence_entry('name = "b"', 'trigger = "elapsed"', "t_s = 999.0", 'action = "terminate"'),
        ),
    )
    resolved, errors = _validate_text(tmp_path, text)
    assert resolved is None
    joined = "\n".join(errors)
    assert "[sequence.1] trigger:" in joined
    assert "can never fire" in joined
    assert "[integrator] dt_s: missing required" in joined
    assert len(errors) >= 3


# ---------------------------------------------------------------------------
# CLI surface: nonzero exit before any core call (exit criterion 1).
# ---------------------------------------------------------------------------


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


def test_cli_mutated_vehicle_exits_2_naming_the_exact_key(tmp_path):
    # The DX-2 exemplar case end to end: delete isp_vac_s from the starter
    # LV's stage-2 engine; `star run` must exit 2 (validation), naming
    # [stage.2.engine.1] isp_vac_s with units and the typical range, before
    # any core call.
    mutated = tmp_path / "electron_mutated.toml"
    mutated.write_text(
        (FLEET_DIR / "electron_class.toml")
        .read_text(encoding="utf-8")
        .replace("isp_vac_s = 343.0\n", ""),
        encoding="utf-8",
    )
    mission = tmp_path / "mission.toml"
    mission.write_text(
        _mission_text(replace={"vehicle": f'vehicle = "{mutated.as_posix()}"'}),
        encoding="utf-8",
    )
    proc = _run_cli("run", str(mission), "-o", str(tmp_path / "out"), cwd=str(REPO_ROOT))
    assert proc.returncode == 2
    assert "[stage.2.engine.1] isp_vac_s: missing required key" in proc.stderr
    assert "units: s; typical range typical chemical range 200-465" in proc.stderr
    assert "No default applied; run aborted." in proc.stderr


def test_cli_valid_vehicle_mission_propagates_with_exit_0(tmp_path):
    # A valid Phase 4 vehicle mission now propagates through run_vehicle (the
    # Phase 4 6DOF path): the default mission is a coasting smallsat in LEO
    # (vehicle reference, cartesian initial state, no sequence). Requires the
    # compiled core; on a core-less checkout the CLI exits 1 with the
    # actionable core-missing hint instead.
    mission = tmp_path / "mission.toml"
    mission.write_text(_mission_text(), encoding="utf-8")
    out = tmp_path / "out"
    proc = _run_cli("run", str(mission), "-o", str(out), cwd=str(REPO_ROOT))
    try:
        from star_reacher import _core  # noqa: F401

        have_core = True
    except ImportError:
        have_core = False
    if have_core:
        assert proc.returncode == 0, proc.stderr
        assert (out / "run.srlog").is_file()
        assert (out / "resolved_vehicle.toml").is_file()
    else:
        assert proc.returncode == 1
        assert "_core is not built" in proc.stderr


def test_cli_strict_promotes_vehicle_warnings_to_exit_2(tmp_path):
    mission = tmp_path / "mission.toml"
    mission.write_text(
        _mission_text(
            replace={
                "vehicle": 'vehicle = "tests/fixtures/vehicles/warning_low_liftoff_tw.toml"'
            }
        ),
        encoding="utf-8",
    )
    strict = _run_cli(
        "run", "--strict", str(mission), "-o", str(tmp_path / "out"), cwd=str(REPO_ROOT)
    )
    assert strict.returncode == 2
    assert "thrust-to-weight" in strict.stderr
    assert "Promoted to an error by --strict" in strict.stderr

    advisory = _run_cli("run", str(mission), "-o", str(tmp_path / "out"), cwd=str(REPO_ROOT))
    # Without --strict the warning is advisory (stderr) and the mission is
    # valid, so it propagates through run_vehicle (exit 0 with the compiled
    # core; exit 1 with the core-missing hint on a core-less checkout).
    try:
        from star_reacher import _core  # noqa: F401

        have_core = True
    except ImportError:
        have_core = False
    assert advisory.returncode == (0 if have_core else 1)
    assert "thrust-to-weight" in advisory.stderr
