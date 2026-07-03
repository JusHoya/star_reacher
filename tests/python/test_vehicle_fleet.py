"""Starter-fleet tests and the Phase 4 exit-criterion 1 mutation gate.

The gate, quoted from the PRD: "deleting any required key from any starter
vehicle yields nonzero exit naming that exact key; one unknown key is
rejected; all four starter vehicles validate under --strict; resolved-config
echo re-validates byte-identically." At the library level "nonzero exit"
means a non-empty accumulated error list (the CLI maps that to exit 2, which
the subprocess tests in test_mission_sequence.py prove end to end).

Every required-key instance is enumerated programmatically from the
committed fleet files against the validator's own REQUIRED_KEYS registry,
deleted by dict surgery on the parsed document, re-serialized through the
canonical writer, and re-validated -- so a schema key added without a
matching abort fails this gate. No compiled core is required.
"""

import tomllib
from pathlib import Path

import pytest

from star_reacher.mission import config_sha256
from star_reacher.vehicle import (
    REQUIRED_KEYS,
    canonical_vehicle_toml,
    validate_vehicle_file,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FLEET_DIR = REPO_ROOT / "vehicles"
FLEET = ["electron_class.toml", "kick_stage.toml", "smallsat.toml", "probe.toml"]

_FLEET_TEXT = {name: (FLEET_DIR / name).read_text(encoding="utf-8") for name in FLEET}


def _iter_required_instances(doc):
    """Every (table_path, key) whose deletion must abort validation."""
    for key in REQUIRED_KEYS["root"]:
        yield ("root", key)
    for key in REQUIRED_KEYS["vehicle"]:
        yield ("vehicle", key)
    for i, stage in enumerate(doc["stage"], 1):
        for key in REQUIRED_KEYS["stage"]:
            yield (f"stage.{i}", key)
        for kind in ("tank", "engine", "rcs", "wheel", "sensor", "jettison"):
            for j, _ in enumerate(stage.get(kind, ()), 1):
                for key in REQUIRED_KEYS[f"stage.{kind}"]:
                    yield (f"stage.{i}.{kind}.{j}", key)
    for i, _ in enumerate(doc.get("aero", ()), 1):
        for key in REQUIRED_KEYS["aero"]:
            yield (f"aero.{i}", key)


def _delete_at(doc, table, key):
    """Delete ``key`` from the (possibly array-indexed) table path."""
    if table == "root":
        del doc[key]
        return
    node = doc
    parts = table.split(".")
    i = 0
    while i < len(parts):
        node = node[parts[i]]
        if i + 1 < len(parts) and parts[i + 1].isdigit():
            node = node[int(parts[i + 1]) - 1]
            i += 2
        else:
            i += 1
    del node[key]


_MUTATIONS = [
    (name, table, key)
    for name in FLEET
    for table, key in _iter_required_instances(tomllib.loads(_FLEET_TEXT[name]))
]


@pytest.mark.parametrize("name", FLEET)
def test_fleet_validates_under_strict(name, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    resolved, errors, warns = validate_vehicle_file(FLEET_DIR / name, strict=True)
    assert errors == [], errors
    assert warns == [], warns
    assert resolved is not None
    assert resolved["provenance"] == "representative"


@pytest.mark.parametrize(
    ("name", "table", "key"),
    _MUTATIONS,
    ids=[f"{n}:{t}:{k}" for n, t, k in _MUTATIONS],
)
def test_deleting_any_required_key_names_that_exact_key(name, table, key, tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    doc = tomllib.loads(_FLEET_TEXT[name])
    _delete_at(doc, table, key)
    mutated = tmp_path / name
    mutated.write_text(canonical_vehicle_toml(doc), encoding="utf-8", newline="")
    resolved, errors, _ = validate_vehicle_file(mutated)
    assert resolved is None
    assert any(
        f"[{table}] {key}:" in e and "missing required" in e for e in errors
    ), (table, key, errors)


@pytest.mark.parametrize("name", FLEET)
def test_one_unknown_key_is_rejected(name, tmp_path, monkeypatch):
    # Appended at the end of the file, the key lands inside whatever table is
    # open last -- a different nesting level per fleet file, which is the
    # point: rejection must hold anywhere in the file.
    monkeypatch.chdir(REPO_ROOT)
    mutated = tmp_path / name
    mutated.write_text(_FLEET_TEXT[name] + "\ntypo_knob = 1.0\n", encoding="utf-8")
    resolved, errors, _ = validate_vehicle_file(mutated)
    assert resolved is None
    assert any("typo_knob: unknown key" in e for e in errors), errors


@pytest.mark.parametrize("name", FLEET)
def test_resolved_echo_revalidates_byte_identically(name, tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    resolved, errors, _ = validate_vehicle_file(FLEET_DIR / name)
    assert errors == [], errors
    echo1 = canonical_vehicle_toml(resolved)
    echo_path = tmp_path / f"echo_{name}"
    echo_path.write_text(echo1, encoding="utf-8", newline="")
    resolved2, errors2, _ = validate_vehicle_file(echo_path)
    assert errors2 == [], errors2
    assert canonical_vehicle_toml(resolved2) == echo1
    assert config_sha256(resolved2) == config_sha256(resolved)


def test_fleet_matches_the_dx3_manifest(monkeypatch):
    """Pin the DX-3 fleet contents the downstream missions rely on."""
    monkeypatch.chdir(REPO_ROOT)

    lv, _, _ = validate_vehicle_file(FLEET_DIR / "electron_class.toml")
    assert [s["name"] for s in lv["stage"]] == ["stage1", "stage2"]
    # Jettisonable fairing plus the payload allowance the capability check
    # in the file header is computed with.
    assert {j["name"] for j in lv["stage"][1]["jettison"]} == {"fairing", "payload_stack"}
    # One aero block per stack configuration.
    assert [a["config"] for a in lv["aero"]] == ["full_stack", "upper_stack"]
    # The 3DOF capability check in the header assumed exactly these numbers.
    assert lv["stage"][0]["engine"][0]["thrust_vac_N"] == 234000.0
    assert lv["stage"][0]["tank"][0]["propellant_mass_kg"] == 9500.0
    assert lv["stage"][1]["engine"][0]["isp_vac_s"] == 343.0

    kick, _, _ = validate_vehicle_file(FLEET_DIR / "kick_stage.toml")
    # Restartable is the point of this vehicle (DX-3).
    assert kick["stage"][0]["engine"][0]["ignitions"] == 4
    # The TLI capability in the file header is computed with this item.
    assert kick["stage"][0]["jettison"][0]["mass_kg"] == 160.0

    sat, _, _ = validate_vehicle_file(FLEET_DIR / "smallsat.toml")
    # RCS + reaction wheels, no main propulsion (DX-3).
    assert len(sat["stage"][0]["wheel"]) == 3
    assert "engine" not in sat["stage"][0]
    assert "tank" not in sat["stage"][0]

    probe, _, _ = validate_vehicle_file(FLEET_DIR / "probe.toml")
    # The AI/ML navigation testbed carries the full Phase 6 sensor
    # complement as placement declarations.
    assert {s["name"] for s in probe["stage"][0]["sensor"]} == {
        "imu0",
        "st0",
        "css0",
        "navfix0",
    }
    assert probe["stage"][0]["engine"][0]["gimbal_max_deg"] == 0.0
