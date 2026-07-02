"""Canonical resolved-config serialization and SHA-256 stability (FR-15,
contract section 5). No compiled core is required."""

import hashlib
from pathlib import Path

from star_reacher.mission import canonical_bytes, config_sha256, validate_mission_file

REPO_ROOT = Path(__file__).resolve().parents[2]

_BASE_MISSION = """\
schema_version = 1

[mission]
name = "hash-test"
epoch_utc = "2026-01-01T00:00:00Z"
duration_s = 600.0

[run]
seed = 7

[integrator]
type = "rk4"
dt_s = 0.1

[initial_state.cartesian]
r_m = [6778137.0, 0.0, 0.0]
v_mps = [0.0, 7668.6, 0.0]
frame = "GCRF"

[environment]
central_body = "earth"
"""

# Same document with the tables and keys written in a different order (the
# root key stays first: TOML assigns bare keys after a [table] header to that
# table). The canonical form must not depend on authoring order.
_REORDERED_MISSION = """\
schema_version = 1

[environment]
central_body = "earth"

[integrator]
dt_s = 0.1
type = "rk4"

[initial_state.cartesian]
frame = "GCRF"
v_mps = [0.0, 7668.6, 0.0]
r_m = [6778137.0, 0.0, 0.0]

[run]
seed = 7

[mission]
duration_s = 600.0
name = "hash-test"
epoch_utc = "2026-01-01T00:00:00Z"
"""


def _resolve(tmp_path, text, name="m.toml"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    resolved, errors = validate_mission_file(path)
    assert errors == [], errors
    return resolved


def test_same_mission_hashes_identically(tmp_path):
    a = _resolve(tmp_path, _BASE_MISSION, "a.toml")
    b = _resolve(tmp_path, _BASE_MISSION, "b.toml")
    assert canonical_bytes(a) == canonical_bytes(b)
    assert config_sha256(a) == config_sha256(b)


def test_toml_table_order_does_not_change_hash(tmp_path):
    a = _resolve(tmp_path, _BASE_MISSION, "a.toml")
    b = _resolve(tmp_path, _REORDERED_MISSION, "b.toml")
    assert config_sha256(a) == config_sha256(b)


def test_changed_value_changes_hash(tmp_path):
    a = _resolve(tmp_path, _BASE_MISSION, "a.toml")
    b = _resolve(tmp_path, _BASE_MISSION.replace("seed = 7", "seed = 8"), "b.toml")
    assert config_sha256(a) != config_sha256(b)


def test_canonical_form_properties():
    # Sorted keys, compact separators, shortest-repr floats: the exact byte
    # recipe of contract section 5.
    assert canonical_bytes({"b": 0.1, "a": 1}) == b'{"a":1,"b":0.1}'
    assert canonical_bytes({"x": 5400.0}) == b'{"x":5400.0}'
    assert canonical_bytes({"x": [6778137.0, 0.0]}) == b'{"x":[6778137.0,0.0]}'


def test_sha_is_over_exactly_the_canonical_bytes():
    resolved = {"schema_version": 1, "x": 0.5}
    assert config_sha256(resolved) == hashlib.sha256(canonical_bytes(resolved)).hexdigest()


def test_reference_mission_hash_is_stable():
    # Golden hash for missions/twobody_leo.toml: guards against silent drift
    # of the canonicalization (key order, float formatting, defaults), which
    # would break run-log/config binding across versions. Regenerate only
    # with a deliberate schema_version bump or mission edit:
    #   python -c "from star_reacher.mission import *; \
    #     print(config_sha256(validate_mission_file('missions/twobody_leo.toml')[0]))"
    resolved, errors = validate_mission_file(REPO_ROOT / "missions" / "twobody_leo.toml")
    assert errors == []
    assert config_sha256(resolved) == (
        "64c57d92780ce5a7f88fd64b96526a8d78fa4ed500dfa64d92c89aee846151f7"
    )


def test_defaults_are_recorded_in_canonical_form(tmp_path):
    # The defaults-applied values must appear in the hashed bytes: a config
    # hash that omitted them could collide across different defaults.
    resolved = _resolve(tmp_path, _BASE_MISSION)
    text = canonical_bytes(resolved).decode("utf-8")
    assert '"mass_kg":1.0' in text
    assert '"truth_rate_hz":10' in text
    assert '"schema_version":1' in text
