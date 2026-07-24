"""``star run --seed`` / ``--set`` overrides (FR-27 re-execution path).

Two things are asserted:

* the CLI parses ``--seed`` and ``--set DOTTED.PATH=VALUE`` (number or JSON
  array of numbers), and a bad path/kind/value is a validation error (exit 2);
* the override path applied by ``star run`` is the same transformation as
  ``Sim.reset(overrides=...)`` -- a run with ``--set`` reproduces the exact
  SHA-256 of a ``Sim.reset(overrides=...)`` run of the same mission -- so the
  two override surfaces cannot drift.

Tests needing the compiled core fail (never skip) when it is absent, the
project's agent-honesty gate; the pure parsing/override tests need no core.
"""

import hashlib
import os
from pathlib import Path

import pytest

from star_reacher.cli import _parse_overrides, _parse_set_value, main

REPO_ROOT = Path(__file__).resolve().parents[2]
MISSION = REPO_ROOT / "missions" / "twobody_leo.toml"
GNC_MISSION = REPO_ROOT / "missions" / "leo_attitude_gnc.toml"

_CORE_MISSING = (
    "star_reacher._core is not built in this environment. These tests require "
    "the compiled core: build and install it with 'pip install .'. This "
    "failure is expected on a core-less checkout and must be green at CI."
)


def _core_or_fail():
    try:
        from star_reacher import _core  # noqa: F401
    except ImportError:
        pytest.fail(_CORE_MISSING)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --- --set value parsing (no core) -----------------------------------------


def test_parse_set_value_number():
    assert _parse_set_value("12") == 12
    assert _parse_set_value("5.4e3") == 5400.0
    assert _parse_set_value("-3.5") == -3.5


def test_parse_set_value_array():
    assert _parse_set_value("[0.5,0.5,0.5]") == [0.5, 0.5, 0.5]
    assert _parse_set_value("[1, 2, 3]") == [1, 2, 3]


@pytest.mark.parametrize(
    "raw",
    [
        '"a string"',      # a JSON string is not a number
        "true",            # a boolean is not a number
        "[]",              # an empty array has no length to match a leaf
        '["a", "b"]',      # an array of non-numbers
        "[1, [2]]",        # a nested/mixed array
        "{}",              # an object
        "not json",        # unparseable
    ],
)
def test_parse_set_value_rejects_non_numeric(raw):
    with pytest.raises(ValueError):
        _parse_set_value(raw)


def test_parse_overrides_builds_a_dict():
    got = _parse_overrides(
        ["mission.duration_s=120", "gnc.control.kp_nm_per_rad=[0.5,0.5,0.5]"]
    )
    assert got == {
        "mission.duration_s": 120,
        "gnc.control.kp_nm_per_rad": [0.5, 0.5, 0.5],
    }


@pytest.mark.parametrize(
    "entries, fragment",
    [
        (["no_equals_sign"], "expected DOTTED.PATH=VALUE"),
        (["=5"], "path before '=' is empty"),
        (["a.b=1", "a.b=2"], "more than once"),
        (["a.b=oops"], "not a number"),
    ],
)
def test_parse_overrides_errors(entries, fragment):
    with pytest.raises(ValueError, match=fragment):
        _parse_overrides(entries)


# --- CLI exit codes (need the core to actually run) ------------------------


def test_cli_set_changes_the_config_hash(tmp_path, capsys):
    """A --set run resolves and hashes to a different config than the base."""
    _core_or_fail()
    base_dir = tmp_path / "base"
    set_dir = tmp_path / "set"
    assert main(["run", str(MISSION), "-o", str(base_dir), "--force"]) == 0
    base_out = capsys.readouterr().out
    assert (
        main(
            [
                "run",
                str(MISSION),
                "--set",
                "mission.duration_s=1200.0",
                "-o",
                str(set_dir),
                "--force",
            ]
        )
        == 0
    )
    set_out = capsys.readouterr().out
    base_hash = _config_hash_from_cli(base_out)
    set_hash = _config_hash_from_cli(set_out)
    assert base_hash != set_hash


def test_cli_bad_set_path_is_a_validation_error(tmp_path, capsys):
    """An override naming no existing leaf exits 2 (validation), not 1."""
    _core_or_fail()
    code = main(
        [
            "run",
            str(MISSION),
            "--set",
            "mission.no_such_key=1.0",
            "-o",
            str(tmp_path / "bad"),
            "--force",
        ]
    )
    err = capsys.readouterr().err
    assert code == 2
    assert "no such key" in err


def test_cli_malformed_set_is_a_validation_error(tmp_path, capsys):
    """A --set with no '=' exits 2 without touching the mission."""
    _core_or_fail()
    code = main(
        ["run", str(MISSION), "--set", "oops", "-o", str(tmp_path / "m"), "--force"]
    )
    err = capsys.readouterr().err
    assert code == 2
    assert "DOTTED.PATH=VALUE" in err


def _config_hash_from_cli(stdout):
    for line in stdout.splitlines():
        if line.startswith("resolved config sha256:"):
            return line.split(":", 1)[1].strip()
    raise AssertionError(f"no config hash in CLI output:\n{stdout}")


# --- the two override paths agree (criterion-1 equivalence in miniature) ---


def test_run_set_reproduces_sim_reset_hash(tmp_path):
    """--set and Sim.reset(overrides=...) are the same transformation.

    A vehicle mission (Sim requires the vehicle path). Both apply the overrides
    before hashing, so the two runs must produce byte-identical logs.
    """
    _core_or_fail()
    from star_reacher.runner import run_mission
    from star_reacher.sim import Sim

    overrides = {
        "mission.duration_s": 30.0,
        "gnc.control.kp_nm_per_rad": [0.5, 0.5, 0.5],
    }
    seed = 424242

    cwd = os.getcwd()
    os.chdir(REPO_ROOT)  # resolve the mission's relative vehicle path
    try:
        run_result = run_mission(
            GNC_MISSION,
            outdir=str(tmp_path / "run"),
            force=True,
            seed=seed,
            overrides=overrides,
        )
        with Sim(GNC_MISSION, tmp_path / "sim", force=True) as sim:
            sim.reset(seed=seed, overrides=overrides)
            sim.run_to_completion()
    finally:
        os.chdir(cwd)

    run_log = tmp_path / "run" / "run.srlog"
    sim_log = tmp_path / "sim" / "run.srlog"
    assert _sha256(run_log) == _sha256(sim_log), (
        "star run --set and Sim.reset(overrides=...) produced different logs; "
        "the two override paths have drifted apart"
    )
    # And the overridden run really is a distinct scenario: its config hash
    # differs from an unoverridden run of the same mission.
    cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        base = run_mission(GNC_MISSION, outdir=str(tmp_path / "base"), force=True)
    finally:
        os.chdir(cwd)
    assert run_result.config_sha256 != base.config_sha256


def test_run_seed_only_reproduces_sim_reset(tmp_path):
    """--seed alone (no --set) also matches the Sim.reset seed path."""
    _core_or_fail()
    from star_reacher.runner import run_mission
    from star_reacher.sim import Sim

    cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        run_mission(
            GNC_MISSION, outdir=str(tmp_path / "run"), force=True, seed=99
        )
        with Sim(GNC_MISSION, tmp_path / "sim", force=True) as sim:
            sim.reset(seed=99)
            sim.run_to_completion()
    finally:
        os.chdir(cwd)

    assert _sha256(tmp_path / "run" / "run.srlog") == _sha256(
        tmp_path / "sim" / "run.srlog"
    )


def test_large_u64_seed_is_accepted(tmp_path):
    """A full-width u64 seed (as SplitMix64 produces) is not misread.

    Regression: routing an integer override through float() rejected an exact
    u64 as 'would be truncated' because float loses precision above 2**53.
    """
    _core_or_fail()
    from star_reacher.runner import run_mission

    big_seed = 17569168275178538554  # > 2**53, a real per-run seed
    result = run_mission(
        MISSION, outdir=str(tmp_path / "big"), force=True, seed=big_seed
    )
    assert len(result.srlog_sha256) == 64
