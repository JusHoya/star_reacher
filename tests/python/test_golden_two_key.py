"""Two-key golden-update mechanism (FR-22 layer 6).

The PRD clause is "goldens regenerate only through the two-key path (CI rejects
otherwise)". These tests exercise the mechanism end to end on the committed
Monte Carlo regression golden:

- ``scripts/golden_update.py`` (key 1, human intent): a dry run detects a
  pending change and emits a diff, and ``--apply`` writes the value file and
  the manifest hash together;
- ``scripts/check_golden_manifests.py`` (key 2, mechanical enforcement): it
  passes on a consistent value/manifest pair and FAILS when a value file is
  hand-edited without a matching manifest update.

The checker tests run on temp COPIES of the committed golden, so a hand-edit
mutation never touches the real tree. The ``--apply`` regeneration test needs
the compiled core (it runs the sweep); it is guarded and fails, never skips,
when the core is absent.
"""

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = REPO_ROOT / "tests" / "golden" / "mc_regression"
VALUE_FILE = GOLDEN_DIR / "energy_stats.toml"
MANIFEST_FILE = GOLDEN_DIR / "manifest.toml"
CHECK = REPO_ROOT / "scripts" / "check_golden_manifests.py"
UPDATE = REPO_ROOT / "scripts" / "golden_update.py"

_CORE_MISSING = (
    "star_reacher._core is not built; golden_update.py --apply runs the sweep "
    "and requires the compiled core. Expected to fail on a core-less checkout."
)


def _core_or_fail():
    try:
        from star_reacher import _core  # noqa: F401
    except ImportError:
        pytest.fail(_CORE_MISSING)


def _run(script, *args):
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )


def _copy_golden(tmp_path):
    """A temp golden root with a copy of the committed mc_regression pair."""
    root = tmp_path / "golden"
    dest = root / "mc_regression"
    dest.mkdir(parents=True)
    shutil.copy(VALUE_FILE, dest / VALUE_FILE.name)
    shutil.copy(MANIFEST_FILE, dest / MANIFEST_FILE.name)
    return root, dest


# ---------------------------------------------------------------------------
# The committed golden is internally consistent right now.


def test_committed_golden_is_consistent():
    """The checker passes on the tree as committed (the invariant CI holds)."""
    result = _run(CHECK)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_recorded_hash_matches_the_value_file_bytes():
    """The manifest's values_sha256 is exactly the value file's SHA-256."""
    import tomllib

    with MANIFEST_FILE.open("rb") as fh:
        manifest = tomllib.load(fh)
    recorded = manifest["file"][0]["values_sha256"]
    actual = hashlib.sha256(VALUE_FILE.read_bytes()).hexdigest()
    assert actual == recorded


# ---------------------------------------------------------------------------
# key 2: the CI checker passes on a consistent pair and fails on a hand-edit.


def test_checker_passes_on_consistent_copy(tmp_path):
    root, _dest = _copy_golden(tmp_path)
    result = _run(CHECK, "--golden-root", str(root), "--dir", "mc_regression")
    assert result.returncode == 0, result.stdout + result.stderr


def test_checker_fails_on_a_hand_edited_value(tmp_path):
    """A byte changed in the value file without a manifest update is rejected.

    This is the whole point of the gate: the hand-edit changes the file's
    bytes but not the manifest's recorded hash, so the recomputed SHA-256
    disagrees and CI fails.
    """
    root, dest = _copy_golden(tmp_path)
    value = dest / "energy_stats.toml"
    data = value.read_bytes()
    edited = data.replace(b"std_readable = 199418", b"std_readable = 299418")
    assert edited != data, "the mutation did not change any bytes"
    value.write_bytes(edited)

    result = _run(CHECK, "--golden-root", str(root), "--dir", "mc_regression")
    assert result.returncode == 1
    assert "does not match" in (result.stdout + result.stderr)


def test_checker_fails_when_required_dir_is_missing(tmp_path):
    """Removing the whole hash-gated directory does not defeat the gate."""
    root = tmp_path / "golden"
    (root / "other").mkdir(parents=True)  # some unrelated dir, no mc_regression
    result = _run(CHECK, "--golden-root", str(root))
    assert result.returncode == 1
    assert "required to be under the hash-gate" in (result.stdout + result.stderr)


def test_checker_fails_on_unmanifested_value_file(tmp_path):
    """A value file dropped into a required dir with no [[file]] entry fails."""
    root, dest = _copy_golden(tmp_path)
    (dest / "sneaked_in.toml").write_text("x = 1\n", encoding="utf-8")
    result = _run(CHECK, "--golden-root", str(root), "--dir", "mc_regression")
    assert result.returncode == 1
    assert "no [[file]] entry" in (result.stdout + result.stderr)


# ---------------------------------------------------------------------------
# key 1: golden_update dry run and --apply.


def test_dry_run_reports_no_pending_change():
    """On an unchanged binary the committed golden matches a fresh sweep."""
    _core_or_fail()
    result = _run(UPDATE)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "no change" in result.stdout


def test_dry_run_detects_a_pending_change_and_emits_a_diff():
    """A stale committed golden makes the dry run exit nonzero with a diff.

    The committed value is tampered in place (with a backup and an
    unconditional restore), so a fresh sweep disagrees with what is on disk; the
    dry run must print a DIFF SUMMARY and exit 1 WITHOUT writing anything. The
    base mission's gravity field path is repo-root-relative, so the tool must
    run against the real tree; the backup/restore keeps the tree unchanged.
    """
    _core_or_fail()
    tampered_value = VALUE_FILE.read_bytes().replace(
        b'std_hex = "0x1.857d1c2e64849p+17"',
        b'std_hex = "0x1.0000000000000p+18"',
    )
    backup = VALUE_FILE.read_bytes()
    assert tampered_value != backup, "the tamper did not change any bytes"
    try:
        VALUE_FILE.write_bytes(tampered_value)
        result = _run(UPDATE)  # dry run against the tampered committed value
        assert result.returncode == 1, result.stdout + result.stderr
        assert "DIFF SUMMARY" in result.stdout
        assert "dry run" in (result.stdout + result.stderr)
        # Nothing was written: the value on disk is still the tampered one.
        assert VALUE_FILE.read_bytes() == tampered_value
    finally:
        VALUE_FILE.write_bytes(backup)
    # After restore the checker is consistent again.
    assert _run(CHECK).returncode == 0


def test_apply_updates_value_and_manifest_together(tmp_path):
    """--apply on a tampered committed golden restores a consistent pair.

    The committed value is tampered in place (with a backup), --apply
    regenerates it from the sweep and rewrites the manifest hash, and the
    checker then passes -- proving --apply writes both halves consistently.
    Restored afterward so the test leaves the tree as it found it.
    """
    _core_or_fail()
    backup_value = VALUE_FILE.read_bytes()
    backup_manifest = MANIFEST_FILE.read_bytes()
    try:
        tampered = backup_value.replace(b"n = 128", b"n = 128\n# tampered\n")
        VALUE_FILE.write_bytes(tampered)
        # Checker fails on the tampered value (hash mismatch).
        assert _run(CHECK).returncode == 1
        # --apply regenerates value + manifest together.
        result = _run(UPDATE, "--apply")
        assert result.returncode == 0, result.stdout + result.stderr
        assert "APPLIED" in result.stdout
        # The pair is consistent again, and the restored value matches the
        # original committed bytes (the sweep is deterministic).
        assert _run(CHECK).returncode == 0
        assert VALUE_FILE.read_bytes() == backup_value
    finally:
        VALUE_FILE.write_bytes(backup_value)
        MANIFEST_FILE.write_bytes(backup_manifest)
