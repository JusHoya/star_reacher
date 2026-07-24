#!/usr/bin/env python3
"""Two-key golden hash gate (FR-22 layer 6): CI rejects unmanaged golden edits.

This is the mechanical half of the two-key golden-update policy. A golden value
file under hash-gate carries a ``values_sha256`` in its directory's
``manifest.toml`` ``[[file]]`` entry, recorded by ``scripts/golden_update.py``
whenever it regenerates the file. This gate recomputes each such file's SHA-256
from its bytes on disk and asserts it equals the manifest's recorded value.

Consequently a hand-edit of a golden value -- bypassing the tool -- changes the
file's bytes but not the manifest's recorded hash, and CI FAILS; a regeneration
through ``golden_update.py --apply`` updates the value and the manifest hash
together, and CI passes. That is the PRD's "CI rejects golden changes without
manifest updates".

Scope. The gate is opt-in and additive: a ``[[file]]`` entry WITHOUT a
``values_sha256`` is reported as "not under hash-gate" and never fails, so the
Phase 1-6 manifests that predate this field are untouched. A directory MAY be
required to be under the gate with ``--require`` (used for the directories this
policy owns), which turns a missing ``values_sha256`` into a failure so a new
golden cannot silently escape the gate. The manifest is also checked for the
well-formedness the schema requires (schema_version, [golden], one [[file]] per
named file, no manifest entry naming an absent file, no un-manifested value
file in a required directory).

Exit status: 0 when every checked manifest is consistent; 1 with each problem
named otherwise; 2 when a required directory or manifest cannot be found.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tomllib
from pathlib import Path

# Directories this policy owns: their manifests must carry a values_sha256 for
# every value file. Extended as new hash-gated golden dirs land.
_REQUIRED_DIRS = ("mc_regression",)

# Files a golden directory holds that are not themselves golden VALUES and so
# carry no values_sha256: the manifest, the regeneration script, and Python
# byte-cache. Everything else in a required directory must be manifested.
_NON_VALUE_NAMES = {"manifest.toml", "generate.py", "__pycache__"}


def repo_root_default() -> Path:
    # The script lives in <root>/scripts/, so the default root is its parent.
    return Path(__file__).resolve().parent.parent


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_manifest(manifest_path: Path, *, require: bool) -> list[str]:
    """Check one golden ``manifest.toml``; return a list of problem strings.

    An empty list means the directory is consistent. ``require`` makes a value
    file without a ``values_sha256`` a problem rather than an untracked skip.
    """
    problems: list[str] = []
    directory = manifest_path.parent
    rel = directory.name
    try:
        with manifest_path.open("rb") as fh:
            doc = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        return [f"{rel}: manifest is unreadable or malformed TOML: {exc}"]

    if doc.get("schema_version") != 1:
        problems.append(
            f"{rel}: schema_version is {doc.get('schema_version')!r}, expected 1"
        )
    if not isinstance(doc.get("golden"), dict):
        problems.append(f"{rel}: missing the [golden] table")

    file_entries = doc.get("file")
    if not isinstance(file_entries, list) or not file_entries:
        problems.append(f"{rel}: manifest declares no [[file]] entries")
        file_entries = []

    manifested: set[str] = set()
    for entry in file_entries:
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            problems.append(f"{rel}: a [[file]] entry has no 'name'")
            continue
        manifested.add(name)
        recorded = entry.get("values_sha256")
        if recorded is None:
            if require:
                problems.append(
                    f"{rel}/{name}: no values_sha256 in the manifest, but this "
                    f"directory is required to be under the hash-gate; "
                    f"regenerate it with scripts/golden_update.py --apply"
                )
            # Not under the hash-gate: this entry predates the field (and its
            # 'name' may even be a descriptive label rather than a single
            # file), so it is not this gate's concern. Skip silently.
            continue
        value_path = directory / name
        if not value_path.is_file():
            # Only reachable for a gated entry: a recorded hash for a file that
            # is not there is itself a broken golden.
            problems.append(
                f"{rel}/{name}: manifest records a values_sha256 but the file "
                f"is absent"
            )
            continue
        actual = _sha256_file(value_path)
        if actual != recorded:
            problems.append(
                f"{rel}/{name}: on-disk SHA-256 {actual} does not match the "
                f"manifest's recorded values_sha256 {recorded}; a golden value "
                f"was changed without going through scripts/golden_update.py "
                f"--apply (which updates both together)"
            )

    if require:
        # Every value file in a required directory must be manifested, so a new
        # golden cannot be dropped in beside the manifest and escape the gate.
        for child in sorted(directory.iterdir()):
            if child.name in _NON_VALUE_NAMES or child.name in manifested:
                continue
            if child.is_file():
                problems.append(
                    f"{rel}/{child.name}: a value file with no [[file]] entry "
                    f"in the manifest"
                )
    return problems


def discover_manifests(golden_root: Path) -> list[Path]:
    """Every ``manifest.toml`` under the golden tree, in sorted directory order."""
    return sorted(golden_root.glob("*/manifest.toml"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    root = repo_root_default()
    parser.add_argument(
        "--golden-root",
        type=Path,
        default=root / "tests" / "golden",
        help="root of the golden tree (default: tests/golden)",
    )
    parser.add_argument(
        "--dir",
        action="append",
        default=None,
        help="check only this golden directory name (repeatable); default: all",
    )
    args = parser.parse_args(argv)

    golden_root = args.golden_root
    if not golden_root.is_dir():
        print(
            f"check_golden_manifests: golden root not found: {golden_root}",
            file=sys.stderr,
        )
        return 2

    if args.dir:
        manifests = []
        for name in args.dir:
            manifest = golden_root / name / "manifest.toml"
            if not manifest.is_file():
                print(
                    f"check_golden_manifests: no manifest for requested "
                    f"directory {name!r}: {manifest}",
                    file=sys.stderr,
                )
                return 2
            manifests.append(manifest)
    else:
        manifests = discover_manifests(golden_root)

    # A required directory must actually be present and checked, so the gate
    # cannot be defeated by removing the whole directory.
    checked_dirs = {m.parent.name for m in manifests}
    all_problems: list[str] = []
    for required in _REQUIRED_DIRS:
        if required not in checked_dirs and (args.dir is None or required in args.dir):
            all_problems.append(
                f"{required}: this directory is required to be under the "
                f"hash-gate but no manifest was found for it"
            )

    checked = 0
    for manifest in manifests:
        require = manifest.parent.name in _REQUIRED_DIRS
        problems = check_manifest(manifest, require=require)
        all_problems.extend(problems)
        checked += 1

    if all_problems:
        print("check_golden_manifests: FAIL", file=sys.stderr)
        for problem in all_problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    print(
        f"check_golden_manifests: OK ({checked} manifest(s) checked; "
        f"hash-gated dirs consistent)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
