#!/usr/bin/env python3
"""FR-29 accretion gate: hold core model code and math-library chapters together.

The invariant, enforced on every CI build: for every entry in
``docs/mathlib/chapter_manifest.toml``, if the listed module file exists in
the tree then (a) the listed chapter file must exist, (b) the chapter must
contain its declared ``\\label{ch:...}``, and (c) every test identifier the
chapter names in its validation-evidence table (marked with the
``\\testid{...}`` macro) must exist in the test tree. A model without its
chapter is a red build. The converse -- a chapter whose module does not exist
yet -- is deliberately allowed, because chapters are written ahead of or in
parallel with the code they document.

Test-identifier resolution is a substring match: an identifier counts as
present if it appears verbatim in the name or the contents of any file under
``cpp/tests/``, ``tests/``, or ``python/star_reacher/verify*.py``. Substring
matching is sufficient because test identifiers are long and unique by
convention (e.g. ``twobody_circular_orbit_analytic``), and it keeps the gate
agnostic to how each test framework spells its registrations.

Exit status: 0 when the invariant holds; 1 with every failure listed
otherwise; 2 on a malformed manifest.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

# \testid{...} is the machine-readable marker for validation-evidence test
# identifiers, defined in the math library preamble precisely so this script
# can extract IDs without parsing LaTeX table structure.
TESTID_RE = re.compile(r"\\testid\{([^}]+)\}")

# Where test identifiers may live (relative to the repository root). Kept in
# one place so the searched set is auditable and matches the docstring.
TEST_TREE_GLOBS = ("cpp/tests/**/*", "tests/**/*")
VERIFY_GLOB = "python/star_reacher/verify*.py"


def repo_root_default() -> Path:
    # The script lives in <root>/scripts/, so the default root is its parent.
    return Path(__file__).resolve().parent.parent


def collect_test_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in TEST_TREE_GLOBS:
        files.extend(p for p in root.glob(pattern) if p.is_file())
    files.extend(p for p in root.glob(VERIFY_GLOB) if p.is_file())
    return files


def test_id_present(test_files: list[Path], test_id: str) -> bool:
    for path in test_files:
        if test_id in path.name:
            return True
        # errors="ignore" so binary fixtures under tests/ cannot crash the
        # gate; a binary file simply cannot satisfy the match.
        try:
            if test_id in path.read_text(encoding="utf-8", errors="ignore"):
                return True
        except OSError:
            continue
    return False


def lint(root: Path) -> int:
    manifest_path = root / "docs" / "mathlib" / "chapter_manifest.toml"
    if not manifest_path.is_file():
        print(f"lint_chapters: manifest not found: {manifest_path}", file=sys.stderr)
        return 2

    try:
        with open(manifest_path, "rb") as fh:
            manifest = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        print(f"lint_chapters: manifest is not valid TOML: {exc}", file=sys.stderr)
        return 2

    entries = manifest.get("chapter", [])
    failures: list[str] = []
    checked = 0
    skipped = 0
    test_files: list[Path] | None = None  # collected lazily; most local runs skip everything

    for i, entry in enumerate(entries):
        missing_keys = [k for k in ("module", "chapter", "label") if k not in entry]
        if missing_keys:
            failures.append(
                f"manifest entry {i}: missing required key(s) {missing_keys}"
            )
            continue

        module = root / entry["module"]
        chapter = root / entry["chapter"]
        label = entry["label"]

        if not module.is_file():
            # Chapter-without-model is fine (docs may land first); the red
            # condition is model-without-chapter, checked below.
            skipped += 1
            continue

        checked += 1
        if not chapter.is_file():
            failures.append(
                f"{entry['module']}: module exists but chapter file "
                f"{entry['chapter']} is missing"
            )
            continue

        chapter_text = chapter.read_text(encoding="utf-8")
        if f"\\label{{{label}}}" not in chapter_text:
            failures.append(
                f"{entry['chapter']}: does not contain \\label{{{label}}}"
            )

        test_ids = TESTID_RE.findall(chapter_text)
        if not test_ids:
            # FR-29 requires a validation-evidence table of test IDs; an
            # implemented model whose chapter names no tests is uncovered.
            failures.append(
                f"{entry['chapter']}: names no \\testid{{...}} validation evidence"
            )
        else:
            if test_files is None:
                test_files = collect_test_files(root)
            for test_id in test_ids:
                if not test_id_present(test_files, test_id):
                    failures.append(
                        f"{entry['chapter']}: test ID '{test_id}' not found under "
                        f"cpp/tests/, tests/, or python/star_reacher/verify*.py"
                    )

    if failures:
        print("lint_chapters: FAIL", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print(
        f"lint_chapters: OK ({checked} module(s) checked, "
        f"{skipped} chapter(s) awaiting their module)"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=repo_root_default(),
        help="repository root to lint (default: the parent of scripts/)",
    )
    args = parser.parse_args(argv)
    return lint(args.root.resolve())


if __name__ == "__main__":
    sys.exit(main())
