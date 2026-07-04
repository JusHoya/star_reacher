#!/usr/bin/env python3
"""Dependency-minimality gate (FR-32; Phase 5 exit criterion 6; D-12).

Run this script WITH THE PYTHON OF THE ENVIRONMENT UNDER AUDIT - in CI, a
fresh venv holding nothing but the built star-reacher wheel and what pip
resolved for it. It enumerates that environment's installed distributions
via importlib.metadata and demands EXACT set equality, both directions,
against the committed expectation file
tests/golden/packaging/runtime_deps.toml:

* an installed distribution missing from the expectation is dependency
  creep (something new rode in and nobody re-derived the closure);
* an expected distribution that is not installed is a silently dropped
  runtime dependency (the wheel no longer pulls what D-12 says it needs).

Either direction is a red build. Stdlib only (tomllib requires Python
>= 3.11, which pyproject.toml already demands), so the fresh venv needs no
extra tooling to audit itself.

The venv bootstrap set is excluded: pip is present in any `python -m venv`
environment, and older interpreters also seed setuptools/wheel. None of the
three is a runtime dependency of star-reacher.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import re
import sys
import tomllib
from pathlib import Path

# Not runtime dependencies: `python -m venv` seeds pip (and, before CPython
# 3.12, setuptools and wheel) into every environment it creates.
BOOTSTRAP = frozenset({"pip", "setuptools", "wheel"})


def normalize(name: str) -> str:
    """PEP 503 name normalization, so 'python-dateutil' == 'python_dateutil'."""
    return re.sub(r"[-_.]+", "-", name).lower()


def installed_distributions() -> set[str]:
    """Normalized names of every distribution visible to this interpreter."""
    names = set()
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        if name:  # tolerate a broken .dist-info rather than crash the gate
            names.add(normalize(name))
    return names


def expected_distributions(expected_path: Path) -> set[str]:
    doc = tomllib.loads(expected_path.read_text(encoding="utf-8"))
    return {normalize(name) for name in doc["distributions"]}


def compare_sets(installed: set[str], expected: set[str]) -> tuple[bool, list[str]]:
    """Exact two-direction comparison; returns (ok, report lines)."""
    lines: list[str] = []
    unexpected = sorted(installed - expected)
    missing = sorted(expected - installed)
    for name in unexpected:
        lines.append(
            f"UNEXPECTED: {name} is installed but not in the committed "
            f"closure (dependency creep - if intended, re-derive "
            f"tests/golden/packaging/runtime_deps.toml per its header)"
        )
    for name in missing:
        lines.append(
            f"MISSING: {name} is in the committed closure but not installed "
            f"(a declared runtime dependency was dropped, or the expectation "
            f"is stale - re-derive it per its header)"
        )
    return not (unexpected or missing), lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="check_min_deps.py",
        description="FR-32 dependency-minimality gate: installed set == committed closure.",
    )
    parser.add_argument(
        "--expected",
        default=str(
            Path(__file__).resolve().parents[1]
            / "tests"
            / "golden"
            / "packaging"
            / "runtime_deps.toml"
        ),
        help="committed closure file (default: tests/golden/packaging/runtime_deps.toml)",
    )
    args = parser.parse_args(argv)

    expected = expected_distributions(Path(args.expected))
    installed = installed_distributions() - BOOTSTRAP
    ok, lines = compare_sets(installed, expected)

    print(f"interpreter: {sys.executable}")
    print(f"installed (bootstrap excluded): {len(installed)}; expected: {len(expected)}")
    for line in lines:
        print(line)
    if ok:
        print("DEP-MINIMALITY: PASS (installed set exactly equals the D-12 closure)")
        return 0
    print("DEP-MINIMALITY: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
