"""Governance-gate tests: the FR-29 chapter lint and the D-18 citation check.

These tests run the two scripts as subprocesses against fabricated temporary
trees rather than against the live repository, for two reasons: (a) they must
not depend on cpp/ or python/star_reacher/ existing (workstreams land in
parallel), and (b) exit criterion 5's negative case -- "deleting a required
chapter while its module exists fails" -- must be proven on every CI run, not
just observed once manually. Fabricated trees make the negative case
constructible without mutating the real tree.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
LINT = SCRIPTS_DIR / "lint_chapters.py"
CHECK = SCRIPTS_DIR / "check_citation.py"

MANIFEST = """\
schema_version = 1

[[chapter]]
module = "cpp/src/models/twobody.cpp"
chapter = "docs/mathlib/chapters/twobody.tex"
label = "ch:twobody"
"""

CHAPTER = r"""\chapter{Two-Body Placeholder}
\label{ch:twobody}
Validation evidence: \testid{twobody_circular_orbit_analytic} and
\testid{V001}.
"""

CFF = """\
cff-version: 1.2.0
message: "If you use this software, please cite it using the metadata from this file."
type: software
title: star_reacher
version: 0.1.0
date-released: 2026-07-02
license: Apache-2.0
repository-code: "https://github.com/JusHoya/star_reacher"
authors:
  - family-names: Hoyer
    given-names: Melvin
    name-suffix: III
"""

README = r"""# fabricated readme

```bibtex
@software{hoyer_star_reacher,
  author  = {Hoyer, III, Melvin},
  title   = {star\_reacher},
  year    = {2026},
  url     = {https://github.com/JusHoya/star_reacher},
  version = {0.1.0}
}
```
"""


def run_script(script: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
    )


@pytest.fixture()
def fake_repo(tmp_path: Path) -> Path:
    """A minimal tree that satisfies the chapter lint: module + manifest +
    labeled chapter + test files containing every declared test ID."""
    (tmp_path / "cpp" / "src" / "models").mkdir(parents=True)
    (tmp_path / "cpp" / "src" / "models" / "twobody.cpp").write_text(
        "// fabricated module for the lint gate test\n", encoding="utf-8"
    )
    (tmp_path / "docs" / "mathlib" / "chapters").mkdir(parents=True)
    (tmp_path / "docs" / "mathlib" / "chapter_manifest.toml").write_text(
        MANIFEST, encoding="utf-8"
    )
    (tmp_path / "docs" / "mathlib" / "chapters" / "twobody.tex").write_text(
        CHAPTER, encoding="utf-8"
    )
    (tmp_path / "cpp" / "tests").mkdir(parents=True)
    (tmp_path / "cpp" / "tests" / "test_twobody.cpp").write_text(
        'TEST_CASE("twobody_circular_orbit_analytic") {}\n', encoding="utf-8"
    )
    (tmp_path / "python" / "star_reacher").mkdir(parents=True)
    (tmp_path / "python" / "star_reacher" / "verify.py").write_text(
        'CHECKS = ["V001"]\n', encoding="utf-8"
    )
    return tmp_path


class TestLintChapters:
    def test_complete_tree_passes(self, fake_repo: Path) -> None:
        result = run_script(LINT, "--root", str(fake_repo))
        assert result.returncode == 0, result.stderr

    def test_missing_chapter_fails(self, fake_repo: Path) -> None:
        # Exit criterion 5, negative case: module present, chapter deleted.
        (fake_repo / "docs" / "mathlib" / "chapters" / "twobody.tex").unlink()
        result = run_script(LINT, "--root", str(fake_repo))
        assert result.returncode != 0
        assert "twobody" in result.stderr

    def test_missing_label_fails(self, fake_repo: Path) -> None:
        chapter = fake_repo / "docs" / "mathlib" / "chapters" / "twobody.tex"
        chapter.write_text(
            CHAPTER.replace(r"\label{ch:twobody}", ""), encoding="utf-8"
        )
        result = run_script(LINT, "--root", str(fake_repo))
        assert result.returncode != 0
        assert "ch:twobody" in result.stderr

    def test_unresolvable_test_id_fails(self, fake_repo: Path) -> None:
        # Renaming the C++ test file's registration breaks the substring
        # match, so the chapter's declared evidence no longer exists.
        (fake_repo / "cpp" / "tests" / "test_twobody.cpp").write_text(
            'TEST_CASE("renamed_away") {}\n', encoding="utf-8"
        )
        result = run_script(LINT, "--root", str(fake_repo))
        assert result.returncode != 0
        assert "twobody_circular_orbit_analytic" in result.stderr

    def test_module_absent_is_skipped(self, fake_repo: Path) -> None:
        # Chapter-without-model is the allowed direction (docs land first).
        (fake_repo / "cpp" / "src" / "models" / "twobody.cpp").unlink()
        result = run_script(LINT, "--root", str(fake_repo))
        assert result.returncode == 0, result.stderr


class TestCheckCitation:
    def write_pair(self, tmp_path: Path, cff: str, readme: str) -> tuple[Path, Path]:
        cff_path = tmp_path / "CITATION.cff"
        readme_path = tmp_path / "README.md"
        cff_path.write_text(cff, encoding="utf-8")
        readme_path.write_text(readme, encoding="utf-8")
        return cff_path, readme_path

    def test_matching_pair_passes(self, tmp_path: Path) -> None:
        cff_path, readme_path = self.write_pair(tmp_path, CFF, README)
        result = run_script(
            CHECK, "--cff", str(cff_path), "--readme", str(readme_path)
        )
        assert result.returncode == 0, result.stderr

    def test_mutated_year_fails(self, tmp_path: Path) -> None:
        cff_path, readme_path = self.write_pair(
            tmp_path, CFF, README.replace("{2026}", "{2025}")
        )
        result = run_script(
            CHECK, "--cff", str(cff_path), "--readme", str(readme_path)
        )
        assert result.returncode != 0
        assert "year" in result.stderr

    def test_mutated_version_fails(self, tmp_path: Path) -> None:
        cff_path, readme_path = self.write_pair(
            tmp_path, CFF.replace("version: 0.1.0", "version: 0.2.0"), README
        )
        result = run_script(
            CHECK, "--cff", str(cff_path), "--readme", str(readme_path)
        )
        assert result.returncode != 0
        assert "version" in result.stderr
