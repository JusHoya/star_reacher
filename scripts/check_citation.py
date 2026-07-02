#!/usr/bin/env python3
"""D-18 citation-consistency gate: CITATION.cff and the README BibTeX agree.

Compares the fields that identify the citation -- title, author (family,
given, suffix), year, URL, and version -- between ``CITATION.cff`` and the
``bibtex`` fenced code block in ``README.md``, and exits nonzero naming each
mismatch. CI runs this after ``cffconvert --validate``, so schema validity is
already established when this comparison runs.

Parsing notes (deliberate, documented tradeoffs):

* CITATION.cff is parsed with a tolerant line-oriented reader instead of a
  YAML library. The file's shape is fully under this repository's control
  (flat scalar fields plus a single-author list), so a YAML dependency would
  buy nothing; ``cffconvert`` performs the real schema validation in CI.
* The README BibTeX block is located as the first fenced code block tagged
  ``bibtex`` and parsed with a field regex. The title comparison unescapes
  ``\\_`` and strips protective braces, since BibTeX requires escaping that
  CFF does not.
* The BibTeX author is expected in the "von Last, Jr, First" form
  ``Family, Suffix, Given`` (e.g. ``Hoyer, III, Melvin``), which is how
  BibTeX represents a name with a suffix.

Exit status: 0 when every field matches; 1 with each mismatch named
otherwise; 2 when a file or required field cannot be found at all.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

BIBTEX_BLOCK_RE = re.compile(r"```bibtex\s*\n(.*?)```", re.DOTALL)
BIBTEX_FIELD_RE = re.compile(r"(\w+)\s*=\s*\{(.*?)\}\s*,?\s*$", re.MULTILINE)


def repo_root_default() -> Path:
    # The script lives in <root>/scripts/, so the default root is its parent.
    return Path(__file__).resolve().parent.parent


def parse_cff(path: Path) -> dict[str, str]:
    """Extract the compared fields from CITATION.cff (line-oriented)."""
    fields: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("#") or ":" not in line:
            continue
        # "- family-names: Hoyer" (author list item) parses the same as a
        # top-level "key: value" once the list dash is stripped; the file
        # contains exactly one author, so no disambiguation is needed.
        line = line.lstrip("- ")
        key, _, value = line.partition(":")
        value = value.strip().strip("'\"")
        if value:
            fields[key.strip()] = value
    return fields


def parse_readme_bibtex(path: Path) -> dict[str, str]:
    """Extract field/value pairs from the README's bibtex fenced block."""
    text = path.read_text(encoding="utf-8")
    match = BIBTEX_BLOCK_RE.search(text)
    if match is None:
        return {}
    return {
        key.lower(): value.strip()
        for key, value in BIBTEX_FIELD_RE.findall(match.group(1))
    }


def normalize_bibtex_text(value: str) -> str:
    # BibTeX needs "\_" and case-protecting braces; CFF stores the plain
    # string. Normalize the BibTeX side so the comparison is on content.
    return value.replace(r"\_", "_").replace("{", "").replace("}", "")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    root = repo_root_default()
    parser.add_argument("--cff", type=Path, default=root / "CITATION.cff")
    parser.add_argument("--readme", type=Path, default=root / "README.md")
    args = parser.parse_args(argv)

    for path in (args.cff, args.readme):
        if not path.is_file():
            print(f"check_citation: file not found: {path}", file=sys.stderr)
            return 2

    cff = parse_cff(args.cff)
    bib = parse_readme_bibtex(args.readme)
    if not bib:
        print(
            f"check_citation: no ```bibtex fenced block found in {args.readme}",
            file=sys.stderr,
        )
        return 2

    mismatches: list[str] = []

    def require(source: dict[str, str], key: str, where: str) -> str | None:
        value = source.get(key)
        if value is None:
            mismatches.append(f"{where}: required field '{key}' is missing")
        return value

    # Title: CFF stores the plain name; BibTeX must escape the underscore.
    cff_title = require(cff, "title", "CITATION.cff")
    bib_title = require(bib, "title", "README bibtex")
    if cff_title and bib_title and normalize_bibtex_text(bib_title) != cff_title:
        mismatches.append(
            f"title: CITATION.cff has '{cff_title}' but README bibtex "
            f"normalizes to '{normalize_bibtex_text(bib_title)}'"
        )

    # Author: BibTeX "Family, Suffix, Given" vs the three CFF name fields.
    bib_author = require(bib, "author", "README bibtex")
    if bib_author:
        parts = [p.strip() for p in normalize_bibtex_text(bib_author).split(",")]
        if len(parts) != 3:
            mismatches.append(
                f"author: README bibtex '{bib_author}' is not in the "
                f"'Family, Suffix, Given' form required for a suffixed name"
            )
        else:
            family, suffix, given = parts
            for cff_key, bib_value in (
                ("family-names", family),
                ("name-suffix", suffix),
                ("given-names", given),
            ):
                cff_value = require(cff, cff_key, "CITATION.cff")
                if cff_value and cff_value != bib_value:
                    mismatches.append(
                        f"author {cff_key}: CITATION.cff has '{cff_value}' "
                        f"but README bibtex has '{bib_value}'"
                    )

    # Year: compared against the year component of CFF date-released.
    cff_date = require(cff, "date-released", "CITATION.cff")
    bib_year = require(bib, "year", "README bibtex")
    if cff_date and bib_year:
        cff_year = cff_date.split("-")[0]
        if cff_year != bib_year:
            mismatches.append(
                f"year: CITATION.cff date-released year is '{cff_year}' "
                f"but README bibtex has '{bib_year}'"
            )

    # URL: the BibTeX url must be the CFF repository-code URL.
    cff_url = require(cff, "repository-code", "CITATION.cff")
    bib_url = require(bib, "url", "README bibtex")
    if cff_url and bib_url and cff_url != bib_url:
        mismatches.append(
            f"url: CITATION.cff repository-code is '{cff_url}' "
            f"but README bibtex has '{bib_url}'"
        )

    # Version: exact string match.
    cff_version = require(cff, "version", "CITATION.cff")
    bib_version = require(bib, "version", "README bibtex")
    if cff_version and bib_version and cff_version != bib_version:
        mismatches.append(
            f"version: CITATION.cff has '{cff_version}' "
            f"but README bibtex has '{bib_version}'"
        )

    if mismatches:
        print("check_citation: FAIL", file=sys.stderr)
        for mismatch in mismatches:
            print(f"  - {mismatch}", file=sys.stderr)
        return 1

    print("check_citation: OK (title, author, year, url, version consistent)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
