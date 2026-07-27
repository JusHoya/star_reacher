"""Implementation of ``star docs`` (FR-20, FR-29 build wrapper).

Wraps latexmk so both PDFs build with one command and with
``SOURCE_DATE_EPOCH`` pinned to the HEAD commit time, which is what makes the
PDFs byte-reproducible across machines and CI runs (FR-29). The LaTeX sources
themselves live in ``docs/mathlib`` and ``docs/report``.

The build is warning-gated, not merely error-gated: a latexmk run that exits 0
while its log still carries LaTeX, Package, Class, or Font warnings -- or the
end-of-run summaries that mean the document never converged -- fails here. The
Phase 8 exit criterion asks the PDFs to build warning-clean, and
``-halt-on-error`` alone only proves them error-clean. Overfull and Underfull
box reports are excluded from the gate and disclosed as a printed count
instead; see :func:`scan_latex_log`.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

# The Warning-tagged diagnostic shapes LaTeX actually emits, plus the
# end-of-run summary classes that mean the document is unconverged or
# incomplete. The summary phrases are matched unanchored because LaTeX prints
# them behind a "LaTeX Warning: " tag that the anchored alternatives already
# cover; a line matching both is still one hit, because the scan tests each
# line once.
_WARNING_RE = re.compile(
    r"^(?:LaTeX Warning:"
    r"|LaTeX Font Warning:"
    r"|Package \S+ Warning:"
    r"|Class \S+ Warning:)"
    r"|There were undefined references"
    r"|Label\(s\) may have changed"
    r"|Citation .*? undefined"
)

# Typographic badness. Deliberately not part of the gate: these are line- and
# page-breaking quality reports, not Warning-tagged diagnostics, and the
# committed documents carry hundreds of them. They are counted and printed so
# the exclusion stays a disclosed, measured number rather than a silent waiver.
_BADNESS_RE = re.compile(r"^(?:Overfull|Underfull) \\[hv]box\b")

# A log line wraps at ~79 columns; a package continues its diagnostic on the
# following lines behind a parenthesized copy of its own name. Skipping those
# keeps one wrapped diagnostic from being counted as several.
_CONTINUATION_RE = re.compile(r"^\([A-Za-z][A-Za-z0-9 ._-]*\)\s")


class LogScan(NamedTuple):
    """Warning hits as ``(line number, line text)``, and the badness count."""

    warnings: list[tuple[int, str]]
    badness: int


def scan_latex_log(text: str) -> LogScan:
    """Classify a latexmk log's diagnostics into gate-failing and disclosed.

    Only the opening line of a diagnostic is reported: that is enough to point
    a maintainer at the source, and counting continuation lines would inflate
    the total. Split out from :func:`build_docs` so the gate is testable
    without a TeX distribution.
    """
    warnings: list[tuple[int, str]] = []
    badness = 0
    for number, line in enumerate(text.splitlines(), 1):
        if _BADNESS_RE.match(line):
            badness += 1
        elif not _CONTINUATION_RE.match(line) and _WARNING_RE.search(line):
            warnings.append((number, line.rstrip()))
    return LogScan(warnings, badness)


def _head_commit_epoch() -> str | None:
    """HEAD commit time as a Unix-epoch string, or None when git cannot say."""
    try:
        proc = subprocess.run(
            ["git", "log", "-1", "--format=%ct"], capture_output=True, text=True
        )
    except OSError:
        return None
    out = proc.stdout.strip()
    return out if proc.returncode == 0 and out.isdigit() else None


def build_docs(mathlib_only: bool = False, report_only: bool = False) -> int:
    """Run latexmk in docs/mathlib and/or docs/report; return the exit code."""
    targets: list[Path] = []
    if not report_only:
        targets.append(Path("docs") / "mathlib")
    if not mathlib_only:
        targets.append(Path("docs") / "report")

    latexmk = shutil.which("latexmk")
    if latexmk is None:
        print(
            "star docs: latexmk not found on PATH. Install a TeX distribution that "
            "provides it: on Debian/Ubuntu 'apt install latexmk texlive-latex-extra "
            "biber', on Windows MiKTeX or TeX Live, on macOS MacTeX.",
            file=sys.stderr,
        )
        return 1

    env = os.environ.copy()
    epoch = _head_commit_epoch()
    if epoch is not None:
        env["SOURCE_DATE_EPOCH"] = epoch
    elif "SOURCE_DATE_EPOCH" in env:
        # git could not supply a commit time (e.g. a source tarball); a
        # caller-preset value still makes the build reproducible, so keep it.
        pass
    else:
        print(
            "star docs: warning: HEAD commit time unavailable (git missing or not a "
            "repository) and SOURCE_DATE_EPOCH is not set; the PDFs will build but "
            "will not be byte-reproducible.",
            file=sys.stderr,
        )

    for target in targets:
        if not target.is_dir():
            print(
                f"star docs: {target} not found; run from the repository root "
                f"(the LaTeX sources live in docs/mathlib and docs/report).",
                file=sys.stderr,
            )
            return 1
        proc = subprocess.run(
            [latexmk, "-pdf", "-halt-on-error", "-interaction=nonstopmode"],
            cwd=target,
            env=env,
        )
        if proc.returncode != 0:
            print(
                f"star docs: latexmk failed in {target} with exit code {proc.returncode}.",
                file=sys.stderr,
            )
            return proc.returncode
        # latexmk's default jobname is the main source's basename, and each
        # target directory is named after its own main source (docs/mathlib ->
        # mathlib.tex, docs/report -> report.tex).
        log_path = target / f"{target.name}.log"
        if not log_path.is_file():
            print(
                f"star docs: {log_path} is missing after a successful latexmk run, so "
                "the build cannot be certified warning-clean. Treating the absent log "
                "as a failure rather than a pass.",
                file=sys.stderr,
            )
            return 1
        scan = scan_latex_log(log_path.read_text(encoding="utf-8", errors="replace"))
        if scan.warnings:
            for number, line in scan.warnings:
                print(f"{log_path}:{number}: {line}", file=sys.stderr)
            print(
                f"star docs: {len(scan.warnings)} LaTeX/Package/Class warning(s) in "
                f"{log_path}. The documents must build warning-clean; fix the sources "
                "behind the lines above and rebuild.",
                file=sys.stderr,
            )
            return 1
        print(
            f"star docs: {target.name}.pdf built warning-clean; {scan.badness} "
            f"Overfull/Underfull box report(s) in {log_path} are excluded from the "
            "gate as typographic badness rather than Warning-tagged diagnostics."
        )
    return 0
