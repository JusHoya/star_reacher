"""Implementation of ``star docs`` (FR-20, FR-29 build wrapper).

Wraps latexmk so both PDFs build with one command and with
``SOURCE_DATE_EPOCH`` pinned to the HEAD commit time, which is what makes the
PDFs byte-reproducible across machines and CI runs (FR-29). The LaTeX sources
themselves live in ``docs/mathlib`` and ``docs/report``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


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
    return 0
