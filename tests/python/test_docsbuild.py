r"""The ``star docs`` warning gate (FR-29, Phase 8 exit criterion 2).

The criterion asks the two PDFs to build *warning-clean*, which
``latexmk -halt-on-error`` alone cannot show: a run that emits
``LaTeX Warning: There were undefined references.`` still exits 0. The scan
that closes that hole lives in ``star_reacher.docsbuild.scan_latex_log`` as a
pure text function precisely so it can be exercised here without a TeX
distribution on the machine.

Every fixture line below is copied verbatim from a real build log in this
repository (``docs/mathlib/mathlib.log`` and ``docs/report/report.log``),
including the ~79-column wraps, so the patterns are tested against the bytes
LaTeX actually writes rather than against an idealization of them.
"""

import subprocess
from pathlib import Path

from star_reacher import docsbuild
from star_reacher.docsbuild import scan_latex_log

# Benign preamble and progress chatter, none of it Warning-tagged.
CLEAN_LOG = r"""This is pdfTeX, Version 3.141592653-2.6-1.40.28 (MiKTeX 25.4)
entering extended mode
**./mathlib.tex
LaTeX2e <2025-11-01>
L3 programming layer <2026-01-19>
(chapters/integrators.tex
Package pdftex.def Info: figures/translunar_transfer.png  used on input line 40
0.
(pdftex.def)             Requested size: 469.75502pt x 216.0898pt.
Package rerunfilecheck Info: Checksums for `report.out':
[7] (report.aux)
Output written on mathlib.pdf (58 pages, 612345 bytes).
"""

FLOAT_WARNING = (
    "LaTeX Warning: Float too large for page by 157.85458pt on input line 314."
)

HYPERREF_WARNING = (
    "Package hyperref Warning: Token not allowed in a PDF string (Unicode):\n"
    "(hyperref)                removing `math shift' on input line 278."
)

# A single diagnostic wrapped across two lines: the trailing "264." is a
# continuation, not a second warning.
WRAPPED_REFERENCE_WARNING = (
    "LaTeX Warning: Reference `sec:vv-translunar' on page 4 undefined on input line \n"
    "264."
)

BADNESS_ONLY = r"""Overfull \hbox (14.35257pt too wide) in paragraph at lines 319--9
"""


def test_clean_log_passes():
    scan = scan_latex_log(CLEAN_LOG)
    assert scan.warnings == []
    assert scan.badness == 0


def test_latex_warning_fails():
    scan = scan_latex_log(CLEAN_LOG + FLOAT_WARNING + "\n")
    assert [line for _, line in scan.warnings] == [FLOAT_WARNING]


def test_package_warning_fails():
    scan = scan_latex_log(CLEAN_LOG + HYPERREF_WARNING + "\n")
    assert len(scan.warnings) == 1
    assert scan.warnings[0][1].startswith("Package hyperref Warning:")


def test_badness_only_passes_and_is_counted():
    # The disclosed exclusion: typographic badness is measured and reported,
    # never gated on.
    scan = scan_latex_log(BADNESS_ONLY)
    assert scan.warnings == []
    assert scan.badness == 1


def test_undefined_references_summary_fails():
    summary = "LaTeX Warning: There were undefined references.\n"
    scan = scan_latex_log(CLEAN_LOG + summary)
    assert len(scan.warnings) == 1
    assert "undefined references" in scan.warnings[0][1]


def test_wrapped_diagnostic_counts_once():
    scan = scan_latex_log(CLEAN_LOG + WRAPPED_REFERENCE_WARNING + "\n")
    assert len(scan.warnings) == 1
    assert scan.warnings[0][1].endswith("undefined on input line")


def test_reported_line_numbers_are_one_based():
    scan = scan_latex_log("first line\n" + FLOAT_WARNING + "\n")
    assert scan.warnings[0][0] == 2


def test_replacement_characters_do_not_hide_a_warning():
    # Logs are read with errors="replace"; a mojibake byte inside a diagnostic
    # must not let it slip past the gate.
    decoded = b"Package hyperref Warning: Token \xff not allowed".decode(
        "utf-8", errors="replace"
    )
    scan = scan_latex_log(decoded + "\n")
    assert len(scan.warnings) == 1


def fake_latexmk(monkeypatch, tmp_path, log_text):
    """Stand ``build_docs`` up over a fabricated docs/mathlib tree with a
    latexmk that always succeeds, so the gate is exercised without a TeX
    distribution. Writing the log only when ``log_text`` is not None also
    covers the missing-log path."""
    target = tmp_path / "docs" / "mathlib"
    target.mkdir(parents=True)
    if log_text is not None:
        (target / "mathlib.log").write_text(log_text, encoding="utf-8")

    def run(cmd, *args, **kwargs):
        # _head_commit_epoch still shells out to git; leave it to the real one
        # only in shape, not in effect, so the test never touches the repo.
        stdout = "1751500000\n" if cmd and cmd[0] == "git" else ""
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(docsbuild.shutil, "which", lambda name: "latexmk")
    monkeypatch.setattr(docsbuild.subprocess, "run", run)
    monkeypatch.chdir(tmp_path)


def test_build_docs_passes_a_clean_log(monkeypatch, tmp_path: Path, capsys):
    fake_latexmk(monkeypatch, tmp_path, CLEAN_LOG + BADNESS_ONLY)
    assert docsbuild.build_docs(mathlib_only=True) == 0
    out = capsys.readouterr().out
    assert "warning-clean" in out
    # The Overfull/Underfull exclusion is disclosed as a number, not waived.
    assert "1 Overfull/Underfull" in out


def test_build_docs_fails_on_a_warning(monkeypatch, tmp_path: Path, capsys):
    fake_latexmk(monkeypatch, tmp_path, CLEAN_LOG + FLOAT_WARNING + "\n")
    assert docsbuild.build_docs(mathlib_only=True) == 1
    err = capsys.readouterr().err
    assert "Float too large" in err
    # The message must name the log so a maintainer can open it.
    assert "mathlib.log" in err


def test_build_docs_fails_when_the_log_is_missing(monkeypatch, tmp_path: Path, capsys):
    fake_latexmk(monkeypatch, tmp_path, None)
    assert docsbuild.build_docs(mathlib_only=True) == 1
    assert "missing after a successful latexmk run" in capsys.readouterr().err
