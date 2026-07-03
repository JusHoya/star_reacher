"""The ``star`` command-line interface (D-4, FR-20, DX-1).

Five subcommands (run, verify, export, docs from Phase 1; data from Phase 2)
and no stubs: every command documented here works. Exit codes: 0 success, 2
validation errors (accumulated per DX-2), 1 runtime errors. ``python -m
star_reacher`` and the installed ``star`` console script both dispatch
through ``main``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from star_reacher import __version__
from star_reacher._corelink import CoreMissingError
from star_reacher.mission import MissionValidationError
from star_reacher.srlog import SrlogError

_EXIT_OK = 0
_EXIT_RUNTIME = 1
_EXIT_VALIDATION = 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="star",
        description=(
            "star_reacher: deterministic 6DOF space-mission simulator "
            f"(frontend {__version__})."
        ),
    )
    sub = parser.add_subparsers(
        dest="command", required=True, metavar="{run,verify,export,docs,data}"
    )

    p_run = sub.add_parser(
        "run",
        help="validate a mission TOML and propagate it, writing run.srlog",
        description=(
            "Validate, resolve, and hash a mission file, then propagate it with the "
            "compiled core, writing run.srlog, resolved_config.json, and meta.json. "
            "Validation errors are accumulated and reported together (exit 2)."
        ),
    )
    p_run.add_argument("mission", help="path to the mission TOML file")
    p_run.add_argument(
        "-o",
        "--outdir",
        default=None,
        help="output directory (default: out/<mission-name>/)",
    )
    p_run.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing run.srlog in the output directory",
    )

    p_verify = sub.add_parser(
        "verify",
        help="run the acceptance check suite (V001-V013)",
        description=(
            "Self-contained acceptance runner: one line per check, ending in "
            "'VERIFY: PASS (N/N)' or 'VERIFY: FAIL (k/N)' plus the failing check "
            "IDs; nonzero exit on any failure. Through Phase 2 the --quick tier "
            "runs the identical check set as the full tier."
        ),
    )
    p_verify.add_argument(
        "--quick",
        action="store_true",
        help="run the smoke tier (through Phase 2: identical to the full check set)",
    )

    p_export = sub.add_parser(
        "export",
        help="export an SRLOG file to CSV, one file per channel group",
        description=(
            "Write one CSV per channel group (truth.csv, events.csv) with a header "
            "row of channel names; vector channels expand to indexed columns and "
            "floats are written via repr, so every value round-trips bit-exactly."
        ),
    )
    p_export.add_argument(
        "--csv",
        action="store_true",
        help="select CSV output (required; the only Phase 1 export format)",
    )
    p_export.add_argument("srlog", help="path to the run.srlog file")
    p_export.add_argument(
        "-o",
        "--outdir",
        default=None,
        help="output directory (default: alongside the input file)",
    )

    p_docs = sub.add_parser(
        "docs",
        help="build the math library and report PDFs with latexmk",
        description=(
            "Run latexmk -pdf -halt-on-error -interaction=nonstopmode in "
            "docs/mathlib and docs/report with SOURCE_DATE_EPOCH pinned to the "
            "HEAD commit time for byte-reproducible PDFs."
        ),
    )
    docs_scope = p_docs.add_mutually_exclusive_group()
    docs_scope.add_argument(
        "--mathlib-only", action="store_true", help="build only docs/mathlib"
    )
    docs_scope.add_argument(
        "--report-only", action="store_true", help="build only docs/report"
    )

    p_data = sub.add_parser(
        "data",
        help="manage fetched datasets (Phase 2: 'fetch de440s')",
        description=(
            "Dataset management (D-8). 'star data fetch de440s' downloads the JPL "
            "DE440s SPK and the DE440 lunar principal-axis PCK with SHA-256 "
            "verification and repacks the 2020-2060 Chebyshev segments into "
            "data/de440s_2020_2060.sreph for the C++ core. Idempotent: with the "
            "files already present it verifies checksums instead of re-downloading."
        ),
    )
    data_sub = p_data.add_subparsers(dest="data_command", required=True, metavar="{fetch}")
    p_fetch = data_sub.add_parser(
        "fetch",
        help="download and repack a named dataset with checksum verification",
    )
    p_fetch.add_argument(
        "dataset",
        choices=["de440s"],
        help="dataset to fetch (Phase 2: de440s only)",
    )
    p_fetch.add_argument(
        "--data-dir",
        default="data",
        help="destination directory for kernels and the repack (default: data/)",
    )
    return parser


def _cmd_run(args: argparse.Namespace, argv: list[str]) -> int:
    from star_reacher.runner import RunnerError, run_mission

    try:
        result = run_mission(
            args.mission, args.outdir, force=args.force, command_line=["star", *argv]
        )
    except MissionValidationError as exc:
        for line in exc.errors:
            print(line, file=sys.stderr)
        print(
            f"star run: {len(exc.errors)} validation error(s) in {args.mission}; "
            f"all are listed above.",
            file=sys.stderr,
        )
        return _EXIT_VALIDATION
    except CoreMissingError as exc:
        print(f"star run: {exc}", file=sys.stderr)
        return _EXIT_RUNTIME
    except (RunnerError, OSError) as exc:
        print(f"star run: {exc}", file=sys.stderr)
        return _EXIT_RUNTIME

    print(f"mission: {result.mission_name}")
    print(f"resolved config sha256: {result.config_sha256}")
    # The summary dict is produced by the core; print it generically so the
    # CLI does not hard-code the core's key names.
    print("final-state summary:")
    for key in sorted(result.summary):
        print(f"  {key}: {result.summary[key]}")
    print(f"run.srlog sha256: {result.srlog_sha256}")
    print(f"outputs: {result.outdir} (run.srlog, resolved_config.json, meta.json)")
    return _EXIT_OK


def _cmd_verify(args: argparse.Namespace) -> int:
    from star_reacher.verify import run_checks

    return run_checks(quick=args.quick)


def _cmd_export(args: argparse.Namespace) -> int:
    if not args.csv:
        print(
            "star export: --csv is required; CSV is the only export format in "
            "Phase 1 (NPZ and Parquet land in Phase 5).",
            file=sys.stderr,
        )
        return _EXIT_VALIDATION
    from star_reacher.export import export_csv

    try:
        written = export_csv(Path(args.srlog), args.outdir)
    except FileNotFoundError:
        print(f"star export: {args.srlog}: no such file.", file=sys.stderr)
        return _EXIT_RUNTIME
    except SrlogError as exc:
        print(f"star export: {exc}", file=sys.stderr)
        return _EXIT_RUNTIME
    except OSError as exc:
        print(f"star export: {exc}", file=sys.stderr)
        return _EXIT_RUNTIME
    for path in written:
        print(f"wrote {path}")
    return _EXIT_OK


def _cmd_docs(args: argparse.Namespace) -> int:
    from star_reacher.docsbuild import build_docs

    return build_docs(mathlib_only=args.mathlib_only, report_only=args.report_only)


def _cmd_data(args: argparse.Namespace) -> int:
    from star_reacher.data_fetch import cli_fetch

    # The subparser is required and 'fetch' is its only member, so dispatch
    # is direct; new data verbs get their own branch when a phase earns them.
    return cli_fetch(args.dataset, args.data_dir)


def main(argv: list[str] | None = None) -> int:
    args_in = list(sys.argv[1:]) if argv is None else list(argv)
    parser = _build_parser()
    args = parser.parse_args(args_in)
    if args.command == "run":
        return _cmd_run(args, args_in)
    if args.command == "verify":
        return _cmd_verify(args)
    if args.command == "export":
        return _cmd_export(args)
    if args.command == "docs":
        return _cmd_docs(args)
    if args.command == "data":
        return _cmd_data(args)
    # Unreachable: the subparser is required and exhaustive.
    parser.error(f"unknown command {args.command!r}")
    return _EXIT_VALIDATION
