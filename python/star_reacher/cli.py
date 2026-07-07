"""The ``star`` command-line interface (D-4, FR-20, DX-1).

Eight subcommands (run, verify, export, docs from Phase 1; data from Phase 2;
view and plot from Phase 5; consistency from Phase 6) and no stubs: every
command documented here works. Exit codes: 0 success, 2 validation errors (accumulated per DX-2), 1
runtime errors. ``python -m star_reacher`` and the installed ``star`` console
script both dispatch through ``main``.
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
        dest="command",
        required=True,
        metavar="{run,verify,export,view,plot,consistency,docs,data}",
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
    p_run.add_argument(
        "--strict",
        action="store_true",
        help="promote validation warnings (the FR-15 vehicle plausibility tier) "
        "to errors",
    )

    p_verify = sub.add_parser(
        "verify",
        help="run the acceptance check suite (V001-V021)",
        description=(
            "Self-contained acceptance runner: one line per check, ending in "
            "'VERIFY: PASS (N/N)' or 'VERIFY: FAIL (k/N)' plus the failing check "
            "IDs; nonzero exit on any failure. Through Phase 3 the --quick tier "
            "runs the identical check set as the full tier."
        ),
    )
    p_verify.add_argument(
        "--quick",
        action="store_true",
        help="run the smoke tier (through Phase 3: identical to the full check set)",
    )

    p_export = sub.add_parser(
        "export",
        help="export an SRLOG file to CSV, NPZ, and/or Parquet",
        description=(
            "Export a log for external tooling (FR-17, D-13). --csv writes one "
            "CSV per channel group with repr-formatted floats (bit-exact round "
            "trip); --npz writes one pickle-free NPZ archive holding every "
            "group, the events, and the header JSON (bit-exact round trip); "
            "--parquet writes one Parquet file per group (requires the pyarrow "
            "optional extra). Vector channels expand to indexed columns "
            "(r_m_0, r_m_1, r_m_2) in every tabular format. Format flags "
            "combine freely; at least one is required."
        ),
    )
    p_export.add_argument(
        "--csv",
        action="store_true",
        help="write one CSV file per channel group",
    )
    p_export.add_argument(
        "--npz",
        action="store_true",
        help="write one NPZ archive containing every group, events, and header",
    )
    p_export.add_argument(
        "--parquet",
        action="store_true",
        help="write one Parquet file per channel group (needs the pyarrow extra)",
    )
    p_export.add_argument("srlog", help="path to the run.srlog file")
    p_export.add_argument(
        "-o",
        "--outdir",
        default=None,
        help="output directory (default: alongside the input file)",
    )

    p_view = sub.add_parser(
        "view",
        help="write a self-contained single-file HTML playback viewer (FR-19)",
        description=(
            "Generate the D-16 WebGL playback viewer for an SRLOG file: one "
            "self-contained HTML file embedding the vendored three.js runtime, "
            "a decimated view stream with a measured position-error bound, and "
            "the coastline overlay; the file makes zero network requests. "
            "Playback consumes only the log (no re-simulation); interpolation "
            "between keyframes is display-only and non-physical."
        ),
    )
    p_view.add_argument("srlog", help="path to the run.srlog file")
    p_view.add_argument(
        "-o",
        "--out",
        default=None,
        help="output HTML path (default: the input path with an .html suffix)",
    )

    p_plot = sub.add_parser(
        "plot",
        help="render the FR-18 quicklook PNG plot set from SRLOG files",
        description=(
            "Render matplotlib quicklook PNGs (headless-safe: the Agg backend "
            "is forced) from one or more SRLOG files: groundtrack with the "
            "embedded coastline, altitude/speed, osculating elements, attitude "
            "and body rates, mass/thrust/throttle, dynamic pressure and Mach, "
            "and per-source force/torque magnitudes, with event markers on "
            "every time axis. With several logs, shared channels overlay on "
            "one axes set per plot, labeled with each log's short resolved-"
            "config hash. Plots a log cannot feed (e.g. no env group) are "
            "skipped with a note. Names, feeding arrays, and conventions are "
            "documented in docs/formats/plots.md."
        ),
    )
    p_plot.add_argument("srlog", nargs="+", help="path(s) to run.srlog file(s)")
    p_plot.add_argument(
        "-o",
        "--outdir",
        default=None,
        help="output directory (default: <first input's directory>/plots)",
    )
    p_plot.add_argument(
        "--plots",
        default=None,
        help="comma-separated subset of plot names (default: all)",
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
        help="manage fetched datasets (de440s ephemeris, FR-5 gravity fields)",
        description=(
            "Dataset management (D-8, FR-5). 'star data fetch de440s' downloads "
            "the JPL DE440s SPK and the DE440 lunar principal-axis PCK with "
            "SHA-256 verification and repacks the 2020-2060 Chebyshev segments "
            "into data/de440s_2020_2060.sreph for the C++ core. 'star data fetch "
            "egm2008 | grgm1200a | mro120f' downloads the published Earth, Moon, "
            "or Mars spherical-harmonic coefficient file with SHA-256 "
            "verification and repacks it (truncated to the FR-5 degree: 70x70, "
            "120x120, 80x80) into a data/<dataset>_n<degree>.srgrav binary for "
            "the C++ core. All fetches are idempotent: with the files already "
            "present they verify checksums instead of re-downloading."
        ),
    )
    data_sub = p_data.add_subparsers(dest="data_command", required=True, metavar="{fetch}")
    p_fetch = data_sub.add_parser(
        "fetch",
        help="download and repack a named dataset with checksum verification",
    )
    p_fetch.add_argument(
        "dataset",
        choices=["de440s", "egm2008", "grgm1200a", "mro120f"],
        help="dataset to fetch",
    )
    p_fetch.add_argument(
        "--data-dir",
        default="data",
        help="destination directory for kernels and the repack (default: data/)",
    )

    # The consistency subparser lives with its handler so the FR-26 gate
    # tooling stays one self-contained module (the import is local for the
    # same reason the command handlers import lazily).
    from star_reacher.consistency_cli import add_consistency_parser

    add_consistency_parser(sub)
    return parser


def _cmd_run(args: argparse.Namespace, argv: list[str]) -> int:
    from star_reacher.runner import RunnerError, run_mission

    try:
        result = run_mission(
            args.mission,
            args.outdir,
            force=args.force,
            command_line=["star", *argv],
            strict=args.strict,
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
    if not (args.csv or args.npz or args.parquet):
        print(
            "star export: select at least one output format: --csv, --npz, "
            "--parquet (flags can be combined in one invocation).",
            file=sys.stderr,
        )
        return _EXIT_VALIDATION
    from star_reacher.export import export_csv, export_npz, export_parquet

    written: list[Path] = []
    try:
        if args.csv:
            written.extend(export_csv(Path(args.srlog), args.outdir))
        if args.npz:
            written.append(export_npz(Path(args.srlog), args.outdir))
        if args.parquet:
            written.extend(export_parquet(Path(args.srlog), args.outdir))
    except FileNotFoundError:
        print(f"star export: {args.srlog}: no such file.", file=sys.stderr)
        return _EXIT_RUNTIME
    except ImportError as exc:
        # A missing optional extra is an environment problem, not a usage
        # error: the exporter's message already names the extra to install.
        print(f"star export: {exc}", file=sys.stderr)
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


def _cmd_view(args: argparse.Namespace) -> int:
    from star_reacher.viewer import ViewerError, generate_view

    # The input check happens before generate_view so a FileNotFoundError
    # raised while writing the output cannot be misattributed to the input
    # log (the generic OSError handler below reports the real path).
    srlog_path = Path(args.srlog)
    if not srlog_path.is_file():
        print(f"star view: {args.srlog}: no such file.", file=sys.stderr)
        return _EXIT_RUNTIME
    try:
        result = generate_view(srlog_path, args.out)
    except (SrlogError, ViewerError, OSError) as exc:
        print(f"star view: {exc}", file=sys.stderr)
        return _EXIT_RUNTIME

    # Both the bound and the measured value are printed (and embedded in the
    # HTML) so the decimation claim is checkable without opening the file.
    print(
        f"view stream: kept {result.keyframes_kept} of {result.truth_records} "
        f"truth samples"
    )
    print(
        f"decimation bound: {result.bound_m:.6g} m "
        f"(= max(100 m, 0.01 % of the {result.position_span_m:.6g} m position span))"
    )
    print(f"decimation measured max error: {result.measured_max_error_m:.6g} m")
    print(f"wrote {result.out_path} ({result.html_bytes} bytes)")
    return _EXIT_OK


def _cmd_plot(args: argparse.Namespace) -> int:
    from star_reacher.plotting import PLOT_NAMES, render_plots
    from star_reacher.srlog import load

    plots = None
    if args.plots is not None:
        plots = [name.strip() for name in args.plots.split(",") if name.strip()]
        unknown = [name for name in plots if name not in PLOT_NAMES]
        if unknown:
            print(
                f"star plot: unknown plot name(s): {', '.join(unknown)}; "
                f"valid names: {', '.join(PLOT_NAMES)}",
                file=sys.stderr,
            )
            return _EXIT_VALIDATION
    runs = []
    for path in args.srlog:
        try:
            runs.append(load(path))
        except FileNotFoundError:
            print(f"star plot: {path}: no such file.", file=sys.stderr)
            return _EXIT_RUNTIME
        except SrlogError as exc:
            print(f"star plot: {exc}", file=sys.stderr)
            return _EXIT_RUNTIME
    outdir = (
        Path(args.outdir)
        if args.outdir is not None
        else Path(args.srlog[0]).parent / "plots"
    )
    try:
        report = render_plots(runs, outdir, plots=plots)
    except CoreMissingError as exc:
        # The groundtrack/altitude derivations use the core's exact frame
        # chain; a core-less environment is an install problem, not usage.
        print(f"star plot: {exc}", file=sys.stderr)
        return _EXIT_RUNTIME
    except OSError as exc:
        print(f"star plot: {exc}", file=sys.stderr)
        return _EXIT_RUNTIME
    for note in report.notes:
        print(f"note: {note}")
    for path in report.written:
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
    if args.command == "view":
        return _cmd_view(args)
    if args.command == "plot":
        return _cmd_plot(args)
    if args.command == "consistency":
        from star_reacher.consistency_cli import cmd_consistency

        return cmd_consistency(args)
    if args.command == "docs":
        return _cmd_docs(args)
    if args.command == "data":
        return _cmd_data(args)
    # Unreachable: the subparser is required and exhaustive.
    parser.error(f"unknown command {args.command!r}")
    return _EXIT_VALIDATION
