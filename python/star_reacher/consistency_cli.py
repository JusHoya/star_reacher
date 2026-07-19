"""``star consistency``: NEES/NIS chi-square gates over SRLOG runs (FR-26).

Thin CLI adapter over :mod:`star_reacher.consistency` (which documents the
gate mathematics and PASS criteria) and the :mod:`star_reacher.srlog`
loader. This module owns path collection, channel extraction, report
formatting, and exit codes; it computes no statistics of its own.

Usage::

    star consistency <run.srlog | directory> [more ...]

Directories are searched recursively for ``*.srlog`` files (sorted). Every
log must carry the FR-26 navigation channel groups:

- ``nav.err`` — the estimation error vector, channel ``e`` (``f64[n]``).
  FR-26 fixes the ``nav.est``/``nav.innov`` channel names but not the
  ``nav.err`` one, so when no channel named ``e`` exists the single
  non-``t_s`` vector channel of the group is accepted instead.
- ``nav.est`` — channel ``P``, the packed row-major upper-triangle
  estimation covariance (``f64[n(n+1)/2]``), logged at the ``nav.err``
  rate (record counts must match).
- ``nav.innov`` — channels ``y`` (``f64[m]``) and ``S``
  (``f64[m(m+1)/2]``), the innovation and its packed covariance.

Report: one time-averaged NEES and NIS gate per run; with two or more runs,
the ensemble per-epoch and pooled gates for both statistics (the ensemble
per-epoch gate is the acceptance instrument). The final line is
``CONSISTENCY: PASS (N/N gates)`` or ``CONSISTENCY: FAIL (k/N gates)``.
Exit codes: 0 when every gate passes; 1 on any gate failure, unreadable or
missing input, or a log without the required groups (the error names the
missing group); 2 for usage errors (argparse).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from star_reacher.consistency import (
    EnsembleGate,
    IntervalGate,
    ensemble_gate,
    nees,
    nis,
    time_average_gate,
)
from star_reacher.srlog import Run, SrlogError, load

_EXIT_OK = 0
_EXIT_RUNTIME = 1


def add_consistency_parser(sub) -> None:
    """Register the ``consistency`` subparser on the ``star`` CLI."""
    p = sub.add_parser(
        "consistency",
        help="compute NEES/NIS chi-square consistency gates from SRLOG runs",
        description=(
            "Compute per-run time-averaged and (for two or more runs) "
            "ensemble NEES/NIS statistics from the nav.err, nav.est, and "
            "nav.innov channel groups, check each against two-sided 95 % "
            "chi-square bounds, and print a PASS/FAIL report per gate "
            "(FR-26). The ensemble per-epoch gate is the estimator "
            "acceptance instrument: it passes when the ensemble average "
            "lies inside the bounds for at least 95 % of epochs. Exits 0 "
            "only when every gate passes."
        ),
    )
    p.add_argument(
        "paths",
        nargs="+",
        metavar="srlog_or_dir",
        help=".srlog file(s) and/or directories searched recursively for *.srlog",
    )


def _collect_paths(raw_paths: list[str]) -> list[Path]:
    """Resolve CLI arguments to a sorted list of .srlog files.

    Raises ``FileNotFoundError`` with an actionable message for a missing
    path or a directory containing no logs.
    """
    files: list[Path] = []
    for raw in raw_paths:
        path = Path(raw)
        if path.is_dir():
            found = sorted(path.rglob("*.srlog"))
            if not found:
                raise FileNotFoundError(
                    f"{raw}: directory contains no *.srlog files (searched "
                    f"recursively)"
                )
            files.extend(found)
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"{raw}: no such file or directory.")
    return files


def _vector_channels(arr: np.ndarray) -> list[str]:
    # A vector channel is a fixed-size f64 subarray field; t_s and other
    # scalars have empty subarray shapes.
    return [name for name in arr.dtype.names if arr.dtype[name].shape]


def _extract_arrays(run: Run, source: str) -> tuple[dict[str, np.ndarray], list[str]]:
    """Pull the FR-26 arrays out of a loaded run.

    Returns ``(arrays, problems)`` where arrays holds ``e``, ``P``, ``y``,
    ``S`` on success and problems lists every missing group/channel with an
    actionable message, so one pass reports all defects of a log together.
    """
    problems: list[str] = []
    arrays: dict[str, np.ndarray] = {}

    err = run.groups.get("nav.err")
    if err is None:
        problems.append(
            f"{source}: missing channel group 'nav.err' (the estimation "
            f"error vector); this log predates the Phase 6 navigation "
            f"channels or was produced without an estimator in the loop"
        )
    else:
        vectors = _vector_channels(err)
        if "e" in vectors:
            arrays["e"] = err["e"]
        elif len(vectors) == 1:
            arrays["e"] = err[vectors[0]]
        else:
            problems.append(
                f"{source}: group 'nav.err' carries no channel named 'e' and "
                f"its vector channels are ambiguous ({vectors}); one error-"
                f"vector channel is required"
            )

    est = run.groups.get("nav.est")
    if est is None:
        problems.append(
            f"{source}: missing channel group 'nav.est' (needs channel 'P', "
            f"the packed upper-triangle estimation covariance per FR-26)"
        )
    elif "P" not in (est.dtype.names or ()):
        problems.append(
            f"{source}: group 'nav.est' carries no channel named 'P' "
            f"(the FR-26 packed upper-triangle estimation covariance); its "
            f"channels are {list(est.dtype.names)}"
        )
    else:
        arrays["P"] = est["P"]

    innov = run.groups.get("nav.innov")
    if innov is None:
        problems.append(
            f"{source}: missing channel group 'nav.innov' (needs channels "
            f"'y' and 'S', the innovation and its packed covariance per "
            f"FR-26)"
        )
    else:
        for name in ("y", "S"):
            if name not in (innov.dtype.names or ()):
                problems.append(
                    f"{source}: group 'nav.innov' carries no channel named "
                    f"{name!r} (FR-26 innovation channels); its channels are "
                    f"{list(innov.dtype.names)}"
                )
            else:
                arrays[name] = innov[name]

    if "e" in arrays and "P" in arrays and len(arrays["e"]) != len(arrays["P"]):
        problems.append(
            f"{source}: 'nav.err' has {len(arrays['e'])} records but "
            f"'nav.est' has {len(arrays['P'])}; FR-26 logs both per "
            f"estimation cycle, so the counts must match"
        )
    return arrays, problems


def _fail_direction(gate: IntervalGate) -> str:
    if gate.mean > gate.upper:
        return " (mean above the upper bound: covariance too small / overconfident)"
    return " (mean below the lower bound: covariance too large / underconfident)"


def _print_interval_gate(label: str, gate: IntervalGate, extra: str) -> bool:
    verdict = "PASS" if gate.passed else "FAIL" + _fail_direction(gate)
    print(
        f"  {label}: mean {gate.mean:.4f} in [{gate.lower:.4f}, "
        f"{gate.upper:.4f}] (chi2 95 %, {extra}): {verdict}"
    )
    return gate.passed


def _print_ensemble_gate(label: str, gate: EnsembleGate, dim_label: str) -> list[bool]:
    if gate.passed:
        verdict = "PASS"
    else:
        verdict = (
            f"FAIL ({100.0 * gate.fraction_below:.1f} % of epochs below, "
            f"{100.0 * gate.fraction_above:.1f} % above)"
        )
    print(
        f"  {label} ensemble: {100.0 * gate.fraction_inside:.1f} % of "
        f"{gate.epoch_mean.shape[0]} epochs in [{gate.lower:.4f}, "
        f"{gate.upper:.4f}] (chi2 95 %, dof {gate.dof}, need >= "
        f"{100.0 * gate.min_fraction:.1f} %): {verdict}"
    )
    pooled_ok = _print_interval_gate(
        f"{label} pooled", gate.pooled, f"dof {gate.pooled.dof}, {dim_label}"
    )
    return [gate.passed, pooled_ok]


def cmd_consistency(args) -> int:
    try:
        paths = _collect_paths(args.paths)
    except FileNotFoundError as exc:
        print(f"star consistency: {exc}", file=sys.stderr)
        return _EXIT_RUNTIME

    per_file: list[tuple[Path, dict[str, np.ndarray]]] = []
    problems: list[str] = []
    for path in paths:
        try:
            run = load(path)
        except (SrlogError, OSError) as exc:
            print(f"star consistency: {exc}", file=sys.stderr)
            return _EXIT_RUNTIME
        arrays, file_problems = _extract_arrays(run, str(path))
        problems.extend(file_problems)
        per_file.append((path, arrays))
    if problems:
        for line in problems:
            print(f"star consistency: {line}", file=sys.stderr)
        return _EXIT_RUNTIME

    results: list[bool] = []
    nees_runs: list[np.ndarray] = []
    nis_runs: list[np.ndarray] = []
    try:
        for path, arrays in per_file:
            eps_nees = nees(arrays["e"], arrays["P"])
            eps_nis = nis(arrays["y"], arrays["S"])
            nees_runs.append(eps_nees)
            nis_runs.append(eps_nis)
            n = arrays["e"].shape[-1]
            m = arrays["y"].shape[-1]
            print(f"run: {path}")
            results.append(
                _print_interval_gate(
                    "NEES time-averaged",
                    time_average_gate(eps_nees, n),
                    f"T={eps_nees.shape[0]}, n={n}",
                )
            )
            results.append(
                _print_interval_gate(
                    "NIS time-averaged",
                    time_average_gate(eps_nis, m),
                    f"T={eps_nis.shape[0]}, m={m}",
                )
            )

        if len(per_file) >= 2:
            for label, runs_eps, dim in (
                ("NEES", nees_runs, per_file[0][1]["e"].shape[-1]),
                ("NIS", nis_runs, per_file[0][1]["y"].shape[-1]),
            ):
                epoch_counts = {eps.shape[0] for eps in runs_eps}
                if len(epoch_counts) != 1:
                    print(
                        f"star consistency: the {label} epoch counts differ "
                        f"across runs ({sorted(epoch_counts)}); ensemble "
                        f"statistics need one common epoch grid (same "
                        f"mission and rates, different seeds)",
                        file=sys.stderr,
                    )
                    return _EXIT_RUNTIME
                print(f"ensemble: R={len(runs_eps)} runs, {label}")
                gate = ensemble_gate(np.stack(runs_eps), dim)
                results.extend(_print_ensemble_gate(label, gate, f"dim {dim}"))
    except ValueError as exc:
        # Dimension mismatches and non-positive-definite covariances arrive
        # here from the engine with their own actionable wording.
        print(f"star consistency: {exc}", file=sys.stderr)
        return _EXIT_RUNTIME

    total = len(results)
    passed = sum(results)
    if passed == total:
        print(f"CONSISTENCY: PASS ({passed}/{total} gates)")
        return _EXIT_OK
    print(f"CONSISTENCY: FAIL ({passed}/{total} gates)")
    return _EXIT_RUNTIME
