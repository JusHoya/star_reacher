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
  estimation covariance (``f64[m(m+1)/2]``), logged at the ``nav.err``
  rate (record counts must match).
- ``nav.innov`` — channels ``y`` (``f64[m_max]``), ``S``
  (``f64[m_max(m_max+1)/2]``), ``sensor_id`` and ``m``.

**Error-state estimators (m = n - 1).** An estimator whose covariance lives
in a different parameterization than its state logs ``P`` at dimension m
while ``nav.err.e`` stays at the state dimension n. The reference
error-state EKF is the case in point: n = 16 with a quaternion attitude,
m = 15 with a three-component attitude error. The documented reduction for
such a quaternion-led estimator (``docs/formats/srlog_v1.md``, ``nav.err``)
is applied here: the leading four components of ``e`` are a
sign-canonicalized error quaternion and collapse to ``dtheta =
2 sgn(dq_w) dq_v``, the remaining n - 4 pass through unchanged. NEES is then
formed against the m-dimensional ``P`` the filter actually reported.

**Per-sensor NIS.** ``nav.innov`` records are zero-padded to ``m_max``, so a
three-dimensional star-tracker update padded into a six-wide record carries
a singular ``S``. Records are therefore grouped by ``sensor_id`` and each
group is trimmed to its own valid dimension ``m`` before gating — which is
also what ch:ekf specifies, since each sensor's NIS has its own chi-square
dimension. A single-sensor run reports one ``NIS`` gate; a multi-sensor run
reports one ``NIS[sensor k]`` gate per sensor.

**What gates, and what only reports.** The exit code is set by the
ensemble statistic of ch:ekf eq:ekf:ensemble and by nothing else. For each
statistic (NEES, and NIS per sensor) two gates are counted:

- the *headline* — the ensemble average over runs, averaged over epochs,
  against the two-sided 95 % chi-square interval at ``dof = R*dim``;
- the *coverage* — the number of epochs whose ensemble average lies inside
  that interval, against the binomial lower-tail threshold derived in
  :func:`star_reacher.consistency.inside_count_threshold`.

The per-run time-averaged numbers and the pooled all-epoch mean are printed
as labelled diagnostics and never reach the exit code, because the
chi-square bounds they would be judged against assume the per-epoch values
are independent while a filter's state error is serially correlated within
a run. On a provably consistent 100-run ensemble the per-run interval
admits only 7 of 100 runs, so gating on it would report FAIL on a correct
filter; :mod:`star_reacher.consistency` derives this in full and ch:ekf
sec:ekf:consistency is the normative statement.

Ensemble statistics are formed for any run count including one: a single
log is its own ensemble average and stays chi-square(dim) per epoch, so
one-log invocations remain gated rather than unconditionally green (they
are simply low-powered).

Report: the per-run diagnostics for every log, then the gates for each
statistic. The final line is ``CONSISTENCY: PASS (N/N gates)`` or
``CONSISTENCY: FAIL (k/N gates)``, counting gates only. Exit codes: 0 when
every gate passes; 1 on any gate failure, unreadable or missing input, or a
log without the required groups (the error names the missing group); 2 for
usage errors (argparse).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from star_reacher.consistency import (
    EnsembleGate,
    IntervalGate,
    ensemble_gate,
    matrix_order,
    nees,
    nis,
    pack_symmetric,
    time_average_gate,
    unpack_symmetric,
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
            "Compute NEES/NIS consistency statistics from the nav.err, "
            "nav.est, and nav.innov channel groups and print a PASS/FAIL "
            "report (FR-26). The estimator acceptance instrument is the "
            "ensemble statistic (ch:ekf eq:ekf:ensemble): per statistic, "
            "the ensemble average must lie inside two-sided 95 % "
            "chi-square bounds, and the number of epochs inside those "
            "bounds must meet a binomial lower-tail threshold. Per-run "
            "time-averaged and pooled numbers are printed as diagnostics "
            "and do not affect the exit code: their bounds assume epochs "
            "are independent, which is false within a run. Exits 0 only "
            "when every gate passes."
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


def _reduce_error(e: np.ndarray, m: int, source: str) -> tuple[np.ndarray, str | None]:
    """Reduce an n-dimensional logged error to the m dimensions P describes.

    Returns ``(reduced, problem)``. ``m == n`` passes straight through. The
    one sanctioned reduction is the quaternion-led error-state form
    ``m == n - 1``: the leading four components are an error quaternion and
    collapse to ``dtheta = 2 sgn(dq_w) dq_v`` (the small-angle extraction
    the estimator's own covariance is expressed in). Any other pairing is a
    genuine mismatch and is reported rather than guessed at.
    """
    n = e.shape[-1]
    if n == m:
        return e, None
    if n == m + 1 and n >= 4:
        w = e[..., 0]
        qv = e[..., 1:4]
        sign = np.where(w >= 0.0, 1.0, -1.0)[..., np.newaxis]
        dtheta = 2.0 * sign * qv
        return np.concatenate([dtheta, e[..., 4:]], axis=-1), None
    return e, (
        f"{source}: 'nav.err' has dimension {n} but 'nav.est' reports a "
        f"{m}-dimensional covariance; the only supported reduction is the "
        f"quaternion-led error state (n = m + 1), so these channels do not "
        f"describe one estimator"
    )


def _group_innovations(
    innov: np.ndarray, source: str
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], list[str]]:
    """Split zero-padded nav.innov records into per-sensor (y, S) arrays.

    Every record is padded to ``m_max``; entries beyond the record's own
    valid dimension ``m`` are zero, which would make S singular if gated as
    written. Each sensor's records are trimmed to its own m, which is also
    the dimension its chi-square bound is taken at.
    """
    problems: list[str] = []
    groups: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    names = innov.dtype.names or ()
    if "sensor_id" not in names or "m" not in names:
        # Pre-Phase-6 or synthetic logs without the tagging channels carry a
        # single full-width update stream; treat them as one group.
        return {0: (innov["y"], innov["S"])}, problems

    m_max = innov["y"].shape[-1]
    for sensor_id in sorted({int(s) for s in innov["sensor_id"]}):
        sel = innov["sensor_id"] == sensor_id
        dims = {int(v) for v in innov["m"][sel]}
        if len(dims) != 1:
            problems.append(
                f"{source}: sensor {sensor_id} logged innovations at more "
                f"than one dimension ({sorted(dims)}); each sensor's NIS is "
                f"chi-square at a single fixed dimension"
            )
            continue
        m = dims.pop()
        y = innov["y"][sel][:, :m]
        # Trim the packed covariance by unpacking at the padded width and
        # repacking the leading m-by-m block, so the zero padding never
        # reaches the Cholesky factorization.
        s_full = unpack_symmetric(innov["S"][sel])
        s_packed = pack_symmetric(s_full[:, :m, :m]) if m < m_max else innov["S"][sel]
        groups[sensor_id] = (y, s_packed)
    return groups, problems


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
        missing = [n for n in ("y", "S") if n not in (innov.dtype.names or ())]
        for name in missing:
            problems.append(
                f"{source}: group 'nav.innov' carries no channel named "
                f"{name!r} (FR-26 innovation channels); its channels are "
                f"{list(innov.dtype.names)}"
            )
        if not missing:
            groups, group_problems = _group_innovations(innov, source)
            problems.extend(group_problems)
            arrays["innov"] = groups

    if "e" in arrays and "P" in arrays and len(arrays["e"]) != len(arrays["P"]):
        problems.append(
            f"{source}: 'nav.err' has {len(arrays['e'])} records but "
            f"'nav.est' has {len(arrays['P'])}; FR-26 logs both per "
            f"estimation cycle, so the counts must match"
        )
    if "e" in arrays and "P" in arrays:
        # The covariance dimension is authoritative: it is what NEES is
        # normalized by, so the error is reduced to it rather than the
        # other way round.
        try:
            m = matrix_order(arrays["P"].shape[-1])
        except ValueError as exc:
            problems.append(f"{source}: {exc}")
        else:
            reduced, problem = _reduce_error(arrays["e"], m, source)
            if problem is not None:
                problems.append(problem)
            else:
                arrays["e"] = reduced
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


def _print_interval_diagnostic(label: str, gate: IntervalGate, extra: str) -> None:
    """Print a non-gating interval statistic.

    Deliberately does not print PASS/FAIL: those words are reserved for
    statistics that move the exit code, and a reader who sees FAIL beside a
    diagnostic will act on a number whose bounds do not apply.
    """
    where = "inside" if gate.passed else "outside"
    print(
        f"  {label}: mean {gate.mean:.4f} {where} [{gate.lower:.4f}, "
        f"{gate.upper:.4f}] (chi2 95 %, {extra}) [diagnostic, not gated]"
    )


def _print_ensemble_gate(label: str, gate: EnsembleGate, dim_label: str) -> list[bool]:
    headline_ok = _print_interval_gate(
        f"{label} ensemble headline", gate.headline, f"dof {gate.dof}, {dim_label}"
    )
    epochs = gate.epoch_mean.shape[0]
    # Name the direction here too: a reader who sees only counts has to
    # re-derive which way the covariance is wrong before they can act.
    lean = (
        "covariance too small / overconfident"
        if gate.fraction_above > gate.fraction_below
        else "covariance too large / underconfident"
    )
    spread = (
        f"{100.0 * gate.fraction_below:.1f} % of epochs below, "
        f"{100.0 * gate.fraction_above:.1f} % above: {lean}"
    )
    if not gate.coverage_gated:
        # Reported without PASS/FAIL: this statistic's epochs are serially
        # correlated, so the binomial threshold beside it is indicative.
        verdict = "inside" if gate.coverage_passed else f"below threshold ({spread})"
        tail = f"{verdict} [diagnostic, not gated: epochs are serially correlated]"
    elif gate.coverage_passed:
        tail = "PASS"
    else:
        tail = f"FAIL ({spread})"
    print(
        f"  {label} ensemble coverage: {gate.inside_count}/{epochs} epochs "
        f"({100.0 * gate.fraction_inside:.1f} %) in [{gate.lower:.4f}, "
        f"{gate.upper:.4f}] (need >= {gate.min_inside}/{epochs} = "
        f"{100.0 * gate.min_fraction:.1f} %, binomial lower tail at "
        f"{100.0 * gate.confidence:.1f} % confidence): {tail}"
    )
    _print_interval_diagnostic(
        f"{label} pooled", gate.pooled, f"dof {gate.pooled.dof}, {dim_label}"
    )
    if not gate.coverage_gated:
        return [headline_ok]
    return [headline_ok, gate.coverage_passed]


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

    # Every run must expose the same sensor set for the ensemble to stack.
    sensor_ids = sorted(per_file[0][1]["innov"].keys())
    for path, arrays in per_file:
        if sorted(arrays["innov"].keys()) != sensor_ids:
            print(
                f"star consistency: {path} logs innovations for sensors "
                f"{sorted(arrays['innov'].keys())} but the first run logs "
                f"{sensor_ids}; ensemble statistics need one common sensor "
                f"set",
                file=sys.stderr,
            )
            return _EXIT_RUNTIME
    # A single-sensor run keeps the bare "NIS" label, so the common case
    # reads the same as it always has.
    def nis_label(sensor_id: int) -> str:
        return "NIS" if len(sensor_ids) == 1 else f"NIS[sensor {sensor_id}]"

    results: list[bool] = []
    nees_runs: list[np.ndarray] = []
    nis_runs: dict[int, list[np.ndarray]] = {sid: [] for sid in sensor_ids}
    print(
        "gating: the ensemble headline and coverage criteria of ch:ekf "
        "eq:ekf:ensemble set the exit code; per-run time-averaged and "
        "pooled numbers are printed as diagnostics only (their chi-square "
        "bounds assume epochs are independent, which is false within a run)."
    )
    try:
        for path, arrays in per_file:
            eps_nees = nees(arrays["e"], arrays["P"])
            nees_runs.append(eps_nees)
            n = arrays["e"].shape[-1]
            print(f"run: {path}")
            _print_interval_diagnostic(
                "NEES time-averaged",
                time_average_gate(eps_nees, n),
                f"T={eps_nees.shape[0]}, n={n}",
            )
            for sid in sensor_ids:
                y, s_packed = arrays["innov"][sid]
                eps_nis = nis(y, s_packed)
                nis_runs[sid].append(eps_nis)
                m = y.shape[-1]
                _print_interval_diagnostic(
                    f"{nis_label(sid)} time-averaged",
                    time_average_gate(eps_nis, m),
                    f"T={eps_nis.shape[0]}, m={m}",
                )

        # The final flag is the structural epoch-independence declaration:
        # a consistent filter's innovations are white, so NIS epochs are
        # independent, while the state error NEES is formed from is a
        # correlated trajectory. It selects whether the binomial coverage
        # criterion gates or merely reports (see consistency.ensemble_gate).
        series: list[tuple[str, list[np.ndarray], int, bool]] = [
            ("NEES", nees_runs, per_file[0][1]["e"].shape[-1], False)
        ]
        for sid in sensor_ids:
            series.append(
                (
                    nis_label(sid),
                    nis_runs[sid],
                    per_file[0][1]["innov"][sid][0].shape[-1],
                    True,
                )
            )
        for label, runs_eps, dim, independent in series:
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
            plural = "run" if len(runs_eps) == 1 else "runs"
            print(f"ensemble: R={len(runs_eps)} {plural}, {label}")
            gate = ensemble_gate(
                np.stack(runs_eps), dim, epochs_independent=independent
            )
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
