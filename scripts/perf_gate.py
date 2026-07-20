#!/usr/bin/env python3
"""Performance gate harness (FR-32, FR-22 layer 7; Phase 5 exit criteria 4-5).

Three metrics, each with an absolute EC-4 threshold, measured sequentially in
a single process (no parallelism anywhere in this harness: D-10's
single-thread discipline extends to the measurement instrument, and parallel
measurement would contaminate the very numbers being gated):

``mission_a_wall_s``
    Wall time of one cold ``star run`` of the Mission A cislunar benchmark in
    a fresh subprocess. EC-4 words the target as end-to-end mission wall time,
    so the measurement deliberately includes interpreter startup, validation,
    hashing, and sidecar writes, not just the propagation loop.
    Gate: < 60 s.

``ascent_rt_factor``
    Simulated seconds per wall second for the scripted open-loop ascent
    (no GNC in the loop; the C++ GNC variant re-gates in Phase 6). The wall
    time is the compiled core's propagation call only - the time loop plus
    its SRLOG writes - because FR-32 states the target for the propagation
    ("RK4 ... >= 100x real time"), and Python-side validation/hashing is a
    duration-independent constant that would turn a real-time factor into an
    interpreter-startup benchmark. The simulated span is read back from the
    produced log's final truth record (the ascent terminates on its
    orbit-insertion event well before duration_s, and crediting unsimulated
    seconds would overstate the factor). Gate: >= 100x.

``ascent_gnc_rt_factor``
    The Phase 6 exit criterion 10 re-gate: the same ratio as
    ``ascent_rt_factor``, measured the same way, but for
    ``missions/ascent_leo_gnc.toml`` - the closed-loop ascent that flies the
    built-in C++ GNC chain (ideal IMU, dead-reckoning navigation,
    pitch_program guidance, pd_attitude control) instead of an open-loop
    pitch-program sequence action. FR-32 states the ascent target once; this
    metric is what holds it "with the built-in C++ GNC stack in the loop".
    The two ascents are deliberately measured as separate metrics rather than
    one being replaced, because their ratio is the cost of the GNC chain and
    the nightly EC-5 rolling gate then tracks each independently.
    Gate: >= 100x.

``srlog_write_mbps``
    Sustained SRLOG write throughput, measured as log bytes divided by the
    wall time of the compiled core's propagation call on a deliberately
    write-dominated run: a two-body point-mass RK4 mission (the cheapest
    force model in the core) logging every integrator step. The SRLOG writer
    is not exposed through the Python bindings (bindings/module.cpp binds no
    writer type), and building a dedicated writer micro-bench would mean
    expanding the public API surface for a measurement, so this bytes/wall
    bound is the honest instrument: the wall time still contains the RK4
    arithmetic, so the reported MB/s UNDERSTATES the pure writer throughput,
    and the gate is conservative. Gate: >= 50 MB/s (decimal: 1 MB = 1e6 B).

Subcommands:

``measure``  - run the metrics, print a human summary, optionally write a
               machine-readable JSON (``--json``); exit 1 if any absolute
               gate fails.
``median``   - reduce N measurement JSONs to a per-metric median file (the
               nightly workflow feeds it the last <= 10 successful runs).
``compare``  - the EC-5 rule: fail (exit 1) when the current measurement
               regresses more than ``--tolerance`` (default 0.10) against
               the median file, direction-aware (wall time regresses UP;
               real-time factor and MB/s regress DOWN).

Dependencies: stdlib plus the installed ``star_reacher`` package only, and
the package is imported lazily so ``median``/``compare`` run anywhere.

Runner honesty (PRD section 9): EC-4's thresholds are written for a
Raspberry Pi 5 single core. No self-hosted Pi runner is attached to this
repository, so CI gates these thresholds on pinned GitHub-hosted runner
classes with ubuntu-24.04-arm as the documented Pi 5 PROXY leg; a number
measured by this script is a number for the machine it ran on (recorded in
the JSON's ``runner`` block) and is never a Pi 5 number unless it ran on a
Pi 5. The literal Pi 5 measurement is the manual pre-release checklist
docs/perf/pi5_checklist.md.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMA_MEASUREMENT = "star-reacher-perf-measurement/1"
SCHEMA_MEDIAN = "star-reacher-perf-median/1"

# One decimal megabyte. FR-32's "MB/s" and the wheel budget both read as SI
# units; the decimal reading also makes the throughput gate the stricter one.
MB = 1_000_000

# The EC-4 absolute gates. "sense" is the regression direction key used by
# both the absolute gate and the EC-5 compare rule:
#   lower_is_better  -> pass while value <  threshold; regresses UP
#   higher_is_better -> pass while value >= threshold; regresses DOWN
GATES = {
    "mission_a_wall_s": {
        "threshold": 60.0,
        "sense": "lower_is_better",
        "units": "s",
        "gate_text": "< 60 s",
    },
    "ascent_rt_factor": {
        "threshold": 100.0,
        "sense": "higher_is_better",
        "units": "x realtime",
        "gate_text": ">= 100x",
    },
    "ascent_gnc_rt_factor": {
        "threshold": 100.0,
        "sense": "higher_is_better",
        "units": "x realtime",
        "gate_text": ">= 100x",
    },
    "srlog_write_mbps": {
        "threshold": 50.0,
        "sense": "higher_is_better",
        "units": "MB/s",
        "gate_text": ">= 50 MB/s",
    },
}

DEFAULT_METRICS = (
    "mission_a_wall_s",
    "ascent_rt_factor",
    "ascent_gnc_rt_factor",
    "srlog_write_mbps",
)


def gate_passes(metric: str, value: float) -> bool:
    """Absolute EC-4 gate verdict for one metric value."""
    gate = GATES[metric]
    if gate["sense"] == "lower_is_better":
        return value < gate["threshold"]
    return value >= gate["threshold"]


def compare_metric(
    metric: str, current: float, median: float, tolerance: float
) -> tuple[bool, float]:
    """The EC-5 rule for one metric: (ok, signed regression fraction).

    The regression fraction is positive when the metric moved in its bad
    direction. Strictly-greater-than semantics: a change of exactly
    ``tolerance`` passes, matching EC-5's ">10 % ... fails" wording.
    """
    if GATES[metric]["sense"] == "lower_is_better":
        regression = (current - median) / median
    else:
        regression = (median - current) / median
    return regression <= tolerance, regression


def compare_documents(current: dict, median: dict, tolerance: float) -> tuple[bool, list[str]]:
    """Apply the EC-5 rule across a measurement and a median document.

    A metric present in the median history but absent from the current
    measurement fails (a silently dropped metric would hide regressions
    forever); a metric with no history is reported and skipped - the rolling
    gate for it becomes live once history exists.
    """
    lines: list[str] = []
    ok = True
    cur_metrics = current.get("metrics", {})
    med_metrics = median.get("metrics", {})
    for name in sorted(set(cur_metrics) | set(med_metrics)):
        if name not in cur_metrics:
            lines.append(f"{name}: FAIL - in median history but missing from the current measurement")
            ok = False
            continue
        if name not in med_metrics:
            lines.append(f"{name}: no history, skipped (rolling gate live from the next run)")
            continue
        cur = float(cur_metrics[name]["value"])
        med = float(med_metrics[name]["median"])
        good, regression = compare_metric(name, cur, med, tolerance)
        verdict = "ok" if good else "FAIL"
        lines.append(
            f"{name}: current {cur:.6g} vs median {med:.6g} "
            f"-> regression {regression * 100:+.2f} % "
            f"(tolerance {tolerance * 100:.0f} %) {verdict}"
        )
        ok = ok and good
    return ok, lines


def median_of_documents(documents: list[dict]) -> dict:
    """Per-metric median over measurement documents (metrics may be sparse)."""
    values: dict[str, list[float]] = {}
    for doc in documents:
        for name, entry in doc.get("metrics", {}).items():
            values.setdefault(name, []).append(float(entry["value"]))
    return {
        "schema": SCHEMA_MEDIAN,
        "source_documents": len(documents),
        "metrics": {
            name: {"median": statistics.median(vals), "n": len(vals)}
            for name, vals in sorted(values.items())
        },
    }


# ---------------------------------------------------------------------------
# measure
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    """Best-available source identity: CI env var, then git, then the core."""
    sha = os.environ.get("GITHUB_SHA")
    if sha:
        return sha
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except OSError:
        pass
    return "unknown"


def _runner_identity() -> dict:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "system": platform.system(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }


class _TimedCore:
    """Proxy over the compiled core that times its propagation entry points.

    The runner calls exactly one of run/run_env/run_vehicle per mission; the
    proxy records that call's wall time (time.perf_counter) and delegates
    everything else untouched. This is the narrowest honest instrument for
    "propagation wall time": it brackets the core time loop including its
    buffered SRLOG writes, and excludes Python-side validation, hashing, and
    sidecar writes. If the runner's core-call structure ever changes, the
    sink stays empty and measurement fails loudly rather than reporting a
    stale number.
    """

    def __init__(self, core, sink: list):
        self._core = core
        self._sink = sink

    def __getattr__(self, name):
        return getattr(self._core, name)

    def _timed(self, fn, cfg, out_path):
        t0 = time.perf_counter()
        result = fn(cfg, out_path)
        self._sink.append(time.perf_counter() - t0)
        return result

    def run(self, cfg, out_path):
        return self._timed(self._core.run, cfg, out_path)

    def run_env(self, cfg, out_path):
        return self._timed(self._core.run_env, cfg, out_path)

    def run_vehicle(self, cfg, out_path):
        return self._timed(self._core.run_vehicle, cfg, out_path)


def _run_in_process_timed(mission_path: Path, outdir: Path) -> tuple[float, Path]:
    """run_mission with the core call timed; returns (core wall s, srlog path)."""
    import star_reacher.runner as runner_mod

    sink: list[float] = []
    real_import_core = runner_mod.import_core
    # Patch the runner's import_core name so run_mission receives the timing
    # proxy; restored unconditionally so later measurements see the real core.
    runner_mod.import_core = lambda: _TimedCore(real_import_core(), sink)
    try:
        result = runner_mod.run_mission(
            str(mission_path),
            str(outdir),
            force=True,
            command_line=["scripts/perf_gate.py", "measure"],
        )
    finally:
        runner_mod.import_core = real_import_core
    if len(sink) != 1:
        raise RuntimeError(
            f"expected exactly one core propagation call, timed {len(sink)}; "
            "the runner's core-call structure changed - update perf_gate.py"
        )
    return sink[0], result.srlog_path


def measure_mission_a_wall_s(mission: Path, workdir: Path) -> tuple[float, dict]:
    """Cold end-to-end `star run` wall time in a fresh subprocess."""
    outdir = workdir / "mission_a"
    cmd = [
        sys.executable,
        "-m",
        "star_reacher",
        "run",
        str(mission),
        "-o",
        str(outdir),
        "--force",
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    wall_s = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"star run {mission} exited {proc.returncode}:\n"
            f"{proc.stdout}\n{proc.stderr}"
        )
    srlog = outdir / "run.srlog"
    detail = {
        "mission": str(mission),
        "measurement": "cold single-process `star run` subprocess wall time",
        "srlog_bytes": srlog.stat().st_size if srlog.exists() else None,
    }
    return wall_s, detail


def measure_ascent_rt_factor(
    mission: Path, workdir: Path, subdir: str = "ascent"
) -> tuple[float, dict]:
    """Simulated span / core propagation wall for an ascent mission.

    Shared by the open-loop (``ascent_rt_factor``) and closed-loop GNC
    (``ascent_gnc_rt_factor``) metrics: the measurement contract is identical
    and only the mission differs, so ``subdir`` keeps their output trees
    apart within one measurement run.
    """
    import star_reacher

    core_wall_s, srlog_path = _run_in_process_timed(mission, workdir / subdir)
    run = star_reacher.load(srlog_path)
    # The mission terminates on its insertion event before duration_s; only
    # the span the core actually simulated is credited.
    simulated_s = float(run.time_s("truth")[-1])
    detail = {
        "mission": str(mission),
        "measurement": "final truth t_s / wall time of the core propagation call",
        "simulated_s": simulated_s,
        "core_wall_s": core_wall_s,
    }
    return simulated_s / core_wall_s, detail


# The write-dominated benchmark mission for srlog_write_mbps, written to a
# temp directory at measurement time (never committed: missions/ is the
# curated example set). Values mirror missions/twobody_leo.toml - the Phase 1
# byte-frozen two-body path, the cheapest compute per logged byte the core
# offers - with 1/(dt_s * truth_rate_hz) = 1 so every integrator step emits a
# truth record. duration_s is derived from --srlog-records at 10 records per
# simulated second.
_SRLOG_BENCH_TEMPLATE = """\
# Write-rate benchmark mission generated by scripts/perf_gate.py (not a
# curated example; see the harness docstring for the measurement contract).
schema_version = 1

[mission]
name = "srlog-write-bench"
epoch_utc = "2026-01-01T00:00:00Z"
duration_s = {duration_s}

[run]
seed = 20260101

[integrator]
type = "rk4"
dt_s = 0.1

[spacecraft]
mass_kg = 150.0

[initial_state.cartesian]
r_m = [6778137.0, 0.0, 0.0]
v_mps = [0.0, 7668.6, 0.0]
frame = "GCRF"

[environment]
central_body = "earth"

[logging]
truth_rate_hz = 10
"""


def measure_srlog_write_mbps(records: int, workdir: Path) -> tuple[float, dict]:
    """Log bytes / core propagation wall on a write-dominated two-body run."""
    duration_s = records / 10.0
    mission = workdir / "srlog_write_bench.toml"
    mission.write_text(
        _SRLOG_BENCH_TEMPLATE.format(duration_s=f"{duration_s:.1f}"),
        encoding="utf-8",
    )
    core_wall_s, srlog_path = _run_in_process_timed(mission, workdir / "srlog_bench")
    size_bytes = srlog_path.stat().st_size
    mbps = (size_bytes / MB) / core_wall_s
    detail = {
        "measurement": (
            "srlog bytes / wall time of the core propagation call on a "
            "write-dominated two-body run; includes RK4 compute, so this "
            "understates pure writer throughput (conservative bound)"
        ),
        "requested_records": records,
        "srlog_bytes": size_bytes,
        "core_wall_s": core_wall_s,
    }
    return mbps, detail


def cmd_measure(args: argparse.Namespace) -> int:
    # Path arguments resolve against the caller's cwd; the harness itself then
    # runs from the repo root so mission-relative assets (vehicle files,
    # committed ephemeris/gravity excerpts) resolve exactly as they do for CI's
    # `star run` steps.
    json_out = Path(args.json).resolve() if args.json else None
    mission_a = Path(args.mission_a).resolve()
    ascent = Path(args.ascent).resolve()
    ascent_gnc = Path(args.ascent_gnc).resolve()
    os.chdir(REPO_ROOT)

    import star_reacher
    from star_reacher._corelink import import_core

    core = import_core()

    metrics_requested = [m.strip() for m in args.metrics.split(",") if m.strip()]
    unknown = sorted(set(metrics_requested) - set(GATES))
    if unknown:
        print(f"perf_gate: unknown metric(s): {', '.join(unknown)}", file=sys.stderr)
        return 2

    document = {
        "schema": SCHEMA_MEASUREMENT,
        "generated_by": "scripts/perf_gate.py measure",
        "package_version": star_reacher.__version__,
        "core_version": core.core_version(),
        "git_sha": _git_sha(),
        "core_git_hash": core.git_hash(),
        "runner": _runner_identity(),
        "metrics": {},
        "all_pass": True,
    }

    print(f"perf_gate: measuring on {document['runner']['platform']}")
    print("perf_gate: sequential, single process at a time (D-10 discipline)")
    with tempfile.TemporaryDirectory(prefix="perf_gate_") as tmp:
        workdir = Path(tmp)
        # Strictly sequential: each measurement finishes (and its subprocess
        # exits) before the next starts.
        for name in metrics_requested:
            if name == "mission_a_wall_s":
                value, detail = measure_mission_a_wall_s(mission_a, workdir)
            elif name == "ascent_rt_factor":
                value, detail = measure_ascent_rt_factor(ascent, workdir)
            elif name == "ascent_gnc_rt_factor":
                value, detail = measure_ascent_rt_factor(
                    ascent_gnc, workdir, subdir="ascent_gnc"
                )
            else:
                value, detail = measure_srlog_write_mbps(args.srlog_records, workdir)
            gate = GATES[name]
            passed = gate_passes(name, value)
            document["metrics"][name] = {
                "value": value,
                "threshold": gate["threshold"],
                "sense": gate["sense"],
                "units": gate["units"],
                "pass": passed,
                "detail": detail,
            }
            document["all_pass"] = document["all_pass"] and passed
            print(
                f"{name} = {value:.6g} {gate['units']} "
                f"(gate: {gate['gate_text']}) {'PASS' if passed else 'FAIL'}"
            )

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"wrote {json_out}")

    print(f"PERF: {'PASS' if document['all_pass'] else 'FAIL'}")
    return 0 if document["all_pass"] else 1


# ---------------------------------------------------------------------------
# median / compare
# ---------------------------------------------------------------------------


def _load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def cmd_median(args: argparse.Namespace) -> int:
    documents = [_load_json(p) for p in args.measurements]
    if not documents:
        print("perf_gate median: no measurement files given", file=sys.stderr)
        return 2
    doc = median_of_documents(documents)
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name, entry in doc["metrics"].items():
        print(f"{name}: median {entry['median']:.6g} over {entry['n']} run(s)")
    print(f"wrote {out}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    current = _load_json(args.current)
    median = _load_json(args.median)
    ok, lines = compare_documents(current, median, args.tolerance)
    for line in lines:
        print(line)
    print(f"PERF REGRESSION GATE: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="perf_gate.py",
        description="star_reacher performance gates (FR-32; Phase 5 EC-4/EC-5).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_measure = sub.add_parser(
        "measure", help="measure the EC-4 metrics and gate the absolutes"
    )
    p_measure.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="comma-separated subset of: " + ", ".join(DEFAULT_METRICS),
    )
    p_measure.add_argument(
        "--mission-a",
        default=str(REPO_ROOT / "missions" / "mission_a_cislunar.toml"),
        help="mission for the wall-time metric (default: Mission A)",
    )
    p_measure.add_argument(
        "--ascent",
        default=str(REPO_ROOT / "missions" / "ascent_leo.toml"),
        help="mission for the real-time-factor metric (default: the ascent)",
    )
    p_measure.add_argument(
        "--ascent-gnc",
        default=str(REPO_ROOT / "missions" / "ascent_leo_gnc.toml"),
        help="mission for the closed-loop GNC real-time-factor metric "
        "(default: the GNC ascent)",
    )
    p_measure.add_argument(
        "--srlog-records",
        type=int,
        default=500_000,
        help="truth records for the write benchmark (default: 500000, ~61 MB)",
    )
    p_measure.add_argument("--json", default=None, help="write the measurement JSON here")
    p_measure.set_defaults(func=cmd_measure)

    p_median = sub.add_parser(
        "median", help="reduce measurement JSONs to a per-metric median file"
    )
    p_median.add_argument("measurements", nargs="+", help="measurement JSON files")
    p_median.add_argument("--json", required=True, help="write the median JSON here")
    p_median.set_defaults(func=cmd_median)

    p_compare = sub.add_parser(
        "compare", help="EC-5: fail on >tolerance regression vs the median"
    )
    p_compare.add_argument("--current", required=True, help="current measurement JSON")
    p_compare.add_argument("--median", required=True, help="median JSON from `median`")
    p_compare.add_argument(
        "--tolerance",
        type=float,
        default=0.10,
        help="allowed fractional regression (default 0.10 per EC-5)",
    )
    p_compare.set_defaults(func=cmd_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
