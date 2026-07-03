#!/usr/bin/env python3
"""Criterion-8 cross-platform divergence tooling (PRD Phase 2 exit criterion 8).

One file owns the final-state interchange format end to end, as three
subcommands used by ``.github/workflows/ci.yml``:

``extract``
    On a ``build-test`` matrix leg: load a ``run.srlog`` produced by
    ``star run missions/twobody_leo.toml``, take the FINAL truth record,
    and write ``finalstate.json`` carrying the leg identity and the state
    at full precision (authoritative fields are ``float.hex()`` strings;
    decimal mirrors are included for human reading only).

``measure``
    On the aggregation job: collect every leg's ``finalstate.json``,
    verify the leg count, leg uniqueness, and bit-identical final epochs,
    compute the maximum pairwise relative divergence, write
    ``measurement.json``, and print machine-readable ``max_rel=`` and
    ``status_state=`` lines (the workflow appends them to
    ``$GITHUB_OUTPUT`` and publishes ``max_rel`` as a commit status, which
    is readable without authentication on a public repository).

``gate``
    Enforce criterion 8 against the committed record
    ``tests/golden/determinism/cross_platform.toml``.

Divergence definition (this docstring is its single home):

    For every unordered pair of legs (i, j):
        rel_pos(i, j) = ||r_i - r_j||_2 / min(||r_i||_2, ||r_j||_2)
        rel_vel(i, j) = ||v_i - v_j||_2 / min(||v_i||_2, ||v_j||_2)
    max_rel = max over all pairs of max(rel_pos, rel_vel)

where r and v are the final truth-record position [m] and velocity [m/s]
vectors and ||.||_2 is the Euclidean norm. The scale is the smaller of the
two state-vector norms - the conservative choice, since it can only
enlarge the ratio. For the reference LEO mission ||r|| ~ 6.8e6 m and
||v|| ~ 7.6e3 m/s, so the denominators are far from zero; a zero-norm
state is rejected as undefined rather than silently scaled.

Gate rules (any failure = nonzero exit; no rule is advisory):

- ``measure`` fails when a leg artifact is missing or duplicated, or when
  the final epochs are not bit-identical across legs (an epoch mismatch is
  a configuration divergence, which would invalidate the comparison).
- ``gate`` fails when the measured ``max_rel`` exceeds the bound (D-10:
  1e-9 relative); when the committed record is missing, unparseable, or
  has ``status = "pending-first-measurement"`` (the designed bootstrap:
  the first CI run fails until the maintainer completes the record per its
  in-file procedure); when the record's ``measured_max_rel`` exceeds the
  bound; or when the record's ``bound_rel`` disagrees with the enforced
  bound (a D-10 revision must update both homes in the same change).
- ``gate`` WARNS, but does not fail, when the fresh measurement and the
  committed record differ by more than a factor of 10: runner-image
  compiler updates legitimately move the last digits, and the warning
  flags a record due for refresh via its recorded procedure.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import combinations
from pathlib import Path


def _fail(message: str) -> int:
    print(f"cross_platform_divergence: {message}", file=sys.stderr)
    return 1


# ---------------------------------------------------------------------------
# Final-state files and the divergence computation
# ---------------------------------------------------------------------------


def load_finalstate(path: Path) -> dict:
    """Parse one finalstate.json into floats (hex fields are authoritative)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("leg", "t_s_hex", "r_m_hex", "v_mps_hex"):
        if key not in data:
            raise ValueError(f"{path}: missing required field {key!r}")
    return {
        "leg": str(data["leg"]),
        "t_s_hex": str(data["t_s_hex"]),
        "t_s": float.fromhex(data["t_s_hex"]),
        "r_m": [float.fromhex(h) for h in data["r_m_hex"]],
        "v_mps": [float.fromhex(h) for h in data["v_mps_hex"]],
    }


def _norm(v: list[float]) -> float:
    return math.hypot(*v)


def _rel(a: list[float], b: list[float], what: str, pair: str) -> float:
    scale = min(_norm(a), _norm(b))
    if scale == 0.0:
        raise ValueError(
            f"zero-norm {what} vector in pair {pair}: the relative scale is undefined"
        )
    return math.dist(a, b) / scale


def max_pairwise_divergence(states: list[dict]) -> dict:
    """Maximum pairwise relative divergence per the module-docstring formula."""
    if len(states) < 2:
        raise ValueError("divergence needs at least two legs")
    max_rel = -1.0
    max_rel_pos = 0.0
    max_rel_vel = 0.0
    worst_pair = ""
    worst_quantity = ""
    for i, j in combinations(range(len(states)), 2):
        a, b = states[i], states[j]
        pair = f"{a['leg']} vs {b['leg']}"
        rel_pos = _rel(a["r_m"], b["r_m"], "position", pair)
        rel_vel = _rel(a["v_mps"], b["v_mps"], "velocity", pair)
        max_rel_pos = max(max_rel_pos, rel_pos)
        max_rel_vel = max(max_rel_vel, rel_vel)
        for quantity, rel in (("position", rel_pos), ("velocity", rel_vel)):
            if rel > max_rel:
                max_rel = rel
                worst_pair = pair
                worst_quantity = quantity
    return {
        "max_rel": max_rel,
        "max_rel_pos": max_rel_pos,
        "max_rel_vel": max_rel_vel,
        "worst_pair": worst_pair,
        "worst_quantity": worst_quantity,
    }


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_extract(args: argparse.Namespace) -> int:
    # Lazy imports: only the build-test legs have the package installed;
    # measure/gate run on a bare checkout with stdlib only.
    import platform

    from star_reacher.srlog import load

    run = load(Path(args.srlog))
    truth = run.groups["truth"]
    if len(truth) == 0:
        return _fail(f"{args.srlog}: empty truth group")
    rec = truth[-1]
    payload = {
        "schema": 1,
        "leg": args.leg,
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        # float.hex() fields are the authoritative full-precision values;
        # the decimal fields mirror them for human reading only.
        "t_s_hex": float(rec["t_s"]).hex(),
        "r_m_hex": [float(x).hex() for x in rec["r_m"]],
        "v_mps_hex": [float(x).hex() for x in rec["v_mps"]],
        "t_s": float(rec["t_s"]),
        "r_m": [float(x) for x in rec["r_m"]],
        "v_mps": [float(x) for x in rec["v_mps"]],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out} (leg {args.leg}, final t = {payload['t_s']!r} s)")
    return 0


def cmd_measure(args: argparse.Namespace) -> int:
    root = Path(args.dir)
    files = sorted(root.rglob("finalstate.json")) if root.is_dir() else []
    if not files:
        return _fail(f"no finalstate.json found under {root}")
    try:
        states = [load_finalstate(f) for f in files]
    except (ValueError, json.JSONDecodeError) as exc:
        return _fail(f"unparseable final-state artifact: {exc}")
    legs = [s["leg"] for s in states]
    if len(states) != args.expect_legs:
        return _fail(
            f"expected {args.expect_legs} leg artifacts, found {len(states)}: {legs} "
            f"(a missing leg means the comparison would silently shrink; failing instead)"
        )
    if len(set(legs)) != len(legs):
        return _fail(f"duplicate leg identifiers: {legs}")
    if len({s["t_s_hex"] for s in states}) != 1:
        detail = ", ".join(f"{s['leg']}: t = {s['t_s']!r} s" for s in states)
        return _fail(f"final epochs are not bit-identical across legs ({detail})")
    try:
        result = max_pairwise_divergence(states)
    except ValueError as exc:
        return _fail(str(exc))
    payload = {
        "schema": 1,
        "expect_legs": args.expect_legs,
        "bound_rel": args.bound,
        "t_s_hex": states[0]["t_s_hex"],
        "legs": {
            s["leg"]: {"r_m_hex": [x.hex() for x in s["r_m"]],
                       "v_mps_hex": [x.hex() for x in s["v_mps"]]}
            for s in states
        },
        **result,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"legs ({len(states)}): {', '.join(sorted(legs))}")
    print(f"final epoch t = {states[0]['t_s']!r} s (bit-identical on all legs)")
    print(
        f"max pairwise relative divergence: position {result['max_rel_pos']:.3e}, "
        f"velocity {result['max_rel_vel']:.3e}; worst {result['worst_quantity']} "
        f"pair: {result['worst_pair']}"
    )
    # Machine-readable lines consumed into $GITHUB_OUTPUT by the workflow.
    print(f"max_rel={result['max_rel']:.3e}")
    print(f"status_state={'success' if result['max_rel'] <= args.bound else 'failure'}")
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    try:
        measurement = json.loads(Path(args.measurement).read_text(encoding="utf-8"))
        measured = float(measurement["max_rel"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return _fail(f"cannot read measurement {args.measurement}: {exc}")

    failures: list[str] = []
    if measured > args.bound:
        failures.append(
            f"measured max_rel {measured:.3e} exceeds the D-10 bound {args.bound:.1e}; "
            f"per exit criterion 8 the bound must be formally revised in the same "
            f"change if this measurement is accepted"
        )

    record_path = Path(args.record)
    record_value: float | None = None
    if not record_path.is_file():
        failures.append(f"committed record {record_path} is missing")
    else:
        import tomllib

        try:
            with open(record_path, "rb") as fh:
                record = tomllib.load(fh)["record"]
        except (tomllib.TOMLDecodeError, KeyError) as exc:
            record = None
            failures.append(f"committed record {record_path} is unparseable: {exc}")
        if record is not None:
            status = record.get("status")
            if status == "pending-first-measurement":
                failures.append(
                    "the committed record is pending-first-measurement: this first "
                    "run is the designed bootstrap failure. Read the measured value "
                    "from the determinism/cross-platform commit status and complete "
                    f"{record_path} per the procedure in its comments."
                )
            elif status != "measured":
                failures.append(f"record status {status!r} is not 'measured'")
            else:
                if "measured_max_rel" not in record:
                    failures.append("record status is 'measured' but measured_max_rel is absent")
                else:
                    record_value = float(record["measured_max_rel"])
                    if record_value > args.bound:
                        failures.append(
                            f"committed measured_max_rel {record_value:.3e} exceeds "
                            f"the bound {args.bound:.1e}"
                        )
            bound_rel = record.get("bound_rel")
            if bound_rel is not None and float(bound_rel) != args.bound:
                failures.append(
                    f"record bound_rel {float(bound_rel):.1e} disagrees with the "
                    f"enforced bound {args.bound:.1e}; a D-10 revision must update "
                    f"both homes in the same change"
                )

    if record_value is not None and not failures:
        # Factor-10 drift between the fresh measurement and the record is a
        # prominent warning, never a failure: runner-image compiler updates
        # legitimately move the last digits.
        lo, hi = sorted((measured, record_value))
        if (lo == 0.0 and hi > 0.0) or (lo > 0.0 and hi > 10.0 * lo):
            banner = "!" * 78
            print(banner)
            print(
                f"WARNING: measured max_rel {measured:.3e} differs from the committed "
                f"record {record_value:.3e} by more than a factor of 10."
            )
            print(
                "Runner-image compiler updates legitimately move the last digits; "
                "refresh tests/golden/determinism/cross_platform.toml via the "
                "procedure recorded in its comments."
            )
            print(banner)

    if failures:
        for f in failures:
            print(f"cross_platform_divergence: gate failure: {f}", file=sys.stderr)
        return 1
    print(
        f"criterion-8 gate passed: measured max_rel {measured:.3e} and committed "
        f"record {record_value:.3e} are both within {args.bound:.1e}"
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cross_platform_divergence.py",
        description="Criterion-8 cross-platform divergence tooling (see module docstring).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract", help="write finalstate.json from a run.srlog")
    p_extract.add_argument("--srlog", required=True, help="path to run.srlog")
    p_extract.add_argument("--leg", required=True, help="CI leg identifier (matrix.os)")
    p_extract.add_argument("--out", required=True, help="output finalstate.json path")

    p_measure = sub.add_parser("measure", help="aggregate leg artifacts and measure")
    p_measure.add_argument(
        "--dir", required=True, help="directory searched recursively for finalstate.json"
    )
    p_measure.add_argument("--expect-legs", type=int, required=True)
    p_measure.add_argument("--bound", type=float, default=1e-9)
    p_measure.add_argument("--out", required=True, help="output measurement.json path")

    p_gate = sub.add_parser("gate", help="enforce criterion 8 against the committed record")
    p_gate.add_argument("--measurement", required=True, help="measurement.json from measure")
    p_gate.add_argument("--record", required=True, help="committed cross_platform.toml record")
    p_gate.add_argument("--bound", type=float, default=1e-9)

    args = parser.parse_args(argv)
    if args.command == "extract":
        return cmd_extract(args)
    if args.command == "measure":
        return cmd_measure(args)
    return cmd_gate(args)


if __name__ == "__main__":
    sys.exit(main())
