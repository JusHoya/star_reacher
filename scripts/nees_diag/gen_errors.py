"""Cache raw per-epoch state errors, so the NEES excess can be decomposed.

``gen_fixtures.py`` caches the NEES statistic itself, which says how large the
excess is but not what kind of defect produces it. The decomposition needs the
error vectors themselves.

For an error e ~ N(b, Sigma) reported under covariance P,

    E[NEES] = trace(P^-1 Sigma) + b' P^-1 b,

so the excess over the state dimension splits into a *covariance* term, which
is non-zero when the reported P mis-states the true spread, and a *bias* term,
which is non-zero when the estimator carries a deterministic error. The two
terms point at disjoint sets of mechanisms: a discretized Phi or Q is a
covariance defect, while an unmodelled deterministic propagation error is a
bias. Separating them is therefore the first discriminating measurement, and
it needs e itself rather than the scalar NEES.

Cached per variant: ``e15`` (R, T, 15) reduced error vectors, ``P`` (R, T, 120)
for the first ``--keep-p`` runs, and the epoch grid.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
# star_reacher is imported from the INSTALLED package, never from
# REPO_ROOT/python: the source tree carries no compiled _core, so putting
# it on sys.path shadows the wheel and makes every core-backed call in
# these diagnostics fail with CoreMissingError.
sys.path.insert(0, str(REPO_ROOT / "tests" / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gen_fixtures as fixtures  # noqa: E402
import test_ekf_consistency as driver  # noqa: E402

from star_reacher import load  # noqa: E402
from star_reacher.runner import run_mission  # noqa: E402


def run_variant(
    name: str,
    spec: dict,
    n_runs: int,
    keep_p: int,
    outdir: Path,
    workdir: Path,
    stride: int = 1,
) -> None:
    base = driver.MISSION.read_text()
    workdir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)

    errors: list[np.ndarray] = []
    covariances: list[np.ndarray] = []
    t_s: np.ndarray | None = None

    cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        for index in range(n_runs):
            mission_path = workdir / ("run%04d.toml" % index)
            mission_path.write_text(fixtures.variant_text(base, index, spec))
            result = run_mission(
                mission_path, workdir / ("run%04d" % index), force=True
            )
            run = load(result.srlog_path)
            # A fine-cadence variant has tens of thousands of epochs; the
            # decomposition is a smooth function of time, so a strided epoch
            # grid carries the same answer at a tenth the cache size.
            errors.append(
                driver.reduce_error(run.groups["nav.err"]["e"])[::stride]
            )
            if index < keep_p:
                covariances.append(run.groups["nav.est"]["P"][::stride])
            t_s = run.groups["nav.err"]["t_s"][::stride]
    finally:
        os.chdir(cwd)

    np.savez_compressed(
        outdir / ("%s_err.npz" % name),
        e15=np.stack(errors),
        P=np.stack(covariances),
        t_s=np.asarray(t_s),
    )
    print("%-16s cached e15 %s" % (name, np.stack(errors).shape), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(REPO_ROOT / "fixtures" / "nees_diag"))
    parser.add_argument("--work", default=None)
    parser.add_argument("--runs", type=int, default=400)
    parser.add_argument("--keep-p", type=int, default=25)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--only",
        nargs="*",
        default=["base", "rate_5hz", "rate_20hz", "only_navfix", "navfix_tight"],
    )
    args = parser.parse_args(argv)

    workdir = Path(
        args.work or (Path(os.environ.get("TEMP", "/tmp")) / "nees_diag_err")
    )
    for name in args.only:
        run_variant(
            name,
            fixtures.VARIANTS[name],
            args.runs,
            args.keep_p,
            Path(args.out),
            workdir / name,
            stride=args.stride,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
