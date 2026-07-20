"""Regenerate the whole EKF NEES-bias attribution from the cached fixtures.

One entry point that reproduces every number the diagnosis rests on, so the
conclusion can be re-derived without re-executing the native core. Run
``gen_fixtures.py`` and ``gen_errors.py`` first to populate the cache.

The sections correspond to the discriminating measurements:

1. **Variant sweep.** The ensemble NEES excess under every scenario knob. The
   cadence sweep moves the propagation count at a fixed update count; the
   aiding-rate sweep moves the update count at a fixed propagation count; the
   measurement-noise sweep moves the covariance the bias is measured against.

2. **Covariance/bias decomposition.** ``E[NEES] = trace(P^-1 Sigma) +
   b' P^-1 b`` evaluated per variant. A covariance-side mechanism moves the
   first term, an estimator bias the second.

3. **Candidate magnitudes.** What each named candidate actually contributes on
   the reference trajectory, in NEES units.

4. **Criterion-3 pass probability.** The gate applied to disjoint 100-run
   ensembles, and where the pinned seeds 0-99 draw sits among them.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

# Grouped so the sweep reads as the experiment it is rather than an
# alphabetical list.
GROUPS = (
    ("reference", ("base",)),
    (
        "cadence sweep (propagation steps vary, 60 updates fixed)",
        ("rate_5hz", "base", "rate_20hz", "rate_50hz", "rate_100hz"),
    ),
    ("duration sweep", ("dur_15s", "base", "dur_120s", "dur_240s")),
    (
        "aiding-rate sweep (updates vary, 600 propagation steps fixed)",
        ("base", "st_5hz", "st_10hz", "navfix_5hz", "navfix_10hz"),
    ),
    (
        "aiding-suite sweep",
        ("no_aiding", "only_st", "only_alt", "only_navfix", "base"),
    ),
    (
        "noise sweep (moves the covariance the bias is measured against)",
        ("navfix_loose", "base", "navfix_tight", "imu_noise_x10", "imu_noise_x01"),
    ),
)


def variant_sweep(root: Path) -> None:
    print("=" * 78)
    print("1. VARIANT SWEEP -- ensemble NEES excess over the expected 15")
    for title, names in GROUPS:
        print("\n  %s" % title)
        for name in names:
            path = root / ("%s.npz" % name)
            if not path.exists():
                print("    %-14s (not cached)" % name)
                continue
            data = np.load(path)
            eps = data["nees"].astype(np.float64)
            grand = float(eps.mean())
            print(
                "    %-14s R=%4d T=%5d  NEES %8.4f  excess %+7.2f %%"
                % (name, eps.shape[0], eps.shape[1], grand,
                   100.0 * (grand / 15.0 - 1.0))
            )


def run(script: str, *args: str) -> None:
    # The child writes straight to this process's stdout, so the parent's own
    # buffer has to be drained first or the sections come out interleaved.
    sys.stdout.flush()
    subprocess.run([sys.executable, str(HERE / script), *args], check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures", default=str(REPO_ROOT / "fixtures" / "nees_diag")
    )
    args = parser.parse_args(argv)
    root = Path(args.fixtures)

    variant_sweep(root)

    print()
    print("=" * 78)
    print("2. COVARIANCE / BIAS DECOMPOSITION")
    run(
        "decompose.py",
        "--fixtures",
        str(root),
        "--variants",
        "rate_5hz",
        "base",
        "rate_20hz",
        "rate_50hz",
        "navfix_tight",
    )

    print()
    print("=" * 78)
    print("3. CANDIDATE MAGNITUDES ON THE REFERENCE TRAJECTORY")
    run("mechanisms.py", "--fixtures", str(root))

    print()
    print("=" * 78)
    print("4. MEASURED BIAS AGAINST THE MECHANIZATION PREDICTION")
    run("confirm.py", "--fixtures", str(root))

    print()
    print("=" * 78)
    print("5. CRITERION-3 PASS PROBABILITY")
    run("pass_probability.py", "--fixtures", str(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
