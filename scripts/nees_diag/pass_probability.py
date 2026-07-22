"""Measure how often exit criterion 3 passes over independent ensembles.

Criterion 3 gates on one 100-run ensemble drawn from a pinned seed sequence.
That is a single realization of a random quantity, so whether it passes says
less than it appears to: the question the phase needs answered is what
fraction of equally valid 100-run ensembles would pass.

The script partitions a large cached ensemble into disjoint 100-run blocks --
disjoint, not bootstrap-resampled, because resampling with replacement from
one ensemble understates the spread when the underlying statistic is
correlated across epochs. Each block is put through the same gate the driver
applies: the epoch-averaged ensemble NEES against
``[chi2_0.025(100 n)/100, chi2_0.975(100 n)/100]``, and the same for each
per-sensor NIS.

Reported: the pass fraction per statistic, the pass fraction for the
conjunction that criterion 3 actually requires, where the pinned seeds 0-99
ensemble sits within the distribution of block headlines, and the margin the
headline has to the upper bound expressed in units of the block-to-block
standard deviation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
# star_reacher is imported from the INSTALLED package, never from
# REPO_ROOT/python: the source tree carries no compiled _core, so putting
# it on sys.path shadows the wheel and makes every core-backed call in
# these diagnostics fail with CoreMissingError.

from star_reacher.chi2 import chi2_ppf  # noqa: E402

NEES_DIM = 15
NIS_DIM_BY_SENSOR = {1: 3, 2: 6, 3: 1}


def bounds(dim: int, n_runs: int) -> tuple[float, float]:
    dof = n_runs * dim
    return chi2_ppf(0.025, dof) / n_runs, chi2_ppf(0.975, dof) / n_runs


def block_headlines(eps: np.ndarray, block: int) -> np.ndarray:
    """Epoch-averaged ensemble statistic for each disjoint block of runs."""
    n_blocks = eps.shape[0] // block
    trimmed = eps[: n_blocks * block]
    return trimmed.reshape(n_blocks, block, -1).mean(axis=(1, 2))


def report(name: str, eps: np.ndarray, dim: int, block: int) -> np.ndarray:
    lower, upper = bounds(dim, block)
    headlines = block_headlines(eps, block)
    passed = (headlines >= lower) & (headlines <= upper)
    print(
        "  %-10s bounds [%.4f, %.4f]  blocks %3d  pass %3d (%5.1f %%)  "
        "mean %.4f  sd %.4f"
        % (
            name,
            lower,
            upper,
            len(headlines),
            int(passed.sum()),
            100.0 * passed.mean(),
            float(headlines.mean()),
            float(headlines.std(ddof=1)),
        )
    )
    return passed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures", default=str(REPO_ROOT / "fixtures" / "nees_diag")
    )
    parser.add_argument("--variant", default="base_4000")
    parser.add_argument("--block", type=int, default=100)
    args = parser.parse_args(argv)

    data = np.load(Path(args.fixtures) / ("%s.npz" % args.variant))
    nees_eps = data["nees"].astype(np.float64)
    block = args.block

    print("=" * 78)
    print(
        "criterion-3 pass probability from %s: %d runs in %d disjoint "
        "%d-run ensembles"
        % (args.variant, nees_eps.shape[0], nees_eps.shape[0] // block, block)
    )
    print()
    all_passed = report("NEES", nees_eps, NEES_DIM, block)
    for sensor_id, dim in sorted(NIS_DIM_BY_SENSOR.items()):
        key = "nis_%d" % sensor_id
        if key not in data.files:
            continue
        passed = report(
            "NIS s%d" % sensor_id, data[key].astype(np.float64), dim, block
        )
        all_passed = all_passed & passed
    print()
    print(
        "  criterion 3 (all four statistics inside): %d/%d blocks pass "
        "(%.1f %%)"
        % (int(all_passed.sum()), len(all_passed), 100.0 * all_passed.mean())
    )

    lower, upper = bounds(NEES_DIM, block)
    headlines = block_headlines(nees_eps, block)
    pinned = float(nees_eps[:block].mean())
    sd = float(headlines.std(ddof=1))
    print()
    print("  pinned seeds 0-%d headline      %.4f" % (block - 1, pinned))
    print("  mean over all blocks           %.4f" % headlines.mean())
    print(
        "  pinned percentile among blocks %.1f %%"
        % (100.0 * float((headlines < pinned).mean()))
    )
    print(
        "  margin of the pinned ensemble to the upper bound: %.4f "
        "(%.2f block sd)" % (upper - pinned, (upper - pinned) / sd)
    )
    print(
        "  margin of the block mean to the upper bound:      %.4f "
        "(%.2f block sd)"
        % (upper - headlines.mean(), (upper - headlines.mean()) / sd)
    )

    # The epoch profile is what the coverage criterion sees, and it is where a
    # monotonic drift shows up most plainly.
    epoch_mean = nees_eps.mean(axis=0)
    t_s = data["t_s"]
    print()
    print("  epoch profile of the %d-run ensemble mean:" % nees_eps.shape[0])
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        k = int(frac * (len(t_s) - 1))
        print("    t = %6.1f s   %.4f" % (t_s[k], epoch_mean[k]))
    print(
        "  epochs above the upper bound %.4f: %d of %d"
        % (upper, int((epoch_mean > upper).sum()), len(epoch_mean))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
