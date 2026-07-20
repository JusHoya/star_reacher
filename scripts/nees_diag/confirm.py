"""Confirm the measured estimator bias against the mechanization prediction.

The decomposition in ``decompose.py`` shows the NEES excess is a bias term
carried almost entirely by the velocity block. This script tests whether that
bias is the filter's gravity-mechanization truncation error, using the one
check a magnitude comparison alone cannot give: direction.

The truncation error is a specific vector. Running the filter's own
mechanization noise-free from the true initial state and differencing against
an RK4 solution of the same dynamics predicts, at each epoch, both how large
the velocity error should be and which way it should point. A bias that
matches in magnitude but not in direction would be a coincidence; one that
matches in both is the mechanism.

Reported per epoch:

* ``|b_v|``  -- the ensemble-mean velocity error, i.e. the realized bias;
* ``|dv|``   -- the free-running mechanization truncation error;
* ``cos``    -- the cosine between them;
* ``sigma``  -- the filter's own reported velocity 1-sigma, which sets how
  much of the bias the NEES gate sees.

The realized bias is smaller than the free-running truncation error because
the aiding measurements continually correct part of it; the ratio is the
fraction the filter fails to remove, and it is reported rather than assumed.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mechanisms as mech  # noqa: E402

from star_reacher.consistency import unpack_symmetric  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures", default=str(REPO_ROOT / "fixtures" / "nees_diag")
    )
    parser.add_argument("--variant", default="base")
    parser.add_argument("--dt", type=float, default=0.1)
    args = parser.parse_args(argv)

    root = Path(args.fixtures)
    errors = np.load(root / ("%s_err.npz" % args.variant))
    e15 = errors["e15"]
    t_s = errors["t_s"]
    p_mean = unpack_symmetric(errors["P"].mean(axis=0))

    raw = mech.load_raw(root / ("%s.npz" % args.variant))
    truth_r = raw["truth_r_m"]
    truth_v = raw["truth_v_mps"]

    n_steps = truth_r.shape[0] - 1
    ps, vs = mech.mechanization_truncation(
        truth_r[0], truth_v[0], args.dt, n_steps
    )
    pr, vr = mech.rk4_reference(truth_r[0], truth_v[0], args.dt, n_steps)
    trunc_v = vs - vr
    trunc_p = ps - pr

    bias = e15.mean(axis=0)
    bias_v = bias[:, 3:6]
    bias_p = bias[:, 6:9]

    print("=" * 78)
    print(
        "%s: measured ensemble bias against the mechanization truncation "
        "prediction" % args.variant
    )
    print("  R = %d runs, dt = %g s" % (e15.shape[0], args.dt))
    print()
    print(
        "   t_s     |b_v| m/s   |dv|trunc    ratio   cos     "
        "sigma_v      |b_v|/sigma"
    )
    marks = [int(f * (len(t_s) - 1)) for f in (0.1, 0.25, 0.5, 0.75, 1.0)]
    for k in marks:
        bv = bias_v[k]
        tv = trunc_v[k]
        nb = float(np.linalg.norm(bv))
        nt = float(np.linalg.norm(tv))
        cos = float(bv @ tv / (nb * nt)) if nb > 0 and nt > 0 else float("nan")
        sigma = float(np.sqrt(np.trace(p_mean[k, 3:6, 3:6]) / 3.0))
        print(
            "  %5.1f    %.4e  %.4e   %5.3f   %+.3f   %.4e   %6.3f"
            % (t_s[k], nb, nt, nb / nt, cos, sigma, nb / sigma)
        )

    print()
    print("  position channel, same comparison:")
    for k in marks:
        bp = bias_p[k]
        tp = trunc_p[k]
        nb = float(np.linalg.norm(bp))
        nt = float(np.linalg.norm(tp))
        cos = float(bp @ tp / (nb * nt)) if nb > 0 and nt > 0 else float("nan")
        sigma = float(np.sqrt(np.trace(p_mean[k, 6:9, 6:9]) / 3.0))
        print(
            "  %5.1f    %.4e  %.4e   %5.3f   %+.3f   %.4e   %6.3f"
            % (t_s[k], nb, nt, nb / nt, cos, sigma, nb / sigma)
        )

    # The covariance term at epoch 0 measures whether the initial error was
    # drawn from the covariance the filter starts with; a deficit there is an
    # initialization mismatch rather than a propagation defect, and it is
    # reported per block so the responsible states are named.
    print()
    print("  epoch-0 covariance term per block (each block expects 3.000):")
    centered = e15[:, 0, :] - bias[0]
    sigma0 = centered.T @ centered / (e15.shape[0] - 1)
    for index, name in enumerate(("att", "vel", "pos", "bg", "ba")):
        sl = slice(3 * index, 3 * index + 3)
        term = np.trace(np.linalg.solve(p_mean[0, sl, sl], sigma0[sl, sl]))
        print("    %-4s %6.3f" % (name, term))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
