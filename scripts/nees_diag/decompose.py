"""Split the ensemble NEES excess into a covariance term and a bias term.

For a state error e ~ N(b, Sigma) reported under covariance P, the expected
NEES separates exactly:

    E[e' P^-1 e] = trace(P^-1 Sigma) + b' P^-1 b.

The first term is 15 when the reported covariance matches the true spread, so
``trace(P^-1 Sigma) - 15`` measures *covariance* error and ``b' P^-1 b``
measures *deterministic estimator bias*. The two are produced by disjoint
mechanisms:

* a truncated state-transition matrix or a truncated process-noise
  discretization mis-states P and moves the covariance term only;
* an unmodelled deterministic propagation error -- for instance a
  mechanization whose integration truncation error the filter's process noise
  does not describe -- moves the bias term only.

Reporting the split is therefore the measurement that decides which family of
mechanism is responsible, before any candidate is examined individually.

The covariance is averaged over the runs whose P was cached; P depends on the
trajectory only weakly, and the script reports the spread across those runs so
the reader can see that using a mean P is not what produces the answer.
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

from star_reacher.consistency import unpack_symmetric  # noqa: E402

BLOCK_NAMES = ("att", "vel", "pos", "bg", "ba")


def decompose(e15: np.ndarray, p_packed: np.ndarray) -> dict:
    """Covariance/bias decomposition of the ensemble NEES, epoch by epoch."""
    runs = e15.shape[0]
    p_mean = unpack_symmetric(p_packed.mean(axis=0))  # (T, 15, 15)
    chol = np.linalg.cholesky(p_mean)

    bias = e15.mean(axis=0)  # (T, 15)
    centered = e15 - bias
    # Sigma_k as the ensemble scatter at each epoch, with the mean removed.
    sigma = np.einsum("rti,rtj->tij", centered, centered) / (runs - 1)

    # trace(P^-1 Sigma) via the Cholesky factor: trace(L^-1 Sigma L^-T).
    whitened = np.linalg.solve(chol, sigma)
    whitened = np.linalg.solve(chol, np.swapaxes(whitened, -1, -2))
    cov_term = np.einsum("tii->t", whitened)

    zb = np.linalg.solve(chol, bias[..., np.newaxis])[..., 0]
    bias_term = np.einsum("ti,ti->t", zb, zb)

    # Per-block attribution of the bias term, against each block's marginal
    # covariance: it names which states carry the deterministic error.
    block_bias = np.empty((e15.shape[1], 5))
    for index in range(5):
        sl = slice(3 * index, 3 * index + 3)
        sub_chol = np.linalg.cholesky(p_mean[:, sl, sl])
        zz = np.linalg.solve(sub_chol, bias[:, sl, np.newaxis])[..., 0]
        block_bias[:, index] = np.einsum("ti,ti->t", zz, zz)

    return {
        "cov_term": cov_term,
        "bias_term": bias_term,
        "block_bias": block_bias,
        "bias": bias,
        "p_mean": p_mean,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures", default=str(REPO_ROOT / "fixtures" / "nees_diag")
    )
    parser.add_argument("--variants", nargs="*", default=None)
    args = parser.parse_args(argv)

    root = Path(args.fixtures)
    names = args.variants or [
        p.name[: -len("_err.npz")] for p in sorted(root.glob("*_err.npz"))
    ]

    for name in names:
        path = root / ("%s_err.npz" % name)
        if not path.exists():
            print("%-16s (no cached errors)" % name)
            continue
        data = np.load(path)
        e15 = data["e15"]
        result = decompose(e15, data["P"])
        t_s = data["t_s"]

        cov_mean = float(result["cov_term"].mean())
        bias_mean = float(result["bias_term"].mean())
        print("=" * 72)
        print(
            "%s  R=%d  T=%d  duration %.0f s"
            % (name, e15.shape[0], e15.shape[1], t_s[-1])
        )
        print(
            "  headline NEES = %.4f   covariance term %.4f (%+.2f %% of 15)"
            "   bias term %.4f (%+.2f %%)"
            % (
                cov_mean + bias_mean,
                cov_mean,
                100.0 * (cov_mean / 15.0 - 1.0),
                bias_mean,
                100.0 * bias_mean / 15.0,
            )
        )
        # The epoch profile of each term: a bias that accumulates with time
        # and a covariance error that does not look very different in the
        # headline but completely different here.
        marks = [0, len(t_s) // 4, len(t_s) // 2, 3 * len(t_s) // 4, len(t_s) - 1]
        print("   t_s      cov_term   bias_term   " + "  ".join(
            "%6s" % b for b in BLOCK_NAMES))
        for index in marks:
            print(
                "  %6.1f   %8.4f   %9.4f   " % (
                    t_s[index],
                    result["cov_term"][index],
                    result["bias_term"][index],
                )
                + "  ".join("%6.3f" % v for v in result["block_bias"][index])
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
