"""Compare two cross-tool trajectories on their shared fixed 60 s grid.

Maintainer-side reporting tool behind the RMS numbers recorded in
``tests/golden/crosstool/manifest.toml`` (the CI gate itself is
``tests/python/test_crosstool_frozen_truth.py``, which implements the same
comparison independently). Each argument is either a frozen-truth CSV
(``t_s,x_m,...`` on the 60 s grid) or a mission ``run.srlog`` (1 Hz truth
log, sampled at the 60 s epochs). Prints position/velocity RMS, the maximum
position difference, and the radial/in-track/cross-track RMS decomposition.

Usage (repo root, main Python with star_reacher installed):

    python scripts/crosstool/compare_rms.py <a.csv|a.srlog> <b.csv|b.srlog>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

STEP_S = 60.0
N_ROWS = 10081


def load_grid(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(position, velocity) arrays on the exact 60 s grid, shape (10081, 3)."""
    if path.suffix == ".srlog":
        import star_reacher

        truth = star_reacher.load(path).groups["truth"]
        idx = np.arange(N_ROWS) * int(STEP_S)
        expected = idx.astype(np.float64)
        if not np.array_equal(truth["t_s"][idx], expected):
            raise SystemExit(f"{path}: truth log is not on the exact 1 Hz grid")
        return truth["r_m"][idx], truth["v_mps"][idx]
    rows = [
        line
        for line in path.read_text(encoding="ascii").splitlines()
        if line and not line.startswith("#")
    ]
    if rows[0] != "t_s,x_m,y_m,z_m,vx_mps,vy_mps,vz_mps":
        raise SystemExit(f"{path}: unexpected CSV column header {rows[0]!r}")
    data = np.array([[float(v) for v in line.split(",")] for line in rows[1:]])
    if data.shape != (N_ROWS, 7) or not np.array_equal(
        data[:, 0], np.arange(N_ROWS) * STEP_S
    ):
        raise SystemExit(f"{path}: CSV is not the exact 60 s grid over 7 days")
    return data[:, 1:4], data[:, 4:7]


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    (ra, va), (rb, vb) = (load_grid(Path(p)) for p in sys.argv[1:3])
    dr, dv = ra - rb, va - vb
    rms = float(np.sqrt(np.mean(np.sum(dr * dr, axis=1))))
    vrms = float(np.sqrt(np.mean(np.sum(dv * dv, axis=1))))
    dist = np.linalg.norm(dr, axis=1)
    # Radial / in-track / cross-track split in the first trajectory's frame.
    rn = ra / np.linalg.norm(ra, axis=1, keepdims=True)
    h = np.cross(ra, va)
    hn = h / np.linalg.norm(h, axis=1, keepdims=True)
    tn = np.cross(hn, rn)
    comps = [
        float(np.sqrt(np.mean(np.sum(dr * u, axis=1) ** 2))) for u in (rn, tn, hn)
    ]
    print(f"position RMS [m]      : {rms:.6e}")
    print(f"velocity RMS [m/s]    : {vrms:.6e}")
    print(f"max |dr| [m]          : {dist.max():.6e} at t = {60.0 * int(np.argmax(dist))} s")
    print(f"RMS radial/in-track/cross [m]: {comps[0]:.6e} / {comps[1]:.6e} / {comps[2]:.6e}")


if __name__ == "__main__":
    main()
