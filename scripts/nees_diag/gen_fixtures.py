"""Generate and cache the ensembles the EKF NEES-bias diagnosis runs against.

The diagnosis measures how the reference filter's ensemble NEES excess
responds to scenario knobs that move the propagation count and the update
count independently. Every ensemble is produced once here and written to a
``.npz`` per variant, so the analysis scripts read cached arrays and never
re-execute the core -- the native module is a shared build slot and cannot be
assumed stable across a long analysis session.

Each variant derives from ``missions/leo_ekf_consistency.toml`` through the
same per-run initial-estimate draw the exit-criterion-3 driver uses
(``tests/python/test_ekf_consistency.py``), so a variant differs from the
committed gate only in the knobs it names.

Cached per variant:

* ``nees`` (R, T) -- the full 15-state per-epoch NEES of every run;
* ``block_nees`` (R, T, 5) -- the per-epoch NEES of the five 3-vector blocks
  against their own marginal covariance, which localizes an excess to a block;
* ``t_s`` (T,) -- the epoch grid;
* ``nis_<id>`` (R, U) -- per-sensor NIS, to confirm the measurement side stays
  on target under the same knob.

A small number of runs per variant additionally cache the raw log arrays
needed to re-run the filter offline (``--raw N``): the IMU increments, every
aiding measurement, truth, and the filter's own ``x_hat``/``P``. The offline
re-filter is what separates candidate mechanisms that a scenario knob alone
cannot.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT / "tests" / "python"))

import test_ekf_consistency as driver  # noqa: E402

from star_reacher import load  # noqa: E402
from star_reacher.consistency import nees, nis, unpack_symmetric  # noqa: E402
from star_reacher.runner import run_mission  # noqa: E402

BLOCK_NAMES = ("att", "vel", "pos", "bg", "ba")


def set_in_section(text: str, section: str | None, key: str, value: str) -> str:
    """Set ``key`` inside ``section`` only, leaving same-named keys elsewhere.

    ``sample_rate_hz`` appears under four different sensor tables, so a
    document-wide substitution would silently retune instruments the variant
    never named.
    """
    lines = text.splitlines()
    current: str | None = None
    hit = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current = stripped[1:-1]
            continue
        if current != section:
            continue
        if re.match(r"^\s*%s\s*=" % re.escape(key), line):
            lines[index] = "%s = %s" % (key, value)
            hit = True
    if not hit:
        raise KeyError("key %r not found in section %r" % (key, section))
    return "\n".join(lines) + "\n"


def drop_section(text: str, section: str) -> str:
    """Remove a whole table, which is how a variant disables one sensor."""
    lines = text.splitlines()
    out: list[str] = []
    skipping = False
    found = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            skipping = stripped[1:-1] == section
            found = found or skipping
        if not skipping:
            out.append(line)
    if not found:
        raise KeyError("section %r not found" % section)
    return "\n".join(out) + "\n"


def variant_text(base: str, run_index: int, spec: dict) -> str:
    """One ensemble member of one variant."""
    text = driver.mission_text_for_run(base, run_index)
    for section in spec.get("drop", ()):
        text = drop_section(text, section)
    for (section, key), value in spec.get("set", {}).items():
        text = set_in_section(text, section, key, value)
    return text


def rate_spec(rate_hz: int, duration_s: float) -> dict:
    """The tied cadence knobs: D-5 pins dt_s = 1 / control_rate_hz exactly.

    The IMU samples at the control rate and truth is logged at the control
    rate, so a cadence variant has to move all four together or the run is no
    longer the reference scenario at a different step size. The schema takes
    every rate as an integer Hz, which is what bounds the sweep to divisors of
    a whole second.
    """
    return {
        "set": {
            ("integrator", "dt_s"): repr(1.0 / rate_hz),
            ("gnc", "control_rate_hz"): str(int(rate_hz)),
            ("logging", "truth_rate_hz"): str(int(rate_hz)),
            ("sensors.imu", "sample_rate_hz"): str(int(rate_hz)),
            ("mission", "duration_s"): repr(duration_s),
        }
    }


def block_nees(e15: np.ndarray, p_packed: np.ndarray) -> np.ndarray:
    """Per-epoch NEES of each 3-vector block against its marginal covariance."""
    p_full = unpack_symmetric(p_packed)
    out = np.empty(e15.shape[:-1] + (5,), dtype=np.float64)
    for block in range(5):
        sl = slice(3 * block, 3 * block + 3)
        sub = p_full[..., sl, sl]
        vec = e15[..., sl]
        chol = np.linalg.cholesky(sub)
        z = np.linalg.solve(chol, vec[..., np.newaxis])[..., 0]
        out[..., block] = np.einsum("...i,...i->...", z, z)
    return out


RAW_FIELDS = {
    "truth": ("t_s", "r_m", "v_mps", "q_i2b", "w_b_radps"),
    "sensors.imu": ("t_s", "dtheta_b_rad", "dv_b_mps"),
    "sensors.startracker": ("t_s", "q_meas_i2b", "valid"),
    "sensors.navfix": ("t_s", "r_meas_m", "v_meas_mps"),
    "sensors.altimeter": ("t_s", "alt_meas_m"),
    "nav.est": ("t_s", "x_hat", "P"),
    "nav.err": ("t_s", "e"),
    "nav.innov": ("t_s", "sensor_id", "m", "y", "S"),
}


def run_variant(
    name: str, spec: dict, n_runs: int, n_raw: int, outdir: Path, workdir: Path
) -> None:
    base = driver.MISSION.read_text()
    workdir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)

    nees_runs: list[np.ndarray] = []
    block_runs: list[np.ndarray] = []
    nis_runs: dict[int, list[np.ndarray]] = {}
    raw: dict[str, np.ndarray] = {}
    t_s: np.ndarray | None = None

    cwd = os.getcwd()
    os.chdir(REPO_ROOT)  # the mission's vehicle path is repository-relative
    try:
        for index in range(n_runs):
            mission_path = workdir / ("run%04d.toml" % index)
            mission_path.write_text(variant_text(base, index, spec))
            result = run_mission(
                mission_path, workdir / ("run%04d" % index), force=True
            )
            run = load(result.srlog_path)
            e15 = driver.reduce_error(run.groups["nav.err"]["e"])
            p_packed = run.groups["nav.est"]["P"]
            nees_runs.append(nees(e15, p_packed))
            block_runs.append(block_nees(e15, p_packed))
            t_s = run.groups["nav.err"]["t_s"]
            for sensor_id, (y, s) in driver._per_sensor_innovations(
                run.groups["nav.innov"]
            ).items():
                nis_runs.setdefault(sensor_id, []).append(nis(y, s))
            if index < n_raw:
                for group, fields in RAW_FIELDS.items():
                    for field in fields:
                        raw["raw%02d_%s_%s" % (index, group, field)] = run.groups[
                            group
                        ][field]
            if (index + 1) % 100 == 0:
                print("  %s: %d/%d" % (name, index + 1, n_runs), flush=True)
    finally:
        os.chdir(cwd)

    payload = {
        "nees": np.stack(nees_runs),
        "block_nees": np.stack(block_runs),
        "t_s": np.asarray(t_s),
        "n_runs": np.asarray(n_runs),
    }
    for sensor_id, arrays in nis_runs.items():
        payload["nis_%d" % sensor_id] = np.stack(arrays)
    payload.update(raw)
    path = outdir / ("%s.npz" % name)
    np.savez_compressed(path, **payload)
    grand = payload["nees"].mean()
    print(
        "%-16s R=%d T=%d  NEES grand mean %.4f  (excess %+.2f %%)"
        % (name, payload["nees"].shape[0], payload["nees"].shape[1], grand,
           100.0 * (grand / 15.0 - 1.0)),
        flush=True,
    )


# The variant battery. The cadence sweep moves the propagation count while
# holding every continuous noise density and every aiding rate fixed; the
# duration sweep moves both counts together; the aiding-rate sweep moves the
# update count at a fixed propagation count. Between them the three separate a
# per-propagation-step mechanism from a per-update one.
VARIANTS: dict[str, dict] = {
    "base": {},
    "rate_5hz": rate_spec(5, 60.0),
    "rate_20hz": rate_spec(20, 60.0),
    "rate_50hz": rate_spec(50, 60.0),
    "rate_100hz": rate_spec(100, 60.0),
    "dur_120s": {"set": {("mission", "duration_s"): "120.0"}},
    "dur_240s": {"set": {("mission", "duration_s"): "240.0"}},
    "dur_15s": {"set": {("mission", "duration_s"): "15.0"}},
    # Aiding-rate sweep at the pinned 10 Hz cadence: the propagation count is
    # unchanged, only how often a reset runs. The schema's integer-Hz rule
    # bounds this to raising a rate, so the sweep runs upward from the
    # reference 1 Hz and the "slower" end is covered by the drop variants.
    "st_5hz": {"set": {("sensors.startracker", "sample_rate_hz"): "5"}},
    "st_10hz": {"set": {("sensors.startracker", "sample_rate_hz"): "10"}},
    "navfix_5hz": {"set": {("sensors.navfix", "sample_rate_hz"): "5"}},
    "navfix_10hz": {"set": {("sensors.navfix", "sample_rate_hz"): "10"}},
    # Single-aiding-sensor variants isolate which update chain carries the
    # excess, and the no-aiding variant is pure propagation.
    "only_navfix": {"drop": ("sensors.startracker", "sensors.altimeter")},
    "only_st": {"drop": ("sensors.navfix", "sensors.altimeter")},
    "only_alt": {"drop": ("sensors.navfix", "sensors.startracker")},
    "no_aiding": {
        "drop": ("sensors.navfix", "sensors.startracker", "sensors.altimeter")
    },
    # Noise-scale variants: a mechanism driven by an unmodelled deterministic
    # error grows relative to the noise floor as the noise is reduced, while a
    # mechanism that is a fixed fraction of the covariance does not move.
    "imu_noise_x10": {
        "set": {
            ("sensors.imu", "gyro_arw_rad_per_sqrt_s"): "1.0e-4",
            ("sensors.imu", "accel_vrw_mps_per_sqrt_s"): "1.0e-3",
        }
    },
    "imu_noise_x01": {
        "set": {
            ("sensors.imu", "gyro_arw_rad_per_sqrt_s"): "1.0e-6",
            ("sensors.imu", "accel_vrw_mps_per_sqrt_s"): "1.0e-5",
        }
    },
    "navfix_tight": {
        "set": {
            ("sensors.navfix", "sigma_r_m"): "[1.0, 1.0, 1.0]",
            ("sensors.navfix", "sigma_v_mps"): "[0.01, 0.01, 0.01]",
        }
    },
    "navfix_loose": {
        "set": {
            ("sensors.navfix", "sigma_r_m"): "[100.0, 100.0, 100.0]",
            ("sensors.navfix", "sigma_v_mps"): "[1.0, 1.0, 1.0]",
        }
    },
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(REPO_ROOT / "fixtures" / "nees_diag"))
    parser.add_argument("--work", default=None)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--base-runs", type=int, default=1000)
    parser.add_argument("--raw", type=int, default=3)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args(argv)

    outdir = Path(args.out)
    workdir = Path(
        args.work
        or (Path(os.environ.get("TEMP", "/tmp")) / "nees_diag_work")
    )
    names = args.only if args.only else list(VARIANTS)
    for name in names:
        if name not in VARIANTS:
            raise SystemExit("unknown variant %r" % name)
        n_runs = args.base_runs if name == "base" else args.runs
        run_variant(
            name, VARIANTS[name], n_runs, args.raw, outdir, workdir / name
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
