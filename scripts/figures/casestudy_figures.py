"""Regenerate the scientific-report case-study figures (FR-30, Phase 8).

The report's two case studies --- a translunar transfer and an Earth--Mars
transfer --- are illustrated by figures that this script regenerates
deterministically. It is split into two strictly separated stages so the
"regenerate from committed seeds reproduces bit-for-bit" guarantee (Phase 8
exit criterion 3) is both true and cheap to check:

- ``--extract`` re-runs the two committed missions with ``star run``,
  decimates each truth log to a small fixed-stride sample, derives the
  figure quantities, and writes one committed ``.npz`` per mission under
  ``docs/report/figures/data/`` plus a provenance manifest. This stage needs
  the compiled core and the committed ephemeris excerpts; it is run by the
  maintainer when a mission or a model changes, and its inputs (the mission
  TOML plus its seed) are committed, so the extracted data traces to them.
- ``--render`` (the default) reads only the committed ``.npz`` inputs and
  matplotlib to write the PNGs. It never touches the core, the network, or
  the clock, and its PNG bytes are a pure function of the committed inputs
  and the pinned matplotlib/FreeType versions (see the README beside this
  file). This is the stage Phase 8 criterion 3 gates.

Determinism discipline (mirrors ``python/star_reacher/plotting.py``):

- ``SOURCE_DATE_EPOCH`` is honored --- matplotlib reads it to stamp the PNG
  ``tIME`` chunk, so a fixed value makes the byte stream reproducible; the
  script sets a fixed fallback when the variable is absent so a bare
  ``--render`` is reproducible too.
- The Agg backend is forced before ``pyplot`` import (headless, no display
  server), ``metadata={"Software": None}`` drops the matplotlib version
  string from the PNG, and the figure geometry and DPI are fixed.
- No wall-clock, hostname, or other host input enters a figure.

Usage (from the repository root, with the project's Python):

    python scripts/figures/casestudy_figures.py            # render (default)
    python scripts/figures/casestudy_figures.py --extract  # re-derive inputs
    python scripts/figures/casestudy_figures.py --verify-repro  # crit-3 gate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# Repository root: this file lives at <root>/scripts/figures/.
_ROOT = Path(__file__).resolve().parents[2]
_FIG_DIR = _ROOT / "docs" / "report" / "figures"
# Committed figure-input data. Named "figure_data", not "data": the repo-root
# .gitignore ignores every directory named "data/" (the git-ignored fetched
# ephemeris, D-8), and that pattern is path-agnostic, so a "data/" subdir here
# would be silently untracked.
_DATA_DIR = _FIG_DIR / "figure_data"
_MANIFEST = _DATA_DIR / "manifest.json"

# A fixed SOURCE_DATE_EPOCH fallback keeps a bare render reproducible when the
# caller has not exported one (CI and the FR-29 docs build always set it from
# the HEAD commit time; the value chosen here is irrelevant to the figures
# because no wall-clock quantity is plotted). 2020-01-01T00:00:00Z.
_FIXED_EPOCH = "1577836800"

# Fixed render geometry and DPI: the PNG bytes must not depend on a host
# default, so both are pinned here rather than read from rcParams.
_DPI = 100

# Decimation target: ~2000 samples across each arc is plenty for a smooth
# trajectory curve and keeps the committed .npz inputs small (both arcs are
# ~4.5e5-6e5 truth records at 1 Hz). A fixed stride (not a fixed count) keeps
# the selection a pure function of the log length.
_TARGET_SAMPLES = 2000

# The two case-study missions and their committed figure identities. The
# ``sha`` fields are filled by --extract from the actual run and frozen into
# the manifest; --render reads them back for the figure captions so a figure
# can never be silently misattributed to a different mission revision.
_CASES = {
    "translunar": {
        "mission": "missions/tli.toml",
        "central_body": "earth",
        "png": "translunar_transfer.png",
        "npz": "translunar_transfer.npz",
    },
    "mars_transfer": {
        "mission": "missions/mars_cruise.toml",
        "central_body": "sun",
        "png": "mars_transfer.png",
        "npz": "mars_transfer.npz",
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Extraction (needs the compiled core; run by the maintainer)
# ---------------------------------------------------------------------------


def _run_mission(mission: str, outdir: Path) -> Path:
    """Propagate a committed mission with ``star run`` into ``outdir``.

    Shells out to the installed console script rather than re-implementing
    the config resolution, so the extracted data derives from exactly the
    path ``star run`` takes (and the same committed SHA-256).
    """
    outdir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [sys.executable, "-m", "star_reacher", "run", mission,
         "-o", str(outdir), "--force"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"star run failed for {mission} (exit {proc.returncode}):\n"
            f"{proc.stdout}\n{proc.stderr}"
        )
    log = outdir / "run.srlog"
    if not log.is_file():
        raise SystemExit(f"star run produced no run.srlog for {mission}")
    return log


def _decimate_indices(n: int) -> np.ndarray:
    """Fixed-stride sample indices covering the whole arc including the end.

    A pure function of ``n``: the stride is ``ceil(n / target)`` and the last
    sample is always included so the arc's terminus (the SOI crossing, the
    end of the cruise window) is on the curve.
    """
    if n <= _TARGET_SAMPLES:
        return np.arange(n)
    stride = -(-n // _TARGET_SAMPLES)  # ceil division
    idx = np.arange(0, n, stride)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    return idx


def _extract_case(key: str, tmp: Path) -> dict:
    """Run one case's mission and write its committed .npz figure input."""
    from star_reacher import load  # local import: core needed only here
    from star_reacher.derived import central_body_gm

    case = _CASES[key]
    log_path = _run_mission(case["mission"], tmp / key)
    run = load(log_path)
    truth = run.groups["truth"]
    n = len(truth)
    idx = _decimate_indices(n)

    t = np.asarray(truth["t_s"], dtype=np.float64)[idx]
    r = np.asarray(truth["r_m"], dtype=np.float64)[idx]
    v = np.asarray(truth["v_mps"], dtype=np.float64)[idx]
    dist = np.sqrt(np.einsum("ij,ij->i", r, r))

    # Osculating semi-major axis about the central body, from the decimated
    # samples (vis-viva); the loader's full-precision elements are not needed
    # for a figure and this keeps the .npz self-contained.
    gm = central_body_gm(run.header["central_body"])
    speed2 = np.einsum("ij,ij->i", v, v)
    energy = 0.5 * speed2 - gm / dist
    a_m = -gm / (2.0 * energy)

    ev = run.events
    ev_t = np.asarray(ev["t_s"], dtype=np.float64)
    ev_detail = np.array([str(d) for d in ev["detail"]])

    out = _DATA_DIR / case["npz"]
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Save without compression: np.savez writes a deterministic (uncompressed)
    # ZIP whose member order is the save order, so the .npz bytes are a pure
    # function of the arrays; compression would introduce a timestamp.
    np.savez(
        out,
        t_s=t,
        r_m=r,
        v_mps=v,
        dist_m=dist,
        a_m=a_m,
        ev_t_s=ev_t,
        ev_detail=ev_detail,
    )
    return {
        "mission": case["mission"],
        "central_body": run.header["central_body"],
        "config_sha256": run.header.get("config_sha256", ""),
        "run_srlog_sha256": _sha256(log_path),
        "truth_records": int(n),
        "decimated_samples": int(len(idx)),
        "npz": case["npz"],
        "npz_sha256": _sha256(out),
    }


def extract() -> None:
    """Re-run both missions and rewrite the committed figure-input .npz set."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        entries = {key: _extract_case(key, tmp) for key in _CASES}
    manifest = {
        "note": (
            "Figure-input data for the scientific-report case studies "
            "(scripts/figures/casestudy_figures.py). Regenerate with "
            "'python scripts/figures/casestudy_figures.py --extract'."
        ),
        "cases": entries,
    }
    _MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for key, e in entries.items():
        print(f"extracted {key}: {e['npz']} "
              f"({e['decimated_samples']} samples, "
              f"npz sha256 {e['npz_sha256'][:12]})")
    print(f"wrote {_MANIFEST.relative_to(_ROOT)}")


# ---------------------------------------------------------------------------
# Rendering (needs only the committed .npz inputs and matplotlib)
# ---------------------------------------------------------------------------

# Okabe-Ito colorblind-safe palette (Okabe & Ito 2008; Wong, Nature Methods
# 8:441, 2011), matching python/star_reacher/plotting.py so the report
# figures and the quicklook plots share one visual identity.
_BLUE = "#0072B2"
_ORANGE = "#E69F00"
_GREEN = "#009E73"
_VERM = "#D55E00"
_EVENT_COLOR = "0.45"


def _load_case(key: str) -> dict:
    npz = _DATA_DIR / _CASES[key]["npz"]
    if not npz.is_file():
        raise SystemExit(
            f"missing figure input {npz.relative_to(_ROOT)}; run "
            f"'python scripts/figures/casestudy_figures.py --extract' first."
        )
    with np.load(npz, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def _short_hash(key: str) -> str:
    """First 12 hex of the case's committed config SHA-256 for the caption."""
    if not _MANIFEST.is_file():
        return "unknown-config"
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    sha = manifest.get("cases", {}).get(key, {}).get("config_sha256", "")
    return sha[:12] if sha else "unknown-config"


def _mark_events(ax, ev_t, ev_detail) -> None:
    for t, detail in zip(ev_t, ev_detail):
        ax.axvline(t / 86400.0, color=_EVENT_COLOR, linewidth=0.7,
                   linestyle="--", zorder=1)
        ax.annotate(
            str(detail), xy=(t / 86400.0, 0.98),
            xycoords=ax.get_xaxis_transform(),
            rotation=90, va="top", ha="right", fontsize=6, color="0.3",
            clip_on=True,
        )


def _render_translunar(plt, data: dict, out: Path) -> None:
    """Earth-centered transfer: XY trajectory + geocentric distance vs time."""
    fig = plt.figure(figsize=(10.0, 4.6), dpi=_DPI, layout="constrained")
    ax_xy, ax_d = fig.subplots(1, 2)

    r = data["r_m"] / 1.0e6  # Mm, so LEO-to-lunar spans a readable range
    ax_xy.plot(r[:, 0], r[:, 1], color=_BLUE, linewidth=1.2, zorder=3,
               label="transfer arc")
    ax_xy.scatter([0.0], [0.0], marker="o", s=60, color=_GREEN, zorder=4,
                  label="Earth")
    ax_xy.scatter([r[0, 0]], [r[0, 1]], marker="s", s=22, color=_VERM,
                  zorder=5, label="TLI parking state")
    ax_xy.scatter([r[-1, 0]], [r[-1, 1]], marker="v", s=26, color=_ORANGE,
                  zorder=5, label="lunar SOI crossing")
    ax_xy.set_aspect("equal")
    ax_xy.grid(True, linewidth=0.4, alpha=0.35)
    ax_xy.tick_params(labelsize=8)
    ax_xy.set_xlabel("GCRF x [Mm]", fontsize=9)
    ax_xy.set_ylabel("GCRF y [Mm]", fontsize=9)
    ax_xy.set_title("translunar transfer (GCRF x--y projection)", fontsize=10)
    ax_xy.legend(fontsize=7, framealpha=0.6, loc="best")

    d_earth = data["dist_m"] / 1.0e6  # Mm
    ax_d.plot(data["t_s"] / 86400.0, d_earth, color=_BLUE, linewidth=1.2,
              zorder=3)
    ax_d.grid(True, linewidth=0.4, alpha=0.35)
    ax_d.tick_params(labelsize=8)
    ax_d.set_xlabel("elapsed time [day]", fontsize=9)
    ax_d.set_ylabel("geocentric distance [Mm]", fontsize=9)
    ax_d.set_title("geocentric distance", fontsize=10)
    _mark_events(ax_d, data["ev_t_s"], data["ev_detail"])

    fig.savefig(out, dpi=_DPI, metadata={"Software": None})
    plt.close(fig)


def _render_mars(plt, data: dict, out: Path) -> None:
    """Heliocentric arc: XY trajectory + osculating semi-major axis vs time."""
    fig = plt.figure(figsize=(10.0, 4.6), dpi=_DPI, layout="constrained")
    ax_xy, ax_a = fig.subplots(1, 2)

    au = 1.495978707e11
    r = data["r_m"] / au
    ax_xy.plot(r[:, 0], r[:, 1], color=_ORANGE, linewidth=1.4, zorder=3,
               label="cruise arc (7 d)")
    ax_xy.scatter([0.0], [0.0], marker="o", s=70, color="#F0C000", zorder=4,
                  label="Sun")
    ax_xy.scatter([r[0, 0]], [r[0, 1]], marker="s", s=22, color=_VERM,
                  zorder=5, label="departure handoff")
    ax_xy.scatter([r[-1, 0]], [r[-1, 1]], marker="v", s=26, color=_BLUE,
                  zorder=5, label="window end (t = 7 d)")
    ax_xy.set_aspect("equal")
    ax_xy.grid(True, linewidth=0.4, alpha=0.35)
    ax_xy.tick_params(labelsize=8)
    ax_xy.set_xlabel("heliocentric x [au]", fontsize=9)
    ax_xy.set_ylabel("heliocentric y [au]", fontsize=9)
    ax_xy.set_title("Earth--Mars cruise (heliocentric x--y projection)",
                    fontsize=10)
    ax_xy.legend(fontsize=7, framealpha=0.6, loc="best")

    ax_a.plot(data["t_s"] / 86400.0, data["a_m"] / au, color=_ORANGE,
              linewidth=1.4, zorder=3)
    ax_a.grid(True, linewidth=0.4, alpha=0.35)
    ax_a.tick_params(labelsize=8)
    ax_a.set_xlabel("elapsed time [day]", fontsize=9)
    ax_a.set_ylabel("osculating semi-major axis [au]", fontsize=9)
    ax_a.set_title("heliocentric osculating semi-major axis", fontsize=10)

    fig.savefig(out, dpi=_DPI, metadata={"Software": None})
    plt.close(fig)


_RENDERERS = {
    "translunar": _render_translunar,
    "mars_transfer": _render_mars,
}


def render(outdir: Path | None = None) -> list[Path]:
    """Render both case-study PNGs from the committed .npz inputs."""
    # A fixed SOURCE_DATE_EPOCH fallback so a bare render is reproducible even
    # outside the docs build; a caller-set value (CI, star docs) is honored.
    os.environ.setdefault("SOURCE_DATE_EPOCH", _FIXED_EPOCH)
    # Agg before pyplot, unconditionally: the render must never touch a
    # display and must not inherit another process's backend choice.
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    out = _FIG_DIR if outdir is None else Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for key, case in _CASES.items():
        data = _load_case(key)
        path = out / case["png"]
        _RENDERERS[key](plt, data, path)
        written.append(path)
        print(f"rendered {path.name} (config {_short_hash(key)})")
    return written


def verify_repro() -> int:
    """Render twice to fresh dirs and assert the PNGs are byte-identical.

    This is the local proof of Phase 8 exit criterion 3: with the pinned
    matplotlib version and SOURCE_DATE_EPOCH set, regenerating the figures
    reproduces them bit-for-bit. Returns a process exit code.
    """
    os.environ.setdefault("SOURCE_DATE_EPOCH", _FIXED_EPOCH)
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        first = render(Path(a))
        second = render(Path(b))
        ok = True
        for p1, p2 in zip(first, second):
            h1, h2 = _sha256(p1), _sha256(p2)
            match = h1 == h2
            ok = ok and match
            print(f"{'MATCH' if match else 'MISMATCH'}  {p1.name}  {h1}")
            if not match:
                print(f"                    second run  {h2}")
    if ok:
        print("crit-3: all case-study figures reproduce bit-for-bit")
        return 0
    print("crit-3: FAILED --- figures are not bit-reproducible", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--extract", action="store_true",
                   help="re-run the missions and rewrite the committed .npz inputs")
    g.add_argument("--verify-repro", action="store_true",
                   help="render twice and assert the PNGs are byte-identical")
    args = ap.parse_args(argv)
    if args.extract:
        extract()
        return 0
    if args.verify_repro:
        return verify_repro()
    render()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
