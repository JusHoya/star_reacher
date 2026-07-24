"""Regenerate the README example-gallery images from committed missions.

The gallery is a *documentation* artifact (PRD FR-31 "example gallery",
Phase 8 deliverable), not a regression instrument: it exists so a reader sees
representative ``star plot`` / ``star view`` output before installing anything.
Every image here is reproducible from a committed mission by a single
documented command, and this script is the one-shot regenerator that runs
those commands and copies the results into ``docs/gallery/`` under stable
names. The commands it runs are the same ones the README and the fresh-machine
walkthrough (``docs/walkthrough.md``) tell a user to run; keeping them in one
executable place is why the gallery cannot drift from the documented CLI.

Determinism: the ``star plot`` PNG bytes and the 3D-trajectory PNG below are a
pure function of the log bytes and the pinned matplotlib version (fixed figure
geometry, fixed PNG metadata, no timestamps) -- the FR-21 discipline applied to
a derived artifact, the same rule ``star_reacher.plotting`` already follows.
The one non-``star plot`` figure (the cislunar 3D trajectory) is a static
preview of what the interactive ``star view`` HTML plays back, rendered from
the *same* truth channel the viewer decimates; it is a still, not a substitute
for the live viewer, and the gallery links the live viewer alongside it.

Regenerating requires the compiled core (the missions must run) and matplotlib
(a mandatory runtime dependency, D-12):

    .venv312/Scripts/python docs/gallery/generate.py

Images are written into docs/gallery/. The three source missions --
missions/leo_gravity_8x8.toml (inclined LEO), missions/ascent_leo.toml
(scripted ascent), and missions/mission_a_cislunar.toml (trans-lunar coast) --
all run on a clean clone with no fetched data (the cislunar mission uses the
committed DE440 ephemeris excerpt).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
# Same rule as tests/golden/plots/generate.py: prefer an installed wheel's
# compiled _core; fall back to the source tree only when no package is
# installed, so a stale source core never shadows the built binary.
if importlib.util.find_spec("star_reacher") is None:
    sys.path.insert(0, str(REPO_ROOT / "python"))

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from star_reacher import load  # noqa: E402

# The installed console script drives the same code path a user runs; calling
# it as a subprocess (rather than importing the plotting module directly)
# keeps this generator honest -- it exercises exactly the documented command.
STAR = REPO_ROOT / ".venv312" / "Scripts" / "star.exe"
if not STAR.exists():  # non-Windows or a differently-named venv
    STAR = "star"


def _run(args: list[str]) -> None:
    """Run a ``star`` subcommand, echoing it, and fail loudly on nonzero exit."""
    cmd = [str(STAR), *args]
    print("$ star " + " ".join(args))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def _plot_to_gallery(srlog: Path, plot: str, dest_name: str, tmp: Path) -> None:
    """Render one named ``star plot`` figure and copy it under a stable name."""
    out = tmp / dest_name
    _run(["plot", str(srlog), "--plots", plot, "-o", str(out)])
    src = out / f"{plot}.png"
    (HERE / dest_name).write_bytes(src.read_bytes())
    print(f"  -> docs/gallery/{dest_name}")


def _trajectory_3d(srlog: Path, dest_name: str) -> None:
    """Static 3D still of the trajectory the interactive viewer plays back.

    Read from the truth channel through the pure-NumPy loader (no re-simulation,
    exactly what ``star view`` consumes), stride-decimated with a fixed stride
    so the PNG stays a deterministic function of the log. Earth is drawn at its
    WGS-84 display radius purely for scale, never as dynamics.
    """
    run = load(srlog)
    r = np.asarray(run.groups["truth"]["r_m"], dtype=np.float64) / 1000.0  # km
    stride = max(1, len(r) // 4000)  # cap the polyline; deterministic
    r = r[::stride]

    fig = plt.figure(figsize=(7.5, 7.0), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        r[:, 0], r[:, 1], r[:, 2],
        color="#0072B2", linewidth=1.0, label="trans-lunar trajectory (GCRF)",
    )
    # Earth display sphere at the origin (WGS-84 semi-major axis, km).
    radius_km = 6378.137
    u = np.linspace(0.0, 2.0 * np.pi, 30)
    v = np.linspace(0.0, np.pi, 15)
    xs = radius_km * np.outer(np.cos(u), np.sin(v))
    ys = radius_km * np.outer(np.sin(u), np.sin(v))
    zs = radius_km * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, color="#4a6fa5", alpha=0.45, linewidth=0, zorder=1)
    ax.scatter([r[0, 0]], [r[0, 1]], [r[0, 2]], color="#009E73", s=25, label="start (TLI cutoff)")
    ax.scatter([r[-1, 0]], [r[-1, 1]], [r[-1, 2]], color="#D55E00", s=25, label="end (4.5 d)")
    ax.set_xlabel("x [km]", fontsize=8)
    ax.set_ylabel("y [km]", fontsize=8)
    ax.set_zlabel("z [km]", fontsize=8)
    ax.set_title("Mission A cislunar transfer (GCRF, Earth-centered)", fontsize=10)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="upper left")
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:  # older matplotlib without box_aspect on 3D axes
        pass
    # Fixed view angle so the PNG is reproducible across regenerations.
    ax.view_init(elev=24, azim=-58)
    dest = HERE / dest_name
    fig.savefig(dest, dpi=100, metadata={"Software": "star_reacher"})
    plt.close(fig)
    print(f"  -> docs/gallery/{dest_name}")


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        # 1) Inclined LEO, first ~6 orbits (a documented --set duration override,
        #    the DX-4 edit-run knob) -> classic sinusoidal groundtrack + the six
        #    osculating elements showing J2 short-period and secular structure.
        leo = tmp / "leo"
        _run([
            "run", "missions/leo_gravity_8x8.toml",
            "--set", "mission.duration_s=18000",
            "-o", str(leo), "--force",
        ])
        leo_log = leo / "run.srlog"
        _plot_to_gallery(leo_log, "groundtrack", "groundtrack_inclined_leo.png", tmp)
        _plot_to_gallery(leo_log, "elements", "elements_inclined_leo.png", tmp)

        # 2) Scripted ascent -> the rich launch panels: altitude/speed with
        #    staging event ticks, the per-source force/torque budget, mass/thrust.
        asc = tmp / "ascent"
        _run(["run", "missions/ascent_leo.toml", "-o", str(asc), "--force"])
        asc_log = asc / "run.srlog"
        _plot_to_gallery(asc_log, "altitude_speed", "altitude_speed_ascent.png", tmp)
        _plot_to_gallery(asc_log, "forces_by_source", "forces_by_source_ascent.png", tmp)
        _plot_to_gallery(asc_log, "mass_thrust_throttle", "mass_thrust_ascent.png", tmp)

        # 3) Cislunar coast -> a static 3D trajectory still previewing the
        #    interactive `star view` HTML (which the gallery also links live).
        cis = tmp / "cislunar"
        _run(["run", "missions/mission_a_cislunar.toml", "-o", str(cis), "--force"])
        _trajectory_3d(cis / "run.srlog", "trajectory3d_cislunar.png")

    print("gallery regenerated into docs/gallery/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
