"""Run the Orekit cross-tool cases and export fixed-cadence GCRF truth CSVs.

Maintainer-side only (D-15): runs on the pinned portable toolchain
(orekit-venv: orekit_jpype 13.1.5.0 = Orekit 13.1.5, Temurin 21 JDK) against
the curated zero-EOP data directory assembled by
``build_orekit_zeroeop_data.py``. CI never runs this; CI consumes only the
committed truth CSV.

Cases (both replicate the committed missions; replication specification in
tests/golden/crosstool/README.md):

- ``drag``  missions/leo_drag_hp.toml -- EGM2008 8x8 (the committed
            earth_egm2008_8x8.gfc) + Harris-Priester drag (Orekit's built-in
            mean-activity table, cosine exponent n = 4, WGS84 ellipsoid,
            DE440 Sun). THE frozen baseline for exit-criterion gate 2;
            output committed as truth_orekit_leo_drag_hp.csv.
- ``grav``  missions/leo_gravity_8x8.toml -- gravity only. Informational
            corroboration of the GMAT baseline (D-15 names Orekit the
            tie-breaker); output stays in the run directory, only its RMS
            numbers are recorded in the manifest.

Controlled-comparison configuration (documented in the manifest): the
curated data directory carries a zeroed finals2000A.all, so the IERS-2010
ITRF chain evaluates with polar motion = 0, UT1 = UTC, dX = dY = 0 --
the closest legitimate Orekit equivalent of the simulator's no-EOP
convention -- and ``simpleEOP=True`` disables Orekit's sub-daily tidal EOP
interpolation model, which the simulator's chain does not carry either.
The script asserts all of this at startup and refuses to run otherwise.
Remaining chain difference (IAU 2000A vs the simulator's 2000B nutation,
~1 mas) is an irreducible tool difference and part of what the criterion
measures.

Usage (repo root):

    C:/Users/hoyer/WorkSpace/tools/orekit-venv/Scripts/python.exe \
        scripts/crosstool/run_orekit.py --case drag [--dp 1e-6]

``--dp`` is the position tolerance handed to Orekit's tolerance provider
(default 1e-6 m, mirroring the mission integrator's atol_pos_m); rerunning
with a coarser value gives the integrator-convergence evidence recorded in
the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CROSSTOOL = REPO_ROOT / "tests" / "golden" / "crosstool"
RUN_DIR = Path(r"C:\Users\hoyer\WorkSpace\tools\crosstool-runs")
JDK_HOME = r"C:\Users\hoyer\WorkSpace\tools\jdk"
DATA_DIR = r"C:\Users\hoyer\WorkSpace\tools\orekit-data-zeroeop"

STEP_S = 60.0
DURATION_S = 604800.0
N_STEPS = int(DURATION_S / STEP_S)  # 10080; output rows = N_STEPS + 1

CASES = {
    # name: (r0 [m], v0 [m/s], drag?, output path)
    "grav": (
        (7000000.0, 0.0, 0.0),
        (0.0, 6900.0, 3000.0),
        False,
        RUN_DIR / "orekit_leo_gravity_8x8.csv",
    ),
    "drag": (
        (6878000.0, 0.0, 0.0),
        (0.0, 7350.0, 2000.0),
        True,
        CROSSTOOL / "truth_orekit_leo_drag_hp.csv",
    ),
}

MASS_KG = 500.0
DRAG_AREA_M2 = 1.0
DRAG_CD = 2.2  # area * cd / mass = 0.0044 m^2/kg, the mission's cd_a_over_m
WGS84_A_M = 6378137.0
WGS84_F = 1.0 / 298.257223563


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=sorted(CASES), required=True)
    ap.add_argument("--dp", type=float, default=1e-6)
    ap.add_argument("--out", type=Path, default=None,
                    help="override output path (for convergence reruns)")
    args = ap.parse_args()
    r0, v0, with_drag, out_path = CASES[args.case]
    if args.out is not None:
        out_path = args.out

    # Per-process only; orekit_jpype.initVM needs JAVA_HOME even with jvmpath.
    os.environ["JAVA_HOME"] = JDK_HOME
    import orekit_jpype

    orekit_jpype.initVM(jvmpath=JDK_HOME + r"\bin\server\jvm.dll")
    from orekit_jpype.pyhelpers import setup_orekit_data

    setup_orekit_data(filenames=DATA_DIR, from_pip_library=False)

    from org.hipparchus.geometry.euclidean.threed import Vector3D
    from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
    from org.orekit.bodies import CelestialBodyFactory, OneAxisEllipsoid
    from org.orekit.forces.drag import DragForce, IsotropicDrag
    from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel
    from org.orekit.forces.gravity.potential import GravityFieldFactory
    from org.orekit.frames import FramesFactory
    from org.orekit.models.earth.atmosphere import HarrisPriester
    from org.orekit.orbits import CartesianOrbit, OrbitType
    from org.orekit.propagation import SpacecraftState, ToleranceProvider
    from org.orekit.propagation.numerical import NumericalPropagator
    from org.orekit.time import AbsoluteDate, TimeScalesFactory
    from org.orekit.utils import IERSConventions, PVCoordinates

    utc = TimeScalesFactory.getUTC()
    epoch = AbsoluteDate(2026, 1, 1, 0, 0, 0.0, utc)  # mission epoch_utc
    gcrf = FramesFactory.getGCRF()

    # Controlled-comparison preconditions: the curated data directory must
    # actually produce the zero-EOP chain; refuse to freeze anything else.
    eop = FramesFactory.getEOPHistory(IERSConventions.IERS_2010, True)
    pole = eop.getPoleCorrection(epoch)
    nut = eop.getNonRotatinOriginNutationCorrection(epoch)
    dut1 = eop.getUT1MinusUTC(epoch)
    assert dut1 == 0.0 and pole.getXp() == 0.0 and pole.getYp() == 0.0, (
        f"EOP not zeroed: dUT1={dut1}, xp={pole.getXp()}, yp={pole.getYp()}"
    )
    assert nut[0] == 0.0 and nut[1] == 0.0, "CIP corrections not zeroed"
    assert utc.offsetFromTAI(epoch).toDouble() == -37.0, "TAI-UTC != 37 s"
    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)

    provider = GravityFieldFactory.getNormalizedProvider(8, 8)
    assert provider.getMu() == 398600441500000.0, provider.getMu()
    assert provider.getAe() == 6378136.3, provider.getAe()
    # Spot-check the parsed field against the committed excerpt values so a
    # stray potential file in the data directory cannot slip through.
    c20 = provider.onDate(epoch).getNormalizedCnm(2, 0)
    assert c20 == -0.000484165143790815, repr(c20)

    forces = [HolmesFeatherstoneAttractionModel(itrf, provider)]  # central term included
    if with_drag:
        sun = CelestialBodyFactory.getSun()  # DE440 from the curated directory
        earth = OneAxisEllipsoid(WGS84_A_M, WGS84_F, itrf)
        atmosphere = HarrisPriester(sun, earth)  # built-in M&G table, n = 4
        forces.append(DragForce(atmosphere, IsotropicDrag(DRAG_AREA_M2, DRAG_CD)))
        rho0 = atmosphere.getDensity(epoch, Vector3D(*r0), gcrf)
        print(f"HP density at initial state: {rho0:.6e} kg/m^3")

    orbit = CartesianOrbit(
        PVCoordinates(Vector3D(*r0), Vector3D(*v0)), gcrf, epoch, provider.getMu()
    )
    tol = ToleranceProvider.getDefaultToleranceProvider(args.dp).getTolerances(
        orbit, OrbitType.CARTESIAN
    )
    integrator = DormandPrince853Integrator(1e-6, STEP_S, tol[0], tol[1])
    prop = NumericalPropagator(integrator)
    prop.setOrbitType(OrbitType.CARTESIAN)
    prop.setInitialState(SpacecraftState(orbit, MASS_KG))
    for f in forces:
        prop.addForceModel(f)

    generator = prop.getEphemerisGenerator()
    final = prop.propagate(epoch.shiftedBy(DURATION_S))
    print("final |r| [m]:", final.getPVCoordinates(gcrf).getPosition().getNorm())
    ephemeris = generator.getGeneratedEphemeris()

    rows = []
    for k in range(N_STEPS + 1):
        t = STEP_S * k
        pv = ephemeris.propagate(epoch.shiftedBy(t)).getPVCoordinates(gcrf)
        p, v = pv.getPosition(), pv.getVelocity()
        rows.append(
            "%r,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e"
            % (t, p.getX(), p.getY(), p.getZ(), v.getX(), v.getY(), v.getZ())
        )

    header = [
        "# Frozen Orekit 13.1.5 truth for missions/leo_drag_hp.toml"
        if with_drag
        else "# Orekit 13.1.5 corroboration run for missions/leo_gravity_8x8.toml",
        "# (cross-tool Phase 3 exit criterion 5, D-15). GCRF Cartesian state on",
        "# the exact 60 s grid; provenance, tool versions, command line, and",
        "# configuration hashes in manifest.toml.",
        "t_s,x_m,y_m,z_m,vx_mps,vy_mps,vz_mps",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(header + rows) + "\n", newline="\n", encoding="ascii")
    print(f"{out_path}: {out_path.stat().st_size} bytes")
    print(f"  sha256 {hashlib.sha256(out_path.read_bytes()).hexdigest()}")
    print(f"  case={args.case} dp={args.dp!r} rows={len(rows)}")


if __name__ == "__main__":
    main()
