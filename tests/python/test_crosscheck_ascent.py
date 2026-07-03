"""Phase 4 EC-11: independent 3DOF cross-check of the 6DOF ascent.

EC-11 requires the ascent insertion orbit to match an independently coded 3DOF
point-mass ascent flying the same pitch program to within 2 % on insertion
apogee and perigee altitudes, with max-q altitude, Mach-1 altitude, and burnout
velocity inside representative Electron-class sanity bands.

This test drives two fully independent simulators of missions/ascent_leo.toml and
compares their insertion orbits at the EXACT perigee crossing:

* the project's own 6DOF, run as a BLACK BOX (propagated through the compiled
  core via ``run_mission``; its truth trajectory is reduced to osculating
  apsides). No 6DOF model source is read by the cross-check.
* an independent 3DOF point-mass oracle, ``tests/crosscheck/ascent_3dof.py``,
  which shares no code with the 6DOF and reimplements its own RK4 integrator,
  U.S. Standard Atmosphere 1976, spherical point-mass gravity, vacuum-Isp
  back-pressure thrust, staged mass model, and osculating reduction. It parses
  only the vehicle and mission TOML. Provenance and every tolerance derivation
  are in ``tests/crosscheck/manifest.toml``.

Both trajectories are interpolated to the same exact osculating perigee of
180.000 km (the 6DOF's insertion-gate condition) before comparison, so the
perigee agreement is exact by construction and the apogee comparison reflects
only the true insertion-energy difference, not the discrete step at which each
run trips the gate. The ascent inserts into an eccentric 180 x 3444 km orbit
(missions/ascent_leo.toml): an eccentric apogee is well conditioned, so the
small insertion-energy difference between the two independent simulators (~0.3 %
in semi-major axis) maps to a sub-2 % apogee-altitude difference rather than the
~30x amplification a near-circular insertion would impose.

The test REQUIRES the compiled core (the 6DOF reference is regenerated live) and
fails-with-hint, never skips, when the core is absent, per the project's
agent-honesty gate (see ``test_integration_core.py``).
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# The independent 3DOF oracle lives under tests/crosscheck (outside the package).
sys.path.insert(0, str(REPO_ROOT / "tests" / "crosscheck"))
import ascent_3dof  # noqa: E402

MISSION = REPO_ROOT / "missions" / "ascent_leo.toml"
VEHICLE = REPO_ROOT / "vehicles" / "electron_class.toml"

# The 6DOF insertion gate: both trajectories are compared at this exact perigee.
INSERTION_PERIGEE_M = 180000.0

# Literal EC-11 tolerances on BOTH insertion apsides.
APOGEE_REL_TOL = 0.02
PERIGEE_REL_TOL = 0.02
# Insertion inertial speed (EC-11 "burnout velocity"), the well-conditioned
# cross-check; agrees far tighter than this in practice.
SPEED_REL_TOL = 0.005

# Representative Electron-class sanity bands (generous; see manifest.toml).
MAXQ_ALT_BAND_KM = (8.0, 15.0)
MACH1_ALT_BAND_KM = (2.0, 10.0)
# Burnout is the perigee velocity of the eccentric 180 x 3444 km insertion
# (higher than a circular-LEO burnout); band spans circular LEO to this eccentric
# insertion from the vis-viva equation.
BURNOUT_SPEED_BAND_KMPS = (7.4, 8.8)

# Mutation check: a wrong 3DOF must fail the apogee gate (the gate is not vacuous).
MUTATION_THRUST_SCALE = 1.03  # +3 % stage-2 thrust -> ~+17 % apogee, well past 2 %

_CORE_MISSING_MESSAGE = (
    "star_reacher._core is not built in this environment. This EC-11 cross-check "
    "regenerates the 6DOF reference as a black box and so requires the compiled "
    "core: build and install it with 'pip install .' from the repository root. "
    "This failure is expected on a core-less checkout and must be green at CI."
)


def _osculating_at_perigee_6dof(srlog_path):
    """Reduce the 6DOF truth to (apo, per, speed) at the exact perigee crossing."""
    from star_reacher import load

    run = load(srlog_path)
    truth = run.groups["truth"]
    samples = (
        (float(truth["t_s"][i]), truth["r_m"][i], truth["v_mps"][i])
        for i in range(len(truth["t_s"]))
    )
    hit = ascent_3dof.interpolate_to_perigee(samples, INSERTION_PERIGEE_M)
    if hit is None:  # gate never crossed in the log; use the terminal state
        r = tuple(float(x) for x in truth["r_m"][-1])
        v = tuple(float(x) for x in truth["v_mps"][-1])
    else:
        r, v, _ = hit
    apo, per, _, _ = ascent_3dof.osculating_apsides(r, v)
    return apo, per, ascent_3dof._norm(v)


@pytest.fixture(scope="module")
def ref_6dof(tmp_path_factory):
    """Run the 6DOF ascent once as a black box; share the reduced reference."""
    try:
        from star_reacher import _core  # noqa: F401
    except ImportError:
        pytest.fail(_CORE_MISSING_MESSAGE)
    from star_reacher.runner import run_mission

    outdir = tmp_path_factory.mktemp("ref6dof")
    result = run_mission(MISSION, outdir)
    apo, per, speed = _osculating_at_perigee_6dof(result.srlog_path)
    return {"apo": apo, "per": per, "speed": speed}


def test_ec11_ascent_crosscheck(ref_6dof):
    apo6, per6, speed6 = ref_6dof["apo"], ref_6dof["per"], ref_6dof["speed"]

    ins = ascent_3dof.run_to_perigee(VEHICLE, MISSION, INSERTION_PERIGEE_M, dt=0.02)
    assert ins.result.reached_insertion, (
        "independent 3DOF did not reach the perigee >= 180 km insertion gate; "
        f"final perigee {ins.periapsis_alt_m / 1e3:.1f} km, "
        f"apogee {ins.apoapsis_alt_m / 1e3:.1f} km"
    )

    apo3, per3, speed3 = ins.apoapsis_alt_m, ins.periapsis_alt_m, ins.speed_mps
    apo_rel = abs(apo3 - apo6) / apo6
    per_rel = abs(per3 - per6) / per6
    speed_rel = abs(speed3 - speed6) / speed6

    summary = (
        "EC-11 3DOF-vs-6DOF at exact perigee = 180.000 km (measured):\n"
        f"  apogee  alt : 3DOF {apo3/1e3:8.2f} km | 6DOF {apo6/1e3:8.2f} km | {100*(apo3-apo6)/apo6:+.2f} %\n"
        f"  perigee alt : 3DOF {per3/1e3:8.3f} km | 6DOF {per6/1e3:8.3f} km | {100*(per3-per6)/per6:+.4f} %\n"
        f"  insertion V : 3DOF {speed3:8.1f} m/s | 6DOF {speed6:8.1f} m/s | {100*(speed3-speed6)/speed6:+.3f} %\n"
        f"  max-q       : {ins.result.maxq_pa/1e3:.1f} kPa at {ins.result.maxq_alt_m/1e3:.2f} km, Mach {ins.result.maxq_mach:.2f}\n"
        f"  Mach-1 alt  : {ins.result.mach1_alt_m/1e3:.2f} km\n"
        f"  s2 residual : {ins.result.s2_prop_residual_kg:.1f} kg"
    )
    print("\n" + summary)

    # --- Literal EC-11: 2 % on BOTH apsides (perigee is exact by construction).
    assert apo_rel <= APOGEE_REL_TOL, (
        f"insertion apogee altitude differs {100*apo_rel:.2f} % (> {100*APOGEE_REL_TOL:.0f} %)\n{summary}"
    )
    assert per_rel <= PERIGEE_REL_TOL, (
        f"insertion perigee altitude differs {100*per_rel:.4f} % (> {100*PERIGEE_REL_TOL:.0f} %)\n{summary}"
    )
    # Insertion inertial speed (burnout velocity) cross-check, well conditioned.
    assert speed_rel <= SPEED_REL_TOL, (
        f"insertion inertial speed differs {100*speed_rel:.3f} % (> {100*SPEED_REL_TOL:.1f} %)\n{summary}"
    )

    # --- Sanity bands (membership, not equality): a grossly wrong ascent fails.
    maxq_alt_km = ins.result.maxq_alt_m / 1e3
    assert MAXQ_ALT_BAND_KM[0] <= maxq_alt_km <= MAXQ_ALT_BAND_KM[1], (
        f"max-q altitude {maxq_alt_km:.2f} km outside band {MAXQ_ALT_BAND_KM} km\n{summary}"
    )
    mach1_alt_km = ins.result.mach1_alt_m / 1e3
    assert MACH1_ALT_BAND_KM[0] <= mach1_alt_km <= MACH1_ALT_BAND_KM[1], (
        f"Mach-1 altitude {mach1_alt_km:.2f} km outside band {MACH1_ALT_BAND_KM} km\n{summary}"
    )
    burnout_kmps = speed3 / 1e3
    assert BURNOUT_SPEED_BAND_KMPS[0] <= burnout_kmps <= BURNOUT_SPEED_BAND_KMPS[1], (
        f"burnout velocity {burnout_kmps:.3f} km/s outside band {BURNOUT_SPEED_BAND_KMPS} km/s\n{summary}"
    )


def test_ec11_gate_rejects_wrong_physics(ref_6dof, monkeypatch):
    """Mutation check: a deliberately wrong 3DOF must fail the 2 % apogee gate.

    A green acceptance gate proves nothing until it is shown it can fail. Scaling
    the stage-2 vacuum thrust +3 % is a physics error the gate must catch; it
    perturbs the insertion energy well outside the near-2 % apogee band.
    """
    apo6 = ref_6dof["apo"]
    real_load_vehicle = ascent_3dof.load_vehicle

    def mutated_load_vehicle(path):
        veh = real_load_vehicle(path)
        veh.s2_engine.thrust_vac_n *= MUTATION_THRUST_SCALE
        return veh

    monkeypatch.setattr(ascent_3dof, "load_vehicle", mutated_load_vehicle)
    ins = ascent_3dof.run_to_perigee(VEHICLE, MISSION, INSERTION_PERIGEE_M, dt=0.05)
    apo_rel = abs(ins.apoapsis_alt_m - apo6) / apo6
    assert apo_rel > APOGEE_REL_TOL, (
        f"mutation check FAILED to fail: a +{100*(MUTATION_THRUST_SCALE-1):.0f} % stage-2 "
        f"thrust error moved the apogee only {100*apo_rel:.2f} % (<= {100*APOGEE_REL_TOL:.0f} %), "
        f"so the EC-11 apogee gate is not discriminating."
    )
