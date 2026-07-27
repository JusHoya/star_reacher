"""The tli ballistic-burnout epoch is t = 354 s, one control cycle after the
commanded cutoff (D-5 zero-order hold).

This pins the invariant the Phase 8 trans-lunar cross-tool case rests on.
``missions/tli.toml`` commands ``cutoff_engine`` at t = 353 s, but the delivered
thrust does not vanish at that instant: ``advance_cycle`` in
``cpp/src/vehicle_cycle.cpp`` runs the RK4 translational step and only afterward
advances the engine states, so the [353, 354) step still integrates at the
pre-command throttle level and t = 354 s is the first ballistic epoch. The
mathlib "6DOF vehicle" chapter documents that per-cycle ordering and the D-5
zero-order hold; the propulsion chapter's spool model is not what decides this
(with ``spool_time_s`` = 0.5 s against a 1 s cycle the level clamps to zero in a
single advance, so the answer would be 354 s for any spool time below the step).

Why this test exists. ``tests/golden/crosstool/gmat_translunar.script`` starts
its GMAT coast from the t = 354 s truth state, and the frozen truth
``truth_gmat_translunar.csv`` is only valid for that state. The first freeze of
that case used t = 353 s -- a mid-burn state -- and missed lunar arrival by
75,189 km, costing a full maintainer round trip. Moving ``engine_advance`` ahead
of ``rk4.step`` would silently restore that defect: the mission would still run,
every existing propulsion unit test would still pass (PROP-SPOOL-RAMP exercises
``engine_advance`` in isolation and never touches the run loop), and only the
cross-tool gate would fail, on a machine that needs GMAT to diagnose it. This
test fails immediately instead, in the same terms.

The discriminator is taken from the truth log alone -- no gravitational
constant, no force-model reconstruction -- because ``tli.toml`` sets
``forces_rate_hz = 0`` and logs no by-source thrust channel. Under thrust the
per-second velocity increment is ~16.6 m/s and rising smoothly; ballistic it is
the ~8.8 m/s gravitational increment. The step under test is unambiguous.

Requires the compiled core and fails, never skips, without it.
"""

import re
from pathlib import Path

import numpy as np

from star_reacher.runner import run_mission

import star_reacher

REPO_ROOT = Path(__file__).resolve().parents[2]

# The meco event epoch in missions/tli.toml, and the ballistic burnout one
# control cycle later. dt_s = 1.0 and truth_rate_hz = 1, so every epoch below is
# an exact integer-second row of the truth log.
MECO_COMMAND_T_S = 353.0
BURNOUT_T_S = 354.0

# Propagating the full 600000 s coast to inspect four seconds around MECO is
# wasteful; the burn and its cutoff are complete well before this.
_SHORT_DURATION_S = 400.0


def _run_short_tli(tmp_path: Path):
    """Run tli.toml truncated to the burn, and return its truth group."""
    text = (REPO_ROOT / "missions" / "tli.toml").read_text(encoding="utf-8")
    text, n = re.subn(
        r"^duration_s = .*$",
        f"duration_s = {_SHORT_DURATION_S}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    assert n == 1, "missions/tli.toml no longer has a single duration_s line"
    mission = tmp_path / "tli_short.toml"
    mission.write_text(text, encoding="utf-8")
    # The mission's vehicle and ephemeris paths are CWD-relative, so the copy
    # resolves them exactly as the committed mission does when run from root.
    result = run_mission(mission, tmp_path / "run")
    return star_reacher.load(result.srlog_path).groups["truth"]


def _velocity_increment(truth, t_s: float) -> float:
    """|v(t+1) - v(t)|: the total delta-v delivered over the cycle at t_s."""
    t = truth["t_s"]
    i = int(np.searchsorted(t, t_s))
    assert t[i] == t_s and t[i + 1] == t_s + 1.0, (
        f"the tli truth log lacks exact integer-second rows at {t_s} and "
        f"{t_s + 1.0} s; this test assumes dt_s = 1 and truth_rate_hz = 1"
    )
    return float(np.linalg.norm(truth["v_mps"][i + 1] - truth["v_mps"][i]))


def test_cutoff_delivers_a_full_thrust_step_after_the_command(tmp_path):
    """The cycle commanded at MECO still delivers thrust; the next one does not.

    Directly asserts the ordering: a cutoff commanded at t = 353 s leaves the
    [353, 354) integration at the pre-command throttle, so its velocity
    increment stays on the burn trend, and only [354, 355) is gravitational.
    """
    truth = _run_short_tli(tmp_path)

    in_burn = _velocity_increment(truth, MECO_COMMAND_T_S - 1.0)  # [352, 353)
    at_cutoff = _velocity_increment(truth, MECO_COMMAND_T_S)  # [353, 354)
    ballistic = _velocity_increment(truth, BURNOUT_T_S)  # [354, 355)
    after = _velocity_increment(truth, BURNOUT_T_S + 1.0)  # [355, 356)

    print(
        f"tli velocity increments (m/s): [352,353) {in_burn:.6f}, "
        f"[353,354) {at_cutoff:.6f}, [354,355) {ballistic:.6f}, "
        f"[355,356) {after:.6f}"
    )

    # The commanded-cutoff cycle is a full-thrust cycle: it continues the burn
    # trend rather than dropping to the gravitational increment. The burn's
    # increment grows slowly as the stage lightens, so a 2 % band around the
    # preceding cycle is tight while carrying that drift.
    assert at_cutoff == 0 or abs(at_cutoff / in_burn - 1.0) < 0.02, (
        f"the cycle commanded at MECO (t = {MECO_COMMAND_T_S} s) delivered a "
        f"{at_cutoff:.6f} m/s increment against {in_burn:.6f} m/s for the "
        f"preceding burn cycle: thrust appears to stop AT the command instead "
        f"of one control cycle later. If engine_advance now runs before "
        f"rk4.step, tli burnout moves back to {MECO_COMMAND_T_S} s and "
        f"tests/golden/crosstool/truth_gmat_translunar.csv (frozen from the "
        f"t = {BURNOUT_T_S} s state) is invalidated -- see this module's docstring."
    )

    # The following cycle is ballistic: gravity alone, roughly half the thrust
    # increment, and near-constant from there on.
    assert ballistic < 0.75 * at_cutoff, (
        f"the cycle at t = {BURNOUT_T_S} s delivered {ballistic:.6f} m/s, not "
        f"the expected gravity-only increment below 0.75x the {at_cutoff:.6f} "
        f"m/s thrust cycle: burnout is not at {BURNOUT_T_S} s"
    )
    assert abs(after / ballistic - 1.0) < 0.01, (
        f"the increments after burnout ({ballistic:.6f}, {after:.6f} m/s) are "
        f"not the near-constant gravitational signature of a ballistic coast"
    )


def test_specific_energy_is_constant_from_burnout(tmp_path):
    """Two-body specific orbital energy stops rising exactly at t = 354 s.

    The independent check on the same invariant, in the quantity the trans-lunar
    freeze actually cares about: the energy of the transfer ellipse. Computed
    with the energy's own gravitational parameter cancelled out by differencing,
    so no constant enters -- the test compares the step-to-step change in
    |v|^2/2 against the change across the coast.
    """
    truth = _run_short_tli(tmp_path)
    t = truth["t_s"]
    r = truth["r_m"]
    v = truth["v_mps"]

    def energy_change(t_s: float) -> float:
        """d(|v|^2/2) - d(mu/r) reduced to what a ballistic step conserves.

        Over one second the two-body specific energy |v|^2/2 - mu/|r| is
        constant for a ballistic step (to the integrator's accuracy and the
        third-body perturbation, both negligible here) and jumps under thrust.
        Differencing consecutive steps removes mu entirely: the ratio of the
        kinetic-term change to the potential-term change is what separates them.
        """
        i = int(np.searchsorted(t, t_s))
        dke = 0.5 * (
            float(np.dot(v[i + 1], v[i + 1])) - float(np.dot(v[i], v[i]))
        )
        # 1/|r| change carries the potential term up to the factor mu.
        dinv_r = 1.0 / float(np.linalg.norm(r[i + 1])) - 1.0 / float(
            np.linalg.norm(r[i])
        )
        return dke, dinv_r

    # For a ballistic step d(|v|^2/2) = mu * d(1/|r|) exactly, so the implied mu
    # is the real Earth GM. Under thrust the same ratio is far off it.
    dke_cut, dinv_cut = energy_change(MECO_COMMAND_T_S)
    dke_coast, dinv_coast = energy_change(BURNOUT_T_S)
    dke_coast2, dinv_coast2 = energy_change(BURNOUT_T_S + 1.0)

    mu_coast = dke_coast / dinv_coast
    mu_coast2 = dke_coast2 / dinv_coast2
    mu_cut = dke_cut / dinv_cut
    print(
        f"implied mu (m^3/s^2): [353,354) {mu_cut:.6e}, [354,355) "
        f"{mu_coast:.6e}, [355,356) {mu_coast2:.6e}"
    )

    # The two post-burnout cycles agree on a consistent gravitational parameter.
    assert abs(mu_coast / mu_coast2 - 1.0) < 1e-3, (
        f"the two cycles after t = {BURNOUT_T_S} s imply inconsistent "
        f"gravitational parameters ({mu_coast:.6e}, {mu_coast2:.6e}); they are "
        f"not both ballistic"
    )
    # The commanded-cutoff cycle does not: it is still adding energy, so its
    # implied parameter is nowhere near the ballistic one.
    assert abs(mu_cut / mu_coast - 1.0) > 0.5, (
        f"the cycle commanded at MECO (t = {MECO_COMMAND_T_S} s) conserves "
        f"two-body energy like a ballistic step (implied mu {mu_cut:.6e} vs "
        f"{mu_coast:.6e} after burnout), so thrust stopped AT the command. "
        f"Burnout would then be {MECO_COMMAND_T_S} s, invalidating the frozen "
        f"trans-lunar cross-tool truth -- see this module's docstring."
    )
