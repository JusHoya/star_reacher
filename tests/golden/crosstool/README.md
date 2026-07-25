# Cross-tool mission definitions (Phase 3 exit criterion 5, D-15)

This directory documents the two Phase 3 cross-tool cases so an external
maintainer can configure GMAT (case 1) and Orekit (case 2) without
reverse-engineering the implementation. The mission files themselves are
`missions/leo_gravity_8x8.toml` and `missions/leo_drag_hp.toml`; both run on
a clean clone (`star run missions/<file>`) because every input they need is
committed. The frozen external truth, once generated, lands in this
directory with its own provenance manifest (D-15: GMAT and Orekit are run
offline by a maintainer; CI never installs them).

Acceptance gates (PRD Phase 3 exit criterion 5):

| Case | External baseline | Gate |
|---|---|---|
| 1. `leo-gravity-8x8` | GMAT | position RMS < 10 m over 7 days |
| 2. `leo-drag-hp` | Orekit (GMAT lacks Harris-Priester) | position RMS < 100 m over 7 days |

## Shared configuration (both cases)

- **Epoch**: 2026-01-01T00:00:00 UTC. TAI-UTC = 37 s at this epoch (IERS
  Bulletin C series, table bundled in the core); TT = TAI + 32.184 s
  exactly; TDB per the truncated Fairhead-Bretagnon series (Kaplan, USNO
  Circular 179, 2005, eq. 2.6; |error| < 30 us, irrelevant at these gates).
- **State frame**: GCRF orientation, Earth-centered. Initial states are
  osculating Cartesian in this frame, exactly as written in the TOML.
- **Earth orientation** (for the body-fixed gravity evaluation and the drag
  co-rotation): the CIO-based IAU 2006/2000B chain of IERS Conventions
  (2010) Chapter 5 — IAU 2006 precession, IAU 2000B 77-term nutation, CIO
  locator s, then R3(ERA). **Polar motion is neglected** (~0.3 urad) and
  **dUT1 = 0** (no EOP series; |UT1-UTC| < 0.9 s bounds the spin error).
  Configure the external tool the same way where possible; the residual
  orientation differences are far below both gates for these cases.
- **Duration**: 604800 s (7 days). **Truth log**: 1 Hz, GCRF position and
  velocity in `run.srlog` (read with `star_reacher.load`, or export CSV via
  `star export --csv`).
- **Integrator**: adaptive RKF7(8) (Fehlberg, NASA TR R-287), rtol = 1e-11,
  atol = 1e-6 m (position) / 1e-9 m/s (velocity), h_init = h_max = 30 s.
  Integration error is orders below both gates; any external integrator of
  comparable tightness is acceptable. Logged states are cubic-Hermite
  dense-output samples; with h_max = 30 s the interpolation error bound
  h^4/384 * max|d4r/dt4| is ~2e-2 m at LEO — negligible against both gates.
- **Spacecraft**: constant point mass 500 kg. Ballistic parameters are
  supplied pre-normalized (Cd*A/m, Cr*A/m), so only the products matter.

## Case 1 — `leo-gravity-8x8` (GMAT baseline)

Force model: Earth spherical-harmonic gravity ONLY. No third body, no SRP,
no drag.

- **Initial state** (GCRF, Cartesian):
  - r = [7000000.0, 0.0, 0.0] m
  - v = [0.0, 6900.0, 3000.0] m/s
  (perigee ~6919 km, apogee 7000 km, inclination ~23.5 deg)
- **Gravity field**: EGM2008 (Pavlis et al. 2012), truncated to degree and
  order 8, fully normalized, tide-free. **Use the EGM2008 header constants
  with the field**: GM = 3.986004415e14 m^3/s^2, R = 6378136.3 m (NOT the
  IERS TN36 GM used elsewhere in this project — each field is evaluated
  with its own self-consistent GM/R/coefficient triple).
  The exact coefficients are committed in
  `tests/golden/gravity/earth_egm2008_n20.csv` (full-precision decimal) and
  `.srgrav` (binary, what the run loads; provenance and source SHA-256 pins
  in `tests/golden/gravity/manifest.toml`). Evaluating the 20x20 excerpt
  truncated to 8x8 is bit-identical to loading an 8x8 field (doctest
  `gravity_truncation_consistency`).
- GMAT configuration notes: `EarthEGM96`-style harmonic gravity with the
  EGM2008 coefficient file, degree 8, order 8; two-body + harmonics only;
  no SRP/drag/point masses.

## Case 2 — `leo-drag-hp` (Orekit baseline)

Force model: case 1's gravity (identical field, degree/order, constants)
PLUS Harris-Priester drag. No third-body force, no SRP (the ephemeris is
used only for the Sun DIRECTION in the density bulge).

- **Initial state** (GCRF, Cartesian):
  - r = [6878000.0, 0.0, 0.0] m
  - v = [0.0, 7350.0, 2000.0] m/s
  (perigee ~6878 km, apogee ~6895 km — altitude ~500-517 km — inclination
  ~15.2 deg)
- **Drag**: cannonball, Cd*A/m = 0.0044 m^2/kg (Cd = 2.2, A = 1 m^2,
  m = 500 kg), acceleration a = -1/2 rho (Cd*A/m) |v_rel| v_rel.
- **Air-relative velocity** (FR-8): v_rel = v - omega_earth x r with
  omega_earth = 7.292115e-5 rad/s (IERS TN36, Table 1.1) about the
  Earth-fixed z-axis expressed in GCRF (third row of the GCRF->ITRF matrix).
- **Atmosphere**: Harris-Priester, Montenbruck & Gill "Satellite Orbits"
  (2000) Sect. 3.5.2 formulation and mean-solar-activity coefficient table
  (the table Orekit's `HarrisPriester` class ships; the committed
  transcription is `tests/golden/atmosphere/harris_priester_table.toml`):
  - cosine exponent n = 4.0 (the Orekit default, written explicitly in the
    mission file);
  - diurnal-bulge apex = geocentric Sun direction rotated +30 deg about the
    +z axis (right ascension advanced, declination preserved), evaluated in
    GCRF; cos(psi) is the dot product of the GCRF position unit vector with
    the apex direction;
  - altitude argument = geodetic altitude over the WGS84 ellipsoid
    (a = 6378137.0 m, 1/f = 298.257223563; NIMA TR8350.2);
  - density is zero above the 1000 km table ceiling (Orekit-compatible)
    and log-linear in altitude between table nodes.
- **Sun position**: JPL DE440 (Park et al. 2021). The run reads the
  committed continuous excerpt
  `tests/golden/ephemeris/excerpt_de440s_crosstool.sreph` (verbatim DE440
  Chebyshev records for sun/emb/earth/moon covering 2025-12-26 to
  2026-01-11 TDB; provenance in `tests/golden/ephemeris/manifest.toml`).
  Geocentric Sun = sun(SSB) - (emb(SSB) + earth(EMB)). Any DE440-family
  Sun source is equivalent at this model's sensitivity (the bulge geometry
  tolerates arcminute-level Sun-direction differences).

## Comparison procedure

Propagate the same initial state, epoch, and force model in the external
tool; sample both trajectories on the shared 1 Hz grid (or any common
subsample, e.g. 60 s); compute the RMS of the position difference
magnitudes in the shared inertial frame over the full 7 days; compare
against the case's gate. Freeze the external tool's output, versions, and
scripts in this directory with a provenance manifest when the comparison
is executed (workstream E).

## Frozen truth (2026-07-03)

Both baselines are frozen in this directory and gated in CI by
`tests/python/test_crosstool_frozen_truth.py`, which re-propagates each
mission and compares on the exact 60 s grid (10081 epochs, no
interpolation). Configuration, hashes, command lines, and the full method
record live in `manifest.toml`; the freeze scripts are
`scripts/crosstool/` (maintainer-run only; CI never installs GMAT/Orekit).

| Gate (test id) | Baseline file | Gate | Frozen measurement |
|---|---|---|---|
| XTOOL-LEO-GRAV-GMAT | `truth_gmat_leo_gravity_8x8.csv` (GMAT R2026a) | RMS < 10 m | **0.015243 m** |
| XTOOL-LEO-DRAG-OREKIT | `truth_orekit_leo_drag_hp.csv` (Orekit 13.1.5) | RMS < 100 m | **3.376229 m** |

Corroboration (informational, D-15 tie-breaker): the gravity-only case run
through Orekit under the same controlled configuration gives position RMS
0.0013130 m vs the simulator and 0.0159378 m vs GMAT, localizing the
sim-vs-GMAT residual to GMAT's FK5/IAU-76 Earth-orientation chain (the
simulator and Orekit both use IAU 2006/2000 CIO chains). Both external
tools ran with zeroed EOP (this file's "Earth orientation" convention) and
with gravity fields generated from the same committed coefficient excerpt
the missions load; a stock-EOP GMAT control run (RMS 3.140 m vs 0.015 m)
confirms the zero-EOP override was applied and material.

## Determinism note

Both missions are covered by `tests/python/test_crosstool_missions.py`:
end-to-end double runs must produce bit-identical `run.srlog` files
(D-10), the gravity-only orbit must stay bounded with a drift-free
Keplerian energy trend, and the drag case must lose energy secularly.

## Phase 8 cross-tool cases (exit criterion 1, D-15)

Five additional GMAT cases extend the table to the full validation campaign.
Their gates are PRD Phase 8 exit criterion 1 verbatim:

| Case | Mission | External baseline | Gate |
|---|---|---|---|
| Molniya | `missions/molniya.toml` | GMAT | position RMS < 100 m over 7 days |
| Lunar orbiter | `missions/lunar_orbiter.toml` | GMAT | position RMS < 100 m over 7 days |
| Mars orbiter | `missions/mars_orbiter.toml` | GMAT | position RMS < 100 m over 7 days |
| Trans-lunar | `missions/tli.toml` (coast) | GMAT | position < 1 km at lunar arrival |
| Earth-Mars cruise | `missions/mars_cruise.toml` | GMAT | position < 100 km at arrival |

Plus an illustrative LRO-ephemeris comparison (`missions/lro_illustrative.toml`,
a ~50 km LRO-class mapping orbit) carried as a **report case study only, NOT a
validation gate**: the real LRO's maneuvers and SRP/attitude history are not
modeled, so its free-flight arc diverges from the published LRO ephemeris by
construction and no tolerance is asserted.

Each case's controlled-comparison configuration follows the same discipline as
the Phase 3 cases and is recorded in `manifest.toml` and in the case's
committed `gmat_<case>.script`: identical (GM, R, coefficients) gravity fields
(the committed `earth_egm2008_8x8.cof`, `moon_grgm1200a_50x50.cof`, and
`mars_mro120f_20x20.cof`, each with its own self-consistent GM/R triple),
GMAT spherical (cannonball) SRP with Cr and area chosen so Cr*A/m matches the
mission where SRP is on, body-centred ICRF coordinate systems (EarthICRF,
LunaICRF, MarsICRF, SunICRF) as the GMAT equivalents of the mission's
body-centred GCRF frame, fixed-step RK89 at 60 s so rows land on the shared
comparison grid, and the simulator's exact t = 0 Cartesian state in each
script. The Earth-regime cases (Molniya, trans-lunar) reuse the same
zero-EOP startup override as Phase 3.

### Configuration notes per case

- **Molniya** (Earth): EGM2008 8x8 harmonic gravity plus the Sun and Luna as
  point masses (the Earth-regime third-body pair). No SRP/drag. sma 26554 km,
  ecc 0.74, inc 63.4 deg (critical), argp 270 deg — period ~11.97 h, apogee
  ~40,100 km where the third-body torque is the leading non-Keplerian signal.
- **Lunar orbiter** (Moon): GRGM1200A 50x50 harmonic gravity in the Moon
  principal-axis frame (the simulator builds that frame from the DE440 lunar
  librations — this is the case that needs the `moon_librations` excerpt
  segment, i.e. the committed `excerpt_de440s_lunar.sreph`; GMAT's own
  orientation source is covered under "Planetary ephemeris source" below),
  plus the Sun and Earth point masses (the lunar-regime pair) and cannonball
  SRP (Moon occulter). ~100 km circular polar mapping orbit, stable
  unmaneuvered over 7 days.
- **Mars orbiter** (Mars): MRO120F 20x20 harmonic gravity in the analytic
  IAU 2015 Mars body-fixed frame (no ephemeris orientation data needed), plus
  the Sun point mass and cannonball SRP (Mars occulter). ~400 km near-polar
  MRO-class science orbit.
- **Trans-lunar** (Earth coast): the ballistic coast of `tli.toml` from its
  exact t = 354 s ballistic-burnout state under Earth+Sun+Luna point masses
  (the tli coast force set), compared at the lunar-SOI arrival epoch. The
  finite TLI burn is a simulator-internal maneuver and is not replicated; the
  gate measures the transfer trajectory. Burnout is t = 354 s, not the meco
  event time 353 s: the cutoff is commanded at 353 s, but the D-5 zero-order
  hold applies it one cycle late, because the control cycle advances the RK4
  translational step before it advances the engine states (the per-cycle
  ordering in `docs/mathlib/chapters/vehicle6dof.tex`, implemented in
  `cpp/src/vehicle_cycle.cpp`), so the [353, 354) step integrates at the
  pre-command throttle level. The kick stage's `spool_time_s` = 0.5 s is
  shorter than the 1 s cycle, so no ramp is spread over steps and the level
  clamps to zero in that single advance. The truth log therefore gains
  +15.3 m/s of equivalent prograde delta-v over [353, 354) and 354 s is the
  first ballistic epoch (a first freeze from the 353 s state was invalidated
  by exactly this; see the freeze history under "Phase 8 frozen truth" below).
- **Earth-Mars cruise** (heliocentric): Sun two-body plus Earth/Luna/Venus/
  Mars/Jupiter point masses and cannonball SRP (no occulter), over the
  committed 7-day report arc. The full 259-day "arrival SOI" propagation is
  the maintainer variant noted in `gmat_mars_cruise.script`.

### Planetary ephemeris source

The two tools do not read the same planetary ephemeris, and the case `.script`
headers name the intended source rather than the one GMAT used. As-run: no
Phase 8 `.script` sets `SolarSystem.EphemerisSource`, so GMAT fell back to its
default, and its run log records the planetary source as **DE405**, loaded
from `data/planetary_ephem/spk/DE405AllPlanets.bsp`. The maintainer's portable
GMAT R2026a install ships only `leDE1941.405`, `leDE1900.421`, and
`leDE18002100.424`, so no DE440 source was selectable on it. The simulator
side reads the committed DE440 excerpts throughout. The `.script` files are
committed as-run artifacts whose SHA-256 pins `manifest.toml` carries, and the
frozen truth was generated from exactly those bytes, so they are not edited;
this note and the matching statement in `manifest.toml` are the correction of
record.

The resulting DE440-vs-DE405 difference in third-body positions and in GMAT's
lunar orientation is an irreducible tool difference of the same kind as the
FK5/IAU-76-vs-CIO frame-chain difference already recorded for GMAT — the kind
of difference exit criterion 1 exists to measure rather than to eliminate. It
is bounded empirically below every gate by the five measured residuals in the
next section, and most tightly by the trans-lunar case (18.089824 m against a
1 km gate), the most lunar-ephemeris-sensitive case in the set.

## Phase 8 frozen truth (deferral discharged 2026-07-25)

GMAT was not installed on the Phase 8 execution host (the same class of
blocker as the Phase 3 maintainer boundary, D-15), so the Phase 8 truth CSVs
and the lunar libration excerpt were **deferred through the PRD section 9
valve** to `docs/release_checklist.md` item 10. That item is now discharged:
the lunar excerpt and all five truth CSVs are committed, and all five gates
measure and are met.

| Gate (test id) | Baseline file (GMAT R2026a) | Gate | Frozen measurement |
|---|---|---|---|
| XTOOL-MOLNIYA-GMAT | `truth_gmat_molniya.csv` | RMS < 100 m / 7 d | **0.160317 m** |
| XTOOL-LUNAR-GMAT | `truth_gmat_lunar_orbiter.csv` | RMS < 100 m / 7 d | **7.232688 m** |
| XTOOL-MARS-ORBITER-GMAT | `truth_gmat_mars_orbiter.csv` | RMS < 100 m / 7 d | **79.478630 m** |
| XTOOL-TRANSLUNAR-GMAT | `truth_gmat_translunar.csv` | < 1 km at lunar arrival | **18.089824 m** |
| XTOOL-MARS-CRUISE-GMAT | `truth_gmat_mars_cruise.csv` | < 100 km at arrival | **952.550 m** |

The command lines, artifact SHA-256 pins, and the toolchain provenance behind
each value are recorded in `manifest.toml`. The Mars-orbiter case carries the
thinnest margin (~1.26x); the Molniya case the widest (~624x).

Freeze history (recorded, not erased): the trans-lunar case was frozen twice.
The first freeze started the GMAT coast from the mid-burn t = 353 s state
instead of the t = 354 s ballistic-burnout state, which left a 75,189 km miss
at the matched epoch and invalidated the CSV; a ~45 km comparison-epoch defect
was fixed alongside it. The 2026-07-25 re-freeze ran
`python scripts/crosstool/run_gmat_phase8.py --case translunar` with no
`--arrival-s` override, so the committed 455401 s default span stands, and
GMAT's last reported row is `ElapsedSecs` = 455401.000000129 s — the requested
span reproduced to 1.29e-7 s. The arrival mission time is 455755.0 s (455401 s
of CSV elapsed time plus the 354 s burnout offset), where the `tli` truth log
has an exact row and where both tools place the spacecraft ~397,665 km from
Earth (simulator 397664.930 km, GMAT 397664.947 km); the companion velocity
difference is 5.379870e-05 m/s. In the same session the Molniya and
Mars-cruise CSVs were regenerated byte-identically, and a toolchain canary run
first reproduced the committed Phase 3 truth `truth_gmat_leo_gravity_8x8.csv`
byte-identically, tying the freeze to the install pinned in `manifest.toml`.

A gate whose truth CSV is absent from a checkout still skips with an explicit
message naming that checklist item — never a vacuous pass. That the comparison
machinery is correct independent of the external truth is proven by
`test_rms_machinery_is_correct_on_sim_own_states`, which feeds the sim's own
re-propagated states through the identical CSV/grid/RMS path and measures a
position RMS of 4.618e-09 m (i.e. the RMS, grid-alignment, and CSV round-trip
contribute nothing to any residual above).
