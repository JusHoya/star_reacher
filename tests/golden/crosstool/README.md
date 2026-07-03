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

## Determinism note

Both missions are covered by `tests/python/test_crosstool_missions.py`:
end-to-end double runs must produce bit-identical `run.srlog` files
(D-10), the gravity-only orbit must stay bounded with a drift-free
Keplerian energy trend, and the drag case must lose energy secularly.
