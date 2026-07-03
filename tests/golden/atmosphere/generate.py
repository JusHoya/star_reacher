"""Regenerate the atmosphere and drag golden-vector files in this directory.

Four models are covered (FR-8, FR-9, Phase 3 exit criteria 4, 8, 9):

- U.S. Standard Atmosphere 1976 (USSA76). The below-86-km rows and the
  above-86-km density nodes are TRANSCRIBED from the official document:
  NOAA/NASA/USAF, "U.S. Standard Atmosphere, 1976", NOAA-S/T 76-1562
  (NASA-TM-X-74335), U.S. Government Printing Office, Washington, D.C.,
  October 1976; Part 4, Table I ("Temperature, pressure, and density for
  geopotential and geometric altitudes in metric units"), geometric-altitude
  pages, document pages 51-73. Each transcribed case records its document
  page. This script re-derives every below-86-km row from the defining
  constants (Part 1, Table 2 and Table 4; equations 18, 33a, 33b, 42, 50)
  and refuses to write any file if a transcribed row disagrees with the
  analytic model at print precision (4 significant figures) - the
  transcription-error gate. The above-86-km nodes pass a log-density
  smoothness screen and an internal rho/rho0-column consistency spot check
  performed during transcription; an independent full-model implementation
  (Public Domain Aeronautical Software, https://www.pdas.com/bigtables.html)
  was compared at 12 altitudes at transcription time: agreement is within
  one unit in the 4th significant figure at and above 150 km, with
  deviations up to ~0.7 % between 86 and 120 km, where that implementation
  is known to depart from the printed tables. The committed values follow
  the printed document.

- Harris-Priester (atmosphere_hp). The 50-row lower/upper density table for
  mean solar activity, 100-1000 km, as tabulated in Montenbruck & Gill,
  "Satellite Orbits: Models, Methods and Applications", Springer, 2000,
  Sect. 3.5.2 (The Harris-Priester Density Model, pp. 89-91); underlying
  model: Harris & Priester (1962). The numeric transcription is taken
  from, and is digit-for-digit identical to, the table embedded in
  Orekit's HarrisPriester class (src/main/java/org/orekit/models/earth/
  atmosphere/HarrisPriester.java, develop branch, retrieved 2026-07-02),
  which implements the same book table in kg/m^3 and is the frozen D-15
  cross-tool baseline for this model. Off-node golden densities are
  computed here with mpmath (50 significant digits) using the identical
  formulation (exponential interpolation + cos^n(psi/2) bulge term).

- Mars piecewise-exponential atmosphere (PRD A-3, confidence: low). Node
  densities are evaluated from the NASA Glenn Research Center "Mars
  Atmosphere Model" curve fits (metric form: T_C = -31 - 0.000998 h below
  7000 m, T_C = -23.4 - 0.00222 h above; p_kPa = 0.699 exp(-0.00009 h);
  rho = p / (0.1921 (T_C + 273.1))), https://www.grc.nasa.gov/www/k-12/
  airplane/atmosmrm.html, retrieved 2026-07-02. PROVENANCE PROVISIONAL PER
  PRD A-3: the source is an educational curve fit to Mars Global Surveyor
  data with no stated uncertainty or altitude datum; the whole model is
  flagged confidence: low. Node values are committed as binary64 hex
  literals so the C++ table can reproduce them bit-exactly.

- Cannonball drag (FR-9). Reference acceleration vectors computed with
  mpmath (50 significant digits) from a_drag = -1/2 rho (Cd A/m) |v_rel|
  v_rel at committed double inputs.

Running this script rewrites the .toml golden files byte-identically; any
diff after regeneration means the script or the goldens were edited by
hand, which the FR-22 golden-update discipline forbids.
"""

from __future__ import annotations

import math
import pathlib

import mpmath

HERE = pathlib.Path(__file__).resolve().parent

mpmath.mp.dps = 50

# ---------------------------------------------------------------------------
# USSA76 defining constants (Part 1, Table 2, document p. 2; Table 4, p. 3).
# These mirror cpp/src/models/atmosphere_ussa76.cpp value for value.
# ---------------------------------------------------------------------------

T0_K = 288.15            # sea-level temperature [K] (Table 2)
P0_PA = 101325.0         # sea-level pressure [Pa] (Table 2)
G0P = 9.80665            # g0' [m^2/(s^2 m')] (Table 2)
RSTAR = 8314.32          # R* [N m/(kmol K)] (Table 2: 8.31432e3)
M0 = 28.9644             # sea-level mean molecular weight [kg/kmol] (p. 9)
R0_M = 6356766.0         # effective Earth radius for geopotential [m] (p. 8)
GAMMA = 1.40             # ratio of specific heats (Table 2)

# Table 4: geopotential layer bases H_b [m'] and molecular-scale temperature
# gradients L_M,b [K/m'] (seven layers, top at 84852.0 m' = 86 km geometric).
HB_MP = [0.0, 11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0, 84852.0]
LB_KPM = [-0.0065, 0.0, 0.0010, 0.0028, 0.0, -0.0028, -0.0020]


def ussa76_below86(z_m: float) -> tuple[float, float, float, float]:
    """Analytic USSA76 state below 86 km: (T_M [K], P [Pa], rho, c_s).

    Mirrors the C++ implementation: eq. (18) geopotential conversion,
    Table 4 layer walk, eqs. (33a)/(33b) pressure, eq. (42) density,
    eq. (50) speed of sound.
    """
    # eq. (18), Gamma = 1 m'/m. The top layer formula is allowed to extend
    # ~5 cm past H_7 = 84852 m' because the document's stated 86 km geometric
    # equivalence of H_7 is a rounded convention: H(86000 m) = 84852.046 m'.
    # The C++ implementation makes the identical choice (branch on z at
    # exactly 86 km) so the analytic and tabulated regions meet without a gap.
    h = R0_M * z_m / (R0_M + z_m)
    if not (-5004.0 <= h <= HB_MP[-1] + 0.05):
        raise ValueError(f"geopotential altitude out of range: {h}")
    # Walk the layers in fixed order, propagating the base values.
    tb = T0_K
    pb = P0_PA
    b = 0
    while b < 6 and h > HB_MP[b + 1]:
        dh_full = HB_MP[b + 1] - HB_MP[b]
        if LB_KPM[b] == 0.0:
            pb = pb * math.exp(-G0P * M0 * dh_full / (RSTAR * tb))
        else:
            tb_next = tb + LB_KPM[b] * dh_full
            pb = pb * (tb / tb_next) ** (G0P * M0 / (RSTAR * LB_KPM[b]))
            tb = tb_next
        b += 1
    dh = h - HB_MP[b]
    tm = tb + LB_KPM[b] * dh
    if LB_KPM[b] == 0.0:
        p = pb * math.exp(-G0P * M0 * dh / (RSTAR * tb))  # eq. (33b)
    else:
        p = pb * (tb / tm) ** (G0P * M0 / (RSTAR * LB_KPM[b]))  # eq. (33a)
    rho = p * M0 / (RSTAR * tm)  # eq. (42)
    cs = math.sqrt(GAMMA * RSTAR * tm / M0)  # eq. (50)
    return tm, p, rho, cs


# ---------------------------------------------------------------------------
# Transcribed Table I rows, geometric-altitude pages, below 86 km.
# Columns kept exactly as printed: T [K], P [mb], rho [kg/m^3]; each entry
# records the document page it was read from. The row set spans all seven
# Table 4 layers (b = 0..6).
# ---------------------------------------------------------------------------

BELOW86_ROWS = [
    # (z_m, T_K, P_mb, rho_kgpm3, document page)
    (0, "288.150", "1.01325e3", "1.2250e0", 53),
    (1000, "281.651", "8.9876e2", "1.1117e0", 53),
    (2000, "275.154", "7.9501e2", "1.0066e0", 53),
    (5000, "255.676", "5.4048e2", "7.3643e-1", 55),
    (10000, "223.252", "2.6499e2", "4.1351e-1", 57),
    (11000, "216.774", "2.2699e2", "3.6480e-1", 59),
    (15000, "216.650", "1.2111e2", "1.9476e-1", 59),
    (20000, "216.650", "5.5293e1", "8.8910e-2", 61),
    (25000, "221.552", "2.5492e1", "4.0084e-2", 61),
    (30000, "226.509", "1.1970e1", "1.8410e-2", 63),
    (32000, "228.490", "8.8906e0", "1.3555e-2", 63),
    (36000, "239.282", "4.9852e0", "7.2579e-3", 63),
    (40000, "250.350", "2.8714e0", "3.9957e-3", 65),
    (45000, "264.164", "1.4910e0", "1.9663e-3", 65),
    (47400, "270.650", "1.1022e0", "1.4187e-3", 65),
    (50000, "270.650", "7.9779e-1", "1.0269e-3", 65),
    (55000, "260.771", "4.2525e-1", "5.6810e-4", 65),
    (60000, "247.021", "2.1958e-1", "3.0968e-4", 67),
    (65000, "233.292", "1.0929e-1", "1.6321e-4", 67),
    (70000, "219.585", "5.2209e-2", "8.2829e-5", 67),
    (75000, "208.399", "2.3881e-2", "3.9921e-5", 67),
    (80000, "198.639", "1.0524e-2", "1.8458e-5", 67),
    (85000, "188.893", "4.4568e-3", "8.2196e-6", 67),
]

# ---------------------------------------------------------------------------
# Transcribed Table I density nodes, 86-1000 km (geometric altitude).
# These are simultaneously (a) the runtime interpolation grid compiled into
# atmosphere_ussa76.cpp and (b) the transcription check copy. Grid spacing
# follows the log-density curvature so that the piecewise-exponential
# interpolation error between nodes stays well below the node accuracy
# (bound derived in the math chapter): 1 km to 100 km, 2 km to 160 km,
# 5 km to 300 km (10 km at 300-310), 20 km to 460 km, then the printed
# 20-25 km rows to 1000 km.
# ---------------------------------------------------------------------------

UPPER_NODES = [
    # (z_m, rho_kgpm3 as printed, document page)
    (86000, "6.958e-6", 68), (87000, "5.824e-6", 68), (88000, "4.875e-6", 68),
    (89000, "4.081e-6", 68), (90000, "3.416e-6", 68), (91000, "2.860e-6", 68),
    (92000, "2.393e-6", 68), (93000, "2.000e-6", 68), (94000, "1.670e-6", 68),
    (95000, "1.393e-6", 68), (96000, "1.162e-6", 68), (97000, "9.685e-7", 68),
    (98000, "8.071e-7", 68), (99000, "6.725e-7", 68), (100000, "5.604e-7", 68),
    (102000, "3.935e-7", 68), (104000, "2.769e-7", 68), (106000, "1.954e-7", 68),
    (108000, "1.381e-7", 68), (110000, "9.708e-8", 68), (112000, "6.838e-8", 68),
    (114000, "4.975e-8", 68), (116000, "3.720e-8", 68), (118000, "2.847e-8", 68),
    (120000, "2.222e-8", 68), (122000, "1.767e-8", 68), (124000, "1.428e-8", 68),
    (126000, "1.171e-8", 68), (128000, "9.717e-9", 68), (130000, "8.152e-9", 68),
    (132000, "6.904e-9", 68), (134000, "5.897e-9", 68), (136000, "5.074e-9", 68),
    (138000, "4.396e-9", 68), (140000, "3.831e-9", 68), (142000, "3.358e-9", 68),
    (144000, "2.958e-9", 68), (146000, "2.618e-9", 68), (148000, "2.326e-9", 68),
    (150000, "2.076e-9", 69), (155000, "1.585e-9", 69), (160000, "1.233e-9", 69),
    (165000, "9.750e-10", 69), (170000, "7.815e-10", 69), (175000, "6.339e-10", 69),
    (180000, "5.194e-10", 69), (185000, "4.295e-10", 69), (190000, "3.581e-10", 69),
    (195000, "3.006e-10", 69), (200000, "2.541e-10", 69), (205000, "2.160e-10", 69),
    (210000, "1.846e-10", 69), (215000, "1.585e-10", 69), (220000, "1.367e-10", 69),
    (225000, "1.184e-10", 69), (230000, "1.029e-10", 70), (235000, "8.979e-11", 70),
    (240000, "7.858e-11", 70), (245000, "6.898e-11", 70), (250000, "6.073e-11", 70),
    (255000, "5.360e-11", 70), (260000, "4.742e-11", 70), (265000, "4.206e-11", 70),
    (270000, "3.738e-11", 70), (275000, "3.329e-11", 70), (280000, "2.971e-11", 70),
    (285000, "2.656e-11", 70), (290000, "2.378e-11", 70), (295000, "2.133e-11", 70),
    (300000, "1.916e-11", 70), (310000, "1.552e-11", 70), (320000, "1.264e-11", 71),
    (340000, "8.503e-12", 71), (360000, "5.805e-12", 71), (380000, "4.013e-12", 71),
    (400000, "2.803e-12", 71), (420000, "1.975e-12", 71), (440000, "1.402e-12", 71),
    (460000, "1.002e-12", 71), (480000, "7.208e-13", 72), (500000, "5.215e-13", 72),
    (525000, "3.509e-13", 72), (550000, "2.384e-13", 72), (575000, "1.637e-13", 72),
    (600000, "1.137e-13", 72), (625000, "7.998e-14", 72), (650000, "5.712e-14", 72),
    (675000, "4.148e-14", 72), (700000, "3.070e-14", 72), (725000, "2.318e-14", 72),
    (750000, "1.788e-14", 72), (775000, "1.410e-14", 72), (800000, "1.136e-14", 72),
    (825000, "9.339e-15", 72), (850000, "7.824e-15", 73), (875000, "6.664e-15", 73),
    (900000, "5.759e-15", 73), (925000, "5.038e-15", 73), (950000, "4.453e-15", 73),
    (975000, "3.968e-15", 73), (1000000, "3.561e-15", 73),
]

# Above-86-km rows gated by exit criterion 4 (a node subset: evaluation at a
# node must return the printed value, which also proves the transcription).
CRITERION_UPPER_Z = [
    86000, 90000, 100000, 110000, 120000, 150000, 200000,
    300000, 400000, 500000, 700000, 1000000,
]

# ---------------------------------------------------------------------------
# Harris-Priester coefficient table (see module docstring for provenance).
# Units: altitude [m], density [kg/m^3] (the book prints g/km^3 = 1e-12
# kg/m^3; the decimal strings below are exactly Orekit's kg/m^3 literals).
# ---------------------------------------------------------------------------

HP_TABLE = [
    ("100000.0", "4.974e-07", "4.974e-07"),
    ("120000.0", "2.490e-08", "2.490e-08"),
    ("130000.0", "8.377e-09", "8.710e-09"),
    ("140000.0", "3.899e-09", "4.059e-09"),
    ("150000.0", "2.122e-09", "2.215e-09"),
    ("160000.0", "1.263e-09", "1.344e-09"),
    ("170000.0", "8.008e-10", "8.758e-10"),
    ("180000.0", "5.283e-10", "6.010e-10"),
    ("190000.0", "3.617e-10", "4.297e-10"),
    ("200000.0", "2.557e-10", "3.162e-10"),
    ("210000.0", "1.839e-10", "2.396e-10"),
    ("220000.0", "1.341e-10", "1.853e-10"),
    ("230000.0", "9.949e-11", "1.455e-10"),
    ("240000.0", "7.488e-11", "1.157e-10"),
    ("250000.0", "5.709e-11", "9.308e-11"),
    ("260000.0", "4.403e-11", "7.555e-11"),
    ("270000.0", "3.430e-11", "6.182e-11"),
    ("280000.0", "2.697e-11", "5.095e-11"),
    ("290000.0", "2.139e-11", "4.226e-11"),
    ("300000.0", "1.708e-11", "3.526e-11"),
    ("320000.0", "1.099e-11", "2.511e-11"),
    ("340000.0", "7.214e-12", "1.819e-11"),
    ("360000.0", "4.824e-12", "1.337e-11"),
    ("380000.0", "3.274e-12", "9.955e-12"),
    ("400000.0", "2.249e-12", "7.492e-12"),
    ("420000.0", "1.558e-12", "5.684e-12"),
    ("440000.0", "1.091e-12", "4.355e-12"),
    ("460000.0", "7.701e-13", "3.362e-12"),
    ("480000.0", "5.474e-13", "2.612e-12"),
    ("500000.0", "3.916e-13", "2.042e-12"),
    ("520000.0", "2.819e-13", "1.605e-12"),
    ("540000.0", "2.042e-13", "1.267e-12"),
    ("560000.0", "1.488e-13", "1.005e-12"),
    ("580000.0", "1.092e-13", "7.997e-13"),
    ("600000.0", "8.070e-14", "6.390e-13"),
    ("620000.0", "6.012e-14", "5.123e-13"),
    ("640000.0", "4.519e-14", "4.121e-13"),
    ("660000.0", "3.430e-14", "3.325e-13"),
    ("680000.0", "2.632e-14", "2.691e-13"),
    ("700000.0", "2.043e-14", "2.185e-13"),
    ("720000.0", "1.607e-14", "1.779e-13"),
    ("740000.0", "1.281e-14", "1.452e-13"),
    ("760000.0", "1.036e-14", "1.190e-13"),
    ("780000.0", "8.496e-15", "9.776e-14"),
    ("800000.0", "7.069e-15", "8.059e-14"),
    ("840000.0", "4.680e-15", "5.741e-14"),
    ("880000.0", "3.200e-15", "4.210e-14"),
    ("920000.0", "2.210e-15", "3.130e-14"),
    ("960000.0", "1.560e-15", "2.360e-14"),
    ("1000000.0", "1.150e-15", "1.810e-14"),
]

# Off-node evaluation points: altitude between nodes, several bulge angles
# and exponents, spanning the low, middle, and high ends of the table.
HP_OFFNODE_CASES = [
    # (name, alt_m, cos_psi, n)
    ("bulge_apex_low", 115000.0, 1.0, 4.0),
    ("bulge_mid_250", 254321.0, 0.5, 4.0),
    ("bulge_quarter_350", 351500.0, 0.0, 4.0),
    ("antapex_450", 452750.0, -1.0, 4.0),
    ("low_incl_n2_600", 611000.0, 0.25, 2.0),
    ("polar_n6_800", 812345.0, 0.75, 6.0),
    ("near_floor", 101000.0, -0.3, 4.0),
    ("near_ceiling", 995000.0, 0.9, 4.0),
]


def hp_density_mp(alt_m: float, cos_psi: float, n: float) -> mpmath.mpf:
    """Harris-Priester density, mpmath replica of the C++ formulation."""
    alts = [float(a) for a, _, _ in HP_TABLE]
    rmin = [float(x) for _, x, _ in HP_TABLE]
    rmax = [float(x) for _, _, x in HP_TABLE]
    if alt_m < alts[0]:
        raise ValueError("below Harris-Priester table floor")
    if alt_m > alts[-1]:
        return mpmath.mpf(0)
    ia = 0
    while ia < len(alts) - 2 and alt_m > alts[ia + 1]:
        ia += 1
    dh = (mpmath.mpf(alts[ia]) - mpmath.mpf(alt_m)) / (
        mpmath.mpf(alts[ia]) - mpmath.mpf(alts[ia + 1])
    )
    rho_min = mpmath.mpf(rmin[ia]) * mpmath.power(
        mpmath.mpf(rmin[ia + 1]) / mpmath.mpf(rmin[ia]), dh
    )
    rho_max = mpmath.mpf(rmax[ia]) * mpmath.power(
        mpmath.mpf(rmax[ia + 1]) / mpmath.mpf(rmax[ia]), dh
    )
    c2 = (1 + mpmath.mpf(cos_psi)) / 2
    cp = mpmath.sqrt(c2)
    cos_pow = c2 * mpmath.power(cp, mpmath.mpf(n) - 2) if cp > 1e-12 else mpmath.mpf(0)
    return rho_min + (rho_max - rho_min) * cos_pow


# ---------------------------------------------------------------------------
# Mars atmosphere (PRD A-3, confidence: low; see module docstring).
# ---------------------------------------------------------------------------

MARS_NODE_STEP_M = 5000.0
MARS_NODE_TOP_M = 100000.0


def mars_glenn_density(h_m: float) -> float:
    """NASA Glenn Mars Atmosphere Model curve fits (metric), plain binary64."""
    if h_m < 7000.0:
        t_c = -31.0 - 0.000998 * h_m
    else:
        t_c = -23.4 - 0.00222 * h_m
    p_kpa = 0.699 * math.exp(-0.00009 * h_m)
    return p_kpa / (0.1921 * (t_c + 273.1))


def mars_nodes() -> list[tuple[float, float]]:
    nodes = []
    z = 0.0
    while z <= MARS_NODE_TOP_M:
        nodes.append((z, mars_glenn_density(z)))
        z += MARS_NODE_STEP_M
    return nodes


def mars_piecewise_mp(z_m: float, nodes: list[tuple[float, float]]) -> mpmath.mpf:
    """Piecewise-exponential evaluation, mpmath replica of the C++ model."""
    zs = [z for z, _ in nodes]
    rs = [r for _, r in nodes]
    i = 0
    while i < len(zs) - 2 and z_m >= zs[i + 1]:
        i += 1
    k = (mpmath.log(mpmath.mpf(rs[i + 1])) - mpmath.log(mpmath.mpf(rs[i]))) / (
        mpmath.mpf(zs[i + 1]) - mpmath.mpf(zs[i])
    )
    return mpmath.mpf(rs[i]) * mpmath.exp(k * (mpmath.mpf(z_m) - mpmath.mpf(zs[i])))


MARS_OFFNODE_Z = [2500.0, 7100.0, 12345.0, 33333.0, 61250.0, 87500.0, 99999.0]

# ---------------------------------------------------------------------------
# Cannonball drag reference vectors (FR-9).
# ---------------------------------------------------------------------------

DRAG_CASES = [
    # (name, rho [kg/m^3], Cd*A/m [m^2/kg], v_rel [m/s])
    ("leo_prograde_x", 2.803e-12, 0.0044, (7550.0, 0.0, 0.0)),
    ("leo_generic", 5.215e-13, 0.0044, (-7100.0, 1200.0, -350.0)),
    ("ascent_dense", 0.736, 0.0088, (450.0, -30.0, 120.0)),
    ("hp_bulge_state", 3.5e-12, 0.02, (1500.5, -7400.25, 25.125)),
    ("zero_velocity", 1.2250, 0.0044, (0.0, 0.0, 0.0)),
]


def drag_accel_mp(rho: float, cdam: float, v: tuple[float, float, float]):
    vm = [mpmath.mpf(c) for c in v]
    speed = mpmath.sqrt(vm[0] ** 2 + vm[1] ** 2 + vm[2] ** 2)
    factor = mpmath.mpf("-0.5") * mpmath.mpf(rho) * mpmath.mpf(cdam) * speed
    return [factor * c for c in vm]


# ---------------------------------------------------------------------------
# Self-checks (run before any file is written; a failure aborts generation)
# ---------------------------------------------------------------------------


def print_ulp4(printed: float) -> float:
    """Half-unit-in-the-last-place of a 4-significant-figure print."""
    return 0.5 * 10.0 ** (math.floor(math.log10(abs(printed))) - 3)


def check_below86() -> float:
    """Analytic model must reproduce every transcribed row at print precision."""
    worst = 0.0
    for z_m, t_s, p_mb_s, rho_s, _page in BELOW86_ROWS:
        t_ref = float(t_s)
        p_ref = float(p_mb_s) * 100.0  # mb -> Pa, exact factor
        rho_ref = float(rho_s)
        tm, p, rho, _cs = ussa76_below86(float(z_m))
        for model, printed in ((tm, t_ref), (p, p_ref), (rho, rho_ref)):
            err = abs(model - printed)
            gate = print_ulp4(printed) * (1.0 + 1e-9)
            worst = max(worst, err / print_ulp4(printed))
            assert err <= gate, (
                f"transcription/model mismatch at z={z_m}: "
                f"model {model!r} vs printed {printed!r}"
            )
    return worst


def check_upper_nodes() -> float:
    """Screen the transcribed 86-1000 km nodes for typos.

    Log-density slopes must decrease in magnitude with altitude (the profile
    stiffens monotonically over this span) and adjacent slopes must not jump
    by more than 25 % - a single-digit transcription error at 4 significant
    figures perturbs a local slope by far more than that. Also anchor the
    86 km node against the analytic below-86 model evaluated at 86 km.
    """
    zs = [z for z, _, _ in UPPER_NODES]
    lr = [math.log(float(r)) for _, r, _ in UPPER_NODES]
    assert zs == sorted(zs) and len(set(zs)) == len(zs)
    slopes = [(lr[i + 1] - lr[i]) / (zs[i + 1] - zs[i]) for i in range(len(zs) - 1)]
    worst_jump = 0.0
    for i in range(len(slopes) - 1):
        assert slopes[i] < 0.0, f"non-decreasing density near z={zs[i]}"
        jump = abs(slopes[i + 1] - slopes[i]) / abs(slopes[i])
        worst_jump = max(worst_jump, jump)
        assert jump < 0.25, (
            f"log-density slope jump {jump:.3f} at z={zs[i + 1]} "
            "- suspected transcription error"
        )
    # Analytic anchor at the 86 km boundary (same defining constants).
    _tm, _p, rho86, _cs = ussa76_below86(86000.0)
    ref = float(UPPER_NODES[0][1])
    assert abs(rho86 - ref) <= print_ulp4(ref), "86 km boundary anchor failed"
    for z in CRITERION_UPPER_Z:
        assert z in zs, f"criterion row {z} is not a committed node"
    return worst_jump


def check_hp_table() -> None:
    alts = [float(a) for a, _, _ in HP_TABLE]
    rmin = [float(x) for _, x, _ in HP_TABLE]
    rmax = [float(x) for _, _, x in HP_TABLE]
    assert len(HP_TABLE) == 50
    assert alts == sorted(alts) and alts[0] == 100000.0 and alts[-1] == 1000000.0
    for lo, hi in zip(rmin, rmax):
        assert lo <= hi
    for seq in (rmin, rmax):
        for a, b in zip(seq, seq[1:]):
            assert b < a, "Harris-Priester densities must decrease with altitude"


def check_mars(nodes: list[tuple[float, float]]) -> None:
    assert len(nodes) == 21
    for (_, a), (_, b) in zip(nodes, nodes[1:]):
        assert 0.0 < b < a, "Mars node densities must be positive and decreasing"


# ---------------------------------------------------------------------------
# Writer (same restricted-TOML emitter as the other golden suites; the C++
# test reader golden_io.hpp parses exactly this shape)
# ---------------------------------------------------------------------------


def emit(path: pathlib.Path, header: str, cases: list[dict]) -> None:
    lines = [f"# {line}" for line in header.strip().splitlines()]
    for case in cases:
        lines.append("")
        lines.append("[[case]]")
        for key, value in case.items():
            if isinstance(value, list):
                lines.append(f"{key} = [")
                lines.extend(f'  "{item}",' for item in value)
                lines.append("]")
            else:
                lines.append(f'{key} = "{value}"')
    path.write_text("\n".join(lines) + "\n", newline="\n", encoding="utf-8")


def mp_to_double_hex(x: mpmath.mpf) -> tuple[str, str]:
    d = float(x)
    return d.hex(), repr(d)


def main() -> None:
    worst_below = check_below86()
    worst_jump = check_upper_nodes()
    check_hp_table()
    nodes = mars_nodes()
    check_mars(nodes)

    # -- ussa76_rows.toml -------------------------------------------------
    row_cases = []
    for z_m, t_s, p_mb_s, rho_s, page in BELOW86_ROWS:
        row_cases.append(
            {
                "name": f"z_{z_m}",
                "region": "analytic",
                "z_m": str(z_m),
                "t_K": t_s,
                "p_mb": p_mb_s,
                "rho_kgpm3": rho_s,
                "document_page": str(page),
            }
        )
    upper = {z: (r, p) for z, r, p in UPPER_NODES}
    for z_m in CRITERION_UPPER_Z:
        rho_s, page = upper[z_m]
        row_cases.append(
            {
                "name": f"z_{z_m}",
                "region": "table",
                "z_m": str(z_m),
                "rho_kgpm3": rho_s,
                "document_page": str(page),
            }
        )
    emit(
        HERE / "ussa76_rows.toml",
        "USSA76 published-row golden vectors (FR-8, Phase 3 exit criterion 4).\n"
        "Rows transcribed from U.S. Standard Atmosphere, 1976 (NOAA-S/T\n"
        "76-1562 / NASA-TM-X-74335), Part 4, Table I, geometric-altitude\n"
        "pages; each case records its document page. region = analytic rows\n"
        "carry T [K] and P [mb] as printed (P in Pa is p_mb * 100 exactly);\n"
        "region = table rows above 86 km carry density only. The consuming\n"
        "test requires 4-significant-figure (print precision) agreement.\n"
        "Provenance and tolerances in manifest.toml. Regenerated by\n"
        "generate.py.",
        row_cases,
    )

    # -- ussa76_upper_nodes.toml ------------------------------------------
    node_cases = [
        {
            "name": f"node_{z}",
            "z_m": str(z),
            "rho_kgpm3": rho_s,
            "document_page": str(page),
        }
        for z, rho_s, page in UPPER_NODES
    ]
    emit(
        HERE / "ussa76_upper_nodes.toml",
        "USSA76 86-1000 km density interpolation nodes (FR-8).\n"
        "Transcribed from U.S. Standard Atmosphere, 1976, Part 4, Table I\n"
        "(geometric altitude, metric units), document pages 68-73; values\n"
        "exactly as printed (4 significant figures). This file is the\n"
        "transcription check copy of the node table compiled into\n"
        "cpp/src/models/atmosphere_ussa76.cpp: the doctest suite requires\n"
        "the compiled table to strtod-match these strings bit for bit.\n"
        "Provenance and tolerances in manifest.toml. Regenerated by\n"
        "generate.py.",
        node_cases,
    )

    # -- harris_priester_table.toml ----------------------------------------
    hp_cases = [
        {
            "name": f"node_{int(float(alt) / 1000)}km",
            "alt_m": alt,
            "rho_min_kgpm3": rmin,
            "rho_max_kgpm3": rmax,
        }
        for alt, rmin, rmax in HP_TABLE
    ]
    emit(
        HERE / "harris_priester_table.toml",
        "Harris-Priester lower/upper density coefficients (FR-8, Phase 3\n"
        "exit criterion 8): mean solar activity, 100-1000 km. Montenbruck &\n"
        "Gill, Satellite Orbits (Springer, 2000), Sect. 3.5.2, pp. 89-91;\n"
        "digit-for-digit transcription of the same table as embedded in\n"
        "Orekit's HarrisPriester class (the D-15 cross-tool baseline),\n"
        "units kg/m^3 (the book prints g/km^3 = 1e-12 kg/m^3). This file is\n"
        "the transcription check copy of the table compiled into\n"
        "cpp/src/models/atmosphere_hp.cpp: the doctest suite requires the\n"
        "compiled table to strtod-match these strings bit for bit.\n"
        "Provenance and tolerances in manifest.toml. Regenerated by\n"
        "generate.py.",
        hp_cases,
    )

    # -- hp_offnode.toml ----------------------------------------------------
    off_cases = []
    for name, alt, cpsi, n in HP_OFFNODE_CASES:
        rho = hp_density_mp(alt, cpsi, n)
        rho_hex, rho_dec = mp_to_double_hex(rho)
        off_cases.append(
            {
                "name": name,
                "alt_m": repr(alt),
                "cos_psi": repr(cpsi),
                "n": repr(n),
                "rho_kgpm3": rho_hex,
                "rho_kgpm3_dec": rho_dec,
            }
        )
    emit(
        HERE / "hp_offnode.toml",
        "Harris-Priester off-node golden densities (FR-8).\n"
        "Independent mpmath (50-digit) evaluation of the published\n"
        "formulation (exponential interpolation between table nodes plus\n"
        "the cos^n(psi/2) diurnal-bulge term) at off-node altitudes and\n"
        "several bulge angles/exponents; rho_kgpm3 is the binary64\n"
        "rounding of the 50-digit value. Provenance and tolerances in\n"
        "manifest.toml. Regenerated by generate.py.",
        off_cases,
    )

    # -- mars_nodes.toml ----------------------------------------------------
    mars_cases = []
    for z, rho in nodes:
        mars_cases.append(
            {
                "name": f"node_{int(z / 1000)}km",
                "z_m": repr(z),
                "rho_kgpm3": rho.hex(),
                "rho_kgpm3_dec": repr(rho),
            }
        )
    emit(
        HERE / "mars_nodes.toml",
        "Mars piecewise-exponential atmosphere nodes (FR-8, PRD A-3;\n"
        "Phase 3 exit criterion 9). CONFIDENCE: LOW - provenance\n"
        "provisional per PRD A-3. Node densities evaluated from the NASA\n"
        "Glenn Research Center Mars Atmosphere Model curve fits (metric\n"
        "units) at 5 km steps, 0-100 km; rho_kgpm3 is the exact binary64\n"
        "value (hex) that cpp/src/models/atmosphere_mars.cpp must return\n"
        "bit-exactly at each node. Provenance and tolerances in\n"
        "manifest.toml. Regenerated by generate.py.",
        mars_cases,
    )

    # -- mars_offnode.toml ---------------------------------------------------
    mars_off = []
    for z in MARS_OFFNODE_Z:
        rho = mars_piecewise_mp(z, nodes)
        rho_hex, rho_dec = mp_to_double_hex(rho)
        mars_off.append(
            {
                "name": f"z_{z:.0f}",
                "z_m": repr(z),
                "rho_kgpm3": rho_hex,
                "rho_kgpm3_dec": rho_dec,
            }
        )
    emit(
        HERE / "mars_offnode.toml",
        "Mars atmosphere off-node golden densities (FR-8, PRD A-3;\n"
        "CONFIDENCE: LOW - provenance provisional per PRD A-3).\n"
        "Independent mpmath (50-digit) evaluation of the piecewise\n"
        "exponential defined by mars_nodes.toml (per-segment scale height\n"
        "from adjacent node densities) at off-node altitudes. Provenance\n"
        "and tolerances in manifest.toml. Regenerated by generate.py.",
        mars_off,
    )

    # -- drag_vectors.toml ---------------------------------------------------
    drag_cases = []
    for name, rho, cdam, v in DRAG_CASES:
        a = drag_accel_mp(rho, cdam, v)
        drag_cases.append(
            {
                "name": name,
                "rho_kgpm3": repr(rho),
                "cd_a_over_m_m2pkg": repr(cdam),
                "v_rel_mps": [float(c).hex() for c in v],
                "v_rel_mps_dec": [repr(float(c)) for c in v],
                "a_mps2": [float(c).hex() for c in a],
                "a_mps2_dec": [repr(float(c)) for c in a],
            }
        )
    emit(
        HERE / "drag_vectors.toml",
        "Cannonball drag golden acceleration vectors (FR-9).\n"
        "a = -1/2 rho (Cd A/m) |v_rel| v_rel evaluated with mpmath\n"
        "(50 digits) at exact binary64 inputs; a_mps2 is the binary64\n"
        "rounding of the 50-digit result. Provenance and tolerances in\n"
        "manifest.toml. Regenerated by generate.py.",
        drag_cases,
    )

    print("golden files regenerated and self-checked")
    print(
        f"below-86 worst |model - printed| = {worst_below:.3f} of a half-unit "
        "in the 4th significant figure"
    )
    print(f"upper-node worst log-density slope jump = {worst_jump:.3f}")


if __name__ == "__main__":
    main()
