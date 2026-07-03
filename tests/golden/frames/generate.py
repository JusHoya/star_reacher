"""Regenerate the reference-frame golden-vector files in this directory.

Covers the FR-3 Earth-orientation chain (IAU 2006/2000B, CIO based), the
Moon principal-axis frame construction, and the Mars IAU 2015 body-fixed
frame (cpp/src/frames.cpp). Four kinds of evidence are produced:

1. earth_chain.toml -- the composed GCRF->ITRF chain at 14 epochs spanning
   2020-2060, generated with ERFA (pyerfa), the reference implementation of
   the IAU SOFA algorithms, by exactly the composition the C++ core
   implements: dpsi,deps = nut00b(tt); rbpn = pn06(tt, dpsi, deps)[-1]
   (equivalently fw2m of the pfw06 Fukushima-Williams angles with the
   nutation added); x, y = bpn2xy(rbpn); s = s06(tt, x, y);
   rc2i = c2ixys(x, y, s); era = era00(ut1); C = Rz(era) * rc2i.
   Polar motion is deliberately absent from both legs (FR-3: neglected,
   bound documented in the frames chapter).
2. cookbook_2006_2000a.toml -- the published worked example of the IAU SOFA
   cookbook "SOFA Tools for Earth Attitude" (2007 April 5, 12:00:00.0 UTC),
   section 5.5 (IAU 2006/2000A, CIO based): the printed no-polar-motion
   celestial-to-terrestrial matrix and X, Y, s, ERA values, transcribed
   verbatim. This anchors the chain to published, human-readable numbers;
   the comparison bound is the IAU 2000B-vs-2000A model difference plus the
   cookbook's applied celestial-pole corrections dX06/dY06, NOT the 1e-11
   ERFA-agreement gate.
3. moon_pa.toml -- the 3-1-3 Moon principal-axis construction
   C = R3(psi) R1(theta) R3(phi) at committed angle sets, assembled from
   ERFA rotation primitives (rz/rx) and cross-checked against an
   independent NumPy element-formula evaluation.
4. mars_iau.toml -- Mars IAU 2015 rotational elements (Archinal et al.
   2018, post-erratum values as distributed in NAIF pck00011.tpc) evaluated
   independently here, with the frame matrix R3(W) R1(pi/2-dec) R3(pi/2+ra)
   assembled from ERFA primitives and the pole direction cross-checked
   against the explicit (cos dec cos ra, cos dec sin ra, sin dec) unit
   vector.

In addition, series_terms.toml commits the transcribed IAU 2000B nutation
and s(X,Y) series tables themselves so the C++ compiled tables can be
checked for transcription equality term by term. The transcription in this
script is validated functionally at generation time against erfa.nut00b /
erfa.s06 / erfa.pfw06 / erfa.obl06 / erfa.era00 at every golden epoch plus
300 seeded pseudo-random epochs spanning 2020-2060.

Time plumbing mirrors tests/golden/time/generate.py (imported directly) so
the committed TAI epochs are bit-identical to what star::time produces.

Running this script rewrites the .toml golden files byte-identically; any
diff after regeneration means the script or the goldens were edited by
hand, which the FR-22 golden-update discipline forbids.
"""

from __future__ import annotations

import importlib.util
import math
import pathlib
import random
import warnings

import erfa
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent

# Reuse the time-system reference implementation (UTC->TAI, two-part JDs,
# TDB series) so frame goldens and time goldens share one epoch definition.
_spec = importlib.util.spec_from_file_location(
    "time_golden_generate", HERE.parent / "time" / "generate.py"
)
time_ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(time_ref)

DJ00 = 2451545.0
DJC = 36525.0
DAS2R = 4.848136811095359935899141e-6  # arcsec -> rad (ERFA ERFA_DAS2R)
TURNAS = 1296000.0  # arcsec in one turn
D2PI = 6.283185307179586476925287
DMAS2R = DAS2R / 1e3
D2R = math.radians(1.0)

# ---------------------------------------------------------------------------
# Transcribed series tables.
#
# Source: IAU 2000B nutation series, McCarthy & Luzum (2003), Cel. Mech.
# Dyn. Astron. 85, 37-49, with the Luzum (2001) planetary-bias offsets; the
# s(X,Y) series and IAU 2006 (P03) precession polynomials per IERS
# Conventions (2010) Chapter 5. All coefficients transcribed verbatim from
# the ERFA 2.0.1 reference source (liberfa/erfa: nut00b.c, s06.c, pfw06.c,
# obl06.c, fal03.c..fapa03.c, era00.c), which implements those publications.
# The same numbers are compiled into cpp/src/frames_series.hpp; the
# series_terms.toml emitted here is the bridge that keeps the two copies
# provably identical.
# ---------------------------------------------------------------------------

# Luni-solar nutation, IAU 2000B: (nl, nlp, nf, nd, nom, ps, pst, pc, ec,
# ect, es). Sine/cosine coefficients in 0.1 microarcsec (and per Julian
# century for the t-multiplied ones).
NUT00B_TERMS = [
    (0, 0, 0, 0, 1, -172064161.0, -174666.0, 33386.0, 92052331.0, 9086.0, 15377.0),
    (0, 0, 2, -2, 2, -13170906.0, -1675.0, -13696.0, 5730336.0, -3015.0, -4587.0),
    (0, 0, 2, 0, 2, -2276413.0, -234.0, 2796.0, 978459.0, -485.0, 1374.0),
    (0, 0, 0, 0, 2, 2074554.0, 207.0, -698.0, -897492.0, 470.0, -291.0),
    (0, 1, 0, 0, 0, 1475877.0, -3633.0, 11817.0, 73871.0, -184.0, -1924.0),
    (0, 1, 2, -2, 2, -516821.0, 1226.0, -524.0, 224386.0, -677.0, -174.0),
    (1, 0, 0, 0, 0, 711159.0, 73.0, -872.0, -6750.0, 0.0, 358.0),
    (0, 0, 2, 0, 1, -387298.0, -367.0, 380.0, 200728.0, 18.0, 318.0),
    (1, 0, 2, 0, 2, -301461.0, -36.0, 816.0, 129025.0, -63.0, 367.0),
    (0, -1, 2, -2, 2, 215829.0, -494.0, 111.0, -95929.0, 299.0, 132.0),
    (0, 0, 2, -2, 1, 128227.0, 137.0, 181.0, -68982.0, -9.0, 39.0),
    (-1, 0, 2, 0, 2, 123457.0, 11.0, 19.0, -53311.0, 32.0, -4.0),
    (-1, 0, 0, 2, 0, 156994.0, 10.0, -168.0, -1235.0, 0.0, 82.0),
    (1, 0, 0, 0, 1, 63110.0, 63.0, 27.0, -33228.0, 0.0, -9.0),
    (-1, 0, 0, 0, 1, -57976.0, -63.0, -189.0, 31429.0, 0.0, -75.0),
    (-1, 0, 2, 2, 2, -59641.0, -11.0, 149.0, 25543.0, -11.0, 66.0),
    (1, 0, 2, 0, 1, -51613.0, -42.0, 129.0, 26366.0, 0.0, 78.0),
    (-2, 0, 2, 0, 1, 45893.0, 50.0, 31.0, -24236.0, -10.0, 20.0),
    (0, 0, 0, 2, 0, 63384.0, 11.0, -150.0, -1220.0, 0.0, 29.0),
    (0, 0, 2, 2, 2, -38571.0, -1.0, 158.0, 16452.0, -11.0, 68.0),
    (0, -2, 2, -2, 2, 32481.0, 0.0, 0.0, -13870.0, 0.0, 0.0),
    (-2, 0, 0, 2, 0, -47722.0, 0.0, -18.0, 477.0, 0.0, -25.0),
    (2, 0, 2, 0, 2, -31046.0, -1.0, 131.0, 13238.0, -11.0, 59.0),
    (1, 0, 2, -2, 2, 28593.0, 0.0, -1.0, -12338.0, 10.0, -3.0),
    (-1, 0, 2, 0, 1, 20441.0, 21.0, 10.0, -10758.0, 0.0, -3.0),
    (2, 0, 0, 0, 0, 29243.0, 0.0, -74.0, -609.0, 0.0, 13.0),
    (0, 0, 2, 0, 0, 25887.0, 0.0, -66.0, -550.0, 0.0, 11.0),
    (0, 1, 0, 0, 1, -14053.0, -25.0, 79.0, 8551.0, -2.0, -45.0),
    (-1, 0, 0, 2, 1, 15164.0, 10.0, 11.0, -8001.0, 0.0, -1.0),
    (0, 2, 2, -2, 2, -15794.0, 72.0, -16.0, 6850.0, -42.0, -5.0),
    (0, 0, -2, 2, 0, 21783.0, 0.0, 13.0, -167.0, 0.0, 13.0),
    (1, 0, 0, -2, 1, -12873.0, -10.0, -37.0, 6953.0, 0.0, -14.0),
    (0, -1, 0, 0, 1, -12654.0, 11.0, 63.0, 6415.0, 0.0, 26.0),
    (-1, 0, 2, 2, 1, -10204.0, 0.0, 25.0, 5222.0, 0.0, 15.0),
    (0, 2, 0, 0, 0, 16707.0, -85.0, -10.0, 168.0, -1.0, 10.0),
    (1, 0, 2, 2, 2, -7691.0, 0.0, 44.0, 3268.0, 0.0, 19.0),
    (-2, 0, 2, 0, 0, -11024.0, 0.0, -14.0, 104.0, 0.0, 2.0),
    (0, 1, 2, 0, 2, 7566.0, -21.0, -11.0, -3250.0, 0.0, -5.0),
    (0, 0, 2, 2, 1, -6637.0, -11.0, 25.0, 3353.0, 0.0, 14.0),
    (0, -1, 2, 0, 2, -7141.0, 21.0, 8.0, 3070.0, 0.0, 4.0),
    (0, 0, 0, 2, 1, -6302.0, -11.0, 2.0, 3272.0, 0.0, 4.0),
    (1, 0, 2, -2, 1, 5800.0, 10.0, 2.0, -3045.0, 0.0, -1.0),
    (2, 0, 2, -2, 2, 6443.0, 0.0, -7.0, -2768.0, 0.0, -4.0),
    (-2, 0, 0, 2, 1, -5774.0, -11.0, -15.0, 3041.0, 0.0, -5.0),
    (2, 0, 2, 0, 1, -5350.0, 0.0, 21.0, 2695.0, 0.0, 12.0),
    (0, -1, 2, -2, 1, -4752.0, -11.0, -3.0, 2719.0, 0.0, -3.0),
    (0, 0, 0, -2, 1, -4940.0, -11.0, -21.0, 2720.0, 0.0, -9.0),
    (-1, -1, 0, 2, 0, 7350.0, 0.0, -8.0, -51.0, 0.0, 4.0),
    (2, 0, 0, -2, 1, 4065.0, 0.0, 6.0, -2206.0, 0.0, 1.0),
    (1, 0, 0, 2, 0, 6579.0, 0.0, -24.0, -199.0, 0.0, 2.0),
    (0, 1, 2, -2, 1, 3579.0, 0.0, 5.0, -1900.0, 0.0, 1.0),
    (1, -1, 0, 0, 0, 4725.0, 0.0, -6.0, -41.0, 0.0, 3.0),
    (-2, 0, 2, 0, 2, -3075.0, 0.0, -2.0, 1313.0, 0.0, -1.0),
    (3, 0, 2, 0, 2, -2904.0, 0.0, 15.0, 1233.0, 0.0, 7.0),
    (0, -1, 0, 2, 0, 4348.0, 0.0, -10.0, -81.0, 0.0, 2.0),
    (1, -1, 2, 0, 2, -2878.0, 0.0, 8.0, 1232.0, 0.0, 4.0),
    (0, 0, 0, 1, 0, -4230.0, 0.0, 5.0, -20.0, 0.0, -2.0),
    (-1, -1, 2, 2, 2, -2819.0, 0.0, 7.0, 1207.0, 0.0, 3.0),
    (-1, 0, 2, 0, 0, -4056.0, 0.0, 5.0, 40.0, 0.0, -2.0),
    (0, -1, 2, 2, 2, -2647.0, 0.0, 11.0, 1129.0, 0.0, 5.0),
    (-2, 0, 0, 0, 1, -2294.0, 0.0, -10.0, 1266.0, 0.0, -4.0),
    (1, 1, 2, 0, 2, 2481.0, 0.0, -7.0, -1062.0, 0.0, -3.0),
    (2, 0, 0, 0, 1, 2179.0, 0.0, -2.0, -1129.0, 0.0, -2.0),
    (-1, 1, 0, 1, 0, 3276.0, 0.0, 1.0, -9.0, 0.0, 0.0),
    (1, 1, 0, 0, 0, -3389.0, 0.0, 5.0, 35.0, 0.0, -2.0),
    (1, 0, 2, 0, 0, 3339.0, 0.0, -13.0, -107.0, 0.0, 1.0),
    (-1, 0, 2, -2, 1, -1987.0, 0.0, -6.0, 1073.0, 0.0, -2.0),
    (1, 0, 0, 0, 2, -1981.0, 0.0, 0.0, 854.0, 0.0, 0.0),
    (-1, 0, 0, 1, 0, 4026.0, 0.0, -353.0, -553.0, 0.0, -139.0),
    (0, 0, 2, 1, 2, 1660.0, 0.0, -5.0, -710.0, 0.0, -2.0),
    (-1, 0, 2, 4, 2, -1521.0, 0.0, 9.0, 647.0, 0.0, 4.0),
    (-1, 1, 0, 1, 1, 1314.0, 0.0, 0.0, -700.0, 0.0, 0.0),
    (0, -2, 2, -2, 1, -1283.0, 0.0, 0.0, 672.0, 0.0, 0.0),
    (1, 0, 2, 2, 1, -1331.0, 0.0, 8.0, 663.0, 0.0, 4.0),
    (-2, 0, 2, 2, 2, 1383.0, 0.0, -2.0, -594.0, 0.0, -2.0),
    (-1, 0, 0, 0, 2, 1405.0, 0.0, 4.0, -610.0, 0.0, 2.0),
    (1, 1, 2, -2, 2, 1290.0, 0.0, 0.0, -556.0, 0.0, 0.0),
]

# Fixed offsets in lieu of planetary nutation (Luzum 2001, "rigorous"
# method), milliarcsec.
NUT00B_DPPLAN_MAS = -0.135
NUT00B_DEPLAN_MAS = 0.388

# s(X,Y) series (SOFA/ERFA S06 form): polynomial coefficients (arcsec) and
# per-order terms (nfa[8] multipliers of l, l', F, D, Om, L_Ve, L_E, pA;
# sine and cosine coefficients in arcsec).
S06_POLY = [94.00e-6, 3808.65e-6, -122.68e-6, -72574.11e-6, 27.98e-6, 15.62e-6]
S06_TERMS_0 = [
    ((0, 0, 0, 0, 1, 0, 0, 0), -2640.73e-6, 0.39e-6),
    ((0, 0, 0, 0, 2, 0, 0, 0), -63.53e-6, 0.02e-6),
    ((0, 0, 2, -2, 3, 0, 0, 0), -11.75e-6, -0.01e-6),
    ((0, 0, 2, -2, 1, 0, 0, 0), -11.21e-6, -0.01e-6),
    ((0, 0, 2, -2, 2, 0, 0, 0), 4.57e-6, 0.00e-6),
    ((0, 0, 2, 0, 3, 0, 0, 0), -2.02e-6, 0.00e-6),
    ((0, 0, 2, 0, 1, 0, 0, 0), -1.98e-6, 0.00e-6),
    ((0, 0, 0, 0, 3, 0, 0, 0), 1.72e-6, 0.00e-6),
    ((0, 1, 0, 0, 1, 0, 0, 0), 1.41e-6, 0.01e-6),
    ((0, 1, 0, 0, -1, 0, 0, 0), 1.26e-6, 0.01e-6),
    ((1, 0, 0, 0, -1, 0, 0, 0), 0.63e-6, 0.00e-6),
    ((1, 0, 0, 0, 1, 0, 0, 0), 0.63e-6, 0.00e-6),
    ((0, 1, 2, -2, 3, 0, 0, 0), -0.46e-6, 0.00e-6),
    ((0, 1, 2, -2, 1, 0, 0, 0), -0.45e-6, 0.00e-6),
    ((0, 0, 4, -4, 4, 0, 0, 0), -0.36e-6, 0.00e-6),
    ((0, 0, 1, -1, 1, -8, 12, 0), 0.24e-6, 0.12e-6),
    ((0, 0, 2, 0, 0, 0, 0, 0), -0.32e-6, 0.00e-6),
    ((0, 0, 2, 0, 2, 0, 0, 0), -0.28e-6, 0.00e-6),
    ((1, 0, 2, 0, 3, 0, 0, 0), -0.27e-6, 0.00e-6),
    ((1, 0, 2, 0, 1, 0, 0, 0), -0.26e-6, 0.00e-6),
    ((0, 0, 2, -2, 0, 0, 0, 0), 0.21e-6, 0.00e-6),
    ((0, 1, -2, 2, -3, 0, 0, 0), -0.19e-6, 0.00e-6),
    ((0, 1, -2, 2, -1, 0, 0, 0), -0.18e-6, 0.00e-6),
    ((0, 0, 0, 0, 0, 8, -13, -1), 0.10e-6, -0.05e-6),
    ((0, 0, 0, 2, 0, 0, 0, 0), -0.15e-6, 0.00e-6),
    ((2, 0, -2, 0, -1, 0, 0, 0), 0.14e-6, 0.00e-6),
    ((0, 1, 2, -2, 2, 0, 0, 0), 0.14e-6, 0.00e-6),
    ((1, 0, 0, -2, 1, 0, 0, 0), -0.14e-6, 0.00e-6),
    ((1, 0, 0, -2, -1, 0, 0, 0), -0.14e-6, 0.00e-6),
    ((0, 0, 4, -2, 4, 0, 0, 0), -0.13e-6, 0.00e-6),
    ((0, 0, 2, -2, 4, 0, 0, 0), 0.11e-6, 0.00e-6),
    ((1, 0, -2, 0, -3, 0, 0, 0), -0.11e-6, 0.00e-6),
    ((1, 0, -2, 0, -1, 0, 0, 0), -0.11e-6, 0.00e-6),
]
S06_TERMS_1 = [
    ((0, 0, 0, 0, 2, 0, 0, 0), -0.07e-6, 3.57e-6),
    ((0, 0, 0, 0, 1, 0, 0, 0), 1.73e-6, -0.03e-6),
    ((0, 0, 2, -2, 3, 0, 0, 0), 0.00e-6, 0.48e-6),
]
S06_TERMS_2 = [
    ((0, 0, 0, 0, 1, 0, 0, 0), 743.52e-6, -0.17e-6),
    ((0, 0, 2, -2, 2, 0, 0, 0), 56.91e-6, 0.06e-6),
    ((0, 0, 2, 0, 2, 0, 0, 0), 9.84e-6, -0.01e-6),
    ((0, 0, 0, 0, 2, 0, 0, 0), -8.85e-6, 0.01e-6),
    ((0, 1, 0, 0, 0, 0, 0, 0), -6.38e-6, -0.05e-6),
    ((1, 0, 0, 0, 0, 0, 0, 0), -3.07e-6, 0.00e-6),
    ((0, 1, 2, -2, 2, 0, 0, 0), 2.23e-6, 0.00e-6),
    ((0, 0, 2, 0, 1, 0, 0, 0), 1.67e-6, 0.00e-6),
    ((1, 0, 2, 0, 2, 0, 0, 0), 1.30e-6, 0.00e-6),
    ((0, 1, -2, 2, -2, 0, 0, 0), 0.93e-6, 0.00e-6),
    ((1, 0, 0, -2, 0, 0, 0, 0), 0.68e-6, 0.00e-6),
    ((0, 0, 2, -2, 1, 0, 0, 0), -0.55e-6, 0.00e-6),
    ((1, 0, -2, 0, -2, 0, 0, 0), 0.53e-6, 0.00e-6),
    ((0, 0, 0, 2, 0, 0, 0, 0), -0.27e-6, 0.00e-6),
    ((1, 0, 0, 0, 1, 0, 0, 0), -0.27e-6, 0.00e-6),
    ((1, 0, -2, -2, -2, 0, 0, 0), -0.26e-6, 0.00e-6),
    ((1, 0, 0, 0, -1, 0, 0, 0), -0.25e-6, 0.00e-6),
    ((1, 0, 2, 0, 1, 0, 0, 0), 0.22e-6, 0.00e-6),
    ((2, 0, 0, -2, 0, 0, 0, 0), -0.21e-6, 0.00e-6),
    ((2, 0, -2, 0, -1, 0, 0, 0), 0.20e-6, 0.00e-6),
    ((0, 0, 2, 2, 2, 0, 0, 0), 0.17e-6, 0.00e-6),
    ((2, 0, 2, 0, 2, 0, 0, 0), 0.13e-6, 0.00e-6),
    ((2, 0, 0, 0, 0, 0, 0, 0), -0.13e-6, 0.00e-6),
    ((1, 0, 2, -2, 2, 0, 0, 0), -0.12e-6, 0.00e-6),
    ((0, 0, 2, 0, 0, 0, 0, 0), -0.11e-6, 0.00e-6),
]
S06_TERMS_3 = [
    ((0, 0, 0, 0, 1, 0, 0, 0), 0.30e-6, -23.42e-6),
    ((0, 0, 2, -2, 2, 0, 0, 0), -0.03e-6, -1.46e-6),
    ((0, 0, 2, 0, 2, 0, 0, 0), -0.01e-6, -0.25e-6),
    ((0, 0, 0, 0, 2, 0, 0, 0), 0.00e-6, 0.23e-6),
]
S06_TERMS_4 = [
    ((0, 0, 0, 0, 1, 0, 0, 0), -0.26e-6, -0.01e-6),
]

# IAU 2006 (P03) bias-precession Fukushima-Williams angle polynomials
# (arcsec, powers of TT Julian centuries since J2000).
FW_GAMB = [-0.052928, 10.556378, 0.4932044, -0.00031238, -0.000002788, 0.0000000260]
FW_PHIB = [84381.412819, -46.811016, 0.0511268, 0.00053289, -0.000000440, -0.0000000176]
FW_PSIB = [-0.041775, 5038.481484, 1.5584175, -0.00018522, -0.000026452, -0.0000000148]
# IAU 2006 mean obliquity polynomial (arcsec).
OBL06 = [84381.406, -46.836769, -0.0001831, 0.00200340, -0.000000576, -0.0000000434]

# Fundamental (Delaunay) arguments, IERS Conventions 2003 full forms
# (arcsec except the last three), used by the s06 series.
FA_L = [485868.249036, 1717915923.2178, 31.8792, 0.051635, -0.00024470]
FA_LP = [1287104.793048, 129596581.0481, -0.5532, 0.000136, -0.00001149]
FA_F = [335779.526232, 1739527262.8478, -12.7512, -0.001037, 0.00000417]
FA_D = [1072260.703692, 1602961601.2090, -6.3706, 0.006593, -0.00003169]
FA_OM = [450160.398036, -6962890.5431, 7.4722, 0.007702, -0.00005939]
FA_VE = [3.176146697, 1021.3285546211]  # radians
FA_E = [1.753470314, 628.3075849991]  # radians
FA_PA = [0.024381750, 0.00000538691]  # radians (t * (c0 + c1 t))

# Simplified linear fundamental arguments used by the IAU 2000B nutation
# itself (arcsec; the truncation is part of the published 2000B model).
NUT00B_FA = {
    "el": [485868.249036, 1717915923.2178],
    "elp": [1287104.79305, 129596581.0481],
    "f": [335779.526232, 1739527262.8478],
    "d": [1072260.70369, 1602961601.2090],
    "om": [450160.398036, -6962890.5431],
}

# Earth rotation angle (Capitaine, Guinot & McCarthy 2000; IERS Conventions
# 2010 eq. 5.15): ERA = 2 pi (f + 0.7790572732640 + 0.00273781191135448 Tu).
ERA_C0 = 0.7790572732640
ERA_C1 = 0.00273781191135448

# ---------------------------------------------------------------------------
# Python mirror of the C++ implementation (transcription validation leg)
# ---------------------------------------------------------------------------


def _poly(coeffs: list[float], t: float) -> float:
    """Horner evaluation matching the nested-parenthesis ERFA/C++ form."""
    acc = coeffs[-1]
    for c in reversed(coeffs[:-1]):
        acc = c + t * acc
    return acc


def nut00b_mirror(t: float) -> tuple[float, float]:
    el = math.fmod(NUT00B_FA["el"][0] + NUT00B_FA["el"][1] * t, TURNAS) * DAS2R
    elp = math.fmod(NUT00B_FA["elp"][0] + NUT00B_FA["elp"][1] * t, TURNAS) * DAS2R
    f = math.fmod(NUT00B_FA["f"][0] + NUT00B_FA["f"][1] * t, TURNAS) * DAS2R
    d = math.fmod(NUT00B_FA["d"][0] + NUT00B_FA["d"][1] * t, TURNAS) * DAS2R
    om = math.fmod(NUT00B_FA["om"][0] + NUT00B_FA["om"][1] * t, TURNAS) * DAS2R
    dp = 0.0
    de = 0.0
    # Summation order fixed smallest-terms-first (reverse table order),
    # matching ERFA and cpp/src/frames.cpp (D-10 fixed evaluation order).
    for nl, nlp, nf, nd, nom, ps, pst, pc, ec, ect, es in reversed(NUT00B_TERMS):
        arg = math.fmod(nl * el + nlp * elp + nf * f + nd * d + nom * om, D2PI)
        sarg = math.sin(arg)
        carg = math.cos(arg)
        dp += (ps + pst * t) * sarg + pc * carg
        de += (ec + ect * t) * carg + es * sarg
    u2r = DAS2R / 1e7
    dpsi = dp * u2r + NUT00B_DPPLAN_MAS * DMAS2R
    deps = de * u2r + NUT00B_DEPLAN_MAS * DMAS2R
    return dpsi, deps


def fundamental_args_03(t: float) -> list[float]:
    fa = [
        math.fmod(_poly(FA_L, t), TURNAS) * DAS2R,
        math.fmod(_poly(FA_LP, t), TURNAS) * DAS2R,
        math.fmod(_poly(FA_F, t), TURNAS) * DAS2R,
        math.fmod(_poly(FA_D, t), TURNAS) * DAS2R,
        math.fmod(_poly(FA_OM, t), TURNAS) * DAS2R,
        math.fmod(FA_VE[0] + FA_VE[1] * t, D2PI),
        math.fmod(FA_E[0] + FA_E[1] * t, D2PI),
        (FA_PA[0] + FA_PA[1] * t) * t,
    ]
    return fa


def s06_mirror(t: float, x: float, y: float) -> float:
    fa = fundamental_args_03(t)
    w = list(S06_POLY)
    for order, terms in enumerate(
        (S06_TERMS_0, S06_TERMS_1, S06_TERMS_2, S06_TERMS_3, S06_TERMS_4)
    ):
        for nfa, sc, cc in reversed(terms):
            a = 0.0
            for j in range(8):
                a += nfa[j] * fa[j]
            w[order] += sc * math.sin(a) + cc * math.cos(a)
    s = (
        w[0] + (w[1] + (w[2] + (w[3] + (w[4] + w[5] * t) * t) * t) * t) * t
    ) * DAS2R - x * y / 2.0
    return s


def pfw06_mirror(t: float) -> tuple[float, float, float, float]:
    gamb = _poly(FW_GAMB, t) * DAS2R
    phib = _poly(FW_PHIB, t) * DAS2R
    psib = _poly(FW_PSIB, t) * DAS2R
    epsa = _poly(OBL06, t) * DAS2R
    return gamb, phib, psib, epsa


def r1_np(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])


def r2_np(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])


def r3_np(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def fw2m_mirror(gamb: float, phib: float, psi: float, eps: float) -> np.ndarray:
    return r1_np(-eps) @ r3_np(-psi) @ r1_np(phib) @ r3_np(gamb)


def c2ixys_mirror(x: float, y: float, s: float) -> np.ndarray:
    r2 = x * x + y * y
    e = math.atan2(y, x) if r2 > 0.0 else 0.0
    d = math.atan(math.sqrt(r2 / (1.0 - r2)))
    return r3_np(-(e + s)) @ r2_np(d) @ r3_np(e)


def era00_mirror(dj1: float, dj2: float) -> float:
    if dj1 < dj2:
        d1, d2 = dj1, dj2
    else:
        d1, d2 = dj2, dj1
    t = d1 + (d2 - DJ00)
    f = math.fmod(d1, 1.0) + math.fmod(d2, 1.0)
    theta = math.fmod(D2PI * (f + ERA_C0 + ERA_C1 * t), D2PI)
    if theta < 0.0:
        theta += D2PI
    return theta


def chain_mirror(tt1: float, tt2: float, ut11: float, ut12: float):
    t = (tt1 - DJ00 + tt2) / DJC
    dpsi, deps = nut00b_mirror(t)
    gamb, phib, psib, epsa = pfw06_mirror(t)
    rnpb = fw2m_mirror(gamb, phib, psib + dpsi, epsa + deps)
    x = rnpb[2, 0]
    y = rnpb[2, 1]
    s = s06_mirror(t, x, y)
    rc2i = c2ixys_mirror(x, y, s)
    era = era00_mirror(ut11, ut12)
    c = r3_np(era) @ rc2i
    return dpsi, deps, x, y, s, era, c


# ---------------------------------------------------------------------------
# ERFA leg (the committed values)
# ---------------------------------------------------------------------------


def chain_erfa(tt1: float, tt2: float, ut11: float, ut12: float):
    dpsi, deps = erfa.nut00b(tt1, tt2)
    rbpn = erfa.pn06(tt1, tt2, dpsi, deps)[-1]
    x, y = erfa.bpn2xy(np.asarray(rbpn))
    s = erfa.s06(tt1, tt2, x, y)
    rc2i = erfa.c2ixys(x, y, s)
    era = erfa.era00(ut11, ut12)
    c = np.asarray(erfa.rz(era, np.asarray(rc2i)))
    return float(dpsi), float(deps), float(x), float(y), float(s), float(era), c


def erfa_times(y, mo, d, h, mi, s, dut1):
    with warnings.catch_warnings():
        # Epochs past ERFA's built-in leap table draw "dubious year"
        # ErfaWarning; ERFA then assumes TAI-UTC stays 37 s, which is
        # exactly the bundled table's documented post-expiry assumption
        # (tests/golden/time/manifest.toml), so the values remain the
        # correct cross-check.
        warnings.simplefilter("ignore", erfa.core.ErfaWarning)
        utc1, utc2 = erfa.dtf2d("UTC", y, mo, d, h, mi, s)
        tai1, tai2 = erfa.utctai(utc1, utc2)
        tt1, tt2 = erfa.taitt(tai1, tai2)
        ut11, ut12 = erfa.utcut1(utc1, utc2, dut1)
    return float(tt1), float(tt2), float(ut11), float(ut12)


# ---------------------------------------------------------------------------
# Golden epochs
# ---------------------------------------------------------------------------

# name, (y, mo, d, h, mi, s UTC), dut1 [s]. Fractional seconds are dyadic
# (exact binary64). 14 epochs spanning 2020-2060 including both endpoints
# of the mission span and nonzero constant-dUT1 cases (FR-3: user-
# suppliable constant dUT1, default 0; |dUT1| < 0.9 s by UTC definition).
EARTH_EPOCHS = [
    ("epoch_2020_01_01", (2020, 1, 1, 0, 0, 0.0), 0.0),
    ("epoch_2022_03_20", (2022, 3, 20, 6, 30, 0.25), 0.0),
    ("epoch_2025_07_04", (2025, 7, 4, 18, 0, 0.0), 0.0),
    ("epoch_2026_06_30_expiry_edge", (2026, 6, 30, 23, 59, 59.0), 0.0),
    ("epoch_2028_11_11", (2028, 11, 11, 11, 11, 11.5), 0.0),
    ("epoch_2032_02_29_leapyear", (2032, 2, 29, 12, 0, 0.0), 0.0),
    ("epoch_2036_09_15_dut1_pos", (2036, 9, 15, 3, 45, 0.0), 0.1),
    ("epoch_2040_01_01", (2040, 1, 1, 0, 0, 0.0), 0.0),
    ("epoch_2044_05_05_dut1_neg", (2044, 5, 5, 20, 20, 20.125), -0.3),
    ("epoch_2048_12_31", (2048, 12, 31, 23, 59, 59.5), 0.0),
    ("epoch_2052_06_15_dut1_pos", (2052, 6, 15, 9, 0, 0.0), 0.25),
    ("epoch_2056_10_01", (2056, 10, 1, 15, 30, 0.0), 0.0),
    ("epoch_2060_01_01", (2060, 1, 1, 0, 0, 0.0), 0.0),
    ("epoch_2060_12_31", (2060, 12, 31, 23, 59, 59.0), 0.0),
]

# Moon principal-axis test angles (phi, theta, psi) [rad], the DE 3-1-3
# Euler angles of C_GCRF->MoonPA = R3(psi) R1(theta) R3(phi). The
# "libration_magnitude" cases use representative DE440 magnitudes (phi
# small, theta near 0.4 rad, psi unbounded); the rest exercise generic and
# 3-1-3-degenerate geometry. The ephemeris evaluator that supplies real
# angles is a separate workstream; these committed sets validate only the
# rotation assembly.
MOON_CASES = [
    ("identity", (0.0, 0.0, 0.0)),
    ("libration_magnitude_1", (0.005, 0.4, 2.9)),
    ("libration_magnitude_2", (-0.002, 0.38, -1.5)),
    ("libration_magnitude_3", (0.0071, 0.4108, 0.6)),
    ("generic", (2.2, 1.2, 4.0)),
    ("theta_near_zero", (0.3, 1e-3, -0.2)),
    ("theta_near_pi", (1.1, math.pi - 1e-3, 0.5)),
]

MARS_EPOCH_NAMES = [
    "epoch_2020_01_01",
    "epoch_2025_07_04",
    "epoch_2032_02_29_leapyear",
    "epoch_2040_01_01",
    "epoch_2052_06_15_dut1_pos",
    "epoch_2060_12_31",
]

# ---------------------------------------------------------------------------
# Mars IAU 2015 rotational elements (Archinal et al. 2018, CMDA 130:22,
# post-erratum values as distributed in NAIF pck00011.tpc, BODY499 blocks).
# Angles in degrees; T = TDB Julian centuries since J2000, d = TDB days
# since J2000 (the report's standard epoch is J2000.0 = JD 2451545.0 TDB).
# ---------------------------------------------------------------------------

MARS_RA_POLY = (317.269202, -0.10927547)
MARS_RA_TRIG = [  # (amplitude_deg, phase_deg, rate_deg_per_century), sin
    (0.000068, 198.991226, 19139.4819985),
    (0.000238, 226.292679, 38280.8511281),
    (0.000052, 249.663391, 57420.7251593),
    (0.000009, 266.183510, 76560.6367950),
    (0.419057, 79.398797, 0.5042615),
]
MARS_DEC_POLY = (54.432516, -0.05827105)
MARS_DEC_TRIG = [  # cos terms
    (0.000051, 122.433576, 19139.9407476),
    (0.000141, 43.058401, 38280.8753272),
    (0.000031, 57.663379, 57420.7517205),
    (0.000005, 79.476401, 76560.6495004),
    (1.591274, 166.325722, 0.5042615),
]
MARS_W_POLY = (176.049863, 350.891982443297)  # + rate * d (days)
MARS_W_TRIG = [  # sin terms
    (0.000145, 129.071773, 19140.0328244),
    (0.000157, 36.352167, 38281.0473591),
    (0.000040, 56.668646, 57420.9295360),
    (0.000001, 67.364003, 76560.2552215),
    (0.000001, 104.792680, 95700.4387578),
    (0.584542, 95.391654, 0.5042615),
]


def mars_elements_mirror(d_days: float) -> tuple[float, float, float]:
    """alpha0, delta0, W in radians; evaluation order mirrors frames.cpp."""
    t = d_days / DJC
    ra = MARS_RA_POLY[0] + MARS_RA_POLY[1] * t
    for amp, phase, rate in MARS_RA_TRIG:
        ra += amp * math.sin(math.fmod(phase + rate * t, 360.0) * D2R)
    dec = MARS_DEC_POLY[0] + MARS_DEC_POLY[1] * t
    for amp, phase, rate in MARS_DEC_TRIG:
        dec += amp * math.cos(math.fmod(phase + rate * t, 360.0) * D2R)
    w = math.fmod(MARS_W_POLY[0] + MARS_W_POLY[1] * d_days, 360.0)
    for amp, phase, rate in MARS_W_TRIG:
        w += amp * math.sin(math.fmod(phase + rate * t, 360.0) * D2R)
    return ra * D2R, dec * D2R, w * D2R


def tdb_jd_two_part(day: int, sec: float) -> tuple[float, float]:
    """Two-part TDB JD exactly as star::time::tdb_jd folds the series."""
    jd1, jd2 = time_ref.tt_jd(day, sec)
    frac = jd2 + time_ref.tdb_minus_tt(day, sec) / 86400.0
    if frac >= 1.0:
        return jd1 + 1.0, frac - 1.0
    if frac < 0.0:
        return jd1 - 1.0, frac + 1.0
    return jd1, frac


# ---------------------------------------------------------------------------
# TOML emission (restricted subset read by cpp/tests/golden_io.hpp)
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


def mat_fields(m: np.ndarray, hexfmt: bool = True) -> dict:
    out = {}
    for i in range(3):
        for j in range(3):
            v = float(m[i, j])
            out[f"c{i}{j}"] = v.hex() if hexfmt else repr(v)
    return out


# ---------------------------------------------------------------------------
# Generation-time validation
# ---------------------------------------------------------------------------


def validate_transcription() -> dict:
    """Mirror vs ERFA over the golden epochs plus 300 seeded random epochs.

    The mirror performs the same operation sequence as ERFA, so residuals
    are libm-only (sub-1e-16 rad); any transcription error in a series
    table is orders of magnitude larger. Bounds are set at 1e-14 rad
    (angles) and 5e-14 (matrix elements) to be safely above libm spread
    and far below the smallest series coefficient (0.01 microarcsec
    ~ 5e-14 rad).
    """
    rng = random.Random(20260702)
    epochs = []
    for name, (y, mo, d, h, mi, s), dut1 in EARTH_EPOCHS:
        epochs.append(erfa_times(y, mo, d, h, mi, s, dut1))
    for _ in range(300):
        mjd = rng.uniform(58849.0, 73825.0)  # 2020-01-01 .. 2061-01-05
        epochs.append((2400000.5, mjd, 2400000.5, mjd - 0.0008))
    worst = {"dpsi": 0.0, "deps": 0.0, "x": 0.0, "y": 0.0, "s": 0.0, "era": 0.0, "mat": 0.0}
    for tt1, tt2, ut11, ut12 in epochs:
        e = chain_erfa(tt1, tt2, ut11, ut12)
        m = chain_mirror(tt1, tt2, ut11, ut12)
        for k, i in (("dpsi", 0), ("deps", 1), ("x", 2), ("y", 3), ("s", 4), ("era", 5)):
            worst[k] = max(worst[k], abs(e[i] - m[i]))
        worst["mat"] = max(worst["mat"], float(np.max(np.abs(e[6] - m[6]))))
    for k in ("dpsi", "deps", "x", "y", "s", "era"):
        assert worst[k] <= 1e-14, (k, worst[k])
    assert worst["mat"] <= 5e-14, worst["mat"]
    return worst


# Published values transcribed from the IAU SOFA cookbook "SOFA Tools for
# Earth Attitude" (C edition), section 5.5 "IAU 2006/2000A, CIO based,
# using classical angles", worked example 2007 April 05 12:00:00.0 UTC with
# UT1-UTC = -0.072073685 s and CIP corrections dX06 = +0.1750 mas,
# dY06 = -0.2259 mas. The matrix below is the printed "celestial to
# terrestrial matrix (no polar motion)"; X, Y include the dX06/dY06
# corrections.
COOKBOOK_UTC = (2007, 4, 5, 12, 0, 0.0)
COOKBOOK_DUT1 = -0.072073685
COOKBOOK_DX06_MAS = 0.1750
COOKBOOK_DY06_MAS = -0.2259
COOKBOOK_MATRIX = [
    ["+0.973104317573127", "+0.230363826247709", "-0.000703332818845"],
    ["-0.230363798804182", "+0.973104570735574", "+0.000120888549586"],
    ["+0.000712264729599", "+0.000044385250426", "+0.999999745354420"],
]
COOKBOOK_X = "+0.000712264729599"
COOKBOOK_Y = "+0.000044385250426"
COOKBOOK_S_ARCSEC = "-0.002200475"
COOKBOOK_ERA_DEG = "13.318492966097"


def main() -> None:
    worst = validate_transcription()

    # ---------------- earth_chain.toml ----------------
    earth_cases = []
    for name, (y, mo, d, h, mi, s), dut1 in EARTH_EPOCHS:
        eday, esec = time_ref.tai_epoch_from_utc(y, mo, d, h, mi, s)
        tt1, tt2, ut11, ut12 = erfa_times(y, mo, d, h, mi, s, dut1)
        dpsi, deps, x, yy, s06v, era, c = chain_erfa(tt1, tt2, ut11, ut12)
        case = {
            "name": name,
            "year": str(y),
            "month": str(mo),
            "day": str(d),
            "hour": str(h),
            "minute": str(mi),
            "second": float(s).hex(),
            "dut1_s": float(dut1).hex(),
            "tai_day": str(eday),
            "tai_sec": esec.hex(),
            "dpsi": dpsi.hex(),
            "deps": deps.hex(),
            "x": x.hex(),
            "y": yy.hex(),
            "s": s06v.hex(),
            "era": era.hex(),
        }
        case.update(mat_fields(c))
        earth_cases.append(case)

    emit(
        HERE / "earth_chain.toml",
        "GCRF -> ITRF chain golden vectors (FR-3), IAU 2006/2000B CIO based,\n"
        "polar motion neglected. All values ERFA-generated by the exact\n"
        "composition cpp/src/frames.cpp implements: dpsi,deps = nut00b(tt);\n"
        "rbpn = pn06(tt,dpsi,deps)[-1]; x,y = bpn2xy(rbpn); s = s06(tt,x,y);\n"
        "rc2i = c2ixys(x,y,s); era = era00(ut1); C = Rz(era) rc2i, with\n"
        "UT1 = UTC + dut1_s (constant). Angles in radians; c00..c22 is\n"
        "C_GCRF->ITRF row-major. tai_day/tai_sec is the two-part TAI epoch\n"
        "from the time-system reference implementation. Provenance and\n"
        "tolerances in manifest.toml. Regenerated by generate.py.",
        earth_cases,
    )

    # ---------------- cookbook_2006_2000a.toml ----------------
    # Compare the 2000B chain (this project's model, no CIP corrections)
    # against the published 2006/2000A + dX/dY matrix to measure the model
    # difference the test tolerance must cover.
    y, mo, d, h, mi, s = COOKBOOK_UTC
    eday, esec = time_ref.tai_epoch_from_utc(y, mo, d, h, mi, s)
    tt1, tt2, ut11, ut12 = erfa_times(y, mo, d, h, mi, s, COOKBOOK_DUT1)
    _, _, xb, yb, sb, erab, cb = chain_erfa(tt1, tt2, ut11, ut12)
    pub = np.array([[float(v) for v in row] for row in COOKBOOK_MATRIX])
    observed = float(np.max(np.abs(cb - pub)))
    # 2000B-vs-2000A is bounded by ~1 mas of CIP error (McCarthy & Luzum
    # 2003) ~ 4.8e-9 in matrix elements, plus the 0.29 mas applied dX/dY
    # corrections; 8e-9 covers both with margin and the observed value.
    bound = 8e-9
    assert observed <= bound, (observed, bound)

    cookbook_case = {
        "name": "sofa_cookbook_2007_04_05",
        "year": str(y),
        "month": str(mo),
        "day": str(d),
        "hour": str(h),
        "minute": str(mi),
        "second": float(s).hex(),
        "dut1_s": float(COOKBOOK_DUT1).hex(),
        "tai_day": str(eday),
        "tai_sec": esec.hex(),
        "pub_x": COOKBOOK_X,
        "pub_y": COOKBOOK_Y,
        "pub_s_arcsec": COOKBOOK_S_ARCSEC,
        "pub_era_deg": COOKBOOK_ERA_DEG,
        "tol_matrix": repr(bound),
        "observed_2000b_delta": repr(observed),
    }
    for i in range(3):
        for j in range(3):
            cookbook_case[f"pub_c{i}{j}"] = COOKBOOK_MATRIX[i][j]

    emit(
        HERE / "cookbook_2006_2000a.toml",
        "Published cross-check golden: IAU SOFA cookbook 'SOFA Tools for\n"
        "Earth Attitude', worked example 2007 April 05 12:00:00.0 UTC,\n"
        "section 5.5 (IAU 2006/2000A, CIO based). pub_c00..pub_c22 is the\n"
        "printed 'celestial to terrestrial matrix (no polar motion)';\n"
        "pub_x/pub_y (radians, include the cookbook's dX06/dY06 CIP\n"
        "corrections), pub_s_arcsec and pub_era_deg as printed. tol_matrix\n"
        "is the documented comparison bound for this project's 2006/2000B\n"
        "chain: the 2000B-vs-2000A model difference (<= ~1 mas CIP error,\n"
        "McCarthy & Luzum 2003) plus the applied 0.175/-0.226 mas CIP\n"
        "corrections -- deliberately far looser than the 1e-11 ERFA gate.\n"
        "Regenerated by generate.py.",
        [cookbook_case],
    )

    # ---------------- moon_pa.toml ----------------
    moon_cases = []
    max_moon = 0.0
    for name, (phi, theta, psi) in MOON_CASES:
        r = np.eye(3)
        r = np.asarray(erfa.rz(phi, r))
        r = np.asarray(erfa.rx(theta, r))
        r = np.asarray(erfa.rz(psi, r))  # R3(psi) R1(theta) R3(phi)
        indep = r3_np(psi) @ r1_np(theta) @ r3_np(phi)
        dmax = float(np.max(np.abs(r - indep)))
        assert dmax <= 1e-15, (name, dmax)
        max_moon = max(max_moon, dmax)
        case = {
            "name": name,
            "phi": float(phi).hex(),
            "theta": float(theta).hex(),
            "psi": float(psi).hex(),
        }
        case.update(mat_fields(r))
        moon_cases.append(case)

    emit(
        HERE / "moon_pa.toml",
        "Moon principal-axis frame golden vectors (FR-3). phi, theta, psi\n"
        "are the DE 3-1-3 libration Euler angles [rad]; c00..c22 is\n"
        "C_GCRF->MoonPA = R3(psi) R1(theta) R3(phi) (Park et al. 2021,\n"
        "DE440), assembled from ERFA rotation primitives (rz/rx) and\n"
        "cross-checked against an independent NumPy element-formula\n"
        "evaluation at generation time. Angle sets are committed test\n"
        "values (representative DE440 libration magnitudes plus degenerate\n"
        "geometry); the ephemeris evaluator supplying real angles is a\n"
        "separate workstream. Provenance and tolerances in manifest.toml.\n"
        "Regenerated by generate.py.",
        moon_cases,
    )

    # ---------------- mars_iau.toml ----------------
    mars_cases = []
    epoch_by_name = {name: (utc, dut1) for name, utc, dut1 in EARTH_EPOCHS}
    for name in MARS_EPOCH_NAMES:
        (y, mo, d, h, mi, s), _ = epoch_by_name[name]
        eday, esec = time_ref.tai_epoch_from_utc(y, mo, d, h, mi, s)
        jd1, jd2 = tdb_jd_two_part(eday, esec)
        d_days = (jd1 - DJ00) + jd2
        ra, dec, w = mars_elements_mirror(d_days)
        # Assembly via ERFA primitives: R3(W) R1(pi/2 - dec) R3(pi/2 + ra).
        r = np.eye(3)
        r = np.asarray(erfa.rz(math.pi / 2.0 + ra, r))
        r = np.asarray(erfa.rx(math.pi / 2.0 - dec, r))
        r = np.asarray(erfa.rz(w, r))
        # Cross-check 1: independent NumPy composition.
        indep = r3_np(w) @ r1_np(math.pi / 2.0 - dec) @ r3_np(math.pi / 2.0 + ra)
        assert float(np.max(np.abs(r - indep))) <= 1e-15, name
        # Cross-check 2: the body z-axis expressed in GCRF (third row of C,
        # i.e. C^T e_z) must be the published pole direction unit vector.
        pole = np.array(
            [
                math.cos(dec) * math.cos(ra),
                math.cos(dec) * math.sin(ra),
                math.sin(dec),
            ]
        )
        assert float(np.max(np.abs(r[2, :] - pole))) <= 1e-15, name
        case = {
            "name": name,
            "tai_day": str(eday),
            "tai_sec": esec.hex(),
            "tdb_jd1": float(jd1).hex(),
            "tdb_jd2": float(jd2).hex(),
            "alpha0": float(ra).hex(),
            "delta0": float(dec).hex(),
            "w": float(w).hex(),
        }
        case.update(mat_fields(r))
        mars_cases.append(case)

    emit(
        HERE / "mars_iau.toml",
        "Mars IAU 2015 body-fixed frame golden vectors (FR-3). alpha0,\n"
        "delta0 (pole RA/Dec) and prime meridian W [rad] evaluated here\n"
        "independently from the published rotational-element polynomials\n"
        "(Archinal et al. 2018, CMDA 130:22; post-erratum values as\n"
        "distributed in NAIF pck00011.tpc), with time argument TDB days /\n"
        "centuries since J2000 via the D-6 TDB series. c00..c22 is\n"
        "C_GCRF->MarsFixed = R3(W) R1(pi/2-delta0) R3(pi/2+alpha0)\n"
        "assembled from ERFA rotation primitives, cross-checked against an\n"
        "independent NumPy composition and the explicit pole unit vector.\n"
        "Provenance and tolerances in manifest.toml. Regenerated by\n"
        "generate.py.",
        mars_cases,
    )

    # ---------------- series_terms.toml ----------------
    term_cases = []
    for i, (nl, nlp, nf, nd, nom, ps, pst, pc, ec, ect, es) in enumerate(NUT00B_TERMS):
        term_cases.append(
            {
                "name": f"nut00b_ls_{i + 1:03d}",
                "nl": str(nl),
                "nlp": str(nlp),
                "nf": str(nf),
                "nd": str(nd),
                "nom": str(nom),
                "ps": repr(ps),
                "pst": repr(pst),
                "pc": repr(pc),
                "ec": repr(ec),
                "ect": repr(ect),
                "es": repr(es),
            }
        )
    for order, terms in enumerate(
        (S06_TERMS_0, S06_TERMS_1, S06_TERMS_2, S06_TERMS_3, S06_TERMS_4)
    ):
        for i, (nfa, sc, cc) in enumerate(terms):
            case = {"name": f"s06_t{order}_{i + 1:02d}"}
            for j, label in enumerate(("l", "lp", "f", "d", "om", "ve", "e", "pa")):
                case[f"n{label}"] = str(nfa[j])
            case["sc"] = repr(sc)
            case["cc"] = repr(cc)
            term_cases.append(case)

    def poly_case(name: str, coeffs: list[float]) -> dict:
        return {"name": name, "coeffs": [repr(c) for c in coeffs]}

    term_cases.append(poly_case("s06_poly", S06_POLY))
    term_cases.append(poly_case("fw_gamb", FW_GAMB))
    term_cases.append(poly_case("fw_phib", FW_PHIB))
    term_cases.append(poly_case("fw_psib", FW_PSIB))
    term_cases.append(poly_case("obl06", OBL06))
    term_cases.append(poly_case("fa_l", FA_L))
    term_cases.append(poly_case("fa_lp", FA_LP))
    term_cases.append(poly_case("fa_f", FA_F))
    term_cases.append(poly_case("fa_d", FA_D))
    term_cases.append(poly_case("fa_om", FA_OM))
    term_cases.append(poly_case("fa_ve", FA_VE))
    term_cases.append(poly_case("fa_e", FA_E))
    term_cases.append(poly_case("fa_pa", FA_PA))
    term_cases.append(poly_case("nutfa_l", NUT00B_FA["el"]))
    term_cases.append(poly_case("nutfa_lp", NUT00B_FA["elp"]))
    term_cases.append(poly_case("nutfa_f", NUT00B_FA["f"]))
    term_cases.append(poly_case("nutfa_d", NUT00B_FA["d"]))
    term_cases.append(poly_case("nutfa_om", NUT00B_FA["om"]))
    term_cases.append(
        {
            "name": "nut00b_planetary_offsets_mas",
            "dpplan": repr(NUT00B_DPPLAN_MAS),
            "deplan": repr(NUT00B_DEPLAN_MAS),
        }
    )
    term_cases.append(
        {"name": "era00_coeffs", "c0": repr(ERA_C0), "c1": repr(ERA_C1)}
    )

    emit(
        HERE / "series_terms.toml",
        "Transcribed IAU 2006/2000B series tables (FR-3): the 77-term IAU\n"
        "2000B luni-solar nutation series (0.1 microarcsec units), the\n"
        "planetary-bias offsets (mas), the s(X,Y) series (arcsec), the IAU\n"
        "2006 Fukushima-Williams / mean-obliquity polynomials (arcsec), the\n"
        "fundamental-argument polynomials, and the ERA constants. Values\n"
        "are decimal shortest-repr strings (exact binary64 round trip).\n"
        "The C++ compiled tables in cpp/src/frames_series.hpp must equal\n"
        "these exactly (term-by-term doctest); the tables here are\n"
        "validated functionally against erfa.nut00b/s06/pfw06/obl06/era00\n"
        "at every golden epoch plus 300 seeded random epochs at generation\n"
        "time. Regenerated by generate.py.",
        term_cases,
    )

    print("frames golden files regenerated and cross-checked")
    print(f"pyerfa {erfa.__version__} (ERFA {erfa.version.erfa_version})")
    print(
        "transcription-mirror maxima vs ERFA [rad]: "
        + ", ".join(f"{k} {v:.3e}" for k, v in worst.items())
    )
    print(
        f"cookbook 2000B-vs-published max element delta: {observed:.3e} "
        f"(documented bound {bound:.1e})"
    )
    print(f"moon PA cross-check max element delta: {max_moon:.3e}")


if __name__ == "__main__":
    main()
