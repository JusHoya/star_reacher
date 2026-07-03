"""Regenerate the gravity golden set in this directory (FR-5, FR-22 layer 1).

Maintainer-side; CI and pytest consume only the committed outputs. Inputs are
the checksummed source coefficient files that ``star data fetch egm2008 |
grgm1200a | mro120f`` leaves in ``data/`` (network only if they are absent).
The script:

1. verifies each source file against the SHA-256 pins in
   ``star_reacher.data_fetch`` (downloading via the fetch pipeline if a
   source is missing);
2. writes the committed coefficient excerpts: Earth EGM2008 truncated to
   20x20 (the same excerpt serves the 8x8 cross-tool case by runtime
   truncation), Moon GRGM1200A to 50x50, Mars MRO120F to 20x20 -- each as a
   full-precision CSV (``*_n<degree>.csv``, human-auditable provenance form)
   and as an SRGRAV v1 binary (``*_n<degree>.srgrav``, the form the C++ core
   loads; docs/formats/srgrav_v1.md). The binary is written from the CSV's
   parsed values so the two committed forms are provably the same data;
3. self-checks pyshtools' MakeGravGridPoint conventions against the closed
   form on a monopole-only field (returns the gravity vector's spherical
   components (g_r, g_theta, g_phi); g_r must equal -GM/r^2 to bit level);
4. evaluates the 20 Phase 3 exit-criterion-1 test states with pyshtools
   MakeGravGridPoint over the SAME committed excerpt coefficients and writes
   ``pyshtools_accel.toml``: body-fixed position and acceleration as binary64
   hex literals. pyshtools is the independent synthesis (different author,
   different algorithm - colatitude Legendre recursion vs the core's Pines
   formulation), satisfying the fresh-point independence rule;
5. cross-checks every golden acceleration against a pure-Python mirror of
   the Pines recursions and records the worst relative difference (a
   generation-time honesty check only; the committed values are pyshtools').

Running it rewrites the outputs deterministically apart from the recorded
generation date.
"""

from __future__ import annotations

import datetime
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from star_reacher import data_fetch as df  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
GENERATION_DATE = datetime.date.today().isoformat()

# Committed excerpt truncation degrees (square: order = degree). Earth >= 20
# also feeds the 8x8 cross-tool states by runtime truncation; Moon >= 50 and
# Mars >= 20 match the exit-criterion state list below.
EXCERPTS = {
    "egm2008": ("earth_egm2008_n20", 20),
    "grgm1200a": ("moon_grgm1200a_n50", 50),
    "mro120f": ("mars_mro120f_n20", 20),
}

# The 20 exit-criterion-1 test states: (body, dataset, n_eval, lat_deg,
# lon_deg, r_m). Latitudes include +/-89.9-class values near both poles to
# exercise the Pines formulation where associated-Legendre-over-longitude
# formulations lose their footing; radii span low orbit to GEO/areosynchronous
# class. All lat/lon/r values are short decimals, exactly representable
# choices are not required because the committed positions are binary64 hex
# of the values actually used on both sides.
TEST_STATES = [
    ("earth", "egm2008", 8, 0.0, 0.0, 6778137.0),
    ("earth", "egm2008", 8, 45.0, -75.0, 6878137.0),
    ("earth", "egm2008", 8, 89.9, 33.0, 6778137.0),
    ("earth", "egm2008", 8, -30.0, 150.0, 42164169.0),
    ("earth", "egm2008", 20, 10.0, 100.0, 6678137.0),
    ("earth", "egm2008", 20, -89.95, -120.0, 6778137.0),
    ("earth", "egm2008", 20, 63.4349, 0.0, 26560000.0),
    ("earth", "egm2008", 20, 28.5, -80.6, 6478137.0),
    ("moon", "grgm1200a", 50, 0.0, 0.0, 1838000.0),
    ("moon", "grgm1200a", 50, 89.9, 45.0, 1788000.0),
    ("moon", "grgm1200a", 50, -89.9, -10.0, 1838000.0),
    ("moon", "grgm1200a", 50, -45.0, 170.0, 1938000.0),
    ("moon", "grgm1200a", 50, 20.0, -60.0, 2538000.0),
    ("moon", "grgm1200a", 50, -75.0, 80.0, 1788000.0),
    ("mars", "mro120f", 20, 0.0, 0.0, 3596000.0),
    ("mars", "mro120f", 20, 89.9, 120.0, 3696000.0),
    ("mars", "mro120f", 20, -89.95, -45.0, 3796000.0),
    ("mars", "mro120f", 20, -25.0, 135.0, 3496000.0),
    ("mars", "mro120f", 20, 55.0, -170.0, 4396000.0),
    ("mars", "mro120f", 20, -60.0, 60.0, 20428000.0),
]


def ensure_sources() -> dict[str, pathlib.Path]:
    """Verify (or fetch) the three pinned source files; return their paths."""
    paths: dict[str, pathlib.Path] = {}
    for dataset, spec in df.GRAVITY_DATASETS.items():
        src = DATA_DIR / spec.source_filename
        if not src.is_file():
            print(f"{spec.source_filename}: absent; running the fetch pipeline")
            df.fetch_gravity(dataset, DATA_DIR)
        got = df._sha256(src)
        if got != spec.source_sha256:
            raise SystemExit(
                f"{src}: SHA-256 mismatch (expected {spec.source_sha256}, got "
                f"{got}); refusing to generate goldens from unverified bytes"
            )
        paths[dataset] = src
    return paths


def parse_source(dataset: str, path: pathlib.Path, n_keep: int) -> df.GravityCoefficients:
    spec = df.GRAVITY_DATASETS[dataset]
    if spec.source_format == "icgem_gfc":
        field = df.parse_icgem_gfc(path, n_keep, spec.source_sha256)
        field.name = spec.field_name
    else:
        field = df.parse_pds_shadr(
            path, n_keep, spec.field_name, spec.tide_system or "unknown", spec.source_sha256
        )
    if spec.tide_system is not None:
        field.tide_system = spec.tide_system
    return field


def write_excerpt_csv(path: pathlib.Path, dataset: str, field: df.GravityCoefficients) -> None:
    spec = df.GRAVITY_DATASETS[dataset]
    lines = [
        f"# {field.name} coefficient excerpt, truncated to {field.n_max}x{field.m_max}",
        "# (FR-5 golden fixture; provenance in manifest.toml; regenerated by",
        "# generate.py in this directory). Values are the source file's fully",
        "# normalized C-bar/S-bar coefficients, printed with repr() so parsing",
        "# recovers the identical binary64. Metadata keys below are consumed by",
        "# star_reacher.data_fetch.read_coeffs_csv.",
        f"# name = {field.name}",
        f"# gm_m3ps2 = {field.gm_m3ps2!r}",
        f"# ref_radius_m = {field.ref_radius_m!r}",
        f"# n_max = {field.n_max}",
        f"# m_max = {field.m_max}",
        f"# tide_system = {field.tide_system}",
        f"# source_sha256 = {field.source_sha256}",
        f"# source_url = {spec.url}",
        f"# retrieval_date = {GENERATION_DATE}",
        "n,m,cbar,sbar",
    ]
    for n in range(field.n_max + 1):
        for m in range(min(n, field.m_max) + 1):
            lines.append(f"{n},{m},{float(field.cbar[n, m])!r},{float(field.sbar[n, m])!r}")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def cart_pos(lat_deg: float, lon_deg: float, r_m: float) -> np.ndarray:
    """Body-fixed Cartesian position for geocentric lat/lon/radius.

    This exact operation sequence defines the committed positions; the C++
    test consumes the resulting hex values directly, so no trigonometric
    reconstruction happens on the consuming side.
    """
    phi = math.radians(lat_deg)
    lam = math.radians(lon_deg)
    return np.array(
        [
            r_m * math.cos(phi) * math.cos(lam),
            r_m * math.cos(phi) * math.sin(lam),
            r_m * math.sin(phi),
        ]
    )


def sph_to_cart(g_sph, lat_deg: float, lon_deg: float) -> np.ndarray:
    """Map MakeGravGridPoint's (g_r, g_theta, g_phi) to body-fixed Cartesian.

    theta-hat points along increasing colatitude (southward), phi-hat east,
    r-hat outward - the standard physics spherical triad, confirmed by the
    monopole self-check in main().
    """
    phi = math.radians(lat_deg)
    lam = math.radians(lon_deg)
    theta = 0.5 * math.pi - phi
    st, ct = math.sin(theta), math.cos(theta)
    sl, cl = math.sin(lam), math.cos(lam)
    rhat = np.array([st * cl, st * sl, ct])
    that = np.array([ct * cl, ct * sl, -st])
    phat = np.array([-sl, cl, 0.0])
    return g_sph[0] * rhat + g_sph[1] * that + g_sph[2] * phat


def cilm_from(field: df.GravityCoefficients, lmax: int) -> np.ndarray:
    cilm = np.zeros((2, lmax + 1, lmax + 1))
    cilm[0] = field.cbar[: lmax + 1, : lmax + 1]
    cilm[1] = field.sbar[: lmax + 1, : lmax + 1]
    return cilm


def pines_mirror(field: df.GravityCoefficients, r_bf: np.ndarray, n_eval: int, m_eval: int) -> np.ndarray:
    """Pure-Python mirror of the C++ Pines evaluator (generation-time check).

    Same recursions and summation order as cpp/src/models/gravity.cpp
    (eq:gravity:* labels in the gravity chapter); used only to cross-check
    the pyshtools goldens at generation time, never as the golden source.
    """
    x, y, z = (float(r_bf[0]), float(r_bf[1]), float(r_bf[2]))
    r = math.sqrt(x * x + y * y + z * z)
    inv_r = 1.0 / r
    s, t, u = x * inv_r, y * inv_r, z * inv_r
    N = n_eval
    A = np.zeros((N + 1, N + 1))
    A[0, 0] = 1.0
    for m in range(1, N + 1):
        f = math.sqrt(3.0) if m == 1 else math.sqrt((2.0 * m + 1.0) / (2.0 * m))
        A[m, m] = f * A[m - 1, m - 1]
    for m in range(0, N):
        A[m + 1, m] = u * math.sqrt(2.0 * m + 3.0) * A[m, m]
    for m in range(0, N + 1):
        for n in range(m + 2, N + 1):
            c1 = math.sqrt((2.0 * n - 1.0) * (2.0 * n + 1.0) / ((n - m) * (n + m)))
            c2 = math.sqrt(
                (2.0 * n + 1.0)
                * (n + m - 1.0)
                * (n - m - 1.0)
                / ((2.0 * n - 3.0) * (n + m) * (n - m))
            )
            A[n, m] = c1 * u * A[n - 1, m] - c2 * A[n - 2, m]
    rm = np.zeros(N + 1)
    im = np.zeros(N + 1)
    rm[0] = 1.0
    for m in range(1, m_eval + 1):
        rm[m] = s * rm[m - 1] - t * im[m - 1]
        im[m] = s * im[m - 1] + t * rm[m - 1]
    rr = field.ref_radius_m * inv_r
    rho = np.zeros(N + 1)
    rho[0] = field.gm_m3ps2 * inv_r
    for n in range(1, N + 1):
        rho[n] = rho[n - 1] * rr
    a1 = a2 = a3 = a4 = 0.0
    for n in range(0, N + 1):
        fac = rho[n] * inv_r
        for m in range(0, min(n, m_eval) + 1):
            cnm = float(field.cbar[n, m])
            snm = float(field.sbar[n, m])
            anm = A[n, m]
            if m < n:
                delta = 0.5 if m == 0 else 1.0
                aprime = math.sqrt((n - m) * (n + m + 1.0) * delta) * A[n, m + 1]
            else:
                aprime = 0.0
            dnm = cnm * rm[m] + snm * im[m]
            if m > 0:
                enm = cnm * rm[m - 1] + snm * im[m - 1]
                fnm = snm * rm[m - 1] - cnm * im[m - 1]
                a1 += fac * m * anm * enm
                a2 += fac * m * anm * fnm
            a3 += fac * aprime * dnm
            a4 += fac * ((n + m + 1) * anm + u * aprime) * dnm
    return np.array([a1 - s * a4, a2 - t * a4, a3 - u * a4])


def main() -> None:
    import pyshtools
    from pyshtools.backends.shtools import MakeGravGridPoint

    ensure_sources()

    # Committed excerpts: parse each source to its excerpt degree and write
    # CSV, then re-read the CSV and write the SRGRAV binary from the re-read
    # values so the two committed forms are the same data by construction.
    fields: dict[str, df.GravityCoefficients] = {}
    for dataset, (stem, n_keep) in EXCERPTS.items():
        src = DATA_DIR / df.GRAVITY_DATASETS[dataset].source_filename
        field = parse_source(dataset, src, n_keep)
        csv_path = HERE / f"{stem}.csv"
        write_excerpt_csv(csv_path, dataset, field)
        reread = df.read_coeffs_csv(csv_path)
        if not (
            np.array_equal(reread.cbar, field.cbar)
            and np.array_equal(reread.sbar, field.sbar)
            and reread.gm_m3ps2 == field.gm_m3ps2
            and reread.ref_radius_m == field.ref_radius_m
        ):
            raise SystemExit(f"{csv_path}: CSV round trip is not bit-exact")
        df.write_srgrav(HERE / f"{stem}.srgrav", reread)
        fields[dataset] = reread
        print(f"wrote {stem}.csv / {stem}.srgrav ({n_keep}x{n_keep})")

    # Monopole convention self-check: pins MakeGravGridPoint's output triad
    # (g_r, g_theta, g_phi) and its sign convention before any golden is
    # written from it.
    probe = fields["egm2008"]
    cilm00 = np.zeros((2, 1, 1))
    cilm00[0, 0, 0] = 1.0
    r_probe = 7000000.0
    g = MakeGravGridPoint(
        cilm00, probe.gm_m3ps2, probe.ref_radius_m, r_probe, 30.0, 40.0, lmax=0, omega=0.0
    )
    expect = -probe.gm_m3ps2 / (r_probe * r_probe)
    if g[0] != expect or g[1] != 0.0 or g[2] != 0.0:
        raise SystemExit(
            f"MakeGravGridPoint convention check failed: got {g}, expected "
            f"({expect}, 0, 0)"
        )

    # Exit-criterion-1 goldens: pyshtools point evaluation over the SAME
    # committed excerpt coefficients at the 20 states.
    lines = [
        "# Independently synthesized spherical-harmonic gravity accelerations",
        "# (FR-5, Phase 3 exit criterion 1) at 20 body-fixed states, produced",
        f"# by pyshtools {pyshtools.__version__} MakeGravGridPoint over the",
        "# committed coefficient excerpts in this directory and mapped to",
        "# Cartesian body-fixed axes. Positions and accelerations are binary64",
        "# hex literals; the consuming doctest (GRAV-XTOOL-20) evaluates the",
        "# core's Pines implementation at the committed positions and requires",
        "# < 1e-12 relative agreement on the acceleration vector. Provenance",
        "# and tolerances in manifest.toml. Regenerated by generate.py.",
    ]
    worst_mirror = 0.0
    for idx, (body, dataset, n_eval, lat, lon, r_m) in enumerate(TEST_STATES, start=1):
        stem, n_keep = EXCERPTS[dataset]
        field = fields[dataset]
        cilm = cilm_from(field, n_eval)
        g_sph = MakeGravGridPoint(
            cilm, field.gm_m3ps2, field.ref_radius_m, r_m, lat, lon, lmax=n_eval, omega=0.0
        )
        g_cart = sph_to_cart(g_sph, lat, lon)
        pos = cart_pos(lat, lon, r_m)
        mirror = pines_mirror(field, pos, n_eval, n_eval)
        rel = float(np.linalg.norm(mirror - g_cart) / np.linalg.norm(g_cart))
        worst_mirror = max(worst_mirror, rel)
        lines += [
            "",
            "[[case]]",
            f'name = "state_{idx:02d}_{body}_{n_eval}x{n_eval}"',
            f'body = "{body}"',
            f'field_file = "{stem}.srgrav"',
            f'n_eval = "{n_eval}"',
            f'm_eval = "{n_eval}"',
            f'lat_deg = "{lat!r}"',
            f'lon_deg = "{lon!r}"',
            f'radius_m = "{r_m!r}"',
            "r_bf_m = [",
            *[f'  "{float(c).hex()}",' for c in pos],
            "]",
            "accel_mps2 = [",
            *[f'  "{float(c).hex()}",' for c in g_cart],
            "]",
        ]
    (HERE / "pyshtools_accel.toml").write_text("\n".join(lines) + "\n", encoding="ascii")
    print(f"wrote pyshtools_accel.toml (20 states)")
    print(
        f"generation-time Pines-mirror cross-check worst relative difference: "
        f"{worst_mirror:.3e} (gate for the committed test is 1e-12)"
    )


if __name__ == "__main__":
    main()
