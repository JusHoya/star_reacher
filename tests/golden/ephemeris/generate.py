"""Regenerate the ephemeris golden set in this directory (FR-4, D-8, FR-22).

Maintainer-side and network-using; CI and `pytest` consume only the committed
outputs. The script:

1. runs the `star data fetch de440s` pipeline (download + SHA-256 pin +
   verbatim Chebyshev repack into ``data/de440s_2020_2060.sreph``);
2. chooses the test epochs: the repacked span endpoints, 17 TDB midnights
   spread across 2020-2060, and two interior Chebyshev record boundaries per
   body (>= 21 epochs per body, satisfying the Phase 2 exit criterion 2
   epoch requirements). All epochs are TDB midnights, exactly representable
   in binary64 both as Julian dates and as seconds since J2000, so no epoch
   quantization enters the comparison;
3. writes ``excerpt_de440s.sreph``: the same SREPH v1 format restricted to
   the records containing the test epochs (plus each boundary's left
   neighbor for continuity checks), so every committed test runs offline;
4. writes ``state_bitlevel.toml``: evaluator outputs at selected epochs as
   binary64 hex literals, produced by the Python reference evaluator in
   ``star_reacher.data_fetch`` (the executable spec of the C++ evaluator)
   and cross-checked against jplephem's independent evaluation;
5. queries JPL Horizons (geometric ICRF states, TDB epochs) for every
   repacked quantity, saving the exact transcripts under ``horizons/`` and
   the parsed vectors in ``horizons_vectors.toml``;
6. writes ``librations_jplephem.toml``: lunar libration angles/rates
   evaluated by jplephem (independent implementation) at the libration test
   epochs;
7. runs the full-span validation of the complete repack against Horizons and
   writes ``full_span_validation.md`` with the max error per body;
8. rewrites ``manifest.toml`` with provenance for every file.

Running it rewrites the outputs deterministically apart from the recorded
generation date and any upstream Horizons ephemeris update.
"""

from __future__ import annotations

import datetime
import math
import pathlib
import sys
import urllib.parse
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from star_reacher import data_fetch as df  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
GENERATION_DATE = datetime.date.today().isoformat()

J2000_JD = 2451545.0  # J2000 epoch, JD 2451545.0 TDB (2000-01-01T12:00 TDB)

# Cross-check tolerances (reference evaluator vs jplephem, same coefficients,
# different summation order). Measured worst case over all bodies/epochs on
# 2026-07-02: 1.4e-4 m, 1.7e-11 m/s, 9.1e-13 rad, 8.5e-22 rad/s - pure
# floating-point rounding at planetary distance scales. Bounds sit two to
# three orders above the measurement.
XCHECK_POS_M = 1e-2
XCHECK_VEL_MPS = 1e-8
XCHECK_ANG_RAD = 1e-10
XCHECK_RATE_RADPS = 1e-19

# Quantities validated against Horizons: (name, command, center, composer,
# horizons_gate_m). `composer` names how the repack reproduces the Horizons
# quantity; the center choice makes both sides the same geometric vector in
# the ICRF. The gate is 1 m (Phase 2 exit criterion 2) except for the two
# lunar quantities: Horizons serves the Moon from DE441 (the transcripts
# record "{source: DE441}"), whose lunar orbit differs from DE440 by design -
# < 2 m over 1970-2020 and ~10 m per century away from present, along-track,
# because DE441 omits the lunar core-mantle tidal damping term (Park et al.
# 2021, AJ 161:105, Section 6). A bit-faithful DE440 repack therefore cannot
# match DE441-sourced lunar states below that floor; the measured 1.8-5.4 m
# over 2020-2060 is that published difference. Lunar DE440 fidelity is gated
# instead against jplephem's independent evaluation of the checksummed DE440
# kernel (< 1 mm, moon_de440_jplephem.toml), with Horizons kept as a 10 m
# DE441-envelope sanity bound.
GATE_M = 1.0
LUNAR_ENVELOPE_M = 10.0
QUANTITIES = (
    ("sun", "10", "500@0", "state:sun", GATE_M),
    ("emb", "3", "500@0", "state:emb", GATE_M),
    ("venus_bary", "2", "500@0", "state:venus_bary", GATE_M),
    ("mars_bary", "4", "500@0", "state:mars_bary", GATE_M),
    ("jupiter_bary", "5", "500@0", "state:jupiter_bary", GATE_M),
    ("earth", "399", "500@3", "state:earth", GATE_M),
    ("moon", "301", "500@3", "state:moon", LUNAR_ENVELOPE_M),
    ("moon_geocentric", "301", "500@399", "moon_minus_earth", LUNAR_ENVELOPE_M),
)

HORIZONS_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"


# ---------------------------------------------------------------------------
# Epoch selection
# ---------------------------------------------------------------------------


def shared_epochs(sreph: df.SrephFile) -> list[float]:
    """Span endpoints plus 17 interior TDB midnights spread over the span."""
    t_lo = sreph.span_start_tdb_s
    t_hi = sreph.span_end_tdb_s
    epochs = [t_lo, t_hi]
    for i in range(1, 18):
        day = math.floor((t_lo + i * (t_hi - t_lo) / 18.0) / 86400.0 - 0.5) + 0.5
        epochs.append(day * 86400.0)
    return sorted(set(epochs))


def boundary_epochs(seg: df.SrephSegment) -> list[float]:
    """Two interior Chebyshev record boundaries of this segment."""
    out = []
    for k in (seg.n_records // 3, (2 * seg.n_records) // 3):
        out.append(seg.init_tdb_s + k * seg.intlen_s)
    return out


def epochs_for(sreph: df.SrephFile, seg_name: str) -> list[float]:
    seg = sreph.segments_named(seg_name)[0]
    return sorted(set(shared_epochs(sreph) + boundary_epochs(seg)))


def bitlevel_picks(sreph: df.SrephFile, seg_name: str) -> list[tuple[str, float]]:
    """(kind, epoch) pairs for the bit-level goldens of one stored segment.

    Span endpoints, one mid-record interior epoch (exercising a non-boundary
    x), and the two interior record boundaries (pinning the boundary-owning
    selection rule at bit level).
    """
    seg = sreph.segments_named(seg_name)[0]
    b1, b2 = boundary_epochs(seg)
    mid = seg.init_tdb_s + (seg.n_records // 2) * seg.intlen_s + seg.intlen_s / 2.0
    prefix = "librations" if seg.kind == df.KIND_ANGLES_RAD else "state"
    return [
        (prefix, sreph.span_start_tdb_s),
        (prefix, mid),
        (prefix, sreph.span_end_tdb_s),
        (f"{prefix}_boundary", b1),
        (f"{prefix}_boundary", b2),
    ]


# ---------------------------------------------------------------------------
# Excerpt construction
# ---------------------------------------------------------------------------


def needed_records(seg: df.SrephSegment, epochs: list[float]) -> set[int]:
    need: set[int] = set()
    for t in epochs:
        if not (seg.init_tdb_s <= t <= seg.end_tdb_s):
            raise SystemExit(f"epoch {t} outside segment {seg.name}")
        k = math.floor((t - seg.init_tdb_s) / seg.intlen_s)
        on_boundary = seg.init_tdb_s + k * seg.intlen_s == t
        if k >= seg.n_records:
            k = seg.n_records - 1
        need.add(k)
        # A boundary epoch belongs to record k, but the continuity tests also
        # evaluate at nextafter(t, -inf), which lands in record k-1.
        if on_boundary and k >= 1:
            need.add(k - 1)
    return need


def build_excerpt(sreph: df.SrephFile, per_segment_epochs: dict[str, list[float]]) -> list[df.SrephSegment]:
    """Mini-segments (contiguous record runs) covering the test epochs."""
    out: list[df.SrephSegment] = []
    for seg in sreph.segments:
        epochs = per_segment_epochs.get(seg.name)
        if not epochs:
            continue
        records = sorted(needed_records(seg, epochs))
        run_start = records[0]
        prev = records[0]
        runs: list[tuple[int, int]] = []
        for k in records[1:]:
            if k == prev + 1:
                prev = k
                continue
            runs.append((run_start, prev))
            run_start = prev = k
        runs.append((run_start, prev))
        for lo, hi in runs:
            out.append(
                df.SrephSegment(
                    name=seg.name,
                    target=seg.target,
                    center=seg.center,
                    kind=seg.kind,
                    init_tdb_s=seg.init_tdb_s + lo * seg.intlen_s,
                    intlen_s=seg.intlen_s,
                    coeffs=seg.coeffs[lo : hi + 1].copy(),
                )
            )
    return out


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


def iso_tdb(tdb_s: float) -> str:
    """Human-readable TDB calendar tag for comments (not consumed by tests)."""
    base = datetime.datetime(2000, 1, 1, 12, 0, 0)
    return (base + datetime.timedelta(seconds=tdb_s)).strftime("%Y-%m-%dT%H:%M:%S") + " TDB"


# ---------------------------------------------------------------------------
# jplephem cross-check helpers (independent evaluation of the same kernels)
# ---------------------------------------------------------------------------


class JplCheck:
    def __init__(self) -> None:
        from jplephem.pck import PCK
        from jplephem.spk import SPK

        self.spk = SPK.open(str(DATA_DIR / "de440s.bsp"))
        self.by_key = {(s.target, s.center): s for s in self.spk.segments}
        pck = PCK.open(str(DATA_DIR / "moon_pa_de440_200625.bpc"))
        self.lib = [
            s
            for s in pck.segments
            if s.body == df.LIBRATION_BODY
            and s.initial_second <= df.SPAN_START_TDB_S
            and s.final_second >= df.SPAN_END_TDB_S
        ][0]

    def state_m(self, target: int, center: int, tdb_s: float) -> tuple[list[float], list[float]]:
        seg = self.by_key[(target, center)]
        p_km, v_kmday = seg.compute_and_differentiate(J2000_JD + tdb_s / 86400.0)
        return [c * 1000.0 for c in p_km], [c * 1000.0 / 86400.0 for c in v_kmday]

    def librations(self, tdb_s: float) -> tuple[list[float], list[float]]:
        a, da = self.lib.compute(J2000_JD + tdb_s / 86400.0, 0.0)
        return [float(x) for x in a], [float(x) for x in da]


# ---------------------------------------------------------------------------
# Horizons
# ---------------------------------------------------------------------------


def horizons_query(command: str, center: str, jd_list: list[float]) -> tuple[str, str]:
    """One vectors query; returns (request_url, response_text)."""
    tlist = " ".join(repr(jd) for jd in jd_list)
    params = {
        "format": "text",
        "COMMAND": f"'{command}'",
        "OBJ_DATA": "'NO'",
        "MAKE_EPHEM": "'YES'",
        "EPHEM_TYPE": "'VECTORS'",
        "CENTER": f"'{center}'",
        "TLIST_TYPE": "'JD'",
        "TLIST": f"'{tlist}'",
        "REF_SYSTEM": "'ICRF'",
        "REF_PLANE": "'FRAME'",
        "VEC_TABLE": "'2'",
        "VEC_CORR": "'NONE'",
        "OUT_UNITS": "'KM-S'",
        "CSV_FORMAT": "'YES'",
    }
    url = HORIZONS_URL + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url) as resp:
        text = resp.read().decode("utf-8", errors="replace")
    return url, text


def parse_horizons_vectors(text: str) -> list[tuple[float, list[str], list[str]]]:
    """Rows between $$SOE/$$EOE: (jd_tdb, [x,y,z] km strings, [vx,vy,vz] km/s strings)."""
    lines = text.splitlines()
    try:
        soe = lines.index("$$SOE")
        eoe = lines.index("$$EOE")
    except ValueError as exc:
        raise SystemExit(f"Horizons response has no $$SOE/$$EOE block:\n{text[:2000]}") from exc
    rows = []
    for line in lines[soe + 1 : eoe]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 8:
            raise SystemExit(f"unexpected Horizons CSV row: {line!r}")
        jd = float(parts[0])
        rows.append((jd, parts[2:5], parts[5:8]))
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    df.fetch_de440s(DATA_DIR)
    full = df.read_sreph(DATA_DIR / df.REPACK_FILENAME)
    check = JplCheck()

    # --- epochs -----------------------------------------------------------
    per_quantity_epochs: dict[str, list[float]] = {}
    for name, _cmd, _center, composer, _gate in QUANTITIES:
        seg_name = "moon" if composer == "moon_minus_earth" else name
        per_quantity_epochs[name] = epochs_for(full, seg_name)
    lib_epochs = epochs_for(full, df.LIBRATION_NAME)

    # Records each stored segment must contribute to the excerpt: the
    # Horizons/libration validation epochs plus the bit-level pick epochs.
    per_segment_epochs: dict[str, list[float]] = {df.LIBRATION_NAME: list(lib_epochs)}
    for name, _cmd, _center, composer, _gate in QUANTITIES:
        targets = ["moon", "earth"] if composer == "moon_minus_earth" else [name]
        for t in targets:
            per_segment_epochs.setdefault(t, [])
            per_segment_epochs[t] = sorted(
                set(per_segment_epochs[t]) | set(per_quantity_epochs[name])
            )
    for seg_name in list(per_segment_epochs):
        per_segment_epochs[seg_name] = sorted(
            set(per_segment_epochs[seg_name])
            | {t for _kind, t in bitlevel_picks(full, seg_name)}
        )

    # --- excerpt ----------------------------------------------------------
    excerpt_segments = build_excerpt(full, per_segment_epochs)
    excerpt_path = HERE / "excerpt_de440s.sreph"
    df.write_sreph(
        excerpt_path,
        excerpt_segments,
        full.source_spk_sha256,
        full.source_pck_sha256,
    )
    excerpt = df.read_sreph(excerpt_path)
    print(
        f"excerpt: {len(excerpt_segments)} mini-segments, "
        f"{excerpt_path.stat().st_size} bytes"
    )

    # Every value below is evaluated FROM THE COMMITTED EXCERPT so the golden
    # bytes and the bytes the tests read are one and the same object.

    # --- bit-level evaluator goldens ---------------------------------------
    bit_cases: list[dict] = []
    for name, _cmd, _center, composer, _gate in QUANTITIES:
        if composer == "moon_minus_earth":
            continue  # composed quantity; covered by the moon/earth segments
        seg_full = full.segments_named(name)[0]
        for kind, t in bitlevel_picks(full, name):
            r_m, v_mps = df.evaluate_state_m(excerpt, name, t)
            jr, jv = check.state_m(seg_full.target, seg_full.center, t)
            assert math.dist(r_m, jr) < XCHECK_POS_M, (name, t)
            assert math.dist(v_mps, jv) < XCHECK_VEL_MPS, (name, t)
            bit_cases.append(
                {
                    "kind": kind,
                    "body": name,
                    "epoch_iso": iso_tdb(t),
                    "tdb_s": float(t).hex(),
                    "r_m": [float(c).hex() for c in r_m],
                    "v_mps": [float(c).hex() for c in v_mps],
                }
            )
    for kind, t in bitlevel_picks(full, df.LIBRATION_NAME):
        a, da = df.evaluate_librations(excerpt, t)
        ja, jda = check.librations(t)
        assert max(abs(x - y) for x, y in zip(a, ja)) < XCHECK_ANG_RAD, t
        assert max(abs(x - y) for x, y in zip(da, jda)) < XCHECK_RATE_RADPS, t
        bit_cases.append(
            {
                "kind": kind,
                "body": df.LIBRATION_NAME,
                "epoch_iso": iso_tdb(t),
                "tdb_s": float(t).hex(),
                "r_m": [float(c).hex() for c in a],
                "v_mps": [float(c).hex() for c in da],
            }
        )
    emit(
        HERE / "state_bitlevel.toml",
        "Bit-level ephemeris evaluator goldens.\n"
        "Values are binary64 hex literals produced by the Python reference\n"
        "evaluator (star_reacher.data_fetch.evaluate_segment) over the\n"
        "committed excerpt_de440s.sreph, cross-checked against jplephem at\n"
        "generation time (tolerances in manifest.toml). The C++ evaluator\n"
        "must reproduce every value bit-exactly (identical operation\n"
        "sequence, D-10 flags). For *_boundary cases the epoch lies exactly\n"
        "on a Chebyshev record boundary; r_m/v_mps hold angles [rad] and\n"
        "rates [rad/s] for the librations kinds.\n"
        "Regenerated by generate.py.",
        bit_cases,
    )
    print(f"state_bitlevel.toml: {len(bit_cases)} cases")

    # --- Horizons vectors ---------------------------------------------------
    horizons_dir = HERE / "horizons"
    horizons_dir.mkdir(exist_ok=True)
    hz_cases: list[dict] = []
    fullspan_rows: list[tuple[str, int, float, float]] = []
    for name, cmd, center, composer, gate_m in QUANTITIES:
        epochs = per_quantity_epochs[name]
        jds = [J2000_JD + t / 86400.0 for t in epochs]
        url, text = horizons_query(cmd, center, jds)
        (horizons_dir / f"{name}.txt").write_text(
            f"# JPL Horizons API transcript for golden quantity '{name}'\n"
            f"# fetched {GENERATION_DATE} by tests/golden/ephemeris/generate.py\n"
            f"# request:\n# {url}\n# response follows verbatim:\n" + text,
            newline="\n",
            encoding="utf-8",
        )
        rows = parse_horizons_vectors(text)
        assert len(rows) == len(epochs), (name, len(rows), len(epochs))
        max_dr = 0.0
        max_dr_t = epochs[0]
        for (jd, xyz_km, vxyz_kmps), t in zip(rows, epochs):
            assert jd == J2000_JD + t / 86400.0, (name, jd, t)
            hz_cases.append(
                {
                    "quantity": name,
                    "command": cmd,
                    "center": center,
                    "composer": composer,
                    "gate_m": repr(float(gate_m)),
                    "epoch_iso": iso_tdb(t),
                    "tdb_s": repr(float(t)),
                    "jd_tdb": repr(jd),
                    "r_km": list(xyz_km),
                    "v_kmps": list(vxyz_kmps),
                }
            )
            # Full-span check runs on the complete repack, not the excerpt.
            if composer == "moon_minus_earth":
                rm_full, _ = df.evaluate_state_m(full, "moon", t)
                re_full, _ = df.evaluate_state_m(full, "earth", t)
                r_full = [a - b for a, b in zip(rm_full, re_full)]
            else:
                r_full, _ = df.evaluate_state_m(full, name, t)
            dr = math.dist(r_full, [float(c) * 1000.0 for c in xyz_km])
            if dr > max_dr:
                max_dr, max_dr_t = dr, t
        assert max_dr < gate_m, (name, max_dr, gate_m)
        fullspan_rows.append((name, len(epochs), max_dr, max_dr_t))
        print(f"horizons {name}: {len(epochs)} epochs, max |dr| = {max_dr:.6f} m")
    emit(
        HERE / "horizons_vectors.toml",
        "JPL Horizons geometric ICRF state vectors (VEC_CORR=NONE,\n"
        "REF_SYSTEM=ICRF, REF_PLANE=FRAME, OUT_UNITS=KM-S, epochs TDB).\n"
        "r_km / v_kmps carry the Horizons output fields verbatim; the raw\n"
        "request/response transcripts live in horizons/<quantity>.txt.\n"
        "'composer' names the repack-side computation of the same geometric\n"
        "quantity: state:<body> is the body's stored segment; moon_minus_earth\n"
        "is the geocentric Moon composed from the EMB-relative moon and earth\n"
        "segments. All epochs are TDB midnights (exact in binary64 as both JD\n"
        "and seconds past J2000), so no epoch quantization enters the\n"
        "comparison. gate_m is 1.0 except for the two lunar quantities, which\n"
        "Horizons serves from DE441, not DE440: DE441's lunar orbit differs\n"
        "from DE440 by < 2 m over 1970-2020, growing ~10 m per century away\n"
        "from present (Park et al. 2021, AJ 161:105, Section 6), so their\n"
        "Horizons bound is a 10 m DE441 envelope and the authoritative DE440\n"
        "lunar gate lives in moon_de440_jplephem.toml (< 1 mm).\n"
        "Regenerated by generate.py.",
        hz_cases,
    )
    print(f"horizons_vectors.toml: {len(hz_cases)} cases")

    # --- lunar DE440 authority: jplephem evaluation of the source kernel ----
    moon_cases: list[dict] = []
    for t in per_quantity_epochs["moon"]:
        jr_m, jv_m = check.state_m(301, 3, t)
        jr_e, jv_e = check.state_m(399, 3, t)
        moon_cases.append(
            {
                "epoch_iso": iso_tdb(t),
                "tdb_s": repr(float(t)),
                "moon_emb_r_m": [float(c).hex() for c in jr_m],
                "moon_emb_v_mps": [float(c).hex() for c in jv_m],
                "moon_geo_r_m": [float(a - b).hex() for a, b in zip(jr_m, jr_e)],
                "moon_geo_v_mps": [float(a - b).hex() for a, b in zip(jv_m, jv_e)],
            }
        )
    emit(
        HERE / "moon_de440_jplephem.toml",
        "DE440 lunar reference states evaluated by jplephem (independent\n"
        "implementation) from the checksummed de440s.bsp: the Moon relative\n"
        "to the EMB (segment 301/3) and the geocentric Moon (301/3 minus\n"
        "399/3), at the same 21 epochs as the Horizons set. This is the\n"
        "authoritative DE440 lunar comparison: Horizons serves the Moon from\n"
        "DE441 (see horizons_vectors.toml header), so the sub-meter DE440\n"
        "fidelity gate for the lunar segments runs against these values\n"
        "(tolerance in manifest.toml). Values are binary64 hex literals of\n"
        "jplephem's output. Regenerated by generate.py.",
        moon_cases,
    )
    print(f"moon_de440_jplephem.toml: {len(moon_cases)} epochs")

    # --- librations vs jplephem --------------------------------------------
    lib_cases: list[dict] = []
    for t in lib_epochs:
        ja, jda = check.librations(t)
        lib_cases.append(
            {
                "epoch_iso": iso_tdb(t),
                "tdb_s": repr(float(t)),
                "angles_rad": [float(c).hex() for c in ja],
                "rates_radps": [float(c).hex() for c in jda],
            }
        )
    emit(
        HERE / "librations_jplephem.toml",
        "DE440 lunar libration angles (Moon PA 3-1-3 Euler angles w.r.t. the\n"
        "ICRF equator) and rates evaluated by jplephem (independent\n"
        "implementation) from moon_pa_de440_200625.bpc. Values are binary64\n"
        "hex literals of jplephem's output; the consuming test compares the\n"
        "repack evaluation within the tolerance recorded in manifest.toml.\n"
        "Regenerated by generate.py.",
        lib_cases,
    )
    print(f"librations_jplephem.toml: {len(lib_cases)} epochs")

    # --- full-span validation summary ---------------------------------------
    worst_nonlunar = max(r[2] for r, q in zip(fullspan_rows, QUANTITIES) if q[4] == GATE_M)
    worst_lunar = max(r[2] for r, q in zip(fullspan_rows, QUANTITIES) if q[4] != GATE_M)
    # Lunar DE440 fidelity: full repack vs the jplephem DE440 states above.
    worst_moon_de440 = 0.0
    for case in moon_cases:
        t = float(case["tdb_s"])
        rm_full, _ = df.evaluate_state_m(full, "moon", t)
        re_full, _ = df.evaluate_state_m(full, "earth", t)
        geo = [a - b for a, b in zip(rm_full, re_full)]
        ref_emb = [float.fromhex(h) for h in case["moon_emb_r_m"]]
        ref_geo = [float.fromhex(h) for h in case["moon_geo_r_m"]]
        worst_moon_de440 = max(
            worst_moon_de440,
            math.dist(rm_full, ref_emb),
            math.dist(geo, ref_geo),
        )
    lines = [
        "# Full-span DE440s repack validation against JPL Horizons",
        "",
        f"Executed {GENERATION_DATE} by `tests/golden/ephemeris/generate.py` on the",
        "maintainer machine (Phase 2 exit criterion 2 evidence). The complete",
        f"repack `data/{df.REPACK_FILENAME}` (not the committed excerpt) was",
        "evaluated by the Python reference evaluator - the executable",
        "specification of `star::Ephemeris` - at every test epoch and compared",
        "against geometric ICRF state vectors fetched from the JPL Horizons API",
        "(transcripts under `horizons/`).",
        "",
        "Sources (SHA-256):",
        "",
        f"- `de440s.bsp` `{full.source_spk_sha256}`",
        f"- `moon_pa_de440_200625.bpc` `{full.source_pck_sha256}`",
        "",
        "Epochs per quantity: the repacked span endpoints, 17 TDB midnights",
        "spread across 2020-2060, and two interior Chebyshev record boundaries",
        "of the underlying segment (21 epochs per quantity, all exact TDB",
        "midnights).",
        "",
        "| Quantity | Horizons COMMAND / CENTER | Epochs | Max position error [m] | At epoch (TDB) | Gate [m] |",
        "|---|---|---|---|---|---|",
    ]
    for (name, cmd, center, _composer, gate_m), (qname, n_epochs, max_dr, max_dr_t) in zip(
        QUANTITIES, fullspan_rows
    ):
        assert name == qname
        lines.append(
            f"| {name} | {cmd} / {center} | {n_epochs} | {max_dr:.6f} | {iso_tdb(max_dr_t)} | {gate_m:g} |"
        )
    lines += [
        "",
        f"Worst non-lunar case: **{worst_nonlunar:.6f} m** against the 1 m gate.",
        "",
        "## Lunar quantities: Horizons serves DE441, not DE440",
        "",
        "The committed transcripts record `{source: DE441}` for the Moon and the",
        "Earth-Moon barycenter. DE441 deliberately omits the lunar core-mantle",
        "tidal damping term so it can span -13200 to +17191; its lunar orbit",
        "consequently differs from DE440 by under 2 m across 1970-2020, growing",
        "to roughly 10 m per century away from the present, predominantly",
        "along-track (Park et al. 2021, AJ 161:105, Section 6). The measured",
        f"lunar difference above ({worst_lunar:.3f} m worst case, growing",
        "secularly from 1.8 m in 2020 to 5.4 m in 2060 with local oscillation",
        "about the trend, and mirrored in the",
        "earth quantity at 1/82.3 of the lunar value - the EMB mass-ratio",
        "signature) is exactly that published DE440/DE441 difference, not a",
        "repack defect. A bit-faithful DE440 repack cannot match DE441-sourced",
        "lunar states more closely.",
        "",
        "The authoritative DE440 lunar comparison therefore runs against",
        "jplephem's independent evaluation of the checksummed `de440s.bsp`",
        "(`moon_de440_jplephem.toml`, same 21 epochs):",
        "",
        "| Quantity | Epochs | Max position error vs jplephem/DE440 [m] | Gate [m] |",
        "|---|---|---|---|",
        f"| moon (w.r.t. EMB) and moon_geocentric | 21 | {worst_moon_de440:.9f} | 0.001 |",
        "",
        "Lunar librations are validated separately against jplephem's",
        "independent evaluation of the same PCK (`librations_jplephem.toml`);",
        "Horizons does not serve the DE440 Euler angles directly.",
        "",
    ]
    (HERE / "full_span_validation.md").write_text(
        "\n".join(lines), newline="\n", encoding="utf-8"
    )
    assert worst_nonlunar < GATE_M and worst_lunar < LUNAR_ENVELOPE_M
    assert worst_moon_de440 < 1e-3
    print(
        f"full_span_validation.md: worst non-lunar {worst_nonlunar:.6f} m, "
        f"lunar vs DE441-Horizons {worst_lunar:.3f} m, "
        f"lunar vs DE440-jplephem {worst_moon_de440:.2e} m"
    )

    # --- manifest ------------------------------------------------------------
    manifest = f'''# Provenance manifest for the ephemeris golden set in this directory
# (FR-22 layer 1: uncited golden = lint failure; schema documented in
# tests/golden/README.md).

schema_version = 1

[golden]
directory = "ephemeris"
date = "{GENERATION_DATE}"
generation = """
All files are written by tests/golden/ephemeris/generate.py, which runs the
`star data fetch de440s` pipeline (download of de440s.bsp and
moon_pa_de440_200625.bpc from JPL with SHA-256 pins, then a verbatim -
never refit - repack of the 2020-2060 Chebyshev records into the SREPH v1
container), excerpts the records containing the test epochs, evaluates the
excerpt with the Python reference evaluator (the executable specification of
cpp/src/ephemeris.cpp), cross-checks every value against jplephem's
independent evaluation of the same kernels, and fetches geometric ICRF state
vectors from the JPL Horizons API for every repacked quantity. Regenerating
is `python tests/golden/ephemeris/generate.py` (network required; the
committed outputs are consumed offline).
"""

[[file]]
name = "excerpt_de440s.sreph"
source = "de440s.bsp (SHA-256 {full.source_spk_sha256}) and moon_pa_de440_200625.bpc (SHA-256 {full.source_pck_sha256}), JPL SSD / NAIF distribution"
citation = "Park, Folkner, Williams, and Boggs, 'The JPL Planetary and Lunar Ephemerides DE440 and DE441', The Astronomical Journal 161:105, 2021"
generation = "build_excerpt() in generate.py: SREPH v1 mini-segments holding the verbatim DE440 Chebyshev records that contain the test epochs (plus each boundary's left neighbor); format per docs/formats/sreph_v1.md"
date = "{GENERATION_DATE}"
tolerance = "not applicable (input fixture; committed binary excerpt is the Phase 2 exception recorded in tests/golden/README.md)"

[[file]]
name = "state_bitlevel.toml"
source = "Python reference evaluator (star_reacher.data_fetch.evaluate_segment) over excerpt_de440s.sreph"
citation = "coefficients: Park et al. 2021 (DE440); evaluation: standard Chebyshev recurrence, see the ephemeris chapter of docs/mathlib"
generation = "generate.py; every value cross-checked against jplephem at generation time to < {XCHECK_POS_M} m position, < {XCHECK_VEL_MPS} m/s velocity, < {XCHECK_ANG_RAD} rad angle, < {XCHECK_RATE_RADPS} rad/s rate (measured worst case 1.4e-4 m, 1.7e-11 m/s, 9.1e-13 rad, 8.5e-22 rad/s)"
date = "{GENERATION_DATE}"
tolerance = "exact (binary64 bit equality): the C++ evaluator implements the identical operation sequence under the D-10 no-FMA/no-fast-math flags, so any bit difference is an implementation divergence"

[[file]]
name = "horizons_vectors.toml"
source = "JPL Horizons API (https://ssd.jpl.nasa.gov/api/horizons.api), geometric ICRF vectors, TDB epochs; raw transcripts in horizons/<quantity>.txt"
citation = "Giorgini et al., JPL Horizons on-line ephemeris system (ssd.jpl.nasa.gov/horizons); Horizons major-body source ephemeris DE441 per the committed transcripts; DE440/DE441 relationship per Park et al. 2021 (AJ 161:105)"
generation = "horizons_query() in generate.py; VEC_TABLE=2, VEC_CORR=NONE, REF_SYSTEM=ICRF, REF_PLANE=FRAME, OUT_UNITS=KM-S, TLIST of exact TDB-midnight Julian dates"
date = "{GENERATION_DATE}"
tolerance = "per-case gate_m field: < 1 m position at every epoch for sun/emb/venus_bary/mars_bary/jupiter_bary/earth (Phase 2 exit criterion 2); < 10 m DE441-envelope for moon and moon_geocentric, because Horizons serves the Moon from DE441 whose lunar orbit differs from DE440 by ~2-10 m across this span by design (Park et al. 2021 Section 6) - the authoritative < 1 mm DE440 lunar gate is moon_de440_jplephem.toml"

[[file]]
name = "moon_de440_jplephem.toml"
source = "jplephem {_jplephem_version()} evaluation of de440s.bsp segments 301/3 and 399/3 (SHA-256 {full.source_spk_sha256})"
citation = "Park et al. 2021 (DE440); kernel from the JPL SSD planets/bsp distribution"
generation = "JplCheck.state_m() in generate.py at the 21 lunar test epochs; geocentric Moon composed as (301/3) - (399/3)"
date = "{GENERATION_DATE}"
tolerance = "< 1e-3 m position at every epoch: same DE440 coefficients evaluated by an independent implementation, so any excess is a repack or evaluator defect (measured worst case 1.4e-4 m, pure summation-order rounding)"

[[file]]
name = "horizons/<quantity>.txt (8 files)"
source = "JPL Horizons API responses, saved verbatim with the exact request URL prepended"
citation = "as horizons_vectors.toml"
generation = "horizons_query() in generate.py; these transcripts are the primary evidence for horizons_vectors.toml, including the '{{source: DE441}}' target lines the lunar gate documentation relies on"
date = "{GENERATION_DATE}"
tolerance = "not applicable (evidence transcripts, not consumed by tests)"

[[file]]
name = "librations_jplephem.toml"
source = "jplephem {_jplephem_version()} PCK evaluation of moon_pa_de440_200625.bpc (SHA-256 {full.source_pck_sha256})"
citation = "Park et al. 2021 (DE440 lunar orientation); kernel from the NAIF generic_kernels/pck distribution"
generation = "JplCheck.librations() in generate.py at the libration test epochs"
date = "{GENERATION_DATE}"
tolerance = "abs 1e-10 rad on angles, 1e-19 rad/s on rates: same coefficients evaluated with a different summation order; measured worst case 9.1e-13 rad / 8.5e-22 rad/s, so the bounds carry two-plus orders of margin while still failing on any wrong record or coefficient"

[[file]]
name = "full_span_validation.md"
source = "full repack (data/{df.REPACK_FILENAME}) vs the horizons_vectors.toml states"
citation = "as horizons_vectors.toml"
generation = "generate.py full-span pass; the committed table records the max observed position error per quantity"
date = "{GENERATION_DATE}"
tolerance = "< 1 m at every epoch for the six DE440-identical quantities (Phase 2 exit criterion 2); lunar quantities < 10 m vs DE441-served Horizons with the authoritative < 1 mm DE440 gate in moon_de440_jplephem.toml (ADR 0002)"
'''
    (HERE / "manifest.toml").write_text(manifest, newline="\n", encoding="utf-8")
    print("manifest.toml written")
    print("golden set regenerated and cross-checked")


def _jplephem_version() -> str:
    import importlib.metadata

    try:
        return importlib.metadata.version("jplephem")
    except importlib.metadata.PackageNotFoundError:
        return "(unknown version)"


if __name__ == "__main__":
    main()
