"""Regenerate the continuous lunar DE440 excerpt in this directory.

The Phase 8 lunar cross-tool cases (``missions/lunar_orbiter.toml``, the
< 100 m frozen-GMAT gate, and ``missions/lro_illustrative.toml``, the
report-only LRO case) propagate about the Moon under GRGM1200A harmonic
gravity in the Moon principal-axis frame. That frame is built from the DE440
lunar libration angles (``cpp/src/models/environment.cpp`` ->
``eph_->lunar_librations`` -> ``frames::c_gcrf_to_moonpa``), so - unlike the
Phase 3 crosstool excerpt - a Moon-centred run needs the ``moon_librations``
segment (kind 1) in addition to the sun/emb/earth/moon position segments the
Sun and Earth third bodies and the geocentric Moon composition consume.

This script cuts a second continuous excerpt carrying all five needed
segments, for each the verbatim DE440 Chebyshev records covering

    [2026-01-01T00:00:00 TDB - 2 d,  2026-01-01T00:00:00 TDB + 9 d]

(the 2026-01-01 mission epoch and 7-day span with two days of margin on both
sides, the same window as the Phase 3 crosstool excerpt), written with the
same SREPH v1 writer as the full repack (never refit). Committing it keeps the
lunar missions runnable in CI and on a clean clone; the provenance entry lives
in ``manifest.toml`` alongside the other excerpts'.

Maintainer-side: requires the fetched kernels (``star data fetch de440s``,
network on first run - the same de440s.bsp and moon_pa_de440_200625.bpc the
other excerpts are cut from, SHA-256-pinned in
``python/star_reacher/data_fetch.py``). Regenerating is
``python tests/golden/ephemeris/generate_lunar.py``; the output bytes are a
pure function of the pinned source kernels. This is the ephemeris-fixture half
of the Phase 8 cross-tool preparation that was registered against the PRD
section 9 valve (``docs/release_checklist.md`` item 10) in case the kernels
proved unfetchable. They were fetchable on the Phase 8 execution host, so the
excerpt was generated and committed there and that half of the item was lifted;
the remaining GMAT frozen truth was completed 2026-07-25 and the item is now
fully discharged. Re-running this generator only re-confirms the committed
bytes.
"""

from __future__ import annotations

import math
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from star_reacher import data_fetch as df  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
OUT_NAME = "excerpt_de440s_lunar.sreph"

# 2026-01-01T00:00:00 TDB (a TDB midnight, exact in binary64): 9496.5 days
# past J2000 (2000-01-01T12:00:00 TDB), matching generate_crosstool.py so the
# lunar excerpt covers the same window as the Molniya crosstool excerpt.
EPOCH_TDB_S = 9496.5 * 86400.0  # 820497600.0
MARGIN_BEFORE_S = 2.0 * 86400.0
SPAN_AFTER_S = 9.0 * 86400.0  # 7-day mission plus 2 days of margin

# sun/emb: third-body Sun position and the Earth/Moon geocentric composition;
# earth/moon: the geocentric Moon (moon - earth) and the central-body SSB
# position; moon_librations: the Moon principal-axis frame the harmonic field
# is evaluated in. The Moon central body needs all five (the crosstool excerpt
# carries only the first four, which is why a separate lunar excerpt exists).
SEGMENTS_NEEDED = (df.LIBRATION_NAME, "sun", "emb", "earth", "moon")


def contiguous_slice(seg: df.SrephSegment, t_lo: float, t_hi: float) -> df.SrephSegment:
    """The verbatim records of ``seg`` covering [t_lo, t_hi].

    Identical record-selection arithmetic to generate_crosstool.py; it is
    reproduced rather than imported so each excerpt generator is a
    self-contained maintainer script. Works for every kind (position and
    libration) because it slices on the directory fields alone.
    """
    if not (seg.init_tdb_s <= t_lo and t_hi <= seg.end_tdb_s):
        raise SystemExit(
            f"segment {seg.name}: requested span [{t_lo}, {t_hi}] exceeds "
            f"stored span [{seg.init_tdb_s}, {seg.end_tdb_s}]"
        )
    k_lo = int(math.floor((t_lo - seg.init_tdb_s) / seg.intlen_s))
    k_hi = int(math.floor((t_hi - seg.init_tdb_s) / seg.intlen_s))
    k_hi = min(k_hi, seg.n_records - 1)
    return df.SrephSegment(
        name=seg.name,
        target=seg.target,
        center=seg.center,
        kind=seg.kind,
        init_tdb_s=seg.init_tdb_s + k_lo * seg.intlen_s,
        intlen_s=seg.intlen_s,
        coeffs=seg.coeffs[k_lo : k_hi + 1].copy(),
    )


def main() -> None:
    df.fetch_de440s(DATA_DIR)
    full = df.read_sreph(DATA_DIR / df.REPACK_FILENAME)

    t_lo = EPOCH_TDB_S - MARGIN_BEFORE_S
    t_hi = EPOCH_TDB_S + SPAN_AFTER_S
    minis = [
        contiguous_slice(full.segments_named(name)[0], t_lo, t_hi)
        for name in SEGMENTS_NEEDED
    ]
    out_path = HERE / OUT_NAME
    df.write_sreph(out_path, minis, full.source_spk_sha256, full.source_pck_sha256)

    check = df.read_sreph(out_path)
    print(f"{OUT_NAME}: {len(minis)} segments, {out_path.stat().st_size} bytes")
    print(
        f"common span: [{check.span_start_tdb_s:.0f}, {check.span_end_tdb_s:.0f}] "
        f"s TDB (need [{t_lo:.0f}, {t_hi:.0f}])"
    )
    for seg in check.segments:
        print(
            f"  {seg.name}: {seg.n_records} records x {seg.intlen_s:.0f} s, "
            f"kind {seg.kind}, [{seg.init_tdb_s:.0f}, {seg.end_tdb_s:.0f}]"
        )
    if not (check.span_start_tdb_s <= t_lo and t_hi <= check.span_end_tdb_s):
        raise SystemExit("excerpt does not cover the requested span")

    # Spot-check the excerpt against the full repack at the mission epoch and
    # mid-span: verbatim records (positions and libration angles alike) must
    # evaluate bit-identically.
    for t in (EPOCH_TDB_S, EPOCH_TDB_S + 3.5 * 86400.0):
        for name in ("sun", "emb", "earth", "moon"):
            r_full, v_full = df.evaluate_state_m(full, name, t)
            r_exc, v_exc = df.evaluate_state_m(check, name, t)
            assert r_full == r_exc and v_full == v_exc, (name, t)
        a_full = df.evaluate_librations(full, t)
        a_exc = df.evaluate_librations(check, t)
        assert a_full == a_exc, ("moon_librations", t)
    print("bit-identity spot check vs the full repack: OK")


if __name__ == "__main__":
    main()
