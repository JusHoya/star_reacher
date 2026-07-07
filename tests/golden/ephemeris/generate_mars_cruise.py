"""Regenerate the continuous Mars-cruise DE440 excerpt in this directory.

The Phase 5 heliocentric example mission (``missions/mars_cruise.toml``) needs
the Sun, Earth, Moon, Venus, Mars, and Jupiter continuously over its 7-day
window: the Sun segment anchors the heliocentric frame (the central body's
SSB position), the emb+earth and emb+moon pairs compose the Earth and Moon
third bodies, and the venus_bary/mars_bary/jupiter_bary segments feed the
remaining perturbers. Neither committed excerpt covers this:
the Phase 2 excerpt holds isolated records around discrete epochs, and the
Phase 3 cross-tool excerpt covers a January 2026 window without the Mars and
Jupiter segments. This script therefore cuts a third CONTIGUOUS excerpt with
the same recipe as ``generate_crosstool.py``: for each needed segment, the
verbatim DE440 Chebyshev records covering

    [2026-12-05T00:00:00 TDB - 2 d,  2026-12-05T00:00:00 TDB + 9 d]

(the mission epoch and 7-day span with margin on both sides), written with the
same SREPH v1 writer as the full repack (never refit). Committing it keeps the
mission runnable in CI and on a clean clone; the provenance entry lives in
``manifest.toml`` alongside the other excerpts'.

Maintainer-side: requires the fetched kernels (``star data fetch de440s``,
network on first run). Regenerating is
``python tests/golden/ephemeris/generate_mars_cruise.py``; the output bytes
are a pure function of the pinned source kernels.
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
OUT_NAME = "excerpt_de440s_mars_cruise.sreph"

# 2026-12-05T00:00:00 TDB (a TDB midnight, exact in binary64): 9834.5 days
# past J2000 (2000-01-01T12:00:00 TDB), i.e. 9834.5 * 86400 s. The mission
# epoch is UTC, offset from TDB by ~69 s; the two-day margin dwarfs it.
EPOCH_TDB_S = 9834.5 * 86400.0  # 849700800.0
MARGIN_BEFORE_S = 2.0 * 86400.0
SPAN_AFTER_S = 9.0 * 86400.0  # 7-day mission plus 2 days of margin

SEGMENTS_NEEDED = (
    "sun",
    "emb",
    "earth",
    "moon",
    "venus_bary",
    "mars_bary",
    "jupiter_bary",
)


def contiguous_slice(seg: df.SrephSegment, t_lo: float, t_hi: float) -> df.SrephSegment:
    """The verbatim records of ``seg`` covering [t_lo, t_hi]."""
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
            f"[{seg.init_tdb_s:.0f}, {seg.end_tdb_s:.0f}]"
        )
    if not (check.span_start_tdb_s <= t_lo and t_hi <= check.span_end_tdb_s):
        raise SystemExit("excerpt does not cover the requested span")

    # Spot-check the excerpt against the full repack at the mission epoch and
    # mid-span: verbatim records must evaluate bit-identically.
    for t in (EPOCH_TDB_S, EPOCH_TDB_S + 3.5 * 86400.0):
        for name in SEGMENTS_NEEDED:
            r_full, v_full = df.evaluate_state_m(full, name, t)
            r_exc, v_exc = df.evaluate_state_m(check, name, t)
            assert r_full == r_exc and v_full == v_exc, (name, t)
    print("bit-identity spot check vs the full repack: OK")


if __name__ == "__main__":
    main()
