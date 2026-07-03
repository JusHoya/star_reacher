"""Regenerate the third-body golden-vector file in this directory.

The values anchor the FR-6 Battin f(q) third-body model
(cpp/src/models/thirdbody.cpp) against Phase 3 exit criterion 7: the
cancellation-safe Battin evaluation must match the naive two-vector
difference

    a = gm * ((s - r)/|s - r|^3 - s/|s|^3)

evaluated in extended precision (mpmath, 60 significant decimal digits) to
better than 1e-12 relative at 10 committed states. The naive formula and the
Battin f(q) formulation are algebraically identical (chapter ch:thirdbody of
the math library derives the identity), so the extended-precision naive
evaluation is an implementation-independent reference: it shares no
floating-point failure mode with either double-precision path under test.

Two double-precision mirrors are evaluated at generation time and their
observed errors recorded in the emitted file header and the manifest:

- battin_double(): the exact scalar operation sequence of the C++
  implementation (models/thirdbody.cpp). Its generation-time error against
  the reference is the margin the consuming doctest re-measures.
- naive_double(): the exact scalar operation sequence the consuming doctest
  uses for the criterion-7 digit-loss demonstration. Both use only IEEE-754
  basic operations (+, -, *, /, sqrt), so under the project's strict FP
  flags (D-10: no FMA contraction, no fast-math) the committed digit-loss
  figure is bit-portable across compilers and architectures.

State selection. Cases 1-9 are representative perturbation geometries for
the FR-6 body set (Sun/Moon at LEO and GEO, Venus and Jupiter at LEO, Earth
and the Sun from low lunar orbit); their component values are round decimal
literals, exact in binary64. GM values are round representative magnitudes
(the model takes GM from the caller; the test verifies formulation
agreement on shared inputs, so physical GM fidelity is irrelevant). Case 10
is the criterion-7 near-alignment demonstration state: a low lunar orbiter
0.0012 rad off the Moon-Jupiter line with Jupiter at a far conjunction
(|r| = 1.787e6 m, |s| = 9.679e11 m). It was selected by search_demo_state()
(deterministic seed, recorded below) as the near-alignment state maximizing
the double-precision naive error, and is committed as hex literals so
regeneration never depends on re-running the search. Observed at selection:
naive 1.373e-10 relative (6.14 of binary64's ~15.95 significant decimal
digits lost), Battin 1.6e-16.

Running this script rewrites states.toml byte-identically; any diff after
regeneration means the script or the golden was edited by hand, which the
FR-22 golden-update discipline forbids. The search is only re-run when
explicitly requested: python generate.py --search
"""

from __future__ import annotations

import math
import pathlib
import sys

import mpmath as mp

mp.mp.dps = 60

HERE = pathlib.Path(__file__).resolve().parent

# Gates enforced at generation time (mirrors of the consuming doctest gates,
# so a bad regeneration fails here instead of in CI).
BATTIN_GATE = 1e-12       # Phase 3 exit criterion 7
DIGIT_LOSS_GATE = 1e-10   # >= 6 of ~16 significant digits lost


# ---------------------------------------------------------------------------
# Extended-precision reference (mpmath, naive difference)
# ---------------------------------------------------------------------------


def naive_mp(gm: float, r: tuple, s: tuple) -> list:
    gm_ = mp.mpf(gm)
    rm = [mp.mpf(x) for x in r]
    sm = [mp.mpf(x) for x in s]
    d = [sm[i] - rm[i] for i in range(3)]
    dn = mp.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
    sn = mp.sqrt(sm[0] * sm[0] + sm[1] * sm[1] + sm[2] * sm[2])
    dn3 = dn * dn * dn
    sn3 = sn * sn * sn
    return [gm_ * (d[i] / dn3 - sm[i] / sn3) for i in range(3)]


# ---------------------------------------------------------------------------
# Double-precision mirrors. Operation order is load-bearing: each mirrors,
# statement for statement, the code whose error it predicts (see docstring).
# ---------------------------------------------------------------------------


def naive_double(gm: float, r: tuple, s: tuple) -> tuple:
    """Mirror of the doctest's naive evaluation (test_thirdbody.cpp)."""
    d0 = s[0] - r[0]
    d1 = s[1] - r[1]
    d2 = s[2] - r[2]
    dn = math.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
    sn = math.sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2])
    dn3 = dn * dn * dn
    sn3 = sn * sn * sn
    return (gm * (d0 / dn3 - s[0] / sn3),
            gm * (d1 / dn3 - s[1] / sn3),
            gm * (d2 / dn3 - s[2] / sn3))


def battin_double(gm: float, r: tuple, s: tuple) -> tuple:
    """Mirror of the C++ implementation (models/thirdbody.cpp)."""
    s2 = s[0] * s[0] + s[1] * s[1] + s[2] * s[2]
    q = (r[0] * (r[0] - 2.0 * s[0]) + r[1] * (r[1] - 2.0 * s[1])
         + r[2] * (r[2] - 2.0 * s[2])) / s2                # eq:thirdbody:q
    opq = 1.0 + q
    f = q * (3.0 + 3.0 * q + q * q) / (1.0 + opq * math.sqrt(opq))
    d0 = r[0] - s[0]
    d1 = r[1] - s[1]
    d2 = r[2] - s[2]
    dn = math.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
    d3 = dn * dn * dn
    c = -gm / d3
    return (c * (r[0] + f * s[0]),                          # eq:thirdbody:accel
            c * (r[1] + f * s[1]),
            c * (r[2] + f * s[2]))


def rel_err(a: tuple, ref: list) -> float:
    dif = [mp.mpf(a[i]) - ref[i] for i in range(3)]
    nd = mp.sqrt(dif[0] ** 2 + dif[1] ** 2 + dif[2] ** 2)
    nr = mp.sqrt(ref[0] ** 2 + ref[1] ** 2 + ref[2] ** 2)
    return float(nd / nr)


# ---------------------------------------------------------------------------
# Committed cases: (name, gm [m^3/s^2], r_sc [m], r_third [m], demo_flag)
# Both positions are relative to the central body in a common frame, the
# convention of star::models::thirdbody_accel.
# ---------------------------------------------------------------------------

CASES = [
    # Sun at LEO: the classic cancellation regime (|r| ~ 7e6, |s| ~ 1.5e11).
    ("sun_leo_align", 1.327e20,
     (6.778137e6, 0.0, 0.0), (1.495978707e11, 0.0, 0.0), False),
    ("sun_leo_perpendicular", 1.327e20,
     (0.0, 6.778137e6, 0.0), (1.495978707e11, 0.0, 0.0), False),
    ("sun_leo_generic", 1.327e20,
     (3.2e6, -4.1e6, 4.3e6), (1.31e11, -6.2e10, -2.7e10), False),
    # Moon at LEO and GEO (always-on Earth-regime third body, FR-6).
    ("moon_leo_generic", 4.9e12,
     (5.1e6, 3.9e6, -2.1e6), (3.1e8, -1.9e8, 1.1e8), False),
    ("moon_geo_generic", 4.9e12,
     (-2.9e7, 3.05e7, 1.2e6), (-3.63e8, 1.21e8, 2.9e7), False),
    # Sun at GEO, near-alignment (larger r/s than LEO, still cancelling).
    ("sun_geo_near_align", 1.327e20,
     (4.2164e7, 8.0e3, -5.0e3), (1.4959787e11, 0.0, 0.0), False),
    # Switchable bodies at LEO (FR-6: Venus/Jupiter per regime).
    ("jupiter_leo_opposition", 1.267e17,
     (6.6e6, 1.4e6, 5.0e5), (5.88e11, 8.0e10, 3.4e10), False),
    ("venus_leo_conjunction", 3.249e14,
     (2.2e6, 6.3e6, -1.2e6), (4.0e10, 8.0e9, 3.0e9), False),
    # Earth as third body from low lunar orbit: mild cancellation
    # (|r|/|s| ~ 5e-3), the benign contrast case the property test uses.
    ("earth_from_llo_generic", 3.986004418e14,
     (1.2e6, 1.1e6, 8.0e5), (-3.6e8, 1.3e8, 6.0e7), False),
    # Criterion-7 near-alignment demonstration state (see module docstring):
    # low lunar orbiter 0.0012 rad off the Moon-Jupiter line, Jupiter at a
    # far conjunction. Hex literals: committed output of search_demo_state().
    ("jupiter_llo_near_align", 1.267e17,
     (float.fromhex("0x1.b45e735dc09e1p+20"),
      float.fromhex("-0x1.c2a74eab00e19p+10"),
      float.fromhex("0x1.0d98574b85c51p+10")),
     (float.fromhex("0x1.c2b92fec3ea9cp+39"), 0.0, 0.0), True),
]


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


def main() -> None:
    out_cases = []
    max_battin = 0.0
    for name, gm, r, s, demo in CASES:
        ref = naive_mp(gm, r, s)
        e_battin = rel_err(battin_double(gm, r, s), ref)
        e_naive = rel_err(naive_double(gm, r, s), ref)
        max_battin = max(max_battin, e_battin)
        assert e_battin < BATTIN_GATE, (name, e_battin)
        if demo:
            assert e_naive >= DIGIT_LOSS_GATE, (name, e_naive)
        case = {
            "name": name,
            "gm_m3ps2": float(gm).hex(),
            "r_sc_m": [float(x).hex() for x in r],
            "r_third_m": [float(x).hex() for x in s],
            # Extended-precision reference, rounded once to binary64.
            "a_ref_mps2": [float(x).hex() for x in ref],
            "digit_loss_demo": "true" if demo else "false",
            # Informational: generation-time double-mirror errors (norm
            # relative, vs the pre-rounding mpmath reference). The consuming
            # doctest recomputes both; these record the expected values.
            "battin_rel_err_observed": f"{e_battin:.3e}",
            "naive_rel_err_observed": f"{e_naive:.3e}",
        }
        out_cases.append(case)
        print(f"{name:26s} naive {e_naive:.3e}  battin {e_battin:.3e}")

    emit(
        HERE / "states.toml",
        "Third-body golden vectors (FR-6, Phase 3 exit criterion 7).\n"
        "gm_m3ps2, r_sc_m, r_third_m are the exact binary64 inputs (both\n"
        "positions relative to the central body in a common frame);\n"
        "a_ref_mps2 is the naive two-vector-difference acceleration\n"
        "gm*((s-r)/|s-r|^3 - s/|s|^3) evaluated with mpmath at 60\n"
        "significant decimal digits from those exact inputs, rounded once\n"
        "to binary64. The case flagged digit_loss_demo demonstrates the\n"
        "criterion-7 near-alignment cancellation (naive double evaluation\n"
        "loses >= 6 significant digits). Provenance and tolerances in\n"
        "manifest.toml. Regenerated by generate.py.",
        out_cases,
    )
    print(f"third-body golden regenerated; mpmath {mp.__version__} at "
          f"{mp.mp.dps} dps; battin max rel err {max_battin:.3e}")


def search_demo_state() -> None:
    """Re-run the near-alignment digit-loss search (NOT part of regeneration).

    Random search plus ulp-scale hill climb over low-lunar-orbit positions
    within 1.7e-3 rad of the Moon-Jupiter line, maximizing the
    double-precision naive error against the mpmath reference. The committed
    case-10 hex literals are the output of this procedure with these exact
    seeds. Re-running may find a different (equally valid) state; committing
    a new one requires updating the manifest per the golden-update policy.
    """
    import numpy as np

    gm = 1.267e17
    rng = np.random.default_rng(20260702)
    best_e, best_r, best_s = 0.0, None, None
    for _ in range(20000):
        ang = float(rng.uniform(0.0, 1.7e-3))
        phi = float(rng.uniform(0.0, 2.0 * math.pi))
        rr = float(1.789e6 + rng.uniform(-3e3, 3e3))
        r = (rr * math.cos(ang), rr * math.sin(ang) * math.cos(phi),
             rr * math.sin(ang) * math.sin(phi))
        s = (float(9.68e11 + rng.uniform(-1e8, 1e8)), 0.0, 0.0)
        e = rel_err(naive_double(gm, r, s), naive_mp(gm, r, s))
        if e > best_e:
            best_e, best_r, best_s = e, r, s
    rng = np.random.default_rng(42)
    for _ in range(6):
        improved = False
        for _ in range(8000):
            scale = 10.0 ** rng.integers(-13, -8)
            r = tuple(float(x * (1.0 + scale * rng.uniform(-1, 1)))
                      for x in best_r)
            s = (float(best_s[0] * (1.0 + scale * rng.uniform(-1, 1))),
                 0.0, 0.0)
            e = rel_err(naive_double(gm, r, s), naive_mp(gm, r, s))
            if e > best_e:
                best_e, best_r, best_s = e, r, s
                improved = True
        if not improved:
            break
    print("search result (commit by hand into CASES + manifest):")
    print("  r =", [x.hex() for x in best_r])
    print("  s =", [x.hex() for x in best_s])
    print(f"  naive rel err {best_e:.4e} "
          f"({math.log10(best_e) + 16:.2f} digits lost)")


if __name__ == "__main__":
    if "--search" in sys.argv:
        search_demo_state()
    else:
        main()
