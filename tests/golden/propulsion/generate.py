"""Regenerate the propulsion golden-vector files in this directory.

The values anchor the FR-10 engine model (cpp/src/models/propulsion.cpp)
against Phase 4 exit criterion 3. Two files are produced:

- engine.toml: delivered thrust magnitude F = lambda F_vac - p_amb A_e
  (back pressure reduces delivered thrust, never the mass flow), mass flow
  mdot = lambda F_vac / (g0 Isp_vac), the gimbal-deflected thrust
  direction (two sequential axis rotations of the nominal axis, chapter
  ch:propulsion, eq:propulsion:direction), and the resulting body force
  and torque about the supplied composite CG (eq:propulsion:forcetorque).
  The zero-throttle case must be exactly zero in every output.
- tsiolkovsky.toml: fixed-attitude vacuum-burn scenarios with the analytic
  velocity increment dv = g0 Isp_vac ln(m0 / m1) where
  m1 = m0 - mdot t_burn (eq:propulsion:tsiolkovsky). The consuming test
  integrates the same burn with the model's mdot and compares within the
  exit-criterion-3 0.1 % gate.

References are evaluated with mpmath at 60 significant decimal digits from
the exact binary64 inputs and rounded once to binary64. The mpmath
evaluation mirrors the model formulation of math-library chapter
ch:propulsion term for term but shares no binary64 rounding with the C++
path, so the committed values check the double-precision implementation to
its own rounding floor. The standard gravity g0 = 9.80665 m/s^2 is the
exact conventional value (BIPM SI Brochure, 9th ed., 2019; the value fixed
by the 3rd CGPM, 1901); sea-level ambient pressure 101325 Pa is the USSA76
standard sea-level pressure (the value cpp/src/models/atmosphere_ussa76.cpp
returns at z = 0, which the consuming test verifies through the actual
model call).

Running this script rewrites both .toml files byte-identically; any diff
after regeneration means the script or the goldens were edited by hand,
which the FR-22 golden-update discipline forbids.
"""

from __future__ import annotations

import pathlib

import mpmath as mp

mp.mp.dps = 60

HERE = pathlib.Path(__file__).resolve().parent

G0 = mp.mpf(9.80665)  # exact conventional standard gravity [m/s^2]


# ---------------------------------------------------------------------------
# mpmath mirror of the propulsion model (chapter ch:propulsion)
# ---------------------------------------------------------------------------


def v3(x):
    return [mp.mpf(c) for c in x]


def vadd(a, b):
    return [a[i] + b[i] for i in range(3)]


def vsub(a, b):
    return [a[i] - b[i] for i in range(3)]


def vscale(s, a):
    return [s * a[i] for i in range(3)]


def vdot(a, b):
    return sum(a[i] * b[i] for i in range(3))


def vcross(a, b):
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


def rodrigues(axis, angle, v):
    """Rotation of v about the unit axis by angle (Rodrigues form,
    eq:propulsion:direction)."""
    c, s = mp.cos(angle), mp.sin(angle)
    return vadd(vadd(vscale(c, v), vscale(s, vcross(axis, v))),
                vscale(vdot(axis, v) * (1 - c), axis))


def thrust_direction(axis, a1, a2, g1, g2):
    """Deflected thrust direction: rotate the nominal axis about gimbal
    axis 1 by g1, then about gimbal axis 2 by g2."""
    return rodrigues(v3(a2), mp.mpf(g2),
                     rodrigues(v3(a1), mp.mpf(g1), v3(axis)))


def thrust_magnitude(f_vac, exit_area, level, p_amb):
    """F = lambda F_vac - p_amb A_e for lambda > 0; exactly zero otherwise
    (eq:propulsion:thrust)."""
    lam = mp.mpf(level)
    if lam == 0:
        return mp.mpf(0)
    return lam * mp.mpf(f_vac) - mp.mpf(p_amb) * mp.mpf(exit_area)


def mdot(f_vac, isp_vac, level):
    """mdot = lambda F_vac / (g0 Isp_vac) (eq:propulsion:mdot); the mass
    flow follows the vacuum rating and throttle only."""
    return mp.mpf(level) * mp.mpf(f_vac) / (G0 * mp.mpf(isp_vac))


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


def hexv(vec):
    return [float(x).hex() for x in vec]


def main() -> None:
    x_axis = (1.0, 0.0, 0.0)
    y_axis = (0.0, 1.0, 0.0)
    z_axis = (0.0, 0.0, 1.0)

    # (name, F_vac, Isp_vac, A_e, level, p_amb, (g1, g2),
    #  axis, a1, a2, position, cg)
    engine_cases = [
        ("vacuum_full_throttle", 180000.0, 340.0, 0.65, 1.0, 0.0,
         (0.0, 0.0), x_axis, y_axis, z_axis,
         (0.0, 0.0, 0.0), (2.8, 0.0, 0.0)),
        ("sea_level_full_throttle", 180000.0, 340.0, 0.65, 1.0, 101325.0,
         (0.0, 0.0), x_axis, y_axis, z_axis,
         (0.0, 0.0, 0.0), (2.8, 0.0, 0.0)),
        ("mid_throttle_vacuum", 180000.0, 340.0, 0.65, 0.55, 0.0,
         (0.0, 0.0), x_axis, y_axis, z_axis,
         (0.0, 0.1, -0.05), (2.8, 0.01, 0.02)),
        ("gimballed_ascent", 180000.0, 340.0, 0.65, 0.85, 26500.0,
         (0.02, -0.035), x_axis, y_axis, z_axis,
         (0.0, 0.1, -0.05), (2.8, 0.01, 0.02)),
        ("zero_throttle_exact_zero", 180000.0, 340.0, 0.65, 0.0, 101325.0,
         (0.01, 0.0), x_axis, y_axis, z_axis,
         (0.0, 0.1, -0.05), (2.8, 0.01, 0.02)),
    ]

    out = []
    for (name, f_vac, isp, ae, level, p_amb, gimbal, axis, a1, a2, pos,
         cg) in engine_cases:
        f_mag = thrust_magnitude(f_vac, ae, level, p_amb)
        md = mdot(f_vac, isp, level)
        direction = thrust_direction(axis, a1, a2, gimbal[0], gimbal[1])
        force = vscale(f_mag, direction)
        torque = vcross(vsub(v3(pos), v3(cg)), force)
        out.append({
            "name": name,
            "thrust_vac_N": float(f_vac).hex(),
            "isp_vac_s": float(isp).hex(),
            "exit_area_m2": float(ae).hex(),
            "throttle_level": float(level).hex(),
            "p_amb_Pa": float(p_amb).hex(),
            "gimbal_rad": [float(gimbal[0]).hex(), float(gimbal[1]).hex()],
            "axis": hexv(axis),
            "gimbal_axis_1": hexv(a1),
            "gimbal_axis_2": hexv(a2),
            "position_m": hexv(pos),
            "cg_m": hexv(cg),
            "thrust_N": float(f_mag).hex(),
            "mdot_kgps": float(md).hex(),
            "force_N": hexv(force),
            "torque_Nm": hexv(torque),
        })
        print(f"{name:26s} F={mp.nstr(f_mag, 12)} N  "
              f"mdot={mp.nstr(md, 12)} kg/s")

    emit(
        HERE / "engine.toml",
        "Engine thrust, mass-flow, and force/torque golden vectors (FR-10,\n"
        "Phase 4 exit criterion 3). thrust_N is\n"
        "F = lambda F_vac - p_amb A_e (chapter ch:propulsion,\n"
        "eq:propulsion:thrust); mdot_kgps is lambda F_vac / (g0 Isp_vac)\n"
        "with g0 = 9.80665 m/s^2 exact (eq:propulsion:mdot); force_N is\n"
        "thrust_N times the gimbal-deflected direction\n"
        "(eq:propulsion:direction, rotation about gimbal_axis_1 by\n"
        "gimbal_rad[0], then about gimbal_axis_2 by gimbal_rad[1]);\n"
        "torque_Nm is (position - cg) x force (eq:propulsion:forcetorque).\n"
        "Evaluated with mpmath at 60 significant decimal digits from the\n"
        "exact binary64 inputs and rounded once to binary64. The\n"
        "zero-throttle case is exactly zero in every output. Provenance\n"
        "and tolerances in manifest.toml. Regenerated by generate.py.",
        out,
    )

    # (name, F_vac, Isp_vac, m0, t_burn)
    tsiolkovsky_cases = [
        ("kick_stage_full_burn", 24000.0, 320.0, 1800.0, 120.0),
        ("deep_throttle_long_burn", 4000.0, 452.0, 9500.0, 600.0),
    ]

    out = []
    for name, f_vac, isp, m0, t_burn in tsiolkovsky_cases:
        md = mdot(f_vac, isp, 1.0)
        m1 = mp.mpf(m0) - md * mp.mpf(t_burn)
        assert m1 > 0, (name, m1)
        dv = G0 * mp.mpf(isp) * mp.log(mp.mpf(m0) / m1)
        out.append({
            "name": name,
            "thrust_vac_N": float(f_vac).hex(),
            "isp_vac_s": float(isp).hex(),
            "m0_kg": float(m0).hex(),
            "t_burn_s": float(t_burn).hex(),
            "m1_kg": float(m1).hex(),
            "dv_mps": float(dv).hex(),
            "dv_decimal": mp.nstr(dv, 17),
        })
        print(f"{name:26s} dv={mp.nstr(dv, 12)} m/s  "
              f"m1={mp.nstr(m1, 12)} kg")

    emit(
        HERE / "tsiolkovsky.toml",
        "Vacuum-burn Tsiolkovsky golden scenarios (Phase 4 exit criterion\n"
        "3): full-throttle fixed-attitude burns in vacuum, with\n"
        "m1 = m0 - mdot t_burn and dv = g0 Isp_vac ln(m0 / m1)\n"
        "(chapter ch:propulsion, eq:propulsion:tsiolkovsky), g0 = 9.80665\n"
        "m/s^2 exact. Evaluated with mpmath at 60 significant decimal\n"
        "digits from the exact binary64 inputs and rounded once to\n"
        "binary64. The consuming test integrates the same burn through the\n"
        "model's mdot and thrust and gates the accumulated dv at 0.1 %\n"
        "relative. Provenance and tolerances in manifest.toml. Regenerated\n"
        "by generate.py.",
        out,
    )
    print(f"propulsion goldens regenerated; mpmath {mp.__version__} at "
          f"{mp.mp.dps} dps")


if __name__ == "__main__":
    main()
