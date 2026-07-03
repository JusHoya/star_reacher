"""Regenerate the vehicle mass-properties golden-vector files in this directory.

The values anchor the FR-10 analytic settled-tank mass-properties model
(cpp/src/models/massprops.cpp) against Phase 4 exit criterion 2. Two files
are produced:

- slug.toml: settled-propellant slug properties for a draining cylindrical
  tank (axis along vehicle +X, liquid settled against the aft face per
  assumption A-2): fill height h = m / (rho pi R^2), slug CG on the tank
  axis at the aft face plus h/2, and the solid-cylinder inertia about the
  slug's own CG (I_xx = m R^2 / 2, I_yy = I_zz = m (3 R^2 + h^2) / 12),
  plus the analytic depletion rates obtained by the chain rule through the
  fill height (chapter ch:massprops, eq:massprops:slug,
  eq:massprops:slugrates).
- composite.toml: composite vehicle mass properties for two fixed rigid
  bodies plus two tank slugs - total mass, mass-weighted CG, and the
  parallel-axis inertia composition about the composite CG
  (eq:massprops:compose) - together with the analytic composite rates for
  the given per-tank mass-flow rates (eq:massprops:composerates) and the
  closed-form single-body removal that models a jettison event
  (eq:massprops:remove).

References are evaluated with mpmath at 60 significant decimal digits from
the exact binary64 inputs and rounded once to binary64. The mpmath
evaluation mirrors the model formulation of math-library chapter
ch:massprops term for term but shares no binary64 rounding with the C++
path, so the committed values check the double-precision implementation to
its own rounding floor. Every formula is an elementary closed form (mass
integrals over a homogeneous cylinder, the parallel-axis decomposition,
and their exact time derivatives) derived in full in ch:massprops.

Slug masses are the propellant-mass inputs verbatim (the model passes the
propellant mass through rather than reconstructing it from the fill
height), so composite masses are plain sums of the committed inputs; the
inputs are round decimals chosen so those sums are exact in binary64 and
the consuming test can require bit equality on mass (the exit-criterion-2
wet-mass identity).

Running this script rewrites both .toml files byte-identically; any diff
after regeneration means the script or the goldens were edited by hand,
which the FR-22 golden-update discipline forbids.
"""

from __future__ import annotations

import pathlib

import mpmath as mp

mp.mp.dps = 60

HERE = pathlib.Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# mpmath mirror of the mass-properties model (chapter ch:massprops)
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


def m3(rows):
    return [[mp.mpf(x) for x in row] for row in rows]


def madd(a, b):
    return [[a[i][j] + b[i][j] for j in range(3)] for i in range(3)]


def msub(a, b):
    return [[a[i][j] - b[i][j] for j in range(3)] for i in range(3)]


def zero3():
    return [[mp.mpf(0)] * 3 for _ in range(3)]


def parallel_axis(m, d):
    """m * (|d|^2 E - d d^T): the parallel-axis inertia increment
    (eq:massprops:parallelaxis) for a point offset d from the reference."""
    d2 = vdot(d, d)
    out = zero3()
    for i in range(3):
        for j in range(3):
            out[i][j] = m * ((d2 if i == j else mp.mpf(0)) - d[i] * d[j])
    return out


def parallel_axis_rate(m, mdot, d, ddot):
    """Exact time derivative of the parallel-axis increment
    (eq:massprops:composerates): mdot * (|d|^2 E - d d^T)
    + m * (2 (d . ddot) E - ddot d^T - d ddot^T)."""
    d2 = vdot(d, d)
    dd = vdot(d, ddot)
    out = zero3()
    for i in range(3):
        for j in range(3):
            eye = mp.mpf(1) if i == j else mp.mpf(0)
            out[i][j] = (mdot * (d2 * eye - d[i] * d[j])
                         + m * (2 * dd * eye - ddot[i] * d[j]
                                - d[i] * ddot[j]))
    return out


def slug_props(radius, length, aft, rho, m_p):
    """Settled-slug mass, CG, and own-CG inertia (eq:massprops:slug)."""
    R, L = mp.mpf(radius), mp.mpf(length)
    rho, m = mp.mpf(rho), mp.mpf(m_p)
    area = mp.pi * R * R
    h = m / (rho * area)
    assert 0 <= h <= L, (h, L)
    cg = vadd(v3(aft), [h / 2, mp.mpf(0), mp.mpf(0)])
    ixx = m * R * R / 2
    iyy = m * (3 * R * R + h * h) / 12
    inertia = zero3()
    inertia[0][0] = ixx
    inertia[1][1] = iyy
    inertia[2][2] = iyy
    return m, cg, inertia, h


def slug_rates(radius, length, aft, rho, m_p, mdot):
    """Slug CG and inertia rates by the chain rule through the fill height
    (eq:massprops:slugrates). mdot is the signed d(m_p)/dt."""
    R = mp.mpf(radius)
    rho, m, md = mp.mpf(rho), mp.mpf(m_p), mp.mpf(mdot)
    area = mp.pi * R * R
    h = m / (rho * area)
    hdot = md / (rho * area)
    cgdot = [hdot / 2, mp.mpf(0), mp.mpf(0)]
    ixxdot = md * R * R / 2
    iyydot = md * (3 * R * R + h * h) / 12 + m * h * hdot / 6
    idot = zero3()
    idot[0][0] = ixxdot
    idot[1][1] = iyydot
    idot[2][2] = iyydot
    return md, cgdot, idot


def compose(bodies):
    """Composite mass, CG, and inertia about the composite CG
    (eq:massprops:compose). bodies: list of (m, cg, I_own_cg)."""
    M = mp.mpf(0)
    for m, _, _ in bodies:
        M += m
    cbar = [mp.mpf(0)] * 3
    for m, c, _ in bodies:
        cbar = vadd(cbar, vscale(m, c))
    cbar = vscale(1 / M, cbar)
    itot = zero3()
    for m, c, inertia in bodies:
        itot = madd(itot, madd(inertia, parallel_axis(m, vsub(c, cbar))))
    return M, cbar, itot


def compose_rates(bodies, rates):
    """Composite mass-flow, CG rate, and inertia rate
    (eq:massprops:composerates). rates: list of (mdot, cgdot, idot)
    aligned with bodies."""
    M, cbar, _ = compose(bodies)
    mdot_tot = mp.mpf(0)
    for md, _, _ in rates:
        mdot_tot += md
    num = [mp.mpf(0)] * 3
    for (m, c, _), (md, cd, _) in zip(bodies, rates):
        num = vadd(num, vadd(vscale(md, c), vscale(m, cd)))
    cbardot = vscale(1 / M, vsub(num, vscale(mdot_tot, cbar)))
    idot_tot = zero3()
    for (m, c, _), (md, cd, idot) in zip(bodies, rates):
        d = vsub(c, cbar)
        ddot = vsub(cd, cbardot)
        idot_tot = madd(idot_tot,
                        madd(idot, parallel_axis_rate(m, md, d, ddot)))
    return mdot_tot, cbardot, idot_tot


def remove_body(comp, item):
    """Closed-form composite after removing one body
    (eq:massprops:remove). comp, item: (m, cg, I)."""
    M, cbar, itot = comp
    mj, cj, ij = item
    m_rem = M - mj
    c_rem = vscale(1 / m_rem, vsub(vscale(M, cbar), vscale(mj, cj)))
    i_rem = msub(msub(itot, madd(ij, parallel_axis(mj, vsub(cj, cbar)))),
                 parallel_axis(m_rem, vsub(c_rem, cbar)))
    return m_rem, c_rem, i_rem


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


def hexm(mat):
    return [float(mat[i][j]).hex() for i in range(3) for j in range(3)]


def main() -> None:
    # (name, radius, length, aft_center, density, propellant_kg, mdot_kgps)
    # Round-decimal inputs (exact binary64); fills well inside [0, capacity]
    # except the deliberate empty case.
    slug_cases = [
        ("generic_half_full", 1.9, 7.0, (2.5, 0.0, 0.0), 810.0,
         30000.0, -250.0),
        ("near_empty", 1.9, 7.0, (2.5, 0.0, 0.0), 810.0, 500.0, -250.0),
        ("near_full", 1.9, 7.0, (2.5, 0.0, 0.0), 810.0, 64000.0, -250.0),
        ("exactly_empty", 1.9, 7.0, (2.5, 0.0, 0.0), 810.0, 0.0, 0.0),
        ("off_axis_tank", 0.6, 2.4, (-1.2, 0.35, -0.8), 1000.0,
         1500.0, -12.5),
    ]

    out = []
    for name, radius, length, aft, rho, m_p, mdot in slug_cases:
        m, cg, inertia, h = slug_props(radius, length, aft, rho, m_p)
        md, cgdot, idot = slug_rates(radius, length, aft, rho, m_p, mdot)
        out.append({
            "name": name,
            "radius_m": float(radius).hex(),
            "length_m": float(length).hex(),
            "aft_center_m": hexv(aft),
            "density_kgpm3": float(rho).hex(),
            "propellant_kg": float(m_p).hex(),
            "mdot_kgps": float(mdot).hex(),
            "fill_height_m": float(h).hex(),
            "cg_m": hexv(cg),
            "inertia_kgm2": hexm(inertia),
            "cg_rate_mps": hexv(cgdot),
            "inertia_rate_kgm2ps": hexm(idot),
        })
        print(f"{name:20s} h={mp.nstr(h, 12)} m  "
              f"Ixx={mp.nstr(inertia[0][0], 12)} kg m^2")

    emit(
        HERE / "slug.toml",
        "Draining-cylinder settled-slug golden vectors (FR-10, Phase 4 exit\n"
        "criterion 2). The tank axis lies along vehicle +X; aft_center_m is\n"
        "the center of the aft (-X) face the settled liquid rests against\n"
        "(assumption A-2). Values are the closed forms of chapter\n"
        "ch:massprops (eq:massprops:slug, eq:massprops:slugrates) evaluated\n"
        "with mpmath at 60 significant decimal digits from the exact\n"
        "binary64 inputs and rounded once to binary64. mdot_kgps is the\n"
        "signed d(m_p)/dt (negative while draining). Provenance and\n"
        "tolerances in manifest.toml. Regenerated by generate.py.",
        out,
    )

    # Composite cases: exactly two fixed bodies plus two tanks each, so the
    # golden reader's flat key space stays fixed. body1 is the jettisoned
    # item of the removal outputs. Inertia inputs are symmetric positive
    # definite with dominant diagonals (physically admissible).
    composite_cases = [
        {
            "name": "stack_burn",
            "bodies": [
                (1200.0, (3.0, 0.02, -0.01),
                 ((900.0, 12.0, -8.0), (12.0, 4200.0, 5.0),
                  (-8.0, 5.0, 4300.0))),
                (150.0, (6.5, 0.0, 0.05),
                 ((40.0, 0.0, 0.0), (0.0, 55.0, 0.0), (0.0, 0.0, 52.0))),
            ],
            "tanks": [
                (0.9, 3.2, (1.0, 0.0, 0.0), 810.0, 4000.0, -85.0),
                (0.9, 2.0, (4.4, 0.0, 0.0), 1140.0, 2500.0, -40.0),
            ],
        },
        {
            "name": "asymmetric_stack",
            "bodies": [
                (800.0, (1.5, -0.3, 0.2),
                 ((500.0, 20.0, -15.0), (20.0, 900.0, 10.0),
                  (-15.0, 10.0, 950.0))),
                (60.0, (0.2, 0.8, -0.4),
                 ((15.0, 1.0, 0.0), (1.0, 18.0, 2.0), (0.0, 2.0, 20.0))),
            ],
            "tanks": [
                (0.5, 1.8, (0.4, 0.6, 0.0), 1000.0, 1000.0, -20.0),
                (0.4, 1.5, (0.4, -0.6, 0.1), 1450.0, 700.0, 0.0),
            ],
        },
    ]

    out = []
    for case in composite_cases:
        bodies = []
        rates = []
        rec: dict = {"name": case["name"]}
        for k, (m, cg, inertia) in enumerate(case["bodies"]):
            bodies.append((mp.mpf(m), v3(cg), m3(inertia)))
            rates.append((mp.mpf(0), [mp.mpf(0)] * 3, zero3()))
            rec[f"body{k}_mass_kg"] = float(m).hex()
            rec[f"body{k}_cg_m"] = hexv(cg)
            rec[f"body{k}_inertia_kgm2"] = hexm(m3(inertia))
        for k, (radius, length, aft, rho, m_p, mdot) in enumerate(
                case["tanks"]):
            m, cg, inertia, _ = slug_props(radius, length, aft, rho, m_p)
            bodies.append((m, cg, inertia))
            rates.append(slug_rates(radius, length, aft, rho, m_p, mdot))
            rec[f"tank{k}_radius_m"] = float(radius).hex()
            rec[f"tank{k}_length_m"] = float(length).hex()
            rec[f"tank{k}_aft_center_m"] = hexv(aft)
            rec[f"tank{k}_density_kgpm3"] = float(rho).hex()
            rec[f"tank{k}_propellant_kg"] = float(m_p).hex()
            rec[f"tank{k}_mdot_kgps"] = float(mdot).hex()

        M, cbar, itot = compose(bodies)
        mdot_tot, cbardot, idot_tot = compose_rates(bodies, rates)
        # Jettison: remove body index 1 (the payload/jettison item) in
        # closed form from the composite.
        m_rem, c_rem, i_rem = remove_body((M, cbar, itot), bodies[1])

        rec["mass_kg"] = float(M).hex()
        rec["cg_m"] = hexv(cbar)
        rec["inertia_kgm2"] = hexm(itot)
        rec["mdot_kgps"] = float(mdot_tot).hex()
        rec["cg_rate_mps"] = hexv(cbardot)
        rec["inertia_rate_kgm2ps"] = hexm(idot_tot)
        rec["jettison_body_index"] = "1"
        rec["post_jettison_mass_kg"] = float(m_rem).hex()
        rec["post_jettison_cg_m"] = hexv(c_rem)
        rec["post_jettison_inertia_kgm2"] = hexm(i_rem)
        out.append(rec)
        print(f"{case['name']:20s} M={mp.nstr(M, 12)} kg  "
              f"cg_x={mp.nstr(cbar[0], 12)} m")

    emit(
        HERE / "composite.toml",
        "Composite vehicle mass-properties golden vectors (FR-10, Phase 4\n"
        "exit criterion 2): two fixed rigid bodies plus two settled tank\n"
        "slugs, composed by mass-weighted CG and parallel-axis inertia\n"
        "summation about the composite CG (chapter ch:massprops,\n"
        "eq:massprops:compose), with the analytic depletion rates\n"
        "(eq:massprops:composerates) and the closed-form removal of body 1\n"
        "modeling a jettison event (eq:massprops:remove). Evaluated with\n"
        "mpmath at 60 significant decimal digits from the exact binary64\n"
        "inputs and rounded once to binary64. Composite masses are exact\n"
        "binary64 sums of the committed inputs by construction. Provenance\n"
        "and tolerances in manifest.toml. Regenerated by generate.py.",
        out,
    )
    print(f"massprop goldens regenerated; mpmath {mp.__version__} at "
          f"{mp.mp.dps} dps")


if __name__ == "__main__":
    main()
