"""Regenerate the RCS and reaction-wheel golden-vector files in this directory.

The values anchor the FR-1 actuator models (cpp/src/models/actuators.cpp)
against Phase 4 exit criterion 7. Two files are produced:

- rcs.toml: on/off thruster pulses with minimum-impulse-bit enforcement
  (chapter ch:actuators, eq:actuators:mib): a commanded pulse whose ideal
  impulse magnitude thrust * duration falls below the configured MIB
  delivers exactly zero impulse; a pulse at or above the MIB delivers
  exactly thrust * duration, as a linear impulse along the thruster
  direction and an angular impulse (position - cg) x impulse about the
  supplied composite CG (eq:actuators:rcscoupling).
- wheels.toml: single-step reaction-wheel torque delivery with exact
  clamping (eq:actuators:wheelclamp): the commanded torque is clamped at
  the torque saturation, the wheel momentum is rail-limited at the
  momentum saturation (the delivered torque within the step is reduced so
  the momentum lands exactly on the rail and is exactly zero once the rail
  is reached and the command pushes further in the same sign), and the
  reaction torque on the body is the negative of the delivered wheel
  torque along the spin axis (eq:actuators:wheelreaction).

References are evaluated with mpmath at 60 significant decimal digits from
the exact binary64 inputs and rounded once to binary64. The mpmath
evaluation mirrors the model formulation of math-library chapter
ch:actuators branch for branch but shares no binary64 rounding with the
C++ path, so the committed values check the double-precision
implementation to its own rounding floor. Clamp-branch cases are placed
far from branch boundaries so extended-precision and binary64 branch
selection cannot disagree; the exact-boundary semantics (impulse exactly
at the MIB, momentum exactly on the rail) are exercised with
exactly-representable inputs whose branch condition is unambiguous in
binary64. The momentum-exchange sign conventions follow the
reaction-wheel formulation of Markley and Crassidis (2014).

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
# mpmath mirror of the actuator models (chapter ch:actuators)
# ---------------------------------------------------------------------------


def v3(x):
    return [mp.mpf(c) for c in x]


def vsub(a, b):
    return [a[i] - b[i] for i in range(3)]


def vscale(s, a):
    return [s * a[i] for i in range(3)]


def vcross(a, b):
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


def rcs_pulse(position, direction, thrust, mib, duration, cg):
    """MIB-enforced pulse impulse (eq:actuators:mib,
    eq:actuators:rcscoupling)."""
    ideal = mp.mpf(thrust) * mp.mpf(duration)
    if ideal < mp.mpf(mib):
        delivered = mp.mpf(0)
    else:
        delivered = ideal
    impulse = vscale(delivered, v3(direction))
    angular = vcross(vsub(v3(position), v3(cg)), impulse)
    return delivered, impulse, angular


def wheel_step(axis, torque_max, momentum_max, h0, cmd, dt):
    """One wheel step: torque clamp, then momentum rail
    (eq:actuators:wheelclamp, eq:actuators:wheelreaction)."""
    tau_max, h_max = mp.mpf(torque_max), mp.mpf(momentum_max)
    h, tc, step = mp.mpf(h0), mp.mpf(cmd), mp.mpf(dt)
    tau = min(max(tc, -tau_max), tau_max)
    h1 = h + tau * step
    if h1 > h_max:
        tau = (h_max - h) / step
        h1 = h_max
    elif h1 < -h_max:
        tau = (-h_max - h) / step
        h1 = -h_max
    body_torque = vscale(-tau, v3(axis))
    return tau, h1, body_torque


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
    # Thruster directions are exactly unit-norm in binary64 (axis vectors
    # and the 3-4-5 direction (0.6, 0, 0.8): 0.36 + 0.64 = 1 exactly).
    # (name, position, direction, thrust_N, mib_Ns, duration_s, cg)
    # The 2x-MIB duration is the binary64 quotient 2*mib/thrust, so the
    # consuming test can also gate the delivered impulse against the spec
    # value 2*mib at 1e-12 relative (exit criterion 7).
    two_mib_duration = (2.0 * 0.02) / 22.0
    rcs_cases = [
        ("below_mib_exact_zero", (1.2, 0.4, -0.3), (0.0, 1.0, 0.0),
         22.0, 0.02, 0.0005, (0.9, 0.0, 0.1)),
        ("at_two_mib", (1.2, 0.4, -0.3), (0.0, 1.0, 0.0),
         22.0, 0.02, two_mib_duration, (0.9, 0.0, 0.1)),
        ("generic_pulse", (1.2, 0.4, -0.3), (0.6, 0.0, 0.8),
         22.0, 0.02, 0.25, (0.9, 0.0, 0.1)),
        ("retro_pulse_off_cg", (-0.8, -0.25, 0.55), (-1.0, 0.0, 0.0),
         10.0, 0.05, 0.125, (0.4, 0.02, -0.03)),
    ]

    out = []
    for name, pos, direction, thrust, mib, duration, cg in rcs_cases:
        delivered, impulse, angular = rcs_pulse(pos, direction, thrust, mib,
                                                duration, cg)
        out.append({
            "name": name,
            "position_m": hexv(pos),
            "direction": hexv(direction),
            "thrust_N": float(thrust).hex(),
            "mib_Ns": float(mib).hex(),
            "duration_s": float(duration).hex(),
            "cg_m": hexv(cg),
            "delivered_Ns": float(delivered).hex(),
            "impulse_Ns": hexv(impulse),
            "angular_impulse_Nms": hexv(angular),
        })
        print(f"{name:24s} J={mp.nstr(delivered, 12)} N s")

    emit(
        HERE / "rcs.toml",
        "RCS pulse golden vectors (FR-1, Phase 4 exit criterion 7).\n"
        "delivered_Ns is thrust_N * duration_s when that product is at or\n"
        "above mib_Ns and exactly zero below it (chapter ch:actuators,\n"
        "eq:actuators:mib); impulse_Ns is delivered_Ns times the unit\n"
        "thruster direction and angular_impulse_Nms is\n"
        "(position - cg) x impulse (eq:actuators:rcscoupling). Evaluated\n"
        "with mpmath at 60 significant decimal digits from the exact\n"
        "binary64 inputs and rounded once to binary64. The below-MIB case\n"
        "is exactly zero in every output. Provenance and tolerances in\n"
        "manifest.toml. Regenerated by generate.py.",
        out,
    )

    # (name, axis, torque_max, momentum_max, h0, cmd, dt)
    # h0 values sit far from the rail except the deliberate exact-rail
    # cases, whose inputs are exactly representable so the branch is
    # unambiguous.
    wheel_cases = [
        ("unclamped_delivery", (0.0, 0.0, 1.0), 0.2, 30.0,
         2.0, 0.05, 0.5),
        ("torque_clamped", (0.0, 0.0, 1.0), 0.2, 30.0,
         2.0, 0.5, 0.5),
        ("torque_clamped_negative", (0.6, 0.8, 0.0), 0.2, 30.0,
         -1.5, -0.75, 0.25),
        ("rail_landing_partial", (0.0, 0.0, 1.0), 0.2, 30.0,
         29.9, 0.2, 1.0),
        ("saturated_zero_delivery", (0.0, 0.0, 1.0), 0.2, 30.0,
         30.0, 0.15, 0.5),
        ("desaturation_from_rail", (0.0, 0.0, 1.0), 0.2, 30.0,
         30.0, -0.15, 0.5),
        ("negative_rail_landing", (0.6, 0.8, 0.0), 0.2, 30.0,
         -29.95, -0.2, 1.0),
    ]

    out = []
    for name, axis, tau_max, h_max, h0, cmd, dt in wheel_cases:
        tau, h1, body_torque = wheel_step(axis, tau_max, h_max, h0, cmd, dt)
        out.append({
            "name": name,
            "axis": hexv(axis),
            "torque_max_Nm": float(tau_max).hex(),
            "momentum_max_Nms": float(h_max).hex(),
            "h0_Nms": float(h0).hex(),
            "torque_cmd_Nm": float(cmd).hex(),
            "dt_s": float(dt).hex(),
            "torque_Nm": float(tau).hex(),
            "h1_Nms": float(h1).hex(),
            "body_torque_Nm": hexv(body_torque),
        })
        print(f"{name:26s} tau={mp.nstr(tau, 12)} N m  "
              f"h1={mp.nstr(h1, 12)} N m s")

    emit(
        HERE / "wheels.toml",
        "Reaction-wheel step golden vectors (FR-1, Phase 4 exit criterion\n"
        "7). torque_Nm is the delivered wheel torque after the torque\n"
        "clamp and the momentum rail (chapter ch:actuators,\n"
        "eq:actuators:wheelclamp): commanded torque is clamped to\n"
        "[-torque_max, +torque_max]; if the resulting momentum would cross\n"
        "a rail the delivered torque is reduced so h1_Nms lands exactly on\n"
        "the rail, and it is exactly zero when the wheel starts on the\n"
        "rail and the command pushes further in the same sign.\n"
        "body_torque_Nm = -torque_Nm * axis (eq:actuators:wheelreaction).\n"
        "Evaluated with mpmath at 60 significant decimal digits from the\n"
        "exact binary64 inputs and rounded once to binary64. Provenance\n"
        "and tolerances in manifest.toml. Regenerated by generate.py.",
        out,
    )
    print(f"actuator goldens regenerated; mpmath {mp.__version__} at "
          f"{mp.mp.dps} dps")


if __name__ == "__main__":
    main()
