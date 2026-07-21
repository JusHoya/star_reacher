#!/usr/bin/env python3
"""Criterion-8 cross-platform divergence tooling (PRD Phase 2 exit criterion 8).

One file owns the final-state interchange format end to end, as three
subcommands used by ``.github/workflows/ci.yml``, plus the three-subcommand
channel-level widening documented under "Channel-level comparison" below:

``extract``
    On a ``build-test`` matrix leg: load a ``run.srlog`` produced by
    ``star run missions/twobody_leo.toml``, take the FINAL truth record,
    and write ``finalstate.json`` carrying the leg identity and the state
    at full precision (authoritative fields are ``float.hex()`` strings;
    decimal mirrors are included for human reading only).

``measure``
    On the aggregation job: collect every leg's ``finalstate.json``,
    verify the leg count, leg uniqueness, and bit-identical final epochs,
    compute the maximum pairwise relative divergence, write
    ``measurement.json``, and print machine-readable ``max_rel=`` and
    ``status_state=`` lines (the workflow appends them to
    ``$GITHUB_OUTPUT`` and publishes ``max_rel`` as a commit status, which
    is readable without authentication on a public repository).

``gate``
    Enforce criterion 8 against the committed record
    ``tests/golden/determinism/cross_platform.toml``.

Divergence definition (this docstring is its single home):

    For every unordered pair of legs (i, j):
        rel_pos(i, j) = ||r_i - r_j||_2 / min(||r_i||_2, ||r_j||_2)
        rel_vel(i, j) = ||v_i - v_j||_2 / min(||v_i||_2, ||v_j||_2)
    max_rel = max over all pairs of max(rel_pos, rel_vel)

where r and v are the final truth-record position [m] and velocity [m/s]
vectors and ||.||_2 is the Euclidean norm. The scale is the smaller of the
two state-vector norms - the conservative choice, since it can only
enlarge the ratio. For the reference LEO mission ||r|| ~ 6.8e6 m and
||v|| ~ 7.6e3 m/s, so the denominators are far from zero; a zero-norm
state is rejected as undefined rather than silently scaled.

Gate rules (any failure = nonzero exit; no rule is advisory):

- ``measure`` fails when a leg artifact is missing or duplicated, or when
  the final epochs are not bit-identical across legs (an epoch mismatch is
  a configuration divergence, which would invalidate the comparison).
- ``gate`` fails when the measured ``max_rel`` exceeds the bound (D-10:
  1e-9 relative); when the committed record is missing, unparseable, or
  has ``status = "pending-first-measurement"`` (the designed bootstrap:
  the first CI run fails until the maintainer completes the record per its
  in-file procedure); when the record's ``measured_max_rel`` exceeds the
  bound; or when the record's ``bound_rel`` disagrees with the enforced
  bound (a D-10 revision must update both homes in the same change).
- ``gate`` WARNS, but does not fail, when the fresh measurement and the
  committed record differ by more than a factor of 10: runner-image
  compiler updates legitimately move the last digits, and the warning
  flags a record due for refresh via its recorded procedure.

Channel-level comparison (``extract-channels`` / ``measure-channels`` /
``gate-channels``)
------------------------------------------------------------------------

The three subcommands above sample exactly two channels of one mission:
``truth.r_m`` and ``truth.v_mps`` of ``missions/twobody_leo.toml``. The
byte-determinism measurement recorded in ``docs/ci/phase6_crossplatform.md``
established that those are precisely the channels that stay bit-identical
across MSVC and GCC in every point-mass mission, so that gate cannot observe
the divergence the same measurement found in the sensor, ``nav.*``,
``env.*`` and guidance channels. It would pass unchanged if cross-platform
divergence grew by orders of magnitude. The channel subcommands close that
hole by comparing every channel of every mission in ``CHANNEL_MISSIONS``.

Each channel is placed in one of two classes, and the classes are gated
differently because they are guaranteed differently:

``exact``
    Compared for **bit-identity**; any difference fails. Membership is
    structural rather than empirical: ``u32``/``u64``/``str16`` channels
    (integers, validity flags, event codes, event detail strings) and every
    group's ``t_s`` channel carry no libm-dependent arithmetic, so a
    difference is a control-flow or time-grid divergence rather than a
    rounding artifact. A mission declared ``arithmetic = "basic-ops-only"``
    puts *every* channel in this class, and a libm-bearing mission may
    additionally declare float channels whose value chain is provably
    restricted to IEEE-754 basic operations (``+ - * /`` and ``sqrt``),
    which the standard requires to be correctly rounded on every conforming
    platform. The SRLOG header is compared for byte-identity on every
    mission; it carries no platform-dependent field, and its one float
    payload (the v1.3 ``gnc.camera`` echo) is encoded as hex bit patterns
    precisely so it crosses exactly.

``tolerance``
    Every remaining float channel: the libm-bearing surface. Cross-binary
    byte identity is **false** here and asserting it would be wrong, not
    strict - IEEE-754 does not require correct rounding for transcendental
    functions, and a 1 ULP disagreement between two conforming libm
    implementations is legitimate. These channels are gated against
    ``CHANNEL_TOLERANCE_REL``, derived below.

Divergence for a tolerance-class channel, for every unordered pair of legs
(i, j):

    diff(i, j)  = max over all elements of |x_i - x_j|
    scale(i, j) = min(rms(x_i), rms(x_j))
    rel(i, j)   = diff(i, j) / scale(i, j)

normalizing by the channel's own root-mean-square rather than pointwise,
because a channel whose values pass through zero yields a meaningless
pointwise ratio (a sign flip at 1e-16 reports rel = 2.0). ``min`` of the two
scales mirrors the conservative choice made for the final-state metric. Both
sides of every comparison are decoded and reduced on the aggregation host
from the raw little-endian payloads the legs upload, so the arithmetic of
the comparison itself is single-platform by construction.

Anti-degeneracy rules. A channel that is identically zero on both legs
compares equal for free, so a gate built only on the tolerance above would
go green while structurally unable to see its target - the failure mode this
widening exists to correct. Each mission therefore declares
``require_active``: channels that must be present AND carry a nonzero RMS,
and ``min_active``: a floor on the number of tolerance-class channels with a
nonzero RMS. A refactor that drops a group, zeroes a sensor, or stops
emitting density fails the gate instead of quietly shrinking it.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import struct
import sys
from itertools import combinations
from pathlib import Path


def _fail(message: str) -> int:
    print(f"cross_platform_divergence: {message}", file=sys.stderr)
    return 1


# ---------------------------------------------------------------------------
# Final-state files and the divergence computation
# ---------------------------------------------------------------------------


def load_finalstate(path: Path) -> dict:
    """Parse one finalstate.json into floats (hex fields are authoritative)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("leg", "t_s_hex", "r_m_hex", "v_mps_hex"):
        if key not in data:
            raise ValueError(f"{path}: missing required field {key!r}")
    return {
        "leg": str(data["leg"]),
        "t_s_hex": str(data["t_s_hex"]),
        "t_s": float.fromhex(data["t_s_hex"]),
        "r_m": [float.fromhex(h) for h in data["r_m_hex"]],
        "v_mps": [float.fromhex(h) for h in data["v_mps_hex"]],
    }


def _norm(v: list[float]) -> float:
    return math.hypot(*v)


def _rel(a: list[float], b: list[float], what: str, pair: str) -> float:
    scale = min(_norm(a), _norm(b))
    if scale == 0.0:
        raise ValueError(
            f"zero-norm {what} vector in pair {pair}: the relative scale is undefined"
        )
    return math.dist(a, b) / scale


def max_pairwise_divergence(states: list[dict]) -> dict:
    """Maximum pairwise relative divergence per the module-docstring formula."""
    if len(states) < 2:
        raise ValueError("divergence needs at least two legs")
    max_rel = -1.0
    max_rel_pos = 0.0
    max_rel_vel = 0.0
    worst_pair = ""
    worst_quantity = ""
    for i, j in combinations(range(len(states)), 2):
        a, b = states[i], states[j]
        pair = f"{a['leg']} vs {b['leg']}"
        rel_pos = _rel(a["r_m"], b["r_m"], "position", pair)
        rel_vel = _rel(a["v_mps"], b["v_mps"], "velocity", pair)
        max_rel_pos = max(max_rel_pos, rel_pos)
        max_rel_vel = max(max_rel_vel, rel_vel)
        for quantity, rel in (("position", rel_pos), ("velocity", rel_vel)):
            if rel > max_rel:
                max_rel = rel
                worst_pair = pair
                worst_quantity = quantity
    return {
        "max_rel": max_rel,
        "max_rel_pos": max_rel_pos,
        "max_rel_vel": max_rel_vel,
        "worst_pair": worst_pair,
        "worst_quantity": worst_quantity,
    }


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_extract(args: argparse.Namespace) -> int:
    # Lazy imports: only the build-test legs have the package installed;
    # measure/gate run on a bare checkout with stdlib only.
    import platform

    from star_reacher.srlog import load

    run = load(Path(args.srlog))
    truth = run.groups["truth"]
    if len(truth) == 0:
        return _fail(f"{args.srlog}: empty truth group")
    rec = truth[-1]
    payload = {
        "schema": 1,
        "leg": args.leg,
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        # float.hex() fields are the authoritative full-precision values;
        # the decimal fields mirror them for human reading only.
        "t_s_hex": float(rec["t_s"]).hex(),
        "r_m_hex": [float(x).hex() for x in rec["r_m"]],
        "v_mps_hex": [float(x).hex() for x in rec["v_mps"]],
        "t_s": float(rec["t_s"]),
        "r_m": [float(x) for x in rec["r_m"]],
        "v_mps": [float(x) for x in rec["v_mps"]],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out} (leg {args.leg}, final t = {payload['t_s']!r} s)")
    return 0


def cmd_measure(args: argparse.Namespace) -> int:
    root = Path(args.dir)
    files = sorted(root.rglob("finalstate.json")) if root.is_dir() else []
    if not files:
        return _fail(f"no finalstate.json found under {root}")
    try:
        states = [load_finalstate(f) for f in files]
    except (ValueError, json.JSONDecodeError) as exc:
        return _fail(f"unparseable final-state artifact: {exc}")
    legs = [s["leg"] for s in states]
    if len(states) != args.expect_legs:
        return _fail(
            f"expected {args.expect_legs} leg artifacts, found {len(states)}: {legs} "
            f"(a missing leg means the comparison would silently shrink; failing instead)"
        )
    if len(set(legs)) != len(legs):
        return _fail(f"duplicate leg identifiers: {legs}")
    if len({s["t_s_hex"] for s in states}) != 1:
        detail = ", ".join(f"{s['leg']}: t = {s['t_s']!r} s" for s in states)
        return _fail(f"final epochs are not bit-identical across legs ({detail})")
    try:
        result = max_pairwise_divergence(states)
    except ValueError as exc:
        return _fail(str(exc))
    payload = {
        "schema": 1,
        "expect_legs": args.expect_legs,
        "bound_rel": args.bound,
        "t_s_hex": states[0]["t_s_hex"],
        "legs": {
            s["leg"]: {"r_m_hex": [x.hex() for x in s["r_m"]],
                       "v_mps_hex": [x.hex() for x in s["v_mps"]]}
            for s in states
        },
        **result,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"legs ({len(states)}): {', '.join(sorted(legs))}")
    print(f"final epoch t = {states[0]['t_s']!r} s (bit-identical on all legs)")
    print(
        f"max pairwise relative divergence: position {result['max_rel_pos']:.3e}, "
        f"velocity {result['max_rel_vel']:.3e}; worst {result['worst_quantity']} "
        f"pair: {result['worst_pair']}"
    )
    # Machine-readable lines consumed into $GITHUB_OUTPUT by the workflow.
    print(f"max_rel={result['max_rel']:.3e}")
    print(f"status_state={'success' if result['max_rel'] <= args.bound else 'failure'}")
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    try:
        measurement = json.loads(Path(args.measurement).read_text(encoding="utf-8"))
        measured = float(measurement["max_rel"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return _fail(f"cannot read measurement {args.measurement}: {exc}")

    failures: list[str] = []
    if measured > args.bound:
        failures.append(
            f"measured max_rel {measured:.3e} exceeds the D-10 bound {args.bound:.1e}; "
            f"per exit criterion 8 the bound must be formally revised in the same "
            f"change if this measurement is accepted"
        )

    record_path = Path(args.record)
    record_value: float | None = None
    if not record_path.is_file():
        failures.append(f"committed record {record_path} is missing")
    else:
        import tomllib

        try:
            with open(record_path, "rb") as fh:
                record = tomllib.load(fh)["record"]
        except (tomllib.TOMLDecodeError, KeyError) as exc:
            record = None
            failures.append(f"committed record {record_path} is unparseable: {exc}")
        if record is not None:
            status = record.get("status")
            if status == "pending-first-measurement":
                failures.append(
                    "the committed record is pending-first-measurement: this first "
                    "run is the designed bootstrap failure. Read the measured value "
                    "from the determinism/cross-platform commit status and complete "
                    f"{record_path} per the procedure in its comments."
                )
            elif status != "measured":
                failures.append(f"record status {status!r} is not 'measured'")
            else:
                if "measured_max_rel" not in record:
                    failures.append("record status is 'measured' but measured_max_rel is absent")
                else:
                    record_value = float(record["measured_max_rel"])
                    if record_value > args.bound:
                        failures.append(
                            f"committed measured_max_rel {record_value:.3e} exceeds "
                            f"the bound {args.bound:.1e}"
                        )
            bound_rel = record.get("bound_rel")
            if bound_rel is not None and float(bound_rel) != args.bound:
                failures.append(
                    f"record bound_rel {float(bound_rel):.1e} disagrees with the "
                    f"enforced bound {args.bound:.1e}; a D-10 revision must update "
                    f"both homes in the same change"
                )

    if record_value is not None and not failures:
        # Factor-10 drift between the fresh measurement and the record is a
        # prominent warning, never a failure: runner-image compiler updates
        # legitimately move the last digits.
        lo, hi = sorted((measured, record_value))
        if (lo == 0.0 and hi > 0.0) or (lo > 0.0 and hi > 10.0 * lo):
            banner = "!" * 78
            print(banner)
            print(
                f"WARNING: measured max_rel {measured:.3e} differs from the committed "
                f"record {record_value:.3e} by more than a factor of 10."
            )
            print(
                "Runner-image compiler updates legitimately move the last digits; "
                "refresh tests/golden/determinism/cross_platform.toml via the "
                "procedure recorded in its comments."
            )
            print(banner)

    if failures:
        for f in failures:
            print(f"cross_platform_divergence: gate failure: {f}", file=sys.stderr)
        return 1
    print(
        f"criterion-8 gate passed: measured max_rel {measured:.3e} and committed "
        f"record {record_value:.3e} are both within {args.bound:.1e}"
    )
    return 0


# ---------------------------------------------------------------------------
# Channel-level comparison: tolerance derivation and mission declarations
# ---------------------------------------------------------------------------

# Worst scale-relative divergence measured anywhere in the five-mission
# byte-determinism pass across Windows/MSVC and Linux/GCC, both x86-64:
# gnc.cmd.w_cmd_b_radps in the 760 s powered ascent, 7,600 integration steps
# through the most transcendental-dense model chain in the project
# (docs/ci/phase6_crossplatform.md, "Result 3 - where the four GNC missions
# differ"). It is the largest error-accumulation chain the mission set offers.
MEASURED_WORST_REL = 1.06e-10

# D-10's hard ceiling on cross-platform divergence, enforced by the
# final-state gate above and repeated here so the channel tolerance can be
# shown to sit strictly inside it.
D10_BOUND_REL = 1e-9

# The channel tolerance is the geometric mean of the two: it sits the SAME
# multiplicative distance (3.07x) above the worst value ever measured as it
# does below the D-10 ceiling. That symmetry is the whole argument for the
# number, and it buys two things that pull in opposite directions:
#
#   - Headroom against a false red. The 1.06e-10 figure comes from one libm
#     pair on one instruction set (MSVC CRT and glibc 2.39, both x86-64).
#     Two of the four CI legs - macOS 15 and ubuntu-24.04-arm - run libm
#     implementations this project has never measured, and aarch64 changes
#     the instruction set rather than merely the library. A tolerance set at
#     the measured worst case would red on any leg that merely disagrees at
#     more arguments than the sampled pair did.
#   - Headroom against masking. A tolerance at or near 1e-9 would make this
#     gate incapable of failing before D-10 already had, which is the exact
#     defect being corrected. At 3.07x below the ceiling, a channel breaches
#     this gate well before it breaches D-10.
#
# Choosing the geometric rather than the arithmetic mean is deliberate:
# divergence of this kind is compared by order of magnitude, so equal ratios
# - not equal differences - are what "the same margin either way" means.
CHANNEL_TOLERANCE_REL = math.sqrt(MEASURED_WORST_REL * D10_BOUND_REL)

# Channels whose name marks them exact-class regardless of dtype. The record
# time grid is emitted from the integrator's step accounting and was measured
# bit-identical in every group of every mission on both platforms; a moved
# t_s means records were written at different epochs, which invalidates the
# comparison rather than merely perturbing it.
_EXACT_CHANNEL_NAMES = frozenset({"t_s"})

# dtypes that carry no floating-point arithmetic at all.
_NON_FLOAT_DTYPES = frozenset({"u32", "u64", "str16"})

CHANNEL_MISSIONS: dict[str, dict] = {
    "twobody_leo": {
        "path": "missions/twobody_leo.toml",
        # Every channel exact: the Phase 1 byte-frozen two-body propagation
        # evaluates add/subtract/multiply/divide/sqrt only. This mission was
        # measured byte-identical whole-file across MSVC and GCC, so the
        # class assignment is both derived and observed.
        "arithmetic": "basic-ops-only",
        "require_active": [],
        "min_active": 0,
    },
    "leo_attitude_gnc": {
        "path": "missions/leo_attitude_gnc.toml",
        "arithmetic": "libm",
        # Point-mass gravity drives translation here with no aero and no
        # thrust, so the translational state and the mass properties stay on
        # the basic-operation surface even though the attitude loop does not.
        "exact_float_channels": [
            "truth.r_m",
            "truth.v_mps",
            "mass.mass_kg",
            "mass.cg_b_m",
            "mass.inertia_b_kgm2",
        ],
        # The attitude loop, its IMU, and dead-reckoning navigation: the
        # channels that motivated including this mission at all. The IMU's
        # dv_b_mps is deliberately NOT required: this mission configures no
        # thrust and no drag, so the specific-force increment is identically
        # zero and would satisfy a naive requirement for free.
        "require_active": [
            "truth.q_i2b",
            "truth.w_b_radps",
            "sensors.imu.dtheta_b_rad",
            "nav.err.e",
            "gnc.cmd.tau_b_nm",
        ],
        "min_active": 12,
    },
    "leo_ekf_consistency": {
        "path": "missions/leo_ekf_consistency.toml",
        "arithmetic": "libm",
        "exact_float_channels": [
            "truth.r_m",
            "truth.v_mps",
            "mass.mass_kg",
            "mass.cg_b_m",
            "mass.inertia_b_kgm2",
        ],
        # The widest numerical surface in the phase: the error-state EKF with
        # aiding updates from a nav fix, a star tracker and an altimeter.
        "require_active": [
            "nav.est.x_hat",
            "nav.est.P",
            "nav.innov.y",
            "nav.innov.S",
            "sensors.altimeter.alt_meas_m",
            "env.alt_m",
        ],
        "min_active": 20,
    },
    "leo_optical_nav": {
        "path": "missions/leo_optical_nav.toml",
        "arithmetic": "libm",
        "exact_float_channels": [
            "truth.r_m",
            "truth.v_mps",
            "mass.mass_kg",
            "mass.cg_b_m",
            "mass.inertia_b_kgm2",
            # The camera pose channels are verbatim copies of the truth
            # translational state (ch:camera exit-criterion-7 clause), so they
            # inherit its exactness on this point-mass mission. Their
            # attitude sibling q_i2b does not, and stays in the tolerance
            # class with the rest of the rotational state.
            "sensors.camera.r_m",
        ],
        # The optical surface: the SRLOG v1.3 camera path, which no other
        # committed mission emits.
        "require_active": [
            "sensors.camera.px_uv",
            "sensors.camera.q_i2b",
            "truth.q_i2b",
            "sensors.imu.dtheta_b_rad",
            "nav.err.e",
            "gnc.cmd.tau_b_nm",
        ],
        "min_active": 14,
    },
    "ascent_leo_gnc": {
        "path": "missions/ascent_leo_gnc.toml",
        "arithmetic": "libm",
        # No exact float channels: this is the one mission whose translational
        # forces pass through the atmosphere and aero models, and truth.r_m
        # and truth.v_mps were measured to diverge here (3.8 nm / 20 pm/s
        # after 760 s) while staying exact in every point-mass mission.
        "exact_float_channels": [],
        # The transcendental-dense chain: exp-bearing density, the aero
        # tables, the pitch-program sin/cos roll reference, and the
        # translational state they feed.
        "require_active": [
            "env.rho_kgpm3",
            "env.q_pa",
            "forces.f_aero_b_n",
            "gnc.cmd.w_cmd_b_radps",
            "truth.r_m",
            "truth.v_mps",
        ],
        "min_active": 25,
    },
}


def _channel_key(group: str, channel: str) -> str:
    return f"{group}.{channel}"


def _channel_class(mission: dict, group: str, channel: str, dtype: str) -> str:
    """'exact' or 'tolerance' for one channel under one mission's declaration."""
    if mission["arithmetic"] == "basic-ops-only":
        return "exact"
    if dtype in _NON_FLOAT_DTYPES or channel in _EXACT_CHANNEL_NAMES:
        return "exact"
    if _channel_key(group, channel) in mission.get("exact_float_channels", ()):
        return "exact"
    return "tolerance"


def _channel_bytes(arr, name: str, dtype: str) -> bytes:
    """Little-endian payload bytes for one channel of a loaded group.

    Fixed-width channels are already stored in explicitly little-endian NumPy
    fields, so their buffer is the wire form on every host. str16 payloads are
    re-packed in the format's own length-prefixed encoding rather than joined
    with a separator, so no detail string can collide with a delimiter.
    """
    column = arr[name]
    if dtype == "str16":
        out = bytearray()
        for value in column:
            encoded = str(value).encode("utf-8")
            out += struct.pack("<H", len(encoded)) + encoded
        return bytes(out)
    return column.tobytes()


def _decode_f64(payload: bytes) -> tuple[float, ...]:
    if len(payload) % 8:
        raise ValueError(f"float payload of {len(payload)} bytes is not a whole number of doubles")
    return struct.unpack(f"<{len(payload) // 8}d", payload)


def _rms(values) -> float:
    if not values:
        return 0.0
    return math.sqrt(math.fsum(x * x for x in values) / len(values))


def _max_abs_diff(a, b) -> float:
    worst = 0.0
    for x, y in zip(a, b):
        d = abs(x - y)
        if d > worst:
            worst = d
    return worst


# ---------------------------------------------------------------------------
# Channel-level subcommands
# ---------------------------------------------------------------------------


def cmd_extract_channels(args: argparse.Namespace) -> int:
    # Lazy import for the same reason as cmd_extract: only the build-test
    # legs have the package installed.
    import platform

    from star_reacher.srlog import load

    mission = CHANNEL_MISSIONS.get(args.mission)
    if mission is None:
        return _fail(
            f"unknown mission {args.mission!r}; declared missions are "
            f"{', '.join(sorted(CHANNEL_MISSIONS))}"
        )

    srlog_path = Path(args.srlog)
    raw = srlog_path.read_bytes()
    run = load(srlog_path)

    # The header bytes are the exact slice the writer emitted, re-derived from
    # the file rather than re-serialized from the parsed dict: a JSON round
    # trip could normalize key order or spacing and hide a real difference.
    header_json_len = struct.unpack_from("<I", raw, 12)[0]
    header_bytes = raw[16 : 16 + header_json_len]

    exact: dict[str, dict] = {}
    tolerance: dict[str, dict] = {}
    for group_meta in run.header["groups"]:
        gname = group_meta["name"]
        arr = run.groups.get(gname)
        if arr is None:
            continue
        for channel_meta in group_meta["channels"]:
            cname = channel_meta["name"]
            dtype = channel_meta["dtype"]
            key = _channel_key(gname, cname)
            payload = _channel_bytes(arr, cname, dtype)
            if _channel_class(mission, gname, cname, dtype) == "exact":
                exact[key] = {
                    "n": int(len(arr)),
                    "dtype": dtype,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            else:
                # The full payload travels, not a sample: the divergence
                # metric is a maximum over every element, and a sampled
                # maximum would understate it by an unknown amount.
                tolerance[key] = {
                    "n": int(len(arr)),
                    "dtype": dtype,
                    "b64": base64.b64encode(payload).decode("ascii"),
                }

    payload = {
        "schema": 1,
        "leg": args.leg,
        "mission": args.mission,
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "srlog_sha256": hashlib.sha256(raw).hexdigest(),
        "header_sha256": hashlib.sha256(header_bytes).hexdigest(),
        "exact": exact,
        "tolerance": tolerance,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"wrote {out} (leg {args.leg}, mission {args.mission}: "
        f"{len(exact)} exact-class and {len(tolerance)} tolerance-class channels)"
    )
    return 0


def compare_mission(artifacts: list[dict]) -> dict:
    """Compare one mission's per-leg channel artifacts.

    Returns the per-mission result: exact-class violations (each one a hard
    failure), the worst tolerance-class divergence with the channel and pair
    that produced it, and the activity census the anti-degeneracy rules read.
    """
    if len(artifacts) < 2:
        raise ValueError("a channel comparison needs at least two legs")
    name = artifacts[0]["mission"]
    mission = CHANNEL_MISSIONS[name]
    violations: list[str] = []

    if len({a["header_sha256"] for a in artifacts}) != 1:
        detail = ", ".join(f"{a['leg']}: {a['header_sha256'][:12]}" for a in artifacts)
        violations.append(f"{name}: SRLOG headers are not byte-identical across legs ({detail})")

    # An exact-class channel present on one leg and absent on another is a
    # structural divergence, so the key sets are compared before the values.
    key_sets = {frozenset(a["exact"]) | frozenset(a["tolerance"]) for a in artifacts}
    if len(key_sets) != 1:
        union = set().union(*key_sets)
        common = set.intersection(*[set(k) for k in key_sets])
        violations.append(
            f"{name}: legs do not declare the same channel set; "
            f"{sorted(union - common)} appear on some legs only"
        )

    for key in sorted(artifacts[0]["exact"]):
        digests = {a["leg"]: a["exact"].get(key, {}).get("sha256") for a in artifacts}
        if len(set(digests.values())) != 1:
            detail = ", ".join(f"{leg}: {(d or 'absent')[:12]}" for leg, d in sorted(digests.items()))
            violations.append(
                f"{name}: exact-class channel {key} is not bit-identical across legs ({detail})"
            )

    decoded: dict[str, dict[str, tuple[float, ...]]] = {}
    rms: dict[str, dict[str, float]] = {}
    for key in sorted(artifacts[0]["tolerance"]):
        decoded[key] = {}
        rms[key] = {}
        for a in artifacts:
            entry = a["tolerance"].get(key)
            if entry is None:
                continue
            values = _decode_f64(base64.b64decode(entry["b64"]))
            decoded[key][a["leg"]] = values
            rms[key][a["leg"]] = _rms(values)

    active = sorted(k for k, per_leg in rms.items() if any(v > 0.0 for v in per_leg.values()))
    inactive = sorted(set(rms) - set(active))
    for key in mission.get("require_active", ()):
        if key not in rms:
            violations.append(
                f"{name}: required tolerance-class channel {key} is absent from the log; "
                f"the gate cannot observe the surface it was widened to cover"
            )
        elif key not in active:
            violations.append(
                f"{name}: required channel {key} is identically zero on every leg, so it "
                f"compares equal for free and gates nothing"
            )
    if len(active) < mission["min_active"]:
        violations.append(
            f"{name}: only {len(active)} tolerance-class channels carry a nonzero RMS, "
            f"below the declared floor of {mission['min_active']}; the comparison has "
            f"lost coverage rather than gained agreement"
        )

    max_rel = 0.0
    worst_channel = ""
    worst_pair = ""
    for key in sorted(decoded):
        legs = sorted(decoded[key])
        for i, j in combinations(range(len(legs)), 2):
            leg_a, leg_b = legs[i], legs[j]
            a_values, b_values = decoded[key][leg_a], decoded[key][leg_b]
            if len(a_values) != len(b_values):
                violations.append(
                    f"{name}: channel {key} has {len(a_values)} elements on {leg_a} and "
                    f"{len(b_values)} on {leg_b}; record counts diverged"
                )
                continue
            diff = _max_abs_diff(a_values, b_values)
            if diff == 0.0:
                continue
            scale = min(rms[key][leg_a], rms[key][leg_b])
            if scale == 0.0:
                violations.append(
                    f"{name}: channel {key} differs by {diff:.3e} between {leg_a} and "
                    f"{leg_b} while one leg is identically zero; the relative scale is "
                    f"undefined and the difference is not a rounding artifact"
                )
                continue
            rel = diff / scale
            if rel > max_rel:
                max_rel = rel
                worst_channel = key
                worst_pair = f"{leg_a} vs {leg_b}"

    return {
        "mission": name,
        "max_rel": max_rel,
        "worst_channel": worst_channel,
        "worst_pair": worst_pair,
        "exact_channels": len(artifacts[0]["exact"]),
        "tolerance_channels": len(rms),
        "active_channels": len(active),
        "inactive_channels": inactive,
        "violations": violations,
    }


def cmd_measure_channels(args: argparse.Namespace) -> int:
    root = Path(args.dir)
    files = sorted(root.rglob("channels-*.json")) if root.is_dir() else []
    if not files:
        return _fail(f"no channels-*.json found under {root}")
    by_mission: dict[str, list[dict]] = {}
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return _fail(f"unparseable channel artifact {path}: {exc}")
        for key in ("leg", "mission", "header_sha256", "exact", "tolerance"):
            if key not in data:
                return _fail(f"{path}: missing required field {key!r}")
        by_mission.setdefault(str(data["mission"]), []).append(data)

    missing = sorted(set(CHANNEL_MISSIONS) - set(by_mission))
    if missing:
        return _fail(
            f"no leg produced artifacts for declared mission(s) {missing}; a mission that "
            f"silently stops being compared is the failure mode this gate exists to prevent"
        )
    unexpected = sorted(set(by_mission) - set(CHANNEL_MISSIONS))
    if unexpected:
        return _fail(f"artifacts for undeclared mission(s) {unexpected}")

    results = []
    for name in sorted(by_mission):
        artifacts = by_mission[name]
        legs = [a["leg"] for a in artifacts]
        if len(artifacts) != args.expect_legs:
            return _fail(
                f"mission {name}: expected {args.expect_legs} leg artifacts, found "
                f"{len(artifacts)}: {legs} (a missing leg would silently shrink the comparison)"
            )
        if len(set(legs)) != len(legs):
            return _fail(f"mission {name}: duplicate leg identifiers: {legs}")
        try:
            results.append(compare_mission(artifacts))
        except ValueError as exc:
            return _fail(f"mission {name}: {exc}")

    violations = [v for r in results for v in r["violations"]]
    max_rel = max(r["max_rel"] for r in results)
    worst = max(results, key=lambda r: r["max_rel"])
    payload = {
        "schema": 1,
        "expect_legs": args.expect_legs,
        "tolerance_rel": args.tolerance,
        "max_rel": max_rel,
        "worst_mission": worst["mission"],
        "worst_channel": worst["worst_channel"],
        "worst_pair": worst["worst_pair"],
        "tolerance_derivation": {
            "measured_worst_rel": MEASURED_WORST_REL,
            "d10_bound_rel": D10_BOUND_REL,
            "rule": "geometric mean of the measured worst case and the D-10 bound",
        },
        "violations": violations,
        "missions": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for r in results:
        print(
            f"{r['mission']}: {r['exact_channels']} exact-class channels bit-compared, "
            f"{r['active_channels']}/{r['tolerance_channels']} tolerance-class channels "
            f"active; max_rel {r['max_rel']:.3e}"
            + (f" on {r['worst_channel']} ({r['worst_pair']})" if r["worst_channel"] else "")
        )
        if r["inactive_channels"]:
            print(f"  identically zero on every leg (gate nothing): {', '.join(r['inactive_channels'])}")
    for v in violations:
        print(f"VIOLATION: {v}")
    print(
        f"worst channel divergence: {max_rel:.3e} on {worst['mission']}"
        + (f".{worst['worst_channel']}" if worst["worst_channel"] else "")
    )
    # Machine-readable lines consumed into $GITHUB_OUTPUT by the workflow.
    print(f"channel_max_rel={max_rel:.3e}")
    ok = not violations and max_rel <= args.tolerance
    print(f"channel_status_state={'success' if ok else 'failure'}")
    return 0


def cmd_gate_channels(args: argparse.Namespace) -> int:
    try:
        measurement = json.loads(Path(args.measurement).read_text(encoding="utf-8"))
        measured = float(measurement["max_rel"])
        violations = list(measurement.get("violations", ()))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return _fail(f"cannot read measurement {args.measurement}: {exc}")

    failures: list[str] = []
    for v in violations:
        failures.append(v)
    if measured > args.tolerance:
        failures.append(
            f"measured channel max_rel {measured:.3e} exceeds the derived tolerance "
            f"{args.tolerance:.4e} (worst: {measurement.get('worst_mission', '?')}."
            f"{measurement.get('worst_channel', '?')}, {measurement.get('worst_pair', '?')}). "
            f"This is a real divergence on a libm-bearing channel, not a bookkeeping "
            f"failure: attribute it before widening the tolerance, and note that the "
            f"tolerance is derived in scripts/cross_platform_divergence.py from the "
            f"measurement recorded in docs/ci/phase6_crossplatform.md"
        )

    record_path = Path(args.record)
    record_value: float | None = None
    if not record_path.is_file():
        failures.append(f"committed record {record_path} is missing")
    else:
        import tomllib

        try:
            with open(record_path, "rb") as fh:
                record = tomllib.load(fh)["channels"]
        except (tomllib.TOMLDecodeError, KeyError) as exc:
            record = None
            failures.append(
                f"committed record {record_path} has no parseable [channels] table: {exc}"
            )
        if record is not None:
            status = record.get("status")
            if status == "pending-first-measurement":
                failures.append(
                    "the committed [channels] record is pending-first-measurement: this "
                    "first run is the designed bootstrap failure, matching the [record] "
                    "table's own procedure. Read the measured value from the "
                    "determinism/cross-platform-channels commit status and complete "
                    f"{record_path} per the procedure in its comments."
                )
            elif status != "measured":
                failures.append(f"[channels] record status {status!r} is not 'measured'")
            elif "measured_max_rel" not in record:
                failures.append("[channels] status is 'measured' but measured_max_rel is absent")
            else:
                record_value = float(record["measured_max_rel"])
                if record_value > args.tolerance:
                    failures.append(
                        f"committed [channels] measured_max_rel {record_value:.3e} exceeds "
                        f"the derived tolerance {args.tolerance:.4e}"
                    )
            tolerance_rel = record.get("tolerance_rel")
            if tolerance_rel is not None and float(tolerance_rel) != args.tolerance:
                failures.append(
                    f"[channels] tolerance_rel {float(tolerance_rel):.4e} disagrees with the "
                    f"enforced tolerance {args.tolerance:.4e}; a revision of the derivation "
                    f"must update both homes in the same change"
                )
            recorded_missions = record.get("missions")
            if recorded_missions is not None and sorted(recorded_missions) != sorted(
                CHANNEL_MISSIONS
            ):
                failures.append(
                    f"[channels] record lists missions {sorted(recorded_missions)} but the "
                    f"gate compares {sorted(CHANNEL_MISSIONS)}; adding or dropping a mission "
                    f"must update both homes in the same change"
                )

    if failures:
        for f in failures:
            print(f"cross_platform_divergence: channel gate failure: {f}", file=sys.stderr)
        return 1
    print(
        f"channel gate passed: measured max_rel {measured:.3e} and committed record "
        f"{record_value:.3e} are both within the derived tolerance {args.tolerance:.4e} "
        f"(= sqrt({MEASURED_WORST_REL:.3e} * {D10_BOUND_REL:.1e}), 3.07x above the worst "
        f"measured divergence and 3.07x below the D-10 ceiling)"
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cross_platform_divergence.py",
        description="Criterion-8 cross-platform divergence tooling (see module docstring).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract", help="write finalstate.json from a run.srlog")
    p_extract.add_argument("--srlog", required=True, help="path to run.srlog")
    p_extract.add_argument("--leg", required=True, help="CI leg identifier (matrix.os)")
    p_extract.add_argument("--out", required=True, help="output finalstate.json path")

    p_measure = sub.add_parser("measure", help="aggregate leg artifacts and measure")
    p_measure.add_argument(
        "--dir", required=True, help="directory searched recursively for finalstate.json"
    )
    p_measure.add_argument("--expect-legs", type=int, required=True)
    p_measure.add_argument("--bound", type=float, default=1e-9)
    p_measure.add_argument("--out", required=True, help="output measurement.json path")

    p_gate = sub.add_parser("gate", help="enforce criterion 8 against the committed record")
    p_gate.add_argument("--measurement", required=True, help="measurement.json from measure")
    p_gate.add_argument("--record", required=True, help="committed cross_platform.toml record")
    p_gate.add_argument("--bound", type=float, default=1e-9)

    p_xc = sub.add_parser(
        "extract-channels", help="write channels-<mission>.json from a run.srlog"
    )
    p_xc.add_argument("--srlog", required=True, help="path to run.srlog")
    p_xc.add_argument("--mission", required=True, choices=sorted(CHANNEL_MISSIONS))
    p_xc.add_argument("--leg", required=True, help="CI leg identifier (matrix.os)")
    p_xc.add_argument("--out", required=True, help="output channels-<mission>.json path")

    p_mc = sub.add_parser(
        "measure-channels", help="aggregate per-leg channel artifacts and measure"
    )
    p_mc.add_argument(
        "--dir", required=True, help="directory searched recursively for channels-*.json"
    )
    p_mc.add_argument("--expect-legs", type=int, required=True)
    p_mc.add_argument("--tolerance", type=float, default=CHANNEL_TOLERANCE_REL)
    p_mc.add_argument("--out", required=True, help="output channel-measurement.json path")

    p_gc = sub.add_parser(
        "gate-channels", help="enforce the channel tolerance against the committed record"
    )
    p_gc.add_argument("--measurement", required=True, help="measurement json from measure-channels")
    p_gc.add_argument("--record", required=True, help="committed cross_platform.toml record")
    p_gc.add_argument("--tolerance", type=float, default=CHANNEL_TOLERANCE_REL)

    args = parser.parse_args(argv)
    if args.command == "extract":
        return cmd_extract(args)
    if args.command == "measure":
        return cmd_measure(args)
    if args.command == "gate":
        return cmd_gate(args)
    if args.command == "extract-channels":
        return cmd_extract_channels(args)
    if args.command == "measure-channels":
        return cmd_measure_channels(args)
    return cmd_gate_channels(args)


if __name__ == "__main__":
    sys.exit(main())
