"""Regenerate the RNG golden-vector files in this directory.

The values are produced by independent pure-Python reference implementations
of the algorithms the C++ core implements (Phase 1 contract section 4):

- FNV-1a 64-bit: offset basis 14695981039346656037, prime 1099511628211
  (IETF draft-eastlake-fnv).
- SplitMix64: Vigna's reference splitmix64.c (public domain); algorithm from
  Steele, Lea, and Flood, "Fast Splittable Pseudorandom Number Generators",
  OOPSLA 2014. Anchored below to the widely published first output for
  seed 0 (0xE220A8397B1DCDAF).
- PCG64: pcg_setseq_128 XSL-RR 128/64 with the reference multiplier and
  srandom seeding (O'Neill, "PCG: A Family of Simple Fast Space-Efficient
  Statistically Good Algorithms for Random Number Generation",
  HMC-CS-2014-0905; pcg-random.org). Every PCG64 case is cross-checked
  against numpy.random.PCG64.random_raw with the same (state, inc), an
  independent C implementation of the same generator.
- Box-Muller normals: G. E. P. Box and M. E. Muller, "A Note on the
  Generation of Random Normal Deviates", Annals of Mathematical Statistics
  29(2), 1958; draw-consumption pattern per the Phase 1 contract section 4.

Running this script rewrites the four .toml golden files byte-identically;
any diff after regeneration means either the script or the goldens were
edited by hand, which the FR-22 golden-update discipline forbids.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent

U64 = (1 << 64) - 1
U128 = (1 << 128) - 1

# --------------------------------------------------------------------------
# Reference implementations
# --------------------------------------------------------------------------


def fnv1a64(data: bytes) -> int:
    """FNV-1a 64-bit per IETF draft-eastlake-fnv."""
    h = 14695981039346656037  # offset basis 0xCBF29CE484222325
    for byte in data:
        h ^= byte
        h = (h * 1099511628211) & U64  # FNV prime 0x100000001B3
    return h


def splitmix64_sequence(seed: int, n: int) -> list[int]:
    """Vigna's reference splitmix64.c, verbatim constants."""
    state = seed & U64
    out = []
    for _ in range(n):
        state = (state + 0x9E3779B97F4A7C15) & U64
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & U64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & U64
        out.append(z ^ (z >> 31))
    return out


PCG_MULT = 0x2360ED051FC65DA44385DF649FCCF645  # PCG_DEFAULT_MULTIPLIER_128


class Pcg64:
    """pcg_setseq_128 XSL-RR 128/64 with reference srandom seeding."""

    def __init__(self, initstate: int, initseq: int) -> None:
        self.state = 0
        self.inc = ((initseq << 1) | 1) & U128
        self._step()
        self.state = (self.state + initstate) & U128
        self._step()

    def _step(self) -> None:
        self.state = (self.state * PCG_MULT + self.inc) & U128

    def next_u64(self) -> int:
        # Reference generation order: step first, output from post-step state.
        self._step()
        xored = ((self.state >> 64) ^ self.state) & U64
        rot = self.state >> 122
        return ((xored >> rot) | (xored << ((64 - rot) & 63))) & U64


def make_stream(master_seed: int, stream_name: str) -> Pcg64:
    """Stream derivation per the Phase 1 contract section 4."""
    sm = splitmix64_sequence(master_seed ^ fnv1a64(stream_name.encode()), 4)
    initstate = (sm[0] << 64) | sm[1]
    initseq = (sm[2] << 64) | sm[3]
    return Pcg64(initstate, initseq)


def box_muller_sequence(master_seed: int, stream_name: str, n: int) -> list[float]:
    """Trigonometric Box-Muller on the named stream (contract section 4)."""
    gen = make_stream(master_seed, stream_name)
    two_pi = 6.283185307179586476925286766559  # 2*pi rounded to binary64
    out: list[float] = []
    cached: float | None = None
    for _ in range(n):
        if cached is not None:
            out.append(cached)
            cached = None
            continue
        u1 = 1.0 - ((gen.next_u64() >> 11) * 2.0**-53)  # (0,1]: avoids log(0)
        u2 = (gen.next_u64() >> 11) * 2.0**-53  # [0,1)
        radius = math.sqrt(-2.0 * math.log(u1))
        angle = two_pi * u2
        out.append(radius * math.cos(angle))
        cached = radius * math.sin(angle)
    return out


# --------------------------------------------------------------------------
# Anchors and cross-checks against independent references
# --------------------------------------------------------------------------


def crosscheck() -> None:
    # FNV-1a of the empty string is the offset basis by definition.
    assert fnv1a64(b"") == 0xCBF29CE484222325
    # Published first output of Vigna's splitmix64 for seed 0.
    assert splitmix64_sequence(0, 1)[0] == 0xE220A8397B1DCDAF
    # Every PCG64 case must match numpy.random.PCG64 (independent C
    # implementation of the same algorithm) given identical (state, inc).
    for initstate, initseq in [
        (42, 54),
        (0x0123456789ABCDEF_FEDCBA9876543210, 0x0F0E0D0C0B0A0908_0706050403020100),
    ]:
        _crosscheck_numpy(Pcg64(initstate, initseq))
    for master_seed, name in [
        (42, "sensors.imu"),
        (42, "dispersions.mass"),
        (0, "truth"),
        (U64, "sensors.imu"),
    ]:
        _crosscheck_numpy(make_stream(master_seed, name))


def _crosscheck_numpy(gen: Pcg64) -> None:
    bg = np.random.PCG64()
    st = bg.state
    st["state"] = {"state": gen.state, "inc": gen.inc}
    bg.state = st
    expected = [int(x) for x in bg.random_raw(16)]
    got = [gen.next_u64() for _ in range(16)]
    assert got == expected, "PCG64 reference diverges from numpy.random.PCG64"


# --------------------------------------------------------------------------
# TOML emission (restricted subset read by cpp/tests/golden_io.hpp)
# --------------------------------------------------------------------------


def hex64(value: int) -> str:
    return f"0x{value:016X}"


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
    crosscheck()

    emit(
        HERE / "fnv1a.toml",
        "FNV-1a 64-bit golden vectors (IETF draft-eastlake-fnv parameters).\n"
        "Regenerated by generate.py; hashes are hex strings (TOML integers\n"
        "are signed 64-bit and cannot carry u64 values).",
        [
            {"name": name or "empty", "input": name, "hash": hex64(fnv1a64(name.encode()))}
            for name in ["", "a", "foobar", "sensors.imu", "dispersions.mass", "truth"]
        ],
    )

    emit(
        HERE / "splitmix64.toml",
        "SplitMix64 golden vectors (Vigna reference splitmix64.c).\n"
        "16 consecutive outputs per seed; regenerated by generate.py.",
        [
            {"name": name, "seed": hex64(seed), "values": [hex64(v) for v in splitmix64_sequence(seed, 16)]}
            for name, seed in [
                ("seed_0", 0),
                ("seed_42", 42),
                ("seed_golden_gamma", 0x9E3779B97F4A7C15),
                ("seed_max", U64),
            ]
        ],
    )

    pcg_cases: list[dict] = []
    for name, initstate, initseq in [
        ("raw_state42_seq54", 42, 54),
        (
            "raw_full128",
            0x0123456789ABCDEF_FEDCBA9876543210,
            0x0F0E0D0C0B0A0908_0706050403020100,
        ),
    ]:
        gen = Pcg64(initstate, initseq)
        pcg_cases.append(
            {
                "name": name,
                "kind": "raw",
                "initstate_hi": hex64(initstate >> 64),
                "initstate_lo": hex64(initstate & U64),
                "initseq_hi": hex64(initseq >> 64),
                "initseq_lo": hex64(initseq & U64),
                "values": [hex64(gen.next_u64()) for _ in range(16)],
            }
        )
    for name, master_seed, stream in [
        ("stream_seed42_sensors_imu", 42, "sensors.imu"),
        ("stream_seed42_dispersions_mass", 42, "dispersions.mass"),
        ("stream_seed0_truth", 0, "truth"),
        ("stream_seedmax_sensors_imu", U64, "sensors.imu"),
    ]:
        gen = make_stream(master_seed, stream)
        pcg_cases.append(
            {
                "name": name,
                "kind": "stream",
                "master_seed": hex64(master_seed),
                "stream": stream,
                "values": [hex64(gen.next_u64()) for _ in range(16)],
            }
        )
    emit(
        HERE / "pcg64.toml",
        "PCG64 (pcg_setseq_128 XSL-RR 128/64) golden vectors.\n"
        "raw cases seed via the reference srandom(initstate, initseq);\n"
        "stream cases derive (initstate, initseq) from (master_seed, stream)\n"
        "per the Phase 1 contract section 4. All cases are cross-checked\n"
        "against numpy.random.PCG64.random_raw at generation time.\n"
        "Regenerated by generate.py.",
        pcg_cases,
    )

    emit(
        HERE / "box_muller.toml",
        "Box-Muller standard-normal golden vectors on named PCG64 streams.\n"
        "Values are IEEE-754 binary64 hex literals (float.hex()) so the file\n"
        "itself is exact; the consuming test compares against libm-computed\n"
        "values with the tolerance recorded in manifest.toml.\n"
        "Odd-count cases exercise the cached second variate.\n"
        "Regenerated by generate.py.",
        [
            {
                "name": name,
                "master_seed": hex64(master_seed),
                "stream": stream,
                "values": [float(v).hex() for v in box_muller_sequence(master_seed, stream, count)],
            }
            for name, master_seed, stream, count in [
                ("normals_seed42_sensors_imu", 42, "sensors.imu", 16),
                ("normals_seed7_dispersions_mass", 7, "dispersions.mass", 15),
                ("normals_seed0_truth", 0, "truth", 8),
            ]
        ],
    )

    print("golden files regenerated and cross-checked")


if __name__ == "__main__":
    main()
