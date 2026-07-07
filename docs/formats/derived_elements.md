# Loader-derived osculating elements

Normative conventions for the osculating orbital elements
`star_reacher.load(...).elements()` derives (PRD FR-16: "osculating elements
are derived in the loader, not logged"; FR-17), implemented in
`python/star_reacher/derived.py`. This is a data-out convention document,
not a math-library chapter: element derivation adds no physical model to the
simulation — it is a pure post-hoc reduction of logged `r_m`/`v_mps`
samples.

## 1. Inputs

Elements are computed per sample from a channel group's `r_m` and `v_mps`
vectors (the `truth` group by default) and the gravitational parameter of
the log header's `central_body`. The GM table in
`python/star_reacher/derived.py` mirrors the single-home values in
`cpp/include/star/constants.hpp` (Earth: IERS Conventions 2010, TN No. 36,
Table 1.1; Moon and Mars system: DE440, Park et al., AJ 161:105, 2021,
Table 2) and is pinned bit-exactly to the core's `gm()` binding by
`tests/python/test_gm_crosscheck.py`.

## 2. Elements

Per sample, `elements()` returns 1-D float64 arrays:

| Key | Quantity | Units |
|---|---|---|
| `a_m` | semi-major axis | m |
| `e` | eccentricity | – |
| `i_rad` | inclination | rad |
| `raan_rad` | right ascension of the ascending node | rad |
| `argp_rad` | argument of periapsis | rad |
| `nu_rad` | true anomaly | rad |
| `energy_m2ps2` | specific orbital energy `v²/2 − μ/r` | m²/s² |
| `hmag_m2ps` | specific angular momentum magnitude `|r × v|` | m²/s |

Formulation: Vallado, *Fundamentals of Astrodynamics and Applications*,
4th ed., Algorithm 9 (RV2COE), with every angle recovered through `atan2`
of in-plane projections rather than `arccos`, so precision does not
collapse near 0 or π.

## 3. Angle ranges

- `i_rad` ∈ [0, π];
- `raan_rad`, `argp_rad`, `nu_rad` ∈ [0, 2π).

## 4. Singular-geometry conventions

The classical elements degenerate for circular and/or equatorial orbits;
the standard alternate-element conventions apply (Vallado Alg. 9), with
tolerances `e < 1e-11` (circular) and `sin i < 1e-11` (equatorial):

| Geometry | Convention |
|---|---|
| circular, inclined | `argp_rad` is exactly 0; `nu_rad` carries the **argument of latitude** (node → position) |
| elliptical, equatorial | `raan_rad` is exactly 0; `argp_rad` carries the **longitude of periapsis** (+X → periapsis) |
| circular, equatorial | `raan_rad` and `argp_rad` exactly 0; `nu_rad` carries the **true longitude** (+X → position) |

In-plane angles are measured **in the direction of motion** (the in-plane
basis is completed with `h_hat × reference`), so a retrograde equatorial
orbit reports angles that advance with the motion.

## 5. Conic types

The recovery is conic-agnostic; no branch selects on orbit type:

- **elliptical** — `e < 1`, `a > 0`, negative energy;
- **hyperbolic** (Mars-cruise heliocentric legs, SOI-exit states) —
  `e > 1`, `a < 0`, positive energy; `nu_rad` stays inside the asymptote
  limit `|ν| < arccos(−1/e)`, with incoming (pre-periapsis) samples in
  (π, 2π) under the [0, 2π) convention;
- **parabolic boundary** — at exactly zero specific energy `a_m` is `+inf`
  (pinned explicitly; the raw `−μ/(2·ε)` limit would take the sign of the
  zero); near-parabolic samples produce a large-magnitude `a_m` and
  `e ≈ 1` without loss of angle precision.

## 6. Degenerate states

Samples with `|r| = 0` or `|h| = 0` (rectilinear motion) have no defined
elements: `a_m`, `e`, and all four angles are NaN for those samples, and
`energy_m2ps2` is NaN only when `|r| = 0` (a radial state still has a
well-defined energy). No exception is raised, so one bad sample cannot make
a whole run un-analyzable.

## 7. Caching

`Run.elements(group)` computes lazily on first call and caches per group;
the arrays are derived views of the log, never part of the file
(`docs/formats/srlog_v1.md` defines what is logged).

## 8. Validation

`tests/python/test_derived_elements.py` reconstructs states from known
elements with an independent COE→RV implementation (Vallado Alg. 10) and
gates recovery of every element, every singular-geometry convention above,
the hyperbolic branch, the parabolic boundary, and the NaN behavior.
