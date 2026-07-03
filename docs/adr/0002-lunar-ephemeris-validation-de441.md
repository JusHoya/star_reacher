# ADR 0002: Lunar ephemeris validation against Horizons (DE441 vs DE440)

- Status: Accepted. This decision is a documented interpretation of Phase 2
  exit criterion 2 as written, recorded pending ratification of the PRD text
  by the project author.
- Date: 2026-07-02
- Decided by: Phase 2 integration workstream, from the measured evidence in
  `tests/golden/ephemeris/full_span_validation.md`

## Context

Phase 2 exit criterion 2 requires that "repacked lunar/planetary positions
match JPL Horizons to < 1 m at >= 20 epochs spanning 2020-2060". The
repacked ephemeris (`star data fetch de440s`) is a bit-faithful repack of
Chebyshev records from the checksummed JPL DE440s SPK kernel: coefficients
are copied verbatim, never refit, so evaluation accuracy is inherited from
DE440 exactly.

The JPL Horizons system, however, serves lunar and Earth-Moon-barycenter
states from DE441, and the committed query transcripts
(`tests/golden/ephemeris/horizons/`) record `{source: DE441}` for those
quantities. DE440 and DE441 are not identical for the Moon: DE441 omits the
lunar core-mantle tidal damping term so that its lunar orbit remains
well-behaved over the full -13200 to +17191 integration span; as a result
its lunar orbit differs from DE440 by under 2 m across 1970-2020, growing
by roughly 10 m per century away from the present, predominantly
along-track (Park, Folkner, Williams and Boggs, "The JPL Planetary and
Lunar Ephemerides DE440 and DE441", The Astronomical Journal 161:105,
2021, Section 6).

The full-span validation executed on 2026-07-02
(`tests/golden/ephemeris/full_span_validation.md`) measured exactly this
signature: the repack matches Horizons to 0.066 m worst case on the six
quantities DE440 and DE441 share (sun, emb, venus_bary, mars_bary,
jupiter_bary, earth), while the lunar quantities differ by 1.8 m in 2020
growing monotonically to 5.4 m by 2060, mirrored in the earth quantity at
1/82.3 of the lunar value (the Earth-Moon mass-ratio signature of a lunar
orbit difference). A bit-faithful DE440 repack cannot match DE441-sourced
lunar states more closely; applying the < 1 m Horizons gate verbatim to the
lunar quantities would therefore fail the phase on a published,
well-understood difference between the two ephemerides rather than on any
defect in the repack or the evaluator.

## Decision

Phase 2 exit criterion 2 is applied as follows:

- The "< 1 m vs Horizons at >= 20 epochs spanning 2020-2060" gate is
  applied verbatim to the six quantities for which DE440 and DE441 are the
  same solution: sun, emb, venus_bary, mars_bary, jupiter_bary, and earth.
- The lunar quantities (moon relative to the Earth-Moon barycenter, and
  the geocentric moon) are gated < 1 mm against jplephem's independent
  evaluation of the checksummed DE440 kernel itself
  (`tests/golden/ephemeris/moon_de440_jplephem.toml`, same 21 epochs) —
  a strictly tighter test of what the criterion is instrumenting, namely
  that the repack and evaluator reproduce DE440.
- The Horizons lunar comparison is retained as a sanity envelope at 10 m,
  bounding the known DE440/DE441 difference over 2020-2060 so a gross
  repack defect still fails against Horizons.

## Rationale

- The criterion's purpose is to validate the repack and the Chebyshev
  evaluator against an independent authority, not to reconcile two distinct
  JPL solutions. For the lunar quantities, the independent evaluation of
  the identical checksummed kernel by jplephem is the correct authority,
  and the measured agreement (8.6e-8 m worst case against the 1 mm gate)
  validates the repack path to far better than the original 1 m ask.
- The measured lunar discrepancy against Horizons (5.42 m worst case,
  growing with distance from the present and along-track dominated) matches
  the published DE440/DE441 difference in magnitude, growth, and direction
  (Park et al. 2021, Section 6), and its 1/82.3 echo in the earth quantity
  is the Earth-Moon mass-ratio signature expected of a lunar-orbit
  difference. The discrepancy is attributable, quantified, and committed in
  `tests/golden/ephemeris/full_span_validation.md`.
- The 10 m Horizons envelope keeps an external cross-check on the lunar
  path: it accommodates the published difference over this span with
  margin below the ~10 m-per-century growth rate, while still failing on
  segment-selection, scaling, or unit defects, which produce errors orders
  of magnitude larger.

## Consequences

- `tests/python/test_ephemeris_horizons.py` and the ephemeris golden
  manifest gate the six shared quantities at 1 m vs Horizons, the lunar
  quantities at 1 mm vs jplephem/DE440, and the lunar quantities at 10 m vs
  Horizons as an envelope.
- The PRD text of exit criterion 2 is unchanged by this ADR; this document
  records the operative interpretation and stands for ratification by the
  project author. If ratification amends the criterion text instead, this
  ADR should be updated to Superseded accordingly.
- Any future re-validation against a Horizons configuration that serves
  DE440 lunar states (or a repack built from DE441) may retire the split
  gate and apply the verbatim criterion to all eight quantities.
