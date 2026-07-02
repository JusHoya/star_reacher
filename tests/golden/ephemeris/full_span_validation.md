# Full-span DE440s repack validation against JPL Horizons

Executed 2026-07-02 by `tests/golden/ephemeris/generate.py` on the
maintainer machine (Phase 2 exit criterion 2 evidence). The complete
repack `data/de440s_2020_2060.sreph` (not the committed excerpt) was
evaluated by the Python reference evaluator - the executable
specification of `star::Ephemeris` - at every test epoch and compared
against geometric ICRF state vectors fetched from the JPL Horizons API
(transcripts under `horizons/`).

Sources (SHA-256):

- `de440s.bsp` `c1c7feeab882263fc493a9d5a5b2ddd71b54826cdf65d8d17a76126b260a49f2`
- `moon_pa_de440_200625.bpc` `60cd55aa401ea2ea97360636f567554bfe4e37bb829f901b4460a455dfaf783f`

Epochs per quantity: the repacked span endpoints, 17 TDB midnights
spread across 2020-2060, and two interior Chebyshev record boundaries
of the underlying segment (21 epochs per quantity, all exact TDB
midnights).

| Quantity | Horizons COMMAND / CENTER | Epochs | Max position error [m] | At epoch (TDB) | Gate [m] |
|---|---|---|---|---|---|
| sun | 10 / 500@0 | 21 | 0.000000 | 2060-01-02T00:00:00 TDB | 1 |
| emb | 3 / 500@0 | 21 | 0.000150 | 2060-01-02T00:00:00 TDB | 1 |
| venus_bary | 2 / 500@0 | 21 | 0.000061 | 2040-01-01T00:00:00 TDB | 1 |
| mars_bary | 4 / 500@0 | 21 | 0.000098 | 2033-04-09T00:00:00 TDB | 1 |
| jupiter_bary | 5 / 500@0 | 21 | 0.000244 | 2024-06-10T00:00:00 TDB | 1 |
| earth | 399 / 500@3 | 21 | 0.065823 | 2060-01-02T00:00:00 TDB | 1 |
| moon | 301 / 500@3 | 21 | 5.351424 | 2060-01-02T00:00:00 TDB | 10 |
| moon_geocentric | 301 / 500@399 | 21 | 5.417247 | 2060-01-02T00:00:00 TDB | 10 |

Worst non-lunar case: **0.065823 m** against the 1 m gate.

## Lunar quantities: Horizons serves DE441, not DE440

The committed transcripts record `{source: DE441}` for the Moon and the
Earth-Moon barycenter. DE441 deliberately omits the lunar core-mantle
tidal damping term so it can span -13200 to +17191; its lunar orbit
consequently differs from DE440 by under 2 m across 1970-2020, growing
to roughly 10 m per century away from the present, predominantly
along-track (Park et al. 2021, AJ 161:105, Section 6). The measured
lunar difference above (5.417 m worst case, growing
monotonically from 1.8 m in 2020 to 5.4 m in 2060, and mirrored in the
earth quantity at 1/82.3 of the lunar value - the EMB mass-ratio
signature) is exactly that published DE440/DE441 difference, not a
repack defect. A bit-faithful DE440 repack cannot match DE441-sourced
lunar states more closely.

The authoritative DE440 lunar comparison therefore runs against
jplephem's independent evaluation of the checksummed `de440s.bsp`
(`moon_de440_jplephem.toml`, same 21 epochs):

| Quantity | Epochs | Max position error vs jplephem/DE440 [m] | Gate [m] |
|---|---|---|---|
| moon (w.r.t. EMB) and moon_geocentric | 21 | 0.000000086 | 0.001 |

Lunar librations are validated separately against jplephem's
independent evaluation of the same PCK (`librations_jplephem.toml`);
Horizons does not serve the DE440 Euler angles directly.
