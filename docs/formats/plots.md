# Quicklook plots and their feeding arrays

Normative reference for `star plot` (PRD FR-18), implemented in
`python/star_reacher/plotting.py`. Like the derived-elements document, this
is a data-out convention page, not a math-library chapter: plotting adds no
physical model — it is a pure post-hoc reduction of logged channels plus the
already-documented loader derivations.

The module is split into two strictly separated layers:

1. **Array preparation** — pure functions from a loaded `Run` to the named
   NumPy "plot-feeding arrays" tabulated below. This layer never imports
   matplotlib and is what the committed golden vectors in
   `tests/golden/plots/` gate (Phase 5 exit criterion 1: data-level
   regression).
2. **Rendering** — a thin matplotlib layer that turns prepared arrays into
   PNGs. matplotlib is imported lazily and the **Agg backend is forced
   before pyplot import**, so `star plot` is headless-safe by construction
   (CI runners, a display-less Pi 5) and never touches a display server.

## 1. CLI

```
star plot <run.srlog> [<more.srlog> ...] [-o OUTDIR] [--plots name,name,...]
```

- Default output directory: `<first input's directory>/plots`.
- `--plots` selects a subset of the names below; an unknown name exits 2
  naming the valid vocabulary.
- Output filenames are stable: `<plot name>.png`, one PNG per named plot.
- A plot no given log can feed is **skipped with a `note:` line on stdout
  and exit code 0** (graceful degradation); exit 1 is reserved for genuinely
  unreadable input (missing file, corrupt SRLOG, missing compiled core) and
  exit 2 for usage errors.

## 2. The named plot set and its feeding arrays

| Plot (`<name>.png`) | Feeding arrays | Source channels |
|---|---|---|
| `groundtrack` | `t_s`, `lon_deg` in [−180, 180), `lat_deg`; event markers `ev_t_s`, `ev_lon_deg`, `ev_lat_deg` | `truth.r_m` via the exact core GCRF→ITRF chain; events |
| `altitude_speed` | `alt_t_s`, `alt_m`, `speed_t_s`, `speed_mps` | `env.alt_m` when logged, else derived (section 3); `truth.v_mps` norm |
| `elements` | `t_s`, `a_km`, `e`, `i_deg`, `raan_deg`, `argp_deg`, `nu_deg` | loader osculating elements (`docs/formats/derived_elements.md`), rescaled to display units |
| `attitude_rates` | `t_s`, `qw`, `qx`, `qy`, `qz`, `wx_dps`, `wy_dps`, `wz_dps` | `truth.q_i2b`, `truth.w_b_radps` (deg/s) |
| `mass_thrust_throttle` | `mass_t_s`, `mass_kg`; `thrust_t_s`, `thrust_n`; `throttle_t_s`, `throttle` | `mass.mass_kg` (fallback `truth.mass_kg`); &#124;`forces.f_thrust_b_n`&#124;; a `throttle` channel in any group |
| `qbar_mach` | `t_s`, `q_pa`, `mach` | `env.q_pa`, `env.mach` |
| `forces_by_source` | `t_s`, `<src>_force_n`, `<src>_torque_nm` per source | &#124;`forces.f_<src>_b_n`&#124;, &#124;`forces.tq_<src>_b_nm`&#124;, sources from the channel dictionary in declaration order |

Partial-content rules (each announced by a `note:`):

- `mass_thrust_throttle`: SRLOG v1.1 defines no throttle channel
  (`docs/formats/srlog_v1.md` section 3.1), so the throttle panel renders
  only when a future additive-minor-version log carries a `throttle`
  channel; the thrust panel requires a forces group with a `thrust` source.
- `qbar_mach` and `forces_by_source` require the optional `env` / `forces`
  groups (absent, for example, from the two-body reference log).
- `groundtrack` is implemented for `central_body = "earth"` only in
  Phase 5: the coastline asset and the exact ITRF chain are Earth's, and a
  lunar principal-axis track would additionally need ephemeris libration
  angles the log alone does not carry.

## 3. Conventions and their reasons

- **Groundtrack frame (exact, not display-grade).** Positions rotate
  GCRF→ITRF with the core's `gcrf_to_itrf` (IAU 2006/2000B chain, dUT1 = 0
  per FR-3) at TAI = header epoch + `t_s` — the same Earth-fixed frame the
  propagation itself used — rather than the viewer's display-grade ERA
  approximation. Latitude is **WGS-84 geodetic** via Bowring's closed form
  with the same fixed two-pass refinement as the core's
  `geodetic_altitude` (Bowring, Survey Review 23(181), 1976;
  `cpp/src/models/atmosphere_hp.cpp`, eq:hp:geodetic); the NumPy
  transcription is pinned to the core binding by
  `tests/python/test_plot_golden.py`. Longitudes wrap to [−180, 180) and
  the rendered track breaks at antimeridian crossings.
- **Altitude source order.** The logged `env.alt_m` when present (the
  model's own geodetic altitude, preferred for model scrutiny); otherwise,
  for Earth, the same exact-frame geodetic derivation as the groundtrack;
  otherwise geocentric `|r| − R_body` with the IAU 2015 radii (Archinal et
  al., Celest. Mech. Dyn. Astron. 130:22, 2018) as a labeled display
  convention.
- **Attitude representation.** The logged quaternion components (Hamilton,
  scalar-first, D-7) are plotted directly: the logged representation is
  continuous and singularity-free, which any Euler-angle view is not.
  Rates are body-frame `w_b_radps` in deg/s.
- **Force/torque form.** Per-source **magnitudes on a log-scaled axis**
  (two stacked panels: forces, torques), the standard perturbation-budget
  view (cf. Montenbruck & Gill, *Satellite Orbits*, Sect. 3.1). Sources
  oppose in sign, so a stacked signed area would misrepresent
  cancellation, while log magnitudes keep a 1e-9-relative perturbation and
  the dominant force readable on one axis — the form that makes model
  scrutiny easiest (the PRD calls forces "the model-scrutiny channel").
  Exactly-zero samples have no log representation and are masked; an
  all-zero series is omitted from its panel; the axis floor sits 12 decades
  below the panel's dominant source.
- **Event markers.** Every time-axis panel gets one vertical line per
  event; label text (the event `detail` string) is drawn on the top panel
  of each figure in single-run mode. The groundtrack marks events at the
  nearest truth sample. In overlay mode each run's markers use that run's
  linestyle and labels are suppressed (they would collide unreadably).
- **Colors and overlays.** Colors come from the Okabe–Ito colorblind-safe
  palette (Okabe & Ito 2008; Wong, *Nature Methods* 8:441, 2011) and are
  assigned **fixed per entity, never cycled**: each canonical force source
  has a permanent color, quaternion/rate components have fixed colors, and
  single-channel panels color by run. Run identity within an axes is
  carried by linestyle (solid, dashed, dash-dot, dotted by command-line
  position). Overlay series are labeled `<channel> (<hash>)` where
  `<hash>` is the first 12 hex digits of the log header's
  `config_sha256` — the resolved-config hash that binds a run to its exact
  inputs (48 bits: far beyond collision range for any practical overlay
  set). `star plot a.srlog b.srlog` overlays shared channels on one axes
  set per plot; a plot renders when **any** given run can feed it.
- **Deterministic output.** Fixed figure geometry (per-plot figsize,
  100 dpi), fixed PNG metadata (no matplotlib version string, no
  timestamps): the PNG bytes are a pure function of the log bytes and the
  package — the FR-21 discipline applied to a derived artifact, asserted by
  `star verify` check V021 and the render tests.

## 4. Golden regression (Phase 5 exit criterion 1)

`tests/golden/plots/` freezes every feeding array of both reference runs
(`missions/twobody_leo.toml`, `missions/ascent_leo.toml`) at fixed probe
indices, bound to each run's resolved-config SHA-256, plus the full event
stream. `tests/python/test_plot_golden.py` re-runs each mission and
compares under per-array rules recorded in the golden files:

| Rule | Meaning |
|---|---|
| `reltol` | `max(abs(got − ref)) ≤ rtol · scale`, `scale` = committed max probe magnitude; an all-zero array must stay exactly zero |
| `abs` | absolute degrees, for bounded angles (`i_deg`, `lat_deg`) |
| `circular_deg` | wrap-safe angular difference (mod 360), for `raan/argp/nu/lon` |
| events | bit-exact times (fixed-step grid arithmetic), exact codes/details/counts |

Tolerance tiers (full derivations in `tests/golden/plots/manifest.toml`):
`rtol = 1e-9` / `1e-6 deg` for the two-body run, whose basic-IEEE-op
propagation has a committed cross-platform divergence of 0.0
(`tests/golden/determinism/cross_platform.toml`); `rtol = 1e-6` /
`1e-3 deg` for the ascent run, whose libm-bearing models (USSA76, frames
series) carry an estimated ≤ ~1e-8 relative cross-platform state spread —
each tier keeps ≥ 2 orders of margin over its platform-noise estimate while
failing on any model or pipeline change at the ppb/ppm level. Goldens
change only by rerunning `tests/golden/plots/generate.py` and committing
the diff with a manifest update (`tests/golden/README.md` update policy).

## 5. Validation

- `tests/python/test_plot_golden.py` — data-level regression of every
  feeding array on both reference runs; the geodetic-transcription pin;
  the golden-manifest lint.
- `tests/python/test_plot_render.py` — headless PNG production for both
  missions, overlay mode and labeling, degradation notes and exit codes,
  byte-identical re-rendering.
- `star verify` check V021 — closed-form circular-orbit element
  preparation plus a headless Agg render with byte-identical repetition,
  self-contained on a bare wheel.
