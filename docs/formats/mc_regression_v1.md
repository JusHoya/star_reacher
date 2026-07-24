# Monte Carlo regression gate and golden (v1)

The Monte Carlo regression layer (FR-22 layer 6, Phase 7 exit criterion 2)
freezes the statistics of a seeded `star mc` ensemble as a golden and gates a
re-run's statistics against it with two complementary 99 % tests, so a change
to the physics or the numerics that moves the outcome distribution is caught
while a bit-identical re-run passes. This document specifies the ensemble, the
metric, the golden file, the two gates, and the two-key update policy.

The exit-criterion-2 clause it satisfies:

> MC ensemble statistics match frozen goldens within chi-square/Anderson-Darling
> 99 % bounds; goldens regenerate only through the two-key path (CI rejects
> otherwise).

## The ensemble and the outcome metric

The reference ensemble is `missions/mc_regression_sweep.toml`: a 128-run Latin
hypercube (`star_reacher.sweep`, deterministic in `master_seed`) that disperses
the initial in-plane (+Y) velocity of the committed EGM2008 8x8 LEO mission
`missions/leo_gravity_8x8.toml` across [6850, 6950] m/s, each run flying a
distinct bound orbit over a single ~90 min window. It runs on committed gravity
data with no fetched ephemeris, in about a second on 8 workers.

The per-run **outcome metric** is the final osculating specific mechanical
energy of the truth trajectory,

    E = |v|^2 / 2 - GM / |r|   [m^2/s^2]

read as `run.elements("truth")["energy_m2ps2"][-1]` through the SRLOG loader
(the osculating set is derived in the loader, not logged; FR-16). It is a
physically meaningful, conservative quantity, monotone in the dispersed initial
velocity, so the ensemble carries a genuine spread to test. Because the ensemble
is bit-reproducible (same master seed → same per-run seeds → same logged bytes →
same metric), the metric vector, and hence every statistic below, is an exact
function of the frozen seed.

## The golden file

`tests/golden/mc_regression/energy_stats.toml` freezes the ensemble size and the
metric's first two moments:

```toml
metric = "energy_m2ps2"
mission = "leo_gravity_8x8.toml"
n = 128
mean_hex = "-0x1.b5003aedbc07ap+24"   # metric mean, exact binary64
std_hex = "0x1.857d1c2e64849p+17"     # sample std (ddof=1), exact binary64
mean_readable = -28639290.928650357    # decimal echo, never read back
std_readable = 199418.22016579125      # decimal echo, never read back
```

`mean`/`std` ride as `float.hex()` literals because the statistics are exact, not
rounded (the same discipline the rng/box_muller golden uses). The `*_readable`
decimals are for the eye only; `star_reacher.mc_regression.load_golden_stats`
reads only the hex fields. Provenance, citation, tolerance, and the
`values_sha256` hash-gate live in the directory's `manifest.toml`.

## The two gates (both at 99 %)

`star_reacher.mc_regression.regression_gate(metric, golden)` standardizes the
re-run's metric by the golden, `z_i = (x_i - mu_g) / sigma_g`, and applies both:

### chi-square (scale)

Under the regression hypothesis that the re-run's metric has the golden mean and
variance, `S = sum_i z_i^2` is chi-square(n) distributed. It is checked against
the two-sided 99 % interval `[chi2_ppf(0.005, n), chi2_ppf(0.995, n)]` (the same
two-sided construction `star_reacher.consistency` gates NEES on, via the
first-principles `star_reacher.chi2`). A variance inflation drives `S` above the
upper bound; a collapse drives it below the lower.

### Anderson-Darling (shape and location)

The standardized metric is Anderson-Darling tested against the standard normal
CDF — i.e. the reference `N(mu_g, sigma_g)` — using `star_reacher.anderson`
(the A2 statistic of Anderson & Darling 1954 with the finite-n distribution of
Marsaglia & Marsaglia 2004, SciPy-free per D-12). The gate passes when the
p-value is at least 0.01. A-D weights the tails, so it catches a distribution
**shift** the symmetric sum-of-squares chi-square tolerates: a half-sigma mean
shift leaves `S` inside its interval yet fails A-D outright.

`RegressionGate.passed` is the conjunction: an ensemble matches the golden only
if it holds both its spread (chi-square) and its shape/location (A-D).

**Why `N(mu_g, sigma_g)` as the A-D reference.** The regression hypothesis is
that the distribution is unchanged from the golden, and the maximum-entropy
continuous distribution consistent with a frozen mean and variance is the
normal. It is also the distribution the ensemble's standardized metric
empirically follows: on the frozen 128-run ensemble the A-D statistic against
`N(mu_g, sigma_g)` is A2 = 1.408 (p = 0.200), comfortably inside the gate.

### Measured statistics (frozen 128-run ensemble)

| quantity | pass point | +0.5 sigma shift | x1.3 scale | x0.7 scale |
|---|---|---|---|---|
| chi-square S | 127.000 | 159.0 (pass) | 214.6 (**red**) | 62.2 (**red**) |
| chi-square interval | [90.543, 172.957] | — | — | — |
| A-D A2 | 1.408 | 16.40 | 7.59 | 3.08 |
| A-D p-value | 0.200 | 5e-6 (**red**) | 1.8e-4 (**red**) | 0.025 (pass) |

The mutations show each gate goes red on a different failure mode, and the x1.3
inflation trips both at once. The `star verify` V029 check runs a self-contained
64-run analogue (a synthesized J2 field, golden inlined for the bare wheel) and
re-measures the +0.5 sigma and x1.4 mutations on the live ensemble, so a gate
that went insensitive fails the acceptance suite. The pytest mutation battery is
`tests/python/test_mc_regression.py`.

## Two-key golden-update policy

A golden regenerates **only** through the two-key path; CI rejects any other
change.

### Key 1 — `scripts/golden_update.py` (human intent)

The only sanctioned way to change the golden. It re-runs the sweep, recomputes
the statistics, and:

* by default (**dry run**) prints a unified diff summary of the committed vs
  regenerated statistics and exits nonzero if a change is pending, writing
  nothing;
* with `--apply` writes the new value file **and** rewrites the manifest with a
  fresh `date`, `generation` note, and the value file's new `values_sha256`.

The value file and its recorded hash are only ever written together here.

```
python scripts/golden_update.py            # dry run: diff summary, exit 1 if stale
python scripts/golden_update.py --apply     # write value + manifest together
```

### Key 2 — `scripts/check_golden_manifests.py` (mechanical enforcement)

The CI gate. For each golden `[[file]]` that carries a `values_sha256`, it
recomputes the value file's SHA-256 from its bytes on disk and asserts it equals
the recorded hash; it also checks the manifest is well-formed and that no value
file in a hash-gated directory escapes a `[[file]]` entry.

A hand-edit of a golden value changes the file's bytes but not the manifest's
recorded hash, so the recomputed SHA-256 disagrees and CI fails. A regeneration
through `--apply` updates both consistently and CI passes. Thus "CI rejects
golden changes without manifest updates".

```
python scripts/check_golden_manifests.py            # all golden dirs
python scripts/check_golden_manifests.py --dir mc_regression
```

The gate is opt-in and additive: a `[[file]]` entry without a `values_sha256` is
reported as "not under hash-gate" and never fails, so the Phase 1-6 manifests
that predate the field are untouched. The `mc_regression` directory is required
to be under the gate (a missing hash there is a failure), and `tests/golden/mc/`
opts in as well. The `values_sha256` field is an additive extension of the
golden manifest schema documented in `tests/golden/README.md`.

## Public API

`star_reacher.mc_regression`:

| name | purpose |
|---|---|
| `ensemble_metric(manifest, manifest_dir)` | the per-run outcome metric of a completed `star mc` ensemble |
| `summarize_metric(metric, mission=...)` | reduce a metric array to the frozen `GoldenStats` (n, mean, std) |
| `regression_gate(metric, golden, prob=0.99)` | the two-part gate; returns a `RegressionGate` |
| `load_golden_stats(path)` / `format_golden_toml(stats)` | read/write the golden value file |
| `GOLDEN_METRIC`, `GOLDEN_VALUE_FILE`, `REGRESSION_PROB` | the metric name, value-file name, and gate probability |

`star_reacher.anderson`:

| name | purpose |
|---|---|
| `anderson_darling(samples, cdf)` | A2 and p-value against a fully specified CDF |
| `anderson_darling_uniform(samples)` | the U(0,1) specialization (PIT path) |
| `adinf(z)`, `errfix(n, z)`, `ad_cdf(a2, n)` | the Marsaglia & Marsaglia (2004) distribution routines |
