# Phase 6 independent code review

Scope: the `main...phase-6-gnc-sensors` diff, reviewed as code. This is a
defect review — correctness, determinism, and latent traps. It is not an
evidence or acceptance-gate audit; that is covered separately by
`docs/audit/phase6_evidence_audit.md`.

Review base: `phase-6-gnc-sensors` at `ea0e5e9`, worktree branch
`ws-p6-review`. Nothing was compiled during this review; every finding is
either an airtight static read (CONFIRMED) or is marked SUSPECTED with the
experiment that would settle it.

## Headline result: the `auto` + Eigen hunt found nothing

The standing project lesson is that an Eigen expression template bound to
`auto` outlives its temporary operands and reads freed memory — benign on one
compiler, garbage on another, and visible only under a cross-compiler
`-Werror` build. Every line of this phase has been compiled by MSVC release
only; the Linux `-Werror` doctest leg has never run on it. That made this the
highest-value thing to look for.

Every `auto` in the phase's new and changed C++ was enumerated and classified:

| File | `auto` uses | Classification |
| --- | --- | --- |
| `cpp/src/gnc/ekf.cpp` | 2 | `std::map` iterator, range-for over `std::map` |
| `cpp/src/gnc/builtin.cpp` | 4 | map iterators and range-for |
| `cpp/src/gnc/component.cpp` | 4 | map/registry iterators and range-for |
| `cpp/src/vehicle_cycle.cpp` | 4 | map iterator, two RHS lambdas, range-for over `std::unique_ptr` |
| `cpp/src/sensors/imu.cpp` | 5 | map iterators, range-for, one setup lambda |
| `cpp/src/sensors/optical.cpp` | 7 | map iterators and range-for |
| `cpp/src/sensors/radio.cpp` | 6 | map iterators and range-for |
| `cpp/src/sensors/camera.cpp` | 6 | map iterators and range-for |
| `cpp/src/models/environment.cpp` | 0 | — |
| `cpp/src/srlog_writer.cpp` | 0 | — |

**Not one `auto` is bound to an Eigen expression.** Every Eigen intermediate
in `ekf.cpp` — the dense, Eigen-heavy file that was the primary concern — is
bound to a named concrete type (`Eigen::Vector3d`, `Eigen::Matrix3d`,
`Matrix15d`, `Eigen::Matrix<double, M, kM>`, …). This is disciplined code and
the discipline is uniform, not accidental.

Two adjacent shapes were checked and are also clean:

- **Aliasing.** `symmetrize()` (`cpp/src/gnc/ekf.cpp:90`) writes
  `p = 0.5 * (p + p.transpose()).eval()`. The `.eval()` materialises the
  transposed sum into a temporary before the assignment reads it, so the
  transpose aliasing hazard is correctly defused. `joseph_update()`
  (`cpp/src/gnc/ekf.cpp:377-397`) assigns through the named intermediate
  `p_post`, never into `p_` from an expression containing `p_`.
- **References outliving temporaries.** The only two `const Eigen::…&`
  bindings in the phase — `cpp/src/gnc/component.cpp:230` and
  `cpp/src/sensors/camera.cpp:121` — bind to members of a live object, not to
  temporaries.

The pybind11 trampoline deserves specific credit here. Both interface methods
that return a reference to a container — `innovations()` and
`error_layout()` — are the textbook pybind11 dangling-reference footgun, and
both are correctly cached in `mutable` members (`bindings/module.cpp:518-526`,
`548-556`, members at `569-570`) rather than returning a reference to the
caster's temporary.

The `auto` + Eigen hunt is a genuine negative result across every file listed
above.

## Findings

### 1. Unvalidated innovation payload overflows fixed log buffers (CONFIRMED, HIGH)

**Where:** `cpp/src/vehicle_cycle.cpp:1110-1138` (writes), sized at
`cpp/src/vehicle_cycle.cpp:812-820`.

The nav-innovation log buffers are sized once, at GNC activation, from the
component's declared maximum:

```
innov_mm = nav->innov_max_dim();
innov_y_buf.assign(innov_mm, 0.0);
innov_s_buf.assign(innov_mm * (innov_mm + 1) / 2, 0.0);
```

Each cycle, every `InnovationSample` the component returns is copied into
those buffers with **no check that it fits**:

- `cpp/src/vehicle_cycle.cpp:1117` —
  `std::copy(s.y.begin(), s.y.end(), innov_y_buf.begin())` copies `s.y.size()`
  doubles into a buffer of `innov_mm`.
- `cpp/src/vehicle_cycle.cpp:1131` —
  `innov_s_buf[row0 + (j - i)] = s.s_upper[src++]` indexes the destination by
  `m = s.y.size()` and reads the source for `m(m+1)/2` entries.

`m` is taken from `s.y.size()` at `cpp/src/vehicle_cycle.cpp:1116` and is never
compared against `innov_mm`; `s.s_upper.size()` is never compared against
`m(m+1)/2`.

**Reachability.** This is reachable from pure Python with no unsafe API.
`InnovationSample` is exposed with a default constructor and read-write access
to both vectors (`bindings/module.cpp:1248-1257`), and a Python nav component
supplies both `innov_max_dim()` and `innovations()` through the trampoline
(`bindings/module.cpp:512-526`).

**Failure scenario.** A Python nav component that declares
`innov_max_dim() -> 1` but returns an `InnovationSample` with `y` of length 6
and `s_upper` of length 21 causes a write of 6 doubles into a 1-element vector
(40 bytes past the end) and a write into `innov_s_buf` at indices up to 20 in a
1-element vector (160 bytes past the end) — heap corruption on the first cycle
that applies an aiding update. The mirror case, declaring
`innov_max_dim() -> 6` and returning a short `s_upper`, is an out-of-bounds
**read** that silently writes uninitialised heap into the `nav.innov` channel.

**Why this reads as an oversight rather than a design choice.** Every other
variable-length quantity a Python component returns *is* length-checked:
`state()` and `covariance_upper()` go through `copy_fixed()`, which refuses a
wrong length by name (`bindings/module.cpp:454-468`), and `error_layout()` is
validated by `validate_error_layout()`, which requires the declared blocks to
tile `[0, state_dim)` exactly (`cpp/src/gnc/component.cpp:136-183`).
`innovations()` is the one gap in an otherwise complete perimeter.

The built-in `error_state_ekf` is safe — it declares `innov_max_dim() == 6`
(`cpp/src/gnc/ekf.cpp:210`) and its widest update is the 6-dimensional nav fix
— so no shipped configuration triggers this. It is a plugin-boundary defect.

**Remedy.** Validate in the loop, before either copy, that
`s.y.size() <= innov_mm` and `s.s_upper.size() == s.y.size() * (s.y.size() + 1) / 2`,
and throw a named `std::length_error` in the same style as `copy_fixed()`.
Validating in the loop rather than in the trampoline also covers a
hypothetical third-party C++ component registered through the same registry.

### 2. The nav fix reports `valid = true` before it has ever been sampled (CONFIRMED, MEDIUM)

**Where:** `cpp/src/vehicle_cycle.cpp:1009-1015`.

```
in.navfix.valid = true;  // the nav fix carries no gating flag
in.navfix.fresh = navfix_fresh;
in.navfix.r_i_m = navfix->last_position_m();
in.navfix.v_i_mps = navfix->last_velocity_mps();
```

`valid` is hardcoded `true` unconditionally, including on every cycle before
the sensor's first sample instant. The star tracker and altimeter, two lines
below and above, correctly forward the sensor's own flag via `last_valid()`.

`r_meas_` and `v_meas_` are properly zero-initialised
(`cpp/include/star/sensors/radio.hpp:65-66`), so this is not an uninitialised
read — but zero is precisely the "plausible payload" shape: a consumer sees a
nav fix flagged valid that places the vehicle at the centre of the central
body, at rest.

**Failure scenario.** The built-in EKF is protected, because it gates on
`fresh && valid` (`cpp/src/gnc/ekf.cpp:184`) and `fresh` is correctly false
until the first sample. But the same struct is copied verbatim onto the FR-24
observation surface (`cpp/src/vehicle_cycle.cpp:1063`, `obs.navfix = in.navfix`)
and exposed to a stepping driver. A Python nav component or a stepping driver
that checks `valid` — the flag whose entire purpose is to answer "may I trust
this payload" — and not also `fresh` will fold a zero position/velocity fix
into its estimate on every cycle before the first sample. With a 1 Hz nav fix
on a 100 Hz control cycle, that is 100 consecutive spurious updates at the
origin.

**Remedy.** Give `NavFix` a `sampled_` flag set in `sample()` and forward it,
so `valid` means "this payload is real" for all four aiding sensors uniformly.
The inline comment is right that the nav fix has no *gating* flag; it does not
follow that it has no *validity* flag.

### 3. The altimeter update is silently discarded without a body-fixed frame (CONFIRMED, MEDIUM)

**Where:** `cpp/src/gnc/ekf.cpp:496-502`.

```
if (!env.bodyfixed_valid || !(ellipsoid_a_m_ > 0.0)) {
  return;
}
```

The altimeter **sensor** does not skip in this case: without a valid body-fixed
frame it falls back to the closed spherical form `|r| - a` and emits the
measurement flagged valid (`cpp/src/sensors/radio.cpp:189-190`, gate at
`203-205`). The filter takes the opposite branch and drops it.

The early return produces no `InnovationSample`, writes nothing to
`nav.innov`, and sets no flag. The measurement is consumed and discarded with
zero observability.

**Failure scenario.** A run configured with an altimeter on a central body
whose body-fixed frame is unavailable logs a full `sensor.altimeter` channel
with `valid = 1` throughout, and an empty `nav.innov` for that sensor. The
filter behaves as if the altimeter were not configured, and nothing in the log
distinguishes "the altimeter was never fresh" from "every altimeter update was
refused". A NEES gate then fails or passes for a reason the log cannot
explain.

Skipping is defensible — the comment argues it, and folding a measurement in
against the wrong frame would be worse. The defect is that the skip is
invisible.

**Remedy.** Emit an `InnovationSample` with a zero-width or explicitly-flagged
payload for a refused update, or add a counter to the run summary. The
project's own standing lesson on making silent rejections observable applies
directly.

### 4. The ellipsoid is defined twice with incompatible sphere sentinels (CONFIRMED, MEDIUM)

**Where:** `cpp/src/models/environment.cpp:291-311` versus
`cpp/src/vehicle_cycle.cpp:717-721`.

The same reference ellipsoid reaches the sensor and the filter by two
independent paths that disagree on how a sphere is encoded:

- The **sensor** path reads `SensorCycleTruth::geom.ellipsoid_inv_f`, filled by
  `EnvironmentModel::central_ellipsoid()`, which encodes the Moon as
  `inv_f = 0.0` (`cpp/src/models/environment.cpp:301-304`). The altimeter tests
  `spherical = !(geom.ellipsoid_inv_f > 1.0)`
  (`cpp/src/sensors/radio.cpp:187`) and takes the closed spherical branch.
- The **filter** path reads `GncInitContext::ellipsoid_inv_f`, filled from
  `planet_inv_f` (`cpp/src/vehicle_cycle.cpp:963-964`), which encodes the Moon
  as `inv_f = 1.0e12` (`cpp/src/vehicle_cycle.cpp:721`). The EKF has no
  spherical branch and runs the Bowring conversion
  (`cpp/src/gnc/ekf.cpp:509`).

So for a lunar mission the altimeter measures `|r| - R_moon` while the filter
predicts a Bowring geodetic height at `f = 1e-12`.

**This does not currently misbehave**, and I want to be precise about that: I
initially expected a throw, and checked. `geodetic_lat_lon_alt()` rejects
`inv_f <= 1.0` with `std::domain_error` (`cpp/src/models/atmosphere_hp.cpp:157`),
but the `1.0e12` sentinel clears that guard, and at `f = 1e-12` the Bowring
result differs from `|r| - a` by order `a·e²` ≈ 3 µm — negligible against any
configured altimeter noise. Mars (`MARS_ELLIPSOID_INV_F`) and Earth
(`WGS84_INV_F = 298.257223563`) are both well above 1 on both paths. The Sun
has no body-fixed frame, so the EKF returns early and no vehicle can be
configured there anyway (`python/star_reacher/mission.py:2585`).

The finding is therefore **latent, not active**: two sources of truth for one
physical constant, using mutually incompatible conventions for the same
degenerate case (`0.0` means sphere on one path and would mean "invalid,
throw" on the other), with correctness resting on a magic `1.0e12` whose only
documentation is a nine-word trailing comment. Any future edit that
harmonises one path to the other — the obvious cleanup — converts this into an
immediate `std::domain_error` thrown from inside the deterministic time loop
on every lunar altimeter update.

**Remedy.** Have `GncInitContext` take its ellipsoid from
`EnvironmentModel::central_ellipsoid()`, the same call the sensor geometry
uses, and give the EKF the same explicit spherical branch the altimeter has.
One ellipsoid, one sphere convention, one branch test.

### 5. `LDLT::info()` is never checked on the innovation covariance (CONFIRMED, MEDIUM)

**Where:** `cpp/src/gnc/ekf.cpp:387-388`.

```
const Eigen::LDLT<Eigen::Matrix<double, M, M>> ldlt(s);
const Eigen::Matrix<double, M, kM> kt = ldlt.solve(pht.transpose());
```

If the decomposition fails, `solve()` returns a result that is silently wrong
rather than signalling. The gain `k` then propagates NaN or garbage into `dx`,
into the reset, and into `p_` through the Joseph form — after which every
subsequent cycle is contaminated and the covariance logged to `nav.est.P` is
meaningless.

`S = H P Hᵀ + R` is positive definite whenever `R` is, and `R` is built from
configured sigmas. The exposure is therefore a run whose sigmas are zero: the
sensor parsers accept `sigma_rad`, `sigma_r_m`, and `sigma_v_mps` entries that
are `>= 0` and reject only negatives (`cpp/src/sensors/optical.cpp:87-90`,
`cpp/src/sensors/radio.cpp:63-66`), and `NavSensorModel`'s sigma members
default to zero (`cpp/include/star/gnc/component.hpp:139-140, 146-147`). A
noiseless sensor is a configuration a user can plausibly write when
constructing a controlled test case.

I could not determine without running whether `python/star_reacher/mission.py`
independently forbids a zero sigma; that check is in the Python layer's
section below if it was covered there.

**Failure scenario.** Configure `[sensors.startracker]` with
`sigma_rad = [0.0, 0.0, 0.0]`. `R` is exactly zero, `S = H P Hᵀ`, and once the
attitude block of `P` collapses toward zero after repeated zero-noise updates,
`LDLT` on a numerically singular `S` yields a garbage solve that is never
detected. The run completes and writes a log that looks structurally valid.

**Remedy.** Check `ldlt.info() != Eigen::Success` and throw a named error, and
separately require every configured sigma that reaches `R` to be strictly
positive at parse time — this filter's own `require_sigma3()`
(`cpp/src/gnc/ekf.cpp:55-66`) already argues exactly this case for `P0`, with
the reasoning that a zero variance makes NEES undefined rather than merely
large. The same argument applies to `R`.

### 6. An invalid IMU sample freezes the covariance while time advances (CONFIRMED, MEDIUM)

**Where:** `cpp/src/gnc/ekf.cpp:176-178`.

```
if (input.imu_fresh && input.imu.valid && input.imu.dt_s > 0.0) {
  propagate(input.imu);
}
```

When the guard fails, neither the nominal state nor `P` advances — but the
aiding updates below still run, and truth keeps moving. The filter's stated
uncertainty stops growing while its actual error grows.

**Failure scenario.** Any interval during which the IMU is fresh but flagged
invalid produces a filter that is systematically overconfident: `P` is frozen
at its last propagated value while the true error accumulates at the full
unaided drift rate. A subsequent aiding update then computes a gain from a
covariance that understates the error, under-corrects, and drives NEES up —
with no entry anywhere in the log identifying the skipped propagations as the
cause. This is the same overconfidence failure the freshness comment at
`cpp/src/vehicle_cycle.cpp:993-996` is careful to prevent for reprocessing,
reached from the opposite direction.

I did not find a path in the shipped `Imu` that sets `valid = false` after the
first sample — `cpp/src/sensors/imu.cpp:255` sets it true and nothing clears
it — so this is currently unreachable with the built-in IMU. It becomes
reachable with any sensor plugin or future IMU gating.

**Remedy.** On a skipped propagation, still advance `P` by the process-noise
term for the elapsed control period, or refuse the cycle outright. Silently
holding the covariance is the one option that produces a plausible-looking
wrong answer.

### 7. Unguarded division by `r³` in the filter's gravity model (CONFIRMED, LOW)

**Where:** `cpp/src/gnc/ekf.cpp:355-369`.

`gravity()` computes `-mu_ * p / (r*r*r)` and `gravity_gradient()` computes
`u = p / r`, both without a guard on `r == 0`. A position *estimate* at the
origin yields inf/NaN that propagates into `v_hat_`, `p_hat_`, and the `F`
matrix, corrupting the run from that cycle on.

`p0_m` is a required parameter with no positivity constraint
(`cpp/src/gnc/ekf.cpp:124` calls `require_vector`, which checks only presence,
length, and finiteness), so `p0_m = [0, 0, 0]` is accepted and produces NaN on
the first propagation. This is a configuration error rather than a realistic
run, and it fails loudly enough in the output to be diagnosed — hence LOW —
but the project's own `require_sigma3()` sets the precedent for rejecting a
degenerate parameter at construction instead.

**Remedy.** Reject a zero `p0_m` in the constructor, and guard `r` in both
gravity routines.

### 8. Sun sensor emits a normalised pure-noise direction when the ephemeris is absent (CONFIRMED, LOW — previously identified, still present)

**Where:** `cpp/src/sensors/optical.cpp:266-283`.

With `geom.ephemeris_valid == false`, `u_b` stays zero, so `sum = eta` is pure
noise and line 283 normalises it to a unit vector. The sample is correctly
flagged `valid = 0`, but the payload is a perfectly plausible unit direction
rather than something a consumer can recognise as meaningless.

This is recorded as already found during the phase. It is confirmed still
present at `ea0e5e9`. Emitting the zero vector, or NaN, when
`geometry == false` would make the invalidity self-evident in the payload as
well as the flag.

## Areas reviewed and found clean

- **`auto` + Eigen, aliasing, and reference lifetime** across all ten new and
  changed C++ source files — see the table above. Genuine negative result.
- **Quaternion conventions.** Scalar-first construction is correct throughout
  (`cpp/src/gnc/ekf.cpp:122`, `219-220`; `cpp/src/sensors/optical.cpp:107`).
  `quat_normalize()` refuses a zero or non-finite quaternion rather than
  fabricating an attitude (`cpp/src/rotation.cpp:64-74`), which closes the
  degenerate-`q0` hole that would otherwise mirror finding 7.
- **The star tracker forward model and the filter's prediction agree
  exactly.** The sensor composes
  `q_meas = q_ab ⊗ q_true ⊗ dq_n` with `q_ab = quat_exp(-b_I × β)`
  (`cpp/src/sensors/optical.cpp:185-201`); the filter predicts
  `q_pred = q_ab ⊗ q_hat` with the identical construction
  (`cpp/src/gnc/ekf.cpp:464-472`). The innovation
  `dq_y = q_pred⁻¹ ⊗ q_meas` therefore telescopes to `dq_true ⊗ dq_n`
  exactly, leaving only the second-order difference between evaluating the
  boresight at `q_true` versus `q_hat`. This is the single easiest place in
  the phase to invert a sign or a transpose, and it is right.
- **The EKF error-dynamics matrix `F`.** Every block was derived
  independently against the local (body-frame) multiplicative error
  convention `δq = q̂⁻¹ ⊗ q_true` and the truth-minus-estimate sign of the
  additive blocks, and all six populated blocks
  (`cpp/src/gnc/ekf.cpp:307-319`) are correct and mutually consistent with
  `reset()`'s right-multiplication (`cpp/src/gnc/ekf.cpp:405-413`) and with
  the star tracker's `H = [I 0 0 0 0]` (`cpp/src/gnc/ekf.cpp:483`).
- **`nav.innov` structural zero-padding.** The row-by-row embedding at
  `cpp/src/vehicle_cycle.cpp:1126-1134` correctly places an `m×m` packed upper
  triangle into the leading corner of an `m_max×m_max` one; the naive flat
  copy the comment warns against would indeed have scattered it. Verified by
  hand for the offsets. (The bounds defect of finding 1 is orthogonal to
  this — the arithmetic itself is right.)
- **Determinism inside the time loop.** No `unordered_map` or `unordered_set`
  anywhere in the phase; every string-keyed container is `std::map`
  (`cpp/include/star/gnc/config.hpp:31-33, 43-46`), so iteration order is
  deterministic. No clock read, network access, or text parsing inside the
  loop. Seeds are threaded explicitly through `rng::make_stream(master_seed,
  <name>)` with a pure 64-bit derivation and no allocation
  (`cpp/src/rng.cpp:104-114`).
- **RNG draw schedules do not depend on gating.** Both optical sensors draw
  their three normals unconditionally, before and independently of the
  validity gate (`cpp/src/sensors/optical.cpp:190-192, 279-280`), and the IMU's
  initialisation and per-sample schedules are unconditional by construction,
  multiplying by a zero sigma rather than skipping a draw
  (`cpp/src/sensors/imu.cpp:180-198`, `216-241`). The nav fix's optional
  Gauss-Markov draws are conditional on configuration only
  (`cpp/src/sensors/radio.cpp:128-129`), which is constant across a run and
  enters the FR-15 resolved-config hash. This is exactly right and is the
  hazard most often gotten wrong.
- **Per-kind RNG stream names cannot collide.** Each sensor derives its stream
  from its kind string, which would correlate two instances of the same kind —
  but `[sensors.<kind>]` is a TOML table keyed by kind
  (`python/star_reacher/mission.py:2129-2135, 2156-2157`), so a second instance
  of a kind is structurally impossible. Closed by construction.
- **Sensor member initialisation.** Every measurement-holding member in the
  four sensor headers carries an explicit initialiser
  (`cpp/include/star/sensors/radio.hpp:59-66, 99-101`;
  `cpp/include/star/sensors/optical.hpp:96-97, 128-129`), as does every field
  of `SensorCycleTruth` (`cpp/include/star/sensors/sensor.hpp:55-82`) and
  `NavSensorModel` (`cpp/include/star/gnc/component.hpp:129-152`). No
  read-before-assign found.
- **The camera hook's bit-exactness claim holds by construction.** The pose
  channels are copies of the truth doubles, not recomputations
  (`cpp/src/sensors/camera.cpp:121-123`), and `px_` is sized from the landmark
  count at construction (`cpp/src/sensors/camera.cpp:110-111`).
- **The FR-24 truth boundary is structural.** No virtual on `IGncComponent`
  takes a `TruthState`; the loop computes `nav.err` itself from the
  component's declared layout (`cpp/src/vehicle_cycle.cpp:1082-1107`), and
  `GncInput.oracle` is populated only under `cfg.oracle`
  (`cpp/src/vehicle_cycle.cpp:1037-1047`). The guarantee does not rest on a
  rule an implementation is asked to honour.
- **The Python-component registry avoids the static-destruction crash.** The
  name→factory table is a deliberately leaked heap allocation
  (`bindings/module.cpp:577-585`) rather than a static `std::map<std::string,
  py::object>`, which would release Python references after interpreter
  finalisation.

## Not reviewed

Stated explicitly so this review is not read as broader coverage than it has:

- `cpp/tests/*` (the new doctest files) were not reviewed as code.
- `docs/mathlib/chapters/*.tex` derivations were not checked against the
  implementations beyond the specific equation cross-checks noted above.
- `scripts/nees_diag/*` and `tests/refs/*` were not reviewed.
- The mission TOML fixtures under `missions/` were read only for the specific
  questions above.
