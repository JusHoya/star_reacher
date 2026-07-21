# Phase 6 C++ test-coverage audit

Two Phase 6 code paths had been found dead at C++ test time — the
`joseph_update<1>` altimeter instantiation and the `nav.innov` consumer block
in `cpp/src/vehicle_cycle.cpp`. Both were found incidentally while chasing
other questions, not by looking. Two of two is a sampling result, so this
document measures the whole Phase 6 C++ surface rather than sampling it
further.

Source under test: branch `phase-6-gnc-sensors` at
`5fe1351` (`Merge ws-p6-tiers: gate criteria 2, 3, and 9 in the acceptance
suite`), working tree clean.

## Headline result

The Phase 6 surface is **not** substantially unexercised, but the coverage is
distributed very unevenly across the two test tiers, and the tier that carries
most of it is the one no sanitizer instruments.

| Measure | Phase 6 core C++ (9 files, 2,759 distinct lines) |
|---|---|
| C++ doctest suite alone | **75.5% line, 45.6% branch** |
| Python pytest tier alone | **91.7% line, 60.5% branch** |
| Union of both tiers | **91.5% line** (234 lines dead in both) |

The single most consequential number is `cpp/src/vehicle_cycle.cpp`: **53.4%
line and 28.9% branch under the doctest suite, against 94.0% / 69.3% under the
Python tier.** 399 of its lines execute only under Python. `vehicle_cycle.cpp`
is the integration point where every Phase 6 subsystem meets — sensor
scheduling, the GNC component call, the innovation consumer, the SRLOG writer
calls — and it is the file the sanitizers can see least of.

The two previously known findings are not outliers. They are two instances of
one structural pattern: **the C++ tier tests components in isolation and the
Python tier tests them integrated, so integration logic is systematically
outside the reach of ASan, UBSan, and `-Werror`.**

Against that, the good news is real and should not be buried: `imu.cpp` is at
98.0%, `ekf.cpp` at 97.5%, `srlog_writer.cpp` at 93.6%, and `builtin.cpp` at
91.6% — all under the doctest suite alone. The physics and estimation code is
well covered. What is thin is configuration parsing, validity gating, and
mission-sequence behaviour.

Four findings are classified DEAD-CRITICAL below. One of them — the star
tracker's exclusion gating — is the reason this audit was worth running.

## Methodology and toolchain

**Linux.** Ubuntu 24.04.4 LTS under WSL2, GCC 13.3.0
(`Ubuntu 13.3.0-6ubuntu2~24.04.1`), `gcov` 13.3.0, CMake 3.28.3, GNU Make,
gcovr 8.6. The CI `cpp-tests` job runs on `ubuntu-24.04`, whose default
compiler is the same GCC 13.3.0, so this is the CI toolchain rather than an
approximation of it.

All builds were performed in a `git clone` of the repository inside the WSL
filesystem at `/home/hoyer/sr`, not on `/mnt/c`, which produces clock-skew
warnings. The clone's `HEAD` was verified equal to `5fe1351` with a clean
working tree before building. Every build ran with
`CMAKE_BUILD_PARALLEL_LEVEL=2` and `--parallel 2`, one at a time.

**Tier 1 — the C++ doctest suite.**

```
cmake -S . -B build/cov -DCMAKE_BUILD_TYPE=Debug -DSTAR_BUILD_TESTING=ON \
      -DCMAKE_CXX_FLAGS="--coverage -O0 -g -fprofile-abs-path" \
      -DCMAKE_EXE_LINKER_FLAGS="--coverage"
cmake --build build/cov --parallel 2
./build/cov/star_tests
```

61 translation units compiled from an empty binary directory. The test binary
was invoked directly rather than only through CTest, and reported **163 cases
and 56,670 assertions, all passing** — an exact match to the recorded Windows
and Linux baselines, so the instrumented build is the same suite.

**Tier 2 — the Python tier against an instrumented extension module.** This
was practical and was done, so the Python-tier number below is measured rather
than estimated.

```
cmake -S . -B build/covpy -DCMAKE_BUILD_TYPE=Debug -DSTAR_BUILD_TESTING=OFF \
      -DSTAR_BUILD_PYTHON_BINDINGS=ON \
      -DCMAKE_CXX_FLAGS="--coverage -O0 -g -fprofile-abs-path" \
      -DCMAKE_SHARED_LINKER_FLAGS="--coverage"
cmake --build build/covpy --parallel 2
python -m pytest tests/python -q
```

35 translation units, producing an instrumented
`_core.cpython-312-x86_64-linux-gnu.so` placed inside the `star_reacher`
package on `PYTHONPATH`. Separate binary directories mean separate `.gcda`
sets, so the two tiers are measured independently and can be differenced line
by line. Result: **985 passed, 1 failed, 3 skipped**. The recorded Windows
baseline is 993 passed; the difference is platform-conditional selection, not
a regression, and the single failure is discussed under "Incidental findings".

Coverage was extracted with `gcovr --json` per tier and differenced with a
scratch script that aggregates duplicate line records (template instantiations
emit several records per source line) by maximum execution count. Percentages
in the per-file table are gcovr's own totals, computed without
`--exclude-unreachable-branches` on both tiers so the two columns are
comparable.

**Reading the branch percentages.** Branch coverage over C++ compiled at `-O0`
counts a large number of implicit branches — exception-handling edges emitted
around every allocating expression, and Eigen's internal dispatch. A branch
figure near 50% is normal for this style of code and does not by itself
indicate a gap. The branch column is used here to locate *specific* two-way
conditions with only one side taken, not as a headline metric.

## Per-file coverage, Phase 6 surface

Line and branch coverage, each tier independently. `n/a` means the file is not
built into that tier's binary.

| File | Lines | doctest line | doctest branch | pytest line | pytest branch | union line |
|---|---:|---:|---:|---:|---:|---:|
| `cpp/src/sensors/imu.cpp` | 150 | 98.0% | 75.0% | 98.0% | 75.0% | 98.0% |
| `cpp/src/gnc/ekf.cpp` | 353 | 97.5% | 60.7% | 97.5% | 61.2% | 97.0%¹ |
| `cpp/src/srlog_writer.cpp` | 563 | 93.6% | 61.9% | 93.6% | 61.9% | 93.6% |
| `cpp/src/gnc/builtin.cpp` | 203 | 91.6% | 56.9% | 91.6% | 56.9% | 91.6% |
| `cpp/src/gnc/component.cpp` | 190 | 70.5% | 43.5% | 74.7% | 47.7% | 74.7% |
| `cpp/src/sensors/optical.cpp` | 153 | 71.2% | 19.4% | 83.7% | 38.8% | 83.7% |
| `cpp/src/sensors/radio.cpp` | 109 | 67.9% | 20.8% | 79.8% | 36.2% | 79.8% |
| `cpp/src/sensors/camera.cpp` | 103 | 69.9% | 39.7% | 85.4% | 57.9% | 85.4% |
| `cpp/src/vehicle_cycle.cpp` | 983 | **53.4%** | **28.9%** | 94.0% | 69.3% | 94.0% |
| `bindings/module.cpp` | 863 | n/a | n/a | 95.8% | 50.5% | 95.8% |
| **Total, 9 core files** | **2,807** | **75.5%** | **45.6%** | **91.7%** | **60.5%** | **91.5%** |

¹ The union column counts distinct source lines; `ekf.cpp` has 353 coverage
records over 305 distinct lines because `joseph_update` is instantiated three
times. The per-tier columns use gcovr's record counts.

`bindings/module.cpp` is reachable only from Python by construction — it is
the pybind11 module and is not linked into `star_tests`. Its 95.8% is
therefore entirely outside sanitizer reach, and that is structural rather than
a coverage defect.

`cpp/src/run.cpp` (84.4% union, 58.2% doctest) is Phase 2/3 code, outside this
audit's scope, and is reported for context only.

## What the doctest-only number means for sanitizer reach

The ASan, UBSan, and `-Wall -Wextra -Werror` legs recorded in
`docs/ci/phase6_crossplatform.md` were run over the doctest binary. Their reach
is exactly the doctest column above and nothing more.

Concretely, over the Phase 6 core surface:

- **75.5% of lines** are within sanitizer reach.
- **24.5% of lines** are not — of which most are live code that the Python
  tier does execute, against a wheel carrying no instrumentation.
- For `vehicle_cycle.cpp` specifically, **399 lines execute only under
  Python**. Six functions in that file are entered only from Python, including
  `gimbal_basis`, `osc_perigee_alt`, `tdb_s_at`, `VehicleCycle::time_s()` and
  `VehicleCycle::external_command()`.

**The doctest suite's 56,670 assertions do not bound this.** Assertion count
measures how much is checked about the code that runs; it says nothing about
what fraction of the code runs. 56,670 assertions over 75.5% of lines and
45.6% of branches is a dense check of a partial surface. The number is
evidence of thoroughness *within* the covered region and is not evidence of
breadth. It should never be cited as a coverage proxy.

An important correction to the earlier record follows from this. The
`docs/ci/phase6_crossplatform.md` finding that the `nav.innov` consumer guard
is dead at C++ test time is confirmed and still true at `5fe1351` — but the
guards at `cpp/src/vehicle_cycle.cpp:1128` and `:1140` **are** exercised, with
real assertions on the exception message, by
`tests/python/test_gnc_python_component.py:1131`
(`test_an_innovation_wider_than_declared_is_refused`) and `:1149`
(`test_a_short_innovation_covariance_is_refused`). Both throw bodies show
non-zero execution counts under the Python tier. The correct statement is
therefore not "those memory-safety fixes are untested" but "those
memory-safety fixes are tested only in the tier that carries no sanitizer
instrumentation." That is a narrower and more accurate claim than the earlier
document supports, and it is the one this audit stands behind.

## Classified uncovered paths

Classification is by consequence, not by line count.

### DEAD-CRITICAL

Safety, correctness, or validity logic that **no test executes in either
tier**. Each is stated with the measured execution counts that establish it.

---

**C-1. Star tracker sun-exclusion and central-body-exclusion gating is never
executed by any test.**
`cpp/src/sensors/optical.cpp:210-222`.

Measured execution counts on the guard and its body:

| Line | Code | doctest | pytest |
|---|---|---:|---:|
| 209 | `if (latest_.geom.ephemeris_valid) {` | 2,016 | 78,056 |
| 210 | `if (cfg_.sun_exclusion_rad > 0.0) {` | **0** | 1,200 |
| 211-214 | sun-exclusion angle test | **0** | **0** |
| 216 | `if (cfg_.central_body_exclusion_rad > 0.0) {` | **0** | 1,200 |
| 219-221 | central-body-exclusion angle test | **0** | **0** |

This is the `eq:optical:gating` validity computation and it has never run.
The mechanism is a two-configuration blind spot that neither test could see on
its own:

- `tests/python/test_gnc_missions.py:567-568` configures
  `sun_exclusion_rad = 0.5236` and `central_body_exclusion_rad = 0.4363`, but
  that mission's geometry carries `ephemeris_valid == false`, so line 209
  rejects every one of its samples before the exclusion code is reached.
- `tests/python/test_p6_optical_gates.py:183-184` runs with a valid ephemeris
  — it is the only configuration that reaches line 210 at all — but sets both
  radii to `0.0` deliberately, so the `> 0.0` guard is false on all 1,200
  evaluations.

The mission that configures the feature cannot reach it; the mission that can
reach it switches it off. The result is a validity flag that is written to
`sensors.startracker.valid` on every sample of every run and whose computation
has never been exercised. An inverted comparison (`>=` for `<=`), a wrong
reference direction, or a units error in either radius would pass the entire
suite. This is the same shape as the two defect classes the project has
already shipped: a validity flag whose payload looked plausible when invalid,
and a gate that reported PASS on empty data.

---

**C-2. The rotation-vector attitude error forms are never computed.**
`cpp/src/gnc/component.cpp:218-222`.

`compute_error_state` writes the `nav.err` channel. Its attitude branch has
three shapes; execution counts:

| Line | Form | doctest | pytest |
|---|---|---:|---:|
| 205-208 | `kQuatDifferenceAligned` | 510 | 53,140 |
| 214-217 | `kQuatErrorLocal` / `kQuatErrorGlobal` | 4 | 734,036 |
| 220-222 | `kRotationVectorLocal` / `kRotationVectorGlobal` | **0** | **0** |

`ErrorForm::kRotationVectorLocal` and `kRotationVectorGlobal` are declared in
the public header at `cpp/include/star/gnc/component.hpp:291,293` and are sized
by `error_block_size` at `cpp/src/gnc/component.cpp:116-117`, so a component
author can select them today. The write itself — `e[o] = 2.0 * dq.x()` and the
two following — has never executed. The three-element rotation vector is the
conventional error-state parameterization for an error-state EKF and is the
form a NEES computation would most naturally consume. The small-angle factor
of 2, the choice of `dq.x/y/z` over `dq.w`, and the local/global composition
side are all unverified.

The comment at line 219 asserts an invariant that carries the correctness of
the branch — "dq is already in the +w hemisphere, so 2 sgn(dq_w) dq_v is
2 dq_v" — and no test confirms it.

---

**C-3. The NavFix Gauss-Markov correlated error model is entirely dead.**
`cpp/src/sensors/radio.cpp:101-119`.

| Line | Code | doctest | pytest |
|---|---|---:|---:|
| 101 | `if (cfg_.gm_r.sigma > 0.0 && cfg_.gm_r.tau_s > 0.0)` | 6 | 1,250 |
| 102-106 | position GM setup and stationary init | **0** | **0** |
| 108 | velocity GM guard | 6 | 1,250 |
| 109-111 | velocity GM setup and stationary init | **0** | **0** |
| 117-119 | `NavFix::advance_gm` | **0** | **0** |

`NavFix::advance_gm` is a function that no test in either tier has ever
entered. The parameters that enable it — `gm_position_tau_s`,
`gm_velocity_sigma_mps`, `gm_velocity_tau_s` — are documented user-facing
mission parameters at `python/star_reacher/mission.py:416-418`. Three specific
correctness claims are unverified: the stationary variance relation
`w_sigma = sigma * sqrt(1 - phi^2)` (lines 103, 110), the stationary
initialization `c = sigma * N(0,1)` (lines 106, 111), and the draw schedule
documented in the comment at lines 122-127, which fixes the RNG consumption
order and therefore the bit-reproducibility of any run that enables the
feature. Enabling this in a mission today produces numbers no test has ever
checked.

---

**C-4. `IGncComponent::state()` and `IGncComponent::covariance_upper()` base
implementations never execute.**
`cpp/src/gnc/component.cpp:18-28`.

Both are functions never entered in either tier. They are the base-class
contract violation guards: a component that declares `state_dim() > 0` but does
not override the accessor should get a `std::logic_error`, not a silent read of
an unwritten buffer. Given that the two heap defects this phase already fixed
were both "a component returned something the loop was not sized for", the
guard for the adjacent failure mode being unexecuted is the same class of
exposure. It is listed as critical for that reason rather than for its own
complexity, which is trivial.

### DEAD-REACHABLE

Real behaviour nothing exercises, with lower consequence.

**R-1. Two mission sequence actions never execute.**
`vehicle_cycle.cpp:1278-1279` (`attitude_hold`) and `:1283-1286`, `:1334-1336`,
`:1489-1500` (`rate_command`). Both are documented actions
(`python/star_reacher/mission.py:77`, `:81-82`, `:238`). The only appearance of
`rate_command` in the test suite is
`tests/python/test_mission_sequence.py:488`, a **negative** case using
`frame = "lvlh"` that the Python validator rejects before the core sees it. So
the C++ implementation of open-loop rate command — including the quaternion
integration at lines 1492-1500 and the GCRF-versus-body frame selection at
1335-1336 and 1490-1491 — has never run.

**R-2. The drag-enabled branch of the 6DOF environment spec never executes.**
`vehicle_cycle.cpp:353-359`. `atmosphere_from_name` is a function never entered
in either tier. It is called only at `vehicle_cycle.cpp:557`, on the true side
of `cfg.drag_enabled ? ... : ...`, so no 6DOF mission in the suite enables
drag. The three-way name mapping and its rejection of an unknown name are
unexercised in the core; a typo would be caught only by the Python validator.

**R-3. Non-Earth central bodies never execute in the 6DOF path.**
`vehicle_cycle.cpp:345-350` and `:721-725`. `central_body_from_name` is entered,
but only its `earth` branch; the `moon`, `mars`, and `sun` returns and the
unknown-body throw are dead, as are the Mars and Moon reference-radius
assignments. Lunar and Mars work in this project runs through `run.cpp`, not
the 6DOF vehicle path, so this is a real but currently unused capability.

**R-4. Configuration-rejection throws across the sensor and GNC parsers.**
Roughly 90 lines. Every `throw std::invalid_argument` in
`sensors/optical.cpp` (lines 28-29, 42-43, 50-51, 64-66, 71-72, 82-84, 88-89,
163, 252), `sensors/radio.cpp` (26-27, 39-40, 47-48, 58-60, 64-65, 95, 165),
`sensors/camera.cpp` (26-27, 43-44, 51-52, 70-71, 79-81, 86, 93-95, 108),
`sensors/imu.cpp` (121, 129-130), `gnc/ekf.cpp` (45-47, 109-111, 160-163) and
`gnc/builtin.cpp` (40-42, 50-52, 68-70, 189-197, 292-298) is dead in the
doctest tier; a subset is reached from Python and the remainder is dead in
both. These reject malformed mission configuration. Individually each is a
one-line guard whose failure mode is a confusing error message rather than a
wrong number, which is why they are classified reachable rather than critical.
Collectively they are the largest single block of untested code in the phase,
and the parameter-name allow-lists in particular (`reject_unknown`) are the
mechanism that stops a silently ignored typo in a mission file from producing
a plausible-looking run with a default value.

**R-5. The error-layout validator's rejection paths are dead in the C++ tier.**
`gnc/component.cpp:122-125`, `:128-131`, `:140-143`, `:160-165`, `:170-173`.
Only the "blocks do not cover state_dim" rejection at `:178-182` is reached,
and only from Python. The validator's contract is that a declared layout tiles
the state vector contiguously from index 0 with no gaps and no overlaps; the
gap/overlap rejection at `:160-165` is precisely the check that stops a
mis-declared layout from making `compute_error_state` write outside the error
buffer, and it is untested in the tier that would catch the resulting overrun.

**R-6. The JSON string escaper's escape branches never execute.**
`srlog_writer.cpp:34-45`. The `"` case, the `\` case, and the control-character
`\u00xx` case are all dead; only the pass-through default runs. The function is
applied to header strings at `srlog_writer.cpp:58-64`, `:260-274` and `:289` —
version, git hash, config digest, epoch, central body, sensor kind, and channel
metadata. A quote or backslash reaching any of those would today produce
syntactically invalid JSON in `meta.json` with no test detecting it. The
function's own comment concedes the inputs are "ASCII in practice", which is an
assumption, not an enforced invariant.

**R-7. Camera extrinsics and landmarks are parsed only from Python.**
`sensors/camera.cpp:69-99`. `r_cam_b_m`, `q_b2c`, and `landmarks_fixed_m`
parsing, including the quaternion normalization at `:88` and the modulo-3
landmark grouping at `:92-99`, is dead in the doctest tier. The C++ tier only
ever constructs a camera with default extrinsics and no landmarks.

**R-8. `VehicleCycle::cycle()` is never called.**
`vehicle_cycle.cpp:1651`. A public accessor on the stepping API, entered by no
test in either tier.

### UNREACHABLE-BY-DESIGN

Defensive branches that cannot fire given a stated invariant. Naming the
invariant is the point; where the invariant is not actually enforced, the item
is listed above instead.

**U-1. The SRLOG writer's "group was not declared at header-write time"
guards.** `srlog_writer.cpp:650-651`, `:661-663`, `:691-693`, `:706-708`,
`:719-721`, `:734-736`, `:833-834`. Each `write_*` method throws
`std::logic_error` if its group is absent from the header. The invariant is
that `SrlogWriter` writes its header in the constructor from a fields struct,
and every `write_*` call site in `vehicle_cycle.cpp` is guarded by the same
`fields.*_enabled` flag that populated the header. These fire only on an
internal inconsistency between two parts of the same object, which is what a
`logic_error` is for. Correctly untested.

**U-2. `truth_vector`'s default case and throw.**
`gnc/component.cpp:80-84`. All five 3-vector quantities are covered (the switch
at line 69 shows 5 of 6 branches taken). The `default` reaches only
`kAttitude` and `kMass`, and both are intercepted by the caller at
`compute_error_state` lines 191 and 226 before `truth_vector` is called. The
invariant is that interception; it holds by construction in the one call site.

**U-3. `quantity_name`.** `gnc/component.cpp:42-59`. A function never entered,
called only from inside the validator throw messages of R-5. It becomes live
the moment any R-5 test is added; it needs no test of its own.

**U-4. `optical.cpp:131`, `if (n == 0.0) return u_hat;`** — the comment at that
line already states the reasoning: unreachable for `|beta| < 1`, retained to
make the function total. Correctly untested.

**U-5. `ekf.cpp:516-520`, the altimeter update's body-fixed-frame guard.** The
early return is dead in both tiers. The invariant is that
`update_altimeter` is called at `ekf.cpp:191-192` only when
`sensors_.altimeter_present`, and the altimeter sensor itself only produces a
sample when the run supplies a body-fixed frame. This one is borderline: the
invariant is a property of the current call graph, not an enforced
precondition, and the guard's whole purpose is to be defensive about an
environment the EKF does not control. It is listed here rather than as
DEAD-REACHABLE because a wrong outcome is a skipped update, not a wrong number
— but it is the weakest "by design" claim in this section and a test would be
cheap.

## Covered but not asserted

Coverage tools count execution, not verification. Three cases in the Phase 6
surface are executed by tests that never check the result.

**A-1. The star tracker and sun sensor validity flags are computed but never
asserted, in either tier.** `optical.cpp:225` (slew limit) executes 3,600 times
and `:291` (sun sensor field of view) executes 2,400 times under the Python
tier, and both write `valid_`, which reaches
`sensors.startracker.valid` and `sensors.sunsensor.valid` in the log. No test
in the repository asserts on either channel's value. Searching the suite for
assertions on those groups finds only reads of `q_meas_i2b`
(`tests/python/test_gnc_missions.py:694`) and the sun vector
(`tests/python/test_p6_optical_gates.py:316`, `:432`, `:692`); the
`assert ... "valid"` hits elsewhere in the suite are all on `gnc.cmd` or on
observation dictionaries, not on these sensors. On the C++ side,
`StarTracker::last_valid()` and `SunSensor::last_valid()` appear in no test;
the `in.startracker.valid` assignments at `cpp/tests/test_ekf.cpp:281`, `:385`
and `:496` are hand-constructed EKF inputs, which test the EKF's *response* to
the flag but not the sensor's *computation* of it.

The contrast with the altimeter is instructive and shows this is a gap rather
than a house style: `cpp/tests/test_sensors.cpp:1050` and `:1057` assert
`alt.last_valid()` both true and false around the configured band. The
altimeter's flag is verified; the two optical flags are not.

Combined with C-1, the position is that the star tracker's validity flag has
two exclusion terms that never execute and one slew term that executes without
ever being checked. Nothing in the suite would detect the flag being wrong.

**A-2. `camera.cpp:181-182`, the Sun direction in camera frame.** Executed
under the Python tier only. It computes `sun_c_` through an aberration and two
frame rotations. The sun-vector assertions in
`tests/python/test_p6_optical_gates.py` are on the *sun sensor* group, not the
camera's `sun_c_`. Worth a targeted check of whether any test consumes this
value before treating the line's coverage as verification.

**A-3. The `bindings/module.cpp` 95.8%.** Much of this is pybind11 glue
executed as a side effect of every Python test that imports the module. High
coverage there should not be read as high verification of the binding layer's
type conversions; it means the module loads and its functions are called.

## Ranked test recommendations

Ranked by the probability that the absence of the test lets a real defect
through, not by line count. These are recommendations only; implementation is a
separate workstream.

**1. Drive the star tracker through both exclusion gates, with assertions on
the validity flag.** Closes C-1 and A-1 together — the highest-value item by a
wide margin. A C++ doctest case constructing a `StarTracker` with a nonzero
`sun_exclusion_rad`, feeding `SensorCycleTruth` with `ephemeris_valid == true`
and a boresight placed first outside and then inside the exclusion cone,
asserting `last_valid()` is true then false, and repeating for
`central_body_exclusion_rad`. This is the one gap where a wrong sign or a wrong
reference direction would today pass the entire suite while silently
mis-flagging every star tracker sample in every run. Placing it in the C++ tier
also brings the code inside sanitizer reach.

**2. Add a C++ doctest case driving `VehicleCycle` with a component that
returns innovations.** Closes the largest sanitizer-reach hole. The `nav.innov`
consumer at `vehicle_cycle.cpp:1117-1173` is correctly guarded and the guards
*are* asserted from Python, so this is not about correctness of the guards — it
is about giving ASan and UBSan reach over the embedding loop at `:1165-1169`,
where the two shipped heap defects lived. The experiment recorded as open in
`docs/ci/phase6_crossplatform.md` — whether ASan can see a consumer-side
overrun at all, given that `std::vector` overruns within capacity are invisible
without container annotations — cannot be run until this test exists, and it
should be run immediately after, because a negative result would mean the ASan
leg should not be relied on for this defect class regardless of coverage.

**3. Pin the rotation-vector error forms against an analytic value.** Closes
C-2. A small-angle rotation about a known axis has a closed-form
`2 sgn(dq_w) dq_v`, so this can be an analytic case rather than a regenerated
golden, in the style of the altimeter case at
`cpp/tests/test_ekf.cpp` that closed the `joseph_update<1>` gap. Cover both the
local and global composition sides, and include one case with the truth
quaternion in the negative hemisphere so the `+w` canonicalization the comment
at line 219 depends on is actually exercised.

**4. Exercise the NavFix Gauss-Markov path.** Closes C-3. Two assertions carry
most of the value: that the sample variance of the correlated component is
stationary at `sigma^2` over a long run (which tests the
`sqrt(1 - phi^2)` relation and the stationary initialization together), and
that the RNG draw order matches the documented schedule, since that fixes
bit-reproducibility for every mission that enables the feature. A
`test_gm_crosscheck.py`-style comparison against a reference implementation
already exists as a pattern in the suite.

**5. Add negative cases for the error-layout validator.** Closes R-5 and,
incidentally, U-3. Five short `CHECK_THROWS_AS` cases — overlapping blocks, a
gap, a non-quaternion form on an attitude block, a bias block with no truth
counterpart, and a layout not starting at 0. The gap/overlap check is the one
that matters: it is what prevents `compute_error_state` from writing outside
the caller's error buffer, and it is currently unverified in the tier that
would catch the overrun.

**6. Add the two base-class contract guards.** Closes C-4. Two
`CHECK_THROWS_AS` cases against a component declaring `state_dim() > 0` without
overriding the accessors. Cheap, and directly adjacent to two defects this
phase already shipped.

**7. Cover one mission-sequence action end to end: `rate_command`.** Closes the
larger half of R-1. It is the only untested action carrying real arithmetic —
quaternion integration and a frame selection — as opposed to a flag
assignment. `attitude_hold` is a two-line state change and can be left.

**8. Add one JSON-escape unit test.** Closes R-6. A three-line case asserting
that a string containing `"` and `\` round-trips through
`append_json_string` into parseable JSON. Low probability of a defect today
given the constrained inputs, but the cost is near zero and the failure mode —
a malformed `meta.json` that every downstream reader rejects — is loud and
total.

**Not recommended before phase close.** The bulk of R-4 (roughly 90 individual
configuration-rejection throws), R-2, R-3, R-7, R-8, and everything under
UNREACHABLE-BY-DESIGN. Adding a test per rejection message would add
significant suite weight for defects whose worst outcome is a poor error
message. The exception within R-4 is the `reject_unknown` allow-list mechanism
itself: **one** test per sensor parser confirming that an unknown parameter
name is rejected rather than ignored is worth having, because a silently
ignored typo produces a run that looks correct and used a default value. That
is a different failure mode from the rest of the block, and the Python tier
already covers part of it.

## Nothing here is a phase-close blocker

The brief asked for anything so dangerous that leaving it untested through
phase close would be indefensible. There is nothing in that category, and it
would be wrong to manufacture one.

C-1 is the closest, and it deserves to be stated precisely rather than
escalated: an unexercised validity gate on a sensor whose measurements are
still logged and still folded into the EKF. If the gate is wrong, the effect is
that star tracker samples which should have been rejected are accepted, which
degrades an estimate rather than corrupting memory. It is a correctness risk on
a research instrument, not a safety-of-flight defect, and recommendation 1 is a
small piece of work. The honest framing is that C-1 should be closed before
phase close because it is cheap and because the gate is currently a claim the
project makes without evidence — not because shipping it would be
indefensible.

## `-Wshadow` on the Linux CI leg

Measured on the real toolchain rather than by inspection, as required.

```
cmake -S . -B build/wshadow -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-Wall -Wextra -Wshadow -Werror"
cmake --build build/wshadow --parallel 2
```

Ubuntu 24.04.4, GCC 13.3.0, from an empty binary directory:

| Metric | Result |
|---|---|
| Translation units compiled | 61 |
| Configure exit status | 0 |
| Build exit status | 0 |
| `-Wshadow` warnings | **0** |
| Warnings or errors of any category | **0** |
| `star_tests` binary | 2,643,136 B, SHA-256 `73ad431f…` |
| Suite result, binary invoked directly | 163 cases, 56,670 assertions, all passing |

The measured cost is zero, confirming the earlier claim on the toolchain that
gates CI. The flag was applied at `.github/workflows/ci.yml:173` with a WHY
comment recording which defect class it guards: four name-shadowing sites
reached this branch and were caught only by MSVC `/W4` (C4457/C4458), which no
CI job builds. `-Wall -Wextra` does not imply `-Wshadow`; GCC diagnoses the
same four sites once it is present.

## Incidental findings

**`tests/python/test_sim.py:586` fails on Linux.**
`test_close_releases_the_log_of_an_abandoned_run` asserts that `run.srlog` has
non-zero size after two `Sim.step()` calls without a close. On this
Linux/GCC/`-O0` build the file exists but is 0 bytes, so nothing has reached
disk yet. The test passes on Windows/MSVC. This is consistent with a
stream-buffering difference — libstdc++ has not flushed at the point MSVC's
runtime would have — rather than a coverage artifact, but it was observed
under an instrumented build and has not been reproduced against an optimized
Linux build, so the attribution is provisional. It is not a Phase 6 GNC
defect and is recorded here only because it was observed. It does suggest the
test asserts on flush timing that is not part of any documented contract.

**Python-tier test counts differ by platform.** 985 passed / 1 failed /
3 skipped here, against the recorded 993 passed on Windows. The difference is
platform-conditional selection and the failure above, not a regression in the
Phase 6 code.

## What could not be measured

- **Whether ASan detects a consumer-side `nav.innov` overrun when the path
  actually executes.** Still open, and still blocked on recommendation 2. This
  audit establishes the coverage gap that makes the question unanswerable; it
  does not answer it. `std::vector` overruns within an allocation's capacity
  are invisible to ASan without libstdc++ container annotations, so the answer
  may be negative.
- **MSVC coverage.** All coverage here is GCC/gcov on Linux. MSVC compiles
  additional or different code under `#ifdef`, and no Windows coverage
  instrumentation was run. The per-file percentages are Linux figures.
- **Sanitizer coverage of the Python tier.** The instrumented module built
  here carries `--coverage`, not `-fsanitize=address`. Building the extension
  module under ASan and running pytest against it is feasible and would close
  the reach gap directly, but was not attempted; it needs `LD_PRELOAD` of the
  ASan runtime into the Python process and a leak-suppression file for
  CPython's own allocations, which is a larger detour than this audit's scope.
- **`bindings/module.cpp` under the doctest tier.** Structurally impossible —
  the module is not linked into `star_tests`. Its 36 lines dead in the Python
  tier were not classified, as the binding layer is outside the Phase 6 GNC
  and sensor surface this audit was scoped to.
- **Whether each of the 2,525 covered lines is *asserted* about.** Three cases
  are documented under "Covered but not asserted" and were found by following
  specific validity-flag and sensor-output paths. A complete assertion audit
  is a different and much larger exercise; the coverage numbers in this
  document bound execution, not verification, and should be read that way.
- **aarch64 / the `pi5` preset.** No target hardware, unchanged from the
  earlier record.
