# Phase 6 coverage-audit closure

What was done about the findings of
[`phase6_coverage_audit.md`](phase6_coverage_audit.md), what the numbers are
now, and what remains open. The audit measured; this document records the
work that answers it and re-measures on the same toolchain and method.

Source under test: branch `phase-6-gnc-sensors`, working tree clean. The
audit's baseline is its own measurement at `5fe1351`.

## Headline

| Measure | Audit baseline | Now |
|---|---:|---:|
| C++ doctest suite, Phase 6 core (9 files) | 75.5% line, 45.6% branch | **78.3% line, 49.7% branch** |
| Union of both tiers, same 9 files | 91.5% line, 234 lines dead in both | **92.7% line, 202 lines dead in both** |
| doctest cases / assertions | 163 / 56,670 | **173 / 57,488** |

All four DEAD-CRITICAL findings and both "covered but not asserted" optical
findings are closed, in the C++ tier, so each is now inside the reach of the
ASan, UBSan, and warnings-as-errors legs. The line-coverage gain is modest
because the findings were small blocks of consequential code, not large ones;
the branch gain on the two sensor files is where the work shows.

## Toolchain

Ubuntu 24.04.4 LTS under WSL2, GCC 13.3.0 (`Ubuntu 13.3.0-6ubuntu2~24.04.1`),
`gcov` 13.3.0, gcovr 8.6, CMake 3.28.3, in a clone at `/home/hoyer/sr` inside
the WSL filesystem. Every build ran with `CMAKE_BUILD_PARALLEL_LEVEL=2` and
`--parallel 2`, one at a time. The same suite was also built and run on
Windows with MSVC; the two toolchains agree exactly at 173 cases and 57,488
assertions.

A separate `-Wall -Wextra -Wshadow -Werror` build of the whole tree, from an
empty binary directory, compiles the added tests with zero warnings of any
category.

## Per-file, C++ doctest tier

Line and branch, doctest tier alone, against the audit's table.

| File | Audit line | Now line | Audit branch | Now branch |
|---|---:|---:|---:|---:|
| `cpp/src/sensors/imu.cpp` | 98.0% | 98.0% | 75.0% | **79.5%** |
| `cpp/src/gnc/ekf.cpp` | 97.5% | 97.5% | 60.7% | 60.7% |
| `cpp/src/srlog_writer.cpp` | 93.6% | 93.6% | 61.9% | 61.9% |
| `cpp/src/gnc/builtin.cpp` | 91.6% | 91.6% | 56.9% | 56.9% |
| `cpp/src/gnc/component.cpp` | 70.5% | **76.8%** | 43.5% | **47.7%** |
| `cpp/src/sensors/optical.cpp` | 71.2% | **85.0%** | 19.4% | **45.9%** |
| `cpp/src/sensors/radio.cpp` | 67.9% | **80.7%** | 20.8% | **40.0%** |
| `cpp/src/sensors/camera.cpp` | 69.9% | **70.9%** | 39.7% | **42.1%** |
| `cpp/src/vehicle_cycle.cpp` | 53.4% | **56.7%** | 28.9% | **32.4%** |
| **Total, 9 files** | **75.5%** | **78.3%** | **45.6%** | **49.7%** |

`imu.cpp` gains branch coverage without gaining lines because the added
unknown-parameter case walks its allow-list loops for the first time.
`vehicle_cycle.cpp` remains the file the doctest tier sees least of: the
innovation consumer is now covered, but the mission-sequence actions, the
drag-enabled environment branch, and the non-Earth central bodies (audit
items R-1 through R-3) are unchanged and were explicitly not recommended
before phase close.

## Finding-by-finding

Executed lines within each finding's line range, doctest tier, measured with
gcov.

| Finding | Location | Audit | Now |
|---|---|---:|---:|
| C-1 star tracker exclusion gating | `optical.cpp:209-223` | 0 of 10 | **10 of 10** |
| C-2 rotation-vector error forms | `component.cpp:218-223` | 0 of 3 | **3 of 3** |
| C-3 NavFix Gauss-Markov model | `radio.cpp:101-119` | 0 of 13 | **13 of 13** |
| C-4 base-class accessor guards | `component.cpp:18-28` | 0 of 6 | **6 of 6** |
| A-1 star tracker slew flag | `optical.cpp:224-227` | executed, unasserted | **asserted both ways** |
| A-1 sun sensor validity flag | `optical.cpp:289-293` | executed, unasserted | **asserted both ways** |
| Recommendation 2, innovation consumer | `vehicle_cycle.cpp:1116-1176` | dead in the C++ tier | **31 of 31** |

## Tests added

Each is stated with what makes its fixture non-degenerate, because a gate
whose fixture cannot reach its target reads as coverage while checking
nothing --- which is exactly how C-1 came to exist.

**`sensors_startracker_exclusion_gating`** (`cpp/tests/test_sensors.cpp`).
Both exclusion terms, each with the other switched off, at 40/20 degrees and
then 30.5/29.5 degrees around a 30-degree sun radius, and 35/15 then
25.5/24.5 around a 25-degree central-body radius. Non-degenerate because
`geom.ephemeris_valid` is true (the mission that configured these radii
carried an invalid ephemeris and was rejected before reaching them), both
velocities are zero so the aberration factor is the identity and the tested
angle is the geometric separation exactly, and `sigma_rad` and
`slew_limit_radps` are zero so neither noise nor the slew term can carry the
result. The case also asserts that the same excluded geometry reports valid
without an ephemeris, which pins the trap itself.

**`sensors_startracker_slew_limit_flag`**. Both directions plus a
three-axis case whose per-component rates are each inside the limit while
the norm is outside. Non-degenerate because the exclusion radii are zero and
no ephemeris is supplied, so the flag is the slew comparison alone.

**`sensors_sunsensor_validity_flag`**. Field of view at 10/30 and 19.5/20.5
degrees against a 20-degree half-angle, total umbra, penumbra, and a missing
ephemeris. Non-degenerate because `sigma_rad` is zero so the measured
direction is exactly the true one, and each of the three gating reasons is
exercised with the other two satisfied.

**`sensors_navfix_gauss_markov_correlated_errors`**. The exact draw schedule
replayed against an independently constructed stream for 25 samples, then
20,000 samples for the stationary variance and the lag-one autocorrelation.
Non-degenerate because the white standard deviations are zero, so the
measured fix minus truth is exactly the correlated component --- while the
white draws are still consumed, which is what makes the replayed schedule the
real one.

**`gnc_rotation_vector_error_forms_match_the_analytic_reduction`**
(`cpp/tests/test_gnc.cpp`). The closed form `2 sin(theta/2) u` for a rotation
of `theta = 1e-3` about a known axis, on both composition sides, with the
truth quaternion also negated to exercise the `+w` canonicalization the
branch's comment depends on. Analytic, not a regenerated golden.
Non-degenerate because the estimate is a non-identity 120-degree rotation, so
the two composition sides are genuinely different and the case asserts they
disagree while preserving the rotation magnitude.

**`gnc_base_component_accessors_refuse_an_undeclared_state`**.
Non-degenerate because the probe declares `state_dim() == 3`, which is the
path a real misuse takes: the loop sizes a buffer from the declaration and
then calls an accessor the author did not supply.

**`sensors_parsers_reject_an_unknown_parameter_name`**. One misspelling per
sensor parser, scalar and vector. Non-degenerate because each case first
asserts that the base configuration is accepted, so a rejection can only come
from the unknown name.

**`gnc_cycle_innovation_consumer_pads_and_embeds`**,
**`..._refuses_a_malformed_sample`**, **`..._run_is_byte_identical`**
(`cpp/tests/test_gnc_cycle.cpp`). A `VehicleCycle` run driven by a component
declaring `innov_max_dim() == 3` that alternates a two- and a
three-dimensional innovation. Non-degenerate because padding and embedding
are both no-ops when `m` always equals `m_max`: the short sample is what
makes the row stride observable, and the assertion distinguishes the
structural embedding `[4, 0.5, 0, 9, 0, 0]` from the flat copy
`[4, 0.5, 9, 0, 0, 0]` a naive implementation produces.

## Mutation evidence

Every added test was shown to fail under a mutation of the thing it claims to
check. Each mutation was applied to the pristine source, built, and run
against the single named test case; the source was restored afterwards and
the tree verified clean. Twenty-four mutations, twenty-four detected.

| # | Mutation | Case | Result |
|---|---|---|---|
| M1 | sun exclusion comparison inverted | exclusion gating | 5 of 11 assertions fail |
| M2 | sun exclusion radius scaled by 0.9 | exclusion gating | 1 of 11 fail |
| M3 | central-body comparison inverted | exclusion gating | 6 of 11 fail |
| M4 | central-body reference direction sign flipped | exclusion gating | 3 of 11 fail |
| M5 | ephemeris guard removed from the gate | exclusion gating | 1 of 11 fail |
| M6 | slew comparison inverted | slew limit | 3 of 4 fail |
| M7 | slew tested per-axis, not on the norm | slew limit | 1 of 4 fail |
| M8 | sun sensor field-of-view comparison inverted | sun sensor validity | 5 of 8 fail |
| M9 | umbra gate admits total shadow | sun sensor validity | 1 of 8 fail |
| M10 | unknown-parameter allow-list disabled | parser rejection | 2 of 13 fail |
| M11 | drive variance loses the square root | Gauss-Markov | 76 of 159 fail |
| M12 | stationary initialization zeroed | Gauss-Markov | 75 of 159 fail |
| M13 | draw order: velocity before position | Gauss-Markov | 150 of 159 fail |
| M14 | recursion drops the `phi` term | Gauss-Markov | 151 of 159 fail |
| M15 | rotation-vector reduction loses the factor of two | rotation vector | 7 of 17 fail |
| M16 | rotation-vector reduction takes the scalar part | rotation vector | 4 of 17 fail |
| M17 | `+w` canonicalization removed | rotation vector | 3 of 17 fail |
| M18 | composition side forced to local | rotation vector | 3 of 17 fail |
| M19 | base guard throws the wrong exception type | base accessors | 1 of 6 fail |
| M20 | covariance embedded with a flat row stride | innovation consumer | 22 of 594 fail |
| M21 | covariance zero-pad removed | innovation consumer | 30 of 594 fail |
| M24 | innovation-vector zero-pad removed | innovation consumer | 10 of 594 fail |
| M22 | innovation width guard removed | malformed sample | 1 of 1 fail |
| M23 | covariance-length guard removed | malformed sample | 1 of 2 fail |

M2 is the one that matters most for C-1's original framing: a 10% misreading
of the configured radius --- a units error, a half-angle-versus-full-angle
error, or a stale constant --- is caught by the half-degree straddle and by
nothing else in the case.

M20, M21, and M24 required a helper to survive `-Werror`, which is itself a
small piece of evidence that the warnings-as-errors leg now reaches this
code: removing the padding leaves `mm` unused and the build fails before the
test can run.

**M15–M18 no longer have a target.** Their case, the rotation-vector
attitude error forms, has since been removed from `ErrorForm` — see the
inconsistency this document reports below, which was the reason. The four
results above stand as measured against the code as it was; they are not
re-runnable, and the mutations they describe cannot be applied to the current
source because the arithmetic they mutate is gone. The case that replaced
them, `gnc_attitude_block_last_layout_cannot_outrun_the_state_buffer`
(`cpp/tests/test_gnc.cpp`), targets the descriptor invariant rather than the
reduction: it re-attempts the out-of-bounds construction through
`validate_error_layout` and pins that the write stays inside the declared
width.

## The ASan reach experiment

The audit left one question open and blocked on recommendation 2: **can ASan
detect a consumer-side `nav.innov` overrun at all?** The test now exists, so
the experiment was run.

Method. Build the doctest binary with `-fsanitize=address,undefined` (GCC
13.3.0, Debug, `-fno-omit-frame-pointer`), remove one of the consumer's two
length guards so the unguarded access actually executes, and run the
malformed-sample case. The variable under study is whether the overrun leaves
the heap allocation or stays inside the container's spare capacity.

| Scenario | Buffer state | Detected |
|---|---|---|
| A: write past `innov_y_buf` | capacity equals size | **yes** --- `heap-buffer-overflow`, "0 bytes after 24-byte region" |
| B: read past `s.s_upper` | capacity equals size | **yes** --- `heap-buffer-overflow` at `vehicle_cycle.cpp:1167`, "0 bytes after 40-byte region" |
| C: write past `innov_y_buf` | spare capacity reserved | **no** |
| D: read past `s.s_upper` | spare capacity reserved | **no** |

The answer is therefore conditional, and the condition is not a property of
the consumer's code. A standalone probe confirms the mechanism directly: on
this toolchain a `std::vector` with `reserve(64)` and `assign(3, ...)`
accepts a write to index 3 with no report, so libstdc++ container
annotations are **not** active in a plain `-fsanitize=address` build.

What this implies, stated as plainly as the audit asked for it. The specific
defect class this phase shipped twice *would* be caught today, because
`innov_y_buf` and `innov_s_buf` are `assign`ed once on fresh vectors and
libstdc++ allocates exactly the requested size, so an overrun leaves the
allocation. But that is an accident of an allocation pattern, not a property
of the guard or of the sanitizer. Any later change that reserves, shrinks, or
reuses either buffer moves the identical defect into scenario C and makes it
invisible --- with no test failing and no ASan report to say so. **The ASan
leg is not a substitute for the length guards at
`vehicle_cycle.cpp:1128`/`:1140`; it is a second line that happens to hold
under the current allocation pattern.** The guards, which are now asserted in
both tiers, are what actually bound this.

Two secondary results. The full 173-case suite runs clean under ASan and
UBSan together, including all the newly added cases. And scenarios C and D
required correcting a first attempt that reserved capacity on the local
`InnovationSample` --- `push_back` copies it and the copy has exact capacity
--- and that made `y` wider than `m_max`, which also drives the covariance
embedding loop out of a *different* buffer. Both of those first-attempt
readings were false positives for detection, and are recorded here because
the corrected result is the opposite of the uncorrected one.

## A correction to the audit's pytest column

The audit's per-file `pytest line` column and its `union line` column are
identical for all ten rows. An independent re-measurement, with each tier's
gcovr invocation restricted to its own object directory, produces columns
that differ --- for example `vehicle_cycle.cpp` at 91.6% pytest against 94.0%
union, where the audit records 94.0% for both.

The most likely mechanism is that the audit's second gcovr invocation
resolved both `build/cov` and `build/covpy`, so its "pytest tier" figure is
in fact the union of both tiers. This is an inference from the numbers; it
was not reproduced at `5fe1351`. The audit's **doctest** column is unaffected
(a merged run would have reported `vehicle_cycle.cpp` at 94.0%, not 53.4%),
and the doctest column is the one that bounds sanitizer reach, so the audit's
central argument stands unchanged.

The re-measured pytest tier alone, at the current commit, is **88.1% line and
51.6% branch** over the ten files including `bindings/module.cpp`. The Python
tier result on Linux is 985 passed, 1 failed, 3 skipped --- the same
`test_sim.py:586` flush-timing failure the audit recorded as an incidental
finding, unchanged and unrelated to this work.

## Not closed

**C-2 is closed as coverage but leaves a defect open.** Writing the test
surfaced an inconsistency the audit did not report:
`compute_error_state` reads **four** quaternion slots at an attitude block's
offset (`cpp/src/gnc/component.cpp:194`), while `error_block_size` reports
**three** slots for `kRotationVectorLocal` and `kRotationVectorGlobal`
(`:116-118`), which is the width `validate_error_layout` tiles the state
vector with (`:158`, `:177`). A layout that passes validation therefore
cannot supply the fourth slot from within its own block: if the
rotation-vector block is last, the read is one `double` past the end of the
loop's `x_hat_buf`; if another block follows, the quaternion's `z` component
is read out of the neighbouring block.

There is no reading of the current code under which the two forms work. If
the state's attitude block is three slots, there is no absolute attitude in
the state vector to difference truth against. If it is four, the error vector
has a slot nothing writes, and `nav.err` is declared with the same dimension
as `nav.est`, so that slot logs as zero --- which the design explicitly
forbids, on the grounds that zero in an error channel reads as "no error"
rather than "not known".

Choosing between those is a design decision with a log-format consequence,
not a test. The added case therefore pins the arithmetic
`compute_error_state` performs on the quaternion it is given, calling it
directly rather than through `validate_error_layout`, and says so in its own
comment. The dead lines are now executed and their reduction is verified; the
layout inconsistency is recorded here and is unresolved. No built-in
component selects either form today (`ekf.cpp:248` uses `kQuatErrorLocal` and
`builtin.cpp:164` uses `kQuatDifferenceAligned`), so nothing in the suite
reaches it --- but both forms are public API a plugin author can select, and
`docs/gnc_plugins.md:204` documents them.

> **Resolved since: the two forms were removed.** The defect this section
> records was decided in favour of removal rather than repair. The
> consequence was first reproduced live — two probe navigators flown through
> `star run --gnc-plugin` both passed `validate_error_layout` and ran to
> completion with well-formed `nav.err`, while the attitude-block-last one
> read a `double` past `x_hat_buf`; the value came back as exactly 0.0 on
> three byte-identical runs, so the defect was silent and stable. Removal
> was chosen over repair because the reduction is not a lost capability:
> `docs/formats/srlog_v1.md` already defines and applies
> `dtheta = 2 sgn(dq_w) dq_v` in the consistency evaluator, naming the
> built-in EKF's `n = 16` / `m = 15` as exactly that case. What removal does
> **not** do is serve a three-parameter attitude state (MRP/Gibbs) — that
> remains unserved, because `attitude_error` needs a `q_est` such a state
> does not publish. See `docs/gnc_plugins.md` and the `ErrorForm` comment in
> `cpp/include/star/gnc/component.hpp`.

**Deliberately left, per the audit's own recommendation.** R-1 through R-4
(beyond the one `reject_unknown` case per parser, which is added), R-7, R-8,
and everything under UNREACHABLE-BY-DESIGN. R-5 (the error-layout
validator's rejection paths) and R-6 (the JSON escaper) were recommendations
5 and 8 and are not closed here.

**Not measured.** MSVC coverage, aarch64, and sanitizer coverage of the
Python tier remain as the audit left them.
