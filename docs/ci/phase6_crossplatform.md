# Phase 6 cross-platform build and sanitizer record

Every line of Phase 6 C++ had, before this record, been compiled by MSVC in
the Release configuration only. This document records the first pass of
that code through the Linux/GCC warnings-as-errors leg, the AddressSanitizer
and UndefinedBehaviorSanitizer legs, and the MSVC `/W4 /WX` leg, and states
which claims are now measured and which remain unverified.

Source under test: branch `phase-6-gnc-sensors` at
`242d92d38064fae152f813de26ac912b6b58339d`, working tree clean.

## Headline results

| Leg | Platform / toolchain | Result |
|---|---|---|
| CI `cpp-tests` replication (`-Wall -Wextra -Werror`) | Ubuntu 24.04.4, GCC 13.3.0 | **FAIL to build** — 16 errors, one site |
| Doctest suite, warnings not fatal | Ubuntu 24.04.4, GCC 13.3.0 | **PASS** — 162 cases, 56,537 assertions |
| `asan` preset | Ubuntu 24.04.4, Clang 18.1.3 | **PASS**, zero findings — but see coverage caveat |
| ASan + UBSan at `-O2` | Ubuntu 24.04.4, GCC 13.3.0 | **PASS**, zero findings — but see coverage caveat |
| `ci` preset (`/W4 /WX`) | Windows 11, MSVC 14.44.35207 | **FAIL to build** — 4 warnings, one file |
| `pi5` preset | not run | no aarch64 target available |

Two legs fail to build. Neither failure is a memory-safety defect; both are
diagnostics that the MSVC Release-only history could not surface.

Both have since been closed, and one claim made below about `-Wshadow` has
been superseded by measurement. This section records the state at the commit
named above and is left as that snapshot; see "Resolution at `1c717d1`" at the
end of this document for the current state.

The sanitizer legs are clean, but a mutation test performed here shows they
are clean partly because they do not reach the code that motivated them. The
coverage findings are the more consequential result of this pass and are
recorded in full below.

## Toolchains

**Linux.** Ubuntu 24.04.4 LTS under WSL2, GCC 13.3.0
(`Ubuntu 13.3.0-6ubuntu2~24.04.1`), Clang 18.1.3, glibc 2.39, CMake 3.28.3,
GNU Make. The CI `cpp-tests` job runs on `ubuntu-24.04`, whose default
compiler is the same GCC 13.3.0, so this leg is a faithful proxy for that
job rather than an approximation of it.

**Windows.** Windows 11 26200, MSVC toolset 14.44.35207 (Visual Studio 2022
Build Tools), CMake 4.2.1, Visual Studio 17 2022 generator.

The Linux builds were performed in a `git clone` of the repository placed in
the WSL filesystem at `/home/hoyer/sr`, not on `/mnt/c`. A previous
cross-filesystem build of this project (`build/wsl-ci`, retained in the tree)
emitted `Clock skew detected. Your build may be incomplete.`, which is the
documented reason the in-WSL tree was used here. The clone's `HEAD` was
verified equal to the main tree's `HEAD` before building.

All builds ran with `CMAKE_BUILD_PARALLEL_LEVEL=2` and `--parallel 2`, one
build at a time, never a Windows and a WSL build concurrently.

## Proof that the builds were real

A green result is worth nothing if the build was a no-op against a stale
artifact. Each Linux configuration was built from an empty binary directory
and produced its own object set and its own linked binary:

| Build directory | Compiler invocations | Object files | `star_tests` size |
|---|---|---|---|
| `build/inv` (GCC `-O2`) | 61 | 61 | 2,635,296 B |
| `build/asan` (Clang ASan) | 61 | 61 | 86,478,544 B |
| `build/gccasan` (GCC ASan+UBSan) | 61 | 61 | 35,306,568 B |

The counts are `Building CXX object` lines in the build logs and `*.o` files
on disk. The three binaries differ in size by more than an order of
magnitude, which is inconsistent with any of them being a stale copy of
another.

The suite completes in 0.11 s under `ctest`. That is fast enough to resemble
a test binary that ran nothing, so the binary was additionally invoked
directly rather than through CTest; it reports its full case and assertion
counts in the same 0.11 s. The timing is genuine.

## Leg 1 — Linux, warnings as errors

The CI job at `.github/workflows/ci.yml:161` was replicated exactly:

```
cmake -S . -B build/ci -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-Wall -Wextra -Werror"
cmake --build build/ci --parallel 2
```

(The job uses `--parallel 4`; 2 was substituted under this machine's
build-concurrency limit. Parallelism does not affect which diagnostics are
emitted.)

**The build fails.** Sixteen errors are emitted, all of category
`-Werror=array-bounds=`, and all sixteen originate from a single source
line:

```
cpp/src/gnc/ekf.cpp:388:40
  const Eigen::Matrix<double, M, kM> kt = ldlt.solve(pht.transpose());
```

The errors are reported inside `/usr/lib/gcc/x86_64-linux-gnu/13/include/
emmintrin.h` (lines 134 and 176) and `/usr/include/c++/13/bits/move.h`
(lines 198 and 199), with `ekf.cpp:388` named as the inlining root. GCC
reports accesses at offsets 120, 136, 152, 168, 184, 200, 216, and 232 into
the local `kt`, whose size is 120 bytes — that is, eight two-double packets
past the end of the object.

The failing instantiation is `joseph_update<M = 1>`, reached only from the
altimeter update at `cpp/src/gnc/ekf.cpp:531`. The `M = 6` (nav fix) and
`M = 3` (star tracker) instantiations are not diagnosed. For `M = 1` the
destination type is `Eigen::Matrix<double, 1, 15>`, which Eigen requires to
be row-major; the diagnostic arises inside Eigen's
`LDLT::_solve_impl_transposed`, in the unrolled, SSE-vectorized row swap of
`Transpositions<1,1,int>` applied to that single-row destination.

**Attribution: new to Phase 6.** `cpp/src/gnc/ekf.cpp` does not exist on
`main` (`git cat-file -e main:cpp/src/gnc/ekf.cpp` fails), so this
diagnostic cannot be pre-existing.

**Consequence.** Because the CI runner and this host carry the same compiler
version, the `cpp-tests` job will fail on this branch as it stands.

### Minimal reproduction and trigger

Compiling `ekf.cpp` alone at `-O1`, `-O2`, or `-O3` with `-Wall -Wextra`
produces **no** warning. The diagnostic depends on the full flag set CMake
applies, recorded in `build/inv/CMakeFiles/star_core.dir/flags.make`:

```
-Wall -Wextra -O3 -DNDEBUG -std=c++17 -fPIC -O2 \
-fno-fast-math -ffp-contract=off -frounding-math
```

Bisecting that set isolates a single necessary trigger:

| Flag set | `-Warray-bounds` count |
|---|---|
| exact CMake flags | 16 |
| minus `-DNDEBUG` | **0** |
| minus `-fPIC` | 16 |
| minus leading `-O3` | 16 |
| minus `-frounding-math` | 16 |

`-DNDEBUG` compiles out Eigen's `eigen_assert` bounds checks, which is what
permits GCC to unroll and vectorize the swap and then mis-model its trip
count. Removing any other flag leaves the diagnostic intact.

### Adjudication: the diagnostic is a false positive

The warning was not accepted or dismissed on inspection. It was tested.

First, coverage. Under a `--coverage -O0` build, `gcov` on `ekf.cpp` reports:

| Call site | Line | Executions |
|---|---|---|
| `joseph_update<6>` (nav fix) | 451 | 21 |
| `joseph_update<3>` (star tracker) | 491 | 21 |
| `joseph_update<1>` (altimeter) | 531 | **`#####` — never executed** |

No test in the committed suite sets `altimeter.fresh`. The altimeter is
configured in the shared EKF fixture (`cpp/tests/test_ekf.cpp:79-82`) but no
altimeter sample is ever fed, so the warned code path never runs. **A
sanitizer run over the committed suite therefore says nothing whatsoever
about this diagnostic.** That is a coverage finding in its own right and is
recorded separately below.

Second, to obtain evidence rather than inference, a scratch probe was
written that drives the altimeter update through the public
`IGncComponent` interface over eight cycles, causing `joseph_update<1>` to
execute (confirmed by a non-empty innovation record). The probe is scratch
apparatus and is not committed. It was linked against four independently
built configurations of the core:

| Configuration | Sanitizer findings | Reported `pos_x` |
|---|---|---|
| GCC `-O2`, no sanitizer (the warned codegen) | n/a | `7.00002190731299575e+06` |
| GCC `-O2` + ASan + UBSan, `-fno-sanitize-recover=all` | none | `7.00002190731299575e+06` |
| Clang `-O2` + ASan | none | `7.00002190731299575e+06` |
| GCC `-O2` + `-fstack-protector-all` | no stack smashing | `7.00002190731299575e+06` |

All four agree to all seventeen printed significant digits, and no sanitizer
reports a read or write outside `kt`. Had the unsanitized `-O2` build really
written 112 bytes past a 120-byte stack local, it would have overwritten the
adjacent locals of `joseph_update` (`k`, `ikh`, `p_post`, `dx`) and could not
have reproduced the sanitized builds' result bit for bit.

The conclusion is that GCC 13.3.0 mis-models Eigen 3.4.0's unrolled
transposition swap for a row-major 1×15 destination under `-DNDEBUG`. **The
project's code at `ekf.cpp:388` is dimensionally and semantically correct:**
`pht` is 15×M, `pht.transpose()` is M×15, the decomposition is M×M, and the
result is the M×15 `kt` the declaration names.

One caveat is stated plainly: the ASan builds do not emit the diagnostic,
because sanitizer instrumentation suppresses the vectorization that provokes
it, so the ASan runs do not execute the byte-for-byte instruction sequence
GCC warned about. The unsanitized and stack-protector runs do, and they agree
numerically with the sanitized ones. The evidence is convergent rather than
single-source, which is why the verdict is stated as measured.

### What the fix should be

No fix was applied; the diagnosis is the deliverable. Three options, in
descending order of preference:

1. **Narrow, documented suppression.** Wrap the `ldlt.solve` call in
   `#pragma GCC diagnostic push` / `ignored "-Warray-bounds"` / `pop`,
   guarded on `__GNUC__` and not Clang, with a WHY comment citing this
   record and the `-DNDEBUG` trigger. This keeps `-Werror` meaningful
   everywhere else. It is the smallest change that restores CI.
2. **Restructure the `M = 1` path** so the destination is not a row-major
   single-row matrix. Note the tradeoff: `joseph_update` is deliberately
   templated so "the three sensors cannot drift into three different update
   algebras" (the comment at `ekf.cpp:373-376`), and special-casing `M = 1`
   works against that stated intent.
3. **Relax the flag globally** (`-Wno-error=array-bounds`). Not recommended:
   `-Warray-bounds` is a diagnostic worth keeping fatal, and this would
   disable it for all files to accommodate one known-benign site.

Whichever is chosen, the altimeter coverage gap below should be closed
independently, because it is the reason the defect class could not be
adjudicated from the suite alone.

### The rest of the tree is clean

With `-Werror` relaxed to `-Wall -Wextra`, all 61 translation units compile
and the **only** warning category emitted anywhere in the Phase 6 C++ is the
`-Warray-bounds` site above — 16 warnings, zero others. There are no
unused-parameter, sign-compare, uninitialized, dangling-reference, or
shadowing diagnostics from GCC across the whole tree.

## Leg 2 — the `auto`-holding-an-Eigen-expression question

The standing hazard is an Eigen product bound to `auto`, whose temporary
operands die at the end of the full expression, leaving the `auto` object
reading freed memory. It is benign under one compiler and garbage under
another, which is exactly why MSVC-only testing cannot settle it.

A prior code review enumerated every `auto` in the new and changed C++
sources and reported a negative result: `ekf.cpp` holds exactly two `auto`s,
both `std::map` iterators, and every Eigen intermediate is bound to a named
concrete type. **That claim was tested with compilers and it held.**

- The two `auto`s in `ekf.cpp` are confirmed at lines 36
  (`cfg.vectors.find(key)`) and 113 (a range-`for` over `cfg.vectors`).
  Both are `std::map` accesses. Neither holds an Eigen expression.
- `joseph_update`, the densest Eigen algebra in the phase, binds every
  intermediate — `pht`, `s`, `ldlt`, `kt`, `k`, `ikh`, `p_post` — to a named
  concrete type. No `auto` appears in it.
- Neither GCC 13.3.0 nor Clang 18.1.3 emitted any dangling, use-after-free,
  or lifetime diagnostic anywhere in the tree.
- ASan under two compilers found no use-after-free or use-after-scope over
  the whole suite.

The compile-time evidence and the runtime evidence agree. Note that this
class of defect is generally **not** diagnosable at compile time — it is
caught at runtime by ASan — so the ASan legs, not the `-Werror` leg, are
what carry this result, and they are bounded by coverage (see below).

## Leg 3 — AddressSanitizer and UndefinedBehaviorSanitizer

Recent fixes closed two buffer overruns on the plugin path
(`vehicle_cycle.cpp` and the innovation payload copies). Reverting those
fixes reproduced `0xC0000374` `STATUS_HEAP_CORRUPTION` on Windows, so the
defects were real. The fixing agent predicted that ASan would find nothing
on the fixed code, and explicitly flagged that as a prediction rather than a
measurement. It is now a measurement.

**`asan` preset (Clang 18.1.3, Debug, `-fsanitize=address`).** Built from
empty, 61 objects. Run with
`detect_stack_use_after_return=1`, `strict_string_checks=1`,
`check_initialization_order=1`, `detect_leaks=1`.

Result: 162 cases, 56,537 assertions, all passed. Exit 0. Zero
AddressSanitizer errors, zero leaks reported.

**GCC 13.3.0, `-O2`, `-fsanitize=address,undefined
-fno-sanitize-recover=all`.** This configuration was added because the
`asan` preset pins Clang at `-O0`-style Debug, and the codegen that the
`-Werror` leg complained about is GCC's at `-O2`. Built from empty, 61
objects. Run with `print_stacktrace=1`.

Result: 162 cases, 56,537 assertions, all passed. Exit 0. Zero
AddressSanitizer findings, zero UndefinedBehaviorSanitizer findings. UBSan
was run with `-fno-sanitize-recover=all`, so any undefined-behavior report
would have aborted the process rather than printing and continuing.

**The prediction held.** No sanitizer finding was produced by any
configuration, on the suite or on the targeted altimeter probe.

**But it held vacuously in two places, and that matters more than the green
result.** First, `joseph_update<1>` is never executed by the committed
suite, so the suite's ASan runs could not have observed the one path the
`-Werror` leg flagged; that gap was closed here only by the scratch probe,
which is not committed. Second, and more seriously, the nav.innov consumer
path where the two fixed overruns actually lived is not executed by the C++
suite either. The mutation test below establishes that by measurement.

### The short-`s_upper` mutation, and why ASan cannot see it

One recorded mutation — a short `s_upper` innovation payload — exited 1
without raising, a silent out-of-bounds read. That shape was reproduced here
and the result changes how the clean ASan runs above should be read.

In the disposable WSL clone only, two mutations were applied together: the
producer at `cpp/src/gnc/ekf.cpp:424` was made to emit m(m+1)/2 − 1 packed
entries, and the consumer guard at `cpp/src/vehicle_cycle.cpp:1138` was
neutralized so it could not reject the short payload. The `asan` preset was
rebuilt (exactly 2 translation units recompiled) and the suite rerun.

Result: exit 1, 162 cases with 1 failed, 56,537 assertions with 1 failed,
and **zero AddressSanitizer reports**. The single failure is
`cpp/tests/test_ekf.cpp:362`, `CHECK( innov[0].s_upper.size() == 21u )` —
a producer-side unit test asserting the payload length directly. This
reproduces the recorded shape exactly: exit 1, no raise.

The reason ASan is silent is structural, and `gcov` over the coverage build
establishes it. In `cpp/src/vehicle_cycle.cpp`, the guard at line 1138, the
zero-padding at lines 1148-1151, and the embedding loop at lines 1163-1167 —
the code that performs the read — are all `#####`: **never executed by the
C++ doctest suite.** The short payload was caught by a producer-side length
assertion before any consumer ever read it, so no out-of-bounds access
occurred for ASan to observe.

This has a consequence that must not be understated. The two buffer overruns
the recent fixes closed lived on this consumer path, and the guard at line
1138 was itself the fix. That path is exercised only by the Python tier —
`tests/python/test_ekf_channels.py` drives a real run and asserts on the
`nav.innov` group including its structural padding — against the installed
wheel, which is MSVC-built and carries no sanitizer instrumentation. The
ASan legs in this document were run over the C++ doctest binary, which does
not reach that code at all.

**Therefore: the "ASan will find nothing" prediction held as a literal
statement, but the ASan legs as constituted have no reach into the code
where the defects lived.** The clean ASan result is not evidence that those
fixes are correct, and not evidence that a recurrence would be caught. The
gate has been shown, by mutation, to be blind to this defect class.

To give ASan teeth over this path, either add C++ doctest cases that drive
`VehicleCycle` with a component returning innovations — covering the guard
and the embedding loop, and closing the coverage hole permanently — or build
the extension module with ASan and run the Python tier under it. The first is
cheaper and is the recommended route.

## Leg 4 — the `ci` preset on Windows

```
cmake --preset ci          # MSVC, Release, STAR_WARNINGS_AS_ERRORS=ON -> /W4 /WX
cmake --build build/ci --preset ci
```

**The build fails**, with one `error C2220` ("the following warning is
treated as an error") raised from four `/W4` warnings, all in one file, all
name-shadowing:

| Location | Warning | Text |
|---|---|---|
| `cpp/src/vehicle_cycle.cpp:855:38` | C4457 | declaration of `c` hides function parameter |
| `cpp/src/vehicle_cycle.cpp:864:43` | C4457 | declaration of `c` hides function parameter |
| `cpp/src/vehicle_cycle.cpp:873:41` | C4457 | declaration of `c` hides function parameter |
| `cpp/src/vehicle_cycle.cpp:1163:28` | C4458 | declaration of `i` hides class member |

The three C4457 sites are the local `const sensors::NavFixCfg c`,
`const sensors::StarTrackerCfg c`, and `const sensors::AltimeterCfg c` in
the sensor-configuration parse block. The C4458 site is a loop index `i`
shadowing the member `star::VehicleCycle::Impl::i`.

**Attribution: new to Phase 6.** `cpp/src/vehicle_cycle.cpp` does not exist
on `main`, and `git blame` places all four lines in Phase 6 commits
(`cb4a073` for lines 855/864/873, `629b9b7` for line 1163).

**Not a CI blocker.** No CI job builds the `ci` preset on Windows; the only
warnings-as-errors job is `cpp-tests` on Ubuntu, which uses a plain
configure. These four are nonetheless real failures of the project's own
`ci` preset on the platform where the phase was developed.

**Why GCC did not report them.** `-Wshadow` is not implied by `-Wall
-Wextra`, whereas MSVC `/W4` includes C4456-C4459. The two warnings-as-errors
legs are genuinely complementary; neither subsumes the other.

**What the fix should be.** Rename the shadowing locals — the three `c`
declarations to distinct names, and the loop index at line 1163. These are
mechanical and behaviour-preserving. No change was applied here.

## Presets run, and presets not run

| Preset | Windows | Linux | Note |
|---|---|---|---|
| `ci` | **run — fails** | equivalent flags run via the CI job's plain configure — fails | the CI job deliberately does not use the preset |
| `asan` | not run | **run — passes** | preset is gated `hostSystemName == Linux`; MSVC ASan is a different mechanism and is not what the preset describes |
| `release` | previously the project's only leg | covered by `build/inv` | |
| `debug` | not run | not run | adds no diagnostic surface the above legs lack |
| `pi5` | not applicable | **not run** | requires an aarch64 Cortex-A76 target; none available on this host, and `-mcpu=cortex-a76` will not build on x86-64 |

The `ci` preset was not run on Linux under its own name. The CI job it
represents uses a plain configure rather than the preset, by explicit design
noted at `.github/workflows/ci.yml:157-160`, and that plain configure is what
was replicated. The `asan` preset was used under its own name on Linux.

## What is now measured

- The Phase 6 C++ compiles under GCC 13.3.0 on Ubuntu 24.04 with exactly one
  warning category, at exactly one source line, and that line's diagnostic is
  a compiler false positive, established by four independent build
  configurations agreeing bit for bit with no sanitizer finding.
- The full doctest suite passes under Linux/GCC with **162 cases and 56,537
  assertions**, an exact match to the Windows/MSVC baseline of 162 cases and
  56,537 assertions. Case and assertion counts agree across compiler,
  standard library, operating system, and sanitizer configuration.
- The `auto`-holding-an-Eigen-expression review result holds under two
  compilers and two AddressSanitizer configurations.
- ASan and UBSan report nothing on the committed suite, under both Clang and
  GCC, with leak detection and stack-use-after-return enabled and with
  UBSan set to abort rather than recover.
- The altimeter EKF update path (`joseph_update<1>`) is dead at test time.
- The nav.innov consumer path in `cpp/src/vehicle_cycle.cpp` — the guard at
  line 1138 and the embedding loop at lines 1163-1167 — is dead at C++ test
  time, and is reached only through the Python tier against a non-sanitized
  wheel.
- The ASan gate is **blind to the short-`s_upper` defect class** as
  currently constituted. Mutating the payload short and disabling the guard
  produces exit 1 with zero ASan reports, caught only by a producer-side
  length assertion.
- MSVC `/W4 /WX` fails on four name-shadowing warnings in
  `cpp/src/vehicle_cycle.cpp`, all introduced in Phase 6.
- The CI `cpp-tests` job will fail on this branch as it stands.

## What remains unverified

- **Whether ASan would catch a consumer-side overrun if the path were
  executed.** The mutation above shows the C++ suite never reaches the
  embedding loop, so it measures the gate's coverage, not ASan's detection
  power on that code. Whether an instrumented run that actually executes the
  loop with a short payload raises a heap-buffer-overflow is still unmeasured.
  The experiment is: add a C++ test driving `VehicleCycle` with a component
  returning innovations, then repeat the mutation under the `asan` preset.
  Note that `std::vector` overruns within the allocation's capacity are
  invisible to ASan without libstdc++ container annotations, so this
  experiment may return a negative result and should be run before relying on
  ASan for this defect class at all.
- **The `pi5` preset and any aarch64 leg.** Not run; no target hardware. The
  `-mcpu=cortex-a76` build has never been exercised for Phase 6 code.
- **MSVC AddressSanitizer.** Not run. The `0xC0000374` heap corruption that
  motivated the fixes was a Windows-side observation; no Windows sanitizer
  pass confirms the fixed state on that platform.
- **Sanitizer coverage of unexecuted paths generally.** The ASan results
  bound only the code the C++ suite actually runs. Two gaps were found here
  by name — `joseph_update<1>` and the nav.innov consumer block — but both
  were found while chasing specific questions, not by a systematic audit. No
  per-line coverage review of the Phase 6 sources was performed, so further
  unexecuted branches may exist and would be equally outside the sanitizers'
  reach. Given that two of two paths investigated turned out to be dead, such
  an audit is worth doing before the ASan legs are treated as broad evidence.
  **Superseded at `5fe1351`:** that audit was performed and is recorded in
  [`docs/review/phase6_coverage_audit.md`](../review/phase6_coverage_audit.md).
  It measures the doctest tier at 75.5% line and 45.6% branch over the Phase 6
  core surface — the exact bound on what the sanitizer legs can observe — and
  classifies every uncovered path. Four further dead paths are classified
  critical there. The audit also narrows one claim made in this document: the
  nav.innov consumer guards are dead in the C++ tier but are exercised with
  assertions by two Python cases, so they are untested where sanitizers reach
  rather than untested outright.
- **Clang warnings-as-errors.** Clang was used only for the `asan` preset,
  which does not set `-Wall -Wextra -Werror`. Clang's warning surface over
  the Phase 6 code is unmeasured.
- **Determinism across platforms for Phase 6 outputs.** This pass compared
  test counts and one probe value, not logged run output. The FR-30
  cross-platform divergence gate is a separate CI job and was not run here.
  **Superseded at `668b9fc`:** logged bytes were compared across Windows/MSVC
  and Linux/GCC; see "Byte-level output determinism across platforms" at the
  end of this document.

## Resolution at `1c717d1`

Both failing legs are now green and the altimeter coverage gap is closed.
Measured on the same host and the same two toolchains as the pass above.

| Leg | Result at `1c717d1` |
|---|---|
| CI `cpp-tests` replication (`-Wall -Wextra -Werror`), GCC 13.3.0 | **PASS** — clean build from an empty binary directory, zero warnings |
| Doctest suite, Ubuntu 24.04.4 / GCC 13.3.0 | **PASS** — 163 cases, 56,670 assertions |
| `ci` preset (`/W4 /WX`), Windows 11 / MSVC 14.44.35207 | **PASS** — configures and builds clean, zero warnings |
| `ctest --preset release`, Windows 11 / MSVC | **PASS** — 163 cases, 56,670 assertions |

Proof the builds were real: the Linux `-Werror` build emitted 61 `Building CXX
object` lines and left 61 `.o` files from an empty directory, producing
`build/ci/star_tests` at 2,643,136 B, SHA-256 `fef030d2…`. The Windows `ci`
build left 62 `.obj` files and relinked `star_tests.exe` from 1,976,832 B to
1,984,000 B; the `release` binary is 1,984,000 B, SHA-256 `851acfd0…`. Both
binaries were additionally invoked directly rather than only through CTest,
and both report the same 163 cases and 56,670 assertions.

Case and assertion counts move from 162 / 56,537 to 163 / 56,670 on both
toolchains, the difference being the 133 assertions of the new altimeter case.

### The array-bounds diagnostic: mechanism pinned, then suppressed

The false-positive verdict recorded above was re-derived from Eigen's sources
rather than inherited, and the mechanism is now identified exactly rather than
described as a mis-modelled trip count.

The inlining chain GCC prints runs from `ekf.cpp:388` through
`LDLT::_solve_impl_transposed` (`Eigen/src/Cholesky/LDLT.h:610`, the
`dst = m_transpositions * rhs` line) into
`transposition_matrix_product::run` at
`Eigen/src/Core/ProductEvaluators.h:1128`, whose body is
`dst.row(k).swap(dst.row(j))` with `j = tr.coeff(k)`. Both sides of that swap
are `Block<Matrix<double, 1, 15>, 1, 15, true>`.

Two readings of the reported offsets are possible, and the offset pattern
discriminates between them:

- If GCC were continuing row 0's unrolled packets past the end of the object,
  the notes would fall at bytes 128, 144, 160, … — packet subscript 8 begins
  at byte 128.
- If GCC is modelling `j = 1`, the block base sits at byte 120, and that
  row's own seven vector packets fall at 120, 136, 152, 168, 184, 200 and
  216, with its scalar tail element at 232.

The 16 diagnostics report offsets 120, 136, 152, 168, 184, 200, 216 and 232 —
seven packet offsets and a tail, each once as a load (`emmintrin.h:134`) and
once as a store (`emmintrin.h:176`), with the 232 entry reported from
`bits/move.h` at double subscript 29. Only the second reading produces that
set, so GCC is modelling a swap against row 1 of a single-row matrix.

That index cannot be 1. `ldlt_inplace<Lower>::unblocked` returns early on
`size <= 1` having called `transpositions.setIdentity()`, and `setIdentity`
assigns `coeffRef(i) = i`, so `tr.coeff(0) == 0` and the swap is row 0
against itself. Eigen's own guarantee of that bound is the `eigen_assert`
inside `row()`, which `-DNDEBUG` compiles out — which is why `-DNDEBUG` is
the single necessary trigger in the bisection above.

Eigen's unroller also cannot run off the end on its own account:
`dense_assignment_loop<…, LinearVectorizedTraversal, CompleteUnrolling>`
computes `alignedSize = (15/2)*2 = 14` and instantiates
`copy_using_evaluator_innervec_CompleteUnrolling<Kernel, 0, 14>`, which emits
packets at inner indices 0 through 12 and leaves element 14 to the scalar
remainder unrolling. Every access it generates for row 0 is inside the
120-byte object.

**Verdict unchanged, now independently grounded: the diagnostic is a false
positive and the code at that line is correct.** The remedy applied is a
suppression scoped to the single declaration, naming only `-Warray-bounds`
and guarded to GCC proper, with the mechanism above recorded at the site.
`-Warray-bounds` remains fatal everywhere else in the tree. Restructuring the
`M = 1` path was rejected: the destination of `ldlt.solve` must have `M` rows,
so for `M = 1` it is necessarily a single-row matrix, which Eigen requires to
be row-major — the shape cannot be avoided without either special-casing
`M = 1` against the template's stated purpose or replacing the LDLT the
specification pins.

### `joseph_update<1>` is no longer dead

`gcov`, rerun over a `--coverage -O0` build of the same source, now reports
the altimeter call site executed once where it previously reported `#####`:

| Call site | Line | Executions |
|---|---|---|
| `joseph_update<6>` (nav fix) | 470 | 21 |
| `joseph_update<3>` (star tracker) | 510 | 21 |
| `joseph_update<1>` (altimeter) | 550 | **1** |

Line coverage of `ekf.cpp` is 97.61% of 377 lines. The line numbers moved by
19 relative to the table above because of the comment and pragma block added
at the suppression site.

The case that closes the gap,
`ekf_altimeter_update_matches_the_closed_form_scalar_solution` in
`cpp/tests/test_ekf.cpp`, pins the `M = 1` update against an analytic result
rather than a regenerated golden: on the equator along +x under an identity
body-fixed rotation the ellipsoidal normal is exactly `(1, 0, 0)`, so `H`
selects the position-x error state alone, `S` is the sum of the fixture's
50 m position variance and 20 m measurement variance, and the Joseph form
reduces to `P+ = P R / (P + R)`. The correctness of that path no longer rests
on code reading plus an uncommitted scratch probe.

### `-Wshadow`: the two legs are redundant, not complementary

The claim above that "the two warnings-as-errors legs are genuinely
complementary; neither subsumes the other" is **superseded** for the
shadowing class. It was inferred from the two legs' behaviour rather than
measured, and the measurement does not support it:

- The whole tree at `1c717d1` compiles under `-Wall -Wextra -Wshadow` with
  **zero** warnings across all 61 translation units.
- The pre-fix `cpp/src/vehicle_cycle.cpp` (from `9364b88^`), compiled with
  the same flag set plus `-Wshadow`, emits **exactly four** `-Wshadow`
  warnings, at lines 855, 864, 873 and 1163 — the same four sites, at the
  same lines, that MSVC `/W4` reported as C4457 and C4458. GCC names three of
  them "shadows a parameter" and the fourth "shadows a member of
  'star::VehicleCycle::Impl'", matching MSVC's split between C4457 and C4458.

GCC therefore sees these defects; the Linux leg missed them only because
`-Wshadow` is not enabled, not because the diagnostic is outside its reach.
Adding `-Wshadow` to the `cpp-tests` configure at
`.github/workflows/ci.yml:168` would cost zero renames today and would have
caught all four Phase 6 sites on the leg that actually gates CI, rather than
on a Windows preset no CI job builds. The change is recommended; it is not
applied here, because enabling a new fatal diagnostic in CI is a project
policy decision rather than part of closing this build failure.

## Re-run at `0cbb52b` (139 commits of new C++ since the pass above)

The legs above were last measured at `1c717d1`. A large body of C++ landed
afterwards and had, before this run, been compiled by MSVC only: the
pitch-program roll-reference fix (`8f09032`, `models::pitch_program_roll_ref`
and both call sites), the camera-intrinsics header echo and the SRLOG bump to
format 1.3 (header JSON keys carrying IEEE-754 doubles as hex bit patterns),
removal of the `ROTATION_VECTOR_*` error forms and the replacement width
tests, the IMU stochastic bit-identity test and its non-degeneracy companion,
the closed-loop insertion gate (EC-6 and EC-11), the noise-sigma validator,
and the NavFix validity flag. `-Wshadow` had also been added to the CI leg
many commits earlier and had not been re-measured since.

Source under test: branch `phase-6-gnc-sensors` at
`0cbb52bfbcbffed86ccdf1cf444152d04c9e3c9c`, working tree clean. The Linux
clone at `/home/hoyer/sr` was fast-forwarded to that commit and verified clean
before every leg.

### Headline results

| Leg | Platform / toolchain | Result |
|---|---|---|
| CI `cpp-tests` replication (`-Wall -Wextra -Wshadow -Werror`) | Ubuntu 24.04, GCC 13.3.0 | **PASS** - clean build from empty, **zero** diagnostics of any category |
| Doctest suite | Ubuntu 24.04, GCC 13.3.0 | **PASS** - 179 cases, 65,348 assertions |
| ASan + UBSan at `-O2`, **container annotations enabled** | Ubuntu 24.04, GCC 13.3.0 | **PASS** - zero ASan, zero UBSan, zero leak reports |
| Python suite | Ubuntu 24.04, CPython 3.12.3 | **PASS** - 1017 passed, 0 failed, 0 skipped |

Case and assertion counts match the Windows/MSVC baseline of 179 / 65,348
**exactly**, as they have on every previous pass. The Python total matches the
Windows baseline of 1017 exactly once the `[pandas,parquet]` extras are
installed, which is what the CI `build-test` job does on the `ubuntu-24.04`
leg specifically (`.github/workflows/ci.yml:42-43`).

### Toolchain

Ubuntu 24.04 under WSL2, `c++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0`,
CMake 3.28.3, GNU Make, CPython 3.12.3, glibc 2.39. This is the same GCC
version as the `ubuntu-24.04` CI runner, so the leg is a faithful proxy for
the `cpp-tests` job rather than an approximation of it. All builds ran with
`CMAKE_BUILD_PARALLEL_LEVEL=2` and `--parallel 2`, one at a time, never
concurrently with a Windows build. The CI job uses `--parallel 4`;
parallelism does not affect which diagnostics are emitted.

### Method

Each leg was run from a shell script copied into the WSL filesystem and
executed there, rather than as an inline command. This was not a stylistic
choice - see "A methodology defect found and corrected during this run" below.
The scripts capture every exit code inside WSL and print object counts,
compiler invocation counts and binary hashes alongside the result.

### Leg 1 - Linux warnings-as-errors

`.github/workflows/ci.yml:165-177` replicated with its current flag set,
which now includes `-Wshadow`:

```
cmake -S . -B build/ci -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-Wall -Wextra -Wshadow -Werror"
cmake --build build/ci --parallel 2
ctest --test-dir build/ci --output-on-failure
```

Configure returned 0, build returned 0, `ctest` returned 0, and the binary
invoked directly returned 0.

**Zero warnings and zero errors, of any category, across all 61 translation
units.** There is nothing to attribute: no new diagnostic, and no pre-existing
one. Every item in the list of unverified changes at the top of this section
compiles clean under GCC with `-Werror`. `-Wshadow` in particular is
re-confirmed at zero cost at this commit, 139 commits after the measurement
that justified adding it.

The `auto`-holding-an-Eigen-expression hazard - the defect class this leg
exists to catch, benign under one compiler and garbage under another - is
therefore unrefuted by the compile-time leg and, more importantly, unrefuted
by the ASan leg below, which is what actually detects it at runtime.

### Leg 2 - ASan and UBSan, now with container annotations

```
cmake -S . -B build/gccasan -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-O2 -g -fsanitize=address,undefined \
                     -fno-sanitize-recover=all -D_GLIBCXX_SANITIZE_VECTOR=1" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined"
```

Run with `detect_stack_use_after_return=1`, `strict_string_checks=1`,
`check_initialization_order=1`, `detect_leaks=1`, `print_stacktrace=1`, and
UBSan set to abort rather than recover.

Result: exit 0, 179 cases, 65,348 assertions, **zero** AddressSanitizer
reports, **zero** UndefinedBehaviorSanitizer runtime errors, **zero**
LeakSanitizer reports.

**The container-annotation caveat is now closed, and the answer is that
annotations can be enabled here and they work.** Every previous clean ASan
result in this document carried the caveat that libstdc++ container
annotations are inactive under plain `-fsanitize=address` on GCC 13.3.0, so a
`std::vector` overrun *within the allocation's capacity* was invisible; the
buffers that mattered were detected by accident of allocation pattern rather
than by any property of the sanitizer.

This libstdc++ does support them. `bits/stl_vector.h:72` gates the annotation
calls on `_GLIBCXX_SANITIZE_STD_ALLOCATOR && _GLIBCXX_SANITIZE_VECTOR`, and
`bits/c++allocator.h:54-58` already defines the first automatically whenever
`__SANITIZE_ADDRESS__` is set, so only `_GLIBCXX_SANITIZE_VECTOR` needs
supplying. It was supplied, and its presence on the compile line was confirmed
in `build/gccasan/CMakeFiles/star_core.dir/flags.make` rather than assumed.

That the annotations are actually *active* was established by mutation rather
than by inspection. A probe reserves capacity 64, assigns 8 elements, and
writes to index 8 - past `size()`, inside `capacity()`:

| Build | Exit | ASan reports |
|---|---|---|
| `g++ -O2 -fsanitize=address` | 0 | **0 - the overrun is invisible** |
| `g++ -O2 -fsanitize=address -D_GLIBCXX_SANITIZE_VECTOR=1` | 1 | **1 - `container-overflow`** |

The first row reproduces the old caveat exactly; the second retires it. The
suite's clean result above was produced with annotations on, so it now covers
within-capacity `std::vector` overruns as well as ordinary heap and stack
overruns.

**What a clean run here does and does not establish.** It establishes that no
sanitizer finding arises on the code the C++ doctest suite actually executes,
now including within-capacity vector overruns. It does **not** establish that
the tree is free of such defects, because ASan reports only what runs, and the
coverage audit in [`docs/review/phase6_coverage_audit.md`](../review/phase6_coverage_audit.md)
measures the doctest tier at 75.5% line and 45.6% branch over the Phase 6 core
surface. That figure, not the sanitizer, is the binding limit. In particular
the `nav.innov` consumer path in `cpp/src/vehicle_cycle.cpp` remains outside
the C++ tier's reach, so the clean result still says nothing about it. What
has changed is that the *stated blocker* on answering that question - "the
answer may be negative because annotations are unavailable" - is removed: if a
C++ case is written that drives `VehicleCycle` with a component returning
innovations, the annotated build will now detect a within-capacity overrun on
it. The experiment is unblocked, not performed.

### Leg 3 - the `ekf.cpp` array-bounds suppression is still load-bearing

`cpp/src/gnc/ekf.cpp:400-407` carries the one deliberate suppression in the
tree, scoped to a single declaration, naming only `-Warray-bounds`, guarded to
`__GNUC__ && !__clang__`. A tree-wide grep confirms it is the **only**
`diagnostic ignored` or `pragma warning` anywhere under `cpp/`.

It was verified to be still necessary rather than assumed. With the `ignored`
line replaced by a comment and nothing else changed, the build returns 2 and
emits the same **16** `-Werror=array-bounds=` errors at `ekf.cpp:404`, with
`cc1plus: all warnings being treated as errors`, in the
`joseph_update<M = 1>` instantiation described earlier in this document. The
source was restored and the tree re-verified clean afterwards. The suppression
is live, correctly scoped, and not dead code left behind by a compiler
upgrade.

### Leg 4 - the Python tier and `test_sim.py:586`

The wheel was built and installed into a venv inside the WSL filesystem
(`/home/hoyer/sr/.venv`), deliberately not into the Windows environment.

`test_close_releases_the_log_of_an_abandoned_run` asserted that `run.srlog`
was non-empty after two `Sim.step()` calls with no intervening close. It
passed under the MSVC runtime and failed on Linux with a 0-byte file. The
earlier record called the attribution provisional because the failure had been
seen only under an instrumented build.

**It is no longer provisional. The failure reproduces against the optimized
Linux wheel**, and the mechanism was measured directly by stepping a run and
sampling the file size after every step:

| After step | File size |
|---|---|
| 1 | 0 |
| 8 | 8,195 |
| 21 | 16,391 |
| 33 | 24,589 |
| 45 | 32,785 |

Bytes reach disk in 8,192-byte units - the libstdc++ `filebuf` buffer - with
nothing on disk at all until step 8. This is stream buffering and not a
defect. The MSVC runtime happens to have surrendered bytes earlier, which is
the only reason the assertion ever passed.

**Adjudication: the flush timing is not contractual and should not become
so.** Three things point the same way. The project already has an explicit
contract that says the opposite of what the test asserted -
`cpp/include/star/vehicle_cycle.hpp:110-112` describes `close()` as the flush,
and the C++ sibling case at `cpp/tests/test_gnc_cycle.cpp:736` measures the
abandoned prefix *after* `close()` for exactly this reason. The format has no
consumer for mid-run bytes: `docs/formats/srlog_v1.md` section 8 already makes
a trailing partial record `SrlogCorruptError`, so reading a log that is still
being written is outside the contract regardless. And per-record flushing
would trade one buffered write per 8 KB for a syscall per record on the core's
hot loop, buying a durability property no reader is permitted to use.

The resolution was therefore to write the contract down where a reader of the
format will find it, and to assert it at the point it holds - not to delete
the assertion:

- `docs/formats/srlog_v1.md` gains a normative section 5.1 stating that the
  file is guaranteed complete only after `close()`, that its size before then
  is unspecified and may legitimately be 0, that no code may use a mid-run
  size as a progress or liveness signal, and why per-record flushing is not
  offered.
- `tests/python/test_sim.py` now asserts that the log **exists** while the run
  is open - which is the handle lifetime the case is actually about - and that
  its size is nonzero **after** `close()`. The test still fails if `close()`
  stops flushing; it no longer fails because of a standard library's buffer
  size.

Full Python suite on Linux after the change: **1017 passed, 0 failed, 0
skipped**, exit 0. Without the `[pandas,parquet]` extras the same suite is
1010 passed / 3 skipped; the three skips are `pyarrow` and `pandas` import
guards in `test_export_parquet.py`, `test_export_cli_formats.py` and
`test_to_pandas.py`. The extras were installed so the suite runs with no
skips, matching what CI does on this leg. There are no platform-conditional
skips in the Python tier - a grep for `sys.platform`, `platform.system` and
`skipif` finds no selection outside those three extras guards - so the
earlier report of a platform-dependent Python total is fully explained by the
extras and by the now-fixed failure.

Note on provenance: the installed extension module reports
`git_hash = a742f8dâ€¦`, the commit the wheel was built from. The subsequent
commit changed only `docs/formats/srlog_v1.md` and `tests/python/test_sim.py`
and no C++ or installed Python source, so the wheel under test is the correct
binary for the source that produced these results.

### Proof that the builds were real

| Build directory | Compiler invocations | `.o` on disk | `star_tests` size | SHA-256 |
|---|---|---|---|---|
| `build/ci` (GCC `-Werror`, Release) | 61 | 61 | 2,746,416 B | `ee5ae01c4863â€¦` |
| `build/gccasan` (GCC ASan+UBSan `-O2 -g`) | 61 | 61 | 170,912,408 B | `3e9c92ddcd1aâ€¦` |

Both were configured and built from a removed binary directory, so neither
count can be a no-op. The two binaries differ by a factor of 62 in size, which
is inconsistent with either being a stale copy of the other.

Two further pieces of evidence, because a count can be produced by a build
that recompiles but emits nothing new:

- The `build/ci` binary hash **changed** between the `a742f8d` build
  (`f9263814e464â€¦`) and the `0cbb52b` build (`ee5ae01c4863â€¦`) even though the
  commit in between touched only a Markdown file and a Python test. The cause
  is `cpp/src/version.cpp.in`, which bakes `STAR_GIT_HASH` in at configure
  time. The binary demonstrably tracks the commit it was built from.
- After the suppression probe removed a pragma, rebuilt, restored the source
  and rebuilt again, `build/ci/star_tests` hashed back to `ee5ae01c4863â€¦` -
  bit-identical to the pre-probe binary. The build is reproducible and the
  probe left nothing behind.

The suite completes in 0.12 s under `ctest`, which is fast enough to resemble
a binary that ran nothing, so as in the earlier passes the binary was also
invoked directly; it reports 179 cases and 65,348 assertions in the same
0.12 s. The timing is genuine.

### A methodology defect found and corrected during this run

This is recorded because it invalidated intermediate readings before it was
caught, and because it would silently affect any future agent driving WSL the
same way.

Exit codes were initially read with an inline
`wsl -d Ubuntu -- bash -lc '... ; echo "RC=$?"'`. **`$?` in that position is
expanded by the outer Windows-side shell before the string ever reaches WSL,
so it reports the outer shell's status, not the build's.** The defect was
caught when a suppression probe printed `BUILD_EXIT=0` while the same log
ended in `gmake: *** [Makefile:101: all] Error 2`; a follow-up using an
intermediate variable printed an empty string, and a direct control -
`bash -lc 'false; echo $?'` - printed `0`, confirming the mechanism.

Escaping as `\$?` reports correctly, and running from a script file copied
into WSL avoids the class entirely. Every result in this section was
re-measured with the script method after the discovery; the pre-discovery
readings are not relied upon anywhere above. No conclusion changed as a
result, because the substantive evidence in each leg was the log *content*
(diagnostic counts, doctest totals, sanitizer report counts) rather than the
exit code - but the exit codes are now independently correct as well.

This is the same failure mode as the recorded stale-artifact-after-masked-
build-failure lesson, arriving through a different door: there, a pipe
swallowed an installer's status; here, an outer shell answered the question
before the inner one could.

### What is now measured at `0cbb52b`

- All 139 commits of C++ added since `1c717d1`, including every item listed at
  the top of this section, compile under GCC 13.3.0 with
  `-Wall -Wextra -Wshadow -Werror` with **zero diagnostics of any category**
  across 61 translation units.
- `-Wshadow` remains zero-cost on this tree at this commit.
- The doctest suite passes on Linux/GCC with **179 cases and 65,348
  assertions**, an exact match to the Windows/MSVC baseline.
- ASan, UBSan and LeakSanitizer report nothing over the suite **with
  libstdc++ container annotations enabled**, and the annotations were proven
  active by a mutation that they catch and that plain ASan misses.
- The `ekf.cpp` `-Warray-bounds` suppression is still load-bearing: removing
  it reproduces 16 errors and fails the build. It is the only suppression in
  the tree.
- The `test_sim.py:586` failure is stream buffering, measured at 8,192-byte
  granularity against an **optimized** Linux wheel, not an instrumentation
  artifact.
- The Python suite passes on Linux with 1017 passed, 0 failed and 0 skipped,
  matching the Windows total exactly.

### What remains unverified

- **Whether ASan detects a consumer-side `nav.innov` overrun when the path
  executes.** Still not measured. The path is still dead in the C++ tier, so
  the experiment still requires a doctest case driving `VehicleCycle` with a
  component returning innovations. The annotation blocker on that experiment
  is now removed, which is the only part of this item that changed.
- **Sanitizer reach generally.** Bounded by the doctest tier's 75.5% line and
  45.6% branch coverage over the Phase 6 core surface. A clean sanitizer run
  is evidence about executed code only.
- **Clang warnings-as-errors.** Clang was not used at all in this pass. Its
  warning surface over the current tree is unmeasured.
- **aarch64 and the `pi5` preset.** Not run; no target hardware.
- **MSVC AddressSanitizer.** Not run.
- **Cross-platform output determinism for the new Phase 6 code.** This pass
  compared test counts, not logged run bytes. The FR-30 divergence gate is a
  separate CI job and was not exercised here; the SRLOG 1.3 header echo, which
  carries doubles as hex bit patterns, has not been compared across platforms.
  **Superseded at `668b9fc`:** both were measured; the header echo crosses
  byte-identically and the record streams diverge at libm magnitude. See the
  final section of this document.
- **macOS.** No leg. Two of the four CI `build-test` legs (macOS and ARM) have
  no local proxy.

## Byte-level output determinism across platforms, at `668b9fc`

Every cross-platform pass recorded above compared **test counts**. None
compared **logged bytes**. FR-21's determinism contract - "the same inputs on
the same binary always produce bit-identical outputs" - was therefore verified
within a platform and never across platforms, and the SRLOG 1.3 camera header
echo, which carries IEEE-754 doubles as 16-hex-digit bit patterns precisely so
that header floats are portable, had never been tested against the thing it was
designed for.

This section is the first byte-for-byte comparison of `run.srlog` files
produced by Windows/MSVC and Linux/GCC.

Source under test: branch `phase-6-gnc-sensors` at
`668b9fc196246bb1f8964ad950bfb08b8ead14e9`, working tree clean on both hosts.

### Headline result

| Mission | File bytes | Header | Record stream | Whole file |
|---|---|---|---|---|
| `missions/twobody_leo.toml` | 6,589,151 | identical | identical | **BYTE-IDENTICAL** |
| `missions/leo_attitude_gnc.toml` | 399,756 | identical | differs | differs |
| `missions/leo_ekf_consistency.toml` | 979,081 | identical | differs | differs |
| `missions/ascent_leo_gnc.toml` | 2,653,436 | identical | differs | differs |
| camera-enabled optical mission | 437,965 | identical | differs | differs |

Four findings, in descending order of consequence:

1. **All five headers are byte-identical, with zero excluded fields** - and
   that includes the v1.3 `gnc.camera` hex echo. The header's portability
   design works.
2. **The two-body reference mission is byte-identical end to end**, whole file,
   no exclusions. Arithmetic restricted to IEEE-754 basic operations crosses
   the platform boundary exactly, as the project's recorded position predicts.
3. **The four Phase 6 GNC missions differ in the record stream only**, at
   last-bit magnitude. File sizes and structure are identical to the byte; only
   float payload values move.
4. **The divergence is attributable to libm**, by measurement rather than
   inference (see "Attribution" below). Within each mission the split is
   clean: every `t_s` channel, every integer and flag channel, `mass.*`, and -
   in all three orbital missions - `truth.r_m` and `truth.v_mps` are
   bit-identical, while the rotational, sensor, and GNC channels differ.

### Missions chosen, and why

Five missions, each earning its place:

- **`missions/twobody_leo.toml` - the control.** The Phase 1 byte-frozen
  two-body path, whose arithmetic is add/subtract/multiply/divide/sqrt only.
  It is the mission the existing FR-30 `cross-platform-divergence` CI job
  already measures at `max_rel = 0`. Including it tests the *method*: if this
  mission had differed, the finding would have been a defect in this
  experiment, not in the simulator.
- **The camera-enabled optical mission** - the SRLOG 1.3 header echo under
  test, and the only configuration that emits a `gnc.camera` header object and
  a `sensors.camera` group. It is the mission text built by
  `tests/python/test_p6_optical_gates.py` (`_OPTICAL_GATE_MISSION`), extracted
  verbatim and written to a file whose SHA-256 was confirmed equal on both
  hosts (`e28cd441cc90b1a18c842b0ec6ba2d3a111378a937a341241aec18fbe4f5ed45`)
  before either run. It is not a committed mission because no committed mission
  declares a camera.
- **`missions/leo_ekf_consistency.toml`** - the widest numerical surface in the
  phase: the built-in error-state EKF with aiding updates from a nav fix, a
  star tracker, and an altimeter, logging `nav.est.P` (72,120 covariance
  entries), `nav.innov.y` and `nav.innov.S`.
- **`missions/ascent_leo_gnc.toml`** - the guidance path. `pitch_program`
  guidance drives the `sin`/`cos` roll reference added in Phase 6, and the
  760 s powered ascent additionally exercises the atmosphere and aero models
  (`exp`, `pow`, table interpolation) over 7,600 integration steps. It is the
  longest error-accumulation chain available.
- **`missions/leo_attitude_gnc.toml`** - the reference GNC mission: closed-loop
  PD attitude control with an IMU and dead-reckoning navigation, and no
  ephemeris, aero, or EKF. It isolates the attitude loop from the ascent
  mission's environment models.

### Excluded fields: none

**No field was excluded from any comparison. The files were compared whole.**

Two fields legitimately differ by build and would ordinarily have to be
excluded. Both were instead made equal, which is a stronger result than
excluding them:

- **`producer.git_hash`** is baked in at configure time from
  `cpp/src/version.cpp.in`. Rather than exclude it, the Windows wheel was
  rebuilt so that both trees sit at `668b9fc`, and both cores report
  `git_hash = 668b9fc196246bb1f8964ad950bfb08b8ead14e9`. The field is
  *compared*, and it matches.
- **`producer.core_version`** is `0.6.0` on both, from the same
  `pyproject.toml`.

A third field was checked rather than assumed:

- **`config_sha256`** is the SHA-256 of the FR-15 canonical resolved config,
  computed host-side in Python. A leaked host path would change it. Every
  path-like string in all five resolved configs was enumerated
  (`vehicle.path`, `environment.ephemeris`) and all are repo-relative with
  forward slashes; no absolute path and no backslash appears. The digests
  match on all five missions.

`meta.json` was not compared and is not part of this result: it records wall
time and a start timestamp and is nondeterministic by construction. FR-21's
contract is about `run.srlog`.

### Method

Both legs ran the identical script (`xplat_run.py`, scratch apparatus, not
committed) from the repository root, invoking `star_reacher.runner.run_mission`
directly and hashing the emitted `run.srlog`. The Linux logs were then copied
to the Windows host and both sides parsed by the **same** loader on the **same**
host, so the comparison measures the bytes the two cores wrote rather than two
readers' behaviour. The transfer was verified lossless by re-hashing every file
on arrival and matching the hash computed in WSL.

Localization descends file to header/stream, then group, channel, and record,
reporting differing-entry counts, maximum absolute difference, and ULP
distance. Pointwise relative error is reported only where it is meaningful: a
channel whose values pass through zero produces `rel = 2.0` from a sign flip at
1e-16, which characterizes nothing, so the tables below normalize by the
channel's own RMS instead.

**Within-platform controls.** Each mission was run twice on each platform. All
five hashes reproduced exactly on Windows and all five reproduced exactly on
Linux. The differences reported here are therefore cross-platform, not
run-to-run, and FR-21's within-platform clause is re-confirmed on both hosts as
a by-product.

### Toolchains

**Windows.** Windows 11 10.0.26200, MSVC 19.44.35222 (x64), CMake 4.2.1,
Visual Studio 17 2022 generator, CPython 3.14.0. Compile flags confirmed from
`build/skbuild-cp314-cp314-win_amd64/star_core.dir/Release/star_core.tlog/CL.command.1.tlog`
rather than assumed: `/O2 /Ob2 /EHsc /MD /fp:strict /std:c++17`, on 34 command
entries.

**Linux.** Ubuntu 24.04.4 LTS under WSL2, GCC 13.3.0
(`Ubuntu 13.3.0-6ubuntu2~24.04.1`), glibc 2.39, CMake 3.28.3, CPython 3.12.3.
Compile flags confirmed from
`build/skbuild-cp312-cp312-linux_x86_64/CMakeFiles/star_core.dir/flags.make`:
`-O3 -DNDEBUG -std=c++17 -fPIC -O2 -fno-fast-math -ffp-contract=off
-frounding-math`.

Both flag sets disable FMA contraction and fast-math, so the IEEE-754 basic
operations are correctly rounded on both toolchains. That is what makes the
libm attribution below the only remaining candidate mechanism.

The Linux build was performed in a `git clone` in the WSL filesystem at
`/home/hoyer/sr`, not on `/mnt/c`. All builds ran with
`CMAKE_BUILD_PARALLEL_LEVEL=2`, one at a time, never a Windows and a WSL build
concurrently.

### Proof that both builds were real

Both wheels were rebuilt from a **removed** scikit-build binary directory at
the commit under test:

| Leg | Objects before | Objects after | Extension module | SHA-256 |
|---|---|---|---|---|
| Windows / MSVC | 0 | 36 `.obj` | `_core.cp314-win_amd64.pyd`, 1,142,784 B | `a21424e378c9c828...` |
| Linux / GCC | 0 | 35 `.o` | `_core.cpython-312-x86_64-linux-gnu.so`, 1,530,912 B | `b09d9e1fe2bfc6ff...` |

The one-object difference is the MSVC generator's `CMakeCXXCompilerId.obj`
probe; both legs compile the same 35 translation units.

Two further pieces of evidence, because an object count can be produced by a
build that recompiles and changes nothing:

- The Linux extension module hash **changed** from `cf6083b4fa231cc1...` to
  `b09d9e1fe2bfc6ff...` across the rebuild, and its reported `git_hash` moved
  from `a742f8d74ae5f210...` to `668b9fc196246bb1...`. The binary demonstrably
  tracks the commit it was built from.
- The Windows core's reported `git_hash` likewise moved from
  `b24f8c9639d38d11...` to `668b9fc196246bb1...`.

**A masked build failure was caught during this pass and is recorded because it
would have invalidated the result silently.** The first WSL build attempt
returned exit 2 with `ModuleNotFoundError: No module named 'scikit_build_core'`
- `--no-build-isolation` requires the build backend in the target venv, and
that venv had only NumPy. The installed `.so` was left untouched at
`cf6083b4fa231cc1...` / `a742f8d`, 11 commits behind the tree it was about to
be compared against (`git rev-list --count a742f8d..668b9fc`). The script
printed `PIP_EXIT=2`, `POST_O=0` and the stale
hash side by side, so the failure was visible rather than inferred. Had the
script reported only that the runs completed, the entire comparison would have
been made against a stale artifact. This is the
stale-artifact-after-masked-build-failure lesson arriving through a third door:
here it was neither a pipe nor an outer shell, but a missing build dependency
under a flag that suppresses provisioning.

### Result 1 - the header, including the v1.3 camera echo, crosses exactly

All five headers are byte-identical. For the camera mission the `gnc.camera`
object is character-for-character equal on both platforms:

```json
{"float_encoding":"ieee754-binary64-hex","width_px":1024,"height_px":768,
 "fx_px":"4089000000000000","fy_px":"4082c00000000000",
 "cx_px":"407e600000000000","cy_px":"4079200000000000",
 "q_b2c":["3fe6a09e667f3bcd","3fe6a09e667f3bcd","0000000000000000",
          "0000000000000000"],
 "r_cam_b_m":["3fe0000000000000","bfd0000000000000","3fc0000000000000"]}
```

(shown wrapped; the file carries it inside the single compact header line.)

**The `ieee754-binary64-hex` encoding did exactly what section 3 of
`docs/formats/srlog_v1.md` claims for it.** The design argument was that a
pure-integer shift-and-nibble-lookup encoder admits no float formatter,
rounding mode, or locale that could perturb the header bytes. That argument is
now measured rather than asserted, on the one artifact in the format that
carries doubles.

This is a narrower claim than it may appear, and the narrowness should be
stated. The echoed values are configuration constants that the core copies
from the resolved config without arithmetic, so what is established is that
the **encoder and the header serializer** are portable - not that a computed
double would survive. That is nonetheless the property the encoding was
designed to provide.

### Result 2 - the two-body control is byte-identical

`missions/twobody_leo.toml` produces the identical 6,589,151-byte file on both
platforms, SHA-256
`0897fa314e47a0902f89261f4548e569016b6c432f33c7a7869ff906690ba568`, whole file,
no exclusions.

This independently corroborates the existing FR-30 record in
`tests/golden/determinism/cross_platform.toml`, which measures `max_rel = 0` on
this mission across four CI legs - but at a strictly stronger level. That gate
compares **final-state truth records**; this compares **every byte of the
file**, including all 5,400 s of intermediate records and the header.

### Result 3 - where the four GNC missions differ

Every differing file has the **identical byte length** and identical group and
channel structure. Only float payloads move.

Divergence is normalized by each channel's RMS, and only the largest few
channels per mission are listed. "Identical" counts channels that are equal to
the byte.

**`missions/leo_attitude_gnc.toml`** - first differing byte at offset 8,681;
5,116 of 399,756 bytes differ. 30 channels bit-identical.

| Channel | max abs diff | RMS scale | rel to scale |
|---|---|---|---|
| `nav.err.e` | 5.551e-16 | 9.722e-06 | 5.71e-11 |
| `gnc.cmd.tau_b_nm` | 9.107e-17 | 3.267e-03 | 2.79e-14 |
| `truth.w_b_radps` | 3.253e-17 | 2.168e-03 | 1.50e-14 |
| `sensors.imu.dtheta_b_rad` | 3.253e-18 | 2.170e-04 | 1.50e-14 |
| `truth.q_i2b` | 3.331e-16 | 5.000e-01 | 6.66e-16 |

**Camera-enabled optical mission** - first differing byte at offset 10,284;
12,593 of 437,965 bytes differ. 37 channels bit-identical.

| Channel | max abs diff | RMS scale | rel to scale |
|---|---|---|---|
| `nav.err.e` | 1.110e-16 | 1.006e-05 | 1.10e-11 |
| `gnc.cmd.tau_b_nm` | 4.510e-17 | 3.170e-03 | 1.42e-14 |
| `sensors.camera.px_uv` | 5.684e-13 px | 3.632e+02 px | 1.57e-15 |
| `sensors.startracker.q_meas_i2b` | 1.110e-16 | 5.000e-01 | 2.22e-16 |
| `sensors.sunsensor.sun_b` | 2.220e-16 | 5.774e-01 | 3.85e-16 |

The logged landmark pixels differ by at most **5.7e-13 pixels**, against the
`ch:camera` exit-criterion-7 tolerance of 1e-6 pixels - seven orders of
magnitude inside it. `sensors.camera.t_s` and `sensors.camera.r_m` are
bit-identical; only `q_i2b` and `px_uv` move.

**`missions/leo_ekf_consistency.toml`** - first differing byte at offset 5,546;
35,459 of 979,081 bytes differ. 36 channels bit-identical.

| Channel | max abs diff | RMS scale | rel to scale |
|---|---|---|---|
| `sensors.altimeter.alt_meas_m` | 9.313e-10 m | 6.219e+05 m | 1.50e-15 |
| `env.alt_m` | 9.313e-10 m | 6.219e+05 m | 1.50e-15 |
| `nav.innov.y` | 1.776e-15 | 6.913e+00 | 2.57e-16 |
| `nav.est.P` | 8.674e-19 | 5.105e+01 | 1.70e-20 |
| `nav.innov.S` | 6.776e-20 | 9.243e+01 | 7.33e-22 |

The covariance is the most-perturbed channel by raw count - 25,495 of 72,120
entries differ - and simultaneously the least perturbed by magnitude, at
1.7e-20 of its own scale. `nav.innov.sensor_id` and `nav.innov.m` (the integer
identity and dimension channels) are bit-identical, so **no aiding update fired
on a different sensor, in a different order, or with a different measurement
dimension on the two platforms.** The filter's discrete decisions agree
exactly; only its arithmetic moves in the last bits.

**`missions/ascent_leo_gnc.toml`** - first differing byte at offset 3,420;
338,849 of 2,653,436 bytes differ. 20 channels bit-identical. This is the worst
case in the set, as expected from 7,600 integration steps through the most
transcendental-dense model chain in the project.

| Channel | max abs diff | RMS scale | rel to scale |
|---|---|---|---|
| `gnc.cmd.w_cmd_b_radps` | 4.139e-13 rad/s | 3.909e-03 | 1.06e-10 |
| `nav.err.e` | 4.829e-15 | 1.246e-04 | 3.88e-11 |
| `gnc.cmd.tau_b_nm` | 7.309e-10 N*m | 7.878e+01 | 9.28e-12 |
| `forces.f_aero_b_n` | 5.086e-09 N | 6.058e+03 | 8.40e-13 |
| `env.q_pa` | 1.189e-08 Pa | 1.685e+04 | 7.06e-13 |
| `env.rho_kgpm3` | 1.090e-13 | 3.614e-01 | 3.02e-13 |
| `truth.v_mps` | 2.012e-11 m/s | 2.427e+03 | 8.29e-15 |
| `truth.r_m` | 3.842e-09 m | 3.758e+06 | 1.02e-15 |

After 760 s of powered ascent the two platforms' trajectories differ by
**3.8 nanometres in position and 20 picometres per second in velocity**.

### The structural split, and what it shows

Across all four differing missions the same channels stay exact:

- **Every `t_s` channel in every group** is bit-identical in every mission. The
  time grid does not drift.
- **Every integer and flag channel** is bit-identical: `events.code`,
  `events.detail` (the `str16` payload), `gnc.cmd.valid`, `sensors.*.valid`,
  `nav.innov.sensor_id`, `nav.innov.m`. **No control-flow decision differed
  between the platforms in any mission** - no event fired at a different step,
  no sensor was gated differently, no update was skipped.
- **`mass.mass_kg`, `mass.cg_b_m`, `mass.inertia_b_kgm2`** are bit-identical
  everywhere.
- **`truth.r_m` and `truth.v_mps` are bit-identical in all three orbital
  missions** and diverge only in the ascent.

That last point is the sharpest evidence in this pass. In the attitude, EKF,
and camera missions the translational dynamics run on point-mass gravity -
basic operations only - and cross the platform boundary **exactly**, while the
rotational state, the sensors, and the GNC channels in the *same file, on the
same integration steps* diverge. The ascent mission is the one whose
translational forces are fed by the atmosphere and aero models, and it is the
one whose `truth.r_m` moves. The split follows the arithmetic, not the mission.

### Attribution: it is libm, measured

The mechanism was not inferred from the pattern above. A probe
(`libm_probe.cpp`, scratch apparatus, not committed) was compiled on each
toolchain with that toolchain's project flags (`/O2 /fp:strict`;
`-O2 -fno-fast-math -ffp-contract=off -frounding-math`) and evaluated the libm
functions the core actually calls, at 17 arguments spanning the regimes the
missions exercise, printing every result as its exact binary64 bit pattern.
The core's call inventory under `cpp/src/` is `sin` (40 sites), `sqrt` (31),
`cos` (28), `atan2` (19), `pow` (9), `exp` (8), `log` (5), `asin` (3), `acos`
(2), `atan` (1).

| Function | cases | differing | max ULP |
|---|---|---|---|
| `basicops` (a `+ - * /` and `sqrt` chain) | 17 | **0** | 0 |
| `sqrt` | 17 | **0** | 0 |
| `cos`, `tan`, `exp`, `log`, `pow`, `asin`, `atan2` | 17 each | 0 | 0 |
| `sin` | 17 | 1 | 1 |
| `acos` | 11 | 1 | 1 |
| `atan` | 17 | 1 | 1 |

The clearest single instance is `sin(pi/4)`: MSVC's CRT returns
`0.7071067811865476`, glibc 2.39 returns `0.7071067811865475` - adjacent
doubles, 1 ULP apart.

This is the expected state of affairs rather than a defect in either library.
IEEE-754 requires correct rounding for `+`, `-`, `*`, `/` and `sqrt`; it does
**not** require it for the transcendental functions, and no mainstream libm
provides it. Two conforming implementations may legitimately differ in the last
place.

**The project's recorded position is confirmed, and one part of it is
sharpened.** The position - that code using only IEEE-754 basic operations
under strict FP flags with contraction off is bit-identical across compilers,
and that only libm-bearing paths need a cross-platform tolerance budget - holds
in both directions here: the basic-operation control differs in zero of 17
cases and the two-body mission is byte-identical, while every mission touching
a transcendental differs. The sharpening is that the zero-difference rows in
the table above are **sample results, not guarantees**: `cos` and `exp` agreed
at these 17 arguments, and `env.rho_kgpm3` (an `exp`-bearing channel) still
differs in the ascent mission, so agreement at probe arguments does not
generalize to agreement everywhere. Only the basic-operation rows are
guaranteed by the standard.

### Relation to the FR-30 / D-10 gate

The D-10 bound is 1e-9 relative on final-state divergence. Recomputed here on
the final `truth` record of each mission:

| Mission | final abs dr | final abs dv | max component rel | vs 1e-9 |
|---|---|---|---|---|
| `twobody_leo` | 0 | 0 | 0 | PASS |
| `leo_attitude_gnc` | 0 | 0 | 0 | PASS |
| `leo_ekf_consistency` | 0 | 0 | 0 | PASS |
| Camera optical | 0 | 0 | 0 | PASS |
| `ascent_leo_gnc` | 5.35e-09 m | 2.67e-11 m/s | 2.74e-14 | PASS |

Every mission passes D-10 with margin. Note what the final-state metric misses,
though: four of these five rows read `0` while the files differ in thousands of
bytes, because the final-state gate looks only at `truth.r_m` and `truth.v_mps`
- the two channels that are bit-identical in the orbital missions. **The FR-30
gate as constituted cannot see the divergence documented in this section.** It
is not wrong; it measures a different and weaker property than FR-21 states.
The widest scale-relative divergence found anywhere in this pass is 1.06e-10
(`gnc.cmd.w_cmd_b_radps`, ascent), which is inside 1e-9 but by only one order
of magnitude, on a channel the gate does not sample.

**Addressed on the Phase 6 closeout.** The gate was widened rather than
replaced: the final-state path above is unchanged and still enforces D-10, and
a channel-level path was added beside it (`extract-channels`,
`measure-channels`, `gate-channels` in `scripts/cross_platform_divergence.py`,
recorded in the `[channels]` table of
`tests/golden/determinism/cross_platform.toml`). It compares **every** channel
of five missions — the four measured in this section plus the new
`missions/leo_optical_nav.toml` — splitting them into an exact class asserted
bit-identical and a tolerance class gated against a derived bound. The exact
class is where this section's structural split is turned into an assertion:
every integer and flag channel, every `t_s`, the whole of the basic-ops-only
two-body mission, and the point-mass missions' `truth.r_m`, `truth.v_mps` and
`mass.*`. The tolerance class is the libm-bearing remainder, where this
section established that cross-binary byte identity is false rather than merely
unmeasured, and is gated at sqrt(1.06e-10 * 1e-9) = 3.2558e-10 — the geometric
mean of the worst value measured here and the D-10 ceiling, sitting a factor of
3.07 above the former and the same factor below the latter. Two caveats from
this section carry straight into that gate and are stated in its record: the
1.06e-10 figure is a two-leg x86-64 measurement, so macOS and aarch64 remain
the reason for the upper margin; and it is a worst case over five missions and
one seed each, not a bound over the mission space.

### What is now measured

- Windows/MSVC and Linux/GCC produce **byte-identical** `run.srlog` files for
  `missions/twobody_leo.toml`, whole file, zero excluded fields.
- All five headers, at all five missions, are byte-identical - including the
  v1.3 `gnc.camera` `ieee754-binary64-hex` echo, which is character-identical
  across the crossing. `config_sha256` matches on all five, and the FR-15
  canonical bytes were confirmed to carry no host-absolute path.
- `producer.git_hash` and `producer.core_version` were **compared, not
  excluded**, by rebuilding both wheels at `668b9fc`. No field was excluded
  from any comparison in this section.
- The four Phase 6 GNC missions differ in the record stream only, at identical
  file length and identical structure, with worst-case scale-relative
  divergence 1.06e-10 and worst-case final-state divergence 5.35e-09 m after
  760 s of powered ascent.
- No control-flow decision differs across platforms in any mission: every
  `t_s`, event code, validity flag, `nav.innov.sensor_id` and `nav.innov.m`
  channel is bit-identical.
- `truth.r_m` and `truth.v_mps` are bit-identical across platforms in the three
  point-mass-gravity missions and differ only in the ascent, whose
  translational forces pass through the atmosphere and aero models.
- The mechanism is libm: a basic-operation control chain and `sqrt` differ in
  0 of 17 cases, while `sin`, `acos` and `atan` differ by 1 ULP between the
  MSVC CRT and glibc 2.39.
- FR-21's **within-platform** clause re-confirmed on both hosts: five missions
  run twice per platform, all ten hash pairs reproduce exactly.

### What this does and does not establish about FR-21

**Establishes.** FR-21 as written - "the same inputs on the same binary always
produce bit-identical outputs" - is satisfied. That sentence scopes the
guarantee to *the same binary*, and within a binary the contract holds on both
platforms, measured twice each. The format-level determinism claims are also
upheld: no embedded timestamp, no host-dependent content, no layout
nondeterminism appears anywhere in five files across two operating systems,
two compilers, two standard libraries and two Python versions. Identical file
lengths and identical integer channels are strong evidence for that.

**Does not establish.** A byte-identical log across *different* binaries is
**false** for any mission that touches a transcendental function, and this
section is the first measurement to say so. Nothing in the repository claimed
otherwise, but nothing had measured it either, and the distinction is easy to
lose: a reader who takes "bit-reproducible" as a property of the *simulator*
rather than of *a build of the simulator* would be wrong for four of the five
missions tested here. Any workflow that compares SRLOG hashes across platforms
- a cross-platform golden log, a CI artifact hash gate, a distributed Monte
Carlo run whose shards land on mixed workers - must use a tolerance, not a
hash, unless the mission is provably free of libm on every active path.

The honest one-line statement of the measured contract is: **SRLOG output is
bit-reproducible per binary, and bit-reproducible across binaries only for
libm-free missions; elsewhere it agrees to within 1.1e-10 of channel scale on
the missions measured here.**

### What remains unverified

- **macOS and aarch64.** Not run; no local target. Both are CI `build-test`
  legs. The aarch64 case is materially different from the two legs compared
  here, because it changes the instruction set and not merely the libm: the
  basic-operation guarantee still holds by IEEE-754, but nothing in this pass
  bounds an ARM libm's divergence.
- **Clang.** Not used in this pass. A third x86-64 libm was not sampled.
- **Whether the divergence bound generalizes.** The 1.06e-10 figure is the
  worst case over five missions, one seed each. It is not a bound over the
  mission space, over seeds, or over durations longer than the 760 s ascent.
  Error growth in a closed loop is not guaranteed monotone or bounded, and a
  longer or less stable scenario could diverge further; a Monte Carlo over
  seeds and durations would be needed to state a real bound.
- **Whether any *committed* mission emits a camera group.** None did at the
  time of this pass. The camera header echo was tested through the
  optical-gate fixture's mission text, which is committed inside a Python
  test module rather than as a `missions/*.toml`, so the echo was verified
  but not by anything a user could run from `missions/`.
  **Closed on the Phase 6 closeout:** `missions/leo_optical_nav.toml` is
  committed and is the first shipped mission that configures a camera. It
  emits the `sensors.camera` group and the v1.3 `gnc.camera` header echo,
  projects five surface landmarks at 2 Hz over a 60 s run, and is one of the
  five missions the widened gate below compares. Measured on the maintainer
  host: 453 of 600 landmark-samples are fully visible against the
  `eq:camera:nearside` and sensor-bounds tests, every one of the 120 camera
  samples carries at least one visible landmark, and the visible pixels span
  u = 1.7 to 1275.3 of 1280 and v = 95.4 to 910.8 of 1024.
- **The FR-30 CI job itself.** Not run here. This pass replicates its
  *question* at a stronger level on one of its four legs' toolchains, but the
  job's own four-leg measurement was not exercised.
- **Whether the ascent divergence is dominated by one model.** The attribution
  establishes libm as the class. It does not apportion the ascent's divergence
  among the pitch-program `sin`/`cos`, the atmosphere's `exp`, the aero
  tables, and the geodetic conversions. Doing so would need per-model probes
  at the missions' actual argument streams.
