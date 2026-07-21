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
- **Clang warnings-as-errors.** Clang was used only for the `asan` preset,
  which does not set `-Wall -Wextra -Werror`. Clang's warning surface over
  the Phase 6 code is unmeasured.
- **Determinism across platforms for Phase 6 outputs.** This pass compared
  test counts and one probe value, not logged run output. The FR-30
  cross-platform divergence gate is a separate CI job and was not run here.

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
