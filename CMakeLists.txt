# star_reacher build: C++17 core (star::), pybind11 module (star_reacher._core),
# doctest suite. Two build paths share this file (contract section 9):
#   - wheel builds via scikit-build-core (SKBUILD defined): bindings on, tests off;
#   - dev/CI preset builds (CMakePresets.json): tests on, bindings off by default,
#     so ctest needs no Python toolchain.
cmake_minimum_required(VERSION 3.26)

# Version is kept in sync with pyproject.toml [project] version manually
# (Phase 1 contract: both files carry this note).
project(star_reacher VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(DEFINED SKBUILD)
  set(_star_bindings_default ON)
  set(_star_testing_default OFF)
else()
  set(_star_bindings_default OFF)
  set(_star_testing_default ON)
endif()
option(STAR_BUILD_PYTHON_BINDINGS "Build the star_reacher._core pybind11 module"
       ${_star_bindings_default})
option(STAR_BUILD_TESTING "Build the doctest suite" ${_star_testing_default})
option(STAR_WARNINGS_AS_ERRORS "Treat compiler warnings as errors (ci preset)" OFF)

# ---------------------------------------------------------------------------
# Eigen 3.4.0, pinned by URL + SHA256 (D-1: the only mandatory core dependency).
# SOURCE_SUBDIR points at a directory with no CMakeLists.txt on purpose:
# FetchContent_MakeAvailable then only downloads and never runs Eigen's own
# CMake, which still declares compatibility with CMake < 3.5 and would fail
# to configure under CMake 4.x. Eigen is header-only, so an INTERFACE target
# over the source tree is the complete integration.
# ---------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(
  eigen
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
  URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
  SOURCE_SUBDIR cmake_subdir_that_does_not_exist
)
FetchContent_MakeAvailable(eigen)
add_library(star_eigen INTERFACE)
# SYSTEM keeps Eigen's headers out of our warnings-as-errors surface.
target_include_directories(star_eigen SYSTEM INTERFACE "${eigen_SOURCE_DIR}")

# ---------------------------------------------------------------------------
# Determinism flags (D-10), applied UNCONDITIONALLY to every target that
# contains or links core numerics - not via presets - so wheel builds get
# exactly the same floating-point semantics as dev and CI builds.
#   GCC/Clang: -O2 -fno-fast-math -ffp-contract=off -frounding-math
#   MSVC:      /O2 /fp:strict
# Exception, documented deviation: on MSVC, /O2 is suppressed for the Debug
# configuration only, because MSVC's default Debug flags include /RTC1 and
# cl.exe hard-errors on /O2 + /RTC1. /fp:strict (the flag that actually pins
# FP semantics) remains unconditional; every non-Debug configuration -
# including all wheel and CI builds - gets the full flag set.
# ---------------------------------------------------------------------------
function(star_apply_build_flags target)
  if(MSVC)
    target_compile_options(${target} PRIVATE
      /fp:strict
      $<$<NOT:$<CONFIG:Debug>>:/O2>)
  else()
    target_compile_options(${target} PRIVATE
      -O2 -fno-fast-math -ffp-contract=off -frounding-math)
  endif()
  if(STAR_WARNINGS_AS_ERRORS)
    if(MSVC)
      target_compile_options(${target} PRIVATE /W4 /WX)
    else()
      target_compile_options(${target} PRIVATE -Wall -Wextra -Werror)
    endif()
  endif()
endfunction()

# ---------------------------------------------------------------------------
# Build identity: git hash captured at configure time (D-11 producer field).
# "unknown" when git or the repository is unavailable (e.g. sdist builds) -
# the core must never probe repository state at runtime.
# ---------------------------------------------------------------------------
set(STAR_GIT_HASH "unknown")
find_package(Git QUIET)
if(Git_FOUND)
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${CMAKE_CURRENT_SOURCE_DIR}" rev-parse HEAD
    OUTPUT_VARIABLE _star_git_hash
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _star_git_rc
    ERROR_QUIET)
  if(_star_git_rc EQUAL 0)
    set(STAR_GIT_HASH "${_star_git_hash}")
  endif()
endif()
configure_file(cpp/src/version.cpp.in
               "${CMAKE_CURRENT_BINARY_DIR}/generated/version.cpp" @ONLY)

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
add_library(star_core STATIC
  cpp/src/ephemeris.cpp
  cpp/src/models/twobody.cpp
  cpp/src/rng.cpp
  cpp/src/run.cpp
  cpp/src/srlog_writer.cpp
  "${CMAKE_CURRENT_BINARY_DIR}/generated/version.cpp")
target_include_directories(star_core PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/cpp/include")
target_link_libraries(star_core PUBLIC star_eigen)
# The static core is linked into a shared Python module, so it must be PIC on
# ELF platforms regardless of which build path is active.
set_target_properties(star_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
star_apply_build_flags(star_core)

# ---------------------------------------------------------------------------
# pybind11 module star_reacher._core
# ---------------------------------------------------------------------------
if(STAR_BUILD_PYTHON_BINDINGS)
  find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_core bindings/module.cpp)
  target_link_libraries(_core PRIVATE star_core)
  star_apply_build_flags(_core)
  # scikit-build-core installs into the wheel root; the module lands inside
  # the star_reacher package next to the pure-Python frontend.
  install(TARGETS _core DESTINATION star_reacher)
endif()

# ---------------------------------------------------------------------------
# doctest suite (vendored single header; PRD D-14), wired into CTest.
# ---------------------------------------------------------------------------
if(STAR_BUILD_TESTING)
  enable_testing()
  add_executable(star_tests
    cpp/tests/test_main.cpp
    cpp/tests/test_ephemeris.cpp
    cpp/tests/test_rng.cpp
    cpp/tests/test_srlog.cpp
    cpp/tests/test_twobody.cpp)
  target_link_libraries(star_tests PRIVATE star_core)
  # Golden vectors are read from the source tree so the test binary needs no
  # install step or fixture copying; the path is frozen at configure time.
  target_compile_definitions(star_tests PRIVATE
    STAR_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/golden")
  star_apply_build_flags(star_tests)
  add_test(NAME star_doctest COMMAND star_tests)
endif()
