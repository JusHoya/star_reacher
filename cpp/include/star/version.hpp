// Build-identity strings embedded at CMake configure time. They travel in
// every SRLOG header (D-11) so a log is traceable to the exact producer
// binary without any runtime git or filesystem access (the core never reads
// the clock, the network, or repository state at runtime).
#ifndef STAR_VERSION_HPP
#define STAR_VERSION_HPP

namespace star {

// Semantic version of the core, e.g. "0.1.0". Sourced from the CMake
// project() version, which is kept in sync with pyproject.toml manually.
const char* core_version();

// 40-hex git commit hash captured at configure time, or "unknown" when git or
// the repository is unavailable (e.g. building from an sdist).
const char* git_hash();

}  // namespace star

#endif  // STAR_VERSION_HPP
