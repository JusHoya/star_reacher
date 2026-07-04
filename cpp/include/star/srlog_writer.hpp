// SRLOG v1.1 binary log writer (PRD D-11, FR-16; byte format normative in
// docs/formats/srlog_v1.md). The Phase 1 two-group schema (`truth` +
// `events`) is always present; v1.1 adds the optional vehicle channel groups
// (`forces`, `mass`, `env`), declared at header-write time. Later phases
// extend the channel dictionary through minor-version bumps, never layout
// breaks.
//
// The header JSON is hand-rolled with a fixed key order and contains only
// integers, booleans, and strings - no floats - so the exact header bytes are
// a pure function of the config values and the producer identity. That is
// what makes the double-run SHA-256 determinism gate (FR-21) meaningful at
// the whole-file level.
#ifndef STAR_SRLOG_WRITER_HPP
#define STAR_SRLOG_WRITER_HPP

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include <Eigen/Dense>

namespace star {
namespace log {

// Header identity fields. Everything here is input-derived or build-derived;
// wall-clock and host data are banned from the file by D-11 (they live in the
// Python-written meta.json sidecar).
struct SrlogHeaderFields {
  std::string core_version;   // e.g. "0.1.0"
  std::string git_hash;       // 40-hex or "unknown"
  std::string config_sha256;  // 64-hex, computed by the Python validator
  std::uint64_t master_seed;  // serialized as a decimal string (JSON has no u64)
  bool oracle;
  std::string epoch_utc;      // ISO-8601, carried verbatim from the mission file
  std::string central_body;   // "earth" in Phase 1
  std::uint32_t truth_rate_hz;

  // v1.1 vehicle channel groups (FR-16; format doc section 3.1). Each group
  // enters the file only when its rate is nonzero, and every nonzero rate
  // must divide truth_rate_hz exactly: group records are decimated from the
  // truth grid, never interpolated. The defaults keep pre-Phase-4 call sites
  // valid unchanged (no group is silently enabled).
  //
  // force_sources is the enabled subset of the canonical vocabulary
  // (kForceSources), in canonical order without duplicates, so the header
  // bytes are a pure function of the enabled set. A nonzero forces_rate_hz
  // with no sources, or sources with a zero rate, is rejected rather than
  // silently reconciled.
  std::vector<std::string> force_sources;
  std::uint32_t forces_rate_hz = 0;
  std::uint32_t mass_rate_hz = 0;
  std::uint32_t env_rate_hz = 0;
};

// Canonical FR-16 force/torque source vocabulary, in canonical order (format
// doc section 3.1). Extending it is a minor version bump.
inline constexpr const char* kForceSources[] = {
    "gravity", "thirdbody", "srp", "drag", "aero",
    "thrust",  "rcs",       "gravgrad", "wheel"};
inline constexpr std::size_t kForceSourceCount =
    sizeof(kForceSources) / sizeof(kForceSources[0]);

// One per-source sample of a `forces` record: body-frame force [N] and
// torque about the composite CG [N*m].
struct ForceSourceSample {
  Eigen::Vector3d force_b_n;
  Eigen::Vector3d torque_b_nm;
};

class SrlogWriter {
 public:
  // Opens `path` and writes the complete header (magic, version words,
  // length-prefixed JSON). Throws std::runtime_error if the file cannot be
  // opened; throws std::logic_error on a big-endian host (the format is
  // little-endian and every supported platform is little-endian, so this is
  // a guard against silent porting mistakes, not a supported configuration);
  // throws std::invalid_argument on an invalid v1.1 group declaration (rate
  // not a divisor of the truth rate, unknown/misordered/duplicate force
  // source, or a sources/rate mismatch) - before the output file is created,
  // so a rejected configuration never leaves a truncated file behind.
  SrlogWriter(const std::string& path, const SrlogHeaderFields& fields);

  SrlogWriter(const SrlogWriter&) = delete;
  SrlogWriter& operator=(const SrlogWriter&) = delete;

  // One `truth` record (group index 0): t_s, r_m, v_mps, q_i2b
  // (Hamilton scalar-first, D-7), w_b_radps, mass_kg.
  void write_truth(double t_s, const Eigen::Vector3d& r_m,
                   const Eigen::Vector3d& v_mps, const double (&q_i2b)[4],
                   const Eigen::Vector3d& w_b_radps, double mass_kg);

  // One `events` record (group index 1): t_s, code, str16 detail.
  // `detail` must be at most 65535 UTF-8 bytes (str16 length prefix).
  void write_event(double t_s, std::uint32_t code, std::string_view detail);

  // One `forces` record: t_s, then one (force, torque) sample per declared
  // source, in declaration order - samples.size() must equal the declared
  // source count (std::invalid_argument otherwise). Throws std::logic_error
  // if the forces group was not declared at construction: the write_* calls
  // for optional groups are programming errors when the header never
  // announced the group.
  void write_forces(double t_s, const std::vector<ForceSourceSample>& samples);

  // One `mass` record: composite mass, body-frame CG, and the inertia tensor
  // about the CG packed row-major upper-triangle [Ixx, Ixy, Ixz, Iyy, Iyz,
  // Izz] (the FR-26 packing). Throws std::logic_error if undeclared.
  void write_mass(double t_s, double mass_kg, const Eigen::Vector3d& cg_b_m,
                  const double (&inertia_b_kgm2)[6]);

  // One `env` record: geodetic altitude, Mach, dynamic pressure, density,
  // flight-path angle. Throws std::logic_error if undeclared.
  void write_env(double t_s, double alt_m, double mach, double q_pa,
                 double rho_kgpm3, double fpa_rad);

  // Flush and close; called by the destructor if not called explicitly.
  void close();

  ~SrlogWriter();

  // The exact header JSON bytes for `fields` (compact separators, fixed key
  // order). Exposed so tests can assert byte equality without re-implementing
  // the serializer, and so the format document can be checked against code.
  // Performs the same group-declaration validation as the constructor.
  static std::string header_json(const SrlogHeaderFields& fields);

 private:
  void put_u16(std::uint16_t v);
  void put_u32(std::uint32_t v);
  void put_f64(double v);
  void put_bytes(const void* data, std::size_t n);

  std::ofstream out_;
  // Record group indices of the enabled v1.1 groups; -1 marks a group absent
  // from this file (its write_* call is then a logic error).
  int forces_index_ = -1;
  int mass_index_ = -1;
  int env_index_ = -1;
  std::size_t force_source_count_ = 0;
};

}  // namespace log
}  // namespace star

#endif  // STAR_SRLOG_WRITER_HPP
