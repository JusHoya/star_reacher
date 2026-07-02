// SRLOG v1.0 binary log writer (PRD D-11, FR-16; byte format normative in
// docs/formats/srlog_v1.md). Phase 1 writes the fixed two-group schema
// (`truth` + `events`); later phases extend the channel dictionary through
// minor-version bumps, never layout breaks.
//
// The header JSON is hand-rolled with a fixed key order and contains only
// integers, booleans, and strings - no floats - so the exact header bytes are
// a pure function of the config values and the producer identity. That is
// what makes the double-run SHA-256 determinism gate (FR-21) meaningful at
// the whole-file level.
#ifndef STAR_SRLOG_WRITER_HPP
#define STAR_SRLOG_WRITER_HPP

#include <cstdint>
#include <fstream>
#include <string>
#include <string_view>

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
};

class SrlogWriter {
 public:
  // Opens `path` and writes the complete header (magic, version words,
  // length-prefixed JSON). Throws std::runtime_error if the file cannot be
  // opened; throws std::logic_error on a big-endian host (the format is
  // little-endian and every supported platform is little-endian, so this is
  // a guard against silent porting mistakes, not a supported configuration).
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

  // Flush and close; called by the destructor if not called explicitly.
  void close();

  ~SrlogWriter();

  // The exact header JSON bytes for `fields` (compact separators, fixed key
  // order). Exposed so tests can assert byte equality without re-implementing
  // the serializer, and so the format document can be checked against code.
  static std::string header_json(const SrlogHeaderFields& fields);

 private:
  void put_u16(std::uint16_t v);
  void put_u32(std::uint32_t v);
  void put_f64(double v);
  void put_bytes(const void* data, std::size_t n);

  std::ofstream out_;
};

}  // namespace log
}  // namespace star

#endif  // STAR_SRLOG_WRITER_HPP
