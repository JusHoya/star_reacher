// SRLOG v1.0 writer byte-level test: the header round-trips against an
// independently assembled reference byte sequence (contract section 2 /
// docs/formats/srlog_v1.md), and the record stream layout is verified field
// by field. The reference bytes are synthesized here in test code - binary
// fixtures are never committed (contract section 11).
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "star/srlog_writer.hpp"
#include "vendor/doctest.h"

namespace {

std::vector<unsigned char> read_all_bytes(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  REQUIRE(static_cast<bool>(in));
  return std::vector<unsigned char>(std::istreambuf_iterator<char>(in),
                                    std::istreambuf_iterator<char>());
}

// Little-endian field extraction; memcpy avoids alignment and aliasing UB.
template <typename T>
T read_le(const std::vector<unsigned char>& buf, std::size_t offset) {
  REQUIRE(offset + sizeof(T) <= buf.size());
  T v;
  std::memcpy(&v, buf.data() + offset, sizeof(T));
  return v;
}

}  // namespace

TEST_CASE("srlog_writer_header_roundtrip") {
  star::log::SrlogHeaderFields fields;
  fields.core_version = "0.1.0-test";
  fields.git_hash = "0123456789abcdef0123456789abcdef01234567";
  // Recognizable 64-hex digest (SHA-256 of the empty string).
  fields.config_sha256 =
      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
  fields.master_seed = 1234567890ULL;
  fields.oracle = false;
  fields.epoch_utc = "2026-01-01T00:00:00Z";
  fields.central_body = "earth";
  fields.truth_rate_hz = 10;

  // Reference header JSON assembled independently of the serializer, byte for
  // byte (compact separators, fixed key order per contract section 2). Any
  // serializer change that alters the bytes must fail here first.
  const std::string expected_json =
      "{\"format\":{\"name\":\"SRLOG\",\"major\":1,\"minor\":0},"
      "\"producer\":{\"core_version\":\"0.1.0-test\","
      "\"git_hash\":\"0123456789abcdef0123456789abcdef01234567\"},"
      "\"config_sha256\":"
      "\"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\","
      "\"master_seed\":\"1234567890\",\"oracle\":false,"
      "\"epoch_utc\":\"2026-01-01T00:00:00Z\",\"central_body\":\"earth\","
      "\"groups\":["
      "{\"name\":\"truth\",\"rate_hz\":10,\"channels\":["
      "{\"name\":\"t_s\",\"dtype\":\"f64\",\"units\":\"s\",\"frame\":\"\"},"
      "{\"name\":\"r_m\",\"dtype\":\"f64[3]\",\"units\":\"m\",\"frame\":\"GCRF\"},"
      "{\"name\":\"v_mps\",\"dtype\":\"f64[3]\",\"units\":\"m/s\","
      "\"frame\":\"GCRF\"},"
      "{\"name\":\"q_i2b\",\"dtype\":\"f64[4]\",\"units\":\"1\","
      "\"frame\":\"GCRF->body Hamilton scalar-first\"},"
      "{\"name\":\"w_b_radps\",\"dtype\":\"f64[3]\",\"units\":\"rad/s\","
      "\"frame\":\"body\"},"
      "{\"name\":\"mass_kg\",\"dtype\":\"f64\",\"units\":\"kg\",\"frame\":\"\"}"
      "]},"
      "{\"name\":\"events\",\"rate_hz\":0,\"channels\":["
      "{\"name\":\"t_s\",\"dtype\":\"f64\",\"units\":\"s\",\"frame\":\"\"},"
      "{\"name\":\"code\",\"dtype\":\"u32\",\"units\":\"1\",\"frame\":\"\"},"
      "{\"name\":\"detail\",\"dtype\":\"str16\",\"units\":\"\",\"frame\":\"\"}"
      "]}]}";
  CHECK(star::log::SrlogWriter::header_json(fields) == expected_json);

  // Write a minimal file: run_start event, one truth record, run_end event.
  const std::string path = "test_srlog_writer_roundtrip.srlog";
  {
    star::log::SrlogWriter writer(path, fields);
    writer.write_event(0.0, 1, "run_start");
    const double q[4] = {1.0, 0.0, 0.0, 0.0};
    writer.write_truth(1.5, Eigen::Vector3d(6778137.0, 0.0, 0.0),
                       Eigen::Vector3d(0.0, 7668.6, 0.0), q,
                       Eigen::Vector3d::Zero(), 150.0);
    writer.write_event(600.0, 2, "run_end");
    writer.close();
  }
  const std::vector<unsigned char> bytes = read_all_bytes(path);
  std::remove(path.c_str());

  // HEADER. Magic: ASCII "SRLOG", NUL, CR, LF - text-mode mangling tripwire.
  const unsigned char magic[8] = {0x53, 0x52, 0x4C, 0x4F, 0x47, 0x00, 0x0D, 0x0A};
  REQUIRE(bytes.size() >= 16);
  CHECK(std::memcmp(bytes.data(), magic, 8) == 0);
  CHECK(read_le<std::uint16_t>(bytes, 8) == 1);    // version_major
  CHECK(read_le<std::uint16_t>(bytes, 10) == 0);   // version_minor
  const std::uint32_t json_len = read_le<std::uint32_t>(bytes, 12);
  CHECK(json_len == expected_json.size());
  REQUIRE(bytes.size() >= 16 + json_len);
  CHECK(std::string(bytes.begin() + 16, bytes.begin() + 16 + json_len) ==
        expected_json);

  // RECORD STREAM. Sizes: truth payload = 15 doubles = 120 bytes; events
  // payload = 8 (t_s) + 4 (code) + 2 + len (str16).
  std::size_t off = 16 + json_len;

  // events record: run_start.
  CHECK(read_le<std::uint16_t>(bytes, off) == 1);  // group index: events
  CHECK(read_le<double>(bytes, off + 2) == 0.0);
  CHECK(read_le<std::uint32_t>(bytes, off + 10) == 1);
  CHECK(read_le<std::uint16_t>(bytes, off + 14) == 9);
  CHECK(std::string(bytes.begin() + static_cast<std::ptrdiff_t>(off) + 16,
                    bytes.begin() + static_cast<std::ptrdiff_t>(off) + 25) ==
        "run_start");
  off += 2 + 8 + 4 + 2 + 9;

  // truth record. t_s = 1.5 doubles as LE bytes 00..F8 3F - this pins the
  // on-disk encoding to IEEE-754 binary64 little-endian, not merely to
  // "whatever the writer produced".
  CHECK(read_le<std::uint16_t>(bytes, off) == 0);  // group index: truth
  const unsigned char t_le[8] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF8, 0x3F};
  CHECK(std::memcmp(bytes.data() + off + 2, t_le, 8) == 0);
  CHECK(read_le<double>(bytes, off + 10) == 6778137.0);   // r_m[0]
  CHECK(read_le<double>(bytes, off + 18) == 0.0);         // r_m[1]
  CHECK(read_le<double>(bytes, off + 26) == 0.0);         // r_m[2]
  CHECK(read_le<double>(bytes, off + 42) == 7668.6);      // v_mps[1]
  CHECK(read_le<double>(bytes, off + 58) == 1.0);         // q_i2b[0] (scalar first)
  CHECK(read_le<double>(bytes, off + 66) == 0.0);         // q_i2b[1]
  CHECK(read_le<double>(bytes, off + 114) == 150.0);      // mass_kg
  off += 2 + 120;

  // events record: run_end.
  CHECK(read_le<std::uint16_t>(bytes, off) == 1);
  CHECK(read_le<double>(bytes, off + 2) == 600.0);
  CHECK(read_le<std::uint32_t>(bytes, off + 10) == 2);
  CHECK(read_le<std::uint16_t>(bytes, off + 14) == 7);
  CHECK(std::string(bytes.begin() + static_cast<std::ptrdiff_t>(off) + 16,
                    bytes.begin() + static_cast<std::ptrdiff_t>(off) + 23) ==
        "run_end");
  off += 2 + 8 + 4 + 2 + 7;

  // No footer, no trailing bytes: the file ends exactly at the last record.
  CHECK(off == bytes.size());
}
