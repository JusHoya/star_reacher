// SRLOG v1 writer byte-level tests: headers round-trip against independently
// assembled reference byte sequences (contract section 2 /
// docs/formats/srlog_v1.md), and the record stream layout is verified field
// by field, for the always-present Phase 1 groups and the v1.1 vehicle
// channel groups. The reference bytes are synthesized here in test code -
// binary fixtures are never committed (contract section 11).
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
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
      "{\"format\":{\"name\":\"SRLOG\",\"major\":1,\"minor\":1},"
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
  CHECK(read_le<std::uint16_t>(bytes, 10) == 1);   // version_minor
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

namespace {

// Baseline v1.1 declaration used by the vehicle-group tests: all three
// optional groups enabled, two force sources, rates that are distinct exact
// divisors of the 10 Hz truth rate.
star::log::SrlogHeaderFields v11_fields() {
  star::log::SrlogHeaderFields fields;
  fields.core_version = "0.3.0-test";
  fields.git_hash = "0123456789abcdef0123456789abcdef01234567";
  fields.config_sha256 =
      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
  fields.master_seed = 1234567890ULL;
  fields.oracle = false;
  fields.epoch_utc = "2026-01-01T00:00:00Z";
  fields.central_body = "earth";
  fields.truth_rate_hz = 10;
  fields.force_sources = {"gravity", "thrust"};
  fields.forces_rate_hz = 1;
  fields.mass_rate_hz = 2;
  fields.env_rate_hz = 5;
  return fields;
}

}  // namespace

TEST_CASE("srlog_v11_header_declares_vehicle_groups") {
  // Reference JSON assembled independently of the serializer (format doc
  // section 3.1): the enabled groups appear after truth and events in the
  // fixed order forces, mass, env, with the source-derived forces channels
  // in declaration order.
  const std::string expected_json =
      "{\"format\":{\"name\":\"SRLOG\",\"major\":1,\"minor\":1},"
      "\"producer\":{\"core_version\":\"0.3.0-test\","
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
      "]},"
      "{\"name\":\"forces\",\"rate_hz\":1,\"channels\":["
      "{\"name\":\"t_s\",\"dtype\":\"f64\",\"units\":\"s\",\"frame\":\"\"},"
      "{\"name\":\"f_gravity_b_n\",\"dtype\":\"f64[3]\",\"units\":\"N\","
      "\"frame\":\"body\"},"
      "{\"name\":\"tq_gravity_b_nm\",\"dtype\":\"f64[3]\",\"units\":\"N*m\","
      "\"frame\":\"body\"},"
      "{\"name\":\"f_thrust_b_n\",\"dtype\":\"f64[3]\",\"units\":\"N\","
      "\"frame\":\"body\"},"
      "{\"name\":\"tq_thrust_b_nm\",\"dtype\":\"f64[3]\",\"units\":\"N*m\","
      "\"frame\":\"body\"}"
      "]},"
      "{\"name\":\"mass\",\"rate_hz\":2,\"channels\":["
      "{\"name\":\"t_s\",\"dtype\":\"f64\",\"units\":\"s\",\"frame\":\"\"},"
      "{\"name\":\"mass_kg\",\"dtype\":\"f64\",\"units\":\"kg\",\"frame\":\"\"},"
      "{\"name\":\"cg_b_m\",\"dtype\":\"f64[3]\",\"units\":\"m\","
      "\"frame\":\"body\"},"
      "{\"name\":\"inertia_b_kgm2\",\"dtype\":\"f64[6]\",\"units\":\"kg*m^2\","
      "\"frame\":\"body\"}"
      "]},"
      "{\"name\":\"env\",\"rate_hz\":5,\"channels\":["
      "{\"name\":\"t_s\",\"dtype\":\"f64\",\"units\":\"s\",\"frame\":\"\"},"
      "{\"name\":\"alt_m\",\"dtype\":\"f64\",\"units\":\"m\",\"frame\":\"\"},"
      "{\"name\":\"mach\",\"dtype\":\"f64\",\"units\":\"1\",\"frame\":\"\"},"
      "{\"name\":\"q_pa\",\"dtype\":\"f64\",\"units\":\"Pa\",\"frame\":\"\"},"
      "{\"name\":\"rho_kgpm3\",\"dtype\":\"f64\",\"units\":\"kg/m^3\","
      "\"frame\":\"\"},"
      "{\"name\":\"fpa_rad\",\"dtype\":\"f64\",\"units\":\"rad\",\"frame\":\"\"}"
      "]}]}";
  CHECK(star::log::SrlogWriter::header_json(v11_fields()) == expected_json);

  // The single-source subset drives a different (still self-describing)
  // dictionary: only the declared source's channel pair appears.
  star::log::SrlogHeaderFields one = v11_fields();
  one.force_sources = {"srp"};
  one.mass_rate_hz = 0;
  one.env_rate_hz = 0;
  const std::string json = star::log::SrlogWriter::header_json(one);
  CHECK(json.find("\"f_srp_b_n\"") != std::string::npos);
  CHECK(json.find("\"f_gravity_b_n\"") == std::string::npos);
  CHECK(json.find("\"name\":\"mass\"") == std::string::npos);
  CHECK(json.find("\"name\":\"env\"") == std::string::npos);
}

TEST_CASE("srlog_v11_record_stream_roundtrip") {
  const std::string path = "test_srlog_v11_roundtrip.srlog";
  {
    star::log::SrlogWriter writer(path, v11_fields());
    std::vector<star::log::ForceSourceSample> samples(2);
    samples[0].force_b_n = Eigen::Vector3d(1.0, 2.0, 3.0);      // gravity
    samples[0].torque_b_nm = Eigen::Vector3d(0.0, 0.0, 0.0);
    samples[1].force_b_n = Eigen::Vector3d(0.0, 0.0, 900.0);    // thrust
    samples[1].torque_b_nm = Eigen::Vector3d(0.5, 0.0, -0.25);
    writer.write_forces(1.5, samples);
    const double inertia[6] = {10.0, 0.5, 0.25, 12.0, 0.125, 8.0};
    writer.write_mass(1.5, 150.0, Eigen::Vector3d(0.1, 0.0, -0.2), inertia);
    writer.write_env(1.5, 1200.5, 0.8, 24000.0, 0.4135, 0.5061);
    writer.close();
  }
  const std::vector<unsigned char> bytes = read_all_bytes(path);
  std::remove(path.c_str());

  const std::uint32_t json_len = read_le<std::uint32_t>(bytes, 12);
  std::size_t off = 16 + json_len;

  // forces record: group index 2; payload = t_s + 2 sources x 6 doubles
  // = 8 + 96 bytes, samples in declaration order (gravity, then thrust).
  CHECK(read_le<std::uint16_t>(bytes, off) == 2);
  // t_s = 1.5 as LE IEEE-754 bytes pins the encoding, not merely the value.
  const unsigned char t_le[8] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF8, 0x3F};
  CHECK(std::memcmp(bytes.data() + off + 2, t_le, 8) == 0);
  CHECK(read_le<double>(bytes, off + 10) == 1.0);     // f_gravity_b_n[0]
  CHECK(read_le<double>(bytes, off + 26) == 3.0);     // f_gravity_b_n[2]
  CHECK(read_le<double>(bytes, off + 34) == 0.0);     // tq_gravity_b_nm[0]
  CHECK(read_le<double>(bytes, off + 74) == 900.0);   // f_thrust_b_n[2]
  CHECK(read_le<double>(bytes, off + 82) == 0.5);     // tq_thrust_b_nm[0]
  CHECK(read_le<double>(bytes, off + 98) == -0.25);   // tq_thrust_b_nm[2]
  off += 2 + 8 + 96;

  // mass record: group index 3; payload = t_s + mass + cg(3) + inertia(6)
  // = 88 bytes, inertia packed [Ixx, Ixy, Ixz, Iyy, Iyz, Izz].
  CHECK(read_le<std::uint16_t>(bytes, off) == 3);
  CHECK(read_le<double>(bytes, off + 2) == 1.5);      // t_s
  CHECK(read_le<double>(bytes, off + 10) == 150.0);   // mass_kg
  CHECK(read_le<double>(bytes, off + 18) == 0.1);     // cg_b_m[0]
  CHECK(read_le<double>(bytes, off + 34) == -0.2);    // cg_b_m[2]
  CHECK(read_le<double>(bytes, off + 42) == 10.0);    // Ixx
  CHECK(read_le<double>(bytes, off + 50) == 0.5);     // Ixy
  CHECK(read_le<double>(bytes, off + 58) == 0.25);    // Ixz
  CHECK(read_le<double>(bytes, off + 66) == 12.0);    // Iyy
  CHECK(read_le<double>(bytes, off + 74) == 0.125);   // Iyz
  CHECK(read_le<double>(bytes, off + 82) == 8.0);     // Izz
  off += 2 + 88;

  // env record: group index 4; payload = 6 doubles = 48 bytes.
  CHECK(read_le<std::uint16_t>(bytes, off) == 4);
  CHECK(read_le<double>(bytes, off + 2) == 1.5);      // t_s
  CHECK(read_le<double>(bytes, off + 10) == 1200.5);  // alt_m
  CHECK(read_le<double>(bytes, off + 18) == 0.8);     // mach
  CHECK(read_le<double>(bytes, off + 26) == 24000.0); // q_pa
  CHECK(read_le<double>(bytes, off + 34) == 0.4135);  // rho_kgpm3
  CHECK(read_le<double>(bytes, off + 42) == 0.5061);  // fpa_rad
  off += 2 + 48;

  // No footer, no trailing bytes.
  CHECK(off == bytes.size());
}

TEST_CASE("srlog_v11_group_declaration_validation") {
  // Every rejection happens in header_json (shared with the constructor), so
  // the checks need no filesystem. Rates must divide the truth rate exactly.
  {
    star::log::SrlogHeaderFields f = v11_fields();
    f.forces_rate_hz = 3;  // 10 % 3 != 0
    CHECK_THROWS_AS(star::log::SrlogWriter::header_json(f),
                    std::invalid_argument);
  }
  {
    star::log::SrlogHeaderFields f = v11_fields();
    f.mass_rate_hz = 4;  // 10 % 4 != 0
    CHECK_THROWS_AS(star::log::SrlogWriter::header_json(f),
                    std::invalid_argument);
  }
  {
    star::log::SrlogHeaderFields f = v11_fields();
    f.env_rate_hz = 20;  // above the truth rate is not a divisor either
    CHECK_THROWS_AS(star::log::SrlogWriter::header_json(f),
                    std::invalid_argument);
  }
  // The source subset must come from the canonical vocabulary, in canonical
  // order, without duplicates.
  {
    star::log::SrlogHeaderFields f = v11_fields();
    f.force_sources = {"gravity", "warp"};
    CHECK_THROWS_AS(star::log::SrlogWriter::header_json(f),
                    std::invalid_argument);
  }
  {
    star::log::SrlogHeaderFields f = v11_fields();
    f.force_sources = {"gravity", "gravity"};
    CHECK_THROWS_AS(star::log::SrlogWriter::header_json(f),
                    std::invalid_argument);
  }
  {
    star::log::SrlogHeaderFields f = v11_fields();
    f.force_sources = {"thrust", "gravity"};  // canonical order is reversed
    CHECK_THROWS_AS(star::log::SrlogWriter::header_json(f),
                    std::invalid_argument);
  }
  // A half-declared forces group is rejected, not reconciled.
  {
    star::log::SrlogHeaderFields f = v11_fields();
    f.force_sources.clear();  // rate still nonzero
    CHECK_THROWS_AS(star::log::SrlogWriter::header_json(f),
                    std::invalid_argument);
  }
  {
    star::log::SrlogHeaderFields f = v11_fields();
    f.forces_rate_hz = 0;  // sources still declared
    CHECK_THROWS_AS(star::log::SrlogWriter::header_json(f),
                    std::invalid_argument);
  }
  // The full canonical vocabulary in canonical order is valid.
  {
    star::log::SrlogHeaderFields f = v11_fields();
    f.force_sources = {"gravity", "thirdbody", "srp",      "drag", "aero",
                       "thrust",  "rcs",       "gravgrad", "wheel"};
    CHECK(star::log::SrlogWriter::header_json(f).find("\"f_wheel_b_n\"") !=
          std::string::npos);
  }
  // The constructor rejects before creating the output file: a refused
  // declaration must not leave a truncated file behind.
  {
    star::log::SrlogHeaderFields f = v11_fields();
    f.forces_rate_hz = 3;
    const std::string path = "test_srlog_v11_never_created.srlog";
    CHECK_THROWS_AS(star::log::SrlogWriter(path, f), std::invalid_argument);
    std::ifstream probe(path, std::ios::binary);
    CHECK_FALSE(static_cast<bool>(probe));
  }
}

TEST_CASE("srlog_v11_write_calls_guard_declaration") {
  const std::string path = "test_srlog_v11_guards.srlog";
  {
    // Only the mass group is enabled: writing the undeclared groups is a
    // programming error (the header never announced them), while the
    // declared group writes normally.
    star::log::SrlogHeaderFields f = v11_fields();
    f.force_sources.clear();
    f.forces_rate_hz = 0;
    f.env_rate_hz = 0;
    star::log::SrlogWriter writer(path, f);
    CHECK_THROWS_AS(
        writer.write_forces(0.0, std::vector<star::log::ForceSourceSample>{}),
        std::logic_error);
    CHECK_THROWS_AS(writer.write_env(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    std::logic_error);
    const double inertia[6] = {1.0, 0.0, 0.0, 1.0, 0.0, 1.0};
    CHECK_NOTHROW(writer.write_mass(0.0, 1.0, Eigen::Vector3d::Zero(),
                                    inertia));
    writer.close();
  }
  {
    // Only the forces group is enabled, with two sources: a one-sample
    // record is a caller bug the writer must refuse rather than emit a
    // short (corrupt) record, and the other optional writes are guarded.
    star::log::SrlogHeaderFields f = v11_fields();
    f.mass_rate_hz = 0;
    f.env_rate_hz = 0;
    star::log::SrlogWriter writer(path, f);
    std::vector<star::log::ForceSourceSample> one(1);
    CHECK_THROWS_AS(writer.write_forces(0.0, one), std::invalid_argument);
    const double inertia[6] = {1.0, 0.0, 0.0, 1.0, 0.0, 1.0};
    CHECK_THROWS_AS(writer.write_mass(0.0, 1.0, Eigen::Vector3d::Zero(),
                                      inertia),
                    std::logic_error);
    CHECK_THROWS_AS(writer.write_env(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    std::logic_error);
    writer.close();
  }
  std::remove(path.c_str());
}

TEST_CASE("srlog_v11_double_write_is_byte_identical") {
  // Whole-file determinism (FR-21): the same declaration and records must
  // produce the same bytes, twice, including the v1.1 groups.
  auto write_once = [](const std::string& path) {
    star::log::SrlogWriter writer(path, v11_fields());
    writer.write_event(0.0, 1, "run_start");
    std::vector<star::log::ForceSourceSample> samples(2);
    samples[0].force_b_n = Eigen::Vector3d(1.0, 2.0, 3.0);
    samples[0].torque_b_nm = Eigen::Vector3d(0.0, 0.0, 0.0);
    samples[1].force_b_n = Eigen::Vector3d(0.0, 0.0, 900.0);
    samples[1].torque_b_nm = Eigen::Vector3d(0.5, 0.0, -0.25);
    writer.write_forces(0.0, samples);
    const double inertia[6] = {10.0, 0.5, 0.25, 12.0, 0.125, 8.0};
    writer.write_mass(0.0, 150.0, Eigen::Vector3d(0.1, 0.0, -0.2), inertia);
    writer.write_env(0.0, 1200.5, 0.8, 24000.0, 0.4135, 0.5061);
    writer.write_event(600.0, 2, "run_end");
    writer.close();
  };
  const std::string p1 = "test_srlog_v11_det_a.srlog";
  const std::string p2 = "test_srlog_v11_det_b.srlog";
  write_once(p1);
  write_once(p2);
  const std::vector<unsigned char> a = read_all_bytes(p1);
  const std::vector<unsigned char> b = read_all_bytes(p2);
  std::remove(p1.c_str());
  std::remove(p2.c_str());
  REQUIRE(!a.empty());
  CHECK(a == b);
}
