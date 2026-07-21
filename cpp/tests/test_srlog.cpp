// SRLOG v1 writer byte-level tests: headers round-trip against independently
// assembled reference byte sequences (contract section 2 /
// docs/formats/srlog_v1.md), and the record stream layout is verified field
// by field, for the always-present Phase 1 groups, the v1.1 vehicle channel
// groups, and the v1.2 GNC channel groups. The reference bytes are
// synthesized here in test code - binary fixtures are never committed
// (contract section 11).
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
      "{\"format\":{\"name\":\"SRLOG\",\"major\":1,\"minor\":3},"
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
  CHECK(read_le<std::uint16_t>(bytes, 10) == 3);   // version_minor
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
      "{\"format\":{\"name\":\"SRLOG\",\"major\":1,\"minor\":3},"
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

namespace {

// Baseline v1.2 declaration used by the GNC-group tests: two sensors (imu at
// the cycle rate, camera decimated with two landmarks), the nav estimator
// groups at n = 7 / m_max = 3, and the applied-command group, with no v1.1
// vehicle groups so the group indices are compact.
star::log::SrlogHeaderFields v12_fields() {
  star::log::SrlogHeaderFields fields;
  fields.core_version = "0.6.0-test";
  fields.git_hash = "0123456789abcdef0123456789abcdef01234567";
  fields.config_sha256 =
      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
  fields.master_seed = 1234567890ULL;
  fields.oracle = false;
  fields.epoch_utc = "2026-01-01T00:00:00Z";
  fields.central_body = "earth";
  fields.truth_rate_hz = 10;
  fields.cycle_rate_hz = 100;
  fields.latency_cycles = 2;
  fields.sensors = {{"imu", 100, 0}, {"camera", 10, 2}};
  // The camera group and its header echo are declared together (the writer
  // rejects either half alone). Anisotropic focal lengths, an off-axis
  // principal point, a non-identity mount rotation, and a nonzero mount
  // station, so a byte-level header assertion cannot pass while any one of
  // the seven doubles is transposed with another.
  fields.camera_echo_present = true;
  fields.camera.fx_px = 800.0;
  fields.camera.fy_px = 600.0;
  fields.camera.cx_px = 511.5;
  fields.camera.cy_px = 383.5;
  fields.camera.width_px = 1024;
  fields.camera.height_px = 768;
  fields.camera.q_b2c[0] = 0.9659258262890683;
  fields.camera.q_b2c[1] = 0.0;
  fields.camera.q_b2c[2] = 0.25881904510252074;
  fields.camera.q_b2c[3] = 0.0;
  fields.camera.r_cam_b_m[0] = 0.5;
  fields.camera.r_cam_b_m[1] = -0.25;
  fields.camera.r_cam_b_m[2] = 0.125;
  fields.nav_est_rate_hz = 100;
  fields.nav_state_dim = 7;
  fields.nav_err_enabled = true;
  fields.nav_innov_enabled = true;
  fields.nav_innov_max_dim = 3;
  fields.gnc_cmd_rate_hz = 100;
  return fields;
}

}  // namespace

TEST_CASE("srlog_v12_header_declares_gnc_groups") {
  // Reference JSON assembled independently of the serializer (format doc
  // sections 3 and 3.2): the "gnc" object follows central_body, and the
  // declared v1.2 groups follow the v1.1 groups (none here) in the fixed
  // order sensors.* (canonical kind order), nav.est, nav.err, nav.innov,
  // gnc.cmd.
  const std::string expected_json =
      "{\"format\":{\"name\":\"SRLOG\",\"major\":1,\"minor\":3},"
      "\"producer\":{\"core_version\":\"0.6.0-test\","
      "\"git_hash\":\"0123456789abcdef0123456789abcdef01234567\"},"
      "\"config_sha256\":"
      "\"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\","
      "\"master_seed\":\"1234567890\",\"oracle\":false,"
      "\"epoch_utc\":\"2026-01-01T00:00:00Z\",\"central_body\":\"earth\","
      "\"gnc\":{\"cycle_rate_hz\":100,\"latency_cycles\":2,"
      "\"sensors\":[\"imu\",\"camera\"],"
      // The v1.3 camera echo. The hex digits are the IEEE-754 binary64 bit
      // patterns of 800, 600, 511.5, 383.5, the mount quaternion, and the
      // mount station - written out longhand rather than computed here, so
      // this assertion is a statement about the bytes on disk and not a
      // re-run of the encoder it checks.
      "\"camera\":{\"float_encoding\":\"ieee754-binary64-hex\","
      "\"width_px\":1024,\"height_px\":768,"
      "\"fx_px\":\"4089000000000000\",\"fy_px\":\"4082c00000000000\","
      "\"cx_px\":\"407ff80000000000\",\"cy_px\":\"4077f80000000000\","
      "\"q_b2c\":[\"3feee8dd4748bf15\",\"0000000000000000\","
      "\"3fd0907dc1930690\",\"0000000000000000\"],"
      "\"r_cam_b_m\":[\"3fe0000000000000\",\"bfd0000000000000\","
      "\"3fc0000000000000\"]}},"
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
      "{\"name\":\"sensors.imu\",\"rate_hz\":100,\"channels\":["
      "{\"name\":\"t_s\",\"dtype\":\"f64\",\"units\":\"s\",\"frame\":\"\"},"
      "{\"name\":\"dtheta_b_rad\",\"dtype\":\"f64[3]\",\"units\":\"rad\","
      "\"frame\":\"body\"},"
      "{\"name\":\"dv_b_mps\",\"dtype\":\"f64[3]\",\"units\":\"m/s\","
      "\"frame\":\"body\"}"
      "]},"
      "{\"name\":\"sensors.camera\",\"rate_hz\":10,\"channels\":["
      "{\"name\":\"t_s\",\"dtype\":\"f64\",\"units\":\"s\",\"frame\":\"\"},"
      "{\"name\":\"r_m\",\"dtype\":\"f64[3]\",\"units\":\"m\",\"frame\":\"GCRF\"},"
      "{\"name\":\"q_i2b\",\"dtype\":\"f64[4]\",\"units\":\"1\","
      "\"frame\":\"GCRF->body Hamilton scalar-first\"},"
      "{\"name\":\"px_uv\",\"dtype\":\"f64[4]\",\"units\":\"px\","
      "\"frame\":\"image\"}"
      "]},"
      "{\"name\":\"nav.est\",\"rate_hz\":100,\"channels\":["
      "{\"name\":\"t_s\",\"dtype\":\"f64\",\"units\":\"s\",\"frame\":\"\"},"
      "{\"name\":\"x_hat\",\"dtype\":\"f64[7]\",\"units\":\"\",\"frame\":\"\"},"
      "{\"name\":\"P\",\"dtype\":\"f64[28]\",\"units\":\"\",\"frame\":\"\"}"
      "]},"
      "{\"name\":\"nav.err\",\"rate_hz\":100,\"channels\":["
      "{\"name\":\"t_s\",\"dtype\":\"f64\",\"units\":\"s\",\"frame\":\"\"},"
      "{\"name\":\"e\",\"dtype\":\"f64[7]\",\"units\":\"\",\"frame\":\"\"}"
      "]},"
      "{\"name\":\"nav.innov\",\"rate_hz\":0,\"channels\":["
      "{\"name\":\"t_s\",\"dtype\":\"f64\",\"units\":\"s\",\"frame\":\"\"},"
      "{\"name\":\"sensor_id\",\"dtype\":\"u32\",\"units\":\"1\",\"frame\":\"\"},"
      "{\"name\":\"m\",\"dtype\":\"u32\",\"units\":\"1\",\"frame\":\"\"},"
      "{\"name\":\"y\",\"dtype\":\"f64[3]\",\"units\":\"\",\"frame\":\"\"},"
      "{\"name\":\"S\",\"dtype\":\"f64[6]\",\"units\":\"\",\"frame\":\"\"}"
      "]},"
      "{\"name\":\"gnc.cmd\",\"rate_hz\":100,\"channels\":["
      "{\"name\":\"t_s\",\"dtype\":\"f64\",\"units\":\"s\",\"frame\":\"\"},"
      "{\"name\":\"tau_b_nm\",\"dtype\":\"f64[3]\",\"units\":\"N*m\","
      "\"frame\":\"body\"},"
      "{\"name\":\"q_cmd_i2b\",\"dtype\":\"f64[4]\",\"units\":\"1\","
      "\"frame\":\"GCRF->body Hamilton scalar-first\"},"
      "{\"name\":\"w_cmd_b_radps\",\"dtype\":\"f64[3]\",\"units\":\"rad/s\","
      "\"frame\":\"body\"},"
      "{\"name\":\"valid\",\"dtype\":\"u32\",\"units\":\"1\",\"frame\":\"\"}"
      "]}]}";
  CHECK(star::log::SrlogWriter::header_json(v12_fields()) == expected_json);

  // A v1.2 declaration with no GNC groups carries no "gnc" key: pre-Phase-6
  // configurations differ from v1.1 output in the version words alone.
  star::log::SrlogHeaderFields plain = v12_fields();
  plain.cycle_rate_hz = 0;
  plain.latency_cycles = 0;
  plain.sensors.clear();
  plain.camera_echo_present = false;  // the echo follows its group
  plain.nav_est_rate_hz = 0;
  plain.nav_state_dim = 0;
  plain.nav_err_enabled = false;
  plain.nav_innov_enabled = false;
  plain.nav_innov_max_dim = 0;
  plain.gnc_cmd_rate_hz = 0;
  const std::string json = star::log::SrlogWriter::header_json(plain);
  CHECK(json.find("\"gnc\"") == std::string::npos);
  CHECK(json.find("\"sensors.") == std::string::npos);
  CHECK(json.find("\"nav.") == std::string::npos);
}

TEST_CASE("srlog_v12_record_stream_roundtrip") {
  // Group indices with no v1.1 groups declared: truth 0, events 1, then
  // sensors.imu 2, sensors.camera 3, nav.est 4, nav.err 5, nav.innov 6,
  // gnc.cmd 7 (declaration order).
  const std::string path = "test_srlog_v12_roundtrip.srlog";
  {
    star::log::SrlogWriter writer(path, v12_fields());
    writer.write_sensor_imu(0.01, Eigen::Vector3d(1e-4, -2e-4, 3e-4),
                            Eigen::Vector3d(0.05, 0.0, -0.01));
    const double q_cam[4] = {1.0, 0.0, 0.0, 0.0};
    const double px[4] = {320.5, 240.25, 100.0, 900.75};
    writer.write_sensor_camera(0.1, Eigen::Vector3d(7.0e6, 0.0, 0.0), q_cam,
                               px, 4);
    const double x_hat[7] = {1.0, 0.0, 0.0, 0.0, 0.01, -0.02, 0.03};
    double p_upper[28];
    for (int i = 0; i < 28; ++i) p_upper[i] = 0.5 * i;
    writer.write_nav_est(0.01, x_hat, 7, p_upper, 28);
    const double e[7] = {0.0, 1e-6, -1e-6, 0.0, 1e-5, 0.0, -1e-5};
    writer.write_nav_err(0.01, e, 7);
    const double y[3] = {0.25, -0.5, 0.0};
    const double s_upper[6] = {2.0, 0.1, 0.0, 3.0, 0.0, 4.0};
    writer.write_nav_innov(0.01, 1, 2, y, 3, s_upper, 6);
    const double q_cmd[4] = {0.0, 1.0, 0.0, 0.0};
    writer.write_gnc_cmd(0.01, Eigen::Vector3d(0.05, 0.0, -0.05), q_cmd,
                         Eigen::Vector3d::Zero(), 1);
    writer.close();
  }
  const std::vector<unsigned char> bytes = read_all_bytes(path);
  std::remove(path.c_str());

  const std::uint32_t json_len = read_le<std::uint32_t>(bytes, 12);
  std::size_t off = 16 + json_len;

  // sensors.imu: index 2; payload = t_s + 2 x f64[3] = 56 bytes.
  CHECK(read_le<std::uint16_t>(bytes, off) == 2);
  CHECK(read_le<double>(bytes, off + 2) == 0.01);
  CHECK(read_le<double>(bytes, off + 10) == 1e-4);   // dtheta_b_rad[0]
  CHECK(read_le<double>(bytes, off + 26) == 3e-4);   // dtheta_b_rad[2]
  CHECK(read_le<double>(bytes, off + 34) == 0.05);   // dv_b_mps[0]
  CHECK(read_le<double>(bytes, off + 50) == -0.01);  // dv_b_mps[2]
  off += 2 + 56;

  // sensors.camera: index 3; payload = t_s + r(3) + q(4) + px(4) = 96 bytes.
  CHECK(read_le<std::uint16_t>(bytes, off) == 3);
  CHECK(read_le<double>(bytes, off + 2) == 0.1);
  CHECK(read_le<double>(bytes, off + 10) == 7.0e6);    // r_m[0]
  CHECK(read_le<double>(bytes, off + 34) == 1.0);      // q_i2b[0] scalar first
  CHECK(read_le<double>(bytes, off + 66) == 320.5);    // px_uv[0]
  CHECK(read_le<double>(bytes, off + 90) == 900.75);   // px_uv[3]
  off += 2 + 96;

  // nav.est: index 4; payload = t_s + x_hat(7) + P(28) = 288 bytes.
  CHECK(read_le<std::uint16_t>(bytes, off) == 4);
  CHECK(read_le<double>(bytes, off + 2) == 0.01);
  CHECK(read_le<double>(bytes, off + 10) == 1.0);      // x_hat[0] = q_w
  CHECK(read_le<double>(bytes, off + 58) == 0.03);     // x_hat[6]
  CHECK(read_le<double>(bytes, off + 66) == 0.0);      // P[0]
  CHECK(read_le<double>(bytes, off + 282) == 13.5);    // P[27] = 0.5 * 27
  off += 2 + 288;

  // nav.err: index 5; payload = t_s + e(7) = 64 bytes.
  CHECK(read_le<std::uint16_t>(bytes, off) == 5);
  CHECK(read_le<double>(bytes, off + 18) == 1e-6);     // e[1]
  CHECK(read_le<double>(bytes, off + 58) == -1e-5);    // e[6]
  off += 2 + 64;

  // nav.innov: index 6; payload = t_s + u32 + u32 + y(3) + S(6) = 88 bytes.
  CHECK(read_le<std::uint16_t>(bytes, off) == 6);
  CHECK(read_le<double>(bytes, off + 2) == 0.01);
  CHECK(read_le<std::uint32_t>(bytes, off + 10) == 1);  // sensor_id (camera)
  CHECK(read_le<std::uint32_t>(bytes, off + 14) == 2);  // m
  CHECK(read_le<double>(bytes, off + 18) == 0.25);      // y[0]
  CHECK(read_le<double>(bytes, off + 34) == 0.0);       // y[2] zero-filled
  CHECK(read_le<double>(bytes, off + 42) == 2.0);       // S[0] = S_00
  CHECK(read_le<double>(bytes, off + 82) == 4.0);       // S[5] = S_22
  off += 2 + 88;

  // gnc.cmd: index 7; payload = t_s + tau(3) + q(4) + w(3) + u32 = 92 bytes.
  CHECK(read_le<std::uint16_t>(bytes, off) == 7);
  CHECK(read_le<double>(bytes, off + 10) == 0.05);      // tau_b_nm[0]
  CHECK(read_le<double>(bytes, off + 34) == 0.0);       // q_cmd_i2b[0]
  CHECK(read_le<double>(bytes, off + 42) == 1.0);       // q_cmd_i2b[1]
  CHECK(read_le<std::uint32_t>(bytes, off + 90) == 1);  // valid
  off += 2 + 92;

  // No footer, no trailing bytes.
  CHECK(off == bytes.size());
}

TEST_CASE("srlog_v12_declaration_validation") {
  // Every rejection happens in header_json (shared with the constructor).
  // A GNC group without a cycle rate has no grid to decimate from.
  {
    star::log::SrlogHeaderFields f = v12_fields();
    f.cycle_rate_hz = 0;
    CHECK_THROWS_AS(star::log::SrlogWriter::header_json(f),
                    std::invalid_argument);
  }
  // Sensor rates must divide the cycle rate exactly.
  {
    star::log::SrlogHeaderFields f = v12_fields();
    f.sensors = {{"imu", 3, 0}};  // 100 % 3 != 0
    f.camera_echo_present = false;  // isolate the rate defect
    CHECK_THROWS_AS(star::log::SrlogWriter::header_json(f),
                    std::invalid_argument);
  }
  // The kind subset must come from the canonical vocabulary, in canonical
  // order, without duplicates.
  {
    star::log::SrlogHeaderFields f = v12_fields();
    f.sensors = {{"lidar", 10, 0}};
    f.camera_echo_present = false;  // isolate the vocabulary defect
    CHECK_THROWS_AS(star::log::SrlogWriter::header_json(f),
                    std::invalid_argument);
  }
  {
    star::log::SrlogHeaderFields f = v12_fields();
    f.sensors = {{"imu", 100, 0}, {"imu", 50, 0}};
    f.camera_echo_present = false;  // isolate the duplicate-kind defect
    CHECK_THROWS_AS(star::log::SrlogWriter::header_json(f),
                    std::invalid_argument);
  }
  {
    star::log::SrlogHeaderFields f = v12_fields();
    f.sensors = {{"camera", 10, 2}, {"imu", 100, 0}};  // canonical order broken
    CHECK_THROWS_AS(star::log::SrlogWriter::header_json(f),
                    std::invalid_argument);
  }
  // Landmarks are a camera-only parameter.
  {
    star::log::SrlogHeaderFields f = v12_fields();
    f.sensors = {{"imu", 100, 3}};
    f.camera_echo_present = false;  // isolate the landmark-parameter defect
    CHECK_THROWS_AS(star::log::SrlogWriter::header_json(f),
                    std::invalid_argument);
  }
  // v1.3: the camera group and its header echo are declared together, in
  // both directions. A camera log without the echo would silently reopen
  // the gap exit criterion 7's intrinsics clause exists to close, and an
  // echo without a group would describe a camera the file does not carry.
  {
    star::log::SrlogHeaderFields f = v12_fields();
    f.camera_echo_present = false;  // camera group, no echo
    CHECK_THROWS_AS(star::log::SrlogWriter::header_json(f),
                    std::invalid_argument);
  }
  {
    star::log::SrlogHeaderFields f = v12_fields();
    f.sensors = {{"imu", 100, 0}};  // echo, no camera group
    CHECK_THROWS_AS(star::log::SrlogWriter::header_json(f),
                    std::invalid_argument);
  }
  // A singular eq:camera:K would make the echo unusable by the consumer it
  // exists for, so it is refused at the writer boundary.
  {
    star::log::SrlogHeaderFields f = v12_fields();
    f.camera.fx_px = 0.0;
    CHECK_THROWS_AS(star::log::SrlogWriter::header_json(f),
                    std::invalid_argument);
  }
  {
    star::log::SrlogHeaderFields f = v12_fields();
    f.camera.height_px = 0;
    CHECK_THROWS_AS(star::log::SrlogWriter::header_json(f),
                    std::invalid_argument);
  }
  // nav.est needs rate and dimension jointly.
  {
    star::log::SrlogHeaderFields f = v12_fields();
    f.nav_state_dim = 0;
    CHECK_THROWS_AS(star::log::SrlogWriter::header_json(f),
                    std::invalid_argument);
  }
  // nav.err cannot exist without nav.est (shared rate and dimension).
  {
    star::log::SrlogHeaderFields f = v12_fields();
    f.nav_est_rate_hz = 0;
    f.nav_state_dim = 0;
    CHECK_THROWS_AS(star::log::SrlogWriter::header_json(f),
                    std::invalid_argument);
  }
  // The covariance dimension qualifies the nav.est declaration and cannot
  // stand alone.
  {
    star::log::SrlogHeaderFields f = v12_fields();
    f.nav_est_rate_hz = 0;
    f.nav_state_dim = 0;
    f.nav_err_enabled = false;
    f.nav_cov_dim = 15;
    CHECK_THROWS_AS(star::log::SrlogWriter::header_json(f),
                    std::invalid_argument);
  }
  // An independently declared covariance dimension changes only the P
  // channel: the error-state EKF layout (a later workstream) declares
  // n = 16 (q, v, p, b_g, b_a) with m = 15, so P is f64[120] while x_hat
  // and nav.err.e stay f64[16].
  {
    star::log::SrlogHeaderFields f = v12_fields();
    f.nav_state_dim = 16;
    f.nav_cov_dim = 15;
    const std::string j = star::log::SrlogWriter::header_json(f);
    CHECK(j.find("{\"name\":\"x_hat\",\"dtype\":\"f64[16]\"") !=
          std::string::npos);
    CHECK(j.find("{\"name\":\"P\",\"dtype\":\"f64[120]\"") !=
          std::string::npos);
    CHECK(j.find("{\"name\":\"e\",\"dtype\":\"f64[16]\"") !=
          std::string::npos);
  }
  // nav.innov needs the sensor-identity table its records index.
  {
    star::log::SrlogHeaderFields f = v12_fields();
    f.sensors.clear();
    CHECK_THROWS_AS(star::log::SrlogWriter::header_json(f),
                    std::invalid_argument);
  }
  // nav.innov enable flag and maximum dimension are declared jointly.
  {
    star::log::SrlogHeaderFields f = v12_fields();
    f.nav_innov_max_dim = 0;
    CHECK_THROWS_AS(star::log::SrlogWriter::header_json(f),
                    std::invalid_argument);
  }
  // gnc.cmd rate must divide the cycle rate.
  {
    star::log::SrlogHeaderFields f = v12_fields();
    f.gnc_cmd_rate_hz = 30;
    CHECK_THROWS_AS(star::log::SrlogWriter::header_json(f),
                    std::invalid_argument);
  }
  // The constructor rejects before creating the output file.
  {
    star::log::SrlogHeaderFields f = v12_fields();
    f.sensors = {{"warp", 10, 0}};
    const std::string path = "test_srlog_v12_never_created.srlog";
    CHECK_THROWS_AS(star::log::SrlogWriter(path, f), std::invalid_argument);
    std::ifstream probe(path, std::ios::binary);
    CHECK_FALSE(static_cast<bool>(probe));
  }
}

TEST_CASE("srlog_v12_write_calls_guard_declaration_and_dimensions") {
  const std::string path = "test_srlog_v12_guards.srlog";
  {
    // Only sensors.imu and gnc.cmd are declared: the undeclared v1.2 writes
    // are programming errors, the declared ones write normally.
    star::log::SrlogHeaderFields f = v12_fields();
    f.sensors = {{"imu", 100, 0}};
    f.camera_echo_present = false;  // the echo follows its group
    f.nav_est_rate_hz = 0;
    f.nav_state_dim = 0;
    f.nav_err_enabled = false;
    f.nav_innov_enabled = false;
    f.nav_innov_max_dim = 0;
    star::log::SrlogWriter writer(path, f);
    const double x[7] = {1, 0, 0, 0, 0, 0, 0};
    const double p[28] = {0};
    CHECK_THROWS_AS(writer.write_nav_est(0.0, x, 7, p, 28), std::logic_error);
    CHECK_THROWS_AS(writer.write_nav_err(0.0, x, 7), std::logic_error);
    const double y[3] = {0};
    const double s[6] = {0};
    CHECK_THROWS_AS(writer.write_nav_innov(0.0, 0, 1, y, 3, s, 6),
                    std::logic_error);
    const double q[4] = {1, 0, 0, 0};
    CHECK_THROWS_AS(writer.write_sensor_startracker(0.0, q, 1),
                    std::logic_error);
    CHECK_NOTHROW(writer.write_sensor_imu(0.01, Eigen::Vector3d::Zero(),
                                          Eigen::Vector3d::Zero()));
    CHECK_NOTHROW(writer.write_gnc_cmd(0.01, Eigen::Vector3d::Zero(), q,
                                       Eigen::Vector3d::Zero(), 0));
    writer.close();
  }
  {
    // Dimension mismatches are caller bugs the writer must refuse rather
    // than emit a short (corrupt) fixed-stride record.
    star::log::SrlogWriter writer(path, v12_fields());
    const double x[7] = {1, 0, 0, 0, 0, 0, 0};
    const double p[28] = {0};
    CHECK_THROWS_AS(writer.write_nav_est(0.0, x, 6, p, 28),
                    std::invalid_argument);
    CHECK_THROWS_AS(writer.write_nav_est(0.0, x, 7, p, 27),
                    std::invalid_argument);
    CHECK_THROWS_AS(writer.write_nav_err(0.0, x, 6), std::invalid_argument);
    const double y[3] = {0};
    const double s[6] = {0};
    CHECK_THROWS_AS(writer.write_nav_innov(0.0, 0, 4, y, 3, s, 6),
                    std::invalid_argument);  // m > m_max
    CHECK_THROWS_AS(writer.write_nav_innov(0.0, 0, 0, y, 3, s, 6),
                    std::invalid_argument);  // m == 0
    // sensor_id indexes the header's gnc.sensors array, which this file
    // declares with two entries. An id outside it is a record no reader can
    // resolve - IndexError at analysis time, or a silent misattribution if a
    // tool clamps - and misattribution is exactly what the NEES/NIS result
    // cannot survive. Every other v1.2 dimension was already checked here.
    CHECK_THROWS_WITH_AS(writer.write_nav_innov(0.0, 2, 2, y, 3, s, 6),
                         doctest::Contains("sensor_id 2"),
                         std::invalid_argument);
    CHECK_NOTHROW(writer.write_nav_innov(0.0, 1, 2, y, 3, s, 6));  // in range
    const double q[4] = {1, 0, 0, 0};
    const double px[2] = {0.0, 0.0};
    CHECK_THROWS_AS(
        writer.write_sensor_camera(0.0, Eigen::Vector3d::Zero(), q, px, 2),
        std::invalid_argument);  // declaration fixes 2 landmarks = 4 values
    writer.close();
  }
  {
    // Independent covariance dimension (the EKF layout reservation): with
    // n = 16 and m = 15 declared, x_hat carries 16 doubles and P exactly
    // 120 - the n-derived 136 is a caller bug.
    star::log::SrlogHeaderFields f = v12_fields();
    f.nav_state_dim = 16;
    f.nav_cov_dim = 15;
    star::log::SrlogWriter writer(path, f);
    double x16[16] = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    double p136[136] = {0};
    CHECK_NOTHROW(writer.write_nav_est(0.0, x16, 16, p136, 120));
    CHECK_THROWS_AS(writer.write_nav_est(0.01, x16, 16, p136, 136),
                    std::invalid_argument);
    // nav.err stays at the STATE dimension n = 16.
    CHECK_NOTHROW(writer.write_nav_err(0.0, x16, 16));
    CHECK_THROWS_AS(writer.write_nav_err(0.01, x16, 15),
                    std::invalid_argument);
    writer.close();
  }
  std::remove(path.c_str());
}

TEST_CASE("srlog_v12_double_write_is_byte_identical") {
  // Whole-file determinism (FR-21) including every v1.2 group.
  auto write_once = [](const std::string& path) {
    star::log::SrlogWriter writer(path, v12_fields());
    writer.write_event(0.0, 1, "run_start");
    writer.write_sensor_imu(0.01, Eigen::Vector3d(1e-4, -2e-4, 3e-4),
                            Eigen::Vector3d(0.05, 0.0, -0.01));
    const double q_cam[4] = {1.0, 0.0, 0.0, 0.0};
    const double px[4] = {320.5, 240.25, 100.0, 900.75};
    writer.write_sensor_camera(0.1, Eigen::Vector3d(7.0e6, 0.0, 0.0), q_cam,
                               px, 4);
    const double x_hat[7] = {1.0, 0.0, 0.0, 0.0, 0.01, -0.02, 0.03};
    double p_upper[28];
    for (int i = 0; i < 28; ++i) p_upper[i] = 0.5 * i;
    writer.write_nav_est(0.01, x_hat, 7, p_upper, 28);
    const double e[7] = {0.0, 1e-6, -1e-6, 0.0, 1e-5, 0.0, -1e-5};
    writer.write_nav_err(0.01, e, 7);
    const double y[3] = {0.25, -0.5, 0.0};
    const double s_upper[6] = {2.0, 0.1, 0.0, 3.0, 0.0, 4.0};
    writer.write_nav_innov(0.01, 1, 2, y, 3, s_upper, 6);
    const double q_cmd[4] = {0.0, 1.0, 0.0, 0.0};
    writer.write_gnc_cmd(0.01, Eigen::Vector3d(0.05, 0.0, -0.05), q_cmd,
                         Eigen::Vector3d::Zero(), 1);
    writer.write_event(600.0, 2, "run_end");
    writer.close();
  };
  const std::string p1 = "test_srlog_v12_det_a.srlog";
  const std::string p2 = "test_srlog_v12_det_b.srlog";
  write_once(p1);
  write_once(p2);
  const std::vector<unsigned char> a = read_all_bytes(p1);
  const std::vector<unsigned char> b = read_all_bytes(p2);
  std::remove(p1.c_str());
  std::remove(p2.c_str());
  REQUIRE(!a.empty());
  CHECK(a == b);
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
