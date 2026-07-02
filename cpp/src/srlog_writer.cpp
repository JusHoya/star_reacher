// SRLOG v1.0 writer implementation. The byte layout is normative in
// docs/formats/srlog_v1.md; keep the two in lockstep.
#include "star/srlog_writer.hpp"

#include <cstring>
#include <stdexcept>

namespace star {
namespace log {

namespace {

// "SRLOG" NUL CR LF: the NUL stops C-string readers early and the CR/LF pair
// is corrupted by any text-mode transfer, so a mangled file fails the magic
// check immediately instead of misparsing later.
constexpr unsigned char kMagic[8] = {0x53, 0x52, 0x4C, 0x4F,
                                     0x47, 0x00, 0x0D, 0x0A};

bool host_is_little_endian() {
  const std::uint16_t probe = 0x0102;
  unsigned char bytes[2];
  std::memcpy(bytes, &probe, 2);
  return bytes[0] == 0x02;
}

// Minimal JSON string escaper. Header strings are ASCII in practice (hex
// digests, ISO-8601 epochs, body names), but escaping defensively means a
// hostile or buggy input can never produce syntactically invalid JSON.
void append_json_string(std::string& out, std::string_view s) {
  out += '"';
  for (unsigned char c : s) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      default:
        if (c < 0x20) {
          static const char* hex = "0123456789abcdef";
          out += "\\u00";
          out += hex[(c >> 4) & 0xF];
          out += hex[c & 0xF];
        } else {
          out += static_cast<char>(c);
        }
    }
  }
  out += '"';
}

// One channel entry with the fixed key order name, dtype, units, frame.
void append_channel(std::string& out, const char* name, const char* dtype,
                    const char* units, const char* frame) {
  out += "{\"name\":";
  append_json_string(out, name);
  out += ",\"dtype\":";
  append_json_string(out, dtype);
  out += ",\"units\":";
  append_json_string(out, units);
  out += ",\"frame\":";
  append_json_string(out, frame);
  out += '}';
}

}  // namespace

std::string SrlogWriter::header_json(const SrlogHeaderFields& fields) {
  // Hand-rolled, compact, fixed key order (contract section 2). std::to_string
  // on integer types is locale-independent, so no locale can perturb the
  // bytes. No floats appear anywhere in the header by design.
  std::string j;
  j.reserve(1024);
  j += "{\"format\":{\"name\":\"SRLOG\",\"major\":1,\"minor\":0}";
  j += ",\"producer\":{\"core_version\":";
  append_json_string(j, fields.core_version);
  j += ",\"git_hash\":";
  append_json_string(j, fields.git_hash);
  j += "},\"config_sha256\":";
  append_json_string(j, fields.config_sha256);
  // master_seed rides as a decimal string: JSON numbers cannot represent all
  // u64 values without precision loss in common readers.
  j += ",\"master_seed\":";
  append_json_string(j, std::to_string(fields.master_seed));
  j += ",\"oracle\":";
  j += fields.oracle ? "true" : "false";
  j += ",\"epoch_utc\":";
  append_json_string(j, fields.epoch_utc);
  j += ",\"central_body\":";
  append_json_string(j, fields.central_body);
  j += ",\"groups\":[";
  // Group 0: truth.
  j += "{\"name\":\"truth\",\"rate_hz\":";
  j += std::to_string(fields.truth_rate_hz);
  j += ",\"channels\":[";
  append_channel(j, "t_s", "f64", "s", "");
  j += ',';
  append_channel(j, "r_m", "f64[3]", "m", "GCRF");
  j += ',';
  append_channel(j, "v_mps", "f64[3]", "m/s", "GCRF");
  j += ',';
  append_channel(j, "q_i2b", "f64[4]", "1", "GCRF->body Hamilton scalar-first");
  j += ',';
  append_channel(j, "w_b_radps", "f64[3]", "rad/s", "body");
  j += ',';
  append_channel(j, "mass_kg", "f64", "kg", "");
  j += "]}";
  // Group 1: events (rate_hz 0 marks an aperiodic stream).
  j += ",{\"name\":\"events\",\"rate_hz\":0,\"channels\":[";
  append_channel(j, "t_s", "f64", "s", "");
  j += ',';
  append_channel(j, "code", "u32", "1", "");
  j += ',';
  append_channel(j, "detail", "str16", "", "");
  j += "]}]}";
  return j;
}

SrlogWriter::SrlogWriter(const std::string& path,
                         const SrlogHeaderFields& fields) {
  if (!host_is_little_endian()) {
    throw std::logic_error(
        "SRLOG writer requires a little-endian host; big-endian is not a "
        "supported platform");
  }
  out_.open(path, std::ios::binary | std::ios::trunc);
  if (!out_) {
    throw std::runtime_error("SRLOG writer: cannot open output file: " + path);
  }
  const std::string json = header_json(fields);
  put_bytes(kMagic, sizeof(kMagic));
  put_u16(1);  // version_major
  put_u16(0);  // version_minor
  put_u32(static_cast<std::uint32_t>(json.size()));
  put_bytes(json.data(), json.size());
}

SrlogWriter::~SrlogWriter() { close(); }

void SrlogWriter::close() {
  if (out_.is_open()) {
    out_.flush();
    if (!out_) {
      // Failing loudly here (not silently in the destructor path) is why
      // close() should be called explicitly on the success path.
      out_.close();
      throw std::runtime_error("SRLOG writer: flush failed on close");
    }
    out_.close();
  }
}

void SrlogWriter::write_truth(double t_s, const Eigen::Vector3d& r_m,
                              const Eigen::Vector3d& v_mps,
                              const double (&q_i2b)[4],
                              const Eigen::Vector3d& w_b_radps,
                              double mass_kg) {
  put_u16(0);  // group index: truth
  put_f64(t_s);
  for (int i = 0; i < 3; ++i) put_f64(r_m[i]);
  for (int i = 0; i < 3; ++i) put_f64(v_mps[i]);
  for (int i = 0; i < 4; ++i) put_f64(q_i2b[i]);
  for (int i = 0; i < 3; ++i) put_f64(w_b_radps[i]);
  put_f64(mass_kg);
}

void SrlogWriter::write_event(double t_s, std::uint32_t code,
                              std::string_view detail) {
  if (detail.size() > 0xFFFF) {
    throw std::invalid_argument(
        "SRLOG writer: event detail exceeds the str16 65535-byte limit");
  }
  put_u16(1);  // group index: events
  put_f64(t_s);
  put_u32(code);
  put_u16(static_cast<std::uint16_t>(detail.size()));
  put_bytes(detail.data(), detail.size());
}

// The put_* helpers memcpy through a byte buffer: the host is verified
// little-endian in the constructor, so the in-memory representation is the
// on-disk representation, and memcpy avoids strict-aliasing violations.
void SrlogWriter::put_u16(std::uint16_t v) { put_bytes(&v, sizeof v); }
void SrlogWriter::put_u32(std::uint32_t v) { put_bytes(&v, sizeof v); }
void SrlogWriter::put_f64(double v) { put_bytes(&v, sizeof v); }

void SrlogWriter::put_bytes(const void* data, std::size_t n) {
  out_.write(static_cast<const char*>(data), static_cast<std::streamsize>(n));
  if (!out_) {
    throw std::runtime_error("SRLOG writer: write failed");
  }
}

}  // namespace log
}  // namespace star
