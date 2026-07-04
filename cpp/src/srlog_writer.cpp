// SRLOG v1.1 writer implementation. The byte layout is normative in
// docs/formats/srlog_v1.md; keep the two in lockstep.
#include "star/srlog_writer.hpp"

#include <cstring>
#include <stdexcept>
#include <string>

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
void append_channel(std::string& out, std::string_view name, const char* dtype,
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

// A nonzero group rate must divide the truth rate exactly: group records are
// decimated from the truth grid, never interpolated (FR-16). Zero means the
// group is absent from the file.
void check_group_rate(const char* group, std::uint32_t rate_hz,
                      std::uint32_t truth_rate_hz) {
  if (rate_hz == 0) return;
  if (truth_rate_hz == 0 || truth_rate_hz % rate_hz != 0) {
    throw std::invalid_argument(
        std::string("SRLOG writer: ") + group + " rate " +
        std::to_string(rate_hz) + " Hz is not an integer divisor of the truth "
        "rate " + std::to_string(truth_rate_hz) + " Hz (FR-16 allows "
        "decimation only, never interpolation)");
  }
}

// The declared source subset must come from the canonical vocabulary, in
// canonical order, without duplicates, so the header bytes are a pure
// function of the enabled set. Anything else is rejected loudly rather than
// silently normalized.
void check_force_sources(const std::vector<std::string>& sources) {
  std::size_t next = 0;  // first canonical slot the next source may occupy
  for (const std::string& src : sources) {
    std::size_t pos = next;
    while (pos < kForceSourceCount && src != kForceSources[pos]) ++pos;
    if (pos < kForceSourceCount) {
      next = pos + 1;
      continue;
    }
    // Not found at or after `next`: either the name is outside the
    // vocabulary, or it appeared earlier (duplicate / out of order).
    for (std::size_t i = 0; i < kForceSourceCount; ++i) {
      if (src == kForceSources[i]) {
        throw std::invalid_argument(
            "SRLOG writer: force source '" + src + "' is duplicated or out "
            "of canonical order; declare a subset of {gravity, thirdbody, "
            "srp, drag, aero, thrust, rcs, gravgrad, wheel} in that order");
      }
    }
    throw std::invalid_argument(
        "SRLOG writer: unknown force source '" + src + "'; the canonical "
        "vocabulary is {gravity, thirdbody, srp, drag, aero, thrust, rcs, "
        "gravgrad, wheel}");
  }
}

// Enabled/disabled is decided by the rate; a half-declared forces group
// (sources without a rate, or a rate without sources) is a configuration
// error, not something to reconcile silently.
void check_group_fields(const SrlogHeaderFields& fields) {
  check_group_rate("forces", fields.forces_rate_hz, fields.truth_rate_hz);
  check_group_rate("mass", fields.mass_rate_hz, fields.truth_rate_hz);
  check_group_rate("env", fields.env_rate_hz, fields.truth_rate_hz);
  check_force_sources(fields.force_sources);
  if ((fields.forces_rate_hz != 0) != !fields.force_sources.empty()) {
    throw std::invalid_argument(
        fields.force_sources.empty()
            ? "SRLOG writer: forces_rate_hz is set but force_sources is "
              "empty; a forces group needs at least one source"
            : "SRLOG writer: force_sources is set but forces_rate_hz is 0; "
              "set the rate to enable the forces group");
  }
}

}  // namespace

std::string SrlogWriter::header_json(const SrlogHeaderFields& fields) {
  // Hand-rolled, compact, fixed key order (contract section 2). std::to_string
  // on integer types is locale-independent, so no locale can perturb the
  // bytes. No floats appear anywhere in the header by design.
  check_group_fields(fields);
  std::string j;
  j.reserve(1024);
  j += "{\"format\":{\"name\":\"SRLOG\",\"major\":1,\"minor\":1}";
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
  j += "]}";
  // v1.1 vehicle groups (format doc section 3.1), in the fixed order forces,
  // mass, env; each appears only when its rate is nonzero.
  if (fields.forces_rate_hz != 0) {
    j += ",{\"name\":\"forces\",\"rate_hz\":";
    j += std::to_string(fields.forces_rate_hz);
    j += ",\"channels\":[";
    append_channel(j, "t_s", "f64", "s", "");
    for (const std::string& src : fields.force_sources) {
      j += ',';
      append_channel(j, "f_" + src + "_b_n", "f64[3]", "N", "body");
      j += ',';
      append_channel(j, "tq_" + src + "_b_nm", "f64[3]", "N*m", "body");
    }
    j += "]}";
  }
  if (fields.mass_rate_hz != 0) {
    j += ",{\"name\":\"mass\",\"rate_hz\":";
    j += std::to_string(fields.mass_rate_hz);
    j += ",\"channels\":[";
    append_channel(j, "t_s", "f64", "s", "");
    j += ',';
    append_channel(j, "mass_kg", "f64", "kg", "");
    j += ',';
    append_channel(j, "cg_b_m", "f64[3]", "m", "body");
    j += ',';
    // Row-major upper triangle [Ixx, Ixy, Ixz, Iyy, Iyz, Izz] - the FR-26
    // packed-upper-triangle convention.
    append_channel(j, "inertia_b_kgm2", "f64[6]", "kg*m^2", "body");
    j += "]}";
  }
  if (fields.env_rate_hz != 0) {
    j += ",{\"name\":\"env\",\"rate_hz\":";
    j += std::to_string(fields.env_rate_hz);
    j += ",\"channels\":[";
    append_channel(j, "t_s", "f64", "s", "");
    j += ',';
    append_channel(j, "alt_m", "f64", "m", "");
    j += ',';
    append_channel(j, "mach", "f64", "1", "");
    j += ',';
    append_channel(j, "q_pa", "f64", "Pa", "");
    j += ',';
    append_channel(j, "rho_kgpm3", "f64", "kg/m^3", "");
    j += ',';
    append_channel(j, "fpa_rad", "f64", "rad", "");
    j += "]}";
  }
  j += "]}";
  return j;
}

SrlogWriter::SrlogWriter(const std::string& path,
                         const SrlogHeaderFields& fields) {
  if (!host_is_little_endian()) {
    throw std::logic_error(
        "SRLOG writer requires a little-endian host; big-endian is not a "
        "supported platform");
  }
  // Serializing (and thereby validating) the header before opening the file
  // means a rejected group declaration never leaves a truncated file behind.
  const std::string json = header_json(fields);
  // Enabled v1.1 groups take the indices after truth (0) and events (1), in
  // the same fixed order header_json emits them.
  int next_index = 2;
  if (fields.forces_rate_hz != 0) {
    forces_index_ = next_index++;
    force_source_count_ = fields.force_sources.size();
  }
  if (fields.mass_rate_hz != 0) mass_index_ = next_index++;
  if (fields.env_rate_hz != 0) env_index_ = next_index++;
  out_.open(path, std::ios::binary | std::ios::trunc);
  if (!out_) {
    throw std::runtime_error("SRLOG writer: cannot open output file: " + path);
  }
  put_bytes(kMagic, sizeof(kMagic));
  put_u16(1);  // version_major
  put_u16(1);  // version_minor
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

void SrlogWriter::write_forces(double t_s,
                               const std::vector<ForceSourceSample>& samples) {
  if (forces_index_ < 0) {
    throw std::logic_error(
        "SRLOG writer: forces group was not declared at header-write time");
  }
  if (samples.size() != force_source_count_) {
    throw std::invalid_argument(
        "SRLOG writer: forces record carries " +
        std::to_string(samples.size()) + " sample(s), but " +
        std::to_string(force_source_count_) +
        " source(s) were declared; supply one sample per declared source, "
        "in declaration order");
  }
  put_u16(static_cast<std::uint16_t>(forces_index_));
  put_f64(t_s);
  for (const ForceSourceSample& s : samples) {
    for (int i = 0; i < 3; ++i) put_f64(s.force_b_n[i]);
    for (int i = 0; i < 3; ++i) put_f64(s.torque_b_nm[i]);
  }
}

void SrlogWriter::write_mass(double t_s, double mass_kg,
                             const Eigen::Vector3d& cg_b_m,
                             const double (&inertia_b_kgm2)[6]) {
  if (mass_index_ < 0) {
    throw std::logic_error(
        "SRLOG writer: mass group was not declared at header-write time");
  }
  put_u16(static_cast<std::uint16_t>(mass_index_));
  put_f64(t_s);
  put_f64(mass_kg);
  for (int i = 0; i < 3; ++i) put_f64(cg_b_m[i]);
  for (int i = 0; i < 6; ++i) put_f64(inertia_b_kgm2[i]);
}

void SrlogWriter::write_env(double t_s, double alt_m, double mach, double q_pa,
                            double rho_kgpm3, double fpa_rad) {
  if (env_index_ < 0) {
    throw std::logic_error(
        "SRLOG writer: env group was not declared at header-write time");
  }
  put_u16(static_cast<std::uint16_t>(env_index_));
  put_f64(t_s);
  put_f64(alt_m);
  put_f64(mach);
  put_f64(q_pa);
  put_f64(rho_kgpm3);
  put_f64(fpa_rad);
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
