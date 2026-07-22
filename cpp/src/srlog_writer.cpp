// SRLOG v1.2 writer implementation. The byte layout is normative in
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

// A double as its IEEE-754 binary64 bit pattern, 16 lowercase hex digits,
// most significant nibble first (format doc section 3.2, "camera").
// Deliberately not a float formatter: only integer shifting and a lookup
// table run here, so the bytes cannot depend on locale, rounding mode, or
// the libc's dtoa. The rendering is endianness-free because it serializes
// the INTEGER value of the interchange encoding, not its storage order.
void append_f64_bits_hex(std::string& out, double v) {
  std::uint64_t bits = 0;
  std::memcpy(&bits, &v, sizeof(bits));
  static const char* hex = "0123456789abcdef";
  char digits[16];
  for (int i = 15; i >= 0; --i) {
    digits[i] = hex[bits & 0xFU];
    bits >>= 4;
  }
  out += '"';
  out.append(digits, 16);
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

// A nonzero v1.2 group rate must divide the control-cycle rate exactly:
// v1.2 records are emitted on the control-cycle grid (format doc
// section 3.2), never interpolated - the same decimation-only rule the
// v1.1 groups follow against the truth grid.
void check_cycle_rate(const char* group, std::uint32_t rate_hz,
                      std::uint32_t cycle_rate_hz) {
  if (rate_hz == 0) return;
  if (cycle_rate_hz == 0 || cycle_rate_hz % rate_hz != 0) {
    throw std::invalid_argument(
        std::string("SRLOG writer: ") + group + " rate " +
        std::to_string(rate_hz) + " Hz is not an integer divisor of the "
        "control-cycle rate " + std::to_string(cycle_rate_hz) +
        " Hz (v1.2 groups decimate from the control-cycle grid, never "
        "interpolate)");
  }
}

// Declared sensor kinds must come from the canonical vocabulary, in
// canonical order, without duplicates - the force_sources discipline - so
// identical configurations yield identical headers. Landmark counts are a
// camera-only parameter.
void check_sensor_decls(const std::vector<SensorGroupDecl>& sensors,
                        std::uint32_t cycle_rate_hz) {
  std::size_t next = 0;
  for (const SensorGroupDecl& s : sensors) {
    std::size_t pos = next;
    while (pos < kSensorKindCount && s.kind != kSensorKinds[pos]) ++pos;
    if (pos >= kSensorKindCount) {
      for (std::size_t i = 0; i < kSensorKindCount; ++i) {
        if (s.kind == kSensorKinds[i]) {
          throw std::invalid_argument(
              "SRLOG writer: sensor kind '" + s.kind + "' is duplicated or "
              "out of canonical order; declare a subset of {imu, "
              "startracker, sunsensor, navfix, altimeter, camera} in that "
              "order");
        }
      }
      throw std::invalid_argument(
          "SRLOG writer: unknown sensor kind '" + s.kind + "'; the "
          "canonical vocabulary is {imu, startracker, sunsensor, navfix, "
          "altimeter, camera}");
    }
    next = pos + 1;
    if (s.rate_hz == 0) {
      throw std::invalid_argument(
          "SRLOG writer: sensor group 'sensors." + s.kind + "' is declared "
          "with a zero sample rate; a declared sensor needs a rate");
    }
    check_cycle_rate(("sensors." + s.kind).c_str(), s.rate_hz,
                     cycle_rate_hz);
    if (s.landmarks != 0 && s.kind != "camera") {
      throw std::invalid_argument(
          "SRLOG writer: sensor kind '" + s.kind + "' declares " +
          std::to_string(s.landmarks) + " landmark(s); landmark "
          "projections are a camera-only parameter");
    }
  }
}

// v1.2 declarations hang together: a cycle rate of zero forbids every GNC
// group, dimension and rate fields are declared jointly, and nav.innov
// needs the sensor-identity table its records index.
void check_gnc_fields(const SrlogHeaderFields& fields) {
  if (fields.cycle_rate_hz == 0) {
    if (!fields.sensors.empty() || fields.nav_est_rate_hz != 0 ||
        fields.nav_state_dim != 0 || fields.nav_cov_dim != 0 ||
        fields.nav_err_enabled || fields.nav_innov_enabled ||
        fields.nav_innov_max_dim != 0 || fields.gnc_cmd_rate_hz != 0 ||
        fields.latency_cycles != 0 || fields.camera_echo_present) {
      throw std::invalid_argument(
          "SRLOG writer: a v1.2 GNC group (or latency_cycles) is declared "
          "but cycle_rate_hz is 0; the control-cycle rate anchors every "
          "GNC group declaration");
    }
    return;
  }
  check_sensor_decls(fields.sensors, fields.cycle_rate_hz);
  // The camera echo and the camera group stand or fall together. A camera
  // log missing the echo would silently reopen the gap exit criterion 7's
  // intrinsics clause exists to close, and an echo without a camera group
  // would describe a sensor the file does not contain.
  bool camera_declared = false;
  for (const SensorGroupDecl& s : fields.sensors) {
    if (s.kind == "camera") camera_declared = true;
  }
  if (camera_declared != fields.camera_echo_present) {
    throw std::invalid_argument(
        camera_declared
            ? "SRLOG writer: a sensors.camera group is declared without its "
              "header intrinsics echo (camera_echo_present is false); the "
              "echo is what makes a camera log self-contained"
            : "SRLOG writer: a camera intrinsics echo is declared without a "
              "sensors.camera group; the echo describes a camera the file "
              "does not carry");
  }
  if (fields.camera_echo_present) {
    // Mirrors the FR-15 validator's camera constraints at the writer
    // boundary: a zero focal length or a zero image dimension makes
    // eq:camera:K singular, so a consumer composing it from the echo would
    // divide by zero rather than see a malformed configuration.
    if (!(fields.camera.fx_px > 0.0) || !(fields.camera.fy_px > 0.0)) {
      throw std::invalid_argument(
          "SRLOG writer: camera echo needs strictly positive fx_px and "
          "fy_px; eq:camera:K is singular otherwise");
    }
    if (fields.camera.width_px == 0 || fields.camera.height_px == 0) {
      throw std::invalid_argument(
          "SRLOG writer: camera echo needs nonzero width_px and height_px");
    }
  }
  if ((fields.nav_est_rate_hz != 0) != (fields.nav_state_dim != 0)) {
    throw std::invalid_argument(
        "SRLOG writer: nav.est needs both a rate and a state dimension "
        "(nav_est_rate_hz and nav_state_dim are declared jointly)");
  }
  if (fields.nav_cov_dim != 0 && fields.nav_est_rate_hz == 0) {
    throw std::invalid_argument(
        "SRLOG writer: nav_cov_dim is declared without nav.est; the "
        "covariance dimension qualifies the nav.est declaration");
  }
  check_cycle_rate("nav.est", fields.nav_est_rate_hz, fields.cycle_rate_hz);
  if (fields.nav_err_enabled && fields.nav_est_rate_hz == 0) {
    // The consistency tooling computes NEES from nav.err.e and nav.est.P
    // together and requires matching record counts; nav.err therefore
    // exists only alongside nav.est, at nav.est's rate and dimension.
    throw std::invalid_argument(
        "SRLOG writer: nav.err is declared without nav.est; the error "
        "state shares nav.est's rate and dimension and cannot exist alone");
  }
  if (fields.nav_innov_enabled != (fields.nav_innov_max_dim != 0)) {
    throw std::invalid_argument(
        "SRLOG writer: nav.innov needs both the enable flag and a maximum "
        "innovation dimension (nav_innov_enabled and nav_innov_max_dim are "
        "declared jointly)");
  }
  if (fields.nav_innov_enabled && fields.sensors.empty()) {
    throw std::invalid_argument(
        "SRLOG writer: nav.innov is declared but no sensor group is; "
        "sensor_id indexes the header's declared sensor list, which must "
        "not be empty");
  }
  check_cycle_rate("gnc.cmd", fields.gnc_cmd_rate_hz, fields.cycle_rate_hz);
}

// f64[N] dtype string for a dimension chosen at header-write time (the
// nav.est/nav.err/nav.innov and camera-landmark channels).
std::string f64_array_dtype(std::size_t n) {
  return "f64[" + std::to_string(n) + "]";
}

}  // namespace

std::string SrlogWriter::header_json(const SrlogHeaderFields& fields) {
  // Hand-rolled, compact, fixed key order (contract section 2). std::to_string
  // on integer types is locale-independent, so no locale can perturb the
  // bytes. No floats appear anywhere in the header by design.
  check_group_fields(fields);
  check_gnc_fields(fields);
  std::string j;
  j.reserve(1024);
  j += "{\"format\":{\"name\":\"SRLOG\",\"major\":1,\"minor\":3}";
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
  // v1.2: the "gnc" object (format doc section 3.2) rides only in GNC-enabled
  // files, so pre-Phase-6 configurations resolve to headers that differ from
  // v1.1 output in the version words alone. The sensors array doubles as the
  // nav.innov sensor-identity table.
  if (fields.cycle_rate_hz != 0) {
    j += ",\"gnc\":{\"cycle_rate_hz\":";
    j += std::to_string(fields.cycle_rate_hz);
    j += ",\"latency_cycles\":";
    j += std::to_string(fields.latency_cycles);
    j += ",\"sensors\":[";
    bool first = true;
    for (const SensorGroupDecl& s : fields.sensors) {
      if (!first) j += ',';
      first = false;
      append_json_string(j, s.kind);
    }
    j += ']';
    // v1.3: the camera constants, present exactly when a camera group is.
    // float_encoding is emitted first and self-describes the hex bit
    // patterns, so a consumer never has to infer the encoding from the
    // format version it may not know.
    if (fields.camera_echo_present) {
      const CameraEchoDecl& c = fields.camera;
      j += ",\"camera\":{\"float_encoding\":\"ieee754-binary64-hex\"";
      j += ",\"width_px\":";
      j += std::to_string(c.width_px);
      j += ",\"height_px\":";
      j += std::to_string(c.height_px);
      j += ",\"fx_px\":";
      append_f64_bits_hex(j, c.fx_px);
      j += ",\"fy_px\":";
      append_f64_bits_hex(j, c.fy_px);
      j += ",\"cx_px\":";
      append_f64_bits_hex(j, c.cx_px);
      j += ",\"cy_px\":";
      append_f64_bits_hex(j, c.cy_px);
      j += ",\"q_b2c\":[";
      for (int i = 0; i < 4; ++i) {
        if (i != 0) j += ',';
        append_f64_bits_hex(j, c.q_b2c[i]);
      }
      j += "],\"r_cam_b_m\":[";
      for (int i = 0; i < 3; ++i) {
        if (i != 0) j += ',';
        append_f64_bits_hex(j, c.r_cam_b_m[i]);
      }
      j += "]}";
    }
    j += '}';
  }
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
  // v1.2 GNC groups (format doc section 3.2), in the fixed order: the
  // declared sensors.<kind> groups (canonical kind order), then nav.est,
  // nav.err, nav.innov, gnc.cmd; each appears only when declared.
  for (const SensorGroupDecl& s : fields.sensors) {
    j += ",{\"name\":\"sensors.";
    j += s.kind;
    j += "\",\"rate_hz\":";
    j += std::to_string(s.rate_hz);
    j += ",\"channels\":[";
    append_channel(j, "t_s", "f64", "s", "");
    j += ',';
    if (s.kind == "imu") {
      append_channel(j, "dtheta_b_rad", "f64[3]", "rad", "body");
      j += ',';
      append_channel(j, "dv_b_mps", "f64[3]", "m/s", "body");
    } else if (s.kind == "startracker") {
      append_channel(j, "q_meas_i2b", "f64[4]", "1",
                     "GCRF->body Hamilton scalar-first");
      j += ',';
      append_channel(j, "valid", "u32", "1", "");
    } else if (s.kind == "sunsensor") {
      append_channel(j, "sun_b", "f64[3]", "1", "body");
      j += ',';
      append_channel(j, "valid", "u32", "1", "");
    } else if (s.kind == "navfix") {
      append_channel(j, "r_meas_m", "f64[3]", "m", "GCRF");
      j += ',';
      append_channel(j, "v_meas_mps", "f64[3]", "m/s", "GCRF");
    } else if (s.kind == "altimeter") {
      append_channel(j, "alt_meas_m", "f64", "m", "");
    } else {  // camera (the vocabulary is closed by check_sensor_decls)
      append_channel(j, "r_m", "f64[3]", "m", "GCRF");
      j += ',';
      append_channel(j, "q_i2b", "f64[4]", "1",
                     "GCRF->body Hamilton scalar-first");
      if (s.landmarks != 0) {
        j += ',';
        append_channel(j, "px_uv",
                       f64_array_dtype(2 * static_cast<std::size_t>(
                                               s.landmarks)).c_str(),
                       "px", "image");
      }
    }
    j += "]}";
  }
  if (fields.nav_est_rate_hz != 0) {
    const std::size_t n = fields.nav_state_dim;
    // The covariance may live in a different parameterization than the
    // state (error-state estimators): m defaults to n and is declared
    // independently otherwise (header contract in srlog_writer.hpp).
    const std::size_t m =
        fields.nav_cov_dim != 0 ? fields.nav_cov_dim : fields.nav_state_dim;
    j += ",{\"name\":\"nav.est\",\"rate_hz\":";
    j += std::to_string(fields.nav_est_rate_hz);
    j += ",\"channels\":[";
    append_channel(j, "t_s", "f64", "s", "");
    j += ',';
    // x_hat units and component meaning are estimator-defined (the
    // estimator's chapter is normative for them); the layout alone is
    // fixed here.
    append_channel(j, "x_hat", f64_array_dtype(n).c_str(), "", "");
    j += ',';
    append_channel(j, "P", f64_array_dtype(m * (m + 1) / 2).c_str(), "", "");
    j += "]}";
  }
  if (fields.nav_err_enabled) {
    // nav.err rides at nav.est's rate with the same dimension n (contract
    // with the consistency tooling: matching record counts, NEES from
    // nav.err.e + nav.est.P directly).
    j += ",{\"name\":\"nav.err\",\"rate_hz\":";
    j += std::to_string(fields.nav_est_rate_hz);
    j += ",\"channels\":[";
    append_channel(j, "t_s", "f64", "s", "");
    j += ',';
    append_channel(j, "e", f64_array_dtype(fields.nav_state_dim).c_str(),
                   "", "");
    j += "]}";
  }
  if (fields.nav_innov_enabled) {
    const std::size_t mm = fields.nav_innov_max_dim;
    j += ",{\"name\":\"nav.innov\",\"rate_hz\":0,\"channels\":[";
    append_channel(j, "t_s", "f64", "s", "");
    j += ',';
    append_channel(j, "sensor_id", "u32", "1", "");
    j += ',';
    append_channel(j, "m", "u32", "1", "");
    j += ',';
    append_channel(j, "y", f64_array_dtype(mm).c_str(), "", "");
    j += ',';
    append_channel(j, "S", f64_array_dtype(mm * (mm + 1) / 2).c_str(), "",
                   "");
    j += "]}";
  }
  if (fields.gnc_cmd_rate_hz != 0) {
    j += ",{\"name\":\"gnc.cmd\",\"rate_hz\":";
    j += std::to_string(fields.gnc_cmd_rate_hz);
    j += ",\"channels\":[";
    append_channel(j, "t_s", "f64", "s", "");
    j += ',';
    append_channel(j, "tau_b_nm", "f64[3]", "N*m", "body");
    j += ',';
    append_channel(j, "q_cmd_i2b", "f64[4]", "1",
                   "GCRF->body Hamilton scalar-first");
    j += ',';
    append_channel(j, "w_cmd_b_radps", "f64[3]", "rad/s", "body");
    j += ',';
    append_channel(j, "valid", "u32", "1", "");
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
  // Enabled v1.1/v1.2 groups take the indices after truth (0) and events
  // (1), in the same fixed order header_json emits them.
  int next_index = 2;
  if (fields.forces_rate_hz != 0) {
    forces_index_ = next_index++;
    force_source_count_ = fields.force_sources.size();
  }
  if (fields.mass_rate_hz != 0) mass_index_ = next_index++;
  if (fields.env_rate_hz != 0) env_index_ = next_index++;
  // -1 is "this kind is absent from this file"; the default-constructed
  // array holds 0, which is truth's group index.
  sensor_index_.fill(-1);
  sensor_count_ = fields.sensors.size();
  for (const SensorGroupDecl& s : fields.sensors) {
    for (std::size_t k = 0; k < kSensorKindCount; ++k) {
      if (s.kind == kSensorKinds[k]) {
        sensor_index_[k] = next_index;
        if (s.kind == "camera") {
          camera_px_count_ = 2 * static_cast<std::size_t>(s.landmarks);
        }
      }
    }
    ++next_index;
  }
  if (fields.nav_est_rate_hz != 0) {
    nav_est_index_ = next_index++;
    nav_state_dim_ = fields.nav_state_dim;
    nav_cov_dim_ =
        fields.nav_cov_dim != 0 ? fields.nav_cov_dim : fields.nav_state_dim;
  }
  if (fields.nav_err_enabled) nav_err_index_ = next_index++;
  if (fields.nav_innov_enabled) {
    nav_innov_index_ = next_index++;
    nav_innov_max_dim_ = fields.nav_innov_max_dim;
  }
  if (fields.gnc_cmd_rate_hz != 0) gnc_cmd_index_ = next_index++;
  out_.open(path, std::ios::binary | std::ios::trunc);
  if (!out_) {
    throw std::runtime_error("SRLOG writer: cannot open output file: " + path);
  }
  put_bytes(kMagic, sizeof(kMagic));
  put_u16(1);  // version_major
  put_u16(3);  // version_minor
  put_u32(static_cast<std::uint32_t>(json.size()));
  put_bytes(json.data(), json.size());
}

SrlogWriter::~SrlogWriter() {
  // A destructor is implicitly noexcept, so a throw escaping here would call
  // std::terminate: an immediate process abort with no unwinding and no
  // Python traceback, on exactly the I/O failure - a full disk, a dropped
  // share - where a diagnostic matters most. The release of the file handle
  // is what the destructor is responsible for, and close() below performs it
  // before it throws; the failure itself is reported from the explicit
  // close() on the normal path, which VehicleCycle::finish() and
  // VehicleCycle::close() both call.
  try {
    close();
  } catch (...) {  // NOLINT(bugprone-empty-catch) - see above
  }
}

void SrlogWriter::close() {
  if (out_.is_open()) {
    out_.flush();
    if (!out_) {
      // The handle is released before throwing, so a caller that reports
      // this error still gets the file closed - the destructor swallows the
      // throw but must never be the thing that leaks the handle.
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

namespace {

// Canonical-slot lookup for the sensor write methods; the vocabulary is
// closed at declaration time so a miss here is unreachable in practice.
std::size_t sensor_slot(const char* kind) {
  for (std::size_t k = 0; k < kSensorKindCount; ++k) {
    if (std::strcmp(kind, kSensorKinds[k]) == 0) return k;
  }
  throw std::logic_error(std::string("SRLOG writer: unknown sensor kind: ") +
                         kind);
}

}  // namespace

void SrlogWriter::write_sensor_imu(double t_s,
                                   const Eigen::Vector3d& dtheta_b_rad,
                                   const Eigen::Vector3d& dv_b_mps) {
  const int idx = sensor_index_[sensor_slot("imu")];
  if (idx < 0) {
    throw std::logic_error(
        "SRLOG writer: sensors.imu group was not declared at header-write "
        "time");
  }
  put_u16(static_cast<std::uint16_t>(idx));
  put_f64(t_s);
  for (int i = 0; i < 3; ++i) put_f64(dtheta_b_rad[i]);
  for (int i = 0; i < 3; ++i) put_f64(dv_b_mps[i]);
}

void SrlogWriter::write_sensor_startracker(double t_s,
                                           const double (&q_meas_i2b)[4],
                                           std::uint32_t valid) {
  const int idx = sensor_index_[sensor_slot("startracker")];
  if (idx < 0) {
    throw std::logic_error(
        "SRLOG writer: sensors.startracker group was not declared at "
        "header-write time");
  }
  put_u16(static_cast<std::uint16_t>(idx));
  put_f64(t_s);
  for (int i = 0; i < 4; ++i) put_f64(q_meas_i2b[i]);
  put_u32(valid);
}

void SrlogWriter::write_sensor_sunsensor(double t_s,
                                         const Eigen::Vector3d& sun_b,
                                         std::uint32_t valid) {
  const int idx = sensor_index_[sensor_slot("sunsensor")];
  if (idx < 0) {
    throw std::logic_error(
        "SRLOG writer: sensors.sunsensor group was not declared at "
        "header-write time");
  }
  put_u16(static_cast<std::uint16_t>(idx));
  put_f64(t_s);
  for (int i = 0; i < 3; ++i) put_f64(sun_b[i]);
  put_u32(valid);
}

void SrlogWriter::write_sensor_navfix(double t_s,
                                      const Eigen::Vector3d& r_meas_m,
                                      const Eigen::Vector3d& v_meas_mps) {
  const int idx = sensor_index_[sensor_slot("navfix")];
  if (idx < 0) {
    throw std::logic_error(
        "SRLOG writer: sensors.navfix group was not declared at "
        "header-write time");
  }
  put_u16(static_cast<std::uint16_t>(idx));
  put_f64(t_s);
  for (int i = 0; i < 3; ++i) put_f64(r_meas_m[i]);
  for (int i = 0; i < 3; ++i) put_f64(v_meas_mps[i]);
}

void SrlogWriter::write_sensor_altimeter(double t_s, double alt_meas_m) {
  const int idx = sensor_index_[sensor_slot("altimeter")];
  if (idx < 0) {
    throw std::logic_error(
        "SRLOG writer: sensors.altimeter group was not declared at "
        "header-write time");
  }
  put_u16(static_cast<std::uint16_t>(idx));
  put_f64(t_s);
  put_f64(alt_meas_m);
}

void SrlogWriter::write_sensor_camera(double t_s, const Eigen::Vector3d& r_m,
                                      const double (&q_i2b)[4],
                                      const double* px_uv,
                                      std::size_t px_count) {
  const int idx = sensor_index_[sensor_slot("camera")];
  if (idx < 0) {
    throw std::logic_error(
        "SRLOG writer: sensors.camera group was not declared at "
        "header-write time");
  }
  if (px_count != camera_px_count_) {
    throw std::invalid_argument(
        "SRLOG writer: sensors.camera record carries " +
        std::to_string(px_count) + " pixel value(s), but the declaration "
        "fixes " + std::to_string(camera_px_count_) +
        " (2 per declared landmark)");
  }
  put_u16(static_cast<std::uint16_t>(idx));
  put_f64(t_s);
  for (int i = 0; i < 3; ++i) put_f64(r_m[i]);
  for (int i = 0; i < 4; ++i) put_f64(q_i2b[i]);
  for (std::size_t i = 0; i < px_count; ++i) put_f64(px_uv[i]);
}

void SrlogWriter::write_nav_est(double t_s, const double* x_hat,
                                std::size_t n, const double* p_upper,
                                std::size_t p_len) {
  if (nav_est_index_ < 0) {
    throw std::logic_error(
        "SRLOG writer: nav.est group was not declared at header-write time");
  }
  if (n != nav_state_dim_ || p_len != nav_cov_dim_ * (nav_cov_dim_ + 1) / 2) {
    throw std::invalid_argument(
        "SRLOG writer: nav.est record carries x_hat[" + std::to_string(n) +
        "], P[" + std::to_string(p_len) + "], but the declaration fixes n=" +
        std::to_string(nav_state_dim_) + ", m=" +
        std::to_string(nav_cov_dim_) + " (P is m(m+1)/2 packed row-major "
        "upper triangle)");
  }
  put_u16(static_cast<std::uint16_t>(nav_est_index_));
  put_f64(t_s);
  for (std::size_t i = 0; i < n; ++i) put_f64(x_hat[i]);
  for (std::size_t i = 0; i < p_len; ++i) put_f64(p_upper[i]);
}

void SrlogWriter::write_nav_err(double t_s, const double* e, std::size_t n) {
  if (nav_err_index_ < 0) {
    throw std::logic_error(
        "SRLOG writer: nav.err group was not declared at header-write time");
  }
  if (n != nav_state_dim_) {
    throw std::invalid_argument(
        "SRLOG writer: nav.err record carries e[" + std::to_string(n) +
        "], but the error state shares nav.est's dimension n=" +
        std::to_string(nav_state_dim_));
  }
  put_u16(static_cast<std::uint16_t>(nav_err_index_));
  put_f64(t_s);
  for (std::size_t i = 0; i < n; ++i) put_f64(e[i]);
}

void SrlogWriter::write_nav_innov(double t_s, std::uint32_t sensor_id,
                                  std::uint32_t m, const double* y,
                                  std::size_t y_len, const double* s_upper,
                                  std::size_t s_len) {
  if (nav_innov_index_ < 0) {
    throw std::logic_error(
        "SRLOG writer: nav.innov group was not declared at header-write "
        "time");
  }
  // sensor_id indexes the header's gnc.sensors array (format doc section
  // 3.2), and the writer has known that array's size since construction.
  // Every other dimension of a v1.2 record is checked here; without this one
  // a component could write an id no reader can resolve, and a downstream
  // tool would either raise at analysis time or - worse, if it clamps -
  // silently attribute the innovation to the wrong instrument. That
  // attribution is what the whole NEES/NIS result rests on.
  if (sensor_id >= sensor_count_) {
    throw std::invalid_argument(
        "SRLOG writer: nav.innov record names sensor_id " +
        std::to_string(sensor_id) + ", but this file declares " +
        std::to_string(sensor_count_) +
        " sensor(s); sensor_id indexes the header's gnc.sensors array");
  }
  const std::size_t mm = nav_innov_max_dim_;
  if (y_len != mm || s_len != mm * (mm + 1) / 2 || m == 0 || m > mm) {
    throw std::invalid_argument(
        "SRLOG writer: nav.innov record carries y[" + std::to_string(y_len) +
        "], S[" + std::to_string(s_len) + "], m=" + std::to_string(m) +
        ", but the declaration fixes m_max=" + std::to_string(mm) +
        " (y is f64[m_max], S is f64[m_max(m_max+1)/2], 1 <= m <= m_max)");
  }
  put_u16(static_cast<std::uint16_t>(nav_innov_index_));
  put_f64(t_s);
  put_u32(sensor_id);
  put_u32(m);
  for (std::size_t i = 0; i < y_len; ++i) put_f64(y[i]);
  for (std::size_t i = 0; i < s_len; ++i) put_f64(s_upper[i]);
}

void SrlogWriter::write_gnc_cmd(double t_s, const Eigen::Vector3d& tau_b_nm,
                                const double (&q_cmd_i2b)[4],
                                const Eigen::Vector3d& w_cmd_b_radps,
                                std::uint32_t valid) {
  if (gnc_cmd_index_ < 0) {
    throw std::logic_error(
        "SRLOG writer: gnc.cmd group was not declared at header-write time");
  }
  put_u16(static_cast<std::uint16_t>(gnc_cmd_index_));
  put_f64(t_s);
  for (int i = 0; i < 3; ++i) put_f64(tau_b_nm[i]);
  for (int i = 0; i < 4; ++i) put_f64(q_cmd_i2b[i]);
  for (int i = 0; i < 3; ++i) put_f64(w_cmd_b_radps[i]);
  put_u32(valid);
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
