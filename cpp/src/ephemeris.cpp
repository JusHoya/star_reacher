// SREPH v1 loader and Chebyshev evaluator (FR-4, D-8); format specification
// in docs/formats/sreph_v1.md, derivation in the ephemeris chapter of the
// math library. The evaluation path mirrors the Python reference evaluator
// in python/star_reacher/data_fetch.py statement for statement: with FMA
// contraction disabled (D-10 build flags) both produce bit-identical IEEE-754
// results, which is the property the committed bit-level goldens gate.
#include "star/ephemeris.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace star {

namespace {

constexpr unsigned char kMagic[8] = {'S', 'R', 'E', 'P', 'H', 0x00, 0x0D, 0x0A};
constexpr std::uint16_t kVersionMajor = 1;
constexpr std::size_t kHeaderSize = 96;
constexpr std::size_t kDirEntrySize = 64;
// DE440 segments carry at most 14 coefficients per component (Mercury);
// capping the count lets evaluation use fixed stack buffers and rejects
// nonsense counts from a corrupt directory before they can size anything.
constexpr std::uint32_t kMaxCoeffs = 32;

// Little-endian field readers. Every supported platform is little-endian
// (same rule as SRLOG), so memcpy from the file image is the whole decode;
// memcpy rather than reinterpret_cast keeps the reads alignment-safe.
std::uint16_t read_u16(const unsigned char* p) {
  std::uint16_t v;
  std::memcpy(&v, p, sizeof v);
  return v;
}
std::uint32_t read_u32(const unsigned char* p) {
  std::uint32_t v;
  std::memcpy(&v, p, sizeof v);
  return v;
}
std::uint64_t read_u64(const unsigned char* p) {
  std::uint64_t v;
  std::memcpy(&v, p, sizeof v);
  return v;
}
double read_f64(const unsigned char* p) {
  double v;
  std::memcpy(&v, p, sizeof v);
  return v;
}

[[noreturn]] void fail(const std::string& path, const std::string& what) {
  throw std::runtime_error("ephemeris: " + path + ": " + what);
}

}  // namespace

Ephemeris Ephemeris::load_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    fail(path, "cannot open file");
  }
  std::vector<unsigned char> buf((std::istreambuf_iterator<char>(in)),
                                 std::istreambuf_iterator<char>());
  if (buf.size() < kHeaderSize) {
    fail(path, "truncated header (file smaller than the 96-byte SREPH header)");
  }
  if (std::memcmp(buf.data(), kMagic, sizeof kMagic) != 0) {
    fail(path, "bad magic (not an SREPH file, or mangled by text-mode transfer)");
  }
  const std::uint16_t major = read_u16(&buf[8]);
  if (major != kVersionMajor) {
    std::ostringstream msg;
    msg << "SREPH major version " << major << " is not supported (this reader implements "
        << kVersionMajor << "); refusing to guess the layout";
    fail(path, msg.str());
  }
  // Minor version (bytes 10-11) is additive-only by the format contract and
  // nothing this reader consumes changes with it.
  const std::uint32_t segment_count = read_u32(&buf[12]);

  Ephemeris eph;
  eph.span_start_tdb_s_ = read_f64(&buf[16]);
  eph.span_end_tdb_s_ = read_f64(&buf[24]);
  // Bytes 32-95: SHA-256 digests of the source kernels. Provenance rides in
  // the file for auditability but the evaluator has no use for it here.

  const std::size_t dir_end = kHeaderSize + kDirEntrySize * static_cast<std::size_t>(segment_count);
  if (buf.size() < dir_end) {
    fail(path, "truncated segment directory");
  }
  eph.segments_.reserve(segment_count);
  for (std::uint32_t i = 0; i < segment_count; ++i) {
    const unsigned char* e = &buf[kHeaderSize + kDirEntrySize * static_cast<std::size_t>(i)];
    Segment seg;
    const char* name_begin = reinterpret_cast<const char*>(e);
    std::size_t name_len = 0;
    while (name_len < 16 && name_begin[name_len] != '\0') {
      ++name_len;
    }
    seg.name.assign(name_begin, name_len);
    seg.target = read_u32(e + 16);
    seg.center = read_u32(e + 20);
    seg.kind = read_u32(e + 24);
    seg.n_coeffs = read_u32(e + 28);
    seg.init_tdb_s = read_f64(e + 32);
    seg.intlen_s = read_f64(e + 40);
    seg.n_records = read_u32(e + 48);
    const std::uint64_t offset = read_u64(e + 56);
    if (seg.n_coeffs == 0 || seg.n_coeffs > kMaxCoeffs || seg.n_records == 0 ||
        !(seg.intlen_s > 0.0)) {
      std::ostringstream msg;
      msg << "segment " << i << " ('" << seg.name << "') has an invalid directory entry";
      fail(path, msg.str());
    }
    const std::uint64_t n_doubles =
        static_cast<std::uint64_t>(seg.n_records) * 3u * seg.n_coeffs;
    if (offset + n_doubles * 8u > buf.size()) {
      std::ostringstream msg;
      msg << "segment " << i << " ('" << seg.name << "') coefficient block runs past end of file";
      fail(path, msg.str());
    }
    seg.coeffs.resize(static_cast<std::size_t>(n_doubles));
    std::memcpy(seg.coeffs.data(), &buf[static_cast<std::size_t>(offset)],
                static_cast<std::size_t>(n_doubles) * 8u);
    eph.segments_.push_back(std::move(seg));
  }
  if (eph.segments_.empty()) {
    fail(path, "file contains no segments");
  }
  return eph;
}

const Ephemeris::Segment& Ephemeris::find_segment(const std::string& body, double tdb_s,
                                                  std::uint32_t expected_kind) const {
  bool name_seen = false;
  double lo = 0.0;
  double hi = 0.0;
  for (const Segment& seg : segments_) {
    if (seg.name != body || seg.kind != expected_kind) {
      continue;
    }
    if (!name_seen) {
      lo = seg.init_tdb_s;
      hi = seg.end_tdb_s();
      name_seen = true;
    } else {
      lo = std::min(lo, seg.init_tdb_s);
      hi = std::max(hi, seg.end_tdb_s());
    }
    if (tdb_s >= seg.init_tdb_s && tdb_s <= seg.end_tdb_s()) {
      return seg;
    }
  }
  if (!name_seen) {
    std::ostringstream msg;
    msg << "ephemeris: unknown body '" << body << "'; available:";
    for (const std::string& n : bodies()) {
      msg << " " << n;
    }
    throw std::invalid_argument(msg.str());
  }
  std::ostringstream msg;
  msg.precision(17);
  msg << "ephemeris: epoch " << tdb_s << " s TDB is outside the stored records of '" << body
      << "' (coverage [" << lo << ", " << hi << "] s TDB); refusing to extrapolate";
  throw std::out_of_range(msg.str());
}

void Ephemeris::evaluate(const Segment& seg, double tdb_s, double value[3], double rate[3]) {
  // Record selection: the boundary epoch between records k-1 and k belongs
  // to record k; the segment's final epoch (which would select record
  // n_records) clamps to the last record and evaluates at x = +1 exactly.
  // The caller has already established containment, so k lands in range.
  const std::int64_t n = static_cast<std::int64_t>(seg.n_records);
  std::int64_t k = static_cast<std::int64_t>(std::floor((tdb_s - seg.init_tdb_s) / seg.intlen_s));
  if (k >= n) {
    k = n - 1;
  }
  if (k < 0) {
    k = 0;
  }
  // eq:ephemeris:timescale
  const double t0 = seg.init_tdb_s + static_cast<double>(k) * seg.intlen_s;
  const double x = 2.0 * (tdb_s - t0) / seg.intlen_s - 1.0;

  const std::uint32_t ncoef = seg.n_coeffs;
  double T[kMaxCoeffs];
  double dT[kMaxCoeffs];
  T[0] = 1.0;
  dT[0] = 0.0;
  if (ncoef > 1) {
    T[1] = x;
    dT[1] = 1.0;
  }
  // eq:ephemeris:recurrence
  for (std::uint32_t j = 2; j < ncoef; ++j) {
    T[j] = 2.0 * x * T[j - 1] - T[j - 2];
    dT[j] = 2.0 * T[j - 1] + 2.0 * x * dT[j - 1] - dT[j - 2];
  }
  // eq:ephemeris:series - fixed ascending accumulation order (D-10)
  for (int comp = 0; comp < 3; ++comp) {
    const double* c =
        seg.coeffs.data() + (static_cast<std::size_t>(k) * 3u + static_cast<std::size_t>(comp)) *
                                ncoef;
    double acc = 0.0;
    double acc_d = 0.0;
    for (std::uint32_t j = 0; j < ncoef; ++j) {
      acc = acc + c[j] * T[j];
      acc_d = acc_d + c[j] * dT[j];
    }
    value[comp] = acc;
    rate[comp] = acc_d * (2.0 / seg.intlen_s);
  }
}

EphemerisState Ephemeris::state(const std::string& body, double tdb_s) const {
  const Segment& seg = find_segment(body, tdb_s, 0u);
  double v[3];
  double r[3];
  evaluate(seg, tdb_s, v, r);
  EphemerisState out;
  // Source coefficients are km / km-per-second exactly as DE440 stores them
  // (verbatim, never refit); the SI conversion is one exact-by-construction
  // multiply per component at output.
  for (int i = 0; i < 3; ++i) {
    out.r_m[i] = v[i] * 1000.0;
    out.v_mps[i] = r[i] * 1000.0;
  }
  return out;
}

EphemerisState Ephemeris::moon_geocentric(double tdb_s) const {
  const EphemerisState moon = state("moon", tdb_s);
  const EphemerisState earth = state("earth", tdb_s);
  EphemerisState out;
  out.r_m = moon.r_m - earth.r_m;
  out.v_mps = moon.v_mps - earth.v_mps;
  return out;
}

LibrationAngles Ephemeris::lunar_librations(double tdb_s) const {
  const Segment& seg = find_segment("moon_librations", tdb_s, 1u);
  double a[3];
  double da[3];
  evaluate(seg, tdb_s, a, da);
  LibrationAngles out;
  for (int i = 0; i < 3; ++i) {
    out.angles_rad[i] = a[i];
    out.rates_radps[i] = da[i];
  }
  return out;
}

std::vector<std::string> Ephemeris::bodies() const {
  std::vector<std::string> names;
  for (const Segment& seg : segments_) {
    bool present = false;
    for (const std::string& n : names) {
      if (n == seg.name) {
        present = true;
        break;
      }
    }
    if (!present) {
      names.push_back(seg.name);
    }
  }
  std::sort(names.begin(), names.end());
  return names;
}

}  // namespace star
