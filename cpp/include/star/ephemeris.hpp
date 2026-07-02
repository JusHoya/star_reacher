// Trimmed-DE440 Chebyshev ephemeris evaluator (FR-4, D-8).
//
// Loads the SREPH v1 binary container written by `star data fetch de440s`
// (format: docs/formats/sreph_v1.md) and evaluates body positions,
// velocities, and lunar libration angles by summing the DE440 Chebyshev
// coefficients copied verbatim into that file. Binary file I/O only: the
// core never parses text, and it never touches the network (D-2) - the
// Python pipeline produced and checksummed the file before t0.
//
// Math-library traceability (FR-29): the derivation lives in the ephemeris
// chapter of docs/mathlib; the implementation echoes its equation labels
// `eq:ephemeris:timescale`, `eq:ephemeris:recurrence`, and
// `eq:ephemeris:series` at the corresponding code. The Python reference
// evaluator in python/star_reacher/data_fetch.py mirrors the same operation
// sequence statement for statement, which is what makes the committed
// bit-level golden vectors meaningful.
#ifndef STAR_EPHEMERIS_HPP
#define STAR_EPHEMERIS_HPP

#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Dense>

namespace star {

// Position and velocity of a body relative to its stored center, resolved in
// the ICRF-oriented frame of the source ephemeris (DE440 uses the ICRF).
struct EphemerisState {
  Eigen::Vector3d r_m;    // position [m]
  Eigen::Vector3d v_mps;  // velocity [m/s]
};

// Lunar libration angles: the 3-1-3 Euler angles phi, theta, psi of the Moon
// principal-axis frame relative to the ICRF equator, as integrated by DE440,
// plus their time derivatives. psi accumulates (it is not wrapped), exactly
// as the source kernel stores it.
struct LibrationAngles {
  Eigen::Vector3d angles_rad;    // [phi, theta, psi] [rad]
  Eigen::Vector3d rates_radps;   // d/dt of the above [rad/s]
};

class Ephemeris {
 public:
  // Load an SREPH v1 file. Throws std::runtime_error naming the specific
  // defect (bad magic, unsupported major version, truncated directory or
  // coefficient block) on a malformed file.
  static Ephemeris load_file(const std::string& path);

  // State of `body` relative to its stored center at `tdb_s` seconds since
  // J2000 TDB. Bodies in the standard repack: "sun", "emb", "venus_bary",
  // "mars_bary", "jupiter_bary" (centered on the solar system barycenter)
  // and "earth", "moon" (centered on the Earth-Moon barycenter, exactly as
  // DE440 stores them). Throws std::invalid_argument for an unknown body
  // and std::out_of_range for an epoch outside the stored records - the
  // evaluator never extrapolates.
  EphemerisState state(const std::string& body, double tdb_s) const;

  // Geocentric Moon state: the difference of the verbatim EMB-relative
  // "moon" and "earth" segments. Composition happens at evaluation time so
  // no coefficient is ever refitted.
  EphemerisState moon_geocentric(double tdb_s) const;

  // DE440 lunar libration angles and rates (segment "moon_librations").
  LibrationAngles lunar_librations(double tdb_s) const;

  // Distinct body names stored in the file, sorted, for error messages and
  // introspection from the bindings.
  std::vector<std::string> bodies() const;

  // Intersection of all stored segments' coverage: every body is evaluable
  // on [span_start, span_end]. Individual segments may extend further.
  double span_start_tdb_s() const { return span_start_tdb_s_; }
  double span_end_tdb_s() const { return span_end_tdb_s_; }

 private:
  // One Chebyshev segment: n_records contiguous intervals of intlen_s
  // seconds starting at init_tdb_s, each carrying n_coeffs coefficients per
  // component, laid out [record][component][coefficient] in ascending
  // Chebyshev order (docs/formats/sreph_v1.md section 5).
  struct Segment {
    std::string name;
    std::uint32_t target = 0;
    std::uint32_t center = 0;
    std::uint32_t kind = 0;  // 0 = position [km], 1 = Euler angles [rad]
    std::uint32_t n_coeffs = 0;
    std::uint32_t n_records = 0;
    double init_tdb_s = 0.0;
    double intlen_s = 0.0;
    std::vector<double> coeffs;

    double end_tdb_s() const {
      return init_tdb_s + static_cast<double>(n_records) * intlen_s;
    }
  };

  const Segment& find_segment(const std::string& body, double tdb_s,
                              std::uint32_t expected_kind) const;
  static void evaluate(const Segment& seg, double tdb_s, double value[3],
                       double rate[3]);

  double span_start_tdb_s_ = 0.0;
  double span_end_tdb_s_ = 0.0;
  std::vector<Segment> segments_;
};

}  // namespace star

#endif  // STAR_EPHEMERIS_HPP
