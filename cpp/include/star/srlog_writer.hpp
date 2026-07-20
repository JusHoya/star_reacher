// SRLOG v1.2 binary log writer (PRD D-11, FR-16; byte format normative in
// docs/formats/srlog_v1.md). The Phase 1 two-group schema (`truth` +
// `events`) is always present; v1.1 adds the optional vehicle channel groups
// (`forces`, `mass`, `env`) and v1.2 the optional GNC channel groups
// (`sensors.*`, `nav.*`, `gnc.cmd`), all declared at header-write time.
// Later phases extend the channel dictionary through minor-version bumps,
// never layout breaks.
//
// The header JSON is hand-rolled with a fixed key order and contains only
// integers, booleans, and strings - no floats - so the exact header bytes are
// a pure function of the config values and the producer identity. That is
// what makes the double-run SHA-256 determinism gate (FR-21) meaningful at
// the whole-file level.
#ifndef STAR_SRLOG_WRITER_HPP
#define STAR_SRLOG_WRITER_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include <Eigen/Dense>

namespace star {
namespace log {

// One declared v1.2 sensor channel group (format doc section 3.2). `kind`
// names the group `sensors.<kind>` and must come from the canonical sensor
// vocabulary (kSensorKinds); `rate_hz` is the sensor's sample rate;
// `landmarks` is the camera hook's declared landmark-projection count and
// must be zero for every other kind.
struct SensorGroupDecl {
  std::string kind;
  std::uint32_t rate_hz = 0;
  std::uint32_t landmarks = 0;
};

// Header identity fields. Everything here is input-derived or build-derived;
// wall-clock and host data are banned from the file by D-11 (they live in the
// Python-written meta.json sidecar).
struct SrlogHeaderFields {
  std::string core_version;   // e.g. "0.1.0"
  std::string git_hash;       // 40-hex or "unknown"
  std::string config_sha256;  // 64-hex, computed by the Python validator
  std::uint64_t master_seed = 0;  // serialized as a decimal string (JSON has no u64)
  bool oracle = false;
  std::string epoch_utc;      // ISO-8601, carried verbatim from the mission file
  std::string central_body;   // "earth" in Phase 1
  std::uint32_t truth_rate_hz = 0;

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

  // v1.2 GNC channel groups (FR-16 reserved names; format doc section 3.2).
  // cycle_rate_hz is the control-cycle rate that anchors every periodic
  // v1.2 group: each declared rate must divide it exactly (records are
  // emitted on the control-cycle grid; decimation only, never
  // interpolation), and cycle_rate_hz == 0 means no v1.2 group may be
  // declared. cycle_rate_hz, latency_cycles, and the declared sensor kinds
  // are echoed in the header's "gnc" object so a latency study (Phase 6
  // exit criterion 8) and the nav.innov sensor-identity table are readable
  // from the header alone.
  std::uint32_t cycle_rate_hz = 0;
  std::uint32_t latency_cycles = 0;
  // Declared sensor groups, in canonical kind order without duplicates
  // (kSensorKinds), mirroring the force_sources discipline: the header
  // bytes are a pure function of the declared set.
  std::vector<SensorGroupDecl> sensors;
  // nav.est: t_s, x_hat f64[n], P f64[m(m+1)/2] (packed row-major upper
  // triangle, the FR-26 convention shared with the mass group's inertia).
  // Rate and state dimension are declared together or not at all. The
  // covariance dimension m defaults to n (nav_cov_dim == 0) and may be
  // declared independently for estimators whose covariance lives in a
  // different parameterization than the state: the reference error-state
  // EKF (ch:ekf, a later workstream) declares n = 16 (q scalar-first, v,
  // p, b_g, b_a) with m = 15 (3-component attitude error), so P carries
  // 120 doubles.
  std::uint32_t nav_est_rate_hz = 0;
  std::uint32_t nav_state_dim = 0;
  std::uint32_t nav_cov_dim = 0;
  // nav.err: t_s, e f64[n] - the truth-minus-estimate error state in the
  // estimator's own state convention. Its dimension and rate are pinned to
  // nav.est by construction (one enable flag, no independent fields): the
  // consistency tooling computes NEES from nav.err.e and nav.est.P
  // directly and requires matching record counts, so a file cannot even
  // express a mismatched declaration.
  bool nav_err_enabled = false;
  // nav.innov: aperiodic (rate_hz 0), one record per aiding update, tagged
  // by sensor identity; y f64[m_max], S f64[m_max(m_max+1)/2]. Requires at
  // least one declared sensor (sensor_id indexes the "gnc" sensors array).
  bool nav_innov_enabled = false;
  std::uint32_t nav_innov_max_dim = 0;
  // gnc.cmd: the command as applied each control cycle (exit criterion 8's
  // instrument: latency_cycles = k must visibly shift application here).
  std::uint32_t gnc_cmd_rate_hz = 0;
};

// Canonical FR-16 force/torque source vocabulary, in canonical order (format
// doc section 3.1). Extending it is a minor version bump.
inline constexpr const char* kForceSources[] = {
    "gravity", "thirdbody", "srp", "drag", "aero",
    "thrust",  "rcs",       "gravgrad", "wheel"};
inline constexpr std::size_t kForceSourceCount =
    sizeof(kForceSources) / sizeof(kForceSources[0]);

// Canonical FR-23 sensor-kind vocabulary, in canonical order (format doc
// section 3.2). At most one `sensors.<kind>` group per kind exists in a
// v1.2 file; multiple instances of one kind is a future minor bump.
inline constexpr const char* kSensorKinds[] = {
    "imu", "startracker", "sunsensor", "navfix", "altimeter", "camera"};
inline constexpr std::size_t kSensorKindCount =
    sizeof(kSensorKinds) / sizeof(kSensorKinds[0]);

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

  // --- v1.2 GNC group writers (format doc section 3.2) --------------------
  // Every method throws std::logic_error when its group was not declared at
  // header-write time, and std::invalid_argument on a payload whose size
  // disagrees with the declared dimensions (a short record would corrupt
  // the fixed-stride stream).

  // One `sensors.imu` record: accumulated angle and velocity increments
  // over the sample interval, body frame.
  void write_sensor_imu(double t_s, const Eigen::Vector3d& dtheta_b_rad,
                        const Eigen::Vector3d& dv_b_mps);

  // One `sensors.startracker` record: measured attitude quaternion
  // (Hamilton scalar-first, D-7) and the exclusion/slew validity flag.
  void write_sensor_startracker(double t_s, const double (&q_meas_i2b)[4],
                                std::uint32_t valid);

  // One `sensors.sunsensor` record: measured Sun unit vector, body frame,
  // and the field-of-view validity flag.
  void write_sensor_sunsensor(double t_s, const Eigen::Vector3d& sun_b,
                              std::uint32_t valid);

  // One `sensors.navfix` record: measured GCRF position and velocity.
  void write_sensor_navfix(double t_s, const Eigen::Vector3d& r_meas_m,
                           const Eigen::Vector3d& v_meas_mps);

  // One `sensors.altimeter` record: measured altitude.
  void write_sensor_altimeter(double t_s, double alt_meas_m);

  // One `sensors.camera` record: geometric-truth pose, plus 2*landmarks
  // pixel coordinates (u0, v0, u1, v1, ...) when the declaration carried a
  // nonzero landmark count. px_count must equal 2*landmarks exactly.
  void write_sensor_camera(double t_s, const Eigen::Vector3d& r_m,
                           const double (&q_i2b)[4], const double* px_uv,
                           std::size_t px_count);

  // One `nav.est` record: the estimator state x_hat (n doubles) and its
  // covariance P packed as the row-major upper triangle (m(m+1)/2 doubles,
  // the FR-26 packing). n must equal the declared nav_state_dim and p_len
  // the declared covariance dimension's triangle size (m = nav_cov_dim,
  // defaulting to n).
  void write_nav_est(double t_s, const double* x_hat, std::size_t n,
                     const double* p_upper, std::size_t p_len);

  // One `nav.err` record: the truth-minus-estimate error state in the
  // estimator's state convention (n doubles, the declared nav_state_dim).
  void write_nav_err(double t_s, const double* e, std::size_t n);

  // One `nav.innov` record (aperiodic, per aiding update): sensor_id
  // indexes the header's "gnc" sensors array, m is the update's valid
  // innovation dimension (<= the declared maximum), and y/S carry m_max
  // and m_max(m_max+1)/2 doubles with entries beyond m zero-filled.
  void write_nav_innov(double t_s, std::uint32_t sensor_id, std::uint32_t m,
                       const double* y, std::size_t y_len,
                       const double* s_upper, std::size_t s_len);

  // One `gnc.cmd` record: the command as applied this control cycle -
  // post-latency, post-saturation - with valid = 1 for a fresh GNC output
  // and 0 for a held command.
  void write_gnc_cmd(double t_s, const Eigen::Vector3d& tau_b_nm,
                     const double (&q_cmd_i2b)[4],
                     const Eigen::Vector3d& w_cmd_b_radps,
                     std::uint32_t valid);

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
  // Record group indices of the enabled v1.1/v1.2 groups; -1 marks a group
  // absent from this file (its write_* call is then a logic error).
  int forces_index_ = -1;
  int mass_index_ = -1;
  int env_index_ = -1;
  std::size_t force_source_count_ = 0;
  // std::array filled in the constructor rather than a brace-initialised C
  // array: a brace list one element short value-initialises the remainder to
  // 0, and 0 is the group index of `truth`. A seventh sensor kind - which
  // this header sanctions as a minor bump - would therefore pass the
  // `idx < 0` guard in every write_sensor_* and tag a sensor payload as a
  // truth record, producing a SILENTLY corrupt log rather than a crash.
  std::array<int, kSensorKindCount> sensor_index_{};
  int nav_est_index_ = -1;
  int nav_err_index_ = -1;
  int nav_innov_index_ = -1;
  int gnc_cmd_index_ = -1;
  // Declared dimensions the v1.2 write calls are re-checked against.
  std::size_t camera_px_count_ = 0;
  std::size_t nav_state_dim_ = 0;
  std::size_t nav_cov_dim_ = 0;
  std::size_t nav_innov_max_dim_ = 0;
  // Declared sensor count, so nav.innov's sensor_id can be bound-checked
  // against the header array it indexes.
  std::size_t sensor_count_ = 0;
};

}  // namespace log
}  // namespace star

#endif  // STAR_SRLOG_WRITER_HPP
