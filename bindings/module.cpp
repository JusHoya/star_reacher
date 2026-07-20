// pybind11 bindings for star_reacher._core (Phase 1 contract section 3).
// The binding layer is deliberately thin: it converts types and forwards to
// star::; all behavior lives in the C++ core so the doctest suite exercises
// the same code paths the Python frontend uses.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "star/ephemeris.hpp"
#include "star/frames.hpp"
#include "star/gnc/builtin.hpp"
#include "star/gnc/component.hpp"
#include "star/models/atmosphere_hp.hpp"
#include "star/models/atmosphere_mars.hpp"
#include "star/models/atmosphere_ussa76.hpp"
#include "star/models/drag.hpp"
#include "star/models/gravity.hpp"
#include "star/models/srp.hpp"
#include "star/models/thirdbody.hpp"
#include "star/rng.hpp"
#include "star/rotation.hpp"
#include "star/run.hpp"
#include "star/testsupport/acceptance.hpp"
#include "star/testsupport/kepler_ref.hpp"
#include "star/time.hpp"
#include "star/vehicle_cycle.hpp"
#include "star/version.hpp"

namespace py = pybind11;

namespace {

Eigen::Vector3d to_vec3(const std::array<double, 3>& a) {
  return Eigen::Vector3d(a[0], a[1], a[2]);
}

std::array<double, 3> from_vec3(const Eigen::Vector3d& v) {
  return {v[0], v[1], v[2]};
}

// ---------------------------------------------------------------------------
// Test-support entry points (Phase 2 acceptance evidence). These forward to
// star/testsupport/acceptance.hpp -- the SAME drivers the doctest suite
// asserts on -- so the Python-side evidence is the same numbers by
// construction. They are not part of the mission-run surface.
// ---------------------------------------------------------------------------

py::dict propagate_kepler(double mu, const std::array<double, 3>& r0,
                          const std::array<double, 3>& v0, double t) {
  Eigen::Vector3d r, v;
  star::testsupport::propagate_kepler(mu, to_vec3(r0), to_vec3(v0), t, &r,
                                      &v);
  py::dict d;
  d["r_m"] = from_vec3(r);
  d["v_mps"] = from_vec3(v);
  return d;
}

py::list kepler_convergence(double mu, const std::array<double, 3>& r0,
                            const std::array<double, 3>& v0, double t_end,
                            const std::string& method,
                            const std::vector<double>& ladder) {
  star::events::Method m;
  if (method == "rk4") {
    m = star::events::Method::kRk4;
  } else if (method == "rkf78") {
    m = star::events::Method::kRkf78;
  } else {
    throw std::invalid_argument("method must be \"rk4\" or \"rkf78\"");
  }
  const auto pts = star::testsupport::kepler_convergence(
      mu, to_vec3(r0), to_vec3(v0), t_end, m, ladder);
  py::list out;
  for (const auto& p : pts) {
    py::dict d;
    d["h_s"] = p.h_s;
    d["err_m"] = p.err_m;
    out.append(d);
  }
  return out;
}

py::dict twobody_drift(double mu, const std::array<double, 3>& r0,
                       const std::array<double, 3>& v0, double n_orbits,
                       double rtol, double atol_pos_m, double atol_vel_mps,
                       double h_init, double h_max) {
  const auto res = star::testsupport::twobody_drift(
      mu, to_vec3(r0), to_vec3(v0), n_orbits, rtol, atol_pos_m, atol_vel_mps,
      h_init, h_max);
  py::dict d;
  d["max_energy_rel"] = res.max_energy_rel;
  d["max_hmag_rel"] = res.max_hmag_rel;
  d["steps_accepted"] = res.steps_accepted;
  d["steps_rejected"] = res.steps_rejected;
  return d;
}

py::list apsis_events(double mu, const std::array<double, 3>& r0,
                      const std::array<double, 3>& v0, double t_end,
                      double rtol, double atol_pos_m, double atol_vel_mps,
                      double h_init, double h_max, double event_tol_s) {
  const auto res = star::testsupport::apsis_event_scan(
      mu, to_vec3(r0), to_vec3(v0), t_end, rtol, atol_pos_m, atol_vel_mps,
      h_init, h_max, event_tol_s);
  py::list out;
  for (const auto& hit : res.hits) {
    py::dict d;
    d["t_s"] = hit.t_s;
    d["kind"] = hit.periapsis ? "periapsis" : "apoapsis";
    out.append(d);
  }
  return out;
}

py::dict hermite_midstep_max_err(double mu, const std::array<double, 3>& r0,
                                 const std::array<double, 3>& v0,
                                 double t_end, double h) {
  py::dict d;
  d["max_err_m"] = star::testsupport::hermite_midstep_max_err(
      mu, to_vec3(r0), to_vec3(v0), t_end, h);
  return d;
}

std::vector<std::uint64_t> rng_stream_u64(std::uint64_t master_seed,
                                          const std::string& stream_name,
                                          std::size_t n) {
  star::rng::Pcg64 gen = star::rng::make_stream(master_seed, stream_name);
  std::vector<std::uint64_t> out;
  out.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    out.push_back(gen.next());
  }
  return out;
}

std::vector<double> rng_stream_normal(std::uint64_t master_seed,
                                      const std::string& stream_name,
                                      std::size_t n) {
  star::rng::NormalSampler sampler(
      star::rng::make_stream(master_seed, stream_name));
  std::vector<double> out;
  out.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    out.push_back(sampler.next());
  }
  return out;
}

// Time-system wrappers (FR-2, D-6). Epochs cross the binding as plain
// (day, sec) / calendar-field tuples rather than bound classes: the values
// are two ints and a handful of doubles, and tuples keep the Python side
// free of core object lifetimes. std::domain_error from star::time maps to
// Python ValueError via the pybind11 built-in exception translator.
py::tuple utc_to_tai(int year, int month, int day, int hour, int minute,
                     double second) {
  const star::time::TaiEpoch tai =
      star::time::tai_from_utc({year, month, day, hour, minute, second});
  return py::make_tuple(tai.day, tai.sec);
}

py::tuple tai_to_utc(std::int64_t day, double sec) {
  const star::time::UtcTime utc = star::time::utc_from_tai({day, sec});
  return py::make_tuple(utc.year, utc.month, utc.day, utc.hour, utc.minute,
                        utc.second);
}

py::tuple tai_to_jd(std::int64_t day, double sec) {
  const star::time::TwoPartJd jd = star::time::tai_jd({day, sec});
  return py::make_tuple(jd.jd1, jd.jd2);
}

py::tuple tt_to_jd(std::int64_t day, double sec) {
  const star::time::TwoPartJd jd = star::time::tt_jd({day, sec});
  return py::make_tuple(jd.jd1, jd.jd2);
}

py::tuple tdb_to_jd(std::int64_t day, double sec) {
  const star::time::TwoPartJd jd = star::time::tdb_jd({day, sec});
  return py::make_tuple(jd.jd1, jd.jd2);
}

py::tuple tai_add_seconds(std::int64_t day, double sec, double delta_s) {
  const star::time::TaiEpoch out =
      star::time::tai_add_seconds({day, sec}, delta_s);
  return py::make_tuple(out.day, out.sec);
}

py::dict leap_table_info() {
  const star::time::LeapTableInfo info = star::time::leap_table_info();
  py::dict d;
  d["version"] = std::string(info.version);
  d["expiry_utc"] =
      py::make_tuple(info.expiry_year, info.expiry_month, info.expiry_day);
  d["entries"] = info.entries;
  return d;
}

std::vector<double> to_vec3(const Eigen::Vector3d& v) {
  return {v[0], v[1], v[2]};
}

py::tuple state_tuple(const star::EphemerisState& s) {
  return py::make_tuple(to_vec3(s.r_m), to_vec3(s.v_mps));
}

// Rotation and frame wrappers (FR-3, D-7). Quaternions cross the binding
// as scalar-first (w, x, y, z) tuples - never Eigen coeffs() order (the
// D-7 storage trap documented in star/rotation.hpp) - and 3x3 matrices as
// row-major 9-element lists; both stay plain Python types so the binding
// carries no core object lifetimes.
std::array<double, 9> mat_out(const Eigen::Matrix3d& m) {
  return {m(0, 0), m(0, 1), m(0, 2), m(1, 0), m(1, 1),
          m(1, 2), m(2, 0), m(2, 1), m(2, 2)};
}

Eigen::Matrix3d mat_in(const std::array<double, 9>& a) {
  Eigen::Matrix3d m;
  m << a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8];
  return m;
}

py::tuple quat_out(const Eigen::Quaterniond& q) {
  return py::make_tuple(q.w(), q.x(), q.y(), q.z());
}

py::tuple quat_multiply(double pw, double px, double py_, double pz,
                        double qw, double qx, double qy, double qz) {
  return quat_out(star::rotation::quat_multiply(
      Eigen::Quaterniond(pw, px, py_, pz), Eigen::Quaterniond(qw, qx, qy, qz)));
}

py::tuple quat_conjugate(double w, double x, double y, double z) {
  return quat_out(star::rotation::quat_conjugate(Eigen::Quaterniond(w, x, y, z)));
}

py::tuple quat_normalize(double w, double x, double y, double z) {
  return quat_out(star::rotation::quat_normalize(Eigen::Quaterniond(w, x, y, z)));
}

py::tuple quat_transform(double w, double x, double y, double z, double vx,
                         double vy, double vz) {
  const Eigen::Vector3d v = star::rotation::quat_transform(
      Eigen::Quaterniond(w, x, y, z), Eigen::Vector3d(vx, vy, vz));
  return py::make_tuple(v.x(), v.y(), v.z());
}

std::array<double, 9> quat_to_dcm(double w, double x, double y, double z) {
  return mat_out(star::rotation::dcm_from_quat(Eigen::Quaterniond(w, x, y, z)));
}

py::tuple dcm_to_quat(const std::array<double, 9>& m) {
  return quat_out(star::rotation::quat_from_dcm(mat_in(m)));
}

std::array<double, 9> dcm_from_euler321(double a1, double a2, double a3) {
  return mat_out(star::rotation::dcm_from_euler321(a1, a2, a3));
}

std::array<double, 9> dcm_from_euler313(double a1, double a2, double a3) {
  return mat_out(star::rotation::dcm_from_euler313(a1, a2, a3));
}

py::tuple euler321_from_dcm(const std::array<double, 9>& m) {
  double a1;
  double a2;
  double a3;
  star::rotation::euler321_from_dcm(mat_in(m), a1, a2, a3);
  return py::make_tuple(a1, a2, a3);
}

py::tuple euler313_from_dcm(const std::array<double, 9>& m) {
  double a1;
  double a2;
  double a3;
  star::rotation::euler313_from_dcm(mat_in(m), a1, a2, a3);
  return py::make_tuple(a1, a2, a3);
}

py::tuple cip_cio_06b(std::int64_t day, double sec) {
  const star::frames::CipCio c = star::frames::cip_cio_06b({day, sec});
  return py::make_tuple(c.x_rad, c.y_rad, c.s_rad);
}

py::tuple nutation_00b(double tt_centuries) {
  double dpsi;
  double deps;
  star::frames::nutation_00b(tt_centuries, dpsi, deps);
  return py::make_tuple(dpsi, deps);
}

double era_00(std::int64_t day, double sec, double dut1_s) {
  return star::frames::era_00({day, sec}, dut1_s);
}

std::array<double, 9> gcrf_to_cirs(std::int64_t day, double sec) {
  return mat_out(star::frames::c_gcrf_to_cirs({day, sec}));
}

std::array<double, 9> gcrf_to_itrf(std::int64_t day, double sec,
                                   double dut1_s) {
  return mat_out(star::frames::c_gcrf_to_itrf({day, sec}, dut1_s));
}

std::array<double, 9> gcrf_to_moonpa(double phi, double theta, double psi) {
  return mat_out(star::frames::c_gcrf_to_moonpa(phi, theta, psi));
}

py::tuple mars_elements(std::int64_t day, double sec) {
  const star::frames::MarsElements e =
      star::frames::mars_elements_iau2015({day, sec});
  return py::make_tuple(e.alpha0_rad, e.delta0_rad, e.w_rad);
}

std::array<double, 9> gcrf_to_marsfixed(std::int64_t day, double sec) {
  return mat_out(star::frames::c_gcrf_to_marsfixed({day, sec}));
}

py::dict summary_dict(const star::RunSummary& s) {
  py::dict d;
  d["steps"] = s.steps;
  d["final_r_m"] = s.final_r_m;
  d["final_v_mps"] = s.final_v_mps;
  d["truth_records"] = s.truth_records;
  d["event_records"] = s.event_records;
  d["records_written"] = s.truth_records + s.event_records;
  return d;
}

py::dict run_and_summarize(const star::RunConfig& cfg,
                           const std::string& out_path) {
  return summary_dict(star::run_twobody(cfg, out_path));
}

py::dict run_env_and_summarize(const star::RunConfig& cfg,
                               const std::string& out_path) {
  return summary_dict(star::run_env(cfg, out_path));
}

py::dict run_vehicle_and_summarize(const star::RunConfig& cfg,
                                   const std::string& out_path) {
  return summary_dict(star::run_vehicle(cfg, out_path));
}

// ---------------------------------------------------------------------------
// Phase 3 model surface (verify checks V014+ and the pytest suites exercise
// the compiled models directly through these thin wrappers).
// ---------------------------------------------------------------------------

star::models::GravityTier tier_from_name(const std::string& tier) {
  if (tier == "pointmass") return star::models::GravityTier::kPointMass;
  if (tier == "j2") return star::models::GravityTier::kJ2Only;
  if (tier == "full") return star::models::GravityTier::kFull;
  throw std::invalid_argument(
      "tier must be \"pointmass\", \"j2\", or \"full\"");
}

std::array<double, 3> gravity_accel(const std::string& srgrav_path,
                                    const std::string& tier, int degree,
                                    int order,
                                    const std::array<double, 3>& r_bf_m) {
  // Loads the field per call: this entry point serves verification and
  // testing, not the propagation loop (run_env constructs its evaluator
  // once, before the time loop).
  star::models::PinesGravity model(
      star::models::GravityField::load_file(srgrav_path));
  return from_vec3(
      model.acceleration(to_vec3(r_bf_m), tier_from_name(tier), degree, order));
}

std::array<double, 3> thirdbody_accel(double gm_third,
                                      const std::array<double, 3>& r_sc_m,
                                      const std::array<double, 3>& r_third_m) {
  return from_vec3(star::models::thirdbody_accel(gm_third, to_vec3(r_sc_m),
                                                 to_vec3(r_third_m)));
}

double shadow_fraction(const std::array<double, 3>& r_sc_m,
                       const std::array<double, 3>& r_sun_m,
                       double radius_sun_m,
                       const std::array<double, 3>& r_occ_m,
                       double radius_occ_m) {
  return star::models::shadow_fraction(to_vec3(r_sc_m), to_vec3(r_sun_m),
                                       radius_sun_m, to_vec3(r_occ_m),
                                       radius_occ_m);
}

std::array<double, 3> srp_accel(double cr_a_over_m, double nu,
                                const std::array<double, 3>& r_sc_m,
                                const std::array<double, 3>& r_sun_m) {
  return from_vec3(star::models::srp_accel(cr_a_over_m, nu, to_vec3(r_sc_m),
                                           to_vec3(r_sun_m)));
}

std::array<double, 3> drag_accel(double rho_kgpm3, double cd_a_over_m,
                                 const std::array<double, 3>& v_rel_mps) {
  return from_vec3(
      star::models::drag_accel(rho_kgpm3, cd_a_over_m, to_vec3(v_rel_mps)));
}

double geodetic_altitude(const std::array<double, 3>& r_ecef_m, double a_m,
                         double inv_f) {
  return star::models::geodetic_altitude(to_vec3(r_ecef_m), a_m, inv_f);
}

// ---------------------------------------------------------------------------
// Phase 6 GNC/stepping surface (FR-24 Sim, FR-25 Python components).
//
// Marshalling rules, uniform across every struct below:
//   * Eigen::Vector3d crosses as a 3-element list, Eigen::Matrix3d as a
//     9-element row-major list, Eigen::Quaterniond as a scalar-first
//     (w, x, y, z) 4-tuple - NEVER Eigen coeffs() order (the D-7 storage
//     trap documented in star/rotation.hpp).
//   * Every accessor COPIES. Nothing handed to Python is a view into a core
//     buffer, which is what lets observe() satisfy the exit-criterion-4
//     idempotence clause: the dict cannot alias memory a later step()
//     overwrites.
// ---------------------------------------------------------------------------

Eigen::Quaterniond quat_in(const std::array<double, 4>& q) {
  return Eigen::Quaterniond(q[0], q[1], q[2], q[3]);  // scalar-first (D-7)
}

std::array<double, 4> quat_arr(const Eigen::Quaterniond& q) {
  return {q.w(), q.x(), q.y(), q.z()};
}

// Property helpers for the fixed-size Eigen members. Macros rather than
// thirty hand-written lambda pairs: the conversion is mechanical and
// identical everywhere, so writing it once removes the class of bug where
// one field silently uses a different component order.
#define STAR_V3_PROP(Cls, Name, Member)                                    \
  def_property(                                                            \
      Name, [](const Cls& s) { return from_vec3(s.Member); },              \
      [](Cls& s, const std::array<double, 3>& a) { s.Member = to_vec3(a); })
#define STAR_QUAT_PROP(Cls, Name, Member)                                  \
  def_property(                                                            \
      Name, [](const Cls& s) { return quat_arr(s.Member); },               \
      [](Cls& s, const std::array<double, 4>& a) { s.Member = quat_in(a); })
#define STAR_M3_PROP(Cls, Name, Member)                                    \
  def_property(                                                            \
      Name, [](const Cls& s) { return mat_out(s.Member); },                \
      [](Cls& s, const std::array<double, 9>& a) { s.Member = mat_in(a); })

// Copy a Python sequence of n floats into a core buffer. A component that
// returns the wrong length is a configuration error the loop must not
// absorb: writing short would leave stale values in the log and writing long
// would corrupt memory, so both are refused by name.
void copy_fixed(const py::object& value, const char* method, int n,
                double* out) {
  const std::vector<double> v = value.cast<std::vector<double>>();
  if (static_cast<int>(v.size()) != n) {
    throw std::length_error(
        std::string("gnc component method ") + method + " returned " +
        std::to_string(v.size()) + " values; the declared dimension requires " +
        std::to_string(n));
  }
  std::copy(v.begin(), v.end(), out);
}

// The FR-25 pybind11 trampoline: a Python subclass of _core.IGncComponent
// plugs into the same registry and the same chain as a built-in C++
// component. init() and update() are pure virtual and must be overridden;
// the introspection methods fall back to the C++ defaults ("not an
// estimator") when the subclass does not define them, so a Python guidance
// or control law implements exactly two methods.
//
// DETERMINISM: the component runs INSIDE the deterministic time loop, so
// the D-10 guarantee is only as strong as the Python in it. See the
// star_reacher.sim module docstring for the contract a component must
// respect; nothing here can enforce it.
class PyGncComponent : public star::gnc::IGncComponent {
 public:
  using star::gnc::IGncComponent::IGncComponent;

  void init(const star::gnc::GncInitContext& ctx) override {
    PYBIND11_OVERRIDE_PURE(void, IGncComponent, init, ctx);
  }

  star::gnc::GncOutput update(const star::gnc::GncInput& input) override {
    PYBIND11_OVERRIDE_PURE(star::gnc::GncOutput, IGncComponent, update, input);
  }

  // The zero-argument introspection methods are dispatched explicitly rather
  // than through PYBIND11_OVERRIDE: that macro expands to a variadic call
  // with an empty argument list, which is a pedantic-mode diagnostic under
  // the -Werror Linux job.
  int state_dim() const override {
    int v = 0;
    if (int_override("state_dim", &v)) return v;
    return star::gnc::IGncComponent::state_dim();
  }

  int cov_dim() const override {
    int v = 0;
    if (int_override("cov_dim", &v)) return v;
    // The base default is state_dim(), which dispatches back into Python:
    // a subclass that declares only state_dim() gets a square covariance of
    // that dimension without writing a second method.
    return star::gnc::IGncComponent::cov_dim();
  }

  int innov_max_dim() const override {
    int v = 0;
    if (int_override("innov_max_dim", &v)) return v;
    return star::gnc::IGncComponent::innov_max_dim();
  }

  const std::vector<star::gnc::InnovationSample>& innovations() const override {
    py::gil_scoped_acquire gil;
    const py::function ov = py::get_override(this, "innovations");
    if (!ov) return star::gnc::IGncComponent::innovations();
    // Cached in a member because the interface returns a reference: the
    // Python list is converted once per call and outlives the temporary.
    innov_cache_ = ov().cast<std::vector<star::gnc::InnovationSample>>();
    return innov_cache_;
  }

  void state(double* x_hat) const override {
    py::gil_scoped_acquire gil;
    const py::function ov = py::get_override(this, "state");
    if (!ov) return star::gnc::IGncComponent::state(x_hat);
    copy_fixed(ov(), "state", state_dim(), x_hat);
  }

  void covariance_upper(double* p) const override {
    py::gil_scoped_acquire gil;
    const py::function ov = py::get_override(this, "covariance_upper");
    if (!ov) return star::gnc::IGncComponent::covariance_upper(p);
    const int m = cov_dim();
    copy_fixed(ov(), "covariance_upper", m * (m + 1) / 2, p);
  }

  // The FR-24 boundary in the trampoline: what a Python estimator returns
  // here is a DESCRIPTION of its state vector, and no argument carries the
  // state of the world. There is deliberately no truth-bearing virtual for a
  // subclass to override - the loop computes nav.err itself from this
  // declaration (gnc/component.hpp).
  const std::vector<star::gnc::ErrorBlock>& error_layout() const override {
    py::gil_scoped_acquire gil;
    const py::function ov = py::get_override(this, "error_layout");
    if (!ov) return star::gnc::IGncComponent::error_layout();
    // Cached in a member for the same reason as innovations(): the interface
    // returns a reference, so the converted vector must outlive the call.
    layout_cache_ = ov().cast<std::vector<star::gnc::ErrorBlock>>();
    return layout_cache_;
  }

 private:
  // True when the subclass defines `name`, with its value in *out. Kept
  // lazy so the C++ fallback is not evaluated when an override exists.
  bool int_override(const char* name, int* out) const {
    py::gil_scoped_acquire gil;
    const py::function ov = py::get_override(this, name);
    if (!ov) return false;
    *out = ov().cast<int>();
    return true;
  }

  mutable std::vector<star::gnc::InnovationSample> innov_cache_;
  mutable std::vector<star::gnc::ErrorBlock> layout_cache_;
};

// Name -> Python factory table for components registered from Python. It is
// a side table because gnc::GncFactory is a plain function pointer and so
// cannot capture the class object; the single C++ factory below recovers the
// class from cfg.component, which the registry passes through verbatim.
std::map<std::string, py::object>& python_component_table() {
  // Deliberately leaked: a static map of py::object would release Python
  // references during static destruction, which runs after the interpreter
  // has finalized. The table lives as long as the process, which is exactly
  // as long as a registered component can be selected.
  static std::map<std::string, py::object>* table =
      new std::map<std::string, py::object>();
  return *table;
}

// Owns the Python object for the lifetime the core expects of a component
// and forwards the interface to the trampoline inside it. The core's
// unique_ptr then deletes only this handle, dropping one Python reference,
// rather than trying to delete an object Python owns.
class PythonComponentHandle : public star::gnc::IGncComponent {
 public:
  explicit PythonComponentHandle(py::object obj)
      : obj_(std::move(obj)), impl_(obj_.cast<star::gnc::IGncComponent*>()) {}

  ~PythonComponentHandle() override {
    py::gil_scoped_acquire gil;  // releasing a Python reference needs the GIL
    obj_ = py::object();
  }

  void init(const star::gnc::GncInitContext& ctx) override { impl_->init(ctx); }
  star::gnc::GncOutput update(const star::gnc::GncInput& in) override {
    return impl_->update(in);
  }
  int state_dim() const override { return impl_->state_dim(); }
  int cov_dim() const override { return impl_->cov_dim(); }
  int innov_max_dim() const override { return impl_->innov_max_dim(); }
  const std::vector<star::gnc::InnovationSample>& innovations() const override {
    return impl_->innovations();
  }
  void state(double* x) const override { impl_->state(x); }
  void covariance_upper(double* p) const override {
    impl_->covariance_upper(p);
  }
  const std::vector<star::gnc::ErrorBlock>& error_layout() const override {
    return impl_->error_layout();
  }

 private:
  py::object obj_;
  star::gnc::IGncComponent* impl_;
};

std::unique_ptr<star::gnc::IGncComponent> make_python_component(
    const star::gnc::GncComponentCfg& cfg) {
  py::gil_scoped_acquire gil;
  const auto it = python_component_table().find(cfg.component);
  if (it == python_component_table().end()) {
    throw std::invalid_argument(
        "gnc component '" + cfg.component +
        "' is registered as a Python component but its factory is no longer "
        "present; register_python_component must be called before the run");
  }
  py::object obj = it->second(cfg);
  if (!py::isinstance<star::gnc::IGncComponent>(obj)) {
    throw std::invalid_argument(
        "the factory for gnc component '" + cfg.component +
        "' returned an object that does not derive from IGncComponent");
  }
  return std::unique_ptr<star::gnc::IGncComponent>(
      new PythonComponentHandle(std::move(obj)));
}

void register_python_component(const std::string& name, py::object factory) {
  if (!py::hasattr(factory, "__call__")) {
    throw std::invalid_argument(
        "register_python_component: factory must be callable as "
        "factory(cfg) -> IGncComponent (a subclass object is such a "
        "callable)");
  }
  // Register with the core FIRST: a duplicate name throws there, and the
  // side table must not gain an entry the core does not know about.
  star::gnc::register_component(name, &make_python_component);
  python_component_table()[name] = std::move(factory);
}

// -- FR-24 observation marshalling -----------------------------------------

py::dict imu_dict(const star::gnc::ImuSample& s) {
  py::dict d;
  d["valid"] = s.valid;
  d["t_s"] = s.t_s;
  d["dt_s"] = s.dt_s;
  d["dtheta_b_rad"] = from_vec3(s.dtheta_b_rad);
  d["dv_b_mps"] = from_vec3(s.dv_b_mps);
  return d;
}

py::dict navfix_dict(const star::gnc::NavFixSample& s) {
  py::dict d;
  d["valid"] = s.valid;
  d["fresh"] = s.fresh;
  d["sensor_id"] = s.sensor_id;
  d["r_i_m"] = from_vec3(s.r_i_m);
  d["v_i_mps"] = from_vec3(s.v_i_mps);
  return d;
}

py::dict startracker_dict(const star::gnc::StarTrackerSample& s) {
  py::dict d;
  d["valid"] = s.valid;
  d["fresh"] = s.fresh;
  d["sensor_id"] = s.sensor_id;
  d["q_i2b"] = quat_arr(s.q_i2b);
  return d;
}

py::dict altimeter_dict(const star::gnc::AltimeterSample& s) {
  py::dict d;
  d["valid"] = s.valid;
  d["fresh"] = s.fresh;
  d["sensor_id"] = s.sensor_id;
  d["h_m"] = s.h_m;
  return d;
}

py::dict env_dict(const star::gnc::NavEnvironment& e) {
  py::dict d;
  d["ephemeris_valid"] = e.ephemeris_valid;
  d["v_central_ssb_mps"] = from_vec3(e.v_central_ssb_mps);
  d["bodyfixed_valid"] = e.bodyfixed_valid;
  d["c_gcrf_to_bodyfixed"] = mat_out(e.c_gcrf_to_bodyfixed);
  return d;
}

py::dict output_dict(const star::gnc::GncOutput& o) {
  py::dict d;
  d["valid"] = o.valid;
  d["q_i2b"] = quat_arr(o.q_i2b);
  d["omega_b_radps"] = from_vec3(o.omega_b_radps);
  d["torque_b_nm"] = from_vec3(o.torque_b_nm);
  return d;
}

py::dict observation_dict(const star::CycleObservation& o) {
  py::dict d;
  d["cycle"] = o.cycle;
  d["t_s"] = o.t_s;
  d["processed"] = o.processed;
  d["done"] = o.done;
  d["gnc_active"] = o.gnc_active;
  d["imu"] = imu_dict(o.imu);
  d["imu_fresh"] = o.imu_fresh;
  d["navfix"] = navfix_dict(o.navfix);
  d["startracker"] = startracker_dict(o.startracker);
  d["altimeter"] = altimeter_dict(o.altimeter);
  d["env"] = env_dict(o.env);
  d["nav_est"] = output_dict(o.nav_est);
  d["att_cmd"] = output_dict(o.att_cmd);
  d["applied"] = output_dict(o.applied);
  d["nav_x_hat"] = o.nav_x_hat;
  d["nav_p_upper"] = o.nav_p_upper;
  return d;
}

py::dict truth_dict(const star::gnc::TruthState& t) {
  py::dict d;
  d["valid"] = t.valid;
  d["t_s"] = t.t_s;
  d["r_i_m"] = from_vec3(t.r_i_m);
  d["v_i_mps"] = from_vec3(t.v_i_mps);
  d["q_i2b"] = quat_arr(t.q_i2b);
  d["omega_b_radps"] = from_vec3(t.omega_b_radps);
  d["mass_kg"] = t.mass_kg;
  d["imu_bias_valid"] = t.imu_bias_valid;
  d["b_g_radps"] = from_vec3(t.b_g_radps);
  d["b_a_mps2"] = from_vec3(t.b_a_mps2);
  return d;
}

// FR-24 step(commands): fold a command dict into the run's held external
// command. Missing keys hold (the previous value is read back and rewritten
// unchanged, D-5 zero-order hold); an unknown key raises rather than being
// ignored, because a silently dropped command is indistinguishable from a
// vehicle that refused to manoeuvre.
void apply_commands(star::VehicleCycle& sim, const py::dict& commands) {
  star::gnc::GncOutput cmd = sim.external_command();
  bool any = false;
  bool explicit_valid = false;
  for (const auto& item : commands) {
    const std::string key = item.first.cast<std::string>();
    if (key == "torque_b_nm") {
      cmd.torque_b_nm = to_vec3(item.second.cast<std::array<double, 3>>());
    } else if (key == "omega_b_radps") {
      cmd.omega_b_radps = to_vec3(item.second.cast<std::array<double, 3>>());
    } else if (key == "q_i2b") {
      cmd.q_i2b = quat_in(item.second.cast<std::array<double, 4>>());
    } else if (key == "valid") {
      cmd.valid = item.second.cast<bool>();
      explicit_valid = true;
    } else {
      throw std::invalid_argument(
          "Sim.step: unknown command key '" + key +
          "'; accepted keys are 'torque_b_nm', 'omega_b_radps', 'q_i2b', "
          "'valid'");
    }
    any = true;
  }
  // Supplying a command means commanding: the entry becomes live unless the
  // caller explicitly asked for a hold.
  if (any && !explicit_valid) cmd.valid = true;
  if (any) sim.set_external_command(cmd);
}

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() =
      "star_reacher deterministic C++ core: two-body and composed-"
      "environment propagation, SRLOG writing, time/frame/ephemeris kernel, "
      "environment force models, and named-stream RNG.";

  // -- Phase 4 vehicle definition (D-2 plain data; run_vehicle only) --------
  // Each struct mirrors a level of the resolved vehicle/sequence dict; the
  // Python frontend fills them from the WS-C validated dicts. Bound before
  // RunConfig so its vehicle/sequence members resolve.
  py::class_<star::TankCfg>(m, "TankCfg")
      .def(py::init<>())
      .def_readwrite("radius_m", &star::TankCfg::radius_m)
      .def_readwrite("length_m", &star::TankCfg::length_m)
      .def_readwrite("position_m", &star::TankCfg::position_m)
      .def_readwrite("propellant_mass_kg", &star::TankCfg::propellant_mass_kg)
      .def_readwrite("density_kgpm3", &star::TankCfg::density_kgpm3);

  py::class_<star::EngineCfg>(m, "EngineCfg")
      .def(py::init<>())
      .def_readwrite("name", &star::EngineCfg::name)
      .def_readwrite("feeds_tank_index", &star::EngineCfg::feeds_tank_index)
      .def_readwrite("thrust_vac_N", &star::EngineCfg::thrust_vac_N)
      .def_readwrite("isp_vac_s", &star::EngineCfg::isp_vac_s)
      .def_readwrite("exit_area_m2", &star::EngineCfg::exit_area_m2)
      .def_readwrite("position_m", &star::EngineCfg::position_m)
      .def_readwrite("axis", &star::EngineCfg::axis)
      .def_readwrite("gimbal_max_deg", &star::EngineCfg::gimbal_max_deg)
      .def_readwrite("gimbal_rate_dps", &star::EngineCfg::gimbal_rate_dps)
      .def_readwrite("throttle_min", &star::EngineCfg::throttle_min)
      .def_readwrite("throttle_max", &star::EngineCfg::throttle_max)
      .def_readwrite("spool_time_s", &star::EngineCfg::spool_time_s)
      .def_readwrite("ignitions", &star::EngineCfg::ignitions);

  py::class_<star::RcsCfg>(m, "RcsCfg")
      .def(py::init<>())
      .def_readwrite("name", &star::RcsCfg::name)
      .def_readwrite("thrust_N", &star::RcsCfg::thrust_N)
      .def_readwrite("min_impulse_bit_Ns", &star::RcsCfg::min_impulse_bit_Ns)
      .def_readwrite("thruster_positions_m",
                     &star::RcsCfg::thruster_positions_m)
      .def_readwrite("thruster_directions", &star::RcsCfg::thruster_directions);

  py::class_<star::WheelCfg>(m, "WheelCfg")
      .def(py::init<>())
      .def_readwrite("name", &star::WheelCfg::name)
      .def_readwrite("axis", &star::WheelCfg::axis)
      .def_readwrite("max_torque_Nm", &star::WheelCfg::max_torque_Nm)
      .def_readwrite("max_momentum_Nms", &star::WheelCfg::max_momentum_Nms);

  py::class_<star::JettisonCfg>(m, "JettisonCfg")
      .def(py::init<>())
      .def_readwrite("name", &star::JettisonCfg::name)
      .def_readwrite("mass_kg", &star::JettisonCfg::mass_kg)
      .def_readwrite("cg_m", &star::JettisonCfg::cg_m)
      .def_readwrite("inertia_kgm2", &star::JettisonCfg::inertia_kgm2);

  py::class_<star::StageCfg>(m, "StageCfg")
      .def(py::init<>())
      .def_readwrite("name", &star::StageCfg::name)
      .def_readwrite("dry_mass_kg", &star::StageCfg::dry_mass_kg)
      .def_readwrite("dry_cg_m", &star::StageCfg::dry_cg_m)
      .def_readwrite("dry_inertia_kgm2", &star::StageCfg::dry_inertia_kgm2)
      .def_readwrite("tanks", &star::StageCfg::tanks)
      .def_readwrite("engines", &star::StageCfg::engines)
      .def_readwrite("rcs", &star::StageCfg::rcs)
      .def_readwrite("wheels", &star::StageCfg::wheels)
      .def_readwrite("jettison", &star::StageCfg::jettison);

  py::class_<star::AeroCfg>(m, "AeroCfg")
      .def(py::init<>())
      .def_readwrite("config", &star::AeroCfg::config)
      .def_readwrite("ref_area_m2", &star::AeroCfg::ref_area_m2)
      .def_readwrite("ref_diameter_m", &star::AeroCfg::ref_diameter_m)
      .def_readwrite("cmq_per_rad", &star::AeroCfg::cmq_per_rad)
      .def_readwrite("mach", &star::AeroCfg::mach)
      .def_readwrite("ca", &star::AeroCfg::ca)
      .def_readwrite("cnalpha_per_rad", &star::AeroCfg::cnalpha_per_rad)
      .def_readwrite("xcp_m", &star::AeroCfg::xcp_m);

  py::class_<star::VehicleConfig>(m, "VehicleConfig")
      .def(py::init<>())
      .def_readwrite("stages", &star::VehicleConfig::stages)
      .def_readwrite("aero", &star::VehicleConfig::aero);

  // -- Phase 6 GNC configuration (D-2 plain data; run_vehicle only) ---------
  // Component parameters ride as two flat maps so a new built-in (or a
  // future Python plugin) adds fields with zero binding changes.
  py::class_<star::gnc::GncComponentCfg>(m, "GncComponentCfg")
      .def(py::init<>())
      .def_readwrite("component", &star::gnc::GncComponentCfg::component)
      .def_readwrite("scalars", &star::gnc::GncComponentCfg::scalars)
      .def_readwrite("vectors", &star::gnc::GncComponentCfg::vectors);

  py::class_<star::gnc::GncSensorCfg>(m, "GncSensorCfg")
      .def(py::init<>())
      .def_readwrite("kind", &star::gnc::GncSensorCfg::kind)
      .def_readwrite("sample_rate_hz",
                     &star::gnc::GncSensorCfg::sample_rate_hz)
      .def_readwrite("scalars", &star::gnc::GncSensorCfg::scalars)
      .def_readwrite("vectors", &star::gnc::GncSensorCfg::vectors);

  py::class_<star::gnc::GncConfig>(m, "GncConfig")
      .def(py::init<>())
      .def_readwrite("enabled", &star::gnc::GncConfig::enabled)
      .def_readwrite("control_rate_hz", &star::gnc::GncConfig::control_rate_hz)
      .def_readwrite("latency_cycles", &star::gnc::GncConfig::latency_cycles)
      .def_readwrite("nav", &star::gnc::GncConfig::nav)
      .def_readwrite("guidance", &star::gnc::GncConfig::guidance)
      .def_readwrite("control", &star::gnc::GncConfig::control)
      .def_readwrite("sensors", &star::gnc::GncConfig::sensors);

  py::class_<star::SequenceEntry>(m, "SequenceEntry")
      .def(py::init<>())
      .def_readwrite("name", &star::SequenceEntry::name)
      .def_readwrite("trigger", &star::SequenceEntry::trigger)
      .def_readwrite("t_s", &star::SequenceEntry::t_s)
      .def_readwrite("event", &star::SequenceEntry::event)
      .def_readwrite("offset_s", &star::SequenceEntry::offset_s)
      .def_readwrite("condition", &star::SequenceEntry::condition)
      .def_readwrite("altitude_m", &star::SequenceEntry::altitude_m)
      .def_readwrite("perigee_alt_m", &star::SequenceEntry::perigee_alt_m)
      .def_readwrite("body", &star::SequenceEntry::body)
      .def_readwrite("action", &star::SequenceEntry::action)
      .def_readwrite("stage", &star::SequenceEntry::stage)
      .def_readwrite("engine", &star::SequenceEntry::engine)
      .def_readwrite("item", &star::SequenceEntry::item)
      .def_readwrite("azimuth_deg", &star::SequenceEntry::azimuth_deg)
      .def_readwrite("pitch_t_s", &star::SequenceEntry::pitch_t_s)
      .def_readwrite("pitch_deg", &star::SequenceEntry::pitch_deg)
      .def_readwrite("frame", &star::SequenceEntry::frame)
      .def_readwrite("omega_dps", &star::SequenceEntry::omega_dps);

  py::class_<star::RunConfig>(m, "RunConfig",
                              "Mission run configuration. Populate every "
                              "field from the validated, canonicalized "
                              "mission file before calling run().")
      .def(py::init<>())
      .def_readwrite("epoch_utc", &star::RunConfig::epoch_utc)
      .def_readwrite("duration_s", &star::RunConfig::duration_s)
      .def_readwrite("dt_s", &star::RunConfig::dt_s)
      .def_readwrite("integrator", &star::RunConfig::integrator)
      .def_readwrite("central_body", &star::RunConfig::central_body)
      .def_readwrite("r0_m", &star::RunConfig::r0_m)
      .def_readwrite("v0_mps", &star::RunConfig::v0_mps)
      .def_readwrite("mass_kg", &star::RunConfig::mass_kg)
      .def_readwrite("master_seed", &star::RunConfig::master_seed)
      .def_readwrite("truth_rate_hz", &star::RunConfig::truth_rate_hz)
      .def_readwrite("config_sha256", &star::RunConfig::config_sha256)
      .def_readwrite("oracle", &star::RunConfig::oracle)
      // Phase 3 extension (consumed by run_env only; run() ignores it).
      .def_readwrite("epoch_tai_day", &star::RunConfig::epoch_tai_day)
      .def_readwrite("epoch_tai_sec", &star::RunConfig::epoch_tai_sec)
      .def_readwrite("rtol", &star::RunConfig::rtol)
      .def_readwrite("atol_pos_m", &star::RunConfig::atol_pos_m)
      .def_readwrite("atol_vel_mps", &star::RunConfig::atol_vel_mps)
      .def_readwrite("h_init_s", &star::RunConfig::h_init_s)
      .def_readwrite("h_max_s", &star::RunConfig::h_max_s)
      .def_readwrite("gravity_model", &star::RunConfig::gravity_model)
      .def_readwrite("gravity_field_path", &star::RunConfig::gravity_field_path)
      .def_readwrite("gravity_degree", &star::RunConfig::gravity_degree)
      .def_readwrite("gravity_order", &star::RunConfig::gravity_order)
      .def_readwrite("third_bodies", &star::RunConfig::third_bodies)
      .def_readwrite("srp_enabled", &star::RunConfig::srp_enabled)
      .def_readwrite("cr_a_over_m_m2pkg", &star::RunConfig::cr_a_over_m_m2pkg)
      .def_readwrite("srp_occulters", &star::RunConfig::srp_occulters)
      .def_readwrite("drag_enabled", &star::RunConfig::drag_enabled)
      .def_readwrite("atmosphere", &star::RunConfig::atmosphere)
      .def_readwrite("cd_a_over_m_m2pkg", &star::RunConfig::cd_a_over_m_m2pkg)
      .def_readwrite("hp_exponent_n", &star::RunConfig::hp_exponent_n)
      .def_readwrite("ephemeris_path", &star::RunConfig::ephemeris_path)
      // Phase 4 extension (consumed by run_vehicle only; run()/run_env()
      // ignore it).
      .def_readwrite("vehicle", &star::RunConfig::vehicle)
      .def_readwrite("sequence", &star::RunConfig::sequence)
      .def_readwrite("initial_form", &star::RunConfig::initial_form)
      .def_readwrite("launch_lat_deg", &star::RunConfig::launch_lat_deg)
      .def_readwrite("launch_lon_deg", &star::RunConfig::launch_lon_deg)
      .def_readwrite("launch_alt_m", &star::RunConfig::launch_alt_m)
      .def_readwrite("forces_rate_hz", &star::RunConfig::forces_rate_hz)
      .def_readwrite("mass_rate_hz", &star::RunConfig::mass_rate_hz)
      .def_readwrite("env_rate_hz", &star::RunConfig::env_rate_hz)
      // Phase 6 extension (contract section: run.hpp).
      .def_readwrite("gnc", &star::RunConfig::gnc);

  m.def("run", &run_and_summarize, py::arg("config"), py::arg("out_path"),
        "Propagate the configured two-body case and write an SRLOG v1.0 file "
        "to out_path. Returns a summary dict (steps, final_r_m, final_v_mps, "
        "truth_records, event_records, records_written). This Phase 1 path "
        "is byte-frozen; it ignores the Phase 3 RunConfig extension.");

  m.def("run_env", &run_env_and_summarize, py::arg("config"),
        py::arg("out_path"),
        "Propagate the composed-environment case (Phase 3: gravity tiers, "
        "third bodies, SRP, drag; rk4 or adaptive rkf78) and write an SRLOG "
        "v1.0 file to out_path. Returns the same summary dict as run().");

  m.def("run_vehicle", &run_vehicle_and_summarize, py::arg("config"),
        py::arg("out_path"),
        "Propagate the full 6DOF vehicle case (Phase 4: staged vehicle under "
        "the composed environment plus its own thrust, aero, and attitude, "
        "driven by the open-loop [[sequence]]) and write an SRLOG v1.1 file "
        "(truth with real q/omega, forces, mass, env groups) to out_path. "
        "Returns the same summary dict as run().");

  m.def("gm", &star::gm, py::arg("body"),
        "Gravitational parameter GM [m^3/s^2] of a named central body "
        "(\"earth\": IERS Conventions 2010, TN No. 36; \"moon\", \"mars\", "
        "\"sun\": DE440 header values, Park et al. 2021). The single home of "
        "the constant - use this for Keplerian conversions.");

  // -- Phase 3 model surface (verification and test access) ----------------

  m.def("gravity_accel", &gravity_accel, py::arg("srgrav_path"),
        py::arg("tier"), py::arg("degree"), py::arg("order"),
        py::arg("r_bf_m"),
        "Spherical-harmonic gravitational acceleration [m/s^2] at body-fixed "
        "position r_bf_m from an SRGRAV v1 field file (FR-5). tier is "
        "\"pointmass\", \"j2\", or \"full\"; degree/order of -1 mean the "
        "stored band. Loads the field per call (verification surface, not "
        "the propagation loop).");

  m.def("thirdbody_accel", &thirdbody_accel, py::arg("gm_third_m3ps2"),
        py::arg("r_sc_m"), py::arg("r_third_m"),
        "Battin f(q) differential third-body acceleration [m/s^2] (FR-6); "
        "both positions relative to the central body in a common frame.");

  m.def("shadow_fraction", &shadow_fraction, py::arg("r_sc_m"),
        py::arg("r_sun_m"), py::arg("radius_sun_m"), py::arg("r_occ_m"),
        py::arg("radius_occ_m"),
        "Conical-shadow illumination fraction nu in [0, 1] (FR-7): exactly "
        "1 in full sunlight, exactly 0 in total umbra, smooth through the "
        "penumbra, annular case handled.");

  m.def("srp_accel", &srp_accel, py::arg("cr_a_over_m_m2pkg"), py::arg("nu"),
        py::arg("r_sc_m"), py::arg("r_sun_m"),
        "Cannonball SRP acceleration [m/s^2] (FR-7), pushing away from the "
        "Sun, scaled by the illumination fraction nu.");

  m.def("ussa76_density", &star::models::ussa76_density, py::arg("z_m"),
        "U.S. Standard Atmosphere 1976 density [kg/m^3] for geometric "
        "altitude z_m in [-5 km, 1000 km] (FR-8): analytic below 86 km, "
        "committed-node log-linear interpolation above.");

  m.def("hp_density", &star::models::hp_density, py::arg("alt_m"),
        py::arg("cos_psi"), py::arg("n"),
        "Harris-Priester density [kg/m^3] at geodetic altitude alt_m with "
        "cos_psi the cosine of the diurnal-bulge angle and n the bulge "
        "exponent in [2, 6] (FR-8; Montenbruck & Gill Sect. 3.5.2, "
        "Orekit-compatible).");

  m.def("mars_density", &star::models::mars_density, py::arg("z_m"),
        "Mars piecewise-exponential density [kg/m^3] (FR-8, PRD A-3: "
        "provenance provisional, confidence low).");

  m.def("drag_accel", &drag_accel, py::arg("rho_kgpm3"),
        py::arg("cd_a_over_m_m2pkg"), py::arg("v_rel_mps"),
        "Cannonball drag acceleration [m/s^2] (FR-9) for the given density "
        "and air-relative velocity v_rel = v - omega x r (FR-8).");

  m.def("geodetic_altitude", &geodetic_altitude, py::arg("r_ecef_m"),
        py::arg("a_m"), py::arg("inv_f"),
        "Geodetic altitude [m] of a body-fixed Cartesian position above the "
        "(a, 1/f) ellipsoid (Bowring's closed form with one refinement).");

  // -- FR-25 GNC component interface (Python plugin path) ------------------
  // The structs below are the component's whole view of the world. They are
  // bound before IGncComponent so its update() signature resolves.

  py::class_<star::gnc::ImuSample>(
      m, "ImuSample",
      "Accumulated IMU increments over the sample interval ending at t_s "
      "(FR-23). valid is false until the first sample instant has passed.")
      .def(py::init<>())
      .def_readwrite("valid", &star::gnc::ImuSample::valid)
      .def_readwrite("t_s", &star::gnc::ImuSample::t_s)
      .def_readwrite("dt_s", &star::gnc::ImuSample::dt_s)
      .STAR_V3_PROP(star::gnc::ImuSample, "dtheta_b_rad", dtheta_b_rad)
      .STAR_V3_PROP(star::gnc::ImuSample, "dv_b_mps", dv_b_mps);

  py::class_<star::gnc::NavFixSample>(
      m, "NavFixSample",
      "External navigation fix (generalized GNSS, FR-23). fresh is true only "
      "on the cycle the sensor was sampled: folding a held measurement in "
      "twice makes a filter overconfident in a way that is invisible in the "
      "state error but immediate in NEES.")
      .def(py::init<>())
      .def_readwrite("valid", &star::gnc::NavFixSample::valid)
      .def_readwrite("fresh", &star::gnc::NavFixSample::fresh)
      .def_readwrite("sensor_id", &star::gnc::NavFixSample::sensor_id)
      .STAR_V3_PROP(star::gnc::NavFixSample, "r_i_m", r_i_m)
      .STAR_V3_PROP(star::gnc::NavFixSample, "v_i_mps", v_i_mps);

  py::class_<star::gnc::StarTrackerSample>(
      m, "StarTrackerSample",
      "Star-tracker attitude measurement (FR-23), scalar-first q_i2b "
      "relative to the APPARENT inertial frame - a consumer that predicts it "
      "must apply the same velocity-aberration factor. valid echoes the "
      "sensor's exclusion/slew gating.")
      .def(py::init<>())
      .def_readwrite("valid", &star::gnc::StarTrackerSample::valid)
      .def_readwrite("fresh", &star::gnc::StarTrackerSample::fresh)
      .def_readwrite("sensor_id", &star::gnc::StarTrackerSample::sensor_id)
      .STAR_QUAT_PROP(star::gnc::StarTrackerSample, "q_i2b", q_i2b);

  py::class_<star::gnc::AltimeterSample>(
      m, "AltimeterSample",
      "Altimeter measurement (FR-23): geodetic height over the central "
      "body's reference ellipsoid. valid echoes the sensor's band gate.")
      .def(py::init<>())
      .def_readwrite("valid", &star::gnc::AltimeterSample::valid)
      .def_readwrite("fresh", &star::gnc::AltimeterSample::fresh)
      .def_readwrite("sensor_id", &star::gnc::AltimeterSample::sensor_id)
      .def_readwrite("h_m", &star::gnc::AltimeterSample::h_m);

  py::class_<star::gnc::NavEnvironment>(
      m, "NavEnvironment",
      "Ephemeris and frame context a real onboard navigator computes from "
      "time and its own ephemeris - never from truth. Supplied every cycle "
      "so an estimator can predict frame-dependent measurements (star-"
      "tracker aberration, altimeter body-fixed conversion) without reaching "
      "across the FR-25 privileged boundary. c_gcrf_to_bodyfixed is a "
      "row-major 9-element list.")
      .def(py::init<>())
      .def_readwrite("ephemeris_valid",
                     &star::gnc::NavEnvironment::ephemeris_valid)
      .STAR_V3_PROP(star::gnc::NavEnvironment, "v_central_ssb_mps",
                    v_central_ssb_mps)
      .def_readwrite("bodyfixed_valid",
                     &star::gnc::NavEnvironment::bodyfixed_valid)
      .STAR_M3_PROP(star::gnc::NavEnvironment, "c_gcrf_to_bodyfixed",
                    c_gcrf_to_bodyfixed);

  py::class_<star::gnc::TruthState>(
      m, "TruthState",
      "Truth kinematics snapshot: privileged data (FR-24). It reaches a GNC "
      "component on exactly one path, GncInput.oracle, and only when the "
      "scenario sets oracle = true; the loop's own use of it to compute "
      "nav.err never crosses the plugin boundary, because the loop does that "
      "arithmetic itself from the component's declared error_layout(). "
      "Sim.truth() is the separate privileged accessor, available to a "
      "stepping driver rather than to a component. b_g_radps/b_a_mps2 are "
      "the true in-run IMU biases, valid only when the run configures an "
      "IMU, so a bias-carrying estimator reports a complete error instead of "
      "logging zeros that read as 'no error'.")
      .def(py::init<>())
      .def_readwrite("valid", &star::gnc::TruthState::valid)
      .def_readwrite("t_s", &star::gnc::TruthState::t_s)
      .STAR_V3_PROP(star::gnc::TruthState, "r_i_m", r_i_m)
      .STAR_V3_PROP(star::gnc::TruthState, "v_i_mps", v_i_mps)
      .STAR_QUAT_PROP(star::gnc::TruthState, "q_i2b", q_i2b)
      .STAR_V3_PROP(star::gnc::TruthState, "omega_b_radps", omega_b_radps)
      .def_readwrite("mass_kg", &star::gnc::TruthState::mass_kg)
      .def_readwrite("imu_bias_valid", &star::gnc::TruthState::imu_bias_valid)
      .STAR_V3_PROP(star::gnc::TruthState, "b_g_radps", b_g_radps)
      .STAR_V3_PROP(star::gnc::TruthState, "b_a_mps2", b_a_mps2);

  py::class_<star::gnc::NavSensorModel>(
      m, "NavSensorModel",
      "The run's configured sensor-suite parameters, handed to components at "
      "init so an estimator's stochastic model IS the configured truth model "
      "rather than a hand-copied duplicate in the mission file that can "
      "silently drift out of sync with the sensors it describes.")
      .def(py::init<>())
      .def_readwrite("imu_present", &star::gnc::NavSensorModel::imu_present)
      .def_readwrite("imu_id", &star::gnc::NavSensorModel::imu_id)
      .def_readwrite("gyro_arw", &star::gnc::NavSensorModel::gyro_arw)
      .def_readwrite("accel_vrw", &star::gnc::NavSensorModel::accel_vrw)
      .def_readwrite("gyro_gm_sigma", &star::gnc::NavSensorModel::gyro_gm_sigma)
      .def_readwrite("gyro_tau_s", &star::gnc::NavSensorModel::gyro_tau_s)
      .def_readwrite("accel_gm_sigma",
                     &star::gnc::NavSensorModel::accel_gm_sigma)
      .def_readwrite("accel_tau_s", &star::gnc::NavSensorModel::accel_tau_s)
      .def_readwrite("navfix_present",
                     &star::gnc::NavSensorModel::navfix_present)
      .def_readwrite("navfix_id", &star::gnc::NavSensorModel::navfix_id)
      .STAR_V3_PROP(star::gnc::NavSensorModel, "navfix_sigma_r_m",
                    navfix_sigma_r_m)
      .STAR_V3_PROP(star::gnc::NavSensorModel, "navfix_sigma_v_mps",
                    navfix_sigma_v_mps)
      .def_readwrite("startracker_present",
                     &star::gnc::NavSensorModel::startracker_present)
      .def_readwrite("startracker_id",
                     &star::gnc::NavSensorModel::startracker_id)
      .STAR_V3_PROP(star::gnc::NavSensorModel, "startracker_sigma_rad",
                    startracker_sigma_rad)
      .STAR_V3_PROP(star::gnc::NavSensorModel, "startracker_boresight_b",
                    startracker_boresight_b)
      .def_readwrite("altimeter_present",
                     &star::gnc::NavSensorModel::altimeter_present)
      .def_readwrite("altimeter_id", &star::gnc::NavSensorModel::altimeter_id)
      .def_readwrite("altimeter_sigma_noise_m",
                     &star::gnc::NavSensorModel::altimeter_sigma_noise_m)
      .def_readwrite("altimeter_sigma_bias_m",
                     &star::gnc::NavSensorModel::altimeter_sigma_bias_m);

  py::class_<star::gnc::GncOutput>(
      m, "GncOutput",
      "One component's output. The same struct serves all three chain roles "
      "with role-dependent meaning: nav -> q_i2b/omega are the ESTIMATE; "
      "guidance -> they are the COMMAND; control -> torque_b_nm is the "
      "commanded body torque, already saturated. valid == false means HOLD: "
      "the loop keeps applying the previous applied command and logs the "
      "held values with valid = 0.")
      .def(py::init<>())
      .def_readwrite("valid", &star::gnc::GncOutput::valid)
      .STAR_QUAT_PROP(star::gnc::GncOutput, "q_i2b", q_i2b)
      .STAR_V3_PROP(star::gnc::GncOutput, "omega_b_radps", omega_b_radps)
      .STAR_V3_PROP(star::gnc::GncOutput, "torque_b_nm", torque_b_nm);

  py::class_<star::gnc::GncInput>(
      m, "GncInput",
      "Everything a component may read on one control cycle. The loop fills "
      "the chain slots progressively in the fixed order nav -> guidance -> "
      "control: guidance sees nav_est, control sees nav_est and att_cmd. "
      "prev_applied is the command actually applied on the previous cycle "
      "(post-latency), so a component can rate-limit against what the "
      "vehicle really did. oracle is populated if and only if the scenario "
      "set oracle = true; treat oracle.valid == false as 'truth does not "
      "exist'.")
      .def(py::init<>())
      .def_readwrite("cycle", &star::gnc::GncInput::cycle)
      .def_readwrite("t_s", &star::gnc::GncInput::t_s)
      .def_readwrite("dt_s", &star::gnc::GncInput::dt_s)
      .def_readwrite("imu", &star::gnc::GncInput::imu)
      .def_readwrite("imu_fresh", &star::gnc::GncInput::imu_fresh)
      .def_readwrite("nav_est", &star::gnc::GncInput::nav_est)
      .def_readwrite("att_cmd", &star::gnc::GncInput::att_cmd)
      .def_readwrite("prev_applied", &star::gnc::GncInput::prev_applied)
      .def_readwrite("oracle", &star::gnc::GncInput::oracle)
      .def_readwrite("navfix", &star::gnc::GncInput::navfix)
      .def_readwrite("startracker", &star::gnc::GncInput::startracker)
      .def_readwrite("altimeter", &star::gnc::GncInput::altimeter)
      .def_readwrite("env", &star::gnc::GncInput::env);

  py::class_<star::gnc::GncInitContext>(
      m, "GncInitContext",
      "One-time initialization context captured at GNC activation. q0/omega0 "
      "are the attitude state at activation; the pad ENU basis is valid only "
      "for geodetic launch missions. mu_m3ps2 and the ellipsoid pair are the "
      "central-body constants an estimator needs for its own dynamics and "
      "measurement models; sensors is the configured suite.")
      .def(py::init<>())
      .def_readwrite("t0_s", &star::gnc::GncInitContext::t0_s)
      .STAR_QUAT_PROP(star::gnc::GncInitContext, "q0_i2b", q0_i2b)
      .STAR_V3_PROP(star::gnc::GncInitContext, "omega0_b_radps",
                    omega0_b_radps)
      .def_readwrite("pad_basis_valid",
                     &star::gnc::GncInitContext::pad_basis_valid)
      .STAR_V3_PROP(star::gnc::GncInitContext, "up_i", up_i)
      .STAR_V3_PROP(star::gnc::GncInitContext, "east_i", east_i)
      .STAR_V3_PROP(star::gnc::GncInitContext, "north_i", north_i)
      .def_readwrite("control_rate_hz",
                     &star::gnc::GncInitContext::control_rate_hz)
      .def_readwrite("dt_s", &star::gnc::GncInitContext::dt_s)
      .def_readwrite("mu_m3ps2", &star::gnc::GncInitContext::mu_m3ps2)
      .def_readwrite("ellipsoid_a_m", &star::gnc::GncInitContext::ellipsoid_a_m)
      .def_readwrite("ellipsoid_inv_f",
                     &star::gnc::GncInitContext::ellipsoid_inv_f)
      .def_readwrite("sensors", &star::gnc::GncInitContext::sensors);

  py::class_<star::gnc::InnovationSample>(
      m, "InnovationSample",
      "One applied aiding update, reported for nav.innov logging: the "
      "innovation vector y (size m) and the innovation covariance S packed "
      "row-major upper triangle (size m(m+1)/2). sensor_id indexes the run's "
      "configured sensor list.")
      .def(py::init<>())
      .def_readwrite("sensor_id", &star::gnc::InnovationSample::sensor_id)
      .def_readwrite("y", &star::gnc::InnovationSample::y)
      .def_readwrite("s_upper", &star::gnc::InnovationSample::s_upper);

  py::enum_<star::gnc::ErrorQuantity>(
      m, "ErrorQuantity",
      "Which truth quantity a block of an estimator's state vector is "
      "compared against when the loop computes nav.err. Each names a "
      "quantity the simulator knows truly; a state with no truth counterpart "
      "cannot be declared.")
      .value("POSITION", star::gnc::ErrorQuantity::kPosition)
      .value("VELOCITY", star::gnc::ErrorQuantity::kVelocity)
      .value("ATTITUDE", star::gnc::ErrorQuantity::kAttitude)
      .value("ANGULAR_RATE", star::gnc::ErrorQuantity::kAngularRate)
      .value("GYRO_BIAS", star::gnc::ErrorQuantity::kGyroBias)
      .value("ACCEL_BIAS", star::gnc::ErrorQuantity::kAccelBias)
      .value("MASS", star::gnc::ErrorQuantity::kMass);

  py::enum_<star::gnc::ErrorForm>(
      m, "ErrorForm",
      "How the error in a declared quantity is formed. DIFFERENCE is "
      "elementwise truth minus estimate and is the only form for every "
      "quantity except ATTITUDE. An attitude error is a rotation difference: "
      "QUAT_ERROR_LOCAL is dq = conj(q_est) (x) q_true (4 slots, resolved in "
      "the estimated body frame), QUAT_ERROR_GLOBAL is dq = q_true (x) "
      "conj(q_est) (4 slots, resolved in the inertial frame), and the "
      "ROTATION_VECTOR forms are the small-angle reduction 2 sgn(dq_w) dq_v "
      "of the respective dq (3 slots). Every quaternion form is sign "
      "canonicalized to the +w hemisphere. Quaternions are scalar-first "
      "(D-7).")
      .value("DIFFERENCE", star::gnc::ErrorForm::kDifference)
      .value("QUAT_ERROR_LOCAL", star::gnc::ErrorForm::kQuatErrorLocal)
      .value("QUAT_ERROR_GLOBAL", star::gnc::ErrorForm::kQuatErrorGlobal)
      .value("ROTATION_VECTOR_LOCAL",
             star::gnc::ErrorForm::kRotationVectorLocal)
      .value("ROTATION_VECTOR_GLOBAL",
             star::gnc::ErrorForm::kRotationVectorGlobal);

  py::class_<star::gnc::ErrorBlock>(
      m, "ErrorBlock",
      "One contiguous run of an estimator's state vector, and how its error "
      "is formed. offset is the index of the block's first slot, shared by "
      "the state vector and the error vector. The blocks a component returns "
      "from error_layout() must tile [0, state_dim()) exactly.")
      .def(py::init<>())
      .def(py::init([](star::gnc::ErrorQuantity q, star::gnc::ErrorForm f,
                       int offset) {
             star::gnc::ErrorBlock b;
             b.quantity = q;
             b.form = f;
             b.offset = offset;
             return b;
           }),
           py::arg("quantity"), py::arg("form"), py::arg("offset"))
      .def_readwrite("quantity", &star::gnc::ErrorBlock::quantity)
      .def_readwrite("form", &star::gnc::ErrorBlock::form)
      .def_readwrite("offset", &star::gnc::ErrorBlock::offset);

  py::class_<star::gnc::IGncComponent, PyGncComponent>(
      m, "IGncComponent",
      "The FR-25 GNC plugin base. Subclass it in Python and override "
      "init(ctx) and update(input) -> GncOutput; register the subclass with "
      "register_python_component(name, cls) and select it from a mission "
      "file by that name.\n\n"
      "An estimator additionally overrides state_dim() and state(), and may "
      "override cov_dim()/covariance_upper(), innov_max_dim()/innovations(), "
      "and error_layout(). state() returns state_dim() floats and "
      "covariance_upper() returns cov_dim()*(cov_dim()+1)/2 floats (packed "
      "row-major upper triangle); a wrong length raises rather than being "
      "silently truncated.\n\n"
      "error_layout() returns a list of ErrorBlock describing what each slot "
      "of the state vector means. The loop uses it to compute nav.err "
      "itself: no method of this class is ever handed the true state, which "
      "is what makes FR-24's privileged-truth boundary structural rather "
      "than a promise. An estimator that declares no layout gets no nav.err "
      "channel.\n\n"
      "DETERMINISM (D-10): update() runs inside the deterministic time loop. "
      "It must not read the clock, perform I/O, or draw from an unseeded "
      "RNG. See star_reacher.sim for the full contract - the core cannot "
      "enforce it on arbitrary Python.")
      .def(py::init<>())
      .def("init", &star::gnc::IGncComponent::init, py::arg("ctx"))
      .def("update", &star::gnc::IGncComponent::update, py::arg("input"))
      .def("state_dim", &star::gnc::IGncComponent::state_dim)
      .def("cov_dim", &star::gnc::IGncComponent::cov_dim)
      .def("innov_max_dim", &star::gnc::IGncComponent::innov_max_dim)
      .def("error_layout", &star::gnc::IGncComponent::error_layout);

  m.def("register_python_component", &register_python_component,
        py::arg("name"), py::arg("factory"),
        "Register a Python GNC component factory under a config-file name "
        "(FR-25). factory(cfg) must return an IGncComponent subclass "
        "instance; a class object is such a callable. Duplicate names raise, "
        "because two components silently shadowing each other would be a "
        "determinism hazard.");

  // -- FR-24 stepping API ---------------------------------------------------

  py::class_<star::VehicleCycle>(
      m, "Sim",
      "The FR-24 stepping API over the 6DOF vehicle loop. Sim(config, "
      "out_path) opens the SRLOG and writes its header; each step() advances "
      "exactly one GNC control period (D-5) and returns the observation.\n\n"
      "Stepped and batch execution of one scenario produce byte-identical "
      "logs because they are the same code: run_vehicle() is literally a "
      "loop over this same cycle core (Phase 6 exit criterion 4).")
      .def(py::init<const star::RunConfig&, const std::string&>(),
           py::arg("config"), py::arg("out_path"))
      .def(
          "step",
          [](star::VehicleCycle& s, const py::object& commands) {
            if (!commands.is_none()) {
              apply_commands(s, commands.cast<py::dict>());
            }
            s.step();
            return observation_dict(s.observation());
          },
          py::arg("commands") = py::none(),
          "Advance exactly one control period and return the observation "
          "dict for the cycle just processed. commands is an optional dict "
          "with keys 'torque_b_nm', 'omega_b_radps', 'q_i2b' (scalar-first), "
          "and 'valid'; supplied keys replace the held command and missing "
          "keys hold it (D-5 zero-order hold), while an unknown key raises. "
          "Commanding requires the mission to configure an 'external' "
          "gnc.guidance or gnc.control component. Stepping after the run has "
          "ended raises: the log is complete and closed.")
      .def(
          "observe",
          [](const star::VehicleCycle& s) {
            return observation_dict(s.observation());
          },
          "The non-privileged observation of the most recently processed "
          "cycle (before the first step(), of the initial state). Reading is "
          "PURE - it runs no component, draws no random number, consumes no "
          "sensor sample, and returns fresh copies rather than views - so "
          "two calls without an intervening step() return equal dicts (exit "
          "criterion 4).")
      .def(
          "truth",
          [](const star::VehicleCycle& s) { return truth_dict(s.truth()); },
          "PRIVILEGED true state at the instant observe() describes. Never "
          "visible to a GNC component through this path: a component sees "
          "truth only via GncInput.oracle, and only when the scenario set "
          "oracle = true.")
      .def("time", &star::VehicleCycle::time_s,
           "Current cycle time [s] - the time of the next cycle to be "
           "processed, which after a step() is one period past the "
           "observation's t_s.")
      .def("cycle", &star::VehicleCycle::cycle,
           "Current cycle index (0-based), the next cycle to be processed.")
      .def("done", &star::VehicleCycle::done,
           "True once the run has ended (terminal event or final cycle); the "
           "log is then complete and closed and step() must not be called "
           "again.")
      .def(
          "summary",
          [](const star::VehicleCycle& s) { return summary_dict(s.summary()); },
          "Run summary dict (steps, final state, record tallies); valid once "
          "done().")
      .def("has_external_command", &star::VehicleCycle::has_external_command,
           "True when the mission configures an 'external' guidance or "
           "control component, i.e. when step(commands) has an addressee.");

  m.def("gnc_component_names", &star::gnc::component_names,
        "Registered GNC component names (FR-25 registry), sorted. Exposed "
        "so the test suite can assert the mission validator's core-less "
        "component vocabulary (mission.py) never drifts from the core "
        "registry, and so tooling can enumerate the built-ins.");

  m.def("core_version", &star::core_version,
        "Semantic version of the compiled core, e.g. \"0.1.0\".");

  m.def("git_hash", &star::git_hash,
        "Git commit hash embedded at build configure time, or \"unknown\".");

  m.def("rng_stream_u64", &rng_stream_u64, py::arg("master_seed"),
        py::arg("stream_name"), py::arg("n"),
        "First n raw u64 draws of the named PCG64 stream derived from "
        "master_seed (D-9).");

  m.def("rng_stream_normal", &rng_stream_normal, py::arg("master_seed"),
        py::arg("stream_name"), py::arg("n"),
        "First n standard-normal Box-Muller deviates of the named stream "
        "(D-9; see star/rng.hpp for the exact draw-consumption pattern).");

  m.def("utc_to_tai", &utc_to_tai, py::arg("year"), py::arg("month"),
        py::arg("day"), py::arg("hour"), py::arg("minute"), py::arg("second"),
        "Numeric UTC calendar fields -> two-part TAI epoch (day, sec): whole "
        "TAI days since 2000-01-01T00:00:00.0 TAI and TAI seconds of day in "
        "[0, 86400) (D-6). second in [60, 61) is accepted only inside an "
        "inserted leap second. Raises ValueError outside the 1972-onward "
        "table domain or for invalid fields.");

  m.def("tai_to_utc", &tai_to_utc, py::arg("day"), py::arg("sec"),
        "Two-part TAI epoch -> UTC calendar fields (year, month, day, hour, "
        "minute, second); instants inside an inserted leap second come back "
        "with second in [60, 61).");

  m.def("tai_minus_utc", &star::time::tai_minus_utc_s, py::arg("year"),
        py::arg("month"), py::arg("day"),
        "TAI - UTC in whole seconds for a UTC calendar date (bundled IERS "
        "Bulletin C leap-second table; see leap_table_info()). Dates past "
        "the table expiry return the last tabulated value.");

  m.def("tai_to_jd", &tai_to_jd, py::arg("day"), py::arg("sec"),
        "Two-part TAI Julian Date (jd1 half-integer day, jd2 fraction of "
        "day) for a two-part TAI epoch.");

  m.def("tt_jd", &tt_to_jd, py::arg("day"), py::arg("sec"),
        "Two-part TT Julian Date for a two-part TAI epoch "
        "(TT = TAI + 32.184 s exactly).");

  m.def("tdb_jd", &tdb_to_jd, py::arg("day"), py::arg("sec"),
        "Two-part TDB Julian Date for a two-part TAI epoch (TT plus the "
        "truncated Fairhead-Bretagnon series).");

  m.def("tt_julian_centuries", [](std::int64_t day, double sec) {
          return star::time::tt_julian_centuries({day, sec});
        },
        py::arg("day"), py::arg("sec"),
        "TT Julian centuries since J2000 (2000-01-01T12:00:00.0 TT) for a "
        "two-part TAI epoch.");

  m.def("tdb_minus_tt", [](std::int64_t day, double sec) {
          return star::time::tdb_minus_tt_s({day, sec});
        },
        py::arg("day"), py::arg("sec"),
        "TDB - TT in seconds: the seven-term series of Kaplan, USNO "
        "Circular 179 (2005), eq. 2.6 (~30 us truncation budget per D-6).");

  m.def("tai_add_seconds", &tai_add_seconds, py::arg("day"), py::arg("sec"),
        py::arg("delta_s"),
        "Two-part TAI epoch plus delta_s SI seconds, with the "
        "0 <= sec < 86400 invariant restored.");

  m.def("leap_table_info", &leap_table_info,
        "Bundled leap-second table metadata: version (the IERS Bulletin C "
        "state the table was verified against), expiry_utc (y, m, d) - the "
        "first UTC date a leap second not in the table could take effect - "
        "and entries. The Python layer warns on post-expiry epochs; the "
        "core never reads the clock (D-2).");

  py::class_<star::Ephemeris>(m, "Ephemeris",
                              "Trimmed-DE440 Chebyshev ephemeris evaluator over an "
                              "SREPH v1 file produced by 'star data fetch de440s' "
                              "(FR-4, D-8; format: docs/formats/sreph_v1.md). All "
                              "epochs are TDB seconds since J2000 TDB.")
      .def_static("load", &star::Ephemeris::load_file, py::arg("path"),
                  "Load an SREPH v1 file; raises RuntimeError naming the defect "
                  "on a malformed file.")
      .def(
          "state",
          [](const star::Ephemeris& e, const std::string& body, double tdb_s) {
            return state_tuple(e.state(body, tdb_s));
          },
          py::arg("body"), py::arg("tdb_s"),
          "([x,y,z] position [m], [vx,vy,vz] velocity [m/s]) of body relative "
          "to its stored center (SSB for sun/emb/venus_bary/mars_bary/"
          "jupiter_bary; EMB for earth/moon). Raises ValueError for an unknown "
          "body and IndexError for an out-of-span epoch (never extrapolates).")
      .def(
          "moon_geocentric",
          [](const star::Ephemeris& e, double tdb_s) {
            return state_tuple(e.moon_geocentric(tdb_s));
          },
          py::arg("tdb_s"),
          "Geocentric Moon state composed from the verbatim EMB-relative moon "
          "and earth segments.")
      .def(
          "lunar_librations",
          [](const star::Ephemeris& e, double tdb_s) {
            const star::LibrationAngles a = e.lunar_librations(tdb_s);
            return py::make_tuple(to_vec3(a.angles_rad), to_vec3(a.rates_radps));
          },
          py::arg("tdb_s"),
          "([phi,theta,psi] [rad], rates [rad/s]) of the DE440 Moon "
          "principal-axis frame relative to the ICRF equator (3-1-3 Euler).")
      .def("bodies", &star::Ephemeris::bodies,
           "Sorted distinct body names stored in the file.")
      .def("span_start_tdb_s", &star::Ephemeris::span_start_tdb_s,
           "Start of the span on which every stored body is evaluable.")
      .def("span_end_tdb_s", &star::Ephemeris::span_end_tdb_s,
           "End of the span on which every stored body is evaluable.");

  // -- test-support surface (Phase 2 acceptance evidence; not mission API) --

  m.def("propagate_kepler", &propagate_kepler, py::arg("mu"), py::arg("r0_m"),
        py::arg("v0_mps"), py::arg("t_s"),
        "TEST SUPPORT: analytic elliptic two-body state at time t_s past the "
        "epoch state (Vallado ch. 2; star/testsupport/kepler_ref.hpp). "
        "Returns {r_m, v_mps}.");

  m.def("kepler_convergence", &kepler_convergence, py::arg("mu"),
        py::arg("r0_m"), py::arg("v0_mps"), py::arg("t_end_s"),
        py::arg("method"), py::arg("ladder_s"),
        "TEST SUPPORT: fixed-step global-error ladder on the Kepler problem "
        "for method \"rk4\" or \"rkf78\" (fixed-step mode). Returns "
        "[{h_s, err_m}, ...]; t_end_s must be an exact multiple of every "
        "ladder step.");

  m.def("twobody_drift", &twobody_drift, py::arg("mu"), py::arg("r0_m"),
        py::arg("v0_mps"), py::arg("n_orbits"), py::arg("rtol"),
        py::arg("atol_pos_m"), py::arg("atol_vel_mps"), py::arg("h_init_s"),
        py::arg("h_max_s"),
        "TEST SUPPORT: max relative drift of specific orbital energy and "
        "|h| over n_orbits under adaptive RKF7(8). Returns "
        "{max_energy_rel, max_hmag_rel, steps_accepted, steps_rejected}.");

  m.def("apsis_events", &apsis_events, py::arg("mu"), py::arg("r0_m"),
        py::arg("v0_mps"), py::arg("t_end_s"), py::arg("rtol"),
        py::arg("atol_pos_m"), py::arg("atol_vel_mps"), py::arg("h_init_s"),
        py::arg("h_max_s"), py::arg("event_tol_s"),
        "TEST SUPPORT: apsis passages in (0, t_end_s] located by the event "
        "framework (g = r.v, direction-filtered, Brent on dense output). "
        "Returns [{t_s, kind}, ...] in time order.");

  m.def("hermite_midstep_max_err", &hermite_midstep_max_err, py::arg("mu"),
        py::arg("r0_m"), py::arg("v0_mps"), py::arg("t_end_s"), py::arg("h_s"),
        "TEST SUPPORT: worst midstep position error of the cubic Hermite "
        "dense output vs the analytic solution over a fixed-step RKF7(8) "
        "propagation. Returns {max_err_m}.");

  // Rotation kernel (FR-3, D-7). Quaternions are scalar-first (w, x, y, z)
  // Hamilton quaternions; DCMs are row-major 9-element lists mapping frame
  // A to frame B (v_B = C v_A).
  m.def("quat_multiply", &quat_multiply, py::arg("pw"), py::arg("px"),
        py::arg("py"), py::arg("pz"), py::arg("qw"), py::arg("qx"),
        py::arg("qy"), py::arg("qz"),
        "Hamilton product p (x) q, scalar-first components (D-7). "
        "Composition: q_a2c = quat_multiply(*q_a2b, *q_b2c).");
  m.def("quat_conjugate", &quat_conjugate, py::arg("w"), py::arg("x"),
        py::arg("y"), py::arg("z"),
        "Quaternion conjugate (w, -x, -y, -z); the inverse for unit "
        "quaternions.");
  m.def("quat_normalize", &quat_normalize, py::arg("w"), py::arg("x"),
        py::arg("y"), py::arg("z"),
        "q / |q|; raises ValueError for a zero or non-finite quaternion.");
  m.def("quat_transform", &quat_transform, py::arg("w"), py::arg("x"),
        py::arg("y"), py::arg("z"), py::arg("vx"), py::arg("vy"),
        py::arg("vz"),
        "Coordinates of vector v in frame B given q_a2b: v_B tuple.");
  m.def("quat_to_dcm", &quat_to_dcm, py::arg("w"), py::arg("x"), py::arg("y"),
        py::arg("z"),
        "DCM C_A^B (row-major, 9 elements) of the unit frame-transformation "
        "quaternion q_a2b. Unit norm is the caller's invariant: a non-unit "
        "input is not detected and yields the rotation scaled by |q|^2; "
        "normalize first via quat_normalize.");
  m.def("dcm_to_quat", &dcm_to_quat, py::arg("dcm"),
        "Scalar-first quaternion of a proper rotation DCM via Shepperd's "
        "method; the returned w is >= 0.");
  m.def("dcm_from_euler321", &dcm_from_euler321, py::arg("a1"), py::arg("a2"),
        py::arg("a3"),
        "3-2-1 sequence DCM: C = R1(a3) R2(a2) R3(a1), angles [rad] in "
        "application order.");
  m.def("dcm_from_euler313", &dcm_from_euler313, py::arg("a1"), py::arg("a2"),
        py::arg("a3"),
        "3-1-3 sequence DCM: C = R3(a3) R1(a2) R3(a1), angles [rad] in "
        "application order.");
  m.def("euler321_from_dcm", &euler321_from_dcm, py::arg("dcm"),
        "3-2-1 angles (a1, a2, a3) of a DCM; a2 in [-pi/2, pi/2]; at exact "
        "gimbal lock a1 = 0 by convention (see star/rotation.hpp).");
  m.def("euler313_from_dcm", &euler313_from_dcm, py::arg("dcm"),
        "3-1-3 angles (a1, a2, a3) of a DCM; a2 in [0, pi]; at exact lock "
        "a1 = 0 by convention.");

  // Reference frames (FR-3). Epochs are two-part TAI (day, sec) as in the
  // time functions above.
  m.def("cip_cio_06b", &cip_cio_06b, py::arg("day"), py::arg("sec"),
        "CIP coordinates X, Y and CIO locator s [rad], IAU 2006/2000B, at "
        "the two-part TAI epoch.");
  m.def("nutation_00b", &nutation_00b, py::arg("tt_centuries"),
        "IAU 2000B nutation (dpsi, deps) [rad] at TT Julian centuries since "
        "J2000.");
  m.def("era_00", &era_00, py::arg("day"), py::arg("sec"),
        py::arg("dut1_s") = 0.0,
        "Earth rotation angle [rad] at UT1 = UTC + dut1_s (constant "
        "user-supplied dUT1, default 0 per FR-3).");
  m.def("gcrf_to_cirs", &gcrf_to_cirs, py::arg("day"), py::arg("sec"),
        "GCRS -> CIRS matrix (bias-precession-nutation + CIO locator), "
        "row-major 9 elements.");
  m.def("gcrf_to_itrf", &gcrf_to_itrf, py::arg("day"), py::arg("sec"),
        py::arg("dut1_s") = 0.0,
        "C_GCRF->ITRF = R3(ERA) * C_GCRF->CIRS (row-major 9 elements). "
        "Polar motion neglected (~0.3 urad, bound documented in the frames "
        "chapter); constant dut1_s, default 0.");
  m.def("gcrf_to_moonpa", &gcrf_to_moonpa, py::arg("phi"), py::arg("theta"),
        py::arg("psi"),
        "C_GCRF->MoonPA = R3(psi) R1(theta) R3(phi) from the DE 3-1-3 "
        "libration Euler angles [rad] (Park et al. 2021).");
  m.def("mars_elements", &mars_elements, py::arg("day"), py::arg("sec"),
        "Mars IAU 2015 rotational elements (alpha0, delta0, W) [rad] at the "
        "epoch's TDB (Archinal et al. 2018).");
  m.def("gcrf_to_marsfixed", &gcrf_to_marsfixed, py::arg("day"),
        py::arg("sec"),
        "C_GCRF->MarsFixed = R3(W) R1(pi/2-delta0) R3(pi/2+alpha0) at the "
        "epoch's TDB (row-major 9 elements).");
}
