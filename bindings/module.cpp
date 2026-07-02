// pybind11 bindings for star_reacher._core (Phase 1 contract section 3).
// The binding layer is deliberately thin: it converts types and forwards to
// star::; all behavior lives in the C++ core so the doctest suite exercises
// the same code paths the Python frontend uses.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "star/rng.hpp"
#include "star/run.hpp"
#include "star/testsupport/acceptance.hpp"
#include "star/testsupport/kepler_ref.hpp"
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

py::dict run_and_summarize(const star::RunConfig& cfg,
                           const std::string& out_path) {
  const star::RunSummary s = star::run_twobody(cfg, out_path);
  py::dict d;
  d["steps"] = s.steps;
  d["final_r_m"] = s.final_r_m;
  d["final_v_mps"] = s.final_v_mps;
  d["truth_records"] = s.truth_records;
  d["event_records"] = s.event_records;
  d["records_written"] = s.truth_records + s.event_records;
  return d;
}

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() =
      "star_reacher deterministic C++ core: two-body propagation, SRLOG "
      "writing, and named-stream RNG (Phase 1 surface).";

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
      .def_readwrite("oracle", &star::RunConfig::oracle);

  m.def("run", &run_and_summarize, py::arg("config"), py::arg("out_path"),
        "Propagate the configured two-body case and write an SRLOG v1.0 file "
        "to out_path. Returns a summary dict (steps, final_r_m, final_v_mps, "
        "truth_records, event_records, records_written).");

  m.def("gm", &star::gm, py::arg("body"),
        "Gravitational parameter GM [m^3/s^2] of a named central body "
        "(Phase 1: \"earth\" only; IERS Conventions 2010, TN No. 36). The "
        "single home of the constant - use this for Keplerian conversions.");

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
}
