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

#include "star/ephemeris.hpp"
#include "star/rng.hpp"
#include "star/run.hpp"
#include "star/testsupport/acceptance.hpp"
#include "star/testsupport/kepler_ref.hpp"
#include "star/time.hpp"
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
}
