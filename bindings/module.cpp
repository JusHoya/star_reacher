// pybind11 bindings for star_reacher._core (Phase 1 contract section 3).
// The binding layer is deliberately thin: it converts types and forwards to
// star::; all behavior lives in the C++ core so the doctest suite exercises
// the same code paths the Python frontend uses.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <string>
#include <vector>

#include "star/rng.hpp"
#include "star/run.hpp"
#include "star/version.hpp"

namespace py = pybind11;

namespace {

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
}
