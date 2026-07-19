// Plain-data mirror of the resolved [gnc] and [sensors.*] mission surface
// the Python validator produces (D-2: the core never parses text). RunConfig
// carries a GncConfig for the Phase 6 closed-loop attitude path; the Python
// frontend fills every field from the resolved mission dict, and the core
// re-checks only what it needs to stay well-defined (defensive guard, not
// user UX), matching the VehicleConfig pattern.
//
// Component parameters ride as two flat string-keyed maps (scalars and
// double vectors) instead of per-component structs, so a new built-in or a
// future Python plugin adds config fields without touching this header or
// the binding layer - composition over inheritance, plain data end to end.
#ifndef STAR_GNC_CONFIG_HPP
#define STAR_GNC_CONFIG_HPP

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace star {
namespace gnc {

// One GNC chain slot (nav, guidance, or control): the registry name of the
// component plus its parameters. Components validate their own parameters
// at construction and throw std::invalid_argument on violations.
struct GncComponentCfg {
  std::string component;                                  // registry name
  std::map<std::string, double> scalars;                  // e.g. azimuth_deg
  std::map<std::string, std::vector<double>> vectors;     // e.g. kp_nm_per_rad
};

// One configured sensor instance. Kind-specific parameters ride as the same
// two flat string-keyed maps the components use, for the same reason: a new
// FR-23 sensor kind (or a new error term on an existing one) adds config
// fields without touching this header or the binding layer. Each sensor
// validates its own parameter set at construction and throws
// std::invalid_argument on violations.
struct GncSensorCfg {
  std::string kind;                    // canonical sensor kind (srlog_writer.hpp)
  std::uint32_t sample_rate_hz = 0;    // must divide the control rate
  std::map<std::string, double> scalars;               // e.g. gyro_quantum_rad
  std::map<std::string, std::vector<double>> vectors;  // e.g. boresight_b
};

// The whole [gnc] surface. enabled == false leaves the Phase 4 kinematic
// attitude modes in sole authority and declares no v1.2 log groups, so
// pre-Phase-6 missions resolve and log exactly as before.
struct GncConfig {
  bool enabled = false;
  std::uint32_t control_rate_hz = 0;   // must equal 1/dt_s (one cycle per step)
  std::uint32_t latency_cycles = 0;    // FR-25 command-application delay
  GncComponentCfg nav;
  GncComponentCfg guidance;
  GncComponentCfg control;
  std::vector<GncSensorCfg> sensors;   // canonical kind order
};

}  // namespace gnc
}  // namespace star

#endif  // STAR_GNC_CONFIG_HPP
