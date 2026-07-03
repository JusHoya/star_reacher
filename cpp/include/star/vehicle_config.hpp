// Plain-data mirror of the resolved vehicle + [[sequence]] surface the Python
// validator produces (D-2: the core never parses text). RunConfig carries a
// VehicleConfig and a sequence of SequenceEntry for the Phase 4 run_vehicle
// path; the Python frontend fills every field from the WS-C resolved dicts and
// run_vehicle re-checks only what it needs to stay well-defined, matching
// check_config_env's defensive style. Nested std::vector members mirror the
// nested [[stage]]/[[stage.tank]]/... schema so the translation is mechanical.
//
// Units are SI throughout (structural frame +X forward, FR-13). Inertia
// tensors are row-major flattened 3x3 (9 elements) about the block's own CG;
// vec3 quantities are std::array<double, 3>.
#ifndef STAR_VEHICLE_CONFIG_HPP
#define STAR_VEHICLE_CONFIG_HPP

#include <array>
#include <string>
#include <vector>

namespace star {

// One settled cylindrical propellant tank (axis +X, aft face -X; A-2).
struct TankCfg {
  double radius_m = 0.0;
  double length_m = 0.0;
  std::array<double, 3> position_m{{0.0, 0.0, 0.0}};  // cylinder center
  double propellant_mass_kg = 0.0;                    // load at t0
  double density_kgpm3 = 0.0;
};

// One engine (or an equivalent rigid cluster). feeds_tank_index is the index of
// the fed tank within the owning stage (resolved by name in Python).
struct EngineCfg {
  std::string name;
  int feeds_tank_index = -1;
  double thrust_vac_N = 0.0;
  double isp_vac_s = 0.0;
  double exit_area_m2 = 0.0;
  std::array<double, 3> position_m{{0.0, 0.0, 0.0}};
  std::array<double, 3> axis{{1.0, 0.0, 0.0}};  // nominal thrust direction
  double gimbal_max_deg = 0.0;
  double gimbal_rate_dps = 0.0;
  double throttle_min = 1.0;
  double throttle_max = 1.0;
  double spool_time_s = 0.0;
  int ignitions = 1;
};

// One RCS thruster cluster (index-matched positions/directions).
struct RcsCfg {
  std::string name;
  double thrust_N = 0.0;
  double min_impulse_bit_Ns = 0.0;
  std::vector<std::array<double, 3>> thruster_positions_m;
  std::vector<std::array<double, 3>> thruster_directions;
};

// One reaction wheel.
struct WheelCfg {
  std::string name;
  std::array<double, 3> axis{{1.0, 0.0, 0.0}};
  double max_torque_Nm = 0.0;
  double max_momentum_Nms = 0.0;
};

// One discretely droppable item; rides as stack mass until jettisoned.
struct JettisonCfg {
  std::string name;
  double mass_kg = 0.0;
  std::array<double, 3> cg_m{{0.0, 0.0, 0.0}};
  std::array<double, 9> inertia_kgm2{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0}};  // row-major about the item CG
};

// One ordered stage (bottom, first-burning stage first).
struct StageCfg {
  std::string name;
  double dry_mass_kg = 0.0;
  std::array<double, 3> dry_cg_m{{0.0, 0.0, 0.0}};
  std::array<double, 9> dry_inertia_kgm2{
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};  // about the dry CG
  std::vector<TankCfg> tanks;
  std::vector<EngineCfg> engines;
  std::vector<RcsCfg> rcs;
  std::vector<WheelCfg> wheels;
  std::vector<JettisonCfg> jettison;
};

// One per-stack-configuration aero block plus its parsed Mach table (FR-9).
// The mach/ca/cnalpha_per_rad/xcp_m arrays are the CSV columns, parallel on a
// strictly increasing Mach grid; cmq_per_rad == 0.0 disables pitch damping.
struct AeroCfg {
  std::string config;
  double ref_area_m2 = 0.0;
  double ref_diameter_m = 0.0;
  double cmq_per_rad = 0.0;
  std::vector<double> mach;
  std::vector<double> ca;
  std::vector<double> cnalpha_per_rad;
  std::vector<double> xcp_m;
};

// The whole vehicle: ordered stages plus the aero blocks. The aero block used
// at a given moment is selected by the number of stages already separated
// (index min(separations, aero.size()-1)); see run_vehicle.
struct VehicleConfig {
  std::vector<StageCfg> stages;
  std::vector<AeroCfg> aero;
};

// One resolved [[sequence]] entry. Only the members relevant to the entry's
// trigger/action carry meaning; the rest keep their defaults (Python fills
// exactly the keys the resolved dict carries).
struct SequenceEntry {
  std::string name;
  std::string trigger;  // "elapsed" | "after_event" | "condition"
  double t_s = 0.0;     // elapsed
  std::string event;    // after_event: the earlier entry name
  double offset_s = 0.0;
  std::string condition;         // condition: the condition kind
  double altitude_m = 0.0;       // altitude_above / altitude_below
  double perigee_alt_m = 0.0;    // perigee_above
  std::string body;              // soi_transition: the entered body
  std::string action;            // the FR-14 v1 action vocabulary
  std::string stage;             // ignite/cutoff/separate/jettison target
  std::string engine;            // ignite_engine / cutoff_engine
  std::string item;              // jettison
  double azimuth_deg = 0.0;      // pitch_program
  std::vector<double> pitch_t_s;
  std::vector<double> pitch_deg;
  std::string frame;                        // rate_command ("gcrf" | "body")
  std::array<double, 3> omega_dps{{0.0, 0.0, 0.0}};
};

}  // namespace star

#endif  // STAR_VEHICLE_CONFIG_HPP
