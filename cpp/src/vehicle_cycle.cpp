// VehicleCycle implementation: the Phase 4 run_vehicle loop body, factored
// into a stepping core (contracts in vehicle_cycle.hpp), plus the Phase 6
// GNC wiring (sensors, chain, latency FIFO, torque-driven attitude).
//
// The Phase 4 kinematic path is transplanted from run.cpp verbatim - same
// helpers, same evaluation order, same record emission order - so kinematic
// missions remain byte-frozen across the refactor; the pitch-table
// interpolation moved to models::pwl_interp_clamped with unchanged
// arithmetic (vehicle6dof.cpp) so the Phase 6 pitch-program guidance can
// share it. Model math lives in the chapter-tracked modules under
// cpp/src/models/; this file owns only orchestration.
#include "star/vehicle_cycle.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "star/constants.hpp"
#include "star/ephemeris.hpp"
#include "star/frames.hpp"
#include "star/gnc/builtin.hpp"
#include "star/gnc/component.hpp"
#include "star/integrate.hpp"
#include "star/models/actuators.hpp"
#include "star/models/aero.hpp"
#include "star/models/atmosphere_hp.hpp"
#include "star/models/atmosphere_ussa76.hpp"
#include "star/models/environment.hpp"
#include "star/models/gravgrad.hpp"
#include "star/models/massprops.hpp"
#include "star/models/propulsion.hpp"
#include "star/models/rigidbody.hpp"
#include "star/models/twobody.hpp"
#include "star/models/vehicle6dof.hpp"
#include "star/rotation.hpp"
#include "star/sensors/camera.hpp"
#include "star/sensors/imu.hpp"
#include "star/sensors/optical.hpp"
#include "star/sensors/radio.hpp"
#include "star/sensors/sensor.hpp"
#include "star/time.hpp"
#include "star/version.hpp"

namespace star {

namespace {

double deg2rad_v(double d) { return d * (constants::TWO_PI / 360.0); }

Eigen::Vector3d arr3(const std::array<double, 3>& a) {
  return Eigen::Vector3d(a[0], a[1], a[2]);
}

Eigen::Matrix3d mat3_flat(const std::array<double, 9>& a) {
  Eigen::Matrix3d m;
  m << a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8];
  return m;
}

// Two unit deflection axes orthogonal to a thrust axis (the propulsion module
// requires an orthonormal gimbal frame; for axis == +X these are +Y/+Z).
void gimbal_basis(const Eigen::Vector3d& axis, Eigen::Vector3d& g1,
                  Eigen::Vector3d& g2) {
  const Eigen::Vector3d ref = (std::fabs(axis.z()) < 0.9)
                                  ? Eigen::Vector3d::UnitZ()
                                  : Eigen::Vector3d::UnitX();
  g1 = (ref - ref.dot(axis) * axis).normalized();
  g2 = axis.cross(g1);
}

struct EngineRt {
  models::EngineParams params;
  models::EngineState state;
  models::EngineCommand command;
  int stage_idx = -1;
  int tank_idx = -1;  // global index into VehicleRuntime::tanks, or -1
  std::string name;
};
struct TankRt {
  models::TankParams params;
  double mass_kg = 0.0;
  int stage_idx = -1;
};
struct WheelRt {
  models::WheelParams params;
  models::WheelState state;
  int stage_idx = -1;
};
struct JettRt {
  models::BodyProps body;
  bool attached = true;
  int stage_idx = -1;
  std::string name;
};
struct StageRt {
  models::BodyProps dry;
  bool attached = true;
  std::string name;
  std::vector<int> tanks;
  std::vector<int> engines;
  std::vector<int> jett;
};
struct VehicleRuntime {
  std::vector<StageRt> stages;
  std::vector<TankRt> tanks;
  std::vector<EngineRt> engines;
  std::vector<WheelRt> wheels;
  std::vector<JettRt> jett;
  std::vector<models::AeroTables> aero;
};

VehicleRuntime build_vehicle(const VehicleConfig& v) {
  VehicleRuntime rt;
  int stage_idx = 0;
  for (const StageCfg& s : v.stages) {
    StageRt st;
    st.name = s.name;
    st.dry.mass_kg = s.dry_mass_kg;
    st.dry.cg_m = arr3(s.dry_cg_m);
    st.dry.inertia_kgm2 = mat3_flat(s.dry_inertia_kgm2);
    for (const TankCfg& t : s.tanks) {
      TankRt tr;
      tr.params.radius_m = t.radius_m;
      tr.params.length_m = t.length_m;
      // Aft interior face center = cylinder center - (L/2) xhat (axis +X).
      tr.params.aft_center_m =
          arr3(t.position_m) - Eigen::Vector3d(0.5 * t.length_m, 0.0, 0.0);
      tr.params.density_kgpm3 = t.density_kgpm3;
      tr.params.initial_mass_kg = t.propellant_mass_kg;
      tr.mass_kg = t.propellant_mass_kg;
      tr.stage_idx = stage_idx;
      st.tanks.push_back(static_cast<int>(rt.tanks.size()));
      rt.tanks.push_back(tr);
    }
    for (const EngineCfg& e : s.engines) {
      EngineRt er;
      er.name = e.name;
      er.stage_idx = stage_idx;
      er.params.thrust_vac_N = e.thrust_vac_N;
      er.params.isp_vac_s = e.isp_vac_s;
      er.params.exit_area_m2 = e.exit_area_m2;
      er.params.throttle_min = e.throttle_min;
      er.params.throttle_max = e.throttle_max;
      er.params.spool_up_s = e.spool_time_s;
      er.params.spool_down_s = e.spool_time_s;
      er.params.max_ignitions = e.ignitions;
      er.params.gimbal_limit_rad = deg2rad_v(e.gimbal_max_deg);
      er.params.gimbal_rate_radps = deg2rad_v(e.gimbal_rate_dps);
      er.params.position_m = arr3(e.position_m);
      er.params.axis = arr3(e.axis).normalized();
      gimbal_basis(er.params.axis, er.params.gimbal_axis_1,
                   er.params.gimbal_axis_2);
      er.tank_idx = (e.feeds_tank_index >= 0 &&
                     e.feeds_tank_index < static_cast<int>(st.tanks.size()))
                        ? st.tanks[static_cast<std::size_t>(e.feeds_tank_index)]
                        : -1;
      st.engines.push_back(static_cast<int>(rt.engines.size()));
      rt.engines.push_back(er);
    }
    for (const WheelCfg& w : s.wheels) {
      WheelRt wr;
      wr.stage_idx = stage_idx;
      wr.params.axis = arr3(w.axis).normalized();
      wr.params.torque_max_Nm = w.max_torque_Nm;
      wr.params.momentum_max_Nms = w.max_momentum_Nms;
      rt.wheels.push_back(wr);
    }
    for (const JettisonCfg& j : s.jettison) {
      JettRt jr;
      jr.name = j.name;
      jr.stage_idx = stage_idx;
      jr.body.mass_kg = j.mass_kg;
      jr.body.cg_m = arr3(j.cg_m);
      jr.body.inertia_kgm2 = mat3_flat(j.inertia_kgm2);
      st.jett.push_back(static_cast<int>(rt.jett.size()));
      rt.jett.push_back(jr);
    }
    rt.stages.push_back(st);
    ++stage_idx;
  }
  for (const AeroCfg& a : v.aero) {
    models::AeroTables t;
    t.ref_area_m2 = a.ref_area_m2;
    t.ref_diameter_m = a.ref_diameter_m;
    t.cmq_per_rad = a.cmq_per_rad;
    t.mach = a.mach;
    t.ca = a.ca;
    t.cnalpha_per_rad = a.cnalpha_per_rad;
    t.xcp_m = a.xcp_m;
    rt.aero.push_back(t);
  }
  return rt;
}

struct StackProps {
  models::BodyProps composite;
  models::BodyRates rates;
};

// Compose the currently attached stack in the FIXED FR-10 order (fixed bodies
// -- dry stages then attached jettison items -- then one settled slug per
// attached tank), so the composite wet mass is a stable same-order binary64
// sum every cycle (EC-2 bit-exactness).
StackProps compose_stack(const VehicleRuntime& v) {
  std::vector<models::BodyProps> bodies;
  std::vector<models::BodyRates> rates;
  for (const StageRt& s : v.stages) {
    if (!s.attached) continue;
    bodies.push_back(s.dry);
    rates.emplace_back();
    for (int j : s.jett) {
      if (v.jett[static_cast<std::size_t>(j)].attached) {
        bodies.push_back(v.jett[static_cast<std::size_t>(j)].body);
        rates.emplace_back();
      }
    }
  }
  for (const StageRt& s : v.stages) {
    if (!s.attached) continue;
    for (int ti : s.tanks) {
      const TankRt& tk = v.tanks[static_cast<std::size_t>(ti)];
      double consume = 0.0;  // positive propellant consumption [kg/s]
      if (tk.mass_kg > 0.0) {
        for (int ei : s.engines) {
          const EngineRt& en = v.engines[static_cast<std::size_t>(ei)];
          if (en.tank_idx == ti && en.state.throttle_level > 0.0) {
            consume +=
                models::engine_mdot_kgps(en.params, en.state.throttle_level);
          }
        }
      }
      bodies.push_back(models::tank_slug_props(tk.params, tk.mass_kg));
      // massprops uses SIGNED dm/dt; a draining tank is negative.
      rates.push_back(models::tank_slug_rates(tk.params, tk.mass_kg, -consume));
    }
  }
  StackProps sp;
  sp.composite = models::compose(bodies);
  sp.rates = models::compose_rates(bodies, rates);
  return sp;
}

// ZOH context captured for one control cycle (D-5): everything the frozen RHS
// and the log-instant force breakdown read.
struct CycleCtx {
  const VehicleRuntime* v = nullptr;
  bool earth = true;
  Eigen::Matrix3d c_i2b = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d c_b2i = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d c_gcrf_to_itrf = Eigen::Matrix3d::Identity();
  Eigen::Vector3d omega_planet_i = Eigen::Vector3d::Zero();
  Eigen::Vector3d omega_b = Eigen::Vector3d::Zero();
  Eigen::Vector3d cg_b = Eigen::Vector3d::Zero();
  double mass_kg = 1.0;
  int aero_idx = -1;
  double planet_a_m = 0.0;
  double planet_inv_f = 0.0;
};

struct BodyLoads {
  Eigen::Vector3d f_thrust = Eigen::Vector3d::Zero();
  Eigen::Vector3d tq_thrust = Eigen::Vector3d::Zero();
  Eigen::Vector3d f_aero = Eigen::Vector3d::Zero();
  Eigen::Vector3d tq_aero = Eigen::Vector3d::Zero();
  double alt_m = 0.0;
  double mach = 0.0;
  double q_bar_pa = 0.0;
  double rho = 0.0;
};

// Body-frame thrust and aero force/torque about the composite CG, plus the
// air-data diagnostics (FR-8/FR-9). Air-relative velocity is v - omega x r
// (eq:drag:vrel); Mach and dynamic pressure come straight from the aero model
// (its exact q_bar == 0 at |v_rel| == 0 is what makes the on-pad q_bar exactly
// zero, EC-10).
BodyLoads eval_body_loads(const CycleCtx& c, const Eigen::Vector3d& r_i,
                          const Eigen::Vector3d& v_i) {
  BodyLoads out;
  double p_amb = 0.0;
  double sos = 0.0;
  if (c.earth) {
    const Eigen::Vector3d r_ecef = c.c_gcrf_to_itrf * r_i;
    out.alt_m = models::geodetic_altitude(r_ecef, c.planet_a_m, c.planet_inv_f);
    if (out.alt_m >= -5000.0 && out.alt_m < 86000.0) {
      const models::Ussa76State s = models::ussa76_state(out.alt_m);
      out.rho = s.density_kgpm3;
      sos = s.speed_of_sound_mps;
      p_amb = s.pressure_Pa;
    }
  } else {
    out.alt_m = r_i.norm() - c.planet_a_m;
  }
  for (const EngineRt& e : c.v->engines) {
    if (!c.v->stages[static_cast<std::size_t>(e.stage_idx)].attached) continue;
    if (e.tank_idx >= 0 &&
        c.v->tanks[static_cast<std::size_t>(e.tank_idx)].mass_kg <= 0.0) {
      continue;
    }
    const models::EngineForceTorque ft =
        models::engine_force_torque(e.params, e.state, p_amb, c.cg_b);
    out.f_thrust += ft.force_N;
    out.tq_thrust += ft.torque_Nm;
  }
  if (c.aero_idx >= 0 && out.rho > 0.0 && sos > 0.0) {
    const Eigen::Vector3d v_rel_i = v_i - c.omega_planet_i.cross(r_i);
    const Eigen::Vector3d v_rel_b = c.c_i2b * v_rel_i;
    const models::AeroForceTorque a = models::aero_force_torque(
        c.v->aero[static_cast<std::size_t>(c.aero_idx)], v_rel_b, out.rho, sos,
        c.cg_b.x(), c.omega_b);
    out.f_aero = a.force_N;
    out.tq_aero = a.torque_Nm;
    out.mach = a.mach;
    out.q_bar_pa = a.q_bar_Pa;
  }
  return out;
}

double osc_perigee_alt(double mu, const Eigen::Vector3d& r,
                       const Eigen::Vector3d& v, double radius_m) {
  const double rn = r.norm();
  const double energy = 0.5 * v.squaredNorm() - mu / rn;
  const double a = -mu / (2.0 * energy);
  const Eigen::Vector3d h = r.cross(v);
  const Eigen::Vector3d e_vec = v.cross(h) / mu - r / rn;
  return a * (1.0 - e_vec.norm()) - radius_m;
}

// TDB seconds since J2000 TDB for the epoch shifted by t_s (mirrors
// EnvironmentModel::tdb_s_at; used only for the SOI-transition geometry).
double tdb_s_at(const time::TaiEpoch& epoch, double t_s) {
  const time::TaiEpoch tai = time::tai_add_seconds(epoch, t_s);
  const time::TwoPartJd jd = time::tdb_jd(tai);
  return ((jd.jd1 - 2451545.0) + jd.jd2) * 86400.0;
}

models::CentralBody central_body_from_name(const std::string& name) {
  if (name == "earth") return models::CentralBody::kEarth;
  if (name == "moon") return models::CentralBody::kMoon;
  if (name == "mars") return models::CentralBody::kMars;
  if (name == "sun") return models::CentralBody::kSun;
  throw std::invalid_argument(
      "unknown central body: \"" + name +
      "\" (supported: \"earth\", \"moon\", \"mars\", \"sun\")");
}

models::AtmosphereModel atmosphere_from_name(const std::string& name) {
  if (name == "ussa76") return models::AtmosphereModel::kUssa76;
  if (name == "harris_priester") return models::AtmosphereModel::kHarrisPriester;
  if (name == "mars_exponential") return models::AtmosphereModel::kMarsExponential;
  throw std::invalid_argument(
      "unknown atmosphere model: \"" + name +
      "\" (supported: \"ussa76\", \"harris_priester\", \"mars_exponential\")");
}

// The FR-14 v1 sequence attitude actions are open-loop commands; with the
// GNC chain in authority they would fight the controller, so a mission may
// carry one or the other, never both (the Python validator enforces the
// same rule with DX-2 messages).
bool is_attitude_action(const std::string& action) {
  return action == "pitch_program" || action == "attitude_hold" ||
         action == "prograde_hold" || action == "rate_command";
}

void check_config_vehicle(const RunConfig& cfg) {
  if (cfg.central_body == "sun") {
    // The vehicle path's altitude events, pad geometry, and aero assume a
    // planetary central body; the Phase 5 heliocentric regime is served by
    // the point-mass run_env path only (the Python validator enforces the
    // same restriction, mission.py sun-regime rules).
    throw std::invalid_argument(
        "run_vehicle: central_body \"sun\" is not supported by the vehicle "
        "path (heliocentric missions are point-mass, use run_env)");
  }
  if (cfg.integrator != "rk4") {
    throw std::invalid_argument(
        "run_vehicle: integrator must be \"rk4\" in Phase 4");
  }
  if (!(cfg.duration_s > 0.0)) {
    throw std::invalid_argument("run_vehicle: duration_s must be > 0");
  }
  if (!(cfg.dt_s > 0.0)) {
    throw std::invalid_argument("run_vehicle: dt_s must be > 0");
  }
  if (cfg.truth_rate_hz < 1) {
    throw std::invalid_argument("run_vehicle: truth_rate_hz must be >= 1");
  }
  if (cfg.vehicle.stages.empty()) {
    throw std::invalid_argument(
        "run_vehicle: the vehicle must define at least one stage");
  }
  const double steps_exact = cfg.duration_s / cfg.dt_s;
  if (std::fabs(steps_exact - std::llround(steps_exact)) > 1e-9 * steps_exact) {
    throw std::invalid_argument(
        "run_vehicle: duration_s must be an integer multiple of dt_s");
  }
  const double decim_exact =
      1.0 / (cfg.dt_s * static_cast<double>(cfg.truth_rate_hz));
  if (decim_exact < 0.5 ||
      std::fabs(decim_exact - std::llround(decim_exact)) > 1e-9 * decim_exact) {
    throw std::invalid_argument(
        "run_vehicle: 1/(dt_s*truth_rate_hz) must be a positive integer");
  }
  if (cfg.gnc.enabled) {
    if (cfg.gnc.control_rate_hz < 1) {
      throw std::invalid_argument(
          "run_vehicle: [gnc] control_rate_hz must be >= 1");
    }
    // One control cycle per loop step (D-5): the loop's dt IS the control
    // period, so the configured rate must equal 1/dt_s exactly.
    const double cycles = cfg.dt_s * static_cast<double>(cfg.gnc.control_rate_hz);
    if (std::fabs(cycles - 1.0) > 1e-9) {
      throw std::invalid_argument(
          "run_vehicle: [gnc] control_rate_hz must equal 1/dt_s (one "
          "control cycle per integrator step, D-5)");
    }
    for (const gnc::GncSensorCfg& s : cfg.gnc.sensors) {
      if (s.kind == "imu") {
        // The v1 IMU emits exactly one increment pair per major cycle
        // (ch:sensors-imu assumption 1): its sample interval IS the
        // control period.
        if (s.sample_rate_hz != cfg.gnc.control_rate_hz) {
          throw std::invalid_argument(
              "run_vehicle: [sensors.imu] sample_rate_hz must equal "
              "control_rate_hz (the v1 IMU emits one increment pair per "
              "control cycle, D-5)");
        }
      } else if (s.sample_rate_hz < 1 ||
                 cfg.gnc.control_rate_hz % s.sample_rate_hz != 0) {
        throw std::invalid_argument(
            "run_vehicle: [sensors." + s.kind + "] sample_rate_hz must be "
            "a positive integer divisor of control_rate_hz (sensors sample "
            "on the control-cycle grid)");
      }
    }
    for (const SequenceEntry& e : cfg.sequence) {
      if (is_attitude_action(e.action)) {
        throw std::invalid_argument(
            "run_vehicle: sequence action \"" + e.action + "\" is an "
            "open-loop attitude command and cannot be combined with [gnc] "
            "(the GNC chain holds attitude authority)");
      }
    }
  }
}

// The nav component's declared error-state layout, validated against the run
// (gnc/component.hpp). Captured ONCE per run: the declaration is fixed for
// the run, and calling error_layout() once means a component cannot answer
// the validation call and the per-cycle call differently. An empty result
// means the component declared no layout and the run writes no nav.err.
std::vector<gnc::ErrorBlock> capture_error_layout(
    const RunConfig& cfg, const gnc::IGncComponent* nav) {
  if (nav == nullptr || nav->state_dim() <= 0) return {};
  std::vector<gnc::ErrorBlock> layout = nav->error_layout();
  // The true IMU biases exist only when the run flies an IMU, which is
  // decidable from the configured sensor list before the sensors are built.
  bool imu_configured = false;
  for (const gnc::GncSensorCfg& s : cfg.gnc.sensors) {
    if (s.kind == "imu") imu_configured = true;
  }
  gnc::validate_error_layout(layout, nav->state_dim(), imu_configured);
  return layout;
}

// Header declaration shared by the constructor and make_header_fields: a
// pure function of the config plus the nav component's declared dimensions.
log::SrlogHeaderFields build_header_fields(const RunConfig& cfg,
                                           int nav_state_dim,
                                           int nav_cov_dim,
                                           int nav_innov_max_dim,
                                           bool nav_err_enabled) {
  // Enabled force sources in canonical order; the forces group is declared
  // only when its rate is nonzero (unchanged Phase 4 logic).
  std::vector<std::string> sources;
  if (cfg.forces_rate_hz != 0) {
    sources.push_back("gravity");
    if (!cfg.third_bodies.empty()) sources.push_back("thirdbody");
    if (!cfg.vehicle.aero.empty()) sources.push_back("aero");
    sources.push_back("thrust");
    sources.push_back("gravgrad");
  }

  log::SrlogHeaderFields fields;
  fields.core_version = core_version();
  fields.git_hash = git_hash();
  fields.config_sha256 = cfg.config_sha256;
  fields.master_seed = cfg.master_seed;
  fields.oracle = cfg.oracle;
  fields.epoch_utc = cfg.epoch_utc;
  fields.central_body = cfg.central_body;
  fields.truth_rate_hz = cfg.truth_rate_hz;
  fields.force_sources = sources;
  fields.forces_rate_hz = cfg.forces_rate_hz;
  fields.mass_rate_hz = cfg.mass_rate_hz;
  fields.env_rate_hz = cfg.env_rate_hz;
  if (cfg.gnc.enabled) {
    fields.cycle_rate_hz = cfg.gnc.control_rate_hz;
    fields.latency_cycles = cfg.gnc.latency_cycles;
    for (const gnc::GncSensorCfg& s : cfg.gnc.sensors) {
      log::SensorGroupDecl decl;
      decl.kind = s.kind;
      decl.rate_hz = s.sample_rate_hz;
      if (s.kind == "camera") {
        // The camera record's pixel-pair count is fixed at header-write
        // time, so the declaration reads the landmark count out of the
        // resolved config through the same parser the sensor uses.
        decl.landmarks = static_cast<std::uint32_t>(
            sensors::parse_camera_cfg(s).landmarks_fixed_m.size());
      }
      fields.sensors.push_back(decl);
    }
    if (nav_state_dim > 0) {
      // nav.est and nav.err log at the control rate; nav.err shares
      // nav.est's dimension by the writer's construction.
      fields.nav_est_rate_hz = cfg.gnc.control_rate_hz;
      fields.nav_state_dim = static_cast<std::uint32_t>(nav_state_dim);
      if (nav_cov_dim != nav_state_dim) {
        // Error-state estimators declare an independent covariance
        // dimension (srlog_writer.hpp contract); equal dims use the
        // writer's default.
        fields.nav_cov_dim = static_cast<std::uint32_t>(nav_cov_dim);
      }
      // nav.err exists only for an estimator that declared how to read its
      // state vector (gnc::ErrorBlock). An estimator that declares no layout
      // gets no channel rather than a channel of zeros, which would be
      // indistinguishable from a perfect estimate.
      fields.nav_err_enabled = nav_err_enabled;
    }
    if (nav_innov_max_dim > 0) {
      fields.nav_innov_enabled = true;
      fields.nav_innov_max_dim = static_cast<std::uint32_t>(nav_innov_max_dim);
    }
    fields.gnc_cmd_rate_hz = cfg.gnc.control_rate_hz;
  }
  return fields;
}

models::EnvironmentSpec make_env_spec(const RunConfig& cfg) {
  models::EnvironmentSpec spec;
  spec.central_body = central_body_from_name(cfg.central_body);
  spec.epoch_tai = {cfg.epoch_tai_day, cfg.epoch_tai_sec};
  spec.gravity_model = cfg.gravity_model;
  spec.gravity_field_path = cfg.gravity_field_path;
  spec.gravity_degree = cfg.gravity_degree;
  spec.gravity_order = cfg.gravity_order;
  spec.third_bodies = cfg.third_bodies;
  spec.srp_enabled = cfg.srp_enabled;
  spec.cr_a_over_m_m2pkg = cfg.cr_a_over_m_m2pkg;
  spec.srp_occulters = cfg.srp_occulters;
  spec.atmosphere = cfg.drag_enabled ? atmosphere_from_name(cfg.atmosphere)
                                     : models::AtmosphereModel::kNone;
  spec.cd_a_over_m_m2pkg = cfg.cd_a_over_m_m2pkg;
  spec.hp_exponent_n = cfg.hp_exponent_n;
  spec.ephemeris_path = cfg.ephemeris_path;
  return spec;
}

enum class AttMode {
  kPadFixed,
  kPitchProgram,
  kInertialHold,
  kRateCommand,
  kProgradeHold,
  kGnc
};

}  // namespace

struct VehicleCycle::Impl {
  // --- configuration and models (construction order matters: the GNC nav
  // component must exist before the writer so the header can declare its
  // state dimension) -------------------------------------------------------
  RunConfig cfg;
  models::EnvironmentModel env;
  std::unique_ptr<gnc::IGncComponent> nav;
  std::unique_ptr<gnc::IGncComponent> guidance;
  std::unique_ptr<gnc::IGncComponent> control;
  // Declared before the writer because the header's nav.err declaration
  // depends on whether this layout is empty.
  std::vector<gnc::ErrorBlock> nav_layout;
  log::SrlogWriter writer;

  models::CentralBody central;
  double mu;
  bool earth;
  double planet_a_m = constants::WGS84_A_M;
  double planet_inv_f = constants::WGS84_INV_F;
  time::TaiEpoch epoch;
  VehicleRuntime veh;
  std::optional<Ephemeris> soi_eph;
  std::vector<std::string> sources;

  // --- continuous and discrete loop state ---------------------------------
  double y[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  Eigen::Map<Eigen::Vector3d> r_m{y};
  Eigen::Map<Eigen::Vector3d> v_mps{y + 3};
  Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
  Eigen::Vector3d omega_b = Eigen::Vector3d::Zero();
  Eigen::Quaterniond q_hold = Eigen::Quaterniond::Identity();
  Eigen::Vector3d rate_omega_b = Eigen::Vector3d::Zero();
  bool rate_frame_gcrf = true;
  bool released = true;
  AttMode mode = AttMode::kInertialHold;
  double pad_lat = 0.0;
  double pad_lon = 0.0;
  double pad_alt = 0.0;
  Eigen::Vector3d up0 = Eigen::Vector3d::UnitZ();
  Eigen::Vector3d east0 = Eigen::Vector3d::UnitX();
  Eigen::Vector3d north0 = Eigen::Vector3d::UnitY();
  double pitch_az_deg = 0.0;
  std::vector<double> pitch_t;
  std::vector<double> pitch_d;

  std::int64_t steps = 0;
  double dt = 0.0;
  std::int64_t truth_decim = 0;
  std::int64_t forces_decim = 0;
  std::int64_t mass_decim = 0;
  std::int64_t env_decim = 0;

  RunSummary run_summary;
  std::map<std::string, double> fire_time;
  std::vector<char> fired;
  integrate::Rk4 rk4{6};
  CycleCtx ctx;
  std::vector<log::ForceSourceSample> samples;
  bool stop = false;
  int aero_sep_count = 0;
  std::int64_t i = 0;      // current cycle index
  bool finished = false;
  // The log handle has been released, either by finish() or by an explicit
  // close() on an abandoned run. Tracked separately from `finished` because
  // only the former means the file is a complete run.
  bool closed = false;

  // Per-cycle products carried from process_cycle() into advance_cycle().
  StackProps sp;
  BodyLoads bl;

  // Cycle-start endpoint values for the trapezoidal sensor truth, captured
  // at the top of advance_cycle before the in-place state update.
  Eigen::Vector3d r_start_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d v_start_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d sf_start_ = Eigen::Vector3d::Zero();

  // --- GNC runtime ---------------------------------------------------------
  bool gnc_active = false;
  std::int64_t act_cycle = 0;
  std::vector<std::unique_ptr<sensors::ISensor>> sensor_list;
  std::vector<std::int64_t> sensor_decim;
  sensors::Imu* imu = nullptr;  // non-owning view into sensor_list
  // Non-owning views of the aiding sensors, so the GNC block can offer their
  // latest measurements to the nav stage (an aiding estimator has no other
  // route to them). Each is null when the run configures no such sensor.
  sensors::NavFix* navfix = nullptr;
  sensors::StarTracker* startracker = nullptr;
  sensors::Altimeter* altimeter = nullptr;
  // Which sensor index each aiding kind occupies, for the nav.innov records.
  std::uint32_t navfix_id = 0;
  std::uint32_t startracker_id = 0;
  std::uint32_t altimeter_id = 0;
  // The sensor-suite parameters handed to components at init, so an
  // estimator's stochastic model is the configured truth model (ch:ekf
  // assumption 3) rather than a duplicate that can drift out of sync.
  gnc::NavSensorModel sensor_model;
  // Geometry composed at the most recent cycle end - which is exactly the
  // instant the point sensors are sampled at, and therefore the instant the
  // GNC block's environment context describes.
  models::SensorGeometry last_geom;
  // True when any configured sensor consumes ephemeris/shadow/body-fixed
  // geometry; the IMU alone does not, and composing it costs ephemeris
  // evaluations per cycle.
  bool needs_geometry_ = false;
  std::optional<gnc::LatencyFifo> fifo;
  Eigen::Vector3d tau_applied = Eigen::Vector3d::Zero();
  integrate::Rk4 rk4_att{models::kAttitudeStateDim};
  int nav_n = 0;
  int innov_mm = 0;
  std::vector<double> x_hat_buf;
  std::vector<double> p_buf;
  std::vector<double> e_buf;
  std::vector<double> innov_y_buf;
  std::vector<double> innov_s_buf;
  // Non-owning views of the "external" component in whichever chain slots
  // the mission put it, so the stepping API can hand it a command. Null when
  // the mission configured no external slot.
  gnc::ExternalCommand* ext_guidance = nullptr;
  gnc::ExternalCommand* ext_control = nullptr;

  // --- FR-24 observation snapshot -----------------------------------------
  // Refreshed once per processed cycle and never on read, which is what
  // makes observe() idempotent (exit criterion 4). Truth is stored beside it
  // rather than inside it so the privileged accessor stays separate.
  CycleObservation obs;
  gnc::TruthState obs_truth;

  Impl(const RunConfig& c, const std::string& out_path)
      : cfg(c),
        env(make_env_spec(c)),
        nav(c.gnc.enabled ? gnc::make_component(c.gnc.nav) : nullptr),
        guidance(c.gnc.enabled ? gnc::make_component(c.gnc.guidance)
                               : nullptr),
        control(c.gnc.enabled ? gnc::make_component(c.gnc.control) : nullptr),
        nav_layout(capture_error_layout(c, nav.get())),
        writer(out_path,
               build_header_fields(c, nav ? nav->state_dim() : 0,
                                   nav ? nav->cov_dim() : 0,
                                   nav ? nav->innov_max_dim() : 0,
                                   !nav_layout.empty())) {
    central = central_body_from_name(cfg.central_body);
    mu = models::central_body_gm(central);
    earth = central == models::CentralBody::kEarth;
    if (central == models::CentralBody::kMars) {
      planet_a_m = constants::MARS_ELLIPSOID_A_M;
      planet_inv_f = constants::MARS_ELLIPSOID_INV_F;
    } else if (central == models::CentralBody::kMoon) {
      planet_a_m = constants::R_MOON_M;
      planet_inv_f = 1.0e12;  // Moon treated spherical for altitude reference
    }
    epoch = {cfg.epoch_tai_day, cfg.epoch_tai_sec};
    veh = build_vehicle(cfg.vehicle);

    // A Moon ephemeris is loaded only when a soi_transition to the Moon is
    // in the sequence (the geometry needs the Moon's geocentric position).
    bool need_soi_moon = false;
    for (const SequenceEntry& e : cfg.sequence) {
      if (e.trigger == "condition" && e.condition == "soi_transition" &&
          e.body == "moon") {
        need_soi_moon = true;
      }
    }
    if (need_soi_moon) {
      soi_eph.emplace(Ephemeris::load_file(cfg.ephemeris_path));
    }

    if (cfg.forces_rate_hz != 0) {
      sources.push_back("gravity");
      if (!cfg.third_bodies.empty()) sources.push_back("thirdbody");
      if (!veh.aero.empty()) sources.push_back("aero");
      sources.push_back("thrust");
      sources.push_back("gravgrad");
    }

    // --- initial state, attitude, and control mode ------------------------
    if (cfg.initial_form == "geodetic") {
      if (!earth) {
        throw std::invalid_argument(
            "run_vehicle: geodetic launch requires central_body earth");
      }
      pad_lat = deg2rad_v(cfg.launch_lat_deg);
      pad_lon = deg2rad_v(cfg.launch_lon_deg);
      pad_alt = cfg.launch_alt_m;
      const Eigen::Matrix3d c_itrf0 = frames::c_gcrf_to_itrf(epoch, 0.0);
      const models::PadState pad = models::geodetic_pad_state(
          pad_lat, pad_lon, pad_alt, c_itrf0, constants::OMEGA_EARTH_RAD_PER_S,
          constants::WGS84_A_M, constants::WGS84_INV_F);
      up0 = pad.up_i;
      east0 = pad.east_i;
      north0 = pad.north_i;
      r_m = pad.r_i_m;
      v_mps = pad.v_i_mps;
      q = models::attitude_from_body_x(up0, north0);
      q_hold = q;
      released = false;
      mode = AttMode::kPadFixed;
    } else {
      r_m = arr3(cfg.r0_m);
      v_mps = arr3(cfg.v0_mps);
      const Eigen::Vector3d vhat =
          v_mps.norm() > 0.0 ? v_mps.normalized() : Eigen::Vector3d::UnitX();
      q = models::attitude_from_body_x(vhat, r_m);  // body +X prograde
      q_hold = q;
      mode = AttMode::kInertialHold;
    }

    steps = std::llround(cfg.duration_s / cfg.dt_s);
    dt = cfg.dt_s;
    truth_decim =
        std::llround(1.0 / (dt * static_cast<double>(cfg.truth_rate_hz)));
    forces_decim =
        cfg.forces_rate_hz ? std::llround(1.0 / (dt * cfg.forces_rate_hz)) : 0;
    mass_decim =
        cfg.mass_rate_hz ? std::llround(1.0 / (dt * cfg.mass_rate_hz)) : 0;
    env_decim =
        cfg.env_rate_hz ? std::llround(1.0 / (dt * cfg.env_rate_hz)) : 0;

    writer.write_event(0.0, 1, "run_start");
    run_summary.event_records += 1;

    fired.assign(cfg.sequence.size(), 0);

    ctx.v = &veh;
    ctx.earth = earth;
    ctx.planet_a_m = planet_a_m;
    ctx.planet_inv_f = planet_inv_f;

    // One reusable buffer for the whole run: the per-source forces record is
    // rebuilt in place each logged step (clear() retains the capacity) rather
    // than allocating and freeing a fresh vector every cycle. On a
    // high-volume log (order 10^5 records) removing this per-cycle allocation
    // churn is what makes the by-source write path reliable on the build
    // host; the logged bytes are identical either way (see
    // docs/KNOWN_ISSUES.md, KNOWN-ISSUE-P4-1).
    samples.reserve(sources.size());

    // --- GNC construction: components, sensors, logging buffers -----------
    if (cfg.gnc.enabled) {
      nav_n = nav->state_dim();
      innov_mm = nav->innov_max_dim();
      x_hat_buf.assign(static_cast<std::size_t>(nav_n), 0.0);
      const std::size_t nav_m = static_cast<std::size_t>(nav->cov_dim());
      p_buf.assign(nav_m * (nav_m + 1) / 2, 0.0);
      e_buf.assign(static_cast<std::size_t>(nav_n), 0.0);
      innov_y_buf.assign(static_cast<std::size_t>(innov_mm), 0.0);
      innov_s_buf.assign(static_cast<std::size_t>(innov_mm) *
                             (static_cast<std::size_t>(innov_mm) + 1) / 2,
                         0.0);
      for (const gnc::GncSensorCfg& s : cfg.gnc.sensors) {
        sensor_list.push_back(sensors::make_sensor(s, cfg.master_seed));
        sensor_decim.push_back(static_cast<std::int64_t>(
            cfg.gnc.control_rate_hz / s.sample_rate_hz));
        const std::uint32_t id =
            static_cast<std::uint32_t>(sensor_list.size() - 1);
        if (s.kind == "imu") {
          imu = static_cast<sensors::Imu*>(sensor_list.back().get());
          // The filter's process model is the configured instrument's error
          // model (eq:ekf:G): the random-walk coefficients drive Q's
          // attitude/velocity blocks and the Gauss-Markov pair drives its
          // bias blocks, through the same preset mapping the sensor uses.
          const sensors::ImuErrorCfg e = sensors::parse_imu_error_cfg(s);
          sensor_model.imu_present = true;
          sensor_model.imu_id = id;
          sensor_model.gyro_arw = e.gyro.random_walk;
          sensor_model.accel_vrw = e.accel.random_walk;
          sensor_model.gyro_gm_sigma =
              sensors::gm_sigma_from_bias_instability(e.gyro.bias_instability);
          sensor_model.gyro_tau_s = e.gyro.bias_tau_s;
          sensor_model.accel_gm_sigma = sensors::gm_sigma_from_bias_instability(
              e.accel.bias_instability);
          sensor_model.accel_tau_s = e.accel.bias_tau_s;
        } else {
          // Every FR-23 kind except the IMU measures against ephemeris,
          // shadow, or body-fixed geometry.
          needs_geometry_ = true;
          if (s.kind == "navfix") {
            navfix = static_cast<sensors::NavFix*>(sensor_list.back().get());
            navfix_id = id;
            const sensors::NavFixCfg navfix_cfg = sensors::parse_nav_fix_cfg(s);
            sensor_model.navfix_present = true;
            sensor_model.navfix_id = id;
            sensor_model.navfix_sigma_r_m = navfix_cfg.sigma_r_m;
            sensor_model.navfix_sigma_v_mps = navfix_cfg.sigma_v_mps;
          } else if (s.kind == "startracker") {
            startracker =
                static_cast<sensors::StarTracker*>(sensor_list.back().get());
            startracker_id = id;
            const sensors::StarTrackerCfg startracker_cfg =
                sensors::parse_star_tracker_cfg(s);
            sensor_model.startracker_present = true;
            sensor_model.startracker_id = id;
            sensor_model.startracker_sigma_rad = startracker_cfg.sigma_rad;
            sensor_model.startracker_boresight_b = startracker_cfg.boresight_b;
          } else if (s.kind == "altimeter") {
            altimeter =
                static_cast<sensors::Altimeter*>(sensor_list.back().get());
            altimeter_id = id;
            const sensors::AltimeterCfg altimeter_cfg =
                sensors::parse_altimeter_cfg(s);
            sensor_model.altimeter_present = true;
            sensor_model.altimeter_id = id;
            sensor_model.altimeter_sigma_noise_m = altimeter_cfg.sigma_noise_m;
            sensor_model.altimeter_sigma_bias_m = altimeter_cfg.sigma_bias_m;
          }
        }
      }
      // Locate the stepping API's command addressee. dynamic_cast rather
      // than a name comparison against the config: the pointer must be the
      // object the chain will actually call, and deriving it from the object
      // itself cannot disagree with what was constructed.
      ext_guidance = dynamic_cast<gnc::ExternalCommand*>(guidance.get());
      ext_control = dynamic_cast<gnc::ExternalCommand*>(control.get());
      // Free-flying missions close the loop from t = 0; geodetic missions
      // activate at pad release (format doc section 3.2 record-start
      // semantics).
      if (cfg.initial_form != "geodetic") {
        activate_gnc(0);
      }
    }
    // Seed the FR-24 snapshot so observe() before the first step() describes
    // the initial state rather than a default-constructed blank. The stack
    // composition is the same pure function the cycle uses, so the reported
    // initial mass is the one cycle 0 will see.
    sp = compose_stack(veh);
    refresh_observation(0, 0.0, false);
  }

  // Overwrite the FR-24 snapshot with the state of cycle k at time t. Called
  // at the end of every processed cycle and once at construction; NEVER from
  // an accessor, which is what keeps observe() idempotent (exit criterion
  // 4). Only the non-GNC fields are written here - the GNC block fills its
  // own fields earlier in the same cycle, before the chain products are
  // overwritten by the next one.
  void refresh_observation(std::int64_t k, double t, bool processed) {
    obs.cycle = k;
    obs.t_s = t;
    obs.processed = processed;
    obs.done = finished;
    obs.gnc_active = gnc_active;
    if (!gnc_active) {
      // Clear the chain fields rather than leaving the last active cycle's
      // values in place: a driver reading them after GNC deactivation would
      // otherwise see a stale command that nothing is applying.
      obs.imu = gnc::ImuSample();
      obs.imu_fresh = false;
      obs.navfix = gnc::NavFixSample();
      obs.startracker = gnc::StarTrackerSample();
      obs.altimeter = gnc::AltimeterSample();
      obs.env = gnc::NavEnvironment();
      obs.nav_est = gnc::GncOutput();
      obs.att_cmd = gnc::GncOutput();
      obs.applied = gnc::GncOutput();
      obs.nav_x_hat.clear();
      obs.nav_p_upper.clear();
    }
    obs_truth.valid = true;
    obs_truth.t_s = t;
    obs_truth.r_i_m = r_m;
    obs_truth.v_i_mps = v_mps;
    obs_truth.q_i2b = q;
    obs_truth.omega_b_radps = omega_b;
    obs_truth.mass_kg = sp.composite.mass_kg;
    obs_truth.imu_bias_valid = imu != nullptr;
    if (imu != nullptr) {
      obs_truth.b_g_radps = imu->gyro_total_bias_radps();
      obs_truth.b_a_mps2 = imu->accel_total_bias_mps2();
    }
  }

  double t_of(std::int64_t k) const { return static_cast<double>(k) * dt; }

  // Switch the loop into closed-loop authority at cycle k: initialize the
  // chain from the current attitude state, seed the latency FIFO with the
  // neutral (zero-torque, current-attitude) command, and mark the
  // activation cycle that anchors sensor sample instants.
  void activate_gnc(std::int64_t k) {
    gnc_active = true;
    act_cycle = k;
    mode = AttMode::kGnc;
    gnc::GncInitContext ictx;
    ictx.t0_s = t_of(k);
    ictx.q0_i2b = q;
    ictx.omega0_b_radps = omega_b;
    ictx.pad_basis_valid = cfg.initial_form == "geodetic";
    ictx.up_i = up0;
    ictx.east_i = east0;
    ictx.north_i = north0;
    ictx.control_rate_hz = cfg.gnc.control_rate_hz;
    ictx.dt_s = dt;
    // Central-body constants an estimator needs for its own dynamics and
    // measurement models (eq:ekf:mech, eq:ekf:altH).
    ictx.mu_m3ps2 = mu;
    ictx.ellipsoid_a_m = planet_a_m;
    ictx.ellipsoid_inv_f = planet_inv_f;
    ictx.sensors = sensor_model;
    nav->init(ictx);
    guidance->init(ictx);
    control->init(ictx);
    gnc::GncOutput neutral;
    neutral.q_i2b = q;
    neutral.omega_b_radps = Eigen::Vector3d::Zero();
    neutral.torque_b_nm = Eigen::Vector3d::Zero();
    fifo.emplace(cfg.gnc.latency_cycles, neutral);
    tau_applied = Eigen::Vector3d::Zero();
  }

  // Run the GNC block for cycle k at time t: sample due sensors, run the
  // chain nav -> guidance -> control, delay through the latency FIFO, and
  // log gnc.cmd / nav.est / nav.err / nav.innov. The applied torque is
  // stashed for advance_cycle's attitude integration.
  void run_gnc_block(std::int64_t k, double t) {
    const std::int64_t n_act = k - act_cycle;
    bool imu_fresh = false;
    bool navfix_fresh = false;
    bool startracker_fresh = false;
    bool altimeter_fresh = false;
    for (std::size_t si = 0; si < sensor_list.size(); ++si) {
      if (n_act > 0 && n_act % sensor_decim[si] == 0) {
        sensors::ISensor* s = sensor_list[si].get();
        s->sample(t, writer);
        // Freshness is per sensor: an estimator must fold a measurement in
        // exactly once, on the cycle it was produced. Reprocessing a held
        // sample would make the filter overconfident in a way that is
        // invisible in the state error but shows up immediately in NEES.
        if (s == imu) imu_fresh = true;
        if (s == navfix) navfix_fresh = true;
        if (s == startracker) startracker_fresh = true;
        if (s == altimeter) altimeter_fresh = true;
      }
    }

    gnc::GncInput in;
    in.cycle = n_act;
    in.t_s = t;
    in.dt_s = dt;
    if (imu != nullptr) in.imu = imu->last_sample();
    in.imu_fresh = imu_fresh;
    in.prev_applied = fifo->applied();
    if (navfix != nullptr) {
      in.navfix.valid = navfix->last_valid();
      in.navfix.fresh = navfix_fresh;
      in.navfix.sensor_id = navfix_id;
      in.navfix.r_i_m = navfix->last_position_m();
      in.navfix.v_i_mps = navfix->last_velocity_mps();
    }
    if (startracker != nullptr) {
      in.startracker.valid = startracker->last_valid();
      in.startracker.fresh = startracker_fresh;
      in.startracker.sensor_id = startracker_id;
      in.startracker.q_i2b = startracker->last_measurement();
    }
    if (altimeter != nullptr) {
      in.altimeter.valid = altimeter->last_valid();
      in.altimeter.fresh = altimeter_fresh;
      in.altimeter.sensor_id = altimeter_id;
      in.altimeter.h_m = altimeter->last_measurement_m();
    }
    // Ephemeris- and frame-derived context only: a navigator may compute
    // these onboard from time alone, so supplying them crosses no privileged
    // boundary (unlike GncInput.oracle, which is gated on the oracle flag).
    in.env.ephemeris_valid = last_geom.ephemeris_valid;
    in.env.v_central_ssb_mps = last_geom.v_central_ssb_mps;
    in.env.bodyfixed_valid = last_geom.bodyfixed_valid;
    in.env.c_gcrf_to_bodyfixed = last_geom.c_gcrf_to_bodyfixed;
    if (cfg.oracle) {
      // FR-25 privileged boundary: truth enters GncInput if and only if the
      // scenario set oracle = true (already stamped in the header).
      in.oracle.valid = true;
      in.oracle.t_s = t;
      in.oracle.r_i_m = r_m;
      in.oracle.v_i_mps = v_mps;
      in.oracle.q_i2b = q;
      in.oracle.omega_b_radps = omega_b;
      in.oracle.mass_kg = sp.composite.mass_kg;
    }
    in.nav_est = nav->update(in);
    in.att_cmd = guidance->update(in);
    const gnc::GncOutput produced = control->update(in);
    const gnc::GncOutput applied = fifo->push(produced);
    tau_applied = applied.torque_b_nm;

    const double q_cmd[4] = {applied.q_i2b.w(), applied.q_i2b.x(),
                             applied.q_i2b.y(), applied.q_i2b.z()};
    writer.write_gnc_cmd(t, applied.torque_b_nm, q_cmd,
                         applied.omega_b_radps, applied.valid ? 1u : 0u);

    // FR-24 snapshot of this cycle's chain view. Copied out of the same `in`
    // the components were handed, so what a stepping driver observes is
    // exactly what the nav stage saw - not a reconstruction that could drift
    // from it.
    obs.imu = in.imu;
    obs.imu_fresh = in.imu_fresh;
    obs.navfix = in.navfix;
    obs.startracker = in.startracker;
    obs.altimeter = in.altimeter;
    obs.env = in.env;
    obs.nav_est = in.nav_est;
    obs.att_cmd = in.att_cmd;
    obs.applied = applied;

    if (nav_n > 0) {
      nav->state(x_hat_buf.data());
      nav->covariance_upper(p_buf.data());
      writer.write_nav_est(t, x_hat_buf.data(), x_hat_buf.size(),
                           p_buf.data(), p_buf.size());
      // Copy, not a view: observe() must never hand out a pointer into a
      // buffer the next cycle overwrites (exit criterion 4's idempotence
      // clause is about the returned value, not just the call).
      obs.nav_x_hat = x_hat_buf;
      obs.nav_p_upper = p_buf;
      if (!nav_layout.empty()) {
        // nav.err is computed HERE, by the loop, from the layout the
        // estimator declared and the state vector it just published - the
        // truth state below never crosses the plugin boundary (FR-24, and
        // the descriptor commentary in gnc/component.hpp). Differencing
        // against x_hat_buf rather than against a second read of the
        // component also makes nav.err provably the error of the nav.est
        // record written on the same cycle.
        gnc::TruthState truth;
        truth.valid = true;
        truth.t_s = t;
        truth.r_i_m = r_m;
        truth.v_i_mps = v_mps;
        truth.q_i2b = q;
        truth.omega_b_radps = omega_b;
        truth.mass_kg = sp.composite.mass_kg;
        if (imu != nullptr) {
          // An estimator carrying bias states needs the true biases to
          // report a complete error; a layout that declares bias blocks
          // without a configured IMU was already refused at construction.
          truth.imu_bias_valid = true;
          truth.b_g_radps = imu->gyro_total_bias_radps();
          truth.b_a_mps2 = imu->accel_total_bias_mps2();
        }
        gnc::compute_error_state(nav_layout, truth, x_hat_buf.data(),
                                 e_buf.data());
        writer.write_nav_err(t, e_buf.data(), e_buf.size());
      }
    }
    if (innov_mm > 0) {
      for (const gnc::InnovationSample& s : nav->innovations()) {
        // Bound the copies below against the DESTINATION's own capacity, not
        // against anything the component reports now. Both buffers were
        // sized once at activation from innov_max_dim(), and a component --
        // in particular an FR-25 Python one, which supplies both the
        // declaration and the payload -- can return a sample wider than the
        // maximum it declared. Unchecked, that is a heap write past the end
        // of innov_y_buf on the first aiding update. Every other
        // variable-length quantity crossing this boundary is length-checked
        // by name (copy_fixed, validate_error_layout); this closes the gap.
        const std::size_t y_cap = innov_y_buf.size();
        if (s.y.size() > y_cap) {
          throw std::length_error(
              "gnc component returned an innovation of dimension " +
              std::to_string(s.y.size()) +
              "; the maximum it declared through innov_max_dim() is " +
              std::to_string(y_cap));
        }
        // The packed triangle is checked against the sample's OWN dimension:
        // the embedding loop below reads exactly m(m+1)/2 entries from it, so
        // a short s_upper is an out-of-bounds read that would otherwise
        // publish uninitialized heap into the nav.innov channel.
        const std::size_t s_need = s.y.size() * (s.y.size() + 1) / 2;
        if (s.s_upper.size() != s_need) {
          throw std::length_error(
              "gnc component returned an innovation covariance of " +
              std::to_string(s.s_upper.size()) + " packed entries for a " +
              std::to_string(s.y.size()) + "-dimensional innovation; the " +
              "packed upper triangle requires exactly " +
              std::to_string(s_need));
        }
        // Zero-pad each update to the declared maximum dimension (format
        // doc section 3.2 fixed-stride rule).
        std::fill(innov_y_buf.begin(), innov_y_buf.end(), 0.0);
        std::fill(innov_s_buf.begin(), innov_s_buf.end(), 0.0);
        const std::uint32_t m = static_cast<std::uint32_t>(s.y.size());
        std::copy(s.y.begin(), s.y.end(), innov_y_buf.begin());
        // S must be padded STRUCTURALLY, not by a flat copy: the packed
        // upper triangle of an m-by-m block and that of an m_max-by-m_max
        // matrix have different row strides, so copying the short triangle
        // into the front of the long buffer would scatter the block across
        // the first row instead of embedding it in the leading corner. The
        // format doc's rule - entries whose row or column exceeds m are
        // zero - is what this loop reproduces, one row at a time.
        {
          const std::size_t mm = static_cast<std::size_t>(innov_mm);
          std::size_t src = 0;
          std::size_t row0 = 0;  // start of the current row in the m_max triangle
          for (std::size_t row = 0; row < m; ++row) {
            for (std::size_t col = row; col < m; ++col) {
              innov_s_buf[row0 + (col - row)] = s.s_upper[src++];
            }
            row0 += mm - row;  // row `row` holds m_max - row entries
          }
        }
        writer.write_nav_innov(t, s.sensor_id, m, innov_y_buf.data(),
                               innov_y_buf.size(), innov_s_buf.data(),
                               innov_s_buf.size());
      }
    }
  }

  // The Phase 4 loop body for cycle k (verbatim from run.cpp), plus the GNC
  // block. Sets `stop` when a terminal event fired.
  void process_cycle(std::int64_t k) {
    const double t = t_of(k);
    const time::TaiEpoch tai = time::tai_add_seconds(epoch, t);
    Eigen::Matrix3d c_itrf = Eigen::Matrix3d::Identity();
    if (earth) c_itrf = frames::c_gcrf_to_itrf(tai, 0.0);

    // -- sequence: fire due entries, in file order -------------------------
    for (std::size_t sk = 0; sk < cfg.sequence.size(); ++sk) {
      if (fired[sk]) continue;
      const SequenceEntry& e = cfg.sequence[sk];
      bool go = false;
      if (e.trigger == "elapsed") {
        go = t >= e.t_s;
      } else if (e.trigger == "after_event") {
        auto it = fire_time.find(e.event);
        go = it != fire_time.end() && t >= it->second + e.offset_s;
      } else {  // condition
        if (e.condition == "altitude_above" || e.condition == "altitude_below") {
          double alt = r_m.norm() - planet_a_m;
          if (earth) alt = models::geodetic_altitude(c_itrf * r_m,
                                                     planet_a_m, planet_inv_f);
          go = (e.condition == "altitude_above") ? alt >= e.altitude_m
                                                 : alt <= e.altitude_m;
        } else if (e.condition == "perigee_above") {
          go = released && osc_perigee_alt(mu, r_m, v_mps, planet_a_m) >=
                               e.perigee_alt_m;
        } else if (e.condition == "soi_transition" && soi_eph.has_value()) {
          const double tdb = tdb_s_at(epoch, t);
          const Eigen::Vector3d r_moon = soi_eph->moon_geocentric(tdb).r_m;
          const double soi = r_moon.norm() *
                             std::pow(constants::GM_MOON_DE440_M3_PER_S2 /
                                          constants::GM_EARTH_M3_PER_S2,
                                      0.4);
          go = (r_m - r_moon).norm() < soi;
        }
      }
      if (!go) continue;
      fired[sk] = 1;
      fire_time[e.name] = t;

      // -- apply the action ------------------------------------------------
      if (e.action == "pad_release") {
        released = true;
        q_hold = q;
        if (cfg.gnc.enabled) {
          // Closed-loop authority begins at release: the pad clamp ends and
          // the chain initializes from the release attitude state (for
          // free-flying missions activation happened at cycle 0 with the
          // scenario initial attitude).
          activate_gnc(k);
        } else {
          mode = AttMode::kInertialHold;  // held until a pitch/rate action
        }
      } else if (e.action == "ignite_engine" || e.action == "cutoff_engine") {
        for (EngineRt& en : veh.engines) {
          if (veh.stages[static_cast<std::size_t>(en.stage_idx)].name ==
                  e.stage &&
              en.name == e.engine) {
            en.command.run = (e.action == "ignite_engine");
            if (en.command.run) en.command.throttle = en.params.throttle_max;
          }
        }
      } else if (e.action == "separate_stage" || e.action == "jettison") {
        const Eigen::Vector3d cg_old = compose_stack(veh).composite.cg_m;
        if (e.action == "separate_stage") {
          for (StageRt& s : veh.stages) {
            if (s.name == e.stage) {
              s.attached = false;
              for (int j : s.jett) {
                veh.jett[static_cast<std::size_t>(j)].attached = false;
              }
              for (int ei : s.engines) {
                veh.engines[static_cast<std::size_t>(ei)].command.run = false;
              }
            }
          }
          ++aero_sep_count;
        } else {
          for (JettRt& jt : veh.jett) {
            if (jt.name == e.item &&
                veh.stages[static_cast<std::size_t>(jt.stage_idx)].name ==
                    e.stage) {
              jt.attached = false;
            }
          }
        }
        const Eigen::Vector3d cg_new = compose_stack(veh).composite.cg_m;
        const models::SeparationRemap rm = models::separation_remap(
            cg_old, cg_new, r_m, v_mps, q, omega_b);
        r_m = rm.r_new_i_m;
        v_mps = rm.v_new_i_mps;
      } else if (e.action == "pitch_program") {
        mode = AttMode::kPitchProgram;
        pitch_az_deg = e.azimuth_deg;
        pitch_t = e.pitch_t_s;
        pitch_d = e.pitch_deg;
      } else if (e.action == "attitude_hold") {
        mode = AttMode::kInertialHold;
        q_hold = q;
      } else if (e.action == "prograde_hold") {
        mode = AttMode::kProgradeHold;
      } else if (e.action == "rate_command") {
        mode = AttMode::kRateCommand;
        rate_frame_gcrf = e.frame == "gcrf";
        rate_omega_b = arr3(e.omega_dps) * (constants::TWO_PI / 360.0);
        q_hold = q;
      } else if (e.action == "terminate") {
        stop = true;
      }
      // Event detail names what fired: for a condition trigger the condition
      // leads (so e.g. the orbit-insertion and SOI-transition terminal events
      // are self-describing in the log), otherwise the action leads.
      const std::string detail = (e.trigger == "condition")
                                     ? (e.condition + ":" + e.name)
                                     : (e.action + ":" + e.name);
      writer.write_event(t, 3, detail);
      run_summary.event_records += 1;
    }

    // -- mass properties for this cycle ------------------------------------
    sp = compose_stack(veh);

    // -- attitude command for this cycle -----------------------------------
    Eigen::Vector3d omega_i_planet = Eigen::Vector3d::Zero();
    if (earth) {
      omega_i_planet =
          constants::OMEGA_EARTH_RAD_PER_S * c_itrf.row(2).transpose();
    }
    if (mode == AttMode::kPadFixed) {
      const models::PadState pad = models::geodetic_pad_state(
          pad_lat, pad_lon, pad_alt, c_itrf, constants::OMEGA_EARTH_RAD_PER_S,
          constants::WGS84_A_M, constants::WGS84_INV_F);
      if (!released) {
        r_m = pad.r_i_m;  // clamped to the co-rotating pad until release
        v_mps = pad.v_i_mps;
      }
      q = models::attitude_from_body_x(pad.up_i, pad.north_i);
      // The pad co-rotates with the planet: the logged body rate is the
      // planet spin resolved in body axes.
      omega_b = rotation::dcm_from_quat(q) * omega_i_planet;
    } else if (mode == AttMode::kPitchProgram) {
      const double az = deg2rad_v(pitch_az_deg);
      const double p0 =
          deg2rad_v(models::pwl_interp_clamped(pitch_t, pitch_d, t));
      const double p1 =
          deg2rad_v(models::pwl_interp_clamped(pitch_t, pitch_d, t + dt));
      const Eigen::Quaterniond q0 = models::attitude_from_body_x(
          models::pitch_program_axis(az, p0, up0, east0, north0),
          models::pitch_program_roll_ref(az, p0, up0, east0, north0));
      const Eigen::Quaterniond q1 = models::attitude_from_body_x(
          models::pitch_program_axis(az, p1, up0, east0, north0),
          models::pitch_program_roll_ref(az, p1, up0, east0, north0));
      q = q0;
      omega_b = models::omega_from_quaternions(q0, q1, dt);
    } else if (mode == AttMode::kRateCommand) {
      q = q_hold;
      omega_b = rate_frame_gcrf ? rotation::dcm_from_quat(q) * rate_omega_b
                                : rate_omega_b;
    } else if (mode == AttMode::kProgradeHold) {
      // Velocity-pointing open-loop steering: body +X tracks the current
      // inertial velocity each control cycle, so a finite burn stays prograde
      // (the fixed-attitude alternative loses cos(theta) of the burn as the
      // velocity rotates ~orbital-rate through a long burn).
      const Eigen::Vector3d vh =
          v_mps.norm() > 0.0 ? v_mps.normalized() : Eigen::Vector3d::UnitX();
      q = models::attitude_from_body_x(vh, r_m);
      omega_b = Eigen::Vector3d::Zero();
    } else if (mode == AttMode::kGnc) {
      // Closed-loop authority: q/omega_b are the dynamically integrated
      // attitude state carried by advance_cycle; nothing to command here.
    } else {  // inertial hold
      q = q_hold;
      omega_b = Eigen::Vector3d::Zero();
    }

    ctx.c_i2b = rotation::dcm_from_quat(q);
    ctx.c_b2i = ctx.c_i2b.transpose();
    ctx.c_gcrf_to_itrf = c_itrf;
    ctx.omega_planet_i = omega_i_planet;
    ctx.omega_b = omega_b;
    ctx.cg_b = sp.composite.cg_m;
    ctx.mass_kg = sp.composite.mass_kg;
    ctx.aero_idx =
        veh.aero.empty()
            ? -1
            : std::min<int>(aero_sep_count,
                            static_cast<int>(veh.aero.size()) - 1);

    // -- GNC block (sensors, chain, latency, gnc.cmd/nav.* records) --------
    if (gnc_active) {
      run_gnc_block(k, t);
    }

    const bool last = (k == steps) || stop;

    // -- logging (decimated from the truth grid) ---------------------------
    if (k % truth_decim == 0 || last) {
      const double q_arr[4] = {q.w(), q.x(), q.y(), q.z()};
      writer.write_truth(t, r_m, v_mps, q_arr, omega_b, sp.composite.mass_kg);
      run_summary.truth_records += 1;
    }
    bl = eval_body_loads(ctx, r_m, v_mps);
    if (forces_decim && (k % forces_decim == 0 || last)) {
      const Eigen::Vector3d a_env = env.acceleration(t, r_m, v_mps);
      const Eigen::Vector3d a_grav = models::twobody_accel(mu, r_m);
      const Eigen::Vector3d f_grav_b =
          ctx.c_i2b * (sp.composite.mass_kg * a_grav);
      const Eigen::Vector3d f_pert_b =
          ctx.c_i2b * (sp.composite.mass_kg * (a_env - a_grav));
      const Eigen::Vector3d tq_gg = models::gravgrad_torque(
          mu, r_m, q, sp.composite.inertia_kgm2);
      samples.clear();
      for (const std::string& src : sources) {
        // Eigen vectors are not zero-initialized by their default constructor;
        // zeroing both channels keeps a source that produces only a force or
        // only a torque deterministic (D-10) rather than logging stack garbage.
        log::ForceSourceSample s;
        s.force_b_n = Eigen::Vector3d::Zero();
        s.torque_b_nm = Eigen::Vector3d::Zero();
        if (src == "gravity") {
          s.force_b_n = f_grav_b;
        } else if (src == "thirdbody") {
          s.force_b_n = f_pert_b;
        } else if (src == "aero") {
          s.force_b_n = bl.f_aero;
          s.torque_b_nm = bl.tq_aero;
        } else if (src == "thrust") {
          s.force_b_n = bl.f_thrust;
          s.torque_b_nm = bl.tq_thrust;
        } else if (src == "gravgrad") {
          s.torque_b_nm = tq_gg;
        }
        samples.push_back(s);
      }
      writer.write_forces(t, samples);
    }
    if (mass_decim && (k % mass_decim == 0 || last)) {
      const Eigen::Matrix3d& I = sp.composite.inertia_kgm2;
      const double packed[6] = {I(0, 0), I(0, 1), I(0, 2),
                                I(1, 1), I(1, 2), I(2, 2)};
      writer.write_mass(t, sp.composite.mass_kg, sp.composite.cg_m, packed);
    }
    if (env_decim && (k % env_decim == 0 || last)) {
      const double rn = r_m.norm();
      const double vn = v_mps.norm();
      double fpa = 0.0;
      if (rn > 0.0 && vn > 0.0) {
        double s = r_m.dot(v_mps) / (rn * vn);
        s = std::max(-1.0, std::min(1.0, s));
        fpa = std::asin(s);
      }
      writer.write_env(t, bl.alt_m, bl.mach, bl.q_bar_pa, bl.rho, fpa);
    }
  }

  // Advance the continuous and discrete state from cycle k to k + 1
  // (verbatim Phase 4 semantics plus sensor accumulation and torque-driven
  // attitude integration in the GNC mode).
  void advance_cycle(std::int64_t k) {
    const double t = t_of(k);
    if (gnc_active) {
      // Cycle-start endpoint values for the trapezoidal sensor truth
      // (eq:imu:quadrature), captured before the in-place translational
      // step overwrites the state; consumed after the attitude step below.
      r_start_ = r_m;
      v_start_ = v_mps;
      sf_start_ = specific_force(t, r_start_, v_start_);
    }
    if (released) {
      auto rhs = [this](double tt, const double* yin, double* ydot) {
        const Eigen::Map<const Eigen::Vector3d> r(yin);
        const Eigen::Map<const Eigen::Vector3d> vv(yin + 3);
        const Eigen::Vector3d a_env = env.acceleration(tt, r, vv);
        const BodyLoads loads = eval_body_loads(ctx, r, vv);
        const Eigen::Vector3d f_body = loads.f_thrust + loads.f_aero;
        const Eigen::Vector3d a = models::composed_translational_accel(
            a_env, f_body, ctx.c_b2i, ctx.mass_kg);
        ydot[0] = vv[0];
        ydot[1] = vv[1];
        ydot[2] = vv[2];
        ydot[3] = a[0];
        ydot[4] = a[1];
        ydot[5] = a[2];
      };
      const integrate::RhsRef f(rhs);
      rk4.step(f, t, y, dt, y);
      run_summary.steps += 1;
    }
    // Deplete tanks at the ZOH throttle level held over the cycle, then
    // advance each engine's spool/gimbal/ignition state for the next cycle.
    for (EngineRt& en : veh.engines) {
      if (!veh.stages[static_cast<std::size_t>(en.stage_idx)].attached) {
        continue;
      }
      if (en.tank_idx >= 0 && en.state.throttle_level > 0.0) {
        TankRt& tk = veh.tanks[static_cast<std::size_t>(en.tank_idx)];
        if (tk.mass_kg > 0.0) {
          const double dm =
              models::engine_mdot_kgps(en.params, en.state.throttle_level) * dt;
          tk.mass_kg = std::max(0.0, tk.mass_kg - dm);
        }
      }
    }
    for (EngineRt& en : veh.engines) {
      en.state = models::engine_advance(en.params, en.command, en.state, dt);
    }
    if (mode == AttMode::kRateCommand) {
      // Integrate the commanded body rate to carry q into the next cycle:
      // q_{k+1} = q_k (x) dq, dq the body-frame rotation by omega_b dt
      // (frame-transformation composition, ch:rotations).
      const Eigen::Vector3d w_b =
          rate_frame_gcrf ? (rotation::dcm_from_quat(q_hold) * rate_omega_b)
                          : rate_omega_b;
      const double ang = w_b.norm() * dt;
      Eigen::Quaterniond dq = Eigen::Quaterniond::Identity();
      if (ang > 0.0) {
        const Eigen::Vector3d ax = w_b.normalized();
        const double s = std::sin(0.5 * ang);
        dq = Eigen::Quaterniond(std::cos(0.5 * ang), s * ax.x(), s * ax.y(),
                                s * ax.z());
      }
      q_hold = rotation::quat_normalize(rotation::quat_multiply(q_hold, dq));
    }
    if (gnc_active) {
      // Torque-driven attitude dynamics (Phase 6): integrate Euler's
      // equations and the quaternion kinematics over the cycle under the
      // applied command torque, with the composite inertia held constant
      // (ZOH on mass properties, Idot = 0 within the cycle - the
      // depletion-rate coupling into attitude lands with the full torque
      // composition of a later phase; format doc section 3.2 authority
      // scoping). One RK4 step per control cycle on the packed 7-state
      // attitude slice, then the FR-1 post-step renormalization.
      const Eigen::Quaterniond q_start = q;
      const Eigen::Vector3d omega_start = omega_b;
      double y_att[models::kAttitudeStateDim] = {
          q.w(), q.x(), q.y(), q.z(), omega_b[0], omega_b[1], omega_b[2]};
      const Eigen::Matrix3d inertia = sp.composite.inertia_kgm2;
      const Eigen::Matrix3d inertia_dot = Eigen::Matrix3d::Zero();
      const Eigen::Vector3d tau = tau_applied;
      auto rhs_att = [&inertia, &inertia_dot, &tau](double, const double* ya,
                                                    double* yd) {
        models::rigidbody_rhs(ya, inertia, inertia_dot, tau, yd);
      };
      const integrate::RhsRef fa(rhs_att);
      rk4_att.step(fa, t, y_att, dt, y_att);
      models::rigidbody_renormalize(y_att);
      q = Eigen::Quaterniond(y_att[0], y_att[1], y_att[2], y_att[3]);
      omega_b = Eigen::Vector3d(y_att[4], y_att[5], y_att[6]);

      // Sensor accumulation with the cycle's accepted-step ENDPOINT values
      // (eq:imu:quadrature trapezoid; sensors/sensor.hpp contract): rates
      // from the attitude integration endpoints, specific forces per
      // eq:imu:specificforce at the translational endpoints under the
      // cycle's frozen attitude/actuator context.
      sensors::SensorCycleTruth st;
      st.t_s = t;
      st.dt_s = dt;
      st.r_i_m = r_start_;
      st.v_i_mps = v_start_;
      st.q_i2b = q_start;
      st.omega_b_start_radps = omega_start;
      st.omega_b_end_radps = omega_b;
      st.sf_b_start_mps2 = sf_start_;
      st.sf_b_end_mps2 = specific_force(t + dt, r_m, v_mps);
      // Cycle-END state: the truth AT the next sample instant, which is when
      // the point sensors are sampled (the GNC block runs at the top of the
      // following cycle). The camera hook emits these doubles verbatim, so
      // they are assigned, never recomputed downstream.
      st.r_end_i_m = r_m;
      st.v_end_i_mps = v_mps;
      st.q_end_i2b = q;
      // Ephemeris-, shadow-, and frame-derived geometry at that same
      // instant, composed once by the environment model and shared by every
      // sensor (models/environment.hpp). Evaluated only when a sensor needs
      // it, so a GNC run with an IMU alone pays nothing for it.
      if (needs_geometry_) {
        st.geom = env.sensor_geometry(t + dt, r_m);
        // The GNC block of the NEXT cycle runs at exactly this instant, so
        // this composition is the environment context its nav stage sees -
        // stashed rather than recomputed, so the sensors and the estimator
        // predicting them cannot disagree about the geometry.
        last_geom = st.geom;
      }
      for (auto& s : sensor_list) {
        s->accumulate(st);
      }
    }
  }

  // True specific force in body axes at (tt, r, v) under this cycle's
  // frozen attitude and actuator context (eq:imu:specificforce): the total
  // non-gravitational acceleration. Thrust, aero, SRP, and drag are sensed;
  // gravitation (central body + third bodies) is not - the environment's
  // gravitational subset cancels term-exactly against acceleration().
  Eigen::Vector3d specific_force(double tt, const Eigen::Vector3d& r,
                                 const Eigen::Vector3d& v) {
    const Eigen::Vector3d a_env = env.acceleration(tt, r, v);
    const Eigen::Vector3d a_grav = env.gravitational_acceleration(tt, r);
    const BodyLoads loads = eval_body_loads(ctx, r, v);
    return ctx.c_i2b * (a_env - a_grav) +
           (loads.f_thrust + loads.f_aero) / ctx.mass_kg;
  }

  void close_log() {
    // SrlogWriter::close() is guarded by is_open(), so this is idempotent
    // and a second call after finish() is a no-op rather than an error.
    writer.close();
    closed = true;
  }

  void finish() {
    close_log();
    for (int c = 0; c < 3; ++c) {
      run_summary.final_r_m[static_cast<std::size_t>(c)] = r_m[c];
      run_summary.final_v_mps[static_cast<std::size_t>(c)] = v_mps[c];
    }
    finished = true;
  }

  bool step_once() {
    if (finished) {
      throw std::logic_error(
          "VehicleCycle::step called after the run ended; the log is "
          "complete and closed");
    }
    if (closed) {
      // Distinguished from the finished case: this run was abandoned, so
      // its log is a valid prefix rather than a complete file, and no
      // run_end event was written.
      throw std::logic_error(
          "VehicleCycle::step called after close(); the log has been "
          "released and an abandoned run cannot be resumed");
    }
    process_cycle(i);
    if (stop) {
      writer.write_event(t_of(i), 2, "run_end");
      run_summary.event_records += 1;
      finish();
      refresh_observation(i, t_of(i), true);
      return false;
    }
    if (i == steps) {
      // run_end carries the configured duration verbatim, not steps*dt, so
      // the event timestamp is exactly the user's requested span.
      writer.write_event(cfg.duration_s, 2, "run_end");
      run_summary.event_records += 1;
      finish();
      refresh_observation(i, t_of(i), true);
      return false;
    }
    // Snapshot cycle i BEFORE advancing: every observation field then
    // describes one consistent instant, the cycle just processed.
    refresh_observation(i, t_of(i), true);
    advance_cycle(i);
    ++i;
    return true;
  }
};

VehicleCycle::VehicleCycle(const RunConfig& cfg, const std::string& out_path) {
  check_config_vehicle(cfg);
  impl_.reset(new Impl(cfg, out_path));
}

VehicleCycle::~VehicleCycle() = default;

bool VehicleCycle::step() { return impl_->step_once(); }

void VehicleCycle::close() { impl_->close_log(); }

bool VehicleCycle::done() const { return impl_->finished; }

std::int64_t VehicleCycle::cycle() const { return impl_->i; }

double VehicleCycle::time_s() const { return impl_->t_of(impl_->i); }

const RunSummary& VehicleCycle::summary() const { return impl_->run_summary; }

const CycleObservation& VehicleCycle::observation() const {
  return impl_->obs;
}

const gnc::TruthState& VehicleCycle::truth() const { return impl_->obs_truth; }

bool VehicleCycle::has_external_command() const {
  return impl_->ext_guidance != nullptr || impl_->ext_control != nullptr;
}

gnc::GncOutput VehicleCycle::external_command() const {
  if (impl_->ext_guidance != nullptr) return impl_->ext_guidance->command();
  if (impl_->ext_control != nullptr) return impl_->ext_control->command();
  return gnc::GncOutput();
}

void VehicleCycle::set_external_command(const gnc::GncOutput& cmd) {
  if (!has_external_command()) {
    throw std::logic_error(
        "VehicleCycle::set_external_command: this mission configures no "
        "\"external\" gnc.guidance or gnc.control component, so there is "
        "nothing to command; the run is flying its own built-in chain");
  }
  // Both slots receive the same command when both are external: the guidance
  // slot reads it as the attitude command and the control slot as the
  // torque, which are disjoint fields of one GncOutput.
  if (impl_->ext_guidance != nullptr) impl_->ext_guidance->set_command(cmd);
  if (impl_->ext_control != nullptr) impl_->ext_control->set_command(cmd);
}

log::SrlogHeaderFields VehicleCycle::make_header_fields(const RunConfig& cfg) {
  check_config_vehicle(cfg);
  int nav_n = 0;
  int nav_m = 0;
  int innov_mm = 0;
  bool nav_err = false;
  if (cfg.gnc.enabled) {
    // A component instance is the source of truth for its declared
    // dimensions; construction is cheap and draw-free by contract.
    const std::unique_ptr<gnc::IGncComponent> tmp =
        gnc::make_component(cfg.gnc.nav);
    nav_n = tmp->state_dim();
    nav_m = tmp->cov_dim();
    innov_mm = tmp->innov_max_dim();
    nav_err = !capture_error_layout(cfg, tmp.get()).empty();
  }
  return build_header_fields(cfg, nav_n, nav_m, innov_mm, nav_err);
}

}  // namespace star
