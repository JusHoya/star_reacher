// The reusable control-cycle core of the vehicle 6DOF path (Phase 4 loop,
// factored; Phase 6 GNC wiring). One VehicleCycle owns the whole per-run
// state - vehicle runtime, environment, attitude state, sensors, GNC chain,
// latency FIFO, and the SRLOG writer - and advances it exactly one control
// cycle per step() call.
//
// This factoring is load-bearing for the determinism contract between batch
// and stepped execution (FR-24 / Phase 6 exit criterion 4): run_vehicle() is
// literally `VehicleCycle vc(cfg, path); while (vc.step()) {}`, so a
// stepping API that drives step() one control period at a time produces a
// byte-identical log by construction - same code, same order, same writer.
//
// Cycle anatomy (one step() call at cycle index i, time t = i * dt):
//   1. fire due [[sequence]] entries (Phase 4 authority: propulsion,
//      staging, pad release, termination);
//   2. compose the attached stack's mass properties (ZOH for this cycle);
//   3. attitude: kinematic modes set q/omega from their command law
//      (byte-frozen Phase 4 behavior); the GNC mode leaves the
//      dynamically integrated attitude state untouched;
//   4. GNC block (when active): sample due sensors, run the chain
//      nav -> guidance -> control in that fixed order, push the control
//      output through the latency FIFO, log gnc.cmd / nav.est / nav.err;
//   5. log truth / forces / mass / env on their decimated grids;
//   6. on a terminal event or the final cycle: write run_end, close the
//      log, finalize the summary, and return false;
//   7. otherwise advance the state across [t, t + dt): translational RK4,
//      tank depletion, engine spool/gimbal advance, sensor accumulation of
//      the cycle's held kinematics, and - in the GNC mode - torque-driven
//      attitude integration (rigid-body RK4 under the applied command
//      torque, ZOH; see gnc/builtin.hpp and the format doc section 3.2).
#ifndef STAR_VEHICLE_CYCLE_HPP
#define STAR_VEHICLE_CYCLE_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "star/gnc/component.hpp"
#include "star/run.hpp"
#include "star/srlog_writer.hpp"

namespace star {

// The FR-24 `observe()` payload: everything a non-privileged driver may read
// about the most recently processed control cycle. Truth is deliberately
// absent - it is reached only through VehicleCycle::truth(), the privileged
// accessor FR-24 keeps separate from the observation.
//
// This is a VALUE SNAPSHOT, refreshed once per step() and never recomputed
// on read, which is what makes `observe()` idempotent (Phase 6 exit
// criterion 4): reading it runs no component, draws no random number,
// consumes no sensor sample, and mutates nothing. Two reads without an
// intervening step() return the same bytes because they read the same
// stored bytes.
//
// Timing: every field describes the SAME instant - the cycle indexed by
// `cycle` at time `t_s`. Before the first step() the snapshot describes the
// initial state at cycle 0 with the GNC fields at their pre-activation
// defaults (gnc_active == false).
struct CycleObservation {
  std::int64_t cycle = 0;
  double t_s = 0.0;
  // False for the construction-time snapshot, true once a cycle has been
  // processed. Cycle 0 is otherwise ambiguous: the initial state and the
  // result of the first step() both carry index 0, and a driver reading
  // chain products needs to know which one it is holding.
  bool processed = false;
  bool done = false;
  bool gnc_active = false;

  // Sensor measurements offered to the nav stage on this cycle. Each carries
  // its own valid/fresh gating exactly as the nav component saw it.
  gnc::ImuSample imu;
  bool imu_fresh = false;
  gnc::NavFixSample navfix;
  gnc::StarTrackerSample startracker;
  gnc::AltimeterSample altimeter;
  gnc::NavEnvironment env;

  // Chain products for this cycle: the navigation estimate, the guidance
  // command, and the command actually applied after the latency FIFO.
  gnc::GncOutput nav_est;
  gnc::GncOutput att_cmd;
  gnc::GncOutput applied;

  // Estimator introspection, copied out of the nav component during the GNC
  // block. Empty when the nav component declares no state (state_dim() == 0),
  // so a dead-reckoning or guidance-only run carries no dead weight.
  std::vector<double> nav_x_hat;
  std::vector<double> nav_p_upper;
};

class VehicleCycle {
 public:
  // Validates the config (defensive re-checks; throws std::invalid_argument
  // with a named reason), constructs the environment/vehicle/GNC runtime,
  // opens the SRLOG at out_path, and writes the header and the run_start
  // event. No cycle has been processed yet.
  VehicleCycle(const RunConfig& cfg, const std::string& out_path);
  ~VehicleCycle();

  VehicleCycle(const VehicleCycle&) = delete;
  VehicleCycle& operator=(const VehicleCycle&) = delete;

  // Process the current cycle and advance to the next one. Returns false
  // when the run ended on this call (terminal event or final cycle) - the
  // log is then complete and closed, and step() must not be called again.
  bool step();

  bool done() const;
  std::int64_t cycle() const;   // current cycle index (0-based)
  double time_s() const;        // current cycle time, cycle() * dt_s

  // FR-24 observe(): the stored snapshot of the most recently processed
  // cycle. A pure read of a value refreshed only by step() - see
  // CycleObservation for the idempotence argument.
  const CycleObservation& observation() const;

  // FR-24 truth(): the privileged true state at the same instant the
  // observation describes. Kept out of CycleObservation so that handing a
  // driver an observation can never leak truth, and so the FR-25 boundary
  // stays a matter of which accessor was called. Like the observation, this
  // is a stored snapshot, so reading it is pure.
  const gnc::TruthState& truth() const;

  // FR-24 step(commands): replace the command held by the run's "external"
  // component. Throws std::logic_error when the mission configured no
  // external slot - commanding a vehicle that is flying its own built-in
  // guidance would otherwise fail silently, with the command dropped and the
  // log showing an autonomous run.
  void set_external_command(const gnc::GncOutput& cmd);

  // True when the mission's guidance or control slot is the "external"
  // component, i.e. when set_external_command() has an addressee.
  bool has_external_command() const;

  // The command the external component currently holds. Returned by value
  // so a caller can amend one field and set the result back, which is how
  // FR-24's "missing keys hold" is implemented. Returns a neutral hold when
  // the mission configured no external slot.
  gnc::GncOutput external_command() const;

  // Run summary (final state, record tallies); valid once done().
  const RunSummary& summary() const;

  // The SRLOG header declaration this configuration produces, including the
  // v1.2 GNC group declarations (the nav.est/nav.err dimension is obtained
  // from the configured nav component). Exposed for tests and for callers
  // that need the header without opening a file.
  static log::SrlogHeaderFields make_header_fields(const RunConfig& cfg);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace star

#endif  // STAR_VEHICLE_CYCLE_HPP
