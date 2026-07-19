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

#include "star/run.hpp"
#include "star/srlog_writer.hpp"

namespace star {

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
