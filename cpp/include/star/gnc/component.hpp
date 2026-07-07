// FR-25 GNC plugin interface: one abstract base `IGncComponent` with
// `init(...)` and `update(GncInput) -> GncOutput`, a static registry keyed
// by config string, and the latency FIFO that delays command application
// (`latency_cycles`, D-5 zero-order hold at application).
//
// The built-in C++ navigation/guidance/control components are themselves
// plugins selected by name from the registry (gnc/builtin.hpp); a pybind11
// trampoline over this same base lands with the stepping-API workstream, so
// the structs here stay ABI-simple by design: fixed-size Eigen types,
// doubles, bools - no unions, no variable-length payloads.
//
// Privileged-truth boundary (FR-25): `GncInput.oracle` is populated if and
// only if the scenario sets `oracle = true` (stamped into the log header
// since v1.0). Components must treat `oracle.valid == false` as "truth does
// not exist"; the loop never fills it otherwise.
//
// Conventions (ch:notation, D-7): quaternions are Hamilton, scalar-first,
// q_i2b inertial-to-body; vectors are SI, body frame unless suffixed
// otherwise.
#ifndef STAR_GNC_COMPONENT_HPP
#define STAR_GNC_COMPONENT_HPP

#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "star/gnc/config.hpp"

namespace star {
namespace gnc {

// Latest IMU output visible to the GNC chain: accumulated increments over
// the sample interval ending at t_s (sensors/imu.hpp). valid is false until
// the first sample instant has passed.
struct ImuSample {
  bool valid = false;
  double t_s = 0.0;
  double dt_s = 0.0;  // accumulation interval length
  Eigen::Vector3d dtheta_b_rad = Eigen::Vector3d::Zero();
  Eigen::Vector3d dv_b_mps = Eigen::Vector3d::Zero();
};

// Truth kinematics/dynamics snapshot. Used in two distinct roles with two
// distinct trust levels: as GncInput.oracle it crosses the FR-25 privileged
// boundary only under the scenario oracle flag; as the argument of
// IGncComponent::error_state it feeds the nav.err log channel, which is
// analysis output, not component input.
struct TruthState {
  bool valid = false;
  double t_s = 0.0;
  Eigen::Vector3d r_i_m = Eigen::Vector3d::Zero();
  Eigen::Vector3d v_i_mps = Eigen::Vector3d::Zero();
  Eigen::Quaterniond q_i2b = Eigen::Quaterniond::Identity();
  Eigen::Vector3d omega_b_radps = Eigen::Vector3d::Zero();
  double mass_kg = 0.0;
};

// One component's output. The same struct serves all three chain roles with
// role-dependent meaning of the attitude fields:
//   nav      -> q_i2b/omega are the ESTIMATE (torque unused, zero);
//   guidance -> q_i2b/omega are the COMMAND (torque unused, zero);
//   control  -> torque_b_nm is the commanded body torque, already
//               saturated; the attitude fields echo the tracked command.
// valid == false means "hold": the loop keeps applying the previous
// applied command (and logs the held values with valid = 0, format doc
// section 3.2).
struct GncOutput {
  bool valid = false;
  Eigen::Quaterniond q_i2b = Eigen::Quaterniond::Identity();
  Eigen::Vector3d omega_b_radps = Eigen::Vector3d::Zero();
  Eigen::Vector3d torque_b_nm = Eigen::Vector3d::Zero();
};

// Everything a component may read on one control cycle. The loop fills the
// chain slots progressively in the fixed order nav -> guidance -> control:
// guidance sees nav_est, control sees nav_est and att_cmd. prev_applied is
// the command actually applied on the previous cycle (post-latency), so a
// component can implement rate limiting or bumpless transfer against what
// the vehicle really did.
struct GncInput {
  std::int64_t cycle = 0;  // control-cycle index since GNC activation
  double t_s = 0.0;
  double dt_s = 0.0;       // one control period (D-5)
  ImuSample imu;           // latest IMU sample (may be stale; see imu.t_s)
  bool imu_fresh = false;  // true when imu was sampled on this cycle
  GncOutput nav_est;       // filled after the nav stage
  GncOutput att_cmd;       // filled after the guidance stage
  GncOutput prev_applied;  // command applied on the previous cycle
  TruthState oracle;       // FR-25: populated ONLY when oracle == true
};

// One-time initialization context, captured at construction of the run.
// q0/omega0 are the scenario initial attitude state; the pad ENU basis is
// valid only for geodetic launch missions (the pitch-program guidance
// resolves its commanded axis in this basis, exactly like the Phase 4
// open-loop mode).
struct GncInitContext {
  double t0_s = 0.0;                 // GNC activation time
  Eigen::Quaterniond q0_i2b = Eigen::Quaterniond::Identity();
  Eigen::Vector3d omega0_b_radps = Eigen::Vector3d::Zero();
  bool pad_basis_valid = false;
  Eigen::Vector3d up_i = Eigen::Vector3d::UnitZ();
  Eigen::Vector3d east_i = Eigen::Vector3d::UnitX();
  Eigen::Vector3d north_i = Eigen::Vector3d::UnitY();
  std::uint32_t control_rate_hz = 0;
  double dt_s = 0.0;
};

// One applied aiding update, reported by an estimator for nav.innov
// logging: the innovation vector y (size m) and the innovation covariance S
// packed row-major upper triangle (size m(m+1)/2). sensor_id indexes the
// run's configured sensor list (GncConfig::sensors order, which is also the
// log header's "gnc" sensors array).
struct InnovationSample {
  std::uint32_t sensor_id = 0;
  std::vector<double> y;
  std::vector<double> s_upper;
};

// The FR-25 abstract base. update() runs once per control cycle in chain
// order and must be deterministic (D-10): no clock, no I/O, no global
// state. The state/covariance/error introspection exists so the loop can
// log nav.est and nav.err generically for any estimator dimension; the
// default "dimension zero" means the component logs nothing there
// (guidance and control components leave these untouched).
class IGncComponent {
 public:
  virtual ~IGncComponent() = default;

  virtual void init(const GncInitContext& ctx) = 0;
  virtual GncOutput update(const GncInput& input) = 0;

  // Estimator state dimension n for nav.est logging; 0 = not an estimator.
  virtual int state_dim() const { return 0; }
  // Maximum innovation dimension across this estimator's aiding sensors;
  // 0 = no aiding, and nav.innov is then not declared in the log. An
  // aiding estimator (the reference EKF of a later workstream) overrides
  // this and innovations() - no change to this interface is required.
  virtual int innov_max_dim() const { return 0; }
  // The aiding updates applied during the most recent update() call, in
  // application order; the loop logs each to nav.innov zero-padded to
  // innov_max_dim(). Rebuilt by the component on every update(); the
  // default returns an empty list.
  virtual const std::vector<InnovationSample>& innovations() const;
  // State vector into x_hat[0..n). Called only when state_dim() > 0.
  virtual void state(double* x_hat) const;
  // Covariance packed row-major upper triangle into p[0..n(n+1)/2).
  virtual void covariance_upper(double* p) const;
  // Truth-minus-estimate error in the estimator's own state convention,
  // into e[0..n) (nav.err contract: same dimension as the state). The
  // truth argument is log-side analysis data, not component input - it is
  // supplied by the loop for every run regardless of the oracle flag, and
  // implementations must not retain it.
  virtual void error_state(const TruthState& truth, double* e) const;
};

// --- registry (FR-25: built-ins selected by config string) ----------------

using GncFactory = std::unique_ptr<IGncComponent> (*)(const GncComponentCfg&);

// Register a factory under a unique name; duplicate names throw
// std::logic_error (two components silently shadowing each other would be a
// determinism hazard). Returns true so translation units can self-register
// through a namespace-scope initializer.
bool register_component(const std::string& name, GncFactory factory);

// Instantiate cfg.component from the registry, passing cfg to its factory.
// Unknown names throw std::invalid_argument listing every registered name.
std::unique_ptr<IGncComponent> make_component(const GncComponentCfg& cfg);

// Registered names in sorted order (error messages, tests, bindings).
std::vector<std::string> component_names();

// --- latency FIFO (FR-25 latency_cycles, exit criterion 8) ----------------

// A k-deep first-in-first-out delay line between the GNC chain's output and
// its application, pre-filled with k hold entries (valid == false) so the
// first k cycles apply the neutral command. push() inserts this cycle's
// chain output and returns the entry due for application: with k == 0 the
// output passes straight through; with k > 0 the output computed on cycle i
// is applied on cycle i + k, shifting application by exactly k cycles.
// A popped hold entry resolves to the previous applied command with its
// valid flag cleared (ZOH at application, D-5); `neutral` seeds the very
// first hold. Deterministic: pure state machine, no clock.
class LatencyFifo {
 public:
  LatencyFifo(std::uint32_t latency_cycles, const GncOutput& neutral);

  // Push the chain output for this cycle; returns the command to apply
  // this cycle (a fresh output when its delay has elapsed, otherwise the
  // held previous applied command with valid == false).
  GncOutput push(const GncOutput& produced);

  // The command applied on the most recent cycle (the neutral command
  // before the first push).
  const GncOutput& applied() const { return applied_; }

 private:
  std::deque<GncOutput> queue_;
  GncOutput applied_;
};

}  // namespace gnc
}  // namespace star

#endif  // STAR_GNC_COMPONENT_HPP
