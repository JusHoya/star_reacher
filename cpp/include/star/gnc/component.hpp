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
// Privileged-truth boundary (FR-24/FR-25): `GncInput.oracle` is populated if
// and only if the scenario sets `oracle = true` (stamped into the log header
// since v1.0). Components must treat `oracle.valid == false` as "truth does
// not exist"; the loop never fills it otherwise. That is the ONLY route by
// which a component sees the true state: no virtual below takes a TruthState,
// so the guarantee is structural rather than a rule an implementation is
// asked to honour. See the error-layout descriptor further down for how
// nav.err is produced without one.
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

// Truth kinematics/dynamics snapshot. It is privileged data (FR-24) and
// exists on two paths only, neither of which reaches a component
// unconditionally: as GncInput.oracle it crosses the FR-25 boundary if and
// only if the scenario sets oracle = true, and inside the loop it is an
// argument to compute_error_state(), which the LOOP calls to write nav.err.
// No virtual on IGncComponent takes it, so a component cannot receive it by
// overriding anything.
struct TruthState {
  bool valid = false;
  double t_s = 0.0;
  Eigen::Vector3d r_i_m = Eigen::Vector3d::Zero();
  Eigen::Vector3d v_i_mps = Eigen::Vector3d::Zero();
  Eigen::Quaterniond q_i2b = Eigen::Quaterniond::Identity();
  Eigen::Vector3d omega_b_radps = Eigen::Vector3d::Zero();
  double mass_kg = 0.0;
  // True in-run IMU bias states (ch:sensors-imu implementation note 4).
  // An estimator that carries bias states needs these to report a complete
  // truth-minus-estimate error on nav.err; without them the bias rows would
  // have to be logged as zero, which reads as "no error" rather than "not
  // known". Valid only when the run configures an IMU.
  bool imu_bias_valid = false;
  Eigen::Vector3d b_g_radps = Eigen::Vector3d::Zero();
  Eigen::Vector3d b_a_mps2 = Eigen::Vector3d::Zero();
};

// --- aiding measurements visible to the nav stage -------------------------
//
// One slot per FR-23 aiding kind. `fresh` is true only on the cycle the
// sensor was actually sampled: an estimator must not reprocess a held
// sample, because folding one measurement in twice makes the filter
// overconfident and is invisible in the state error until the covariance is
// checked. `valid` echoes the sensor's own gating flag (ch:sensors-optical
// eq:optical:gating and the altimeter band gate), so a gated-out sample is
// skipped rather than trusted. sensor_id indexes GncConfig::sensors, the
// same index nav.innov records carry.

struct NavFixSample {
  bool valid = false;
  bool fresh = false;
  std::uint32_t sensor_id = 0;
  Eigen::Vector3d r_i_m = Eigen::Vector3d::Zero();
  Eigen::Vector3d v_i_mps = Eigen::Vector3d::Zero();
};

struct StarTrackerSample {
  bool valid = false;
  bool fresh = false;
  std::uint32_t sensor_id = 0;
  // Attitude relative to the APPARENT inertial frame (eq:optical:stmodel);
  // a consumer that predicts it must apply the same aberration factor.
  Eigen::Quaterniond q_i2b = Eigen::Quaterniond::Identity();
};

struct AltimeterSample {
  bool valid = false;
  bool fresh = false;
  std::uint32_t sensor_id = 0;
  double h_m = 0.0;  // geodetic height over the central body's ellipsoid
};

// Environment context a navigator may legitimately use: quantities a real
// onboard navigator computes from time and its own ephemeris, never from
// truth. Supplied every cycle so an estimator can predict frame- and
// ephemeris-dependent measurements (the star tracker's aberration, the
// altimeter's body-fixed conversion) without reaching for truth.
struct NavEnvironment {
  bool ephemeris_valid = false;
  Eigen::Vector3d v_central_ssb_mps = Eigen::Vector3d::Zero();
  bool bodyfixed_valid = false;
  Eigen::Matrix3d c_gcrf_to_bodyfixed = Eigen::Matrix3d::Identity();
};

// The configured sensor-suite parameters, handed to components at init.
// An estimator's stochastic model is then the configured truth model
// (ch:ekf assumption 3, the reference-implementation stance) rather than a
// hand-copied duplicate in the mission file that can silently drift out of
// sync with the sensors it is supposed to describe. Fixed-size throughout,
// like every other struct here.
struct NavSensorModel {
  bool imu_present = false;
  std::uint32_t imu_id = 0;
  double gyro_arw = 0.0;         // N_g [rad/sqrt(s)], eq:imu:arw
  double accel_vrw = 0.0;        // N_a [(m/s)/sqrt(s)], eq:imu:arw
  double gyro_gm_sigma = 0.0;    // sigma_GM [rad/s], eq:imu:presetmap
  double gyro_tau_s = 0.0;       // tau_c [s], eq:imu:gm
  double accel_gm_sigma = 0.0;   // sigma_GM [m/s^2]
  double accel_tau_s = 0.0;      // tau_c [s]

  bool navfix_present = false;
  std::uint32_t navfix_id = 0;
  Eigen::Vector3d navfix_sigma_r_m = Eigen::Vector3d::Zero();
  Eigen::Vector3d navfix_sigma_v_mps = Eigen::Vector3d::Zero();

  bool startracker_present = false;
  std::uint32_t startracker_id = 0;
  Eigen::Vector3d startracker_sigma_rad = Eigen::Vector3d::Zero();
  Eigen::Vector3d startracker_boresight_b = Eigen::Vector3d::UnitZ();

  bool altimeter_present = false;
  std::uint32_t altimeter_id = 0;
  double altimeter_sigma_noise_m = 0.0;
  double altimeter_sigma_bias_m = 0.0;
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
  // Aiding measurements offered to the nav stage this cycle (zero-valued and
  // not fresh when the run configures no such sensor).
  NavFixSample navfix;
  StarTrackerSample startracker;
  AltimeterSample altimeter;
  NavEnvironment env;      // ephemeris/frame context, never truth-derived
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
  // Central-body constants an estimator needs for its own dynamics and
  // measurement models: the point-mass gravity parameter (eq:ekf:mech) and
  // the reference ellipsoid the altimeter measures against (eq:ekf:altH).
  double mu_m3ps2 = 0.0;
  double ellipsoid_a_m = 0.0;
  double ellipsoid_inv_f = 0.0;
  // The run's configured sensor suite, so an estimator's stochastic model is
  // the configured truth model rather than a duplicate that can drift.
  NavSensorModel sensors;
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

// --- declared error-state layout (FR-24/FR-25) ----------------------------
//
// nav.err is truth minus estimate expressed in the estimator's own state
// convention, so computing it needs two things: the truth state, and a
// reading of what each slot of the estimator's state vector means. Only the
// component knows the second. The obvious arrangement - hand the component
// the truth state and let it subtract - is the one this interface
// deliberately does NOT use, because it would put the real truth state in
// the hands of every estimator on every cycle whether or not the scenario
// enabled the oracle, which is precisely what FR-24 ("privileged; never
// visible to GNC plugins") and FR-25 ("truth never appears in GncInput")
// exist to forbid. A component that retained the argument would have truth
// available to its next update(), and no signature can prevent that.
//
// So the direction is inverted. The component DECLARES its layout, once,
// through error_layout(); the loop reads the state vector the component
// already publishes through state() and does the differencing itself. What
// crosses the plugin boundary is a description of the state vector, which
// carries no information about the world. nav.err survives for plugin
// estimators - that is the point of the descriptor - and the privileged
// boundary becomes structural instead of contractual.

// Which truth quantity a block of the state vector is compared against. Each
// names a member of TruthState; a quantity with no truth counterpart cannot
// be declared, which is what bounds the mechanism.
enum class ErrorQuantity {
  kPosition,     // TruthState::r_i_m, 3 slots
  kVelocity,     // TruthState::v_i_mps, 3 slots
  kAttitude,     // TruthState::q_i2b, 4 or 3 slots depending on the form
  kAngularRate,  // TruthState::omega_b_radps, 3 slots
  kGyroBias,     // TruthState::b_g_radps, 3 slots (needs a configured IMU)
  kAccelBias,    // TruthState::b_a_mps2, 3 slots (needs a configured IMU)
  kMass,         // TruthState::mass_kg, 1 slot
};

// How the error in that quantity is formed from truth and the estimate.
// Attitude is the reason this enum exists: an attitude error is a rotation
// difference, not a subtraction, and which side it is composed on and how it
// is parameterized are conventions the component owns. Declaring the
// convention lets the loop reproduce the component's own arithmetic exactly
// rather than imposing one.
//
// INVARIANT: a block's error width equals its state width. error_block_size
// serves both roles - validate_error_layout tiles the STATE vector with it,
// and compute_error_state writes that many slots of the ERROR vector at the
// same offset - so a form whose two widths differ makes the two disagree.
// Every attitude form below occupies four slots, which is what keeps the
// invariant true by construction rather than by inspection.
//
// A pair of three-slot rotation-vector forms was removed for violating it:
// they declared three slots while compute_error_state read four quaternion
// components at the block offset, so an attitude-block-last layout that
// PASSED validate_error_layout read one double past the state buffer. The
// convention itself is not lost - the consistency evaluator already carries
// dtheta = 2 sgn(dq_w) dq_v and applies it to the built-in EKF's n=16/m=15
// (docs/formats/srlog_v1.md), so the forms were a second implementation of a
// reduction the pipeline performs downstream, not a capability.
//
// STILL UNSERVED, and removal does not change this: an estimator whose state
// carries a THREE-parameter attitude directly (MRP, Gibbs/Rodrigues, or a
// rotation-vector error state) has no admissible form here. It could not use
// the removed forms either - attitude_error needs a q_est, which it reads as
// four consecutive state slots, and a three-parameter state does not publish
// one. Serving that case needs a way for a component to supply its own
// estimated quaternion independently of the state layout, which is a
// descriptor change rather than an enumerator. Removal makes the gap visible
// instead of appearing to fill it; do not re-add the enumerators believing
// the capability was merely unfinished.
enum class ErrorForm {
  // Elementwise truth minus estimate. The only admissible form for every
  // quantity except kAttitude, and inadmissible for kAttitude.
  kDifference,
  // 4 slots, quaternion: dq = conj(q_est) (x) q_true, the error resolved in
  // the ESTIMATED body frame (a "local" or right-multiplied error), sign
  // canonicalized to the +w hemisphere so the double cover cannot flip the
  // logged value between neighbouring epochs. This is eq:ekf:qerr, the form
  // the built-in error_state_ekf declares.
  kQuatErrorLocal,
  // 4 slots, quaternion: dq = q_true (x) conj(q_est), the error resolved in
  // the inertial frame (a "global" or left-multiplied error), likewise sign
  // canonicalized.
  kQuatErrorGlobal,
  // 4 slots: the plain componentwise difference q_true - q_est, with q_true
  // first sign-aligned to the estimate's hemisphere (q and -q are the same
  // attitude, and alignment is what keeps the difference continuous). This
  // is an ADDITIVE quaternion error rather than a rotation, appropriate to
  // an estimator that treats the four quaternion components as ordinary
  // state entries; the built-in dead_reckoning navigator declares it.
  kQuatDifferenceAligned,
};

// One contiguous run of the state vector, and how its error is formed.
// `offset` is the index of the block's first slot, shared by the state
// vector and the error vector - they have the same layout, which is what
// nav.err's "same dimension as the state" contract already implies.
struct ErrorBlock {
  ErrorQuantity quantity = ErrorQuantity::kPosition;
  ErrorForm form = ErrorForm::kDifference;
  int offset = 0;
};

// Slots a block occupies. Throws std::invalid_argument for a
// quantity/form pairing that has no meaning (an attitude subtraction, or a
// rotation error of a position).
int error_block_size(ErrorQuantity quantity, ErrorForm form);

// Check a declared layout against the state dimension it describes and
// against what the run can supply, throwing std::invalid_argument naming the
// offending block otherwise. The layout must TILE [0, state_dim) exactly:
// no gaps, no overlaps, nothing past the end. A partial layout is refused
// rather than zero-filled, because a zero in nav.err reads as "no error"
// when the truth is "not known" - the same distinction TruthState draws with
// imu_bias_valid. imu_bias_available reports whether the run configures an
// IMU, without which the bias quantities have no truth to difference.
void validate_error_layout(const std::vector<ErrorBlock>& layout,
                           int state_dim, bool imu_bias_available);

// Write truth minus estimate into e[0..state_dim) from the estimate x_hat
// (the component's own state vector, as returned by state()) and the truth
// state, following `layout`. Preconditions are those validate_error_layout
// checks; the loop validates once at run construction and calls this per
// cycle.
void compute_error_state(const std::vector<ErrorBlock>& layout,
                         const TruthState& truth, const double* x_hat,
                         double* e);

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

  // CONSTANCY CONTRACT for the three declared dimensions below. Each is
  // queried once, at GNC activation, and the loop sizes its fixed nav.est /
  // nav.err / nav.innov buffers from what it got; the SRLOG header records
  // the same values as the file's fixed record strides. Nothing resizes
  // either afterwards, so a component MUST return the same value from every
  // later call. An estimator that wants to augment its state mid-run
  // declares the augmented dimension up front. This is enforced, not merely
  // requested, for Python components (bindings/module.cpp pins each on first
  // query and refuses a later divergence), because there the declaration and
  // the payload come from the same mutable object and a divergence would
  // otherwise write past a buffer sized at construction.
  //
  // Estimator state dimension n for nav.est logging; 0 = not an estimator.
  virtual int state_dim() const { return 0; }
  // Covariance dimension m for nav.est's packed P (m(m+1)/2 doubles);
  // defaults to the state dimension. An error-state estimator whose
  // covariance lives in a different parameterization overrides it (the
  // reference EKF of a later workstream declares n = 16 - q scalar-first,
  // v, p, b_g, b_a - with m = 15, a 3-component attitude error).
  virtual int cov_dim() const { return state_dim(); }
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
  // Covariance packed row-major upper triangle into p[0..m(m+1)/2),
  // m = cov_dim(). Called only when state_dim() > 0.
  virtual void covariance_upper(double* p) const;
  // This estimator's state layout, block by block, from which the loop
  // computes nav.err (see the descriptor commentary above). The default is
  // an empty layout: the component declares none, and the run then writes
  // no nav.err channel at all rather than a channel of zeros that would be
  // indistinguishable from a perfect estimate. Called once per run, at
  // construction, so the declaration is fixed for the run.
  virtual const std::vector<ErrorBlock>& error_layout() const;
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
