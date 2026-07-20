// The built-in reference error-state EKF (FR-25, FR-26, ch:ekf): a 15-state
// multiplicative (indirect) extended Kalman filter that mechanizes a nominal
// state from the ch:sensors-imu increments and corrects it with external nav
// fix, star tracker, and altimeter updates.
//
// Registry name and parameters (GncComponentCfg.vectors; every one required,
// so the filter's initial belief is stated explicitly in the mission file and
// never inferred from truth):
//
//   "error_state_ekf" (nav)
//     q0            (4) initial attitude estimate, Hamilton scalar-first
//     v0_mps        (3) initial inertial velocity estimate
//     p0_m          (3) initial inertial position estimate
//     bg0_radps     (3) initial gyro bias estimate
//     ba0_mps2      (3) initial accelerometer bias estimate
//     p0_sigma_att_rad (3) initial 1-sigma attitude error, body axes
//     p0_sigma_vel_mps (3) initial 1-sigma velocity error
//     p0_sigma_pos_m   (3) initial 1-sigma position error
//     p0_sigma_bg_radps(3) initial 1-sigma gyro-bias error
//     p0_sigma_ba_mps2 (3) initial 1-sigma accelerometer-bias error
//
// P0 is diagonal with those variances, in the eq:ekf:staterr ordering
// (attitude, velocity, position, gyro bias, accelerometer bias). The
// measurement and process noise models are NOT parameters here: they are
// taken from the run's configured sensors through GncInitContext::sensors, so
// the filter's stochastic model is the configured truth model (ch:ekf
// assumption 3) and cannot silently drift out of sync with the instruments it
// describes.
//
// Estimator introspection (pinned cross-workstream, format doc section 3.2):
// n = state_dim() = 16 (quaternion scalar-first, then v, p, b_g, b_a),
// m = cov_dim() = 15 (the 3-component attitude error replaces the
// 4-component quaternion), so nav.est.P carries 120 doubles.
// innov_max_dim() = 6, the nav fix's dimension and the largest of the three.
//
// nav.err convention: e is the 16-vector
// [dq (4, scalar-first, sign-canonicalized), dv (3), dp (3), db_g (3),
// db_a (3)], where dq = q_hat^-1 (x) q_true is the MULTIPLICATIVE attitude
// error of eq:ekf:qerr - the estimator's own convention - and the remaining
// entries are additive truth-minus-estimate. The consistency evaluator
// reduces this 16-vector to the 15-vector P describes by
// dtheta = 2 sgn(dq_w) dq_v (docs/formats/srlog_v1.md, nav.err).
//
// The filter does not compute that error itself: it DECLARES the layout
// through error_layout() and the loop differences it against truth, so the
// true state never reaches this component (FR-24; the descriptor rationale
// is in gnc/component.hpp and sec:gnc:errlayout). The declared blocks are
// the multiplicative attitude error at offset 0 followed by velocity,
// position, gyro-bias, and accel-bias differences at 4, 7, 10, and 13.
#ifndef STAR_GNC_EKF_HPP
#define STAR_GNC_EKF_HPP

namespace star {
namespace gnc {

// Idempotent registration hook for the reference filter, called by the
// registry alongside register_builtin_components() for the same reason: a
// named function cannot be dropped when the static core library is linked.
void register_ekf_component();

}  // namespace gnc
}  // namespace star

#endif  // STAR_GNC_EKF_HPP
