// Built-in reference GNC components (FR-25): the dead-reckoning attitude
// navigator, the pitch-program and attitude-hold guidance laws, and the
// quaternion-error PD attitude controller. Each is an IGncComponent selected
// from the registry by name; this header documents the registry names, the
// accepted parameters, and the exact control laws, which are cross-workstream
// contracts (a Python reimplementation of the PD law must match the built-in
// to < 1e-9 N*m, Phase 6 exit criterion 2, and the math-library chapter
// ch:gnc-builtin states these same equations).
//
// Registry names and parameters (GncComponentCfg.scalars / .vectors):
//
//   "dead_reckoning"  (nav)      - no parameters.
//       Initializes q_hat from the scenario initial attitude and omega_hat
//       from the scenario initial rate; on each fresh IMU sample composes
//       the increment as an exact rotation:
//         angle = |dtheta|, axis = dtheta / |dtheta| (identity when
//         |dtheta| == 0), dq = [cos(angle/2), sin(angle/2) * axis],
//         q_hat <- normalize(q_hat (x) dq)     (Hamilton scalar-first, D-7)
//       and estimates the rate as the interval mean omega_hat = dtheta/dt.
//       Estimator introspection: n = 7, x_hat = [q_w, q_x, q_y, q_z,
//       w_x, w_y, w_z], P identically zero (dead reckoning carries no
//       covariance), e = truth - estimate componentwise with the truth
//       quaternion sign-aligned to the estimate first.
//
//   "pitch_program"   (guidance) - scalars: azimuth_deg;
//                                  vectors: pitch_t_s, pitch_deg.
//       Reuses the Phase 4 open-loop machinery verbatim
//       (models::pwl_interp_clamped, models::pitch_program_axis,
//       models::attitude_from_body_x, models::omega_from_quaternions), with
//       the launch-pad ENU basis captured at init, so the commanded
//       attitude equals what the open-loop kPitchProgram mode would command
//       bit-for-bit at the same cycle times. Requires a geodetic launch
//       context (the ENU basis is the resolution frame).
//
//   "attitude_hold"   (guidance) - vectors: q_cmd (optional, 4 entries,
//                                  Hamilton scalar-first, normalized at
//                                  construction).
//       Commands a fixed inertial attitude with zero body rate; when q_cmd
//       is absent, holds the scenario initial attitude.
//
//   "pd_attitude"     (control)  - vectors: kp_nm_per_rad (3),
//                                  kd_nm_per_radps (3), tau_max_nm (3).
//       Quaternion-error PD law with per-axis gains and symmetric per-axis
//       saturation. EXACT arithmetic (normative, cross-workstream):
//         dq    = q_cmd^-1 (x) q_est          (Hamilton, scalar-first)
//         s     = (dq_0 >= 0) ? +1 : -1       (unwinding branch; sign(0) = +1)
//         tau_i = -kp_i * s * dq_vec_i - kd_i * (w_est_i - w_cmd_i)
//         tau_i = clamp(tau_i, -tau_max_i, +tau_max_i)
//       evaluated per axis exactly as written (left-associated products).
//       q_cmd/w_cmd come from the guidance slot, q_est/w_est from the nav
//       slot; when either slot is invalid the output is a hold.
#ifndef STAR_GNC_BUILTIN_HPP
#define STAR_GNC_BUILTIN_HPP

namespace star {
namespace gnc {

// Idempotent registration hook for the built-ins above. The registry calls
// it lazily on first use; it exists as a named function (rather than only a
// namespace-scope initializer) so linking the static core library cannot
// drop the built-ins' translation unit.
void register_builtin_components();

}  // namespace gnc
}  // namespace star

#endif  // STAR_GNC_BUILTIN_HPP
