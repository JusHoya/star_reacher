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
//   "dead_reckoning"  (nav)      - vectors: q0 (required, 4 entries,
//                                  Hamilton scalar-first, normalized at
//                                  construction).
//       Initializes q_hat from the CONFIGURED q0 - the mission file states
//       the initial estimate explicitly, so there is no implicit truth
//       access (ch:gnc-builtin, sec:gnc:deadreckoning) - and omega_hat from
//       zero. On each fresh IMU sample it composes the increment as an
//       exact rotation (eq:gnc:exactrot):
//         angle = |dtheta|, axis = dtheta / |dtheta| (identity when
//         |dtheta| == 0), dq = [cos(angle/2), sin(angle/2) * axis],
//       then propagates and estimates the rate (eq:gnc:drprop):
//         q_hat <- normalize(q_hat (x) dq)     (Hamilton scalar-first, D-7)
//         omega_hat = dtheta / dt              (interval mean)
//       No bias compensation, no coning correction (the omitted term is the
//       single-sample residual bounded by eq:gnc:coningbound). Estimator
//       introspection: n = 7, x_hat = [q_w, q_x, q_y, q_z, w_x, w_y, w_z],
//       P identically zero (dead reckoning carries no covariance),
//       e = truth - estimate componentwise with the truth quaternion
//       sign-aligned to the estimate first.
//
//   "pitch_program"   (guidance) - scalars: azimuth_deg;
//                                  vectors: pitch_t_s, pitch_deg.
//       Reuses the Phase 4 open-loop machinery verbatim
//       (models::pwl_interp_clamped per eq:gnc:interp,
//       models::pitch_program_axis, models::attitude_from_body_x,
//       models::omega_from_quaternions per eq:gnc:cmdrate - the commanded
//       rate is the finite-difference rotation to the next cycle's command,
//       resolved in the commanded frame, 2 sgn(dq_w) dq_vec / dt with
//       sgn(0) = +1), with the launch-pad ENU basis captured at init, so
//       the commanded attitude equals what the open-loop kPitchProgram mode
//       would command bit-for-bit at the same cycle times. Requires a
//       geodetic launch context (the ENU basis is the resolution frame).
//
//   "attitude_hold"   (guidance) - vectors: q_cmd (optional, 4 entries,
//                                  Hamilton scalar-first, normalized at
//                                  construction).
//       Commands a fixed inertial attitude with zero body rate; when q_cmd
//       is absent, holds the attitude state at GNC activation.
//
//   "pd_attitude"     (control)  - vectors: kp_nm_per_rad (3),
//                                  kd_nm_per_radps (3), tau_max_nm (3).
//       Quaternion-error PD law with per-axis gains and symmetric per-axis
//       saturation. EXACT arithmetic (normative, cross-workstream; the
//       chapter equations are echoed by label):
//         dq    = q_cmd^* (x) q_est            (eq:gnc:deltaq; Hamilton,
//                                              scalar-first, cmd-to-body)
//         s     = (dq_0 >= 0) ? +1 : -1        (eq:gnc:sign; sign(0) = +1)
//         w_err = w_est - C(dq) * w_cmd        (eq:gnc:werr; C from
//                                              eq:notation:quat2dcm resolves
//                                              the commanded rate into the
//                                              estimated body frame)
//         tau_i = -kp_i * s * dq_vec_i - kd_i * w_err_i     (eq:gnc:pd)
//         tau_i = clamp(tau_i, -tau_max_i, +tau_max_i)      (eq:gnc:sat)
//       evaluated per axis exactly as written (left-associated products),
//       with NO renormalization of dq - inputs are used as received.
//       q_cmd/w_cmd come from the guidance slot, q_est/w_est from the nav
//       slot; when either slot is invalid the output is a hold.
//
//   "external"        (guidance or control) - no parameters.
//       The FR-24 stepping-API command seam; documented at the
//       ExternalCommand declaration below because the driver, not the
//       mission file, supplies its numbers.
#ifndef STAR_GNC_BUILTIN_HPP
#define STAR_GNC_BUILTIN_HPP

#include "star/gnc/component.hpp"

namespace star {
namespace gnc {

// "external" (guidance or control) - no parameters.
//
// The FR-24 stepping-API authority stand-in: update() returns whatever the
// driver most recently handed to set_command(), unchanged. It is the seam
// that lets `Sim.step(commands)` command the vehicle without special-casing
// the chain - the external command traverses the same nav -> guidance ->
// control ordering, the same LatencyFifo, and the same gnc.cmd logging as a
// built-in component, so a commanded run and an autonomous run differ only
// in who computed the numbers.
//
// Zero-order hold is the whole semantics (D-5): the stored command persists
// across cycles until the driver replaces it, so a step() that supplies no
// command re-applies the previous one. The initial command is a hold
// (valid == false), which the loop resolves to the neutral command.
class ExternalCommand : public IGncComponent {
 public:
  explicit ExternalCommand(const GncComponentCfg& cfg);

  void init(const GncInitContext& ctx) override;
  GncOutput update(const GncInput& input) override;

  // Replace the held command. Deterministic by construction: the value is
  // stored verbatim and read back on the next update() with no arithmetic.
  void set_command(const GncOutput& cmd) { cmd_ = cmd; }
  const GncOutput& command() const { return cmd_; }

 private:
  GncOutput cmd_;
};

// Idempotent registration hook for the built-ins above. The registry calls
// it lazily on first use; it exists as a named function (rather than only a
// namespace-scope initializer) so linking the static core library cannot
// drop the built-ins' translation unit.
void register_builtin_components();

}  // namespace gnc
}  // namespace star

#endif  // STAR_GNC_BUILTIN_HPP
