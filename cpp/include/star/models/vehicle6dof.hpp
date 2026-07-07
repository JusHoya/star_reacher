// Vehicle 6DOF composition layer (Phase 4): the load-bearing pure pieces the
// run_vehicle orchestration composes into a full launch-to-orbit-capable rigid
// body -- the geodetic launch-pad initial state (FR-14, EC-10), the open-loop
// pitch-program attitude command (FR-14), the closed-form staging/jettison
// state remap (FR-10, EC-5), and the fixed-order translational acceleration
// assembly (D-10). Every physical term itself lives in its own chapter-tracked
// model module (environment, gravgrad, massprops, propulsion, actuators,
// aero); this module owns only the geometry and the composition order.
//
// Conventions (ch:vehicle6dof, citing ch:notation, D-7): q_i2b is the Hamilton
// scalar-first inertial-to-body frame-transformation quaternion; the vehicle
// (structural) frame is +X forward per FR-13; all units SI.
//
// Continuous state layout of the run_vehicle translational integrator
// (eq:vehicle6dof:state): y = [r_i(3), v_i(3)], the central-body-centered GCRF
// position and velocity of the composite center of mass. Attitude (q_i2b,
// omega_b), per-tank propellant masses, and the per-engine spool/gimbal/
// ignition states are advanced per control cycle (D-5 zero-order hold), not
// inside the continuous step.
//
// Math-library traceability (FR-29): the derivations live in the vehicle-6DOF
// chapter of docs/mathlib (ch:vehicle6dof); the implementation echoes its
// equation labels eq:vehicle6dof:padstate, eq:vehicle6dof:pitchaxis,
// eq:vehicle6dof:attitude, eq:vehicle6dof:remap, and eq:vehicle6dof:transaccel
// at the corresponding code.
#ifndef STAR_MODELS_VEHICLE6DOF_HPP
#define STAR_MODELS_VEHICLE6DOF_HPP

#include <cstddef>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace star {
namespace models {

// Piecewise-linear table interpolation with endpoint clamping over a
// strictly increasing grid (xs, ys parallel, size >= 2 - the caller's
// invariant, validated at config time). This is the single home of the
// pitch-table arithmetic: the Phase 4 open-loop pitch-program mode and the
// Phase 6 pitch-program guidance component both call it, so their commanded
// attitudes agree bit-for-bit by construction (the Phase 6 closed-loop
// contract), and the byte-frozen Phase 4 missions see the exact arithmetic
// the original in-loop lambda performed.
double pwl_interp_clamped(const std::vector<double>& xs,
                          const std::vector<double>& ys, double x);

// Dimension of the continuous translational state [r(3), v(3)].
inline constexpr std::size_t kVehicleTransStateDim = 6;

// Launch-pad state and the launch-site local tangent basis, all in GCRF.
struct PadState {
  Eigen::Vector3d r_i_m;    // pad position on the reference ellipsoid
  Eigen::Vector3d v_i_mps;  // co-rotating pad velocity, exactly omega_earth x r
  Eigen::Vector3d up_i;     // ellipsoid-normal local up
  Eigen::Vector3d east_i;   // local east
  Eigen::Vector3d north_i;  // local north
};

// Geodetic launch-pad state in GCRF (eq:vehicle6dof:padstate, FR-14/EC-10).
// lat_rad/lon_rad/alt_m are geodetic latitude, east longitude, and height above
// the (a_m, inv_f) reference ellipsoid; c_gcrf_to_itrf is the Earth-fixed
// rotation at the epoch (frames::c_gcrf_to_itrf). The returned inertial
// velocity is exactly omega_earth x r (the pad co-rotates with the planet), so
// the air-relative velocity on the pad is exactly zero and the logged dynamic
// pressure before release is exactly zero (EC-10). The ENU basis uses the
// ellipsoid-normal up. No libm beyond the sin/cos of the two angles.
PadState geodetic_pad_state(double lat_rad, double lon_rad, double alt_m,
                            const Eigen::Matrix3d& c_gcrf_to_itrf,
                            double omega_earth_radps, double a_m, double inv_f);

// Commanded thrust (body +X) direction in GCRF for an open-loop pitch-over
// (eq:vehicle6dof:pitchaxis): elevation `pitch_rad` above the local horizontal
// at flight azimuth `az_rad` (east of north), resolved in the launch-site ENU
// basis held inertially fixed at the pad (a documented Phase 4 approximation:
// the tangent frame does not co-rotate during the ~minutes-long ascent).
//   dhat = cos(pitch)(sin(az) east + cos(az) north) + sin(pitch) up.
Eigen::Vector3d pitch_program_axis(double az_rad, double pitch_rad,
                                   const Eigen::Vector3d& up_i,
                                   const Eigen::Vector3d& east_i,
                                   const Eigen::Vector3d& north_i);

// Attitude q_i2b whose body +X maps to the unit GCRF direction xb_i
// (eq:vehicle6dof:attitude), completing the triad by Gram-Schmidt of ref_i
// against xb_i (body +Y in the xb_i-ref_i plane). The vehicle aerodynamics and
// thrust are axisymmetric about body +X, so the roll orientation is a free,
// deterministic choice; ref_i near-parallel to xb_i falls back to a fixed
// alternate reference. xb_i and ref_i need not be unit (normalized internally).
Eigen::Quaterniond attitude_from_body_x(const Eigen::Vector3d& xb_i,
                                        const Eigen::Vector3d& ref_i);

// Body angular velocity [rad/s] that carries q0 into q1 over dt
// (eq:vehicle6dof:attitude): omega_b from the small relative rotation
// q0^{-1} (x) q1, used to log a real body rate for the prescribed-attitude
// segments. dt must be > 0.
Eigen::Vector3d omega_from_quaternions(const Eigen::Quaterniond& q0_i2b,
                                       const Eigen::Quaterniond& q1_i2b,
                                       double dt_s);

// Tracked translational state after a torque-free staging/jettison separation
// (eq:vehicle6dof:remap, FR-10). The tracked point moves from the old composite
// CG to the retained composite CG (both body-frame stations); r shifts by the
// rotated CG offset and v_new = v + omega x Delta_r_cg (the rotating-stack
// velocity term). omega_b is unchanged by a torque-free separation.
struct SeparationRemap {
  Eigen::Vector3d r_new_i_m;
  Eigen::Vector3d v_new_i_mps;
};
SeparationRemap separation_remap(const Eigen::Vector3d& cg_old_b_m,
                                 const Eigen::Vector3d& cg_new_b_m,
                                 const Eigen::Vector3d& r_old_i_m,
                                 const Eigen::Vector3d& v_old_i_mps,
                                 const Eigen::Quaterniond& q_i2b,
                                 const Eigen::Vector3d& omega_b_radps);

// Composed translational acceleration in GCRF (eq:vehicle6dof:transaccel), in
// the fixed D-10 summation order: the environment acceleration (gravity, then
// third bodies, then SRP, then orbital drag -- ch:environment) plus the summed
// body force (thrust, then RCS, then aero) rotated into GCRF and divided by the
// current composite mass. c_b2i is the body-to-inertial rotation (dcm of
// q_i2b, transposed). mass_kg must be positive.
Eigen::Vector3d composed_translational_accel(const Eigen::Vector3d& a_env_mps2,
                                             const Eigen::Vector3d& f_body_b_n,
                                             const Eigen::Matrix3d& c_b2i,
                                             double mass_kg);

}  // namespace models
}  // namespace star

#endif  // STAR_MODELS_VEHICLE6DOF_HPP
