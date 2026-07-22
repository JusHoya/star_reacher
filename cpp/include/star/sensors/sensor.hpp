// FR-23 sensor abstraction: core-side ISensor modules sampled on the
// control-cycle grid (D-5 major cycle), fed truth kinematics/dynamics
// context by the loop, emitting records into their `sensors.<kind>` SRLOG
// group (format doc section 3.2).
//
// Scheduling contract (normative here; the format doc states the on-disk
// consequence): a sensor declares a sample rate that must be an exact
// integer divisor of the control rate, so it is sampled every
// control_rate_hz / sample_rate_hz cycles, on the cycle grid. The v1 IMU
// is stricter (ch:sensors-imu): it emits one increment pair per major
// cycle, so its sample rate must EQUAL the control rate - faster- or
// slower-than-cycle IMU output is out of scope for v1 and rejected at
// configuration. Each cycle the loop calls accumulate() once with that
// cycle's truth (accepted-step endpoint values, below); at the sensor's
// sample instants (t = k / sample_rate_hz after GNC activation, k >= 1)
// the loop calls sample(), which emits the record and resets any
// accumulation state. Determinism (D-10): sensors draw randomness only
// from their D-9 named stream ("sensors.<kind>"), seeded from the master
// seed at construction; the ideal IMU consumes zero draws, but the seeding
// plumbing is part of this interface so the FR-23 error models land later
// with no interface change (they add config fields of their own through
// the plain-data GncSensorCfg, D-2).
#ifndef STAR_SENSORS_SENSOR_HPP
#define STAR_SENSORS_SENSOR_HPP

#include <cstdint>
#include <memory>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "star/gnc/config.hpp"
#include "star/models/environment.hpp"

namespace star {
namespace log {
class SrlogWriter;
}  // namespace log

namespace sensors {

// One control cycle's truth for sensor accumulation. The loop supplies the
// cycle's ACCEPTED-STEP ENDPOINT values so increment sensors can apply the
// trapezoidal rule of eq:imu:quadrature (ch:sensors-imu): the integrator's
// accepted steps tile each cycle exactly (D-5 forces step termination on
// cycle boundaries; one fixed rk4 step per cycle in this phase), so the
// start/end pairs below are that rule's tau_j / tau_j+1 evaluations.
// omega_b_* are the true body rates at the cycle boundaries (the attitude
// integration endpoints in the GNC mode; equal under kinematic ZOH modes).
// sf_b_* are the true specific forces per eq:imu:specificforce - the total
// non-gravitational acceleration in body axes, C_i2b (a_total - g): thrust,
// aero, SRP, and drag are sensed; gravitation (central body plus third
// bodies) is not. Both endpoints are evaluated with the cycle's frozen
// attitude and actuator context, exactly like the translational RHS (D-5).
struct SensorCycleTruth {
  double t_s = 0.0;
  double dt_s = 0.0;
  Eigen::Vector3d r_i_m = Eigen::Vector3d::Zero();    // cycle-start position
  Eigen::Vector3d v_i_mps = Eigen::Vector3d::Zero();  // cycle-start velocity
  Eigen::Quaterniond q_i2b = Eigen::Quaterniond::Identity();
  Eigen::Vector3d omega_b_start_radps = Eigen::Vector3d::Zero();
  Eigen::Vector3d omega_b_end_radps = Eigen::Vector3d::Zero();
  Eigen::Vector3d sf_b_start_mps2 = Eigen::Vector3d::Zero();
  Eigen::Vector3d sf_b_end_mps2 = Eigen::Vector3d::Zero();

  // Cycle-END state, i.e. the state at t_s + dt_s. Increment sensors use the
  // endpoint pairs above; POINT sensors (star tracker, sun sensor, nav fix,
  // altimeter, camera hook) measure an instant, and the instant they are
  // sampled at is the end of the cycle most recently accumulated - the loop
  // runs its GNC block, which samples, at the top of a cycle, after the
  // previous cycle's integration has landed. These fields are therefore the
  // truth AT the sample instant, and the camera hook emits r_end_i_m and
  // q_end_i2b as the bit-exact copies exit criterion 7 requires.
  Eigen::Vector3d r_end_i_m = Eigen::Vector3d::Zero();
  Eigen::Vector3d v_end_i_mps = Eigen::Vector3d::Zero();
  Eigen::Quaterniond q_end_i2b = Eigen::Quaterniond::Identity();

  // Ephemeris-, shadow-, and frame-derived geometry at the cycle-end state,
  // served by the environment model so every sensor and the force model see
  // one composition (models/environment.hpp).
  models::SensorGeometry geom;
};

class ISensor {
 public:
  virtual ~ISensor() = default;

  // Canonical sensor kind (log::kSensorKinds); names the SRLOG group.
  virtual const char* kind() const = 0;

  // Declared sample rate [Hz]; the loop validated the rate rules above
  // against the control rate before construction.
  virtual std::uint32_t sample_rate_hz() const = 0;

  // Integrate one control cycle of truth into the sensor's internal
  // accumulation state. Called exactly once per cycle, in cycle order.
  virtual void accumulate(const SensorCycleTruth& truth) = 0;

  // Emit the sample for the instant t_s into the sensor's SRLOG group and
  // reset accumulation state. Called only at the sensor's sample instants.
  virtual void sample(double t_s, log::SrlogWriter& writer) = 0;
};

// Instantiate a sensor from its resolved config (D-2 plain data). The
// master seed derives the sensor's D-9 named stream; the ideal IMU stores
// no generator (zero error draws) but every sensor kind constructs through
// this one signature. Unknown kinds throw std::invalid_argument naming the
// supported set (Phase 6 WS1: "imu").
std::unique_ptr<ISensor> make_sensor(const gnc::GncSensorCfg& cfg,
                                     std::uint64_t master_seed);

}  // namespace sensors
}  // namespace star

#endif  // STAR_SENSORS_SENSOR_HPP
