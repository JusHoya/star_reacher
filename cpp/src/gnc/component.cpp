// FR-25 registry and latency-FIFO implementation. Behavior contracts are in
// gnc/component.hpp; the built-in components live in gnc/builtin.cpp.
#include "star/gnc/component.hpp"

#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

#include "star/gnc/builtin.hpp"
#include "star/gnc/ekf.hpp"
#include "star/rotation.hpp"

namespace star {
namespace gnc {

void IGncComponent::state(double*) const {
  throw std::logic_error(
      "IGncComponent::state called on a component that declares no "
      "estimator state (state_dim() == 0)");
}

void IGncComponent::covariance_upper(double*) const {
  throw std::logic_error(
      "IGncComponent::covariance_upper called on a component that declares "
      "no estimator state (state_dim() == 0)");
}

const std::vector<ErrorBlock>& IGncComponent::error_layout() const {
  // The default is "no layout declared", which the loop reads as "write no
  // nav.err for this run". Shared immutable empty vector, same rationale as
  // innovations() below.
  static const std::vector<ErrorBlock> kNone;
  return kNone;
}

// --- declared error-state layout ------------------------------------------

namespace {

const char* quantity_name(ErrorQuantity q) {
  switch (q) {
    case ErrorQuantity::kPosition:
      return "position";
    case ErrorQuantity::kVelocity:
      return "velocity";
    case ErrorQuantity::kAttitude:
      return "attitude";
    case ErrorQuantity::kAngularRate:
      return "angular_rate";
    case ErrorQuantity::kGyroBias:
      return "gyro_bias";
    case ErrorQuantity::kAccelBias:
      return "accel_bias";
    case ErrorQuantity::kMass:
      return "mass";
  }
  return "unknown";
}

bool is_attitude_form(ErrorForm form) {
  return form != ErrorForm::kDifference;
}

// The truth vector a differenced block is measured against. Attitude and
// mass are handled separately because neither is a plain 3-vector.
const Eigen::Vector3d& truth_vector(ErrorQuantity q, const TruthState& truth) {
  switch (q) {
    case ErrorQuantity::kPosition:
      return truth.r_i_m;
    case ErrorQuantity::kVelocity:
      return truth.v_i_mps;
    case ErrorQuantity::kAngularRate:
      return truth.omega_b_radps;
    case ErrorQuantity::kGyroBias:
      return truth.b_g_radps;
    case ErrorQuantity::kAccelBias:
      return truth.b_a_mps2;
    default:
      break;
  }
  throw std::logic_error(
      "gnc::compute_error_state: quantity has no 3-vector truth counterpart");
}

// dq for the declared side of composition, canonicalized to the +w
// hemisphere. Canonicalization is what keeps the logged error continuous:
// q and -q are the same rotation, so without it the four components can
// flip sign between neighbouring epochs for no physical reason.
Eigen::Quaterniond attitude_error(ErrorForm form,
                                  const Eigen::Quaterniond& q_est,
                                  const Eigen::Quaterniond& q_true) {
  // kQuatDifferenceAligned is additive rather than a composition and is
  // handled by the caller; only the multiplicative forms reach here.
  const bool local = form == ErrorForm::kQuatErrorLocal;
  Eigen::Quaterniond dq =
      local ? rotation::quat_multiply(rotation::quat_conjugate(q_est), q_true)
            : rotation::quat_multiply(q_true, rotation::quat_conjugate(q_est));
  if (dq.w() < 0.0) {
    dq = Eigen::Quaterniond(-dq.w(), -dq.x(), -dq.y(), -dq.z());
  }
  return dq;
}

}  // namespace

int error_block_size(ErrorQuantity quantity, ErrorForm form) {
  if (quantity == ErrorQuantity::kAttitude) {
    switch (form) {
      case ErrorForm::kQuatErrorLocal:
      case ErrorForm::kQuatErrorGlobal:
      case ErrorForm::kQuatDifferenceAligned:
        // Every attitude form is four slots, which is what makes this one
        // number serve as both the state width validate_error_layout tiles
        // with and the error width compute_error_state writes.
        return 4;
      case ErrorForm::kDifference:
        break;
    }
    throw std::invalid_argument(
        "gnc error layout: the attitude quantity cannot use the difference "
        "form; an attitude error is a rotation difference, not a subtraction "
        "of quaternion components");
  }
  if (is_attitude_form(form)) {
    throw std::invalid_argument(
        std::string("gnc error layout: quantity '") + quantity_name(quantity) +
        "' cannot use an attitude error form; only the attitude quantity has "
        "a composition side and a rotation parameterization");
  }
  return quantity == ErrorQuantity::kMass ? 1 : 3;
}

void validate_error_layout(const std::vector<ErrorBlock>& layout,
                           int state_dim, int cov_dim,
                           bool imu_bias_available) {
  if (layout.empty()) return;  // "no layout declared" is a valid declaration
  if (state_dim <= 0) {
    throw std::invalid_argument(
        "gnc error layout: a layout was declared by a component whose "
        "state_dim() is zero; only an estimator has a state vector to lay "
        "out");
  }
  // Reconstruct the tiling by walking the declared blocks in offset order.
  // Requiring an exact tile of [0, state_dim) is what makes every slot of
  // nav.err accounted for: a gap would be logged as zero, and zero in an
  // error channel reads as "no error" rather than "not known".
  std::vector<const ErrorBlock*> ordered;
  ordered.reserve(layout.size());
  for (const ErrorBlock& b : layout) ordered.push_back(&b);
  std::sort(ordered.begin(), ordered.end(),
            [](const ErrorBlock* a, const ErrorBlock* b) {
              return a->offset < b->offset;
            });
  int next = 0;
  for (const ErrorBlock* b : ordered) {
    const int size = error_block_size(b->quantity, b->form);
    if (b->offset != next) {
      throw std::invalid_argument(
          "gnc error layout: block '" + std::string(quantity_name(b->quantity)) +
          "' starts at offset " + std::to_string(b->offset) + " where " +
          std::to_string(next) +
          " was expected; the declared blocks must tile the state vector "
          "contiguously from index 0 with no gaps and no overlaps");
    }
    if ((b->quantity == ErrorQuantity::kGyroBias ||
         b->quantity == ErrorQuantity::kAccelBias) &&
        !imu_bias_available) {
      throw std::invalid_argument(
          "gnc error layout: block '" + std::string(quantity_name(b->quantity)) +
          "' was declared but the run configures no IMU, so there is no true "
          "bias to difference against");
    }
    next += size;
  }
  if (next != state_dim) {
    throw std::invalid_argument(
        "gnc error layout: the declared blocks cover " + std::to_string(next) +
        " slots but the component declares state_dim() == " +
        std::to_string(state_dim) +
        "; a layout must describe the whole state vector or be left empty");
  }
  // The quaternion-led rule (header commentary). n == m + 1 with n >= 4 is
  // exactly the shape `star consistency` reduces by collapsing slots 0..3 as
  // an error quaternion, and the log does not carry the layout that would let
  // it check the assumption. Bounding the rule to n >= 4 keeps it to the
  // shape that is silently mangled: a narrower n == m + 1 does not meet the
  // consumer's own n >= 4 guard and is already reported there as a mismatch
  // rather than reduced.
  if (state_dim == cov_dim + 1 && state_dim >= 4 &&
      ordered.front()->quantity != ErrorQuantity::kAttitude) {
    throw std::invalid_argument(
        "gnc error layout: block '" +
        std::string(quantity_name(ordered.front()->quantity)) +
        "' occupies offset 0, but the component declares state_dim() == " +
        std::to_string(state_dim) + " with cov_dim() == " +
        std::to_string(cov_dim) +
        ", the one-slot-wider shape 'star consistency' reduces by collapsing "
        "slots 0..3 as a scalar-first error quaternion; the log carries no "
        "layout for it to check that against, so those slots would be "
        "misread as a rotation and the NEES would be wrong. Declare the "
        "attitude block at offset 0, or declare no layout at all if this "
        "estimator is not meant to be evaluated for consistency");
  }
}

void compute_error_state(const std::vector<ErrorBlock>& layout,
                         const TruthState& truth, const double* x_hat,
                         double* e) {
  for (const ErrorBlock& b : layout) {
    const int o = b.offset;
    if (b.quantity == ErrorQuantity::kAttitude) {
      // The estimate's quaternion is read out of the state vector in the
      // project convention (scalar-first, D-7).
      const Eigen::Quaterniond q_est(x_hat[o], x_hat[o + 1], x_hat[o + 2],
                                     x_hat[o + 3]);
      if (b.form == ErrorForm::kQuatDifferenceAligned) {
        // Additive error: align the truth quaternion to the estimate's
        // hemisphere, then subtract componentwise.
        Eigen::Quaterniond qt = truth.q_i2b;
        const double dot = qt.w() * q_est.w() + qt.x() * q_est.x() +
                           qt.y() * q_est.y() + qt.z() * q_est.z();
        if (dot < 0.0) {
          qt = Eigen::Quaterniond(-qt.w(), -qt.x(), -qt.y(), -qt.z());
        }
        e[o] = qt.w() - q_est.w();
        e[o + 1] = qt.x() - q_est.x();
        e[o + 2] = qt.y() - q_est.y();
        e[o + 3] = qt.z() - q_est.z();
        continue;
      }
      // The remaining multiplicative forms are both four slots, matching the
      // four state slots q_est was read from just above.
      const Eigen::Quaterniond dq = attitude_error(b.form, q_est, truth.q_i2b);
      e[o] = dq.w();
      e[o + 1] = dq.x();
      e[o + 2] = dq.y();
      e[o + 3] = dq.z();
      continue;
    }
    if (b.quantity == ErrorQuantity::kMass) {
      e[o] = truth.mass_kg - x_hat[o];
      continue;
    }
    const Eigen::Vector3d& t = truth_vector(b.quantity, truth);
    for (int i = 0; i < 3; ++i) {
      e[o + i] = t[i] - x_hat[o + i];
    }
  }
}

const std::vector<InnovationSample>& IGncComponent::innovations() const {
  // Non-aiding components share one immutable empty list; a function-local
  // static avoids any global-initialization ordering concern.
  static const std::vector<InnovationSample> kEmpty;
  return kEmpty;
}

namespace {

// Function-local static: initialized on first use, so registration from
// namespace-scope initializers in other translation units is order-safe.
std::map<std::string, GncFactory>& registry() {
  static std::map<std::string, GncFactory> instance;
  return instance;
}

// The built-ins self-register through register_component, but their
// registration objects live in a static library: without a reference from
// this translation unit the linker may drop gnc/builtin.o entirely and the
// registry would come up empty. Calling the builtin registration hook here
// creates that reference; the hook itself is idempotent.
void ensure_builtins() {
  static const bool once = [] {
    register_builtin_components();
    register_ekf_component();
    return true;
  }();
  (void)once;
}

}  // namespace

bool register_component(const std::string& name, GncFactory factory) {
  if (name.empty() || factory == nullptr) {
    throw std::invalid_argument(
        "gnc::register_component: a non-empty name and a non-null factory "
        "are required");
  }
  const auto inserted = registry().emplace(name, factory);
  if (!inserted.second) {
    throw std::logic_error(
        "gnc::register_component: component name '" + name +
        "' is already registered; duplicate names would make configuration "
        "resolution ambiguous");
  }
  return true;
}

std::unique_ptr<IGncComponent> make_component(const GncComponentCfg& cfg) {
  ensure_builtins();
  const auto it = registry().find(cfg.component);
  if (it == registry().end()) {
    std::string known;
    for (const auto& entry : registry()) {
      if (!known.empty()) known += ", ";
      known += entry.first;
    }
    throw std::invalid_argument(
        "gnc::make_component: unknown component '" + cfg.component +
        "'; registered components: {" + known + "}");
  }
  return it->second(cfg);
}

std::vector<std::string> component_names() {
  ensure_builtins();
  std::vector<std::string> names;
  names.reserve(registry().size());
  for (const auto& entry : registry()) {
    names.push_back(entry.first);
  }
  return names;  // std::map iteration is already sorted
}

LatencyFifo::LatencyFifo(std::uint32_t latency_cycles,
                         const GncOutput& neutral) {
  applied_ = neutral;
  applied_.valid = false;
  // Pre-fill with k hold entries so the first k pops apply the neutral
  // command: the output produced on cycle i then surfaces on cycle i + k.
  GncOutput hold = applied_;
  for (std::uint32_t i = 0; i < latency_cycles; ++i) {
    queue_.push_back(hold);
  }
}

GncOutput LatencyFifo::push(const GncOutput& produced) {
  queue_.push_back(produced);
  GncOutput due = queue_.front();
  queue_.pop_front();
  if (!due.valid) {
    // Hold: keep applying the previous applied command, flagged invalid so
    // the gnc.cmd log distinguishes a held application from a fresh one.
    due = applied_;
    due.valid = false;
  }
  applied_ = due;
  return due;
}

}  // namespace gnc
}  // namespace star
