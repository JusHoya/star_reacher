// FR-25 registry and latency-FIFO implementation. Behavior contracts are in
// gnc/component.hpp; the built-in components live in gnc/builtin.cpp.
#include "star/gnc/component.hpp"

#include <map>
#include <stdexcept>
#include <utility>

#include "star/gnc/builtin.hpp"
#include "star/gnc/ekf.hpp"

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

void IGncComponent::error_state(const TruthState&, double*) const {
  throw std::logic_error(
      "IGncComponent::error_state called on a component that declares no "
      "estimator state (state_dim() == 0)");
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
