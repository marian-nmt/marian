#include "scorer.h"

namespace amunmt {

Scorer::Scorer(const std::string& name,
               const YAML::Node& config,
               const DeviceInfo& devInfo,
               size_t tab)
: name_(name), config_(config), deviceInfo_(devInfo), tab_(tab)
{
}

}
