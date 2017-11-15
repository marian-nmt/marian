#include "scorer.h"

namespace amunmt {

Scorer::Scorer(const God &god,
              const std::string& name,
              const YAML::Node& config, size_t tab)
:god_(god)
,name_(name)
,config_(config)
,tab_(tab)
{
}

}
