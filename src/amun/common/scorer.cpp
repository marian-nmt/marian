#include "scorer.h"

using namespace std;

namespace amunmt {

Scorer::Scorer(const God &god,
              const std::string& name,
              const YAML::Node& config, size_t tab,
              const Search &search)
:god_(god)
,name_(name)
,config_(config)
,tab_(tab)
,search_(search)
{
}

}
