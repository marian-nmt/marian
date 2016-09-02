#include "scorer.h"

Scorer::Scorer(const std::string& name,
	   const YAML::Node& config, size_t tab)
: name_(name), config_(config), tab_(tab)
{
}
