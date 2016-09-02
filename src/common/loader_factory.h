#pragma once

#include <vector>
#include <yaml-cpp/yaml.h>

#include "exception.h"
#include "loader.h"

#define IF_MATCH_RETURN(typeVar, typeStr, LoaderType) \
do { \
  if(typeVar == typeStr) { \
    LoaderPtr loader(new LoaderType(name, config)); \
    loader->Load(); \
    return loader; \
  } \
} while(0)

class LoaderFactory {
  public:
    static LoaderPtr Create(const std::string& name,
                            const YAML::Node& config);
};

