#pragma once

#include <vector>
#include <yaml-cpp/yaml.h>

#include "common/exception.h"
#include "common/loader.h"

namespace amunmt {

#define IF_MATCH_RETURN(god, typeVar, typeStr, LoaderType) \
do { \
  if(typeVar == typeStr) { \
    Loader *loader = new LoaderType(name, config); \
    loader->Load(god); \
    return loader; \
  } \
} while(0)

class LoaderFactory {
  public:
    static LoaderPtr Create(const God &god,
    						const std::string& name,
                            const YAML::Node& config,
                            const DeviceType deviceType);

  protected:

    static Loader *CreateCPU(const God &god, const std::string& name,
                            const YAML::Node& config);

    static Loader *CreateGPU(const God &god, const std::string& name,
                            const YAML::Node& config);

    static Loader *CreateFPGA(const God &god, const std::string& name,
                            const YAML::Node& config);
};

}

