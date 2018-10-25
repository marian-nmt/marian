#pragma once

#include <sstream>
#include <string>
#include "common/definitions.h"

#include "3rd_party/yaml-cpp/yaml.h"

#define YAML_REGISTER_TYPE(registered, type)                \
  namespace YAML {                                          \
  template <>                                               \
  struct convert<registered> {                              \
    static Node encode(const registered& rhs) {             \
      type value = static_cast<type>(rhs);                  \
      return Node(value);                                   \
    }                                                       \
    static bool decode(const Node& node, registered& rhs) { \
      type value = node.as<type>();                         \
      rhs = static_cast<registered>(value);                 \
      return true;                                          \
    }                                                       \
  };                                                        \
  }

namespace marian {

/**
 * Container for options stored as key-value pairs. Keys are unique strings.
 */
class Options {
protected:
  YAML::Node options_;

public:
  Options() {}

  Options clone() const {
    auto options = Options();
    options.options_ = YAML::Clone(options_);
    return options;
  }

  YAML::Node& getYaml() { return options_; }
  const YAML::Node& getYaml() const { return options_; }

  void parse(const std::string& yaml) {
    auto node = YAML::Load(yaml);
    for(auto it : node)
      options_[it.first.as<std::string>()] = YAML::Clone(it.second);
  }

  /**
   * @brief Splice options from a YAML node
   *
   * By default, only options with keys that do not already exist in options_ are extracted from
   * node. These options are cloned if overwirte is true.
   *
   * @param node a YAML node to transfer the options from
   * @param overwrite overwrite all options
   */
  void merge(YAML::Node& node, bool overwrite = false) {
    for(auto it : node)
      if(overwrite || !options_[it.first.as<std::string>()])
        options_[it.first.as<std::string>()] = YAML::Clone(it.second);
  }

  void merge(const YAML::Node& node, bool overwrite = false) { merge(node, overwrite); }
  void merge(Ptr<Options> options) { merge(options->getYaml()); }

  std::string str() {
    std::stringstream ss;
    ss << options_;
    return ss.str();
  }

  template <typename T>
  void set(const std::string& key, T value) {
    options_[key] = value;
  }

  template <typename T>
  T get(const std::string& key) {
    ABORT_IF(!has(key), "Required option '{}' has not been set", key);
    return options_[key].as<T>();
  }

  template <typename T>
  T get(const std::string& key, T defaultValue) {
    if(has(key))
      return options_[key].as<T>();
    else
      return defaultValue;
  }

  bool has(const std::string& key) const { return options_[key]; }






// @TODO: remove the function from this class
//
// parse the device-spec parameters (--num-devices, --devices, --cpu-threads) into an array of size_t
// For multi-node, this returns the devices vector for the given rank, where "devices" really
// refers to how many graph instances are used (for CPU, that is the number of threads).
// For CPU, specify --cpu-threads.
// For GPU, specify either --num-devices or --devices.
// For single-MPI-process GPU, if both are given, --num-devices must be equal to size of --devices.
// For multi-MPI-process GPU, if --devices is equal to --num-devices, then the device set is shared
// across all nodes. Alternatively, it can contain a multiple of --num-devices entries. In that case,
// devices lists the set of MPI-process-local GPUs for all MPI processes, concatenated. This last form must be used
// when running a multi-MPI-process MPI job on a single machine with multiple GPUs.
// Examples:
//  - CPU:
//    --cpu-threads 8
//  - single MPI process, single GPU:
//    [no option given]  // will use device 0
//    --num-devices 1    // same
//    --devices 2        // will use device 2
//  - single MPI process, multiple GPU:
//    --num-devices 4    // will use devices 0, 1, 2, and 3
//    --devices 0 1 2 3  // same
//    --devices 4 5 6 7  // will use devices 4, 5, 6, and 7
//  - multiple MPI processes, multiple GPU:
//    --num-devices 4   // will use devices 0, 1, 2, and 3 in all MPI process, respectively
//    --devices 4 5 6 7 // will use devices 4, 5, 6, and 7 in all MPI process, respectively
//    --num-devices 1 --devices 0 1 2 3 4 5 6 7 // this is a 8-process job on a single machine; MPI processes 0..7 use devices 0..7, respectively
//    --num-devices 4 --devices 0 1 2 3 4 5 6 7 // this is a 2-process job on a single machine; MPI process 0 uses 0..3, and MPI process 1 uses 4..7
std::vector<DeviceId> getDevices(size_t myMPIRank = 0, size_t numMPIProcesses = 1) {
  std::vector<DeviceId> devices;
  auto devicesArg = get<std::vector<std::string>>("devices");
  // CPU: devices[] just enumerate the threads (note: --devices refers to GPUs, and is thus ignored)
  if (get<size_t>("cpu-threads") > 0) {
    for (size_t i = 0; i < get<size_t>("cpu-threads"); ++i)
      devices.push_back({ i, DeviceType::cpu });
  }
  // GPU: devices[] are interpreted in a more complex way
  else {
    size_t numDevices = has("num-devices") ? get<size_t>("num-devices") : 0;
    std::vector<size_t> deviceNos;
    for (auto d : devicesArg)
      deviceNos.push_back((size_t)std::stoull(d));
    // if devices[] is empty then default to 0..N-1, where N = numDevices or 1
    if (deviceNos.empty()) {
      if (numDevices == 0) // if neither is given, then we default to 1 device, which is device[0]
        numDevices = 1;
      for(size_t i = 0; i < numDevices; ++i) // default to 0..N-1
        deviceNos.push_back(i);
    }
    // devices[] is not empty
    else if (numDevices == 0) // if device list then num devices defaults to list size
      numDevices = deviceNos.size(); // default to #devices
    // If multiple MPI processes then we can either have one set of devices shared across all MPI-processes,
    // or the full list across all MPI processes concatenated.
    // E.g. --num-devices 1 --devices 0 2 4 5 means 4 processes using devices 0, 2, 4, and 5, respectively.
    // In that case, we cut out and return our own slice. In the above example, for MPI process 1, we would return {2}.
    if (numMPIProcesses == 1) // special-case the error mesage (also caught indirectly below, but with a msg that is confusing when one does not run multi-node)
      ABORT_IF(numDevices != deviceNos.size(), "devices[] size must be equal to numDevices"); // same as requiring numPerMPIProcessDeviceNos == 1
    size_t numPerMPIProcessDeviceNos = deviceNos.size() / numDevices; // how many lists concatenated in devices[]? Allowed is either 1 (=shared) or numWorkers
    ABORT_IF(numDevices * numPerMPIProcessDeviceNos != deviceNos.size(), "devices[] size must be equal to or a multiple of numDevices"); // (check that it is a multiple)
    if (numPerMPIProcessDeviceNos != 1) { // if multiple concatenated lists are given, slice out the one for myMPIRank
      ABORT_IF(numPerMPIProcessDeviceNos != numMPIProcesses, "devices[] must either list a shared set of devices, or one set per MPI process");
      deviceNos.erase(deviceNos.begin(), deviceNos.begin() + myMPIRank * numDevices);
      deviceNos.resize(numDevices);
    }
    // form the final vector
    for (auto d : deviceNos)
      devices.push_back({ d, DeviceType::gpu });
  }
#ifdef MPI_FOUND
  for (auto d : devices)
    LOG(info, "[MPI rank {} out of {}]: {}[{}]", myMPIRank, numMPIProcesses, d.type == DeviceType::cpu ? "CPU" : "GPU", d.no);
#endif
  return devices;
}
};

}  // namespace marian
