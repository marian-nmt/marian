#include "common/config.h"
#include "common/file_stream.h"
#include "common/logging.h"

#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <set>
#include <string>

namespace marian {

size_t Config::seed = (size_t)time(0);

bool Config::has(const std::string& key) const {
  return config_[key];
}

YAML::Node Config::get(const std::string& key) const {
  return config_[key];
}

const YAML::Node& Config::get() const {
  return config_;
}

YAML::Node& Config::get() {
  return config_;
}

void Config::log() {
  YAML::Emitter out;
  cli::OutputYaml(config_, out);
  std::string configString = out.c_str();

  std::vector<std::string> results;
  boost::algorithm::split(results, configString, boost::is_any_of("\n"));
  for(auto& r : results)
    LOG(info, "[config] {}", r);
}

void Config::override(const YAML::Node& params) {
  for(auto& it : params) {
    config_[it.first.as<std::string>()] = it.second;
  }
}

void Config::loadModelParameters(const std::string& name) {
  YAML::Node config;
  io::getYamlFromModel(config, "special:model.yml", name);
  override(config);
}

void Config::loadModelParameters(const void* ptr) {
  YAML::Node config;
  io::getYamlFromModel(config, "special:model.yml", ptr);
  override(config);
}

// parse the device-spec parameters (--num-devices, --devices, --cpu-threads) into an array of size_t
// For multi-node, this returns the devices vector for the given rank.
// For CPU, specify --cpu-threads.
// For GPU, specify either --num-devices or --devices.
// For single-worker GPU, if both are given, --num-devices must be equal to size of --devices.
// For multi-node GPU, if --devices is equal to --num-devices, then the device set is shared
// across all nodes. Alternatively, it can contain a multiple of --num-devices entries. In that case,
// devices lists the set of worker-local GPUs for all workers, concatenated. This last form must be used
// when running a multi-worker MPI job on a single machine with multiple GPUs.
// Examples:
//  - CPU:
//    --cpu-threads 8
//  - single worker, single GPU:
//    [no option given]  // will use device 0
//    --num-devices 1    // same
//    --devices 2        // will use device 2
//  - single worker process, multiple GPU:
//    --num-devices 4    // will use devices 0, 1, 2, and 3
//    --devices 0 1 2 3  // same
//    --devices 4 5 6 7  // will use devices 4, 5, 6, and 7
//  - multiple worker processes, multiple GPU:
//    --num-devices 4   // will use devices 0, 1, 2, and 3 on all worker nodes, respectively
//    --devices 4 5 6 7 // will use devices 4, 5, 6, and 7 on all worker nodes, respectively
//    --num-devices 1 --devices 0 1 2 3 4 5 6 7 // this is a 8-process job on a single machine; workers 0..7 use devices 0..7, respectively
//    --num-devices 4 --devices 0 1 2 3 4 5 6 7 // this is a 2-process job on a single machine; worker 0 uses 0..3, and worker 1 uses 4..7
std::vector<DeviceId> Config::getDevices(size_t myRank /*= 0*/, size_t numWorkers /*= 1*/) {
  std::vector<DeviceId> devices;
  auto devicesArg = get<std::vector<std::string>>("devices");
  // CPU: devices[] just enumerate the threads (--devices is ignored)
  if (get<size_t>("cpu-threads") > 0) {
    ABORT_IF(!devicesArg.empty(), "devices parameter not allowed in CPU mode");
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
    // If multi-node then we can either have one set of devices shared across all workers,
    // or the full list across all workers concatenated.
    // E.g. --num-devices 1 --devices 0 2 4 5 means 4 processes using devices 0, 2, 4, and 5, respectively.
    // In that case, we cut out and return our own slice. In the above example, for worker 1, we would return {2}.
    if (numWorkers == 1) // special-case the error mesage (also caught indirectly below, but with a msg that is confusing when one does not run multi-node)
      ABORT_IF(numDevices != deviceNos.size(), "devices[] size must be equal to numDevices"); // same as requiring numPerWorkerDeviceNos == 1
    size_t numPerWorkerDeviceNos = deviceNos.size() / numDevices; // how many lists concatenated in devices[]? Allowed is either 1 (=shared) or numWorkers
    ABORT_IF(numDevices * numPerWorkerDeviceNos != deviceNos.size(), "devices[] size must be equal to or a multiple of numDevices"); // (check that it is a multiple)
    if (numPerWorkerDeviceNos != 1) { // if multiple concatenated lists are given, slice out the one for myRank
      ABORT_IF(numPerWorkerDeviceNos != numWorkers, "devices[] must either list a shared set of devices, or one set per worker");
      deviceNos.erase(deviceNos.begin(), deviceNos.begin() + myRank * numDevices);
      deviceNos.resize(numDevices);
    }
    // form the final vector
    for (auto d : deviceNos)
      devices.push_back({ d, DeviceType::gpu });
  }
#if 1
  for (auto d : devices)
    LOG(info, "[worker {} out of {}]: {}[{}]", myRank, numWorkers, d.type == DeviceType::cpu ? "CPU" : "GPU", d.no);
#endif
  return devices;
}

}  // namespace marian
