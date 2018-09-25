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
//  - single-worker GPU:
//    --num-devices 4    // will use devices 0, 1, 2, and 3
//    --devices 4 5 6 7  // will use devices 4, 5, 6, and 7
//  - multi-worker GPU:
//    --num-devices 4   // will use devices 0, 1, 2, and 3 on all workers, respectively
//    --devices 4 5 6 7 // will use devices 4, 5, 6, and 7 on all workers, respectively
//    --num-devices 1 --devices 3 4 5   // this is a 3-process job on a single machine; worker 0 will use device 3, worker 1 device 4, and worker 2 device 5
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
      if (numDevices == 0) // default to 1
        numDevices = 1;
      for(size_t i = 0; i < numDevices; ++i) // default to 0..N-1
        deviceNos.push_back(i);
    }
    // devices[] is not empty
    else {
      if (numDevices == 0) // default to 1
        numDevices = deviceNos.size(); // default to #devices
      else
        ABORT_IF(!get<bool>("multi-node") && numDevices != deviceNos.size(), "devices[] size must be equal to numDevices");
      // If multi-node then we can either have one set of devices shared across all workers,
      // or the full list across all workers concatenated.
      // E.g. --num-devices 1 --devices 0 1 2 3 means 4 processes using devices 0, 1, 2, and 3, respectively.
      if (numWorkers == 1) {
        ABORT_IF(numDevices != deviceNos.size(), "devices[] size must be equal to numDevices");
      }
      else {
        size_t numDevicesPerWorker = deviceNos.size() / numDevices;
        ABORT_IF(numDevices * numDevicesPerWorker != deviceNos.size(), "devices[] size must be equal to or a multiple of numDevices");
        ABORT_IF(!get<bool>("multi-node"), "getDevices() called wth numRanks != 1 while not in multi-node mode??");
        size_t numNodeSpecs = deviceNos.size() / numDevicesPerWorker; // devices[] can either list devices for all nodes individually, or list one set shared by all
        if (numNodeSpecs != 1) {
          ABORT_IF(numNodeSpecs != numWorkers, "devices[] must either list a shared set of devices, or one set per worker");
          deviceNos.erase(deviceNos.begin(), deviceNos.begin() + myRank * numDevicesPerWorker);
          deviceNos.resize(numDevicesPerWorker);
        }
      }
    }
    // form the final vector
    for (auto d : deviceNos)
      devices.push_back({ d, DeviceType::gpu });
  }
#if 1
  for (auto d : devices)
    LOG(info, "[{}/{}]: {}[{}]", myRank, numWorkers, d.type == DeviceType::cpu ? "CPU" : "GPU", d.no);
#endif
  return devices;
}

}  // namespace marian
