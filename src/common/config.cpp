#include "common/config.h"
#include "common/config_parser.h"
#include "common/file_stream.h"
#include "common/logging.h"
#include "common/options.h"
#include "common/regex.h"
#include "common/utils.h"
#include "common/version.h"

#include <algorithm>
#include <set>
#include <string>

namespace marian {

// @TODO: keep seed in a single place, now it is kept here and in Config/Options
size_t Config::seed = (size_t)time(0);


Config::Config(ConfigParser const& cp) {
  initialize(cp);
}

Config::Config(int argc, char** argv, cli::mode mode, bool validate /*= true*/)
  : Config(ConfigParser(argc, argv, mode, validate)) {}

Config::Config(const Config& other) : config_(YAML::Clone(other.config_)) {}
Config::Config(const Options& options) : config_(options.cloneToYamlNode()) {}

void Config::initialize(ConfigParser const& cp) {
  config_ = YAML::Clone(cp.getConfig());
  cli::mode mode = cp.getMode();

  createLoggers(this);

  // echo version and command line
  LOG(info, "[marian] Marian {}", buildVersion());
  std::string cmdLine = cp.cmdLine();
  std::string hostname; int pid; std::tie
  (hostname, pid) = utils::hostnameAndProcessId();
  LOG(info, "[marian] Running on {} as process {} with command line:", hostname, pid);
  LOG(info, "[marian] {}", cmdLine);

  // set random seed
  if(get<size_t>("seed") == 0) {
    seed = (size_t)time(0);
  } else {
    seed = get<size_t>("seed");
  }

  // load model parameters
  bool loaded = false;
  if(mode == cli::mode::translation || mode == cli::mode::server) {
    auto model = get<std::vector<std::string>>("models")[0];
    try {
      if(!get<bool>("ignore-model-config"))
        loaded = loadModelParameters(model);
    } catch(std::runtime_error& ) {
      LOG(info, "[config] No model configuration found in model file");
    }
  }
  // if cli::mode::training or cli::mode::scoring
  else {
    auto model = get<std::string>("model");
    if(filesystem::exists(model) && !get<bool>("no-reload")) {
      try {
        if(!get<bool>("ignore-model-config"))
          loaded = loadModelParameters(model);
      } catch(std::runtime_error&) {
        LOG(info, "[config] No model configuration found in model file");
      }
    }
  }

  // guess --tsv-fields, i.e. the number of fields in a TSV input, if not set
  if(get<bool>("tsv") && get<size_t>("tsv-fields") == 0) {
    size_t tsvFields = 0;

    // use the length of --input-types if given
    auto inputTypes = get<std::vector<std::string>>("input-types");
    if(!inputTypes.empty()) {
      tsvFields = inputTypes.size();
    } else {
      if(loaded) {
        // model.npz has properly set vocab dimensions in special:model.yml,
        // so we may use them to determine the number of streams
        for(auto dim : get<std::vector<size_t>>("dim-vocabs"))
          if(dim != 0)  // language models have a fake extra vocab
            ++tsvFields;
        // For translation there is no target stream
        if((mode == cli::mode::translation || mode == cli::mode::server) && tsvFields > 1)
          --tsvFields;
      } else {
        // If parameters from model.npz special:model.yml were not loaded,
        // guess the number of inputs and outputs based on the model type name.
        // TODO: This is very britle, find a better solution
        auto modelType = get<std::string>("type");

        tsvFields = 1;
        if(modelType.find("multi-", 0) != std::string::npos)  // is a dual-source model
          tsvFields += 1;
        if(mode == cli::mode::training || mode == cli::mode::scoring)
          if(modelType.rfind("lm", 0) != 0)  // unless it is a language model
            tsvFields += 1;
      }

      // count fields with guided-alignment or data-weighting too
      if(mode == cli::mode::training) {
        if(has("guided-alignment") && get<std::string>("guided-alignment") != "none")
          tsvFields += 1;
        if(has("data-weighting") && !get<std::string>("data-weighting").empty())
          tsvFields += 1;
      }
    }

    config_["tsv-fields"] = tsvFields;
  }

  // ensures factors backward compatibility whilst keeping the more user friendly CLI
  if(get<std::string>("lemma-dependency").empty()) {
    YAML::Node config;
    int lemmaDimEmb = get<int>("lemma-dim-emb");
    if(lemmaDimEmb > 0) {
      config_["lemma-dependency"] = "re-embedding";
    } else if(lemmaDimEmb == -1) {
      config_["lemma-dependency"] = "lemma-dependent-bias";
    } else if(lemmaDimEmb == -2) {
      config_["lemma-dependency"] = "soft-transformer-layer";
    } else if(lemmaDimEmb == -3) {
      config_["lemma-dependency"] = "hard-transformer-layer";
    }
  }

  // echo full configuration
  log();

  // Log version of Marian that has been used to create the model.
  //
  // Key "version" is present only if loaded from model parameters and is not
  // related to --version flag
  if(has("version")) {
    auto version = get<std::string>("version");

    if(mode == cli::mode::training && version != buildVersion())
      LOG(info,
          "[config] Loaded model has been created with Marian {}, "
          "will be overwritten with current version {} at saving",
          version,
          buildVersion());
    else
      LOG(info, "[config] Loaded model has been created with Marian {}", version);

    // Remove "version" from config to make it consistent among different start-up scenarios
    config_.remove("version");
  }
  // If this is a newly started training
  else if(mode == cli::mode::training) {
    LOG(info, "[config] Model is being created with Marian {}", buildVersion());
  }
}

bool Config::has(const std::string& key) const {
  return config_[key];
}

YAML::Node Config::operator[](const std::string& key) const {
  return get(key);
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

void Config::save(const std::string& name) {
  io::OutputFileStream out(name);
  out << *this;
}

bool Config::loadModelParameters(const std::string& name) {
  YAML::Node config;
  io::getYamlFromModel(config, "special:model.yml", name);
  override(config);
  return true;
}

bool Config::loadModelParameters(const void* ptr) {
  YAML::Node config;
  io::getYamlFromModel(config, "special:model.yml", ptr);
  override(config);
  return true;
}

void Config::override(const YAML::Node& params) {
  for(auto& it : params) {
    config_[it.first.as<std::string>()] = it.second;
  }
}

void Config::log() {
  YAML::Emitter out;
  cli::OutputYaml(config_, out);
  std::string configString = out.c_str();

  // print YAML prepending each line with [config]
  auto lines = utils::split(configString, "\n");
  for(auto& line : lines)
    LOG(info, "[config] {}", line);
}

// Parse the device-spec parameters (--num-devices, --devices, --cpu-threads) into an array of
// size_t For multi-node, this returns the devices vector for the given rank, where "devices" really
// refers to how many graph instances are used (for CPU, that is the number of threads).
//
// For CPU, specify --cpu-threads.
//
// For GPU, specify either --num-devices or --devices.
//
// For single-MPI-process GPU, if both are given, --num-devices must be equal to size of --devices.
//
// For multi-MPI-process GPU, if --devices is equal to --num-devices, then the device set is shared
// across all nodes. Alternatively, it can contain a multiple of --num-devices entries. In that
// case, devices lists the set of MPI-process-local GPUs for all MPI processes, concatenated. This
// last form must be used when running a multi-MPI-process MPI job on a single machine with multiple
// GPUs.
//
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
//    --num-devices 1 --devices 0 1 2 3 4 5 6 7 // this is a 8-process job on a single machine;
//                                              // MPI processes 0..7 use devices 0..7, respectively
//    --num-devices 4 --devices 0 1 2 3 4 5 6 7 // this is a 2-process job on a single machine;
//                                              // MPI process 0 uses 0..3, and MPI process 1 uses 4..7
std::vector<DeviceId> Config::getDevices(Ptr<Options> options,
                                         size_t myMPIRank /*= 0*/,
                                         size_t numMPIProcesses /*= 1*/) {
  std::vector<DeviceId> devices;
  auto devicesArg = options->get<std::vector<std::string>>("devices");

  // CPU: devices[] just enumerate the threads (note: --devices refers to GPUs, and is thus ignored)
  if(options->get<size_t>("cpu-threads") > 0) {
    for(size_t i = 0; i < options->get<size_t>("cpu-threads"); ++i)
      devices.push_back({i, DeviceType::cpu});
  }
  // GPU: devices[] are interpreted in a more complex way
  else {
    size_t numDevices = options->get<size_t>("num-devices", 0);
    std::vector<size_t> deviceNos;
    for(auto d : devicesArg)
      deviceNos.push_back((size_t)std::stoull(d));

    // if devices[] is empty then default to 0..N-1, where N = numDevices or 1
    if (deviceNos.empty()) {
      if(numDevices == 0)  // if neither is given, then we default to 1 device, which is device[0]
        numDevices = 1;
      for(size_t i = 0; i < numDevices; ++i) // default to 0..N-1
        deviceNos.push_back(i);
    }
    // devices[] is not empty
    else if(numDevices == 0) // if device list then num devices defaults to list size
      numDevices = deviceNos.size(); // default to #devices

    // If multiple MPI processes then we can either have one set of devices shared across all
    // MPI-processes, or the full list across all MPI processes concatenated.  E.g. --num-devices 1
    // --devices 0 2 4 5 means 4 processes using devices 0, 2, 4, and 5, respectively.  In that
    // case, we cut out and return our own slice. In the above example, for MPI process 1, we would
    // return {2}.

    // special-case the error message (also caught indirectly below, but with a msg that is
    // confusing when one does not run multi-node)
    if(numMPIProcesses == 1)
      // same as requiring numPerMPIProcessDeviceNos == 1
      // @TODO: improve logging message as devices[] and numDevices are not informative for the user
      ABORT_IF(numDevices != deviceNos.size(), "devices[] size must be equal to numDevices");

    // how many lists concatenated in devices[]? Allowed is either 1 (=shared) or numWorkers
    size_t numPerMPIProcessDeviceNos = deviceNos.size() / numDevices;
    // @TODO: improve logging message as devices[] and numDevices are not informative for the user
    ABORT_IF(numDevices * numPerMPIProcessDeviceNos != deviceNos.size(),
             "devices[] size must be equal to or a multiple of numDevices");  // (check that it is a multiple)

    // if multiple concatenated lists are given, slice out the one for myMPIRank
    if(numPerMPIProcessDeviceNos != 1) {
      ABORT_IF(numPerMPIProcessDeviceNos != numMPIProcesses,
               "devices[] must either list a shared set of devices, or one set per MPI process");
      deviceNos.erase(deviceNos.begin(), deviceNos.begin() + myMPIRank * numDevices);
      deviceNos.resize(numDevices);
    }
    // form the final vector
    for(auto d : deviceNos)
      devices.push_back({ d, DeviceType::gpu });
  }
#ifdef MPI_FOUND
  for(auto d : devices)
    LOG(info, "[MPI rank {} out of {}]: {}[{}]",
        myMPIRank, numMPIProcesses, d.type == DeviceType::cpu ? "CPU" : "GPU", d.no);
#endif
  return devices;
}

Ptr<Options>
parseOptions(int argc, char** argv, cli::mode mode, bool validate){
  ConfigParser cp(mode);
  return cp.parseOptions(argc, argv, validate);
}

std::ostream& operator<<(std::ostream& out, const Config& config) {
  YAML::Emitter outYaml;
  cli::OutputYaml(config.get(), outYaml);
  out << outYaml.c_str();
  return out;
}

}  // namespace marian
