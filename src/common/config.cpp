#include <set>

#include "common/config.h"
#include "common/file_stream.h"
#include "common/exception.h"
#include "common/git_version.h"

#define SET_OPTION(key, type) \
do { if(!vm_[key].defaulted() || !config_[key]) { \
  config_[key] = vm_[key].as<type>(); \
}} while(0)

#define SET_OPTION_NONDEFAULT(key, type) \
do { if(vm_.count(key) > 0) { \
  config_[key] = vm_[key].as<type>(); \
}} while(0)

bool Config::Has(const std::string& key) {
  return config_[key];
}

YAML::Node& Config::Get() {
  return config_;
}

void ProcessPaths(YAML::Node& node, const boost::filesystem::path& configPath, bool isPath) {
  using namespace boost::filesystem;
  std::set<std::string> paths = {"path", "paths", "source-vocab", "target-vocab", "bpe", "softmax-filter"};

  if(isPath) {
    if(node.Type() == YAML::NodeType::Scalar) {
      std::string nodePath = node.as<std::string>();
      if (nodePath.size()) {
        node = canonical(path{nodePath}, configPath).string();
      }
    }
    if(node.Type() == YAML::NodeType::Sequence) {
      for(auto&& sub : node) {
        ProcessPaths(sub, configPath, true);
        break;
      }
    }
  }
  else {
    switch (node.Type()) {
      case YAML::NodeType::Sequence:
        for(auto&& sub : node)
          ProcessPaths(sub, configPath, false);
        break;
      case YAML::NodeType::Map:
        for(auto&& sub : node) {
          std::string key = sub.first.as<std::string>();
          ProcessPaths(sub.second, configPath, paths.count(key) > 0);
        }
        break;
    }
  }
}

void OverwriteModels(YAML::Node& config, std::vector<std::string>& modelPaths) {
  for(size_t i = 0; i < modelPaths.size(); ++i) {
    std::stringstream name;
    name << "F" << i;
    config["scorers"][name.str()]["type"] = "Nematus";
    config["scorers"][name.str()]["path"] = modelPaths[i];
    if(!config["weights"][name.str()])
      config["weights"][name.str()] = 1;
  }
}

void OverwriteSourceVocabs(YAML::Node& config, std::vector<std::string>& sourceVocabPaths) {
    config["source-vocab"] = sourceVocabPaths;
}

void OverwriteTargetVocab(YAML::Node& config, std::string& targetVocabPath) {
    config["target-vocab"] = targetVocabPath;
}

void OverwriteBPE(YAML::Node& config, std::vector<std::string>& bpePaths) {
    config["bpe"] = bpePaths;
}

void Validate(const YAML::Node& config) {
  UTIL_THROW_IF2(!config["scorers"] || config["scorers"].size() == 0,
                 "No scorers given in config file");

  UTIL_THROW_IF2(!config["source-vocab"],
                 "No source-vocab given in config file");

  UTIL_THROW_IF2(!config["target-vocab"],
                 "No target-vocab given in config file");

  UTIL_THROW_IF2(config["weights"].size() != config["scorers"].size(),
                "Different number of models and weights in config file");

  for(auto&& pair: config["weights"])
    UTIL_THROW_IF2(!(config["scorers"][pair.first.as<std::string>()]),
                   "Weight has no scorer: " << pair.first.as<std::string>());

  for(auto&& pair: config["scorers"])
    UTIL_THROW_IF2(!(config["weights"][pair.first.as<std::string>()]), "Scorer has no weight: " << pair.first.as<std::string>());
}

void OutputRec(const YAML::Node node, YAML::Emitter& out) {
  std::set<std::string> flow = { "devices" };
  std::set<std::string> sorter;
  switch (node.Type()) {
    case YAML::NodeType::Null:
      out << node; break;
    case YAML::NodeType::Scalar:
      out << node; break;
    case YAML::NodeType::Sequence:
      out << YAML::BeginSeq;
      for(auto&& n : node)
        OutputRec(n, out);
      out << YAML::EndSeq;
      break;
    case YAML::NodeType::Map:
      for(auto& n : node)
        sorter.insert(n.first.as<std::string>());
      out << YAML::BeginMap;
      for(auto& key : sorter) {
        out << YAML::Key;
        out << key;
        out << YAML::Value;
        if(flow.count(key))
          out << YAML::Flow;
        OutputRec(node[key], out);
      }
      out << YAML::EndMap;
      break;
    case YAML::NodeType::Undefined:
      out << node; break;
  }
}

void LoadWeights(YAML::Node& config, const std::string& path) {
  LOG(info) << "Reading weights from " << path;
  InputFileStream fweights(path);
  std::string name;
  float weight;
  size_t i = 0;
  while(fweights >> name >> weight) {
    if(name.back() == '=')
      name.pop_back();
    LOG(info) << " > " << name << "= " << weight;
    config["weights"][name] = weight;
    i++;
  }
}


void Config::AddOptions(size_t argc, char** argv) {
  namespace po = boost::program_options;
  po::options_description general("General options");

  std::string configPath;
  std::vector<std::string> modelPaths;
  std::vector<std::string> sourceVocabPaths;
  std::string targetVocabPath;
  std::vector<std::string> bpePaths;
  bool debpe;

  std::vector<size_t> devices;

  general.add_options()
    ("config,c", po::value(&configPath),
     "Configuration file")
    ("input-file,i", po::value(&inputPath),
      "Take input from a file instead of stdin")
    ("model,m", po::value(&modelPaths)->multitoken(),
     "Overwrite scorer section in config file with these models. "
     "Assumes models of type Nematus and assigns model names F0, F1, ...")
    ("source-vocab,s", po::value(&sourceVocabPaths)->multitoken(),
     "Overwrite source vocab section in config file with vocab file.")
    ("target-vocab,t", po::value(&targetVocabPath),
     "Overwrite target vocab section in config file with vocab file.")
    ("bpe", po::value(&bpePaths)->multitoken(),
     "Overwrite bpe section in config with bpe code file.")
    ("no-debpe", po::value(&debpe)->zero_tokens()->default_value(false),
     "Providing bpe is on, turn off deBPE of the output.")
#ifdef CUDA
    ("devices,d", po::value(&devices)->multitoken()->default_value(std::vector<size_t>(1, 0), "0"),
     "CUDA device(s) to use, set to 0 by default, "
     "e.g. set to 0 1 to use gpu0 and gpu1. "
     "Implicitly sets minimal number of threads to number of devices.")
    ("gpu-threads", po::value<size_t>()->default_value(1),
     "Number of threads on a single GPU.")
    ("cpu-threads", po::value<size_t>()->default_value(0),
     "Number of threads on the CPU.")
#else
    ("cpu-threads", po::value<size_t>()->default_value(1),
     "Number of threads on the CPU.")
#endif
    ("batch-size", po::value<size_t>()->default_value(1),
     "Number of sentences in one batch.")
    ("show-weights", po::value<bool>()->zero_tokens()->default_value(false),
     "Output used weights to stdout and exit")
    ("load-weights", po::value<std::string>(),
     "Load scorer weights from this file")
    ("wipo", po::value<bool>()->zero_tokens()->default_value(false),
     "Use WIPO specific n-best-list format and non-buffering single-threading")
    ("return-alignment", po::value<bool>()->zero_tokens()->default_value(false),
     "If true, return alignment.")
    ("version,v", po::value<bool>()->zero_tokens()->default_value(false),
     "Print version.")
    ("help,h", po::value<bool>()->zero_tokens()->default_value(false),
     "Print this help message and exit")
  ;

  po::options_description search("Search options");
  search.add_options()
    ("beam-size,b", po::value<size_t>()->default_value(12),
     "Decoding beam-size")
    ("normalize,n", po::value<bool>()->zero_tokens()->default_value(false),
     "Normalize scores by translation length after decoding")
    ("softmax-filter,f", po::value<std::vector<std::string>>()->multitoken()->default_value(std::vector<std::string>(0), ""),
     "Filter final softmax: path to file with alignment [N first words]")
    ("allow-unk,u", po::value<bool>()->zero_tokens()->default_value(false),
     "Allow generation of UNK")
    ("n-best", po::value<bool>()->zero_tokens()->default_value(false),
     "Output n-best list with n = beam-size")
  ;

  po::options_description configuration("Configuration meta options");
  configuration.add_options()
    ("relative-paths", po::value<bool>()->zero_tokens()->default_value(false),
     "All paths are relative to the config file location")
    ("dump-config", po::value<bool>()->zero_tokens()->default_value(false),
     "Dump current (modified) configuration to stdout and exit")
  ;

  po::options_description cmdline_options("Allowed options");
  cmdline_options.add(general);
  cmdline_options.add(search);
  cmdline_options.add(configuration);

  po::variables_map vm_;
  try {
    po::store(po::command_line_parser(argc,argv)
              .options(cmdline_options).run(), vm_);
    po::notify(vm_);
  }
  catch (std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl << std::endl;

    std::cerr << "Usage: " + std::string(argv[0]) +  " [options]" << std::endl;
    std::cerr << cmdline_options << std::endl;
    exit(1);
  }

  if (vm_["help"].as<bool>()) {
    std::cerr << "Usage: " + std::string(argv[0]) +  " [options]" << std::endl;
    std::cerr << cmdline_options << std::endl;
    exit(0);
  }

  if (vm_["version"].as<bool>()) {
    std::cerr << AMUNMT_GIT_VERION << std::endl;
    exit(0);
  }

  if(configPath.size())
    config_ = YAML::Load(InputFileStream(configPath));

  // Simple overwrites
  SET_OPTION("n-best", bool);
  SET_OPTION("normalize", bool);
  SET_OPTION("wipo", bool);
  SET_OPTION("return-alignment", bool);
  SET_OPTION("softmax-filter", std::vector<std::string>);
  SET_OPTION("allow-unk", bool);
  SET_OPTION("no-debpe", bool);
  SET_OPTION("beam-size", size_t);
  SET_OPTION("cpu-threads", size_t);
  SET_OPTION("batch-size", size_t);
#ifdef CUDA
  SET_OPTION("gpu-threads", size_t);
  SET_OPTION("devices", std::vector<size_t>);
#endif
  SET_OPTION("show-weights", bool);
  SET_OPTION_NONDEFAULT("load-weights", std::string);
  SET_OPTION("relative-paths", bool);

  // @TODO: Apply complex overwrites

  if (Has("load-weights")) {
    LoadWeights(config_, Get<std::string>("load-weights"));
  }

  if (modelPaths.size()) {
    OverwriteModels(config_, modelPaths);
  }

  if (sourceVocabPaths.size()) {
    OverwriteSourceVocabs(config_, sourceVocabPaths);
  }

  if (targetVocabPath.size()) {
    OverwriteTargetVocab(config_, targetVocabPath);
  }

  if (bpePaths.size()) {
    OverwriteBPE(config_, bpePaths);
  }

  if (Get<bool>("relative-paths"))
    ProcessPaths(config_, boost::filesystem::path{configPath}.parent_path(), false);

  Validate(config_);

  if(vm_["dump-config"].as<bool>()) {
    YAML::Emitter emit;
    OutputRec(config_, emit);
    std::cout << emit.c_str() << std::endl;
    exit(0);
  }
}

void Config::LogOptions() {
  std::stringstream ss;
  YAML::Emitter out;
  OutputRec(config_, out);
  LOG(info) << "Options: \n" << out.c_str();
}
