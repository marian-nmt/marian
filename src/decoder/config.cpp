#include <set>

#include "config.h"

#define SET_OPTION(key, type) \
if(!vm_[key].defaulted() || !config_[key]) { \
  config_[key] = vm_[key].as<type>(); \
}

#define SET_OPTION_NONDEFAULT(key, type) \
if(vm_.count(key) > 0) { \
  config_[key] = vm_[key].as<type>(); \
}

bool Config::Has(const std::string& key) {
  return config_[key];
}

YAML::Node& Config::Get() {
  return config_;
}

void Config::AddOptions(size_t argc, char** argv) {
  namespace po = boost::program_options;
  po::options_description general("General options");

  std::string configPath;
  std::vector<size_t> devices;
  std::vector<size_t> tabMap;
  std::vector<float> weights;
  
  std::vector<std::string> modelPaths;
  std::vector<std::string> lmPaths;
  std::vector<std::string> sourceVocabPaths;
  std::string targetVocabPath;

  general.add_options()
    ("config,c", po::value(&configPath),
     "Configuration file")  
    ("model,m", po::value(&modelPaths)->multitoken()->required(),
     "Path to neural translation model(s)")
    ("source,s", po::value(&sourceVocabPaths)->multitoken()->required(),
     "Path to source vocabulary file.")
    ("target,t", po::value(&targetVocabPath)->required(),
     "Path to target vocabulary file.")
    ("ape", po::value<bool>()->zero_tokens()->default_value(false),
     "Add APE-penalty")
    ("lm,l", po::value(&lmPaths)->multitoken(),
     "Path to KenLM language model(s)")
    ("tab-map", po::value(&tabMap)->multitoken()->default_value(std::vector<size_t>(1, 0), "0"),
     "tab map")
    ("devices,d", po::value(&devices)->multitoken()->default_value(std::vector<size_t>(1, 0), "0"),
     "CUDA device(s) to use, set to 0 by default, "
     "e.g. set to 0 1 to use gpu0 and gpu1. "
     "Implicitly sets minimal number of threads to number of devices.")
    ("threads-per-device", po::value<size_t>()->default_value(1),
     "Number of threads per device, total thread count equals threads x devices")
    ("help,h", po::value<bool>()->zero_tokens()->default_value(false),
     "Print this help message and exit")
  ;

  po::options_description search("Search options");
  search.add_options()
    ("beam-size,b", po::value<size_t>()->default_value(12),
     "Decoding beam-size")
    ("normalize,n", po::value<bool>()->zero_tokens()->default_value(false),
     "Normalize scores by translation length after decoding")
    ("n-best", po::value<bool>()->zero_tokens()->default_value(false),
     "Output n-best list with n = beam-size")
    ("weights,w", po::value(&weights)->multitoken()->default_value(std::vector<float>(1, 1.0), "1.0"),
     "Model weights (for neural models and KenLM models)")
    ("show-weights", po::value<bool>()->zero_tokens()->default_value(false),
     "Output used weights to stdout and exit")
    ("load-weights", po::value<std::string>(),
     "Load scorer weights from this file")
  ;

  po::options_description kenlm("KenLM specific options");
  kenlm.add_options()
    ("kenlm-batch-size", po::value<size_t>()->default_value(1000),
     "Batch size for batched queries to KenLM")
    ("kenlm-batch-threads", po::value<size_t>()->default_value(4),
     "Concurrent worker threads for batch processing")
  ;

  po::options_description cmdline_options("Allowed options");
  cmdline_options.add(general);
  cmdline_options.add(search);
  cmdline_options.add(kenlm);

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
  
  if(configPath.size())
    config_ = YAML::LoadFile(configPath);
  
  SET_OPTION("model", std::vector<std::string>)
  SET_OPTION_NONDEFAULT("lm", std::vector<std::string>)
  SET_OPTION("ape", bool)
  SET_OPTION("source", std::vector<std::string>)
  SET_OPTION("target", std::string)
  
  SET_OPTION("n-best", bool)
  SET_OPTION("normalize", bool)
  SET_OPTION("beam-size", size_t)
  SET_OPTION("threads-per-device", size_t)
  SET_OPTION("devices", std::vector<size_t>)
  SET_OPTION("tab-map", std::vector<size_t>)
  
  SET_OPTION("weights", std::vector<float>)
  SET_OPTION("show-weights", bool)
  SET_OPTION_NONDEFAULT("load-weights", std::string)
  
  SET_OPTION("kenlm-batch-size", size_t)
  SET_OPTION("kenlm-batch-threads", size_t)
}

void OutputRec(const YAML::Node node, YAML::Emitter& out) {
  std::set<std::string> flow = { "weights", "devices", "tab-map" };
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

void Config::LogOptions() {
  std::stringstream ss;
  YAML::Emitter out;
  OutputRec(config_, out);
  LOG(info) << "Options: \n" << out.c_str();
}
