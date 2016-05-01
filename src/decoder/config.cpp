#include <set>

#include "config.h"
#include "file_stream.h"
#include "exception.h"

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
  
  general.add_options()
    ("config,c", po::value(&configPath)->required(),
     "Configuration file")
    ("devices,d", po::value(&devices)->multitoken()->default_value(std::vector<size_t>(1, 0), "0"),
     "CUDA device(s) to use, set to 0 by default, "
     "e.g. set to 0 1 to use gpu0 and gpu1. "
     "Implicitly sets minimal number of threads to number of devices.")
    ("threads-per-device", po::value<size_t>()->default_value(1),
     "Number of threads per device, total thread count equals threads x devices")
    ("show-weights", po::value<bool>()->zero_tokens()->default_value(false),
     "Output used weights to stdout and exit")
    ("load-weights", po::value<std::string>(),
     "Load scorer weights from this file")
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
  ;

  po::options_description cmdline_options("Allowed options");
  cmdline_options.add(general);
  cmdline_options.add(search);
  
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
  
  config_ = YAML::Load(InputFileStream(configPath));
   
  SET_OPTION("n-best", bool)
  SET_OPTION("normalize", bool)
  SET_OPTION("beam-size", size_t)
  SET_OPTION("threads-per-device", size_t)
  SET_OPTION("devices", std::vector<size_t>)
  SET_OPTION("show-weights", bool)
  SET_OPTION_NONDEFAULT("load-weights", std::string)
    
  Validate();
}

void Config::Validate() {
  UTIL_THROW_IF2(!config_["scorers"] || config_["scorers"].size() == 0,
                 "No scorers given in config file");
  
  UTIL_THROW_IF2(!config_["source-vocab"] || config_["source-vocab"].size() == 0,
                 "No source-vocab given in config file");
  
  UTIL_THROW_IF2(!config_["target-vocab"],
                 "No target-vocab given in config file");
  
  UTIL_THROW_IF2(config_["weights"].size() != config_["scorers"].size(),
                "Different number of models and weights in config file");
}

void OutputRec(const YAML::Node node, YAML::Emitter& out) {
  std::set<std::string> flow = { "weights", "devices"};
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
