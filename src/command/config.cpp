#include "command/config.h"
#include <set>
#include <string>

#include "common/file_stream.h"

#define SET_OPTION(key, type) \
do { if(!vm_[key].defaulted() || !config_[key]) { \
  config_[key] = vm_[key].as<type>(); \
}} while(0)

#define SET_OPTION_NONDEFAULT(key, type) \
do { if(vm_.count(key) > 0) { \
  config_[key] = vm_[key].as<type>(); \
}} while(0)

bool Config::has(const std::string& key) const {
  return config_[key];
}

YAML::Node Config::get(const std::string& key) const {
  return config_[key];
}

const YAML::Node& Config::get() const {
  return config_;
}

void ProcessPaths(YAML::Node& node, const boost::filesystem::path& configPath, bool isPath) {
  using namespace boost::filesystem;
  std::set<std::string> paths = {"model", "trainsets", "vocabs"};

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

void Config::validate() const {
  if (config_["trainsets"]) {
    std::vector<std::string> tmp = config_["trainsets"].as<std::vector<std::string>>();
    if (tmp.size() != 2) {
      std::cerr << "No trainsets!" << std::endl;
      exit(1);
    }
  } else {
    std::cerr << "No trainsets!" << std::endl;
    exit(1);
  }
  if (config_["vocabs"]) {
    if (config_["vocabs"].as<std::vector<std::string>>().size() != 2) {
      std::cerr << "No vocab files!" << std::endl;
      exit(1);
    }
  } else {
    std::cerr << "No vocab files!" << std::endl;
    exit(1);
  }
}

void OutputRec(const YAML::Node node, YAML::Emitter& out) {
  // std::set<std::string> flow = { "devices" };
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
        // if(flow.count(key))
          // out << YAML::Flow;
        OutputRec(node[key], out);
      }
      out << YAML::EndMap;
      break;
    case YAML::NodeType::Undefined:
      out << node; break;
  }
}

void Config::addOptions(int argc, char** argv) {
  std::string configPath;

  namespace po = boost::program_options;
  po::options_description general("General options");

  general.add_options()
    ("config,c", po::value(&configPath),
     "Configuration file")
    ("model,m", po::value<std::string>()->default_value("./model"),
      "Path prefix for model to be saved")
    ("device,d", po::value<int>()->default_value(0),
      "Use device no.  arg")
    ("init,i", po::value<std::string>()->default_value(""),
      "Load weights from  arg  before training")
    ("overwrite", po::value<bool>()->default_value(false),
      "Overwrite model with following checkpoints")
    ("trainsets,t", po::value<std::vector<std::string>>()->multitoken(),
      "Paths to training corpora: source target")
    ("vocabs,v", po::value<std::vector<std::string>>()->multitoken(),
      "Paths to vocabulary files, have to correspond to --trainsets")
    ("after-epochs,e", po::value<size_t>()->default_value(0),
      "Finish after this many epochs, 0 is infinity")
    ("after-batches", po::value<size_t>()->default_value(0),
      "Finish after this many batch updates, 0 is infinity")
    ("disp-freq", po::value<size_t>()->default_value(100),
      "Display information every  arg  updates")
    ("save-freq", po::value<size_t>()->default_value(30000),
      "Save model file every  arg  updates")
    ("workspace,w", po::value<size_t>()->default_value(2048),
      "Preallocate  arg  MB of work space")
  ;

  po::options_description hyper("Hyper-parameters");
  hyper.add_options()
    ("max-length", po::value<size_t>()->default_value(50),
      "Maximum length of a sentence in a training sentence pair")
    ("mini-batch,b", po::value<int>()->default_value(40),
      "Size of mini-batch used during update")
    ("maxi-batch", po::value<int>()->default_value(20),
      "Number of batches to preload for length-based sorting")
    ("lrate,l,", po::value<double>()->default_value(0.0001),
      "Learning rate for Adam algorithm")
    ("clip-norm", po::value<double>()->default_value(1.f),
      "Clip gradient norm to  arg  (0 to disable)")
    ("dim-vocabs", po::value<std::vector<int>>()
      ->multitoken()
      ->default_value(std::vector<int>({50000, 50000}), "50000 50000"),
      "Maximum items in vocabulary ordered by rank")
    ("dim-emb", po::value<int>()->default_value(512), "Size of embedding vector")
    ("dim-rnn", po::value<int>()->default_value(1024), "Size of rnn hidden state")
  ;

  po::options_description configuration("Configuration meta options");
  configuration.add_options()
    ("relative-paths", po::value<bool>()->zero_tokens()->default_value(false),
     "All paths are relative to the config file location")
    ("dump-config", po::value<bool>()->zero_tokens()->default_value(false),
     "Dump current (modified) configuration to stdout and exit")
    ("help,h", po::value<bool>()->zero_tokens()->default_value(false),
      "Print this help message and exit")
  ;

  po::options_description cmdline_options("Allowed options");
  cmdline_options.add(general);
  cmdline_options.add(hyper);
  cmdline_options.add(configuration);

  boost::program_options::variables_map vm_;
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
    config_ = YAML::Load(InputFileStream(configPath));

  // Simple overwrites
  SET_OPTION("model", std::string);
  SET_OPTION("device", int);
  SET_OPTION("init", std::string);
  SET_OPTION("overwrite", bool);
  // SET_OPTION_NONDEFAULT("trainsets", std::vector<std::string>);

  if (!vm_["trainsets"].empty()) {
    config_["trainsets"] = vm_["trainsets"].as<std::vector<std::string>>();
  }
  if (!vm_["vocabs"].empty()) {
    config_["vocabs"] = vm_["vocabs"].as<std::vector<std::string>>();
  }
  // SET_OPTION_NONDEFAULT("vocabs", std::vector<std::string>);
  SET_OPTION("after-epochs", size_t);
  SET_OPTION("after-batches", size_t);
  SET_OPTION("disp-freq", size_t);
  SET_OPTION("save-freq", size_t);
  SET_OPTION("workspace", size_t);
  SET_OPTION("relative-paths", bool);

  SET_OPTION("max-length", size_t);
  SET_OPTION("mini-batch", int);
  SET_OPTION("maxi-batch", int);
  SET_OPTION("lrate", double);
  SET_OPTION("clip-norm", double);
  SET_OPTION("dim-vocabs", std::vector<int>);
  SET_OPTION("dim-emb", int);
  SET_OPTION("dim-rnn", int);

  validate();

  if (get<bool>("relative-paths") && !vm_["dump-config"].as<bool>())
    ProcessPaths(config_, boost::filesystem::path{configPath}.parent_path(), false);

  if(vm_["dump-config"].as<bool>()) {
    YAML::Emitter emit;
    OutputRec(config_, emit);
    std::cout << emit.c_str() << std::endl;
    exit(0);
  }

}

void Config::logOptions() {
  std::stringstream ss;
  YAML::Emitter out;
  OutputRec(config_, out);
  std::cerr << "Options: \n" << out.c_str() << std::endl;
}
