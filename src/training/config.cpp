#include <boost/algorithm/string.hpp>
#include <set>
#include <string>

#include "3rd_party/cnpy/cnpy.h"
#include "common/file_stream.h"
#include "common/logging.h"
#include "training/config.h"

#define SET_OPTION(key, type)                    \
  do {                                           \
    if(!vm_[key].defaulted() || !config_[key]) { \
      config_[key] = vm_[key].as<type>();        \
    }                                            \
  } while(0)

#define SET_OPTION_NONDEFAULT(key, type)  \
  do {                                    \
    if(vm_.count(key) > 0) {              \
      config_[key] = vm_[key].as<type>(); \
    }                                     \
  } while(0)

namespace po = boost::program_options;

namespace marian {

uint16_t guess_terminal_width(uint16_t max_width) {
  struct winsize size;
  ioctl(STDOUT_FILENO, TIOCGWINSZ, &size);
  if(size.ws_col == 0)  // couldn't determine terminal width
    size.ws_col = po::options_description::m_default_line_length;
  return max_width ? std::min(size.ws_col, max_width) : size.ws_col;
}

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

void ProcessPaths(YAML::Node& node,
                  const boost::filesystem::path& configPath,
                  bool isPath) {
  using namespace boost::filesystem;
  std::set<std::string> paths = {"model", "trainsets", "vocabs"};

  if(isPath) {
    if(node.Type() == YAML::NodeType::Scalar) {
      std::string nodePath = node.as<std::string>();
      if(nodePath.size()) {
        try {
          node = canonical(path{nodePath}, configPath).string();
        } catch(boost::filesystem::filesystem_error& e) {
          std::cerr << e.what() << std::endl;
          auto parentPath = path{nodePath}.parent_path();
          node = (canonical(parentPath, configPath) / path{nodePath}.filename())
                     .string();
        }
      }
    }

    if(node.Type() == YAML::NodeType::Sequence) {
      for(auto&& sub : node) {
        ProcessPaths(sub, configPath, true);
      }
    }
  } else {
    switch(node.Type()) {
      case YAML::NodeType::Sequence:
        for(auto&& sub : node) {
          ProcessPaths(sub, configPath, false);
        }
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

void Config::validateOptions(bool translate, bool rescore) const {
  if(translate)
    return;

  UTIL_THROW_IF2(
      !has("train-sets") || get<std::vector<std::string>>("train-sets").empty(),
      "No train sets given in config file or on command line");
  if(has("vocabs")) {
    UTIL_THROW_IF2(get<std::vector<std::string>>("vocabs").size()
                       != get<std::vector<std::string>>("train-sets").size(),
                   "There should be as many vocabularies as training sets");
  }

  if(rescore)
    return;

  if(has("valid-sets")) {
    UTIL_THROW_IF2(get<std::vector<std::string>>("valid-sets").size()
                       != get<std::vector<std::string>>("train-sets").size(),
                   "There should be as many validation sets as training sets");
  }

  // validations for learning rate decaying
  UTIL_THROW_IF2(get<double>("lr-decay") > 1.0,
                 "Learning rate decay factor greater than 1.0 is unusual");
  UTIL_THROW_IF2(
      (get<std::string>("lr-decay-strategy") == "epoch+batches"
       || get<std::string>("lr-decay-strategy") == "epoch+stalled")
          && get<std::vector<size_t>>("lr-decay-start").size() != 2,
      "Decay strategies 'epoch+batches' and 'epoch+stalled' require two "
      "values specified with --lr-decay-start options");
  UTIL_THROW_IF2(
      (get<std::string>("lr-decay-strategy") == "epoch"
       || get<std::string>("lr-decay-strategy") == "batches"
       || get<std::string>("lr-decay-strategy") == "stalled")
          && get<std::vector<size_t>>("lr-decay-start").size() != 1,
      "Single decay strategies require only one value specified with "
      "--lr-decay-start option");
}

void Config::OutputRec(const YAML::Node node, YAML::Emitter& out) const {
  // std::set<std::string> flow = { "devices" };
  std::set<std::string> sorter;
  switch(node.Type()) {
    case YAML::NodeType::Null: out << node; break;
    case YAML::NodeType::Scalar: out << node; break;
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
    case YAML::NodeType::Undefined: out << node; break;
  }
}

void Config::addOptionsCommon(po::options_description& desc,
                              bool translate = false) {
  po::options_description general("General options", guess_terminal_width());
  // clang-format off
  general.add_options()
    ("config,c", po::value<std::string>(),
     "Configuration file")
    ("workspace,w", po::value<size_t>()->default_value(translate ? 512 : 2048),
      "Preallocate  arg  MB of work space")
    ("log", po::value<std::string>(),
     "Log training process information to file given by  arg")
    ("seed", po::value<size_t>()->default_value(0),
     "Seed for all random number generators. 0 means initialize randomly")
    ("relative-paths", po::value<bool>()->zero_tokens()->default_value(false),
     "All paths are relative to the config file location")
    ("dump-config", po::value<bool>()->zero_tokens()->default_value(false),
     "Dump current (modified) configuration to stdout and exit")
    ("help,h", po::value<bool>()->zero_tokens()->default_value(false),
      "Print this help message and exit")
  ;
  // clang-format on
  desc.add(general);
}

void Config::addOptionsModel(po::options_description& desc,
                             bool translate = false,
                             bool rescore = false) {
  po::options_description model("Model options", guess_terminal_width());
  // clang-format off
  if(!translate) {
    model.add_options()
      ("model,m", po::value<std::string>()->default_value("model.npz"),
      "Path prefix for model to be saved/resumed");
  } else {
    model.add_options()
    ("models,m", po::value<std::vector<std::string>>()
      ->multitoken()
      ->default_value(std::vector<std::string>({"model.npz"}), "model.npz"),
     "Paths to model(s) to be loaded");
  }

  model.add_options()
    ("type", po::value<std::string>()->default_value("amun"),
      "Model type (possible values: amun, s2s, multi-s2s)")
    ("dim-vocabs", po::value<std::vector<int>>()
      ->multitoken()
      ->default_value(std::vector<int>({50000, 50000}), "50000 50000"),
     "Maximum items in vocabulary ordered by rank")
    ("dim-emb", po::value<int>()->default_value(512), "Size of embedding vector")
    ("dim-pos", po::value<int>()->default_value(0), "Size of position embedding vector")
    ("dim-rnn", po::value<int>()->default_value(1024), "Size of rnn hidden state")
    ("cell-enc", po::value<std::string>()->default_value("gru"), "Type of RNN cell: gru, lstm, tanh (s2s)")
    ("cell-dec", po::value<std::string>()->default_value("gru"), "Type of RNN cell: gru, lstm, tanh (s2s)")
    ("layers-enc", po::value<int>()->default_value(1), "Number of encoder layers (s2s)")
    ("layers-dec", po::value<int>()->default_value(1), "Number of decoder layers (s2s)")
    ("skip", po::value<bool>()->zero_tokens()->default_value(false),
     "Use skip connections (s2s)")
    ("layer-normalization", po::value<bool>()->zero_tokens()->default_value(false),
     "Enable layer normalization")
    ("special-vocab", po::value<std::vector<size_t>>()->multitoken(),
     "Model-specific special vocabulary ids")
    ("tied-embeddings", po::value<bool>()->zero_tokens()->default_value(false),
     "Tie target embeddings and output embeddings in output layer (s2s)")
    ;

  if(!translate && !rescore) {
    model.add_options()
      ("dropout-rnn", po::value<float>()->default_value(0),
       "Scaling dropout along rnn layers and time (0 = no dropout)")
      ("dropout-src", po::value<float>()->default_value(0),
       "Dropout source words (0 = no dropout)")
      ("dropout-trg", po::value<float>()->default_value(0),
       "Dropout target words (0 = no dropout)")
    ;
  }
  // clang-format on

  modelFeatures_ = {
      "type",
      "dim-vocabs",
      "dim-emb",
      "dim-pos",
      "dim-rnn",
      "cell-enc",
      "cell-dec",
      "layers-enc",
      "layers-dec",
      "skip",
      "layer-normalization",
      "special-vocab",
      "tied-embeddings"
      /*"lexical-table", "vocabs"*/
  };

  desc.add(model);
}

void Config::addOptionsTraining(po::options_description& desc) {
  po::options_description training("Training options", guess_terminal_width());
  // clang-format off
  training.add_options()
    ("overwrite", po::value<bool>()->zero_tokens()->default_value(false),
      "Overwrite model with following checkpoints")
    ("no-reload", po::value<bool>()->zero_tokens()->default_value(false),
      "Do not load existing model specified in --model arg")
    ("train-sets,t", po::value<std::vector<std::string>>()->multitoken(),
      "Paths to training corpora: source target")
    ("vocabs,v", po::value<std::vector<std::string>>()->multitoken(),
      "Paths to vocabulary files have to correspond to --train-sets. "
      "If this parameter is not supplied we look for vocabulary files "
      "source.{yml,json} and target.{yml,json}. "
      "If these files do not exists they are created.")
    ("max-length", po::value<size_t>()->default_value(50),
      "Maximum length of a sentence in a training sentence pair")
    ("after-epochs,e", po::value<size_t>()->default_value(0),
      "Finish after this many epochs, 0 is infinity")
    ("after-batches", po::value<size_t>()->default_value(0),
      "Finish after this many batch updates, 0 is infinity")
    ("disp-freq", po::value<size_t>()->default_value(1000),
      "Display information every  arg  updates")
    ("save-freq", po::value<size_t>()->default_value(10000),
      "Save model file every  arg  updates")
    ("no-shuffle", po::value<bool>()->zero_tokens()->default_value(false),
    "Skip shuffling of training data before each epoch")
    ("tempdir,T", po::value<std::string>()->default_value("/tmp"),
      "Directory for temporary (shuffled) files")
    ("devices,d", po::value<std::vector<int>>()
      ->multitoken()
      ->default_value(std::vector<int>({0}), "0"),
      "GPUs to use for training. Asynchronous SGD is used with multiple devices.")

    ("mini-batch", po::value<int>()->default_value(64),
      "Size of mini-batch used during update")
    ("mini-batch-words", po::value<int>()->default_value(0),
      "Set mini-batch size based on words instead of sentences.")
    ("dynamic-batching", po::value<bool>()->zero_tokens()->default_value(false),
      "Determine mini-batch size dynamically based on sentence-length and reserved memory")
    ("maxi-batch", po::value<int>()->default_value(100),
      "Number of batches to preload for length-based sorting")

    ("optimizer,o", po::value<std::string>()->default_value("adam"),
     "Optimization algorithm (possible values: sgd, adagrad, adam")
    ("learn-rate,l", po::value<double>()->default_value(0.0001),
     "Learning rate")
    ("lr-decay", po::value<double>()->default_value(0.0),
     "Decay factor for learning rate: lr = lr * arg (0 to disable)")
    ("lr-decay-strategy", po::value<std::string>()->default_value("epoch+stalled"),
     "Strategy for learning rate decaying "
     "(possible values: epoch, batches, stalled, epoch+batches, epoch+stalled)")
    ("lr-decay-start", po::value<std::vector<size_t>>()
       ->multitoken()
       ->default_value(std::vector<size_t>({10,1}), "10,1"),
       "The first number of epoch/batches/stalled validations to start "
       "learning rate decaying")
    ("lr-decay-freq", po::value<size_t>()->default_value(50000),
     "Learning rate decaying frequency for batches, "
     "requires --lr-decay-strategy to be batches")

    ("clip-norm", po::value<double>()->default_value(1.f),
     "Clip gradient norm to  arg  (0 to disable)")
    ("moving-average", po::value<bool>()->zero_tokens()->default_value(false),
     "Maintain and save moving average of parameters")
    ("moving-decay", po::value<double>()->default_value(0.999),
     "Decay factor for moving average")
    //("lexical-table", po::value<std::string>(),
    // "Load lexical table")

    ("guided-alignment", po::value<std::string>(),
     "Use guided alignment to guide attention")
    ("guided-alignment-cost", po::value<std::string>()->default_value("ce"),
     "Cost type for guided alignment. Possible values: ce (cross-entropy), "
     "mse (mean square error), mult (multiplication).")
    ("guided-alignment-weight", po::value<double>()->default_value(1),
     "Weight for guided alignment cost")

    ("drop-rate", po::value<double>()->default_value(0),
     "Gradient drop ratio. (read: https://arxiv.org/abs/1704.05021)")
  ;
  // clang-format on
  desc.add(training);
}

void Config::addOptionsValid(po::options_description& desc) {
  po::options_description valid("Validation set options",
                                guess_terminal_width());
  // clang-format off
  valid.add_options()
    ("valid-sets", po::value<std::vector<std::string>>()->multitoken(),
      "Paths to validation corpora: source target")
    ("valid-freq", po::value<size_t>()->default_value(10000),
      "Validate model every  arg  updates")
    ("valid-metrics", po::value<std::vector<std::string>>()
      ->multitoken()
      ->default_value(std::vector<std::string>({"cross-entropy"}),
                      "cross-entropy"),
      "Metric to use during validation: cross-entropy, perplexity, valid-script. "
      "Multiple metrics can be specified")
    ("valid-script-path", po::value<std::string>(),
     "Path to external validation script")
    ("early-stopping", po::value<size_t>()->default_value(10),
     "Stop if the first validation metric does not improve for  arg  consecutive "
     "validation steps")
    ("keep-best", po::value<bool>()->zero_tokens()->default_value(false),
      "Keep best model for each validation metric")
    ("valid-log", po::value<std::string>(),
     "Log validation scores to file given by  arg")
    /*("beam-size", po::value<size_t>()->default_value(12),
      "Beam size used during search with validating translator")
    ("normalize", po::value<bool>()->zero_tokens()->default_value(false),
      "Normalize translation score by translation length")
    ("allow-unk", po::value<bool>()->zero_tokens()->default_value(false),
      "Allow unknown words to appear in output")*/
  ;
  // clang-format on
  desc.add(valid);
}

void Config::addOptionsTranslate(po::options_description& desc) {
  po::options_description translate("Translator options",
                                    guess_terminal_width());
  // clang-format off
  translate.add_options()
    ("input,i", po::value<std::vector<std::string>>()
      ->multitoken()
      ->default_value(std::vector<std::string>({"stdin"}), "stdin"),
      "Paths to input file(s), stdin by default")
    ("vocabs,v", po::value<std::vector<std::string>>()->multitoken(),
      "Paths to vocabulary files have to correspond to --input.")
    ("beam-size,b", po::value<size_t>()->default_value(12),
      "Beam size used during search")
    ("normalize,n", po::value<bool>()->zero_tokens()->default_value(false),
      "Normalize translation score by translation length")
    ("allow-unk", po::value<bool>()->zero_tokens()->default_value(false),
      "Allow unknown words to appear in output")
    ("max-length", po::value<size_t>()->default_value(1000),
      "Maximum length of a sentence in a training sentence pair")
    ("devices,d", po::value<std::vector<int>>()
      ->multitoken()
      ->default_value(std::vector<int>({0}), "0"),
      "GPUs to use for translating.")
    ("mini-batch", po::value<int>()->default_value(1),
      "Size of mini-batch used during update")
    ("maxi-batch", po::value<int>()->default_value(1),
      "Number of batches to preload for length-based sorting")
    ("n-best", po::value<bool>()->zero_tokens()->default_value(false),
      "Display n-best list")
    //("lexical-table", po::value<std::string>(),
    // "Path to lexical table")
    ("weights", po::value<std::vector<float>>()
      ->multitoken(),
      "Scorer weights")
  ;
  // clang-format on
  desc.add(translate);
}

void Config::addOptionsRescore(po::options_description& desc) {
  po::options_description rescore("Rescorer options", guess_terminal_width());
  // clang-format off
  rescore.add_options()
    ("no-reload", po::value<bool>()->zero_tokens()->default_value(false),
      "Do not load existing model specified in --model arg")
    ("train-sets,t", po::value<std::vector<std::string>>()->multitoken(),
      "Paths to corpora to be scored: source target")
    ("vocabs,v", po::value<std::vector<std::string>>()->multitoken(),
      "Paths to vocabulary files have to correspond to --train-sets. "
      "If this parameter is not supplied we look for vocabulary files "
      "source.{yml,json} and target.{yml,json}. "
      "If these files do not exists they are created.")
    ("max-length", po::value<size_t>()->default_value(1000),
      "Maximum length of a sentence in a training sentence pair")
    ("devices,d", po::value<std::vector<int>>()
      ->multitoken()
      ->default_value(std::vector<int>({0}), "0"),
      "GPUs to use for training. Asynchronous SGD is used with multiple devices.")

    ("mini-batch", po::value<int>()->default_value(64),
      "Size of mini-batch used during update")
    ("mini-batch-words", po::value<int>()->default_value(0),
      "Set mini-batch size based on words instead of sentences.")
    ("dynamic-batching", po::value<bool>()->zero_tokens()->default_value(false),
      "Determine mini-batch size dynamically based on sentence-length and reserved memory")
    ("maxi-batch", po::value<int>()->default_value(100),
      "Number of batches to preload for length-based sorting")
    ;
  // clang-format on
  desc.add(rescore);
}

void Config::addOptions(
    int argc, char** argv, bool doValidate, bool translate, bool rescore) {
  UTIL_THROW_IF2(translate && rescore,
                 "Config does not support both modes: translate and rescore!");

  addOptionsCommon(cmdline_options_, translate);
  addOptionsModel(cmdline_options_, translate, rescore);

  if(!translate) {
    if(rescore) {
      addOptionsRescore(cmdline_options_);
    } else {
      addOptionsTraining(cmdline_options_);
      addOptionsValid(cmdline_options_);
    }
  } else {
    addOptionsTranslate(cmdline_options_);
  }

  boost::program_options::variables_map vm_;
  try {
    po::store(
        po::command_line_parser(argc, argv).options(cmdline_options_).run(),
        vm_);
    po::notify(vm_);
  } catch(std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl << std::endl;
    std::cerr << "Usage: " + std::string(argv[0]) + " [options]" << std::endl;
    std::cerr << cmdline_options_ << std::endl;
    exit(1);
  }

  if(vm_["help"].as<bool>()) {
    std::cerr << "Usage: " + std::string(argv[0]) + " [options]" << std::endl;
    std::cerr << cmdline_options_ << std::endl;
    exit(0);
  }

  std::string configPath;
  if(vm_.count("config")) {
    configPath = vm_["config"].as<std::string>();
    config_ = YAML::Load(InputFileStream(configPath));
  } else if(!translate && boost::filesystem::exists(
                              vm_["model"].as<std::string>() + ".yml")
            && !vm_["no-reload"].as<bool>()) {
    configPath = vm_["model"].as<std::string>() + ".yml";
    config_ = YAML::Load(InputFileStream(configPath));
  }

  /** model **/

  if(!translate) {
    SET_OPTION("model", std::string);
  } else {
    SET_OPTION("models", std::vector<std::string>);
  }

  if(!vm_["vocabs"].empty()) {
    config_["vocabs"] = vm_["vocabs"].as<std::vector<std::string>>();
  }

  SET_OPTION("type", std::string);
  SET_OPTION("dim-vocabs", std::vector<int>);
  SET_OPTION("dim-emb", int);
  SET_OPTION("dim-pos", int);
  SET_OPTION("dim-rnn", int);
  SET_OPTION("cell-enc", std::string);
  SET_OPTION("cell-dec", std::string);
  SET_OPTION("layers-enc", int);
  SET_OPTION("layers-dec", int);
  SET_OPTION("skip", bool);
  SET_OPTION("tied-embeddings", bool);
  SET_OPTION("layer-normalization", bool);
  SET_OPTION_NONDEFAULT("special-vocab", std::vector<size_t>);

  if(!translate && !rescore) {
    SET_OPTION("dropout-rnn", float);
    SET_OPTION("dropout-src", float);
    SET_OPTION("dropout-trg", float);
  }
  /** model **/

  /** training start **/
  if(!translate && !rescore) {
    SET_OPTION("overwrite", bool);
    SET_OPTION("no-reload", bool);
    if(!vm_["train-sets"].empty()) {
      config_["train-sets"] = vm_["train-sets"].as<std::vector<std::string>>();
    }
    SET_OPTION("after-epochs", size_t);
    SET_OPTION("after-batches", size_t);
    SET_OPTION("disp-freq", size_t);
    SET_OPTION("save-freq", size_t);
    SET_OPTION("no-shuffle", bool);
    SET_OPTION("tempdir", std::string);

    SET_OPTION("optimizer", std::string);
    SET_OPTION("learn-rate", double);
    SET_OPTION("mini-batch-words", int);
    SET_OPTION("dynamic-batching", bool);

    SET_OPTION("lr-decay", double);
    SET_OPTION("lr-decay-strategy", std::string);
    SET_OPTION("lr-decay-start", std::vector<size_t>);
    SET_OPTION("lr-decay-freq", size_t);

    SET_OPTION("clip-norm", double);
    SET_OPTION("moving-average", bool);
    SET_OPTION("moving-decay", double);
    // SET_OPTION_NONDEFAULT("lexical-table", std::string);

    SET_OPTION_NONDEFAULT("guided-alignment", std::string);
    SET_OPTION("guided-alignment-cost", std::string);
    SET_OPTION("guided-alignment-weight", double);
    SET_OPTION("drop-rate", double);
  }
  /** training end **/
  else if(rescore) {
    SET_OPTION("no-reload", bool);
    if(!vm_["train-sets"].empty()) {
      config_["train-sets"] = vm_["train-sets"].as<std::vector<std::string>>();
    }
    SET_OPTION("mini-batch-words", int);
    SET_OPTION("dynamic-batching", bool);
  }
  /** translation start **/
  else {
    SET_OPTION("input", std::vector<std::string>);
    SET_OPTION("normalize", bool);
    SET_OPTION("n-best", bool);
    SET_OPTION("beam-size", size_t);
    SET_OPTION("allow-unk", bool);
    SET_OPTION_NONDEFAULT("weights", std::vector<float>);
    // SET_OPTION_NONDEFAULT("lexical-table", std::string);
  }

  /** valid **/
  if(!translate && !rescore) {
    if(!vm_["valid-sets"].empty()) {
      config_["valid-sets"] = vm_["valid-sets"].as<std::vector<std::string>>();
    }
    SET_OPTION_NONDEFAULT("valid-sets", std::vector<std::string>);
    SET_OPTION("valid-freq", size_t);
    SET_OPTION("valid-metrics", std::vector<std::string>);
    SET_OPTION_NONDEFAULT("valid-script-path", std::string);
    SET_OPTION("early-stopping", size_t);
    SET_OPTION("keep-best", bool);
    SET_OPTION_NONDEFAULT("valid-log", std::string);

    // SET_OPTION("normalize", bool);
    // SET_OPTION("beam-size", size_t);
    // SET_OPTION("allow-unk", bool);
  }
  /** valid **/

  if(doValidate) {
    try {
      validateOptions(translate, rescore);
    } catch(util::Exception& e) {
      std::cerr << "Error: " << e.what() << std::endl << std::endl;

      std::cerr << "Usage: " + std::string(argv[0]) + " [options]" << std::endl;
      std::cerr << cmdline_options_ << std::endl;
      exit(1);
    }
  }

  SET_OPTION("workspace", size_t);
  SET_OPTION_NONDEFAULT("log", std::string);
  SET_OPTION("seed", size_t);
  SET_OPTION("relative-paths", bool);
  SET_OPTION("devices", std::vector<int>);
  SET_OPTION("mini-batch", int);
  SET_OPTION("maxi-batch", int);
  SET_OPTION("max-length", size_t);

  if(get<bool>("relative-paths") && !vm_["dump-config"].as<bool>())
    ProcessPaths(
        config_, boost::filesystem::path{configPath}.parent_path(), false);
  if(vm_["dump-config"].as<bool>()) {
    YAML::Emitter emit;
    OutputRec(config_, emit);
    std::cout << emit.c_str() << std::endl;
    exit(0);
  }

  if(vm_["seed"].as<size_t>() == 0)
    seed = (size_t)time(0);
  else
    seed = vm_["seed"].as<size_t>();

  if(!translate) {
    if(boost::filesystem::exists(vm_["model"].as<std::string>())
       && !vm_["no-reload"].as<bool>()) {
      try {
        loadModelParameters(vm_["model"].as<std::string>());
      } catch(std::runtime_error& e) {
        // @TODO do this with log
        LOG(info, "No model settings found in model file");
      }
    }
  } else {
    auto models = vm_["models"].as<std::vector<std::string>>();
    auto model = models[0];
    try {
      loadModelParameters(model);
    } catch(std::runtime_error& e) {
      LOG(info, "No model settings found in model file");
    }
  }
}

void Config::log() {
  createLoggers(this);

  YAML::Emitter out;
  OutputRec(config_, out);
  std::string conf = out.c_str();

  std::vector<std::string> results;
  boost::algorithm::split(results, conf, boost::is_any_of("\n"));
  for(auto& r : results)
    LOG(config, r);
}

void Config::override(const YAML::Node& params) {
  // YAML::Emitter out;
  // OutputRec(params, out);
  // std::string conf = out.c_str();
  //
  // std::vector<std::string> results;
  // boost::algorithm::split(results, conf, boost::is_any_of("\n"));
  //
  // LOG(config, "Overriding model parameters:");
  // for(auto &r : results)
  //  LOG(config, r);

  for(auto& it : params) {
    config_[it.first.as<std::string>()] = it.second;
  }
}

YAML::Node Config::getModelParameters() {
  YAML::Node modelParams;
  for(auto& key : modelFeatures_)
    modelParams[key] = config_[key];
  return modelParams;
}

void Config::loadModelParameters(const std::string& name) {
  YAML::Node config;
  GetYamlFromNpz(config, "special:model.yml", name);
  override(config);
}

void Config::GetYamlFromNpz(YAML::Node& yaml,
                            const std::string& varName,
                            const std::string& fName) {
  yaml = YAML::Load(cnpy::npz_load(fName, varName).data);
}

void Config::saveModelParameters(const std::string& name) {
  AddYamlToNpz(getModelParameters(), "special:model.yml", name);
}

void Config::AddYamlToNpz(const YAML::Node& yaml,
                          const std::string& varName,
                          const std::string& fName) {
  YAML::Emitter out;
  OutputRec(yaml, out);
  unsigned shape = out.size() + 1;
  cnpy::npz_save(fName, varName, out.c_str(), &shape, 1, "a");
}
}
