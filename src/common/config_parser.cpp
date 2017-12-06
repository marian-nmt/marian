#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <set>
#include <string>

#include "3rd_party/cnpy/cnpy.h"
#include "common/config.h"
#include "common/config_parser.h"
#include "common/file_stream.h"
#include "common/logging.h"
#include "common/version.h"

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
  uint16_t cols = 0;
#ifdef TIOCGSIZE
  struct ttysize ts;
  ioctl(STDIN_FILENO, TIOCGSIZE, &ts);
  if(ts.ts_cols != 0)
    cols = ts.ts_cols;
#elif defined(TIOCGWINSZ)
  struct winsize ts;
  ioctl(STDIN_FILENO, TIOCGWINSZ, &ts);
  if(ts.ws_col != 0)
    cols = ts.ws_col;
#endif
  if(cols == 0)  // couldn't determine terminal width
    cols = po::options_description::m_default_line_length;
  return max_width ? std::min(cols, max_width) : cols;
}

void OutputYaml(const YAML::Node node, YAML::Emitter& out) {
  // std::set<std::string> flow = { "devices" };
  std::set<std::string> sorter;
  switch(node.Type()) {
    case YAML::NodeType::Null: out << node; break;
    case YAML::NodeType::Scalar: out << node; break;
    case YAML::NodeType::Sequence:
      out << YAML::BeginSeq;
      for(auto&& n : node)
        OutputYaml(n, out);
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
        OutputYaml(node[key], out);
      }
      out << YAML::EndMap;
      break;
    case YAML::NodeType::Undefined: out << node; break;
  }
}

const std::set<std::string> PATHS = {"model",
                                     "models",
                                     "train-sets",
                                     "vocabs",
                                     "embedding-vectors",
                                     "valid-sets",
                                     "valid-script-path",
                                     "valid-log",
                                     "valid-translation-output",
                                     "log"};

void ProcessPaths(YAML::Node& node,
                  const boost::filesystem::path& configPath,
                  bool isPath) {
  using namespace boost::filesystem;

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
          ProcessPaths(sub.second, configPath, PATHS.count(key) > 0);
        }
        break;
    }
  }
}

bool ConfigParser::has(const std::string& key) const {
  return config_[key];
}

void ConfigParser::validateOptions() const {
  //UTIL_THROW_IF2(!has("vocabs"), "No vocabularies provided");

  if(mode_ == ConfigMode::translating) {
    return;
  }

  UTIL_THROW_IF2(
      !has("train-sets") || get<std::vector<std::string>>("train-sets").empty(),
      "No train sets given in config file or on command line");
  UTIL_THROW_IF2(
      has("vocabs")
          && get<std::vector<std::string>>("vocabs").size()
                 != get<std::vector<std::string>>("train-sets").size(),
      "There should be as many vocabularies as training sets");
  UTIL_THROW_IF2(
      has("embedding-vectors")
          && get<std::vector<std::string>>("embedding-vectors").size()
                 != get<std::vector<std::string>>("train-sets").size(),
      "There should be as many files with embedding vectors as "
      "training sets");

  if(mode_ == ConfigMode::rescoring)
    return;

  boost::filesystem::path modelPath(get<std::string>("model"));
  auto modelDir = modelPath.parent_path();
  if(modelDir.empty())
    modelDir = boost::filesystem::current_path();

  UTIL_THROW_IF2(
      !modelDir.empty() && !boost::filesystem::is_directory(modelDir),
      "Model directory does not exist");

  UTIL_THROW_IF2(!modelDir.empty() && !(boost::filesystem::status(modelDir).permissions()
                 & boost::filesystem::owner_write),
                 "No write permission in model directory");

  UTIL_THROW_IF2(
      has("valid-sets")
          && get<std::vector<std::string>>("valid-sets").size()
                 != get<std::vector<std::string>>("train-sets").size(),
      "There should be as many validation sets as training sets");

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

void ConfigParser::addOptionsCommon(po::options_description& desc) {
  int defaultWorkspace = (mode_ == ConfigMode::translating) ? 512 : 2048;

  po::options_description general("General options", guess_terminal_width());
  // clang-format off
  general.add_options()
    ("config,c", po::value<std::string>(),
     "Configuration file")
    ("workspace,w", po::value<size_t>()->default_value(defaultWorkspace),
      "Preallocate  arg  MB of work space")
    ("log", po::value<std::string>(),
     "Log training process information to file given by  arg")
    ("log-level", po::value<std::string>()->default_value("info"),
     "Set verbosity level of logging "
     "(trace - debug - info - warn - err(or) - critical - off)")
    ("quiet", po::value<bool>()->zero_tokens()->default_value(false),
     "Suppress all logging to stderr. Logging to files still works")
    ("quiet-translation", po::value<bool>()->zero_tokens()->default_value(false),
     "Suppress logging for translation")
    ("seed", po::value<size_t>()->default_value(0),
     "Seed for all random number generators. 0 means initialize randomly")
    ("relative-paths", po::value<bool>()->zero_tokens()->default_value(false),
     "All paths are relative to the config file location")
    ("dump-config", po::value<bool>()->zero_tokens()->default_value(false),
     "Dump current (modified) configuration to stdout and exit")
    ("version", po::value<bool>()->zero_tokens()->default_value(false),
      "Print version number and exit")
    ("help,h", po::value<bool>()->zero_tokens()->default_value(false),
      "Print this help message and exit")
  ;
  // clang-format on
  desc.add(general);
}

void ConfigParser::addOptionsModel(po::options_description& desc) {
  po::options_description model("Model options", guess_terminal_width());
  // clang-format off
  if(mode_ == ConfigMode::translating) {
    model.add_options()
    ("models,m", po::value<std::vector<std::string>>()->multitoken(),
     "Paths to model(s) to be loaded");
  } else {
    model.add_options()
      ("model,m", po::value<std::string>()->default_value("model.npz"),
      "Path prefix for model to be saved/resumed");
  }

  model.add_options()
    ("ignore-model-config", po::value<bool>()->zero_tokens()->default_value(false),
     "Ignore the model configuration saved in npz file")
    ("type", po::value<std::string>()->default_value("amun"),
      "Model type (possible values: amun, nematus, s2s, multi-s2s, transformer)")
    ("dim-vocabs", po::value<std::vector<int>>()
      ->multitoken()
      ->default_value(std::vector<int>({0, 0}), "0 0"),
     "Maximum items in vocabulary ordered by rank, 0 uses all items in the provided/created vocabulary file")
    ("dim-emb", po::value<int>()->default_value(512),
     "Size of embedding vector")
    ("dim-rnn", po::value<int>()->default_value(1024),
     "Size of rnn hidden state")
    ("enc-type", po::value<std::string>()->default_value("bidirectional"),
     "Type of encoder RNN : bidirectional, bi-unidirectional, alternating (s2s)")
    ("enc-cell", po::value<std::string>()->default_value("gru"),
     "Type of RNN cell: gru, lstm, tanh (s2s)")
    ("enc-cell-depth", po::value<int>()->default_value(1),
     "Number of tansitional cells in encoder layers (s2s)")
    ("enc-depth", po::value<int>()->default_value(1),
     "Number of encoder layers (s2s)")
    ("dec-cell", po::value<std::string>()->default_value("gru"),
     "Type of RNN cell: gru, lstm, tanh (s2s)")
    ("dec-cell-base-depth", po::value<int>()->default_value(2),
     "Number of tansitional cells in first decoder layer (s2s)")
    ("dec-cell-high-depth", po::value<int>()->default_value(1),
     "Number of tansitional cells in next decoder layers (s2s)")
    ("dec-depth", po::value<int>()->default_value(1),
     "Number of decoder layers (s2s)")
    //("dec-high-context", po::value<std::string>()->default_value("none"),
    // "Repeat attended context: none, repeat, conditional, conditional-repeat (s2s)")
    ("skip", po::value<bool>()->zero_tokens()->default_value(false),
     "Use skip connections (s2s)")
    ("layer-normalization", po::value<bool>()->zero_tokens()->default_value(false),
     "Enable layer normalization")
    ("best-deep", po::value<bool>()->zero_tokens()->default_value(false),
     "Use Edinburgh deep RNN configuration (s2s)")
    ("special-vocab", po::value<std::vector<size_t>>()->multitoken(),
     "Model-specific special vocabulary ids")
    ("tied-embeddings", po::value<bool>()->zero_tokens()->default_value(false),
     "Tie target embeddings and output embeddings in output layer")
    ("tied-embeddings-src", po::value<bool>()->zero_tokens()->default_value(false),
     "Tie source and target embeddings")
    ("tied-embeddings-all", po::value<bool>()->zero_tokens()->default_value(false),
     "Tie all embedding layers and output layer")
    ("transformer-heads", po::value<int>()->default_value(8),
     "Number of head in multi-head attention (transformer)")
    ("transformer-dim-ffn", po::value<int>()->default_value(2048),
     "Size of position-wise feed-forward network (transformer)")
    ("transformer-preprocess", po::value<std::string>()->default_value(""),
     "Operation before each transformer layer: d = dropout, a = add, n = normalize")
    ("transformer-postprocess-emb", po::value<std::string>()->default_value("d"),
     "Operation after transformer embedding layer: d = dropout, a = add, n = normalize")
    ("transformer-postprocess", po::value<std::string>()->default_value("dan"),
     "Operation after each transformer layer: d = dropout, a = add, n = normalize")
#ifdef CUDNN
    ("char-stride", po::value<int>()->default_value(5),
     "Width of max-pooling layer after convolution layer in char-s2s model")
    ("char-highway", po::value<int>()->default_value(4),
     "Number of highway network layers after max-pooling in char-s2s model")
    ("char-conv-filters-num", po::value<std::vector<int>>()
      ->default_value(std::vector<int>({200, 200, 250, 250, 300, 300, 300, 300}),
                                      "200 200 250 250 300 300 300 300")
      ->multitoken(),
     "Numbers of convolution filters of correspoding width in char-s2s model")
    ("char-conv-filters-widths", po::value<std::vector<int>>()
     ->default_value(std::vector<int>({1, 2, 3, 4, 5, 6, 7, 8}), "1 2 3 4 5 6 7 8")
      ->multitoken(),
     "Convolution window widths in char-s2s model")
#endif
    ;

  if(mode_ == ConfigMode::training) {
    model.add_options()
      ("dropout-rnn", po::value<float>()->default_value(0),
       "Scaling dropout along rnn layers and time (0 = no dropout)")
      ("dropout-src", po::value<float>()->default_value(0),
       "Dropout source words (0 = no dropout)")
      ("dropout-trg", po::value<float>()->default_value(0),
       "Dropout target words (0 = no dropout)")
      ("gradient-dropping", po::value<float>()->default_value(0),
       "Gradient Dropping rate (0 = no gradient Dropping)")
      ("transformer-dropout", po::value<float>()->default_value(0),
       "Dropout between transformer layers (0 = no dropout)")
      ("transformer-dropout-attention", po::value<float>()->default_value(0),
       "Dropout for transformer attention (0 = no dropout)")
    ;
  }
  // clang-format on

  desc.add(model);
}

void ConfigParser::addOptionsTraining(po::options_description& desc) {
  po::options_description training("Training options", guess_terminal_width());
  // clang-format off
  training.add_options()
    ("cost-type", po::value<std::string>()->default_value("ce-mean"),
      "Optimization criterion: ce-mean, ce-mean-words, ce-sum, perplexity")
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
      "If these files do not exists they are created")
    ("max-length", po::value<size_t>()->default_value(50),
      "Maximum length of a sentence in a training sentence pair")
    ("max-length-crop", po::value<bool>()->zero_tokens()->default_value(false),
      "Crop a sentence to max-length instead of ommitting it if longer than max-length")
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
      "GPUs to use for training. Asynchronous SGD is used with multiple devices")

    ("mini-batch", po::value<int>()->default_value(64),
      "Size of mini-batch used during update")
    ("mini-batch-words", po::value<int>()->default_value(0),
      "Set mini-batch size based on words instead of sentences")
    ("mini-batch-fit", po::value<bool>()->zero_tokens()->default_value(false),
      "Determine mini-batch size automatically based on sentence-length to fit reserved memory")
    ("maxi-batch", po::value<int>()->default_value(100),
      "Number of batches to preload for length-based sorting")
    ("maxi-batch-sort", po::value<std::string>()->default_value("trg"),
      "Sorting strategy for maxi-batch: trg (default) src none")

    ("optimizer,o", po::value<std::string>()->default_value("adam"),
     "Optimization algorithm (possible values: sgd, adagrad, adam")
    ("optimizer-params",  po::value<std::vector<float>>()
       ->multitoken(),
     "Parameters for optimization algorithm, e.g. betas for adam")
    ("optimizer-delay", po::value<size_t>()->default_value(1),
     "SGD update delay, 1 = no delay")
    ("learn-rate,l", po::value<double>()->default_value(0.0001),
     "Learning rate")
    ("lr-decay", po::value<double>()->default_value(0.0),
     "Decay factor for learning rate: lr = lr * arg (0 to disable)")
    ("lr-decay-strategy", po::value<std::string>()->default_value("epoch+stalled"),
     "Strategy for learning rate decaying "
     "(possible values: epoch, batches, stalled, epoch+batches, epoch+stalled)")
    ("lr-decay-start", po::value<std::vector<size_t>>()
       ->multitoken()
       ->default_value(std::vector<size_t>({10,1}), "10 1"),
       "The first number of epoch/batches/stalled validations to start "
       "learning rate decaying")
    ("lr-decay-freq", po::value<size_t>()->default_value(50000),
     "Learning rate decaying frequency for batches, "
     "requires --lr-decay-strategy to be batches")
    ("lr-decay-reset-optimizer", po::value<bool>()->zero_tokens()->default_value(false),
      "Reset running statistics of optimizer whenever learning rate decays")

    ("lr-decay-repeat-warmup", po::value<bool>()
     ->zero_tokens()->default_value(false),
     "Repeat learning rate warmup when learning rate is decayed")
    ("lr-decay-inv-sqrt", po::value<size_t>()->default_value(0),
     "Decrease learning rate at arg / sqrt(no. updates) starting at arg")

    ("lr-warmup", po::value<size_t>()->default_value(0),
     "Increase learning rate linearly for arg first steps")
    ("lr-warmup-start-rate", po::value<float>()->default_value(0),
     "Start value for learning rate warmup")
    ("lr-warmup-cycle", po::value<bool>()->zero_tokens()->default_value(false),
     "Apply cyclic warmup")
    ("lr-warmup-at-reload", po::value<bool>()->zero_tokens()->default_value(false),
     "Repeat warmup after interrupted training")

    ("lr-report", po::value<bool>()
     ->zero_tokens()->default_value(false),
     "Report learning rate for each update")

    ("batch-flexible-lr", po::value<bool>()->zero_tokens()->default_value(false),
      "Scales the learning rate based on the number of words in a mini-batch")
    ("batch-normal-words", po::value<double>()->default_value(1920.0),
      "Set number of words per batch that the learning rate corresponds to. "
      "The option is only active when batch-flexible-lr is on")
    ("sync-sgd", po::value<bool>()->zero_tokens()->default_value(false),
     "Use synchronous SGD instead of asynchronous for multi-gpu training")
    ("label-smoothing", po::value<double>()->default_value(0),
     "Epsilon for label smoothing (0 to disable)")
    ("clip-norm", po::value<double>()->default_value(1.f),
     "Clip gradient norm to  arg  (0 to disable)")
    ("exponential-smoothing", po::value<float>()->default_value(0.f)->implicit_value(1e-4, "1e-4"),
     "Maintain smoothed version of parameters for validation and saving with smoothing factor arg. "
     " 0 to disable.")
    ("guided-alignment", po::value<std::string>(),
     "Use guided alignment to guide attention")
    ("guided-alignment-cost", po::value<std::string>()->default_value("ce"),
     "Cost type for guided alignment. Possible values: ce (cross-entropy), "
     "mse (mean square error), mult (multiplication)")
    ("guided-alignment-weight", po::value<double>()->default_value(1),
     "Weight for guided alignment cost")

    //("drop-rate", po::value<double>()->default_value(0),
    // "Gradient drop ratio (read: https://arxiv.org/abs/1704.05021)")
    ("embedding-vectors", po::value<std::vector<std::string>>()
      ->multitoken(),
     "Paths to files with custom source and target embedding vectors")
    ("embedding-normalization", po::value<bool>()
      ->zero_tokens()
      ->default_value(false),
     "Enable normalization of custom embedding vectors")
    ("embedding-fix-src", po::value<bool>()
      ->zero_tokens()
      ->default_value(false),
     "Fix source embeddings. Affects all encoders")
    ("embedding-fix-trg", po::value<bool>()
      ->zero_tokens()
      ->default_value(false),
     "Fix target embeddings. Affects all decoders")
  ;
  // clang-format on
  desc.add(training);
}

void ConfigParser::addOptionsValid(po::options_description& desc) {
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
      "Metric to use during validation: cross-entropy, perplexity, "
      "valid-script, translation. "
      "Multiple metrics can be specified")
    ("valid-mini-batch", po::value<int>()->default_value(32),
      "Size of mini-batch used during validation")
    ("valid-max-length", po::value<size_t>()->default_value(1000),
      "Maximum length of a sentence in a validating sentence pair")
    ("valid-script-path", po::value<std::string>(),
     "Path to external validation script. "
     "It should print a single score to stdout. "
     "If the option is used with validating translation, the output "
     "translation file will be passed as a first argument ")
    ("early-stopping", po::value<size_t>()->default_value(10),
     "Stop if the first validation metric does not improve for  arg  consecutive "
     "validation steps")
    ("keep-best", po::value<bool>()->zero_tokens()->default_value(false),
      "Keep best model for each validation metric")
    ("valid-log", po::value<std::string>(),
     "Log validation scores to file given by  arg")

    ("valid-translation-output", po::value<std::string>(),
     "Path to store the translation")
    ("beam-size,b", po::value<size_t>()->default_value(12),
      "Beam size used during search with validating translator")
    ("normalize,n", po::value<float>()->default_value(0.f)->implicit_value(1.f),
      "Divide translation score by pow(translation length, arg) ")
    ("allow-unk", po::value<bool>()->zero_tokens()->default_value(false),
      "Allow unknown words to appear in output")
    ("n-best", po::value<bool>()->zero_tokens()->default_value(false),
      "Generate n-best list")
  ;
  // clang-format on
  desc.add(valid);
}

void ConfigParser::addOptionsTranslate(po::options_description& desc) {
  po::options_description translate("Translator options",
                                    guess_terminal_width());
  // clang-format off
  translate.add_options()
    ("input,i", po::value<std::vector<std::string>>()
      ->multitoken()
      ->default_value(std::vector<std::string>({"stdin"}), "stdin"),
      "Paths to input file(s), stdin by default")
    ("vocabs,v", po::value<std::vector<std::string>>()->multitoken(),
      "Paths to vocabulary files have to correspond to --input")
    ("beam-size,b", po::value<size_t>()->default_value(12),
      "Beam size used during search")
    ("normalize,n", po::value<float>()->default_value(0.f)->implicit_value(1.f),
      "Divide translation score by pow(translation length, arg) ")
    ("allow-unk", po::value<bool>()->zero_tokens()->default_value(false),
      "Allow unknown words to appear in output")
    ("max-length", po::value<size_t>()->default_value(1000),
      "Maximum length of a sentence in a training sentence pair")
    ("max-length-crop", po::value<bool>()->zero_tokens()->default_value(false),
      "Crop a sentence to max-length instead of ommitting it if longer than max-length")
    ("devices,d", po::value<std::vector<int>>()
      ->multitoken()
      ->default_value(std::vector<int>({0}), "0"),
      "GPUs to use for translating")
    ("mini-batch", po::value<int>()->default_value(1),
      "Size of mini-batch used during update")
    ("maxi-batch", po::value<int>()->default_value(1),
      "Number of batches to preload for length-based sorting")
    ("maxi-batch-sort", po::value<std::string>()->default_value("none"),
      "Sorting strategy for maxi-batch: none (default) src")
    ("n-best", po::value<bool>()->zero_tokens()->default_value(false),
      "Display n-best list")
    //("lexical-table", po::value<std::string>(),
    // "Path to lexical table")
    ("weights", po::value<std::vector<float>>()
      ->multitoken(),
      "Scorer weights")
    // TODO: the options should be available only in server
    ("port,p", po::value<size_t>()->default_value(8080),
      "Port number for web socket server")
  ;
  // clang-format on
  desc.add(translate);
}

void ConfigParser::addOptionsRescore(po::options_description& desc) {
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
      "If these files do not exists they are created")
    ("summary", po::value<std::string>()->implicit_value("cross-entropy"),
      "Only print total cost, possible values: cross-entropy (ce-mean), ce-mean-words, ce-sum, perplexity")
    ("max-length", po::value<size_t>()->default_value(1000),
      "Maximum length of a sentence in a training sentence pair")
    ("max-length-crop", po::value<bool>()->zero_tokens()->default_value(false),
      "Crop a sentence to max-length instead of ommitting it if longer than max-length")
    ("devices,d", po::value<std::vector<int>>()
      ->multitoken()
      ->default_value(std::vector<int>({0}), "0"),
      "GPUs to use for training. Asynchronous SGD is used with multiple devices")

    ("mini-batch", po::value<int>()->default_value(64),
      "Size of mini-batch used during update")
    ("mini-batch-words", po::value<int>()->default_value(0),
      "Set mini-batch size based on words instead of sentences")
    ("mini-batch-fit", po::value<bool>()->zero_tokens()->default_value(false),
      "Determine mini-batch size automatically based on sentence-length to fit reserved memory")
    ("maxi-batch", po::value<int>()->default_value(100),
      "Number of batches to preload for length-based sorting")
    ;
  // clang-format on
  desc.add(rescore);
}

void ConfigParser::parseOptions(int argc, char** argv, bool doValidate) {
  addOptionsCommon(cmdline_options_);
  addOptionsModel(cmdline_options_);

  // clang-format off
  switch(mode_) {
    case ConfigMode::translating:
      addOptionsTranslate(cmdline_options_);
      break;
    case ConfigMode::rescoring:
      addOptionsRescore(cmdline_options_);
      break;
    case ConfigMode::training:
      addOptionsTraining(cmdline_options_);
      addOptionsValid(cmdline_options_);
      break;
  }
  // clang-format on

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

  if(mode_ == ConfigMode::translating) {
    if(vm_.count("models") == 0 && vm_.count("config") == 0) {
      std::cerr << "Error: you need to provide at least one model file or a "
                   "config file"
                << std::endl
                << std::endl;

      std::cerr << "Usage: " + std::string(argv[0]) + " [options]" << std::endl;
      std::cerr << cmdline_options_ << std::endl;
      exit(0);
    }
  }

  if(vm_["version"].as<bool>()) {
    std::cerr << PROJECT_VERSION_FULL << std::endl;
    exit(0);
  }

  std::string configPath;
  if(vm_.count("config")) {
    configPath = vm_["config"].as<std::string>();
    config_ = YAML::Load(InputFileStream(configPath));
  } else if((mode_ == ConfigMode::training)
            && boost::filesystem::exists(vm_["model"].as<std::string>()
                                         + ".yml")
            && !vm_["no-reload"].as<bool>()) {
    configPath = vm_["model"].as<std::string>() + ".yml";
    config_ = YAML::Load(InputFileStream(configPath));
  }

  /** model **/

  if(mode_ == ConfigMode::translating) {
    SET_OPTION_NONDEFAULT("models", std::vector<std::string>);
  } else {
    SET_OPTION("model", std::string);
  }

  if(!vm_["vocabs"].empty()) {
    config_["vocabs"] = vm_["vocabs"].as<std::vector<std::string>>();
  }

  SET_OPTION("ignore-model-config", bool);
  SET_OPTION("type", std::string);
  SET_OPTION("dim-vocabs", std::vector<int>);
  SET_OPTION("dim-emb", int);
  SET_OPTION("dim-rnn", int);

  SET_OPTION("enc-type", std::string);
  SET_OPTION("enc-cell", std::string);
  SET_OPTION("enc-cell-depth", int);
  SET_OPTION("enc-depth", int);

  SET_OPTION("dec-cell", std::string);
  SET_OPTION("dec-cell-base-depth", int);
  SET_OPTION("dec-cell-high-depth", int);
  SET_OPTION("dec-depth", int);

  SET_OPTION("skip", bool);
  SET_OPTION("tied-embeddings", bool);
  SET_OPTION("tied-embeddings-src", bool);
  SET_OPTION("tied-embeddings-all", bool);
  SET_OPTION("layer-normalization", bool);
  SET_OPTION("transformer-heads", int);
  SET_OPTION("transformer-preprocess", std::string);
  SET_OPTION("transformer-postprocess", std::string);
  SET_OPTION("transformer-postprocess-emb", std::string);
  SET_OPTION("transformer-dim-ffn", int);
  SET_OPTION("transformer-dim-ffn", int);

#ifdef CUDNN
  SET_OPTION("char-stride", int);
  SET_OPTION("char-highway", int);
  SET_OPTION("char-conv-filters-num", std::vector<int>);
  SET_OPTION("char-conv-filters-widths", std::vector<int>);
#endif

  SET_OPTION("best-deep", bool);
  SET_OPTION_NONDEFAULT("special-vocab", std::vector<size_t>);

  if(mode_ == ConfigMode::training) {
    SET_OPTION("cost-type", std::string);

    SET_OPTION("dropout-rnn", float);
    SET_OPTION("dropout-src", float);
    SET_OPTION("dropout-trg", float);

    SET_OPTION("gradient-dropping", float);

    SET_OPTION("transformer-dropout", float);
    SET_OPTION("transformer-dropout-attention", float);

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
    SET_OPTION_NONDEFAULT("optimizer-params", std::vector<float>);
    SET_OPTION("optimizer-delay", size_t);
    SET_OPTION("learn-rate", double);
    SET_OPTION("sync-sgd", bool);
    SET_OPTION("mini-batch-words", int);
    SET_OPTION("mini-batch-fit", bool);

    SET_OPTION("lr-decay", double);
    SET_OPTION("lr-decay-strategy", std::string);
    SET_OPTION("lr-decay-start", std::vector<size_t>);
    SET_OPTION("lr-decay-freq", size_t);
    SET_OPTION("lr-decay-reset-optimizer", bool);
    SET_OPTION("lr-warmup", size_t);

    SET_OPTION("lr-decay-inv-sqrt", size_t);
    SET_OPTION("lr-warmup-start-rate", float);
    SET_OPTION("lr-warmup-cycle", bool);

    SET_OPTION("lr-decay-repeat-warmup", bool);
    SET_OPTION("lr-warmup-at-reload", bool);
    SET_OPTION("lr-report", bool);

    SET_OPTION("batch-flexible-lr", bool);
    SET_OPTION("batch-normal-words", double);

    SET_OPTION("label-smoothing", double);
    SET_OPTION("clip-norm", double);
    SET_OPTION("exponential-smoothing", float);

    SET_OPTION_NONDEFAULT("guided-alignment", std::string);
    SET_OPTION("guided-alignment-cost", std::string);
    SET_OPTION("guided-alignment-weight", double);
    // SET_OPTION("drop-rate", double);
    SET_OPTION_NONDEFAULT("embedding-vectors", std::vector<std::string>);
    SET_OPTION("embedding-normalization", bool);
    SET_OPTION("embedding-fix-src", bool);
    SET_OPTION("embedding-fix-trg", bool);
  }
  if(mode_ == ConfigMode::rescoring) {
    SET_OPTION("no-reload", bool);
    if(!vm_["train-sets"].empty()) {
      config_["train-sets"] = vm_["train-sets"].as<std::vector<std::string>>();
    }
    SET_OPTION("mini-batch-words", int);
    SET_OPTION("mini-batch-fit", bool);
    SET_OPTION_NONDEFAULT("summary", std::string);
  }
  if(mode_ == ConfigMode::translating) {
    SET_OPTION("input", std::vector<std::string>);
    SET_OPTION("beam-size", size_t);
    SET_OPTION("normalize", float);
    SET_OPTION("allow-unk", bool);
    SET_OPTION("n-best", bool);
    SET_OPTION_NONDEFAULT("weights", std::vector<float>);
    SET_OPTION("port", size_t);
  }

  /** valid **/
  if(mode_ == ConfigMode::training) {
    if(!vm_["valid-sets"].empty()) {
      config_["valid-sets"] = vm_["valid-sets"].as<std::vector<std::string>>();
    }
    SET_OPTION_NONDEFAULT("valid-sets", std::vector<std::string>);
    SET_OPTION("valid-freq", size_t);
    SET_OPTION("valid-metrics", std::vector<std::string>);
    SET_OPTION("valid-mini-batch", int);
    SET_OPTION("valid-max-length", size_t);
    SET_OPTION_NONDEFAULT("valid-script-path", std::string);
    SET_OPTION("early-stopping", size_t);
    SET_OPTION("keep-best", bool);
    SET_OPTION_NONDEFAULT("valid-log", std::string);

    SET_OPTION_NONDEFAULT("valid-translation-output", std::string);
    SET_OPTION("beam-size", size_t);
    SET_OPTION("normalize", float);
    SET_OPTION("allow-unk", bool);
    SET_OPTION("n-best", bool);
  }

  if(doValidate) {
    try {
      validateOptions();
    } catch(util::Exception& e) {
      std::cerr << "Error: " << e.what() << std::endl << std::endl;

      std::cerr << "Usage: " + std::string(argv[0]) + " [options]" << std::endl;
      std::cerr << cmdline_options_ << std::endl;
      exit(1);
    }
  }

  SET_OPTION("workspace", size_t);
  SET_OPTION("log-level", std::string);
  SET_OPTION("quiet", bool);
  SET_OPTION("quiet-translation", bool);
  SET_OPTION_NONDEFAULT("log", std::string);
  SET_OPTION("seed", size_t);
  SET_OPTION("relative-paths", bool);
  SET_OPTION("devices", std::vector<int>);
  SET_OPTION("mini-batch", int);
  SET_OPTION("maxi-batch", int);

  if(mode_ == ConfigMode::training || mode_ == ConfigMode::translating)
    SET_OPTION("maxi-batch-sort", std::string);
  SET_OPTION("max-length", size_t);
  SET_OPTION("max-length-crop", bool);

  if(vm_["best-deep"].as<bool>()) {
    config_["layer-normalization"] = true;
    config_["tied-embeddings"] = true;
    config_["enc-type"] = "alternating";
    config_["enc-cell-depth"] = 2;
    config_["enc-depth"] = 4;
    config_["dec-cell-base-depth"] = 4;
    config_["dec-cell-high-depth"] = 2;
    config_["dec-depth"] = 4;
    config_["skip"] = true;
  }

  if(get<bool>("relative-paths") && !vm_["dump-config"].as<bool>())
    ProcessPaths(
        config_, boost::filesystem::path{configPath}.parent_path(), false);

  if(vm_["dump-config"].as<bool>()) {
    YAML::Emitter emit;
    OutputYaml(config_, emit);
    std::cout << emit.c_str() << std::endl;
    exit(0);
  }
}

YAML::Node ConfigParser::getConfig() const {
  return config_;
}
}
