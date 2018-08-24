#include <algorithm>
#include <set>
#include <stdexcept>
#include <string>

#include <boost/algorithm/string.hpp>

#if MKL_FOUND
#include <mkl.h>
#else
#if BLAS_FOUND
#include <cblas.h>
#endif
#endif

#include "common/definitions.h"

#include "common/cli_helper.h"
#include "common/config.h"
#include "common/config_parser.h"
#include "common/file_stream.h"
#include "common/logging.h"
#include "common/regex.h"
#include "common/version.h"

namespace marian {

uint16_t guess_terminal_width(uint16_t max_width, uint16_t default_width) {
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
  // couldn't determine terminal width
  if(cols == 0)
    cols = default_width;
  return max_width ? std::min(cols, max_width) : cols;
}

// TODO: move to CLIWrapper
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


bool ConfigParser::has(const std::string& key) const {
  return config_[key];
}

void ConfigParser::validateOptions() const {
  if(mode_ == ConfigMode::translating) {
    UTIL_THROW_IF2(
        !has("models") && get<std::vector<std::string>>("config").empty(),
        "You need to provide at least one model file or a config file");

    UTIL_THROW_IF2(
        !has("vocabs") || get<std::vector<std::string>>("vocabs").empty(),
        "Translating, but vocabularies are not given!");

    for(const auto& modelFile : get<std::vector<std::string>>("models")) {
      boost::filesystem::path modelPath(modelFile);
      UTIL_THROW_IF2(!boost::filesystem::exists(modelPath),
                     "Model file does not exist: " + modelFile);
    }

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

  boost::filesystem::path modelPath(get<std::string>("model"));

  if(mode_ == ConfigMode::rescoring) {
    UTIL_THROW_IF2(!boost::filesystem::exists(modelPath),
                   "Model file does not exist: " + modelPath.string());

    UTIL_THROW_IF2(
        !has("vocabs") || get<std::vector<std::string>>("vocabs").empty(),
        "Scoring, but vocabularies are not given!");

    return;
  }

  auto modelDir = modelPath.parent_path();
  if(modelDir.empty())
    modelDir = boost::filesystem::current_path();

  UTIL_THROW_IF2(
      !modelDir.empty() && !boost::filesystem::is_directory(modelDir),
      "Model directory does not exist");

  UTIL_THROW_IF2(!modelDir.empty()
                     && !(boost::filesystem::status(modelDir).permissions()
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

void ConfigParser::validateDevices() const {
  std::string devices = utils::Join(get<std::vector<std::string>>("devices"));
  utils::Trim(devices);

  regex::regex pattern;
  std::string help;
  if(mode_ == ConfigMode::training && get<bool>("multi-node")) {
    // valid strings: '0: 1 2', '0:1 2 1:2 3'
    pattern = "( *[0-9]+ *: *[0-9]+( *[0-9]+)*)+";
    help = "Supported format for multi-node setting: '0:0 1 2 3 1:0 1 2 3'";
  } else {
    // valid strings: '0', '0 1 2 3', '3 2 0 1'
    pattern = "[0-9]+( *[0-9]+)*";
    help = "Supported formats: '0 1 2 3'";
  }

  UTIL_THROW_IF2(!regex::regex_match(devices, pattern),
                 "the argument '(" + devices
                     + ")' for option '--devices' is invalid. " + help);
}

void ConfigParser::addOptionsCommon(cli::CLIWrapper& cli) {
  int defaultWorkspace = (mode_ == ConfigMode::translating) ? 512 : 2048;

  cli.startGroup("General options");

  // clang-format off
  cli.add_nondefault<bool>("--version",
      "Print version number and exit");
  cli.add<std::vector<std::string>>("--config,-c",
     "Configuration file(s). If multiple, later overrides earlier");
  cli.add<size_t>("--workspace,-w",
      "Preallocate  arg  MB of work space",
      defaultWorkspace);
  cli.add_nondefault<std::string>("--log",
     "Log training process information to file given by  arg");
  cli.add<std::string>("--log-level",
     "Set verbosity level of logging: trace, debug, info, warn, err(or), critical, off",
     "info");
  cli.add<bool>("--quiet",
     "Suppress all logging to stderr. Logging to files still works");
  cli.add<bool>("--quiet-translation",
     "Suppress logging for translation");
  cli.add<size_t>("--seed",
     "Seed for all random number generators. 0 means initialize randomly");
  cli.add<float>("--clip-gemm",
     "If not 0 clip GEMM input values to +/- arg");
  cli.add<bool>("--interpolate-env-vars",
     "allow the use of environment variables in paths, of the form ${VAR_NAME}");
  cli.add<bool>("--relative-paths",
     "All paths are relative to the config file location");
  cli.add<bool>("--dump-config",
     "Dump current (modified) configuration to stdout and exit");
  // clang-format on
  cli.endGroup();
}

void ConfigParser::addOptionsModel(cli::CLIWrapper& cli) {
  cli.startGroup("Model options");

  // clang-format off
  if(mode_ == ConfigMode::translating) {
    cli.add_nondefault<std::vector<std::string>>("--models,-m",
      "Paths to model(s) to be loaded");
  } else {
    cli.add<std::string>("--model,-m",
      "Path prefix for model to be saved/resumed",
      "model.npz");

    if(mode_ == ConfigMode::training) {
      cli.add_nondefault<std::string>("--pretrained-model",
        "Path prefix for pre-trained model to initialize model weights");
    }
  }

  cli.add<bool>("--ignore-model-config",
      "Ignore the model configuration saved in npz file");
  cli.add<std::string>("--type",
      "Model type: amun, nematus, s2s, multi-s2s, transformer",
      "amun");
  cli.add<std::vector<int>>("--dim-vocabs",
      "Maximum items in vocabulary ordered by rank, 0 uses all items in the provided/created vocabulary file",
      std::vector<int>({0, 0}));
  cli.add<int>("--dim-emb",
      "Size of embedding vector",
      512);
  cli.add<int>("--dim-rnn",
      "Size of rnn hidden state", 1024);
  cli.add<std::string>("--enc-type",
      "Type of encoder RNN : bidirectional, bi-unidirectional, alternating (s2s)",
      "bidirectional");
  cli.add<std::string>("--enc-cell",
      "Type of RNN cell: gru, lstm, tanh (s2s)", "gru");
  cli.add<int>("--enc-cell-depth",
      "Number of transitional cells in encoder layers (s2s)",
      1);
  cli.add<int>("--enc-depth",
      "Number of encoder layers (s2s)",
      1);
  cli.add<std::string>("--dec-cell",
      "Type of RNN cell: gru, lstm, tanh (s2s)",
      "gru");
  cli.add<int>("--dec-cell-base-depth",
      "Number of transitional cells in first decoder layer (s2s)",
      2);
  cli.add<int>("--dec-cell-high-depth",
      "Number of transitional cells in next decoder layers (s2s)",
      1);
  cli.add<int>("--dec-depth",
      "Number of decoder layers (s2s)",
      1);
  cli.add<bool>("--skip",
      "Use skip connections (s2s)");
  cli.add<bool>("--layer-normalization",
      "Enable layer normalization");
  cli.add<bool>("--right-left",
      "Train right-to-left model");
  cli.add<bool>("--best-deep",
      "Use Edinburgh deep RNN configuration (s2s)");
  cli.add_nondefault<std::vector<size_t>>("--special-vocab",
      "Model-specific special vocabulary ids");
  cli.add<bool>("--tied-embeddings",
      "Tie target embeddings and output embeddings in output layer");
  cli.add<bool>("--tied-embeddings-src",
      "Tie source and target embeddings");
  cli.add<bool>("--tied-embeddings-all",
      "Tie all embedding layers and output layer");

  // Transformer options
  cli.add<int>("--transformer-heads",
      "Number of heads in multi-head attention (transformer)",
      8);
  cli.add<bool>("--transformer-no-projection",
      "Omit linear projection after multi-head attention (transformer)");
  cli.add<int>("--transformer-dim-ffn",
      "Size of position-wise feed-forward network (transformer)",
      2048);
  cli.add<int>("--transformer-ffn-depth",
      "Depth of filters (transformer)",
      2);
  cli.add<std::string>("--transformer-ffn-activation",
      "Activation between filters: swish or relu (transformer)",
      "swish");
  cli.add<int>("--transformer-dim-aan",
      "Size of position-wise feed-forward network in AAN (transformer)",
      2048);
  cli.add<int>("--transformer-aan-depth",
      "Depth of filter for AAN (transformer)",
      2);
  cli.add<std::string>("--transformer-aan-activation",
      "Activation between filters in AAN: swish or relu (transformer)",
      "swish");
  cli.add<bool>("--transformer-aan-nogate",
      "Omit gate in AAN (transformer)");
  cli.add<std::string>("--transformer-decoder-autoreg",
      "Type of autoregressive layer in transformer decoder: self-attention, average-attention (transformer)",
      "self-attention");
  cli.add<std::vector<size_t>>("--transformer-tied-layers",
      "List of tied decoder layers (transformer)");
  cli.add<std::string>("--transformer-preprocess",
      "Operation before each transformer layer: d = dropout, a = add, n = normalize");
  cli.add<std::string>("--transformer-postprocess-emb",
      "Operation after transformer embedding layer: d = dropout, a = add, n = normalize",
      "d");
  cli.add<std::string>("--transformer-postprocess",
      "Operation after each transformer layer: d = dropout, a = add, n = normalize",
      "dan");
#ifdef CUDNN
  cli.add<int>("--char-stride",
      "Width of max-pooling layer after convolution layer in char-s2s model",
      5);
  cli.add<int>("--char-highway",
      "Number of highway network layers after max-pooling in char-s2s model",
      4)
  cli.add<std::vector<int>>("--char-conv-filters-num",
      "Numbers of convolution filters of correspoding width in char-s2s model",
      std::vector<int>({200, 200, 250, 250, 300, 300, 300, 300}));
  cli.add<std::vector<int>>("--char-conv-filters-widths",
      "Convolution window widths in char-s2s model",
      std::vector<int>({1, 2, 3, 4, 5, 6, 7, 8}));
#endif

  if(mode_ == ConfigMode::training) {
    // TODO: add ->range(0,1);
    cli.add<float>("--dropout-rnn",
        "Scaling dropout along rnn layers and time (0 = no dropout)");
    cli.add<float>("--dropout-src",
        "Dropout source words (0 = no dropout)");
    cli.add<float>("--dropout-trg",
        "Dropout target words (0 = no dropout)");
    cli.add<float>("--grad-dropping-rate",
        "Gradient Dropping rate (0 = no gradient Dropping)");
    cli.add<float>("--grad-dropping-momentum",
        "Gradient Dropping momentum decay rate (0.0 to 1.0)");
    cli.add<size_t>("--grad-dropping-warmup",
        "Do not apply gradient dropping for the first arg steps",
        100);
    cli.add<float>("--transformer-dropout",
        "Dropout between transformer layers (0 = no dropout)");
    cli.add<float>("--transformer-dropout-attention",
        "Dropout for transformer attention (0 = no dropout)");
    cli.add<float>("--transformer-dropout-ffn",
        "Dropout for transformer filter (0 = no dropout)");
  }
  // clang-format on

  cli.endGroup();
}

void ConfigParser::addOptionsTraining(cli::CLIWrapper &cli) {
  cli.startGroup("Training options");
  // clang-format off
  cli.add<std::string>("--cost-type",
      "Optimization criterion: ce-mean, ce-mean-words, ce-sum, perplexity", "ce-mean");
  cli.add<bool>("--overwrite",
      "Overwrite model with following checkpoints");
  cli.add<bool>("--no-reload",
      "Do not load existing model specified in --model arg");
  cli.add<std::vector<std::string>>("--train-sets,-t",
      "Paths to training corpora: source target");
  cli.add_nondefault<std::vector<std::string>>("--vocabs,-v",
      "Paths to vocabulary files have to correspond to --train-sets. "
      "If this parameter is not supplied we look for vocabulary files "
      "source.{yml,json} and target.{yml,json}. "
      "If these files do not exist they are created");

  // scheduling options
  cli.add<size_t>("--after-epoch,-e",
      "Finish after this many epochs, 0 is infinity");
  cli.add<size_t>("--after-batches",
      "Finish after this many batch updates, 0 is infinity");
  cli.add<size_t>("--disp-freq",
      "Display information every  arg  updates",
      1000);
  cli.add<bool>("--disp-label-counts",
      "Display label counts when logging loss progress");
  cli.add<size_t>("--save-freq",
      "Save model file every  arg  updates",
      10000);

  // data management options
  cli.add<size_t>("--max-length",
      "Maximum length of a sentence in a training sentence pair",
      50);
  cli.add<bool>("--max-length-crop",
      "Crop a sentence to max-length instead of ommitting it if longer than max-length");
  cli.add<bool>("--no-shuffle",
      "Skip shuffling of training data before each epoch");
  cli.add<bool>("--no-restore-corpus",
      "Skip restoring corpus state after training is restarted");
  cli.add<std::string>("--tempdir,-T",
      "Directory for temporary (shuffled) files and database",
      "/tmp");
  cli.add<std::string>("--sqlite",
      "Use disk-based sqlite3 database for training corpus storage, default"
      " is temporary with path creates persistent storage")
    ->default_val("")->implicit_val("temporary");
  cli.add<bool>("--sqlite-drop",
      "Drop existing tables in sqlite3 database");

  // thread options
  cli.add<std::vector<std::string>>("--devices,-d",
      "GPU ID(s) to use for training",
      std::vector<std::string>({"0"}));
#ifdef USE_NCCL
  cli.add<bool>("--no-nccl",
     "Disable inter-GPU communication via NCCL");
#endif
#ifdef CUDA_FOUND
  cli.add<size_t>("--cpu-threads",
      "Use CPU-based computation with this many independent threads, 0 means GPU-based computation")
    ->default_val("0")->implicit_val("1");
#else
  cli.add<size_t>("--cpu-threads",
      "Use CPU-based computation with this many independent threads, 0 means GPU-based computation")
    ->default_val("1");
#endif

  // optimizer options
  cli.add<int>("--mini-batch",
      "Size of mini-batch used during update", 64);
  cli.add<int>("--mini-batch-words",
      "Set mini-batch size based on words instead of sentences");
  cli.add<bool>("--mini-batch-fit",
      "Determine mini-batch size automatically based on sentence-length to fit reserved memory");
  cli.add<size_t>("--mini-batch-fit-step",
      "Step size for mini-batch-fit statistics",
      10);
  cli.add<int>("--maxi-batch",
      "Number of batches to preload for length-based sorting",
      100);
  cli.add<std::string>("--maxi-batch-sort",
      "Sorting strategy for maxi-batch: trg, src, none",
      "trg");

  cli.add<std::string>("--optimizer,-o",
     "Optimization algorithm: sgd, adagrad, adam",
     "adam");
  cli.add_nondefault<std::vector<float>>("--optimizer-params",
     "Parameters for optimization algorithm, e.g. betas for adam");
  cli.add<size_t>("--optimizer-delay",
     "SGD update delay, 1 = no delay",
     1);

  cli.add<bool>("--sync-sgd",
     "Use synchronous SGD instead of asynchronous for multi-gpu training");

  // learning rate options
  cli.add<double>("--learn-rate,-l",
     "Learning rate",
     0.0001);
  cli.add<bool>("--lr-report",
     "Report learning rate for each update");

  cli.add<double>("--lr-decay",
     "Decay factor for learning rate: lr = lr * arg (0 to disable)");
  cli.add<std::string>("--lr-decay-strategy",
     "Strategy for learning rate decaying: epoch, batches, stalled, epoch+batches, epoch+stalled",
     "epoch+stalled");
  cli.add<std::vector<size_t>>("--lr-decay-start",
     "The first number of epoch/batches/stalled validations to start learning rate decaying",
     std::vector<size_t>({10,1}));
  cli.add<size_t>("--lr-decay-freq",
     "Learning rate decaying frequency for batches, requires --lr-decay-strategy to be batches",
     50000);
  cli.add<bool>("--lr-decay-reset-optimizer",
      "Reset running statistics of optimizer whenever learning rate decays");
  cli.add<bool>("--lr-decay-repeat-warmup",
     "Repeat learning rate warmup when learning rate is decayed");
  cli.add<size_t>("--lr-decay-inv-sqrt",
     "Decrease learning rate at arg / sqrt(no. updates) starting at arg");

  cli.add<size_t>("--lr-warmup",
     "Increase learning rate linearly for arg first steps");
  cli.add<float>("--lr-warmup-start-rate",
     "Start value for learning rate warmup");
  cli.add<bool>("--lr-warmup-cycle",
     "Apply cyclic warmup");
  cli.add<bool>("--lr-warmup-at-reload",
     "Repeat warmup after interrupted training");

  cli.add<bool>("--batch-flexible-lr",
      "Scales the learning rate based on the number of words in a mini-batch");
  cli.add<double>("--batch-normal-words",
      "Set number of words per batch that the learning rate corresponds to. "
      "The option is only active when batch-flexible-lr is on",
      1920.0);

  cli.add<double>("--label-smoothing",
     "Epsilon for label smoothing (0 to disable)");
  cli.add<double>("--clip-norm",
     "Clip gradient norm to  argcli.add<int>(0 to disable)",
     1.f);
  cli.add<float>("--exponential-smoothing",
     "Maintain smoothed version of parameters for validation and saving with smoothing factor. 0 to disable")
    ->implicit_val("1e-4")->default_val("0");

  // options for additional training data
  cli.add_nondefault<std::string>("--guided-alignment",
     "Use guided alignment to guide attention");
  cli.add<std::string>("--guided-alignment-cost",
     "Cost type for guided alignment: ce (cross-entropy), mse (mean square error), mult (multiplication)",
     "ce");
  cli.add<double>("--guided-alignment-weight",
     "Weight for guided alignment cost",
     1);
  cli.add_nondefault<std::string>("--data-weighting",
     "File with sentence or word weights");
  cli.add<std::string>("--data-weighting-type",
     "Processing level for data weighting: sentence, word",
     "sentence");

  // embedding options
  cli.add_nondefault<std::vector<std::string>>("--embedding-vectors",
     "Paths to files with custom source and target embedding vectors");
  cli.add<bool>("--embedding-normalization",
     "Enable normalization of custom embedding vectors");
  cli.add<bool>("--embedding-fix-src",
     "Fix source embeddings. Affects all encoders");
  cli.add<bool>("--embedding-fix-trg",
     "Fix target embeddings. Affects all decoders");

  cli.add<bool>("--multi-node",
     "Enable multi-node training through MPI");
  cli.add<bool>("--multi-node-overlap",
     "Overlap model computations with MPI communication",
     true);
  // clang-format on

  cli.endGroup();
}

void ConfigParser::addOptionsValid(cli::CLIWrapper &cli) {
  cli.startGroup("Validation set options");

  // clang-format off
  cli.add_nondefault<std::vector<std::string>>("--valid-sets",
      "Paths to validation corpora: source target");
  cli.add<size_t>("--valid-freq",
      "Validate model every  arg  updates",
      10000);
  cli.add<std::vector<std::string>>("--valid-metrics",
      "Metric to use during validation: cross-entropy, perplexity, valid-script, translation."
      " Multiple metrics can be specified",
      std::vector<std::string>({"cross-entropy"}));
  cli.add<size_t>("--early-stopping",
     "Stop if the first validation metric does not improve for  arg  consecutive validation steps",
     10);

  // decoding options
  cli.add<size_t>("--beam-size,-b",
      "Beam size used during search with validating translator",
      12);
  cli.add<float>("--normalize,-n",
      "Divide translation score by pow(translation length, arg) ")
      ->default_val("0")->implicit_val("1");
  cli.add<float>("--max-length-factor",
      "Maximum target length as source length times factor",
      3);
  cli.add<float>("--word-penalty",
      "Subtract (arg * translation length) from translation score ");
  cli.add<bool>("--allow-unk",
      "Allow unknown words to appear in output");
  cli.add<bool>("--n-best",
      "Generate n-best list");

  // efficiency options
  cli.add<int>("--valid-mini-batch",
      "Size of mini-batch used during validation",
      32);
  cli.add<size_t>("--valid-max-length",
      "Maximum length of a sentence in a validating sentence pair",
      1000);

  // options for validation script
  cli.add_nondefault<std::string>("--valid-script-path",
     "Path to external validation script."
     " It should print a single score to stdout."
     " If the option is used with validating translation, the output"
     " translation file will be passed as a first argument");
  cli.add_nondefault<std::string>("--valid-translation-output",
     "Path to store the translation");

  cli.add<bool>("--keep-best",
      "Keep best model for each validation metric");
  cli.add_nondefault<std::string>("--valid-log",
     "Log validation scores to file given by  arg");

  // clang-format on
  cli.endGroup();
}

void ConfigParser::addOptionsTranslate(cli::CLIWrapper &cli) {
  cli.startGroup("Translator options");

  // clang-format off
  cli.add<std::vector<std::string>>("--input,-i",
      "Paths to input file(s), stdin by default",
      std::vector<std::string>({"stdin"}));
  cli.add_nondefault<std::vector<std::string>>("--vocabs,-v",
      "Paths to vocabulary files have to correspond to --input");

  // decoding options
  cli.add<size_t>("--beam-size,-b",
      "Beam size used during search with validating translator",
      12);
  cli.add<float>("--normalize,-n",
      "Divide translation score by pow(translation length, arg)")
      ->default_val("0")->implicit_val("1");
  cli.add<float>("--max-length-factor",
      "Maximum target length as source length times factor",
      3);
  cli.add<float>("--word-penalty",
      "Subtract (arg * translation length) from translation score");
  cli.add<bool>("--allow-unk",
      "Allow unknown words to appear in output");
  cli.add<bool>("--n-best",
      "Generate n-best list");
  cli.add<std::string>("--alignment",
     "Return word alignment. Possible values: 0.0-1.0, hard, soft")
    ->implicit_val("1");

  // efficiency options
  cli.add<std::vector<std::string>>("--devices,-d",
      "GPUs to use for translation",
      std::vector<std::string>({"0"}));
#ifdef CUDA_FOUND
  cli.add<size_t>("--cpu-threads",
      "Use CPU-based computation with this many independent threads, 0 means GPU-based computation")
      ->default_val("0")->implicit_val("1");
#else
  cli.add<size_t>("--cpu-threads",
      "Use CPU-based computation with this many independent threads, 0 means GPU-based computation")
      ->default_val("1");
#endif

  cli.add<size_t>("--max-length",
      "Maximum length of a sentence in a training sentence pair",
      1000);
  cli.add<bool>("--max-length-crop",
      "Crop a sentence to max-length instead of ommitting it if longer than max-length");
  cli.add<bool>("--optimize",
      "Optimize speed aggressively sacrificing memory or precision");
  cli.add<int>("--mini-batch",
      "Size of mini-batch used during update",
      1);
  cli.add<int>("--mini-batch-words",
      "Set mini-batch size based on words instead of sentences");
  cli.add<int>("--maxi-batch",
      "Number of batches to preload for length-based sorting",
      1);
  cli.add<std::string>("--maxi-batch-sort",
      "Sorting strategy for maxi-batch: none, src",
      "none");
  cli.add<bool>("--skip-cost",
      "Ignore model cost during translation, not recommended for beam-size > 1");

  cli.add_nondefault<std::vector<std::string>>("--shortlist",
     "Use softmax shortlist: path first best prune");
  cli.add_nondefault<std::vector<float>>("--weights",
      "Scorer weights");

  // TODO: the options should be available only in server
  cli.add<size_t>("--port,-p",
      "Port number for web socket server",
      8080);
  // clang-format on

 cli.endGroup();
}

void ConfigParser::addOptionsRescore(cli::CLIWrapper &cli) {
  cli.startGroup("Rescorer options");

  // clang-format off
  cli.add<bool>("--no-reload",
      "Do not load existing model specified in --model arg");
  // TODO: move options like vocabs and train-sets to a separate procedure as they are defined twice
  cli.add<std::vector<std::string>>("--train-sets,-t",
      "Paths to corpora to be scored: source target");
  cli.add_nondefault<std::vector<std::string>>("--vocabs,-v",
      "Paths to vocabulary files have to correspond to --train-sets."
      " If this parameter is not supplied we look for vocabulary files source.{yml,json} and target.{yml,json}."
      " If these files do not exists they are created");
  cli.add<bool>("--n-best",
      "Score n-best list instead of plain text corpus");
  cli.add<std::string>("--n-best-feature",
      "Feature name to be inserted into n-best list", "Score");
  cli.add_nondefault<std::string>("--summary",
      "Only print total cost, possible values: cross-entropy (ce-mean), ce-mean-words, ce-sum, perplexity")
      ->implicit_val("cross-entropy");
  cli.add<size_t>("--max-length",
      "Maximum length of a sentence in a training sentence pair",
      1000);
  cli.add<bool>("--max-length-crop",
      "Crop a sentence to max-length instead of ommitting it if longer than max-length");
  cli.add<std::vector<std::string>>("--devices,-d",
      "GPUs to use for training",
      std::vector<std::string>({"0"}));
#ifdef CUDA_FOUND
  cli.add<size_t>("--cpu-threads",
      "Use CPU-based computation with this many independent threads, 0 means GPU-based computation")
      ->default_val("0")->implicit_val("1");
#else
  cli.add<size_t>("--cpu-threads",
      "Use CPU-based computation with this many independent threads, 0 means GPU-based computation")
      ->default_val("1");
#endif
  cli.add<bool>("--optimize",
      "Optimize speed aggressively sacrificing memory or precision");
  cli.add<int>("--mini-batch",
      "Size of mini-batch used during update",
      64);
  cli.add<int>("--mini-batch-words",
      "Set mini-batch size based on words instead of sentences");
  cli.add<int>("--maxi-batch",
      "Number of batches to preload for length-based sorting",
      100);
  cli.add<std::string>("--maxi-batch-sort",
      "Sorting strategy for maxi-batch: trg (default), src, none",
      "trg");
  cli.add_nondefault<std::string>("--alignment",
     "Return word alignments. Possible values: 0.0-1.0, hard, soft")
     ->implicit_val("1"),

  // clang-format on
  cli.endGroup();
}

void ConfigParser::parseOptions(int argc, char** argv, bool doValidate) {
  cli::CLIWrapper cli;

  addOptionsCommon(cli);
  addOptionsModel(cli);

  // clang-format off
  switch(mode_) {
    case ConfigMode::translating:
      addOptionsTranslate(cli);
      break;
    case ConfigMode::rescoring:
      addOptionsRescore(cli);
      break;
    case ConfigMode::training:
      addOptionsTraining(cli);
      addOptionsValid(cli);
      break;
  }
  // clang-format on

  try {
    cli.app()->parse(argc, argv);
  } catch(const CLI::ParseError& e) {
    exit(cli.app()->exit(e));
  }

  // TODO: config_ is needed here?
  config_ = cli.getConfig();

  if(has("version")) {
    std::cerr << PROJECT_VERSION_FULL << std::endl;
    exit(0);
  }

  auto configPaths = loadConfigPaths();

  if(!configPaths.empty()) {
    auto config = loadConfigFiles(configPaths);
    config_ = cli.getConfigWithNewDefaults(config);
  }

  // TODO: option expansion should be done at the very end?
  if(cli.has("best-deep")) {
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

  if(get<bool>("interpolate-env-vars")) {
    cli::ProcessPaths(config_, cli::InterpolateEnvVars, PATHS);
  }

  if(get<bool>("relative-paths") && !get<bool>("dump-config")) {
    makeAbsolutePaths(configPaths);
  }

  if(doValidate) {
    try {
      validateOptions();
      validateDevices();
    } catch(util::Exception& e) {
      std::cerr << "Error: " << e.what() << std::endl << std::endl;
      std::cerr << "Usage: " + std::string(argv[0]) + " [options]" << std::endl;
      exit(1);
    }
  }

  config_.remove("config");

  if(get<bool>("dump-config")) {
    config_.remove("dump-config");
    YAML::Emitter emit;
    cli::OutputYaml(config_, emit);
    std::cout << emit.c_str() << std::endl;
    exit(0);
  }
}

void ConfigParser::makeAbsolutePaths(
    const std::vector<std::string>& configPaths) {
  ABORT_IF(configPaths.empty(),
           "--relative-paths option requires at least one config file provided "
           "with --config");
  auto configDir = boost::filesystem::path{configPaths.front()}.parent_path();

  for(const auto& configPath : configPaths)
    ABORT_IF(boost::filesystem::path{configPath}.parent_path() != configDir,
             "--relative-paths option requires all config files to be in the "
             "same directory");

  auto transformFunc = [&](const std::string& nodePath) -> std::string {
    // replace relative path w.r.t. configDir
    using namespace boost::filesystem;
    try {
      return canonical(path{nodePath}, configDir).string();
    } catch(boost::filesystem::filesystem_error& e) {
      // will fail if file does not exist; use parent in that case
      std::cerr << e.what() << std::endl;
      auto parentPath = path{nodePath}.parent_path();
      return (canonical(parentPath, configDir) / path{nodePath}.filename())
          .string();
    }
  };

  cli::ProcessPaths(config_, transformFunc, PATHS);
}

YAML::Node ConfigParser::loadConfigFiles(
    const std::vector<std::string>& paths) {
  YAML::Node config;

  for(auto& path : paths) {
    // later file overrides
    for(const auto& it : YAML::Load(InputFileStream(path))) {
      config[it.first.as<std::string>()] = YAML::Clone(it.second);
    }
  }

  return config;
}

std::vector<std::string> ConfigParser::loadConfigPaths() {
  std::vector<std::string> paths;

  bool interpolateEnvVars = config_["interpolate-env-vars"].as<bool>();
  bool loadConfig = !config_["config"].as<std::vector<std::string>>().empty();

  if(loadConfig) {
    paths = config_["config"].as<std::vector<std::string>>();
    for(auto& path : paths) {
      // (note: this updates the paths array)
      if(interpolateEnvVars)
        path = cli::InterpolateEnvVars(path);
    }
  } else if(mode_ == ConfigMode::training) {
    auto path = config_["model"].as<std::string>() + ".yml";
    if(interpolateEnvVars)
      path = cli::InterpolateEnvVars(path);

    bool reloadConfig
        = boost::filesystem::exists(path) && !config_["no-reload"].as<bool>();

    if(reloadConfig)
      paths = {path};
  }

  return paths;
}

std::vector<DeviceId> ConfigParser::getDevices() {
  std::vector<DeviceId> devices;

  // TODO: why do we use config_[x] and get<> interchanbeably
  try {
    std::string devicesStr
        = utils::Join(config_["devices"].as<std::vector<std::string>>());

    if(mode_ == ConfigMode::training && get<bool>("multi-node")) {
      auto parts = utils::Split(devicesStr, ":");
      for(size_t i = 1; i < parts.size(); ++i) {
        std::string part = parts[i];
        utils::Trim(part);
        auto ds = utils::Split(part, " ");
        if(i < parts.size() - 1)
          ds.pop_back();

        // does this make sense?
        devices.push_back({ds.size(), DeviceType::gpu});
        for(auto d : ds)
          devices.push_back({(size_t)std::stoull(d), DeviceType::gpu});
      }
    } else {
      for(auto d : utils::Split(devicesStr))
        devices.push_back({(size_t)std::stoull(d), DeviceType::gpu});
    }

    if(config_["cpu-threads"].as<size_t>() > 0) {
      devices.clear();
      for(size_t i = 0; i < config_["cpu-threads"].as<size_t>(); ++i)
        devices.push_back({i, DeviceType::cpu});
    }

  } catch(...) {
    ABORT("Problem parsing devices, please report an issue on github");
  }

  return devices;
}

YAML::Node ConfigParser::getConfig() const {
  return config_;
}
}  // namespace marian
