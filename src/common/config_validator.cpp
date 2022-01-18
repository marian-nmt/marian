#include "common/config_validator.h"
#include "common/logging.h"
#include "common/regex.h"
#include "common/utils.h"
#include "common/filesystem.h"

#include <set>

namespace marian {

bool ConfigValidator::has(const std::string& key) const {
  return config_[key];
}

ConfigValidator::ConfigValidator(const YAML::Node& config)
    : config_(config),
      dumpConfigOnly_(config["dump-config"] && !config["dump-config"].as<std::string>().empty()
                      && config["dump-config"].as<std::string>() != "false") {}

ConfigValidator::~ConfigValidator() {}

void ConfigValidator::validateOptions(cli::mode mode) const {
  // clang-format off
  switch(mode) {
    case cli::mode::translation:
      validateOptionsTranslation();
      break;
    case cli::mode::scoring:
      validateOptionsParallelData();
      validateOptionsScoring();
      break;
    case cli::mode::embedding:
      validateOptionsParallelData();
      validateOptionsScoring();
      break;
    case cli::mode::training:
      validateOptionsParallelData();
      validateOptionsTraining();
      break;
    default:
      ABORT("wrong CLI mode");
      break;
  }
  // clang-format on

  validateModelExtension(mode);
  validateDevices(mode);
}

void ConfigValidator::validateOptionsTranslation() const {
  auto models = get<std::vector<std::string>>("models");
  auto configs = get<std::vector<std::string>>("config");

  ABORT_IF(models.empty() && configs.empty(),
           "You need to provide at least one model file or a config file");

  ABORT_IF(get<bool>("model-mmap") && get<size_t>("cpu-threads") == 0,
           "Model MMAP is CPU-only, please use --cpu-threads");

  for(const auto& modelFile : models) {
    filesystem::Path modelPath(modelFile);
    ABORT_IF(!filesystem::exists(modelPath), "Model file does not exist: " + modelFile);
  }

  auto vocabs = get<std::vector<std::string>>("vocabs");
  ABORT_IF(vocabs.empty(), "Translating, but vocabularies are not given");

  for(const auto& vocabFile : vocabs) {
    filesystem::Path vocabPath(vocabFile);
    ABORT_IF(!filesystem::exists(vocabPath), "Vocabulary file does not exist: " + vocabFile);
  }
}

void ConfigValidator::validateOptionsParallelData() const {
  // Do not check these constraints if only goal is to dump config
  if(dumpConfigOnly_)
    return;

  auto trainSets = get<std::vector<std::string>>("train-sets");
  ABORT_IF(trainSets.empty(), "No train sets given in config file or on command line");

  auto numVocabs = get<std::vector<std::string>>("vocabs").size();
  ABORT_IF(!get<bool>("tsv") && numVocabs > 0 && numVocabs != trainSets.size(),
           "There should be as many vocabularies as training files");

  // disallow, for example --tsv --train-sets file1.tsv file2.tsv
  ABORT_IF(get<bool>("tsv") && trainSets.size() != 1,
      "A single file must be provided with --train-sets (or stdin) for a tab-separated input");

  // disallow, for example --train-sets stdin stdin or --train-sets stdin file.tsv
  ABORT_IF(trainSets.size() > 1
               && std::any_of(trainSets.begin(),
                              trainSets.end(),
                              [](const std::string& s) { return (s == "stdin") || (s == "-"); }),
           "Only one 'stdin' or '-' in --train-sets is allowed");
}

void ConfigValidator::validateOptionsScoring() const {
  filesystem::Path modelPath(get<std::string>("model"));
  ABORT_IF(!filesystem::exists(modelPath), "Model file does not exist: " + modelPath.string());

  auto vocabs = get<std::vector<std::string>>("vocabs");
  ABORT_IF(vocabs.empty(), "Scoring, but vocabularies are not given");

  for(const auto& vocabFile : vocabs) {
    filesystem::Path vocabPath(vocabFile);
    ABORT_IF(!filesystem::exists(vocabPath), "Vocabulary file does not exist: " + vocabFile);
  }
}

void ConfigValidator::validateOptionsTraining() const {
  auto trainSets = get<std::vector<std::string>>("train-sets");

  ABORT_IF(has("embedding-vectors")
               && get<std::vector<std::string>>("embedding-vectors").size() != trainSets.size()
               && !get<std::vector<std::string>>("embedding-vectors").empty(),
           "There should be as many embedding vector files as training files");

  filesystem::Path modelPath(get<std::string>("model"));

  auto modelDir = modelPath.parentPath();
  if(modelDir.empty())
    modelDir = filesystem::currentPath();

  ABORT_IF(!modelDir.empty() && !filesystem::isDirectory(modelDir),
           "Model directory does not exist");

  std::string errorMsg = "There should be as many validation files as training files";
  if(get<bool>("tsv"))
    errorMsg += ". If the training set is in the TSV format, validation sets have to also be a single TSV file";

  ABORT_IF(has("valid-sets")
               && get<std::vector<std::string>>("valid-sets").size() != trainSets.size()
               && !get<std::vector<std::string>>("valid-sets").empty(),
           errorMsg);

  // check if --early-stopping-on has proper value
  std::set<std::string> supportedStops = {"first", "all", "any"};
  ABORT_IF(supportedStops.find(get<std::string>("early-stopping-on")) == supportedStops.end(),
           "Supported options for --early-stopping-on are: first, all, any");

  // validations for learning rate decaying
  ABORT_IF(get<float>("lr-decay") > 1.f, "Learning rate decay factor greater than 1.0 is unusual");

  auto strategy = get<std::string>("lr-decay-strategy");

  ABORT_IF((strategy == "epoch+batches" || strategy == "epoch+stalled")
               && get<std::vector<size_t>>("lr-decay-start").size() != 2,
           "Decay strategies 'epoch+batches' and 'epoch+stalled' require two values specified with "
           "--lr-decay-start option");
  ABORT_IF((strategy == "epoch" || strategy == "batches" || strategy == "stalled")
               && get<std::vector<size_t>>("lr-decay-start").size() != 1,
           "Single decay strategies require only one value specified with --lr-decay-start option");

  // validate ULR options
  ABORT_IF((has("ulr") && get<bool>("ulr") && (get<std::string>("ulr-query-vectors") == ""
                                               || get<std::string>("ulr-keys-vectors") == "")),
           "ULR requires query and keys vectors specified with --ulr-query-vectors and "
           "--ulr-keys-vectors option");

  // validate model quantization
  size_t bits = get<size_t>("quantize-bits");
  ABORT_IF(bits > 32, "Invalid quantization bits. Must be from 0 to 32 bits");

  ABORT_IF(bits > 0 && !get<bool>("sync-sgd"), "Model quantization only works with synchronous training (--sync-sgd)");
}

void ConfigValidator::validateModelExtension(cli::mode mode) const {
  std::vector<std::string> models;
  if(mode == cli::mode::translation)
    models = get<std::vector<std::string>>("models");
  else
    models.push_back(get<std::string>("model"));

  for(const auto& modelPath : models) {
    bool hasProperExt = utils::endsWith(modelPath, ".npz") || utils::endsWith(modelPath, ".bin");
    ABORT_IF(!hasProperExt,
             "Unknown model format for file '{}'. Supported file extensions: .npz, .bin",
             modelPath);
  }
}

void ConfigValidator::validateDevices(cli::mode /*mode*/) const {
  std::string devices = utils::join(get<std::vector<std::string>>("devices"));
  utils::trim(devices);

  regex::regex pattern;
  std::string help;

  // valid strings: '0', '0 1 2 3', '3 2 0 1'
  pattern = "[0-9]+( *[0-9]+)*";
  help = "Supported formats: '0 1 2 3'";

  ABORT_IF(!regex::regex_match(devices, pattern),
           "the argument '{}' for option '--devices' is invalid. {}",
           devices,
           help);
}

}  // namespace marian
