#include "common/config_validator.h"
#include "3rd_party/exception.h"
#include "common/logging.h"
#include "common/regex.h"
#include "common/utils.h"
#include "common/filesystem.h"

namespace marian {

bool ConfigValidator::has(const std::string& key) const {
  return config_[key];
}

ConfigValidator::ConfigValidator(const YAML::Node& config)
    : config_(config) {}

ConfigValidator::~ConfigValidator() {}

void ConfigValidator::validateOptions(cli::mode mode) const {
  switch(mode) {
    case cli::mode::translation:
      validateOptionsTranslation();
      break;
    case cli::mode::scoring:
      validateOptionsParallelData();
      validateOptionsScoring();
      break;
    case cli::mode::training:
      validateOptionsParallelData();
      validateOptionsTraining();
      break;
  }

  validateDevices(mode);
}

void ConfigValidator::validateOptionsTranslation() const {
  auto models = get<std::vector<std::string>>("models");
  auto configs = get<std::vector<std::string>>("config");
  UTIL_THROW_IF2(
      models.empty() && configs.empty(),
      "You need to provide at least one model file or a config file");

  auto vocabs = get<std::vector<std::string>>("vocabs");
  UTIL_THROW_IF2(vocabs.empty(),
                 "Translating, but vocabularies are not given!");

  for(const auto& modelFile : models) {
    filesystem::Path modelPath(modelFile);
    UTIL_THROW_IF2(!filesystem::exists(modelPath),
                   "Model file does not exist: " + modelFile);
  }
}

void ConfigValidator::validateOptionsParallelData() const {
  auto trainSets = get<std::vector<std::string>>("train-sets");
  UTIL_THROW_IF2(trainSets.empty(),
                 "No train sets given in config file or on command line");

  auto vocabs = get<std::vector<std::string>>("vocabs");
  UTIL_THROW_IF2(!vocabs.empty() && vocabs.size() != trainSets.size(),
                 "There should be as many vocabularies as training sets");
}

void ConfigValidator::validateOptionsScoring() const {
  filesystem::Path modelPath(get<std::string>("model"));

  UTIL_THROW_IF2(!filesystem::exists(modelPath),
                 "Model file does not exist: " + modelPath.string());
  UTIL_THROW_IF2(get<std::vector<std::string>>("vocabs").empty(),
                 "Scoring, but vocabularies are not given!");
}

void ConfigValidator::validateOptionsTraining() const {
  auto trainSets = get<std::vector<std::string>>("train-sets");

  UTIL_THROW_IF2(
      has("embedding-vectors")
          && get<std::vector<std::string>>("embedding-vectors").size()
                 != trainSets.size(),
      "There should be as many embedding vector files as training sets");

  filesystem::Path modelPath(get<std::string>("model"));

  auto modelDir = modelPath.parentPath();
  if(modelDir.empty())
    modelDir = filesystem::currentPath();

  UTIL_THROW_IF2(
      !modelDir.empty() && !filesystem::isDirectory(modelDir),
      "Model directory does not exist");

  UTIL_THROW_IF2(!modelDir.empty() && !filesystem::canWrite(modelDir),
                 "No write permission in model directory");

  UTIL_THROW_IF2(has("valid-sets")
                     && get<std::vector<std::string>>("valid-sets").size()
                            != trainSets.size(),
                 "There should be as many validation sets as training sets");

  // validations for learning rate decaying
  UTIL_THROW_IF2(get<double>("lr-decay") > 1.0,
                 "Learning rate decay factor greater than 1.0 is unusual");

  auto strategy = get<std::string>("lr-decay-strategy");

  UTIL_THROW_IF2(
      (strategy == "epoch+batches" || strategy == "epoch+stalled")
          && get<std::vector<size_t>>("lr-decay-start").size() != 2,
      "Decay strategies 'epoch+batches' and 'epoch+stalled' require two "
      "values specified with --lr-decay-start option");
  UTIL_THROW_IF2(
      (strategy == "epoch" || strategy == "batches" || strategy == "stalled")
          && get<std::vector<size_t>>("lr-decay-start").size() != 1,
      "Single decay strategies require only one value specified with "
      "--lr-decay-start option");
}

void ConfigValidator::validateDevices(cli::mode mode) const {
  std::string devices = utils::join(get<std::vector<std::string>>("devices"));
  utils::trim(devices);

  regex::regex pattern;
  std::string help;
  if(mode == cli::mode::training && get<bool>("multi-node")) {
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
                     + ")' for option '--devices' is invalid. "
                     + help);
}

}  // namespace marian
