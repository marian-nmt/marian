#include "common/config_validator.h"
#include "3rd_party/exception.h"
#include "common/logging.h"
#include "common/regex.h"
#include "common/utils.h"

#include <boost/filesystem.hpp>

namespace marian {

bool ConfigValidator::has(const std::string& key) const {
  return config_[key];
}

ConfigValidator::ConfigValidator(const YAML::Node& config)
    : config_(config) {}

ConfigValidator::~ConfigValidator() {}

void ConfigValidator::validateOptions(ConfigMode mode) const {
  switch(mode) {
    case ConfigMode::translating:
      validateOptionsTranslation();
      break;
    case ConfigMode::rescoring:
      validateOptionsParallelData();
      validateOptionsScoring();
      break;
    case ConfigMode::training:
      validateOptionsParallelData();
      validateOptionsTraining();
      break;
  }
}

void ConfigValidator::validateOptionsTranslation() const {
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
}

void ConfigValidator::validateOptionsParallelData() const {
  UTIL_THROW_IF2(
      !has("train-sets") || get<std::vector<std::string>>("train-sets").empty(),
      "No train sets given in config file or on command line");
  UTIL_THROW_IF2(
      has("vocabs")
          && get<std::vector<std::string>>("vocabs").size()
                 != get<std::vector<std::string>>("train-sets").size(),
      "There should be as many vocabularies as training sets");
}

void ConfigValidator::validateOptionsScoring() const {
  boost::filesystem::path modelPath(get<std::string>("model"));

  UTIL_THROW_IF2(!boost::filesystem::exists(modelPath),
                 "Model file does not exist: " + modelPath.string());
  UTIL_THROW_IF2(
      !has("vocabs") || get<std::vector<std::string>>("vocabs").empty(),
      "Scoring, but vocabularies are not given!");
}

void ConfigValidator::validateOptionsTraining() const {
  UTIL_THROW_IF2(
      has("embedding-vectors")
          && get<std::vector<std::string>>("embedding-vectors").size()
                 != get<std::vector<std::string>>("train-sets").size(),
      "There should be as many files with embedding vectors as "
      "training sets");

  boost::filesystem::path modelPath(get<std::string>("model"));

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

void ConfigValidator::validateDevices(ConfigMode mode) const {
  std::string devices = utils::Join(get<std::vector<std::string>>("devices"));
  utils::Trim(devices);

  regex::regex pattern;
  std::string help;
  if(mode == ConfigMode::training && get<bool>("multi-node")) {
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
