#include "translator/scorers.h"
#include "common/io.h"

namespace marian {

Ptr<Scorer> scorerByType(const std::string& fname,
                         float weight,
                         const std::string& model,
                         Ptr<Options> options) {
  options->set("inference", true);
  std::string type = options->get<std::string>("type");

  // @TODO: solve this better
  if(type == "lm" && options->has("input")) {
    size_t index = options->get<std::vector<std::string>>("input").size();
    options->set("index", index);
  }

  bool skipCost = options->get<bool>("skip-cost");
  auto encdec = models::createModelFromOptions(
      options, skipCost ? models::usage::raw : models::usage::translation);

  LOG(info, "Loading scorer of type {} as feature {}", type, fname);

  return New<ScorerWrapper>(encdec, fname, weight, model);
}

Ptr<Scorer> scorerByType(const std::string& fname,
                         float weight,
                         const void* ptr,
                         Ptr<Options> options) {
  options->set("inference", true);
  std::string type = options->get<std::string>("type");

  // @TODO: solve this better
  if(type == "lm" && options->has("input")) {
    size_t index = options->get<std::vector<std::string>>("input").size();
    options->set("index", index);
  }

  bool skipCost = options->get<bool>("skip-cost");
  auto encdec = models::createModelFromOptions(
      options, skipCost ? models::usage::raw : models::usage::translation);

  LOG(info, "Loading scorer of type {} as feature {}", type, fname);

  return New<ScorerWrapper>(encdec, fname, weight, ptr);
}

std::vector<Ptr<Scorer>> createScorers(Ptr<Options> options) {
  std::vector<Ptr<Scorer>> scorers;

  auto models = options->get<std::vector<std::string>>("models");

  std::vector<float> weights(models.size(), 1.f);
  if(options->hasAndNotEmpty("weights"))
    weights = options->get<std::vector<float>>("weights");

  bool isPrevRightLeft = false;  // if the previous model was a right-to-left model
  size_t i = 0;
  for(auto model : models) {
    std::string fname = "F" + std::to_string(i);

    // load options specific for the scorer
    auto modelOptions = New<Options>(options->clone());
    try {
      if(!options->get<bool>("ignore-model-config")) {
        YAML::Node modelYaml;
        io::getYamlFromModel(modelYaml, "special:model.yml", model);
        modelOptions->merge(modelYaml, true);
      }
    } catch(std::runtime_error&) {
      LOG(warn, "No model settings found in model file");
    }

    // l2r and r2l cannot be used in the same ensemble
    if(models.size() > 1 && modelOptions->has("right-left")) {
      if(i == 0) {
        isPrevRightLeft = modelOptions->get<bool>("right-left");
      } else {
        // abort as soon as there are two consecutive models with opposite directions
        ABORT_IF(isPrevRightLeft != modelOptions->get<bool>("right-left"),
                 "Left-to-right and right-to-left models cannot be used together in ensembles");
        isPrevRightLeft = modelOptions->get<bool>("right-left");
      }
    }

    scorers.push_back(scorerByType(fname, weights[i], model, modelOptions));
    i++;
  }

  return scorers;
}

std::vector<Ptr<Scorer>> createScorers(Ptr<Options> options, const std::vector<const void*>& ptrs) {
  std::vector<Ptr<Scorer>> scorers;

  std::vector<float> weights(ptrs.size(), 1.f);
  if(options->hasAndNotEmpty("weights"))
    weights = options->get<std::vector<float>>("weights");

  size_t i = 0;
  for(auto ptr : ptrs) {
    std::string fname = "F" + std::to_string(i);

    // load options specific for the scorer
    auto modelOptions = New<Options>(options->clone());
    try {
      if(!options->get<bool>("ignore-model-config")) {
        YAML::Node modelYaml;
        io::getYamlFromModel(modelYaml, "special:model.yml", ptr);
        modelOptions->merge(modelYaml, true);
      }
    } catch(std::runtime_error&) {
      LOG(warn, "No model settings found in model file");
    }

    scorers.push_back(scorerByType(fname, weights[i], ptr, modelOptions));
    i++;
  }

  return scorers;
}

std::vector<Ptr<Scorer>> createScorers(Ptr<Options> options, const std::vector<mio::mmap_source>& mmaps) {
  std::vector<const void*> ptrs;
  for(const auto& mmap : mmaps) {
    ABORT_IF(!mmap.is_mapped(), "Memory mapping did not succeed");
    ptrs.push_back(mmap.data());
  }
  return createScorers(options, ptrs);
}

}  // namespace marian
