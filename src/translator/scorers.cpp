#include "translator/scorers.h"

namespace marian {

Ptr<Scorer> scorerByType(std::string fname,
                         float weight,
                         std::string model,
                         Ptr<Config> config) {
  Ptr<Options> options = New<Options>();
  options->merge(config);
  options->set("inference", true);

  std::string type = options->get<std::string>("type");

  // @TODO: solve this better
  if(type == "lm" && config->has("input")) {
    size_t index = config->get<std::vector<std::string>>("input").size();
    options->set("index", index);
  }

  bool skipCost = config->get<bool>("skip-cost");
  auto encdec = models::from_options(options,
                                     skipCost ? models::usage::raw
                                     : models::usage::translation);

  LOG(info, "Loading scorer of type {} as feature {}", type, fname);

  return New<ScorerWrapper>(encdec, fname, weight, model);
}

std::vector<Ptr<Scorer>> createScorers(Ptr<Config> options) {
  std::vector<Ptr<Scorer>> scorers;

  auto models = options->get<std::vector<std::string>>("models");
  int dimVocab = options->get<std::vector<int>>("dim-vocabs").back();

  std::vector<float> weights(models.size(), 1.f);
  if(options->has("weights"))
    weights = options->get<std::vector<float>>("weights");

  int i = 0;
  for(auto model : models) {
    std::string fname = "F" + std::to_string(i);
    auto modelOptions = New<Config>(*options);

    try {
      if(!options->get<bool>("ignore-model-config"))
        modelOptions->loadModelParameters(model);
    } catch(std::runtime_error& e) {
      LOG(warn, "No model settings found in model file");
    }

    scorers.push_back(scorerByType(fname, weights[i], model, modelOptions));
    i++;
  }

  return scorers;
}
}
