#include "training/validator.h"

namespace marian {

std::vector<Ptr<Validator<data::Corpus>>> Validators(
    std::vector<Ptr<Vocab>> vocabs,
    Ptr<Options> config) {
  std::vector<Ptr<Validator<data::Corpus>>> validators;

  auto validMetrics = config->get<std::vector<std::string>>("valid-metrics");

  std::vector<std::string> ceMetrics
      = {"cross-entropy", "ce-mean", "ce-sum", "ce-mean-words", "perplexity"};

  for(auto metric : validMetrics) {
    if(std::find(ceMetrics.begin(), ceMetrics.end(), metric) != ceMetrics.end()) {
      Ptr<Options> opts = New<Options>(*config);
      opts->set("cost-type", metric);

      auto validator = New<CrossEntropyValidator>(vocabs, opts);
      validators.push_back(validator);
    } else if(metric == "valid-script") {
      auto validator = New<ScriptValidator>(vocabs, config);
      validators.push_back(validator);
    } else if(metric == "translation") {
      auto validator = New<TranslationValidator>(vocabs, config);
      validators.push_back(validator);
    } else if(metric == "bleu") {
      auto validator = New<BleuValidator>(vocabs, config, false);
      validators.push_back(validator);
    } else if(metric == "bleu-detok") {
      auto validator = New<BleuValidator>(vocabs, config, true);
      validators.push_back(validator);
    } else if(metric == "accuracy") {
      auto validator = New<AccuracyValidator>(vocabs, config);
      validators.push_back(validator);
    } else if(metric == "bert-lm-accuracy") {
      auto validator = New<BertAccuracyValidator>(vocabs, config, true);
      validators.push_back(validator);
    } else if(metric == "bert-sentence-accuracy") {
      auto validator = New<BertAccuracyValidator>(vocabs, config, false);
      validators.push_back(validator);
    } else {
      LOG_VALID(warn, "Unrecognized validation metric: {}", metric);
    }
  }

  return validators;
}
}  // namespace marian
