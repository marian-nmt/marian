#pragma once

#include <cstdio>
#include <cstdlib>
#include <limits>

#include "common/config.h"
#include "data/batch_generator.h"
#include "data/corpus.h"
#include "graph/expression_graph.h"
#include "translator/beam_search.h"
#include "translator/history.h"
#include "translator/output_collector.h"
#include "translator/printer.h"

#include "translator/printer.h"
#include "translator/scorers.h"


namespace marian {

template <class DataSet>
class Validator {
protected:
  Ptr<Config> options_;
  std::vector<Ptr<Vocab>> vocabs_;
  float lastBest_;
  size_t stalled_{0};

public:
  Validator(std::vector<Ptr<Vocab>> vocabs, Ptr<Config> options)
      : options_(options),
        vocabs_(vocabs),
        lastBest_{lowerIsBetter() ? std::numeric_limits<float>::max() :
                                    std::numeric_limits<float>::lowest()} {}

  virtual std::string type() = 0;

  virtual void keepBest(Ptr<ExpressionGraph> graph) = 0;

  virtual bool lowerIsBetter() { return true; }

  virtual void initLastBest() {
    lastBest_ = lowerIsBetter() ? std::numeric_limits<float>::max() :
                                  std::numeric_limits<float>::lowest();
  }

  size_t stalled() { return stalled_; }

  virtual float validate(Ptr<ExpressionGraph> graph) {
    using namespace data;

    graph->setInference(true);

    auto validPaths = options_->get<std::vector<std::string>>("valid-sets");
    auto corpus = New<DataSet>(validPaths, vocabs_, options_);
    Ptr<BatchGenerator<DataSet>> batchGenerator
        = New<BatchGenerator<DataSet>>(corpus, options_);
    if(options_->has("valid-mini-batch"))
      batchGenerator->forceBatchSize(options_->get<int>("valid-mini-batch"));
    batchGenerator->prepare(false);

    float val = validateBG(graph, batchGenerator);

    graph->setInference(false);

    if((lowerIsBetter() && lastBest_ > val)
       || (!lowerIsBetter() && lastBest_ < val)) {
      stalled_ = 0;
      lastBest_ = val;
      if(options_->get<bool>("keep-best"))
        keepBest(graph);
    } else {
      stalled_++;
    }

    return val;
  };

protected:
  virtual float validateBG(Ptr<ExpressionGraph>,
                           Ptr<data::BatchGenerator<DataSet>>)
      = 0;
};

template <class Builder>
class CrossEntropyValidator : public Validator<data::Corpus> {
private:
  Ptr<Builder> builder_;

public:
  template <class... Args>
  CrossEntropyValidator(std::vector<Ptr<Vocab>> vocabs,
                        Ptr<Config> options,
                        Args... args)
      : Validator(vocabs, options) {

    Ptr<Options> temp = New<Options>();
    temp->merge(options);
    temp->set("inference", true);
    builder_ = models::from_options(temp);

    initLastBest();
  }

  virtual float validateBG(
      Ptr<ExpressionGraph> graph,
      Ptr<data::BatchGenerator<data::Corpus>> batchGenerator) {
    float cost = 0;
    size_t samples = 0;

    while(*batchGenerator) {
      auto batch = batchGenerator->next();
      auto costNode = builder_->build(graph, batch);
      graph->forward();

      cost += costNode->scalar() * batch->size();
      samples += batch->size();
    }

    return cost / samples;
  }

  virtual void keepBest(Ptr<ExpressionGraph> graph) {
    auto model = options_->get<std::string>("model");
    builder_->save(graph, model + ".best-" + type() + ".npz", true);
  }

  std::string type() { return options_->get<std::string>("cost-type"); }
};

template <class Builder>
class ScriptValidator : public Validator<data::Corpus> {
private:
  Ptr<Builder> builder_;

public:
  template <class... Args>
  ScriptValidator(std::vector<Ptr<Vocab>> vocabs,
                  Ptr<Config> options,
                  Args... args)
      : Validator(vocabs, options) {

    Ptr<Options> temp = New<Options>();
    temp->merge(options);
    temp->set("inference", true);
    builder_ = models::from_options(temp);

    initLastBest();
  }

  std::string exec(const std::string& cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::shared_ptr<std::FILE> pipe(popen(cmd.c_str(), "r"), pclose);
    if(!pipe)
      UTIL_THROW2("popen() failed!");

    while(!std::feof(pipe.get())) {
      if(std::fgets(buffer.data(), 128, pipe.get()) != NULL)
        result += buffer.data();
    }
    return result;
  }

  virtual bool lowerIsBetter() { return false; }

  virtual float validate(Ptr<ExpressionGraph> graph) {
    using namespace data;
    auto model = options_->get<std::string>("model");

    builder_->save(graph, model + ".dev.npz", true);

    UTIL_THROW_IF2(!options_->has("valid-script-path"),
                   "valid-script metric but no script given");
    auto command = options_->get<std::string>("valid-script-path");

    auto valStr = exec(command);
    float val = std::atof(valStr.c_str());

    if((lowerIsBetter() && lastBest_ > val)
       || (!lowerIsBetter() && lastBest_ < val)) {
      stalled_ = 0;
      lastBest_ = val;
      if(options_->get<bool>("keep-best"))
        keepBest(graph);
    } else {
      stalled_++;
    }

    return val;
  };

  virtual float validateBG(
      Ptr<ExpressionGraph> graph,
      Ptr<data::BatchGenerator<data::Corpus>> batchGenerator) {
    return 0;
  }

  virtual void keepBest(Ptr<ExpressionGraph> graph) {
    auto model = options_->get<std::string>("model");
    builder_->save(graph, model + ".best-" + type() + ".npz", true);
  }

  std::string type() { return "valid-script"; }
};


template <class Builder>
class TranslationValidator : public Validator<data::Corpus> {
private:
  Ptr<Builder> builder_;

public:
  template <class... Args>
  TranslationValidator(std::vector<Ptr<Vocab>> vocabs,
                       Ptr<Config> options,
                       Args... args)
      : Validator(vocabs, options) {

    Ptr<Options> opts = New<Options>();
    opts->merge(options);
    opts->set("inference", true);
    builder_ = models::from_options(opts);

    initLastBest();
  }

  virtual bool lowerIsBetter() { return false; }

  virtual float validate(Ptr<ExpressionGraph> graph) {
    using namespace data;

    // Temporary options for translation
    auto opts = New<Config>(*options_);
    opts->set("mini-batch", 1);
    opts->set("maxi-batch", 1);
    opts->set("max-length", 1000);

    // Create corpus
    auto validPaths = options_->get<std::vector<std::string>>("valid-sets");
    std::vector<std::string> srcPaths(validPaths.begin(), validPaths.end() - 1);
    std::vector<Ptr<Vocab>> srcVocabs(vocabs_.begin(), vocabs_.end() - 1);
    auto corpus = New<Corpus>(srcPaths, srcVocabs, opts);

    // Generate batches
    Ptr<BatchGenerator<Corpus>> batchGenerator
        = New<BatchGenerator<Corpus>>(corpus, opts);
    batchGenerator->prepare(false);

    // Create scorer
    auto model = options_->get<std::string>("model");
    Ptr<Scorer> scorer = New<ScorerWrapper>(builder_, "", 1.0f, model);
    std::vector<Ptr<Scorer>> scorers = { scorer };

    // Create output collector
    UTIL_THROW_IF2(!options_->has("trans-output"),
                   "translation but no output file given");
    auto outputFile = options_->get<std::string>("trans-output");
    auto collector = New<OutputCollector>(outputFile);

    size_t sentenceId = 0;

    LOG(valid)->info("Translating...");

    graph->setInference(true);
    {
      while(*batchGenerator) {
        auto batch = batchGenerator->next();

        graph->clear();
        auto search = New<BeamSearch>(options_, scorers);
        auto history = search->search(graph, batch, sentenceId);

        std::stringstream best1;
        std::stringstream bestn;
        Printer(options_, vocabs_.back(), history, best1, bestn);
        collector->Write(history->GetLineNum(),
                         best1.str(),
                         bestn.str(),
                         options_->get<bool>("n-best"));

        int id = batch->getSentenceIds()[0];
        LOG(valid)->info("Best translation {}: {}", id, best1.str());

        sentenceId++;
      }
    }
    graph->setInference(false);

    // TODO: change me!
    return 0.0f;
  };

  virtual float validateBG(
      Ptr<ExpressionGraph> graph,
      Ptr<data::BatchGenerator<data::Corpus>> batchGenerator) {
    return 0;
  }

  virtual void keepBest(Ptr<ExpressionGraph> graph) {
    // TODO: decide what to do with this method
  }

  std::string type() { return "translation"; }
};

template <class Builder, class... Args>
std::vector<Ptr<Validator<data::Corpus>>> Validators(
    std::vector<Ptr<Vocab>> vocabs, Ptr<Config> config, Args... args) {
  std::vector<Ptr<Validator<data::Corpus>>> validators;

  auto validMetrics = config->get<std::vector<std::string>>("valid-metrics");

  std::vector<std::string> ceMetrics = {
    "cross-entropy", "ce-mean", "ce-sum",
    "ce-mean-words", "perplexity"
  };

  for(auto metric : validMetrics) {
    if(std::find(ceMetrics.begin(), ceMetrics.end(), metric)
       != ceMetrics.end()) {
      Ptr<Config> opts = New<Config>(*config);
      opts->set("cost-type", metric);

      auto validator
          = New<CrossEntropyValidator<Builder>>(vocabs, opts, args...);
      validators.push_back(validator);
    } else if(metric == "valid-script") {
      auto validator = New<ScriptValidator<Builder>>(vocabs, config, args...);
      validators.push_back(validator);
    } else if(metric == "translation") {
      auto validator = New<TranslationValidator<Builder>>(vocabs, config, args...);
      validators.push_back(validator);
    } else {
      LOG(valid)->info("Unrecognized validation metric: {}", metric);
    }
  }

  if(validators.empty()) {
    LOG(valid)->info("No validation metric specified, using 'cross-entropy'");

    Ptr<Config> opts = New<Config>(*config);
    opts->set("cost-type", "cross-entropy");

    auto validator
        = New<CrossEntropyValidator<Builder>>(vocabs, opts, args...);
    validators.push_back(validator);
  }

  return validators;
}

}
