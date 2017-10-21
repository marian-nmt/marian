#pragma once

#include <cstdio>
#include <cstdlib>
#include <limits>

#include "common/config.h"
#include "common/utils.h"
#include "data/batch_generator.h"
#include "data/corpus.h"
#include "graph/expression_graph.h"
#include "translator/beam_search.h"
#include "translator/history.h"
#include "translator/output_collector.h"
#include "translator/printer.h"
#include "translator/scorers.h"

namespace marian {

/**
 * @brief Base class for validators
 */
class ValidatorBase {
public:
  ValidatorBase(bool lowerIsBetter)
      : lowerIsBetter_(lowerIsBetter),
        lastBest_{lowerIsBetter_ ? std::numeric_limits<float>::max()
                                 : std::numeric_limits<float>::lowest()} {}

  virtual float validate(Ptr<ExpressionGraph> graph) = 0;
  virtual std::string type() = 0;

  size_t stalled() { return stalled_; }

protected:
  bool lowerIsBetter_{true};
  float lastBest_;
  size_t stalled_{0};
};

template <class DataSet>
class Validator : public ValidatorBase {
public:
  Validator(std::vector<Ptr<Vocab>> vocabs,
            Ptr<Config> options,
            bool lowerIsBetter = true)
      : ValidatorBase(lowerIsBetter), options_(options), vocabs_(vocabs) {}

  virtual float validate(Ptr<ExpressionGraph> graph) {
    using namespace data;

    graph->setInference(true);

    // Update validation options
    auto opts = New<Config>(*options_);
    opts->set("max-length", options_->get<size_t>("valid-max-length"));
    if(options_->has("valid-mini-batch"))
      opts->set("mini-batch", options_->get<size_t>("valid-mini-batch"));

    // Create corpus
    auto validPaths = options_->get<std::vector<std::string>>("valid-sets");
    auto corpus = New<DataSet>(validPaths, vocabs_, opts);

    // Generate batches
    auto batchGenerator = New<BatchGenerator<DataSet>>(corpus, opts);
    batchGenerator->prepare(false);

    // Validate on batches
    float val = validateBG(graph, batchGenerator);
    updateStalled(graph, val);

    graph->setInference(false);

    return val;
  };

protected:
  std::vector<Ptr<Vocab>> vocabs_;
  Ptr<Config> options_;
  Ptr<models::ModelBase> builder_;

  virtual float validateBG(Ptr<ExpressionGraph>,
                           Ptr<data::BatchGenerator<DataSet>>)
      = 0;

  void updateStalled(Ptr<ExpressionGraph> graph, float val) {
    if((lowerIsBetter_ && lastBest_ > val)
       || (!lowerIsBetter_ && lastBest_ < val)) {
      stalled_ = 0;
      lastBest_ = val;
      if(options_->get<bool>("keep-best"))
        keepBest(graph);
    } else {
      stalled_++;
    }
  }

  virtual void keepBest(Ptr<ExpressionGraph> graph) {
    auto model = options_->get<std::string>("model");
    builder_->save(graph, model + ".best-" + type() + ".npz", true);
  }
};

class CrossEntropyValidator : public Validator<data::Corpus> {
public:
  CrossEntropyValidator(std::vector<Ptr<Vocab>> vocabs, Ptr<Config> options)
      : Validator(vocabs, options) {
    Ptr<Options> opts = New<Options>();
    opts->merge(options);
    opts->set("inference", true);
    opts->set("cost-type", "ce-sum");
    builder_ = models::from_options(opts);
  }

  std::string type() { return options_->get<std::string>("cost-type"); }

protected:
  virtual float validateBG(
      Ptr<ExpressionGraph> graph,
      Ptr<data::BatchGenerator<data::Corpus>> batchGenerator) {
    auto ctype = options_->get<std::string>("cost-type");

    float cost = 0;
    size_t samples = 0;
    size_t words = 0;

    while(*batchGenerator) {
      auto batch = batchGenerator->next();
      auto costNode = builder_->build(graph, batch);
      graph->forward();

      cost += costNode->scalar();
      samples += batch->size();
      words += batch->back()->batchWords();
    }

    if(ctype == "perplexity")
      return std::exp(cost / words);
    if(ctype == "ce-mean-words")
      return cost / words;
    if(ctype == "ce-sum")
      return cost;
    else
      return cost / samples;
  }
};

class ScriptValidator : public Validator<data::Corpus> {
public:
  ScriptValidator(std::vector<Ptr<Vocab>> vocabs, Ptr<Config> options)
      : Validator(vocabs, options, false) {
    Ptr<Options> opts = New<Options>();
    opts->merge(options);
    opts->set("inference", true);
    builder_ = models::from_options(opts);

    UTIL_THROW_IF2(!options_->has("valid-script-path"),
                   "valid-script metric but no script given");
  }

  virtual float validate(Ptr<ExpressionGraph> graph) {
    using namespace data;
    auto model = options_->get<std::string>("model");
    builder_->save(graph, model + ".dev.npz", true);

    auto command = options_->get<std::string>("valid-script-path");
    auto valStr = Exec(command);
    float val = std::atof(valStr.c_str());
    updateStalled(graph, val);

    return val;
  };

  std::string type() { return "valid-script"; }

protected:
  virtual float validateBG(
      Ptr<ExpressionGraph> graph,
      Ptr<data::BatchGenerator<data::Corpus>> batchGenerator) {
    return 0;
  }
};

class TranslationValidator : public Validator<data::Corpus> {
public:
  template <class... Args>
  TranslationValidator(std::vector<Ptr<Vocab>> vocabs, Ptr<Config> options)
      : Validator(vocabs, options, false) {
    Ptr<Options> opts = New<Options>();
    opts->merge(options);
    opts->set("inference", true);
    builder_ = models::from_options(opts);

    if(!options_->has("valid-script-path"))
      LOG_VALID(warn,
                "No post-processing script given for validating translator");
  }

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
    auto corpus = New<data::Corpus>(srcPaths, srcVocabs, opts);

    // Generate batches
    auto batchGenerator = New<BatchGenerator<data::Corpus>>(corpus, opts);
    batchGenerator->prepare(false);

    // Create scorer
    auto model = options_->get<std::string>("model");
    Ptr<Scorer> scorer = New<ScorerWrapper>(builder_, "", 1.0f, model);
    std::vector<Ptr<Scorer>> scorers = {scorer};

    // Set up output file
    std::string fileName;
    Ptr<TemporaryFile> tempFile;

    if(options_->has("trans-output")) {
      fileName = options_->get<std::string>("trans-output");
    } else {
      tempFile.reset(
          new TemporaryFile(options_->get<std::string>("tempdir"), false));
      fileName = tempFile->getFileName();
    }

    LOG_VALID(info, "Translating validation set...");

    graph->setInference(true);
    boost::timer::cpu_timer timer;

    {
      auto collector = options_->has("trans-output")
                           ? New<OutputCollector>(fileName)
                           : New<OutputCollector>(*tempFile);

      size_t sentenceId = 0;

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

        // int id = batch->getSentenceIds()[0];
        // LOG(info, "Best translation {}: {}", id, best1.str());

        sentenceId++;
      }
    }

    LOG_VALID(info, "Total translation time: {}", timer.format(5, "%ws"));
    graph->setInference(false);

    float val = 0.0f;

    // Run post-processing script if given
    if(options_->has("valid-script-path")) {
      auto command
          = options_->get<std::string>("valid-script-path") + " " + fileName;
      auto valStr = Exec(command);
      val = std::atof(valStr.c_str());
      updateStalled(graph, val);
    }

    return val;
  };

  std::string type() { return "translation"; }

protected:
  virtual float validateBG(
      Ptr<ExpressionGraph> graph,
      Ptr<data::BatchGenerator<data::Corpus>> batchGenerator) {
    return 0;
  }
};

/**
 * @brief Creates validators from options
 *
 * If no validation metrics are specified in the options, a cross entropy
 * validator is created by default.
 *
 * @param vocabs Source and target vocabularies
 * @param config Config options
 *
 * @return Vector of validator objects
 */
std::vector<Ptr<Validator<data::Corpus>>> Validators(
    std::vector<Ptr<Vocab>> vocabs, Ptr<Config> config);
}
