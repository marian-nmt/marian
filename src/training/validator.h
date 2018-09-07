#pragma once

#include <cstdio>
#include <cstdlib>
#include <limits>

#include "3rd_party/threadpool.h"
#include "common/config.h"
#include "common/utils.h"
#include "data/batch_generator.h"
#include "data/corpus.h"
#include "graph/expression_graph.h"
#include "training/training_state.h"
#include "translator/beam_search.h"
#include "translator/history.h"
#include "translator/output_collector.h"
#include "translator/output_printer.h"
#include "translator/scorers.h"

namespace marian {

/**
 * @brief Base class for validators
 */
class ValidatorBase : public TrainingObserver {
protected:
  bool lowerIsBetter_{true};
  float lastBest_;
  size_t stalled_{0};
  std::mutex mutex_;

public:
  ValidatorBase(bool lowerIsBetter)
      : lowerIsBetter_(lowerIsBetter), lastBest_{initScore()} {}

  virtual float validate(const std::vector<Ptr<ExpressionGraph>>& graphs) = 0;
  virtual std::string type() = 0;

  float lastBest() { return lastBest_; }
  size_t stalled() { return stalled_; }

  virtual float initScore() {
    return lowerIsBetter_ ? std::numeric_limits<float>::max()
                          : std::numeric_limits<float>::lowest();
  }

  virtual void actAfterLoaded(TrainingState& state) override {
    if(state.validators[type()]) {
      lastBest_ = state.validators[type()]["last-best"].as<float>();
      stalled_ = state.validators[type()]["stalled"].as<size_t>();
    }
  }
};

template <class DataSet>
class Validator : public ValidatorBase {
public:
  Validator(std::vector<Ptr<Vocab>> vocabs,
            Ptr<Config> options,
            bool lowerIsBetter = true)
      : ValidatorBase(lowerIsBetter), vocabs_(vocabs), options_(options) {}

  virtual float validate(const std::vector<Ptr<ExpressionGraph>>& graphs) override {
    using namespace data;

    for(auto graph : graphs)
      graph->setInference(true);

    // Update validation options
    auto opts = New<Config>(*options_);
    opts->set("max-length", options_->get<size_t>("valid-max-length"));
    if(options_->has("valid-mini-batch"))
      opts->set("mini-batch", options_->get<size_t>("valid-mini-batch"));
    opts->set("mini-batch-sort", "src");

    // Create corpus
    auto validPaths = options_->get<std::vector<std::string>>("valid-sets");
    auto corpus = New<DataSet>(validPaths, vocabs_, opts);

    // Generate batches
    auto batchGenerator = New<BatchGenerator<DataSet>>(corpus, opts);
    batchGenerator->prepare(false);

    // Validate on batches
    float val = validateBG(graphs, batchGenerator);
    updateStalled(graphs, val);

    for(auto graph : graphs)
      graph->setInference(false);

    return val;
  };

protected:
  std::vector<Ptr<Vocab>> vocabs_;
  Ptr<Config> options_;
  Ptr<models::ModelBase> builder_;

  virtual float validateBG(const std::vector<Ptr<ExpressionGraph>>&,
                           Ptr<data::BatchGenerator<DataSet>>)
      = 0;

  void updateStalled(const std::vector<Ptr<ExpressionGraph>>& graphs,
                     float val) {
    if((lowerIsBetter_ && lastBest_ > val)
       || (!lowerIsBetter_ && lastBest_ < val)) {
      stalled_ = 0;
      lastBest_ = val;
      if(options_->get<bool>("keep-best"))
        keepBest(graphs);
    } else {
      stalled_++;
    }
  }

  virtual void keepBest(const std::vector<Ptr<ExpressionGraph>>& graphs) {
    auto model = options_->get<std::string>("model");
    builder_->save(graphs[0], model + ".best-" + type() + ".npz", true);
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
    builder_ = models::from_options(opts, models::usage::scoring);
  }

  std::string type() override { return options_->get<std::string>("cost-type"); }

protected:
  virtual float validateBG(
      const std::vector<Ptr<ExpressionGraph>>& graphs,
      Ptr<data::BatchGenerator<data::Corpus>> batchGenerator) override {
    auto ctype = options_->get<std::string>("cost-type");

    float cost = 0;
    size_t samples = 0;
    size_t words = 0;

    size_t batchId = 0;

    {
      ThreadPool threadPool(graphs.size(), graphs.size());
      Ptr<Options> opts = New<Options>();
      opts->merge(options_);
      opts->set("inference", true);
      opts->set("cost-type", "ce-sum");

      while(*batchGenerator) {
        auto batch = batchGenerator->next();

        auto task = [=, &cost, &samples, &words](size_t id) {
          thread_local Ptr<ExpressionGraph> graph;
          thread_local auto builder
              = models::from_options(opts, models::usage::scoring);

          if(!graph) {
            graph = graphs[id % graphs.size()];
          }

          builder->clear(graph);
          auto costNode = builder->build(graph, batch);
          graph->forward();

          std::unique_lock<std::mutex> lock(mutex_);
          cost += costNode->scalar();
          samples += batch->size();
          words += batch->back()->batchWords();
        };

        threadPool.enqueue(task, batchId);
        batchId++;
      }
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
    builder_ = models::from_options(opts, models::usage::raw);

    ABORT_IF(!options_->has("valid-script-path"),
             "valid-script metric but no script given");
  }

  virtual float validate(const std::vector<Ptr<ExpressionGraph>>& graphs) override {
    using namespace data;
    auto model = options_->get<std::string>("model");
    builder_->save(graphs[0], model + ".dev.npz", true);

    auto command = options_->get<std::string>("valid-script-path");
    auto valStr = utils::Exec(command);
    float val = (float)std::atof(valStr.c_str());
    updateStalled(graphs, val);

    return val;
  };

  std::string type() override { return "valid-script"; }

protected:
  virtual float validateBG(
      const std::vector<Ptr<ExpressionGraph>>& /*graphs*/,
      Ptr<data::BatchGenerator<data::Corpus>> /*batchGenerator*/) override {
    return 0;
  }
};

class TranslationValidator : public Validator<data::Corpus> {
public:
  TranslationValidator(std::vector<Ptr<Vocab>> vocabs, Ptr<Config> options)
      : Validator(vocabs, options, false),
        quiet_(options_->get<bool>("quiet-translation")) {
    Ptr<Options> opts = New<Options>();
    opts->merge(options);
    opts->set("inference", true);
    builder_ = models::from_options(opts, models::usage::translation);

    if(!options_->has("valid-script-path"))
      LOG_VALID(warn,
                "No post-processing script given for validating translator");
  }

  virtual float validate(const std::vector<Ptr<ExpressionGraph>>& graphs) override {
    using namespace data;

    // Temporary options for translation
    auto opts = New<Config>(*options_);
    opts->set("mini-batch", options_->get<int>("valid-mini-batch"));
    opts->set("maxi-batch", 10);
    opts->set("max-length", 1000);

    // Create corpus
    auto validPaths = options_->get<std::vector<std::string>>("valid-sets");
    std::vector<std::string> paths(validPaths.begin(), validPaths.end());
    auto corpus = New<data::Corpus>(paths, vocabs_, opts);

    // Generate batches
    auto batchGenerator = New<BatchGenerator<data::Corpus>>(corpus, opts);
    batchGenerator->prepare(false);

    // Create scorer
    auto model = options_->get<std::string>("model");

    auto mopts = New<Options>();
    mopts->merge(options_);
    mopts->set("inference", true);

    std::vector<Ptr<Scorer>> scorers;
    for(auto graph : graphs) {
      auto builder = models::from_options(mopts, models::usage::translation);
      Ptr<Scorer> scorer = New<ScorerWrapper>(builder, "", 1.0f, model);
      scorers.push_back(scorer);
    }

    // Set up output file
    std::string fileName;
    Ptr<TemporaryFile> tempFile;

    if(options_->has("valid-translation-output")) {
      fileName = options_->get<std::string>("valid-translation-output");
    } else {
      tempFile.reset(
          new TemporaryFile(options_->get<std::string>("tempdir"), false));
      fileName = tempFile->getFileName();
    }

    for(auto graph : graphs)
      graph->setInference(true);

    if(!quiet_)
      LOG(info, "Translating validation set...");

    boost::timer::cpu_timer timer;
    {
      auto printer = New<OutputPrinter>(options_, vocabs_.back());
      auto collector = options_->has("valid-translation-output")
                           ? New<OutputCollector>(fileName)
                           : New<OutputCollector>(*tempFile);

      if(quiet_)
        collector->setPrintingStrategy(New<QuietPrinting>());
      else
        collector->setPrintingStrategy(New<GeometricPrinting>());

      size_t sentenceId = 0;

      ThreadPool threadPool(graphs.size(), graphs.size());

      // @TODO: unify this and get rid of Config object.
      auto tOptions = New<Options>();
      tOptions->merge(options_);

      while(*batchGenerator) {
        auto batch = batchGenerator->next();

        auto task = [=](size_t id) {
          thread_local Ptr<ExpressionGraph> graph;
          thread_local Ptr<Scorer> scorer;

          if(!graph) {
            graph = graphs[id % graphs.size()];
            scorer = scorers[id % graphs.size()];
          }

          auto search = New<BeamSearch>(tOptions,
                                        std::vector<Ptr<Scorer>>{scorer},
                                        vocabs_.back()->GetEosId(),
                                        vocabs_.back()->GetUnkId());
          auto histories = search->search(graph, batch);

          for(auto history : histories) {
            std::stringstream best1;
            std::stringstream bestn;
            printer->print(history, best1, bestn);
            collector->Write((long)history->GetLineNum(),
                             best1.str(),
                             bestn.str(),
                             options_->get<bool>("n-best"));
          }
        };

        threadPool.enqueue(task, sentenceId);
        sentenceId++;
      }
    }

    if(!quiet_)
      LOG(info, "Total translation time: {}", timer.format(5, "%ws"));

    for(auto graph : graphs)
      graph->setInference(false);

    float val = 0.0f;

    // Run post-processing script if given
    if(options_->has("valid-script-path")) {
      auto command
          = options_->get<std::string>("valid-script-path") + " " + fileName;
      auto valStr = utils::Exec(command);
      val = (float)std::atof(valStr.c_str());
      updateStalled(graphs, val);
    }

    return val;
  };

  std::string type() override { return "translation"; }

protected:
  bool quiet_{false};

  virtual float validateBG(
      const std::vector<Ptr<ExpressionGraph>>& /*graphs*/,
      Ptr<data::BatchGenerator<data::Corpus>> /*batchGenerator*/) override {
    return 0;
  }
};

// @TODO: combine with TranslationValidator (above) to avoid code duplication
class BleuValidator : public Validator<data::Corpus> {
public:
  BleuValidator(std::vector<Ptr<Vocab>> vocabs, Ptr<Config> options)
      : Validator(vocabs, options, false),
        quiet_(options_->get<bool>("quiet-translation")) {
    Ptr<Options> opts = New<Options>();
    opts->merge(options);
    opts->set("inference", true);
    builder_ = models::from_options(opts, models::usage::translation);

    if(!options_->has("valid-script-path"))
      LOG_VALID(warn,
                "No post-processing script given for validating translator");
  }

  virtual float validate(const std::vector<Ptr<ExpressionGraph>>& graphs) override {
    using namespace data;

    // Temporary options for translation
    auto opts = New<Config>(*options_);
    opts->set("mini-batch", options_->get<int>("valid-mini-batch"));
    opts->set("maxi-batch", 10);
    opts->set("max-length", 1000);

    // Create corpus
    auto validPaths = options_->get<std::vector<std::string>>("valid-sets");
    std::vector<std::string> paths(validPaths.begin(), validPaths.end());
    auto corpus = New<data::Corpus>(paths, vocabs_, opts);

    // Generate batches
    auto batchGenerator = New<BatchGenerator<data::Corpus>>(corpus, opts);
    batchGenerator->prepare(false);

    // Create scorer
    auto model = options_->get<std::string>("model");

    auto mopts = New<Options>();
    mopts->merge(options_);
    mopts->set("inference", true);

    std::vector<Ptr<Scorer>> scorers;
    for(auto graph : graphs) {
      auto builder = models::from_options(mopts, models::usage::translation);
      Ptr<Scorer> scorer = New<ScorerWrapper>(builder, "", 1.0f, model);
      scorers.push_back(scorer);
    }

    for(auto graph : graphs)
      graph->setInference(true);

    if(!quiet_)
      LOG(info, "Translating validation set...");

    // 0: 1-grams matched, 1: 1-grams total,
    // ...,
    // 6: 4-grams matched, 7: 4-grams total,
    // 8: reference length
    std::vector<float> stats(9, 0.f);

    boost::timer::cpu_timer timer;
    {
      size_t sentenceId = 0;

      ThreadPool threadPool(graphs.size(), graphs.size());

      // @TODO: unify this and get rid of Config object.
      auto tOptions = New<Options>();
      tOptions->merge(options_);

      while(*batchGenerator) {
        auto batch = batchGenerator->next();

        auto task = [=, &stats](size_t id) {
          thread_local Ptr<ExpressionGraph> graph;
          thread_local Ptr<Scorer> scorer;

          if(!graph) {
            graph = graphs[id % graphs.size()];
            scorer = scorers[id % graphs.size()];
          }

          auto search = New<BeamSearch>(tOptions,
                                        std::vector<Ptr<Scorer>>{scorer},
                                        vocabs_.back()->GetEosId(),
                                        vocabs_.back()->GetUnkId());
          auto histories = search->search(graph, batch);

          size_t no = 0;
          std::lock_guard<std::mutex> statsLock(mutex_);
          for(auto history : histories) {
            auto result = history->Top();
            const auto& words = std::get<0>(result);
            updateStats(stats, words, batch, no, vocabs_.back()->GetEosId());
            no++;
          }
        };

        threadPool.enqueue(task, sentenceId);
        sentenceId++;
      }
    }

    if(!quiet_)
      LOG(info, "Total translation time: {}", timer.format(5, "%ws"));

    for(auto graph : graphs)
      graph->setInference(false);

    float val = calcBLEU(stats);
    updateStalled(graphs, val);

    return val;
  };

  std::string type() override { return "bleu"; }

protected:
  bool quiet_{false};

  void updateStats(std::vector<float>& stats,
                   const Words& cand,
                   const Ptr<data::Batch> batch,
                   size_t no,
                   Word eos) {
    auto corpusBatch = std::static_pointer_cast<data::CorpusBatch>(batch);
    auto subBatch = corpusBatch->back();

    size_t size = subBatch->batchSize();
    size_t width = subBatch->batchWidth();

    Words ref;  // fill ref
    for(size_t i = 0; i < width; ++i) {
      Word w = subBatch->data()[i * size + no];
      if(w == eos)
        break;
      ref.push_back(w);
    }

    std::map<std::vector<Word>, size_t> rgrams;
    for(size_t i = 0; i < ref.size(); ++i) {
      // template deduction for std::min<T> seems to be weird under VS due to
      // macros in windows.h hence explicit type to avoid macro parsing.
      for(size_t l = 1; l <= std::min<size_t>(4ul, ref.size() - i); ++l) {
        std::vector<Word> ngram(l);
        std::copy(ref.begin() + i, ref.begin() + i + l, ngram.begin());
        rgrams[ngram]++;
      }
    }

    std::map<std::vector<Word>, size_t> tgrams;
    for(size_t i = 0; i < cand.size() - 1; ++i) {
      for(size_t l = 1; l <= std::min<size_t>(4ul, cand.size() - 1 - i); ++l) {
        std::vector<Word> ngram(l);
        std::copy(cand.begin() + i, cand.begin() + i + l, ngram.begin());
        tgrams[ngram]++;
      }
    }

    for(auto& ngramcount : tgrams) {
      size_t l = ngramcount.first.size();
      size_t tc = ngramcount.second;
      size_t rc = rgrams[ngramcount.first];

      stats[2 * l - 2] += std::min<size_t>(tc, rc);
      stats[2 * l - 1] += tc;
    }

    stats[8] += ref.size();
  }

  float calcBLEU(const std::vector<float>& stats) {
    float logbleu = 0;
    for(int i = 0; i < 8; i += 2) {
      if(stats[i] == 0.f)
        return 0.f;
      logbleu += std::log(stats[i] / stats[i + 1]);
    }

    logbleu /= 4.f;

    float brev_penalty = 1.f - std::max(stats[8] / stats[1], 1.f);
    return std::exp(logbleu + brev_penalty) * 100;
  }

  virtual float validateBG(
      const std::vector<Ptr<ExpressionGraph>>& /*graphs*/,
      Ptr<data::BatchGenerator<data::Corpus>> /*batchGenerator*/) override {
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
    std::vector<Ptr<Vocab>> vocabs,
    Ptr<Config> config);
}  // namespace marian
