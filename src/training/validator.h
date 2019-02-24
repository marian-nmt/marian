#pragma once

#include "3rd_party/threadpool.h"
#include "common/config.h"
#include "common/timer.h"
#include "common/utils.h"
#include "common/regex.h"
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
#include "models/bert.h"

#include <cstdio>
#include <cstdlib>
#include <limits>

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
  ThreadPool threadPool_;

public:
  ValidatorBase(bool lowerIsBetter) : lowerIsBetter_(lowerIsBetter), lastBest_{initScore()} {}

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
  Validator(std::vector<Ptr<Vocab>> vocabs, Ptr<Options> options, bool lowerIsBetter = true)
      : ValidatorBase(lowerIsBetter),
        vocabs_(vocabs),
        // options_ is a clone of global options, so it can be safely modified within the class
        options_(New<Options>(options->clone())) {
    // set options common for all validators
    options_->set("inference", true);
    if(options_->has("valid-max-length"))
      options_->set("max-length", options_->get<size_t>("valid-max-length"));
    if(options_->has("valid-mini-batch"))
      options_->set("mini-batch", options_->get<size_t>("valid-mini-batch"));
    options_->set("mini-batch-sort", "src");
    options_->set("maxi-batch", 10);
  }

protected:
  void createBatchGenerator(bool isTranslating) {
    // Create the BatchGenerator. Note that ScriptValidator does not use batchGenerator_.

    // Update validation options
    auto opts = New<Options>();
    opts->merge(options_);
    opts->set("inference", true);

    if (isTranslating) { // TranslationValidator and BleuValidator
      opts->set("max-length", 1000);
      opts->set("mini-batch", options_->get<int>("valid-mini-batch"));
      opts->set("maxi-batch", 10);
    }
    else { // CrossEntropyValidator
      opts->set("max-length", options_->get<size_t>("valid-max-length"));
      if(options_->has("valid-mini-batch"))
        opts->set("mini-batch", options_->get<size_t>("valid-mini-batch"));
      opts->set("mini-batch-sort", "src");
    }

    // Create corpus
    auto validPaths = options_->get<std::vector<std::string>>("valid-sets");
    auto corpus = New<DataSet>(validPaths, vocabs_, options_);

    // Create batch generator
    batchGenerator_ = New<data::BatchGenerator<DataSet>>(corpus, opts);
  }
public:

  virtual float validate(const std::vector<Ptr<ExpressionGraph>>& graphs) override {
    for(auto graph : graphs)
      graph->setInference(true);

    batchGenerator_->prepare(false);

    // Validate on batches
    float val = validateBG(graphs);
    updateStalled(graphs, val);

    for(auto graph : graphs)
      graph->setInference(false);

    return val;
  };

protected:
  std::vector<Ptr<Vocab>> vocabs_;
  Ptr<Options> options_;
  Ptr<models::ModelBase> builder_;
  Ptr<data::BatchGenerator<DataSet>> batchGenerator_;

  virtual float validateBG(const std::vector<Ptr<ExpressionGraph>>&)
      = 0;

  void updateStalled(const std::vector<Ptr<ExpressionGraph>>& graphs,
                     float val) {
    if((lowerIsBetter_ && lastBest_ > val)
       || (!lowerIsBetter_ && lastBest_ < val)) {
      stalled_ = 0;
      lastBest_ = val;
      if(options_->get<bool>("keep-best"))
        keepBest(graphs);
    } else /* if (lastBest_ != val) */ { // (special case 0 at start)  @TODO: needed? Seems stall count gets reset each time it does improve. If not needed, remove "if(...)" again.
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
  CrossEntropyValidator(std::vector<Ptr<Vocab>> vocabs, Ptr<Options> options)
      : Validator(vocabs, options) {
    createBatchGenerator(/*isTranslating=*/false);

    // @TODO: check if this is required.
    Ptr<Options> opts = New<Options>();
    opts->merge(options);
    opts->set("inference", true);
    opts->set("cost-type", "ce-sum");
    builder_ = models::from_options(opts, models::usage::scoring);
  }

  std::string type() override { return options_->get<std::string>("cost-type"); }

protected:
  virtual float validateBG(const std::vector<Ptr<ExpressionGraph>>& graphs) override {
    auto ctype = options_->get<std::string>("cost-type");
    options_->set("cost-type", "ce-sum"); // @TODO: check if still needed, most likely not.

    StaticLoss loss;
    size_t samples = 0;
    size_t batchId = 0;

    {
      threadPool_.reserve(graphs.size());

      TaskBarrier taskBarrier;
      for(auto batch : *batchGenerator_) {
        auto task = [=, &loss, &samples](size_t id) {
          thread_local Ptr<ExpressionGraph> graph;
          thread_local auto builder = models::from_options(options_, models::usage::scoring);

          if(!graph) {
            graph = graphs[id % graphs.size()];
          }

          builder->clear(graph);
          auto dynamicLoss = builder->build(graph, batch);
          graph->forward();

          std::unique_lock<std::mutex> lock(mutex_);
          loss  += *dynamicLoss;
          samples += batch->size();
        };

        taskBarrier.push_back(threadPool_.enqueue(task, batchId));
        batchId++;
      }
      // ~TaskBarrier waits until all are done
    }

    // get back to the original cost type
    options_->set("cost-type", ctype); // @TODO: check if still needed, most likely not.

    if(ctype == "perplexity")
      return std::exp(loss.loss / loss.count);
    if(ctype == "ce-mean-words")
      return loss.loss / loss.count;
    if(ctype == "ce-sum")
      return loss.loss;
    else
      return loss.loss / samples; // @TODO: back-compat, to be removed
  }
};

// Used for validating with classifiers. Compute prediction accuary versus groundtruth for a set of classes
class AccuracyValidator : public Validator<data::Corpus> {
public:
  AccuracyValidator(std::vector<Ptr<Vocab>> vocabs, Ptr<Options> options)
      : Validator(vocabs, options, /*lowerIsBetter=*/false) {
    createBatchGenerator(/*isTranslating=*/false);

    // @TODO: check if this is required.
    Ptr<Options> opts = New<Options>();
    opts->merge(options);
    opts->set("inference", true);
    builder_ = models::from_options(opts, models::usage::raw);
  }

  std::string type() override { return "accuracy"; }

protected:
  virtual float validateBG(const std::vector<Ptr<ExpressionGraph>>& graphs) override {

    size_t correct     = 0;
    size_t totalLabels = 0;
    size_t batchId     = 0;

    {
      threadPool_.reserve(graphs.size());

      TaskBarrier taskBarrier;
      for(auto batch : *batchGenerator_) {
        auto task = [=, &correct, &totalLabels](size_t id) {
          thread_local Ptr<ExpressionGraph> graph;
          thread_local auto builder = models::from_options(options_, models::usage::raw);

          if(!graph) {
            graph = graphs[id % graphs.size()];
          }

          // @TODO: requires argmax implementation and integer arithmetics
          // builder->clear(graph);
          // auto predicted = argmax(builder->build(graph, batch), /*axis*/-1);
          // auto labels    = graph->indices(batch->back()->data());
          // auto correct   = sum(flatten(predicted) == labels);
          // graph->forward();

          // std::unique_lock<std::mutex> lock(mutex_);
          // totalLabels += labels->shape().elements();
          // correct     += correct->scalar<IndexType>();

          builder->clear(graph);
          Expr logits = builder->build(graph, batch)->loss();
          graph->forward();

          std::vector<float> vLogits;
          logits->val()->get(vLogits);

          const auto& groundTruth = batch->back()->data();

          IndexType cols = logits->shape()[-1];

          size_t thisCorrect = 0;
          size_t thisLabels  = groundTruth.size();

          for(int i = 0; i < thisLabels; ++i) {
            // CPU-side Argmax
            IndexType bestIndex = 0;
            float bestValue = std::numeric_limits<float>::lowest();
            for(IndexType j = 0; j < cols; ++j) {
              float currValue = vLogits[i * cols + j];
              if(currValue > bestValue) {
                bestValue = currValue;
                bestIndex = j;
              }
            }
            thisCorrect += (size_t)(bestIndex == groundTruth[i]);
          }

          std::unique_lock<std::mutex> lock(mutex_);
          totalLabels += thisLabels;
          correct     += thisCorrect;
        };

        taskBarrier.push_back(threadPool_.enqueue(task, batchId));
        batchId++;
      }
      // ~TaskBarrier waits until all are done
    }

    return (float)correct / (float)totalLabels;
  }
};

class BertAccuracyValidator : public Validator<data::Corpus> {
private:
  bool evalMaskedLM_{true};

public:
  BertAccuracyValidator(std::vector<Ptr<Vocab>> vocabs, Ptr<Options> options, bool evalMaskedLM)
      : Validator(vocabs, options, /*lowerIsBetter=*/false),
        evalMaskedLM_(evalMaskedLM) {
    createBatchGenerator(/*isTranslating=*/false);

    // @TODO: check if this is required.
    Ptr<Options> opts = New<Options>();
    opts->merge(options);
    opts->set("inference", true);
    builder_ = models::from_options(opts, models::usage::raw);
  }

  std::string type() override {
    if(evalMaskedLM_)
      return "bert-lm-accuracy";
    else
      return "bert-sentence-accuracy";
  }

protected:
  virtual float validateBG(const std::vector<Ptr<ExpressionGraph>>& graphs) override {

    size_t correct     = 0;
    size_t totalLabels = 0;
    size_t batchId     = 0;

    {
      threadPool_.reserve(graphs.size());

      TaskBarrier taskBarrier;
      for(auto batch : *batchGenerator_) {
        auto task = [=, &correct, &totalLabels](size_t id) {
          thread_local Ptr<ExpressionGraph> graph;
          thread_local auto builder = models::from_options(options_, models::usage::raw);
          thread_local std::unique_ptr<std::mt19937> engine;

          if(!graph) {
            graph = graphs[id % graphs.size()];
          }

          if(!engine)
            engine.reset(new std::mt19937((unsigned int)(Config::seed + id)));

          auto bertBatch = New<data::BertBatch>(batch,
                                                *engine,
                                                options_->get<float>("bert-masking-fraction"),
                                                options_->get<std::string>("bert-mask-symbol"),
                                                options_->get<std::string>("bert-sep-symbol"),
                                                options_->get<std::string>("bert-class-symbol"),
                                                options_->get<int>("bert-type-vocab-size"));

          builder->clear(graph);
          auto classifierStates = std::dynamic_pointer_cast<BertEncoderClassifier>(builder)->apply(graph, bertBatch, true);
          graph->forward();

          auto maskedLMLogits = classifierStates[0]->getLogProbs();
          const auto& maskedLMLabels = bertBatch->bertMaskedWords();

          auto sentenceLogits = classifierStates[1]->getLogProbs();
          const auto& sentenceLabels = bertBatch->back()->data();

          auto count = [=, &correct, &totalLabels](Expr logits, const std::vector<IndexType>& labels) {
            IndexType cols = logits->shape()[-1];
            size_t thisCorrect = 0;
            size_t thisLabels  = labels.size();

            std::vector<float> vLogits;
            logits->val()->get(vLogits);

            for(int i = 0; i < thisLabels; ++i) {
              // CPU-side Argmax
              IndexType bestIndex = 0;
              float bestValue = std::numeric_limits<float>::lowest();
              for(IndexType j = 0; j < cols; ++j) {
                float currValue = vLogits[i * cols + j];
                if(currValue > bestValue) {
                  bestValue = currValue;
                  bestIndex = j;
                }
              }
              thisCorrect += (size_t)(bestIndex == labels[i]);
            }

            std::unique_lock<std::mutex> lock(mutex_);
            totalLabels += thisLabels;
            correct     += thisCorrect;
          };

          if(evalMaskedLM_)
            count(maskedLMLogits, maskedLMLabels);
          else
            count(sentenceLogits, sentenceLabels);
        };

        taskBarrier.push_back(threadPool_.enqueue(task, batchId));
        batchId++;
      }
      // ~TaskBarrier waits until all are done
    }

    return (float)correct / (float)totalLabels;
  }
};


class ScriptValidator : public Validator<data::Corpus> {
public:
  ScriptValidator(std::vector<Ptr<Vocab>> vocabs, Ptr<Options> options)
      : Validator(vocabs, options, false) {
    builder_ = models::from_options(options_, models::usage::raw);

    ABORT_IF(!options_->hasAndNotEmpty("valid-script-path"),
             "valid-script metric but no script given");
  }

  virtual float validate(const std::vector<Ptr<ExpressionGraph>>& graphs) override {
    using namespace data;
    auto model = options_->get<std::string>("model");
    builder_->save(graphs[0], model + ".dev.npz", true);

    auto command = options_->get<std::string>("valid-script-path");
    auto valStr = utils::exec(command);
    float val = (float)std::atof(valStr.c_str());
    updateStalled(graphs, val);

    return val;
  };

  std::string type() override { return "valid-script"; }

protected:
  virtual float validateBG(const std::vector<Ptr<ExpressionGraph>>& /*graphs*/) override {
    return 0;
  }
};

class TranslationValidator : public Validator<data::Corpus> {
public:
  TranslationValidator(std::vector<Ptr<Vocab>> vocabs, Ptr<Options> options)
      : Validator(vocabs, options, false),
        quiet_(options_->get<bool>("quiet-translation")) {
    builder_ = models::from_options(options_, models::usage::translation);

    if(!options_->hasAndNotEmpty("valid-script-path"))
      LOG_VALID(warn, "No post-processing script given for validating translator");

    createBatchGenerator(/*isTranslating=*/true);
  }

  virtual float validate(const std::vector<Ptr<ExpressionGraph>>& graphs) override {
    using namespace data;

    // Generate batches
    batchGenerator_->prepare(false);

    // Create scorer
    auto model = options_->get<std::string>("model");

    // Temporary options for translation
    auto mopts = New<Options>();
    mopts->merge(options_);
    mopts->set("inference", true);

    std::vector<Ptr<Scorer>> scorers;
    for(auto graph : graphs) {
      auto builder = models::from_options(options_, models::usage::translation);
      Ptr<Scorer> scorer = New<ScorerWrapper>(builder, "", 1.0f, model);
      scorers.push_back(scorer);
    }

    // Set up output file
    std::string fileName;
    Ptr<io::TemporaryFile> tempFile;

    if(options_->hasAndNotEmpty("valid-translation-output")) {
      fileName = options_->get<std::string>("valid-translation-output");
    } else {
      tempFile.reset(new io::TemporaryFile(options_->get<std::string>("tempdir"), false));
      fileName = tempFile->getFileName();
    }

    for(auto graph : graphs)
      graph->setInference(true);

    if(!quiet_)
      LOG(info, "Translating validation set...");

    timer::Timer timer;
    {
      auto printer = New<OutputPrinter>(options_, vocabs_.back());
      // @TODO: This can be simplified. If there is no "valid-translation-output", fileName already
      // contains the name of temporary file that should be used?
      auto collector = options_->hasAndNotEmpty("valid-translation-output")
                           ? New<OutputCollector>(fileName)
                           : New<OutputCollector>(*tempFile);

      if(quiet_)
        collector->setPrintingStrategy(New<QuietPrinting>());
      else
        collector->setPrintingStrategy(New<GeometricPrinting>());

      threadPool_.reserve(graphs.size());

      size_t sentenceId = 0;
      TaskBarrier taskBarrier;
      for(auto batch : *batchGenerator_) {
        auto task = [=](size_t id) {
          thread_local Ptr<ExpressionGraph> graph;
          thread_local Ptr<Scorer> scorer;

          if(!graph) {
            graph = graphs[id % graphs.size()];
            scorer = scorers[id % graphs.size()];
          }

          auto search = New<BeamSearch>(options_,
                                        std::vector<Ptr<Scorer>>{scorer},
                                        vocabs_.back()->getEosId(),
                                        vocabs_.back()->getUnkId());
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

        taskBarrier.push_back(threadPool_.enqueue(task, sentenceId));
        sentenceId++;
      }
      // ~TaskBarrier waits until all are done
    }

    if(!quiet_)
      LOG(info, "Total translation time: {:.5f}s", timer.elapsed());

    for(auto graph : graphs)
      graph->setInference(false);

    float val = 0.0f;

    // Run post-processing script if given
    if(options_->hasAndNotEmpty("valid-script-path")) {
      auto command = options_->get<std::string>("valid-script-path") + " " + fileName;
      auto valStr = utils::exec(command);
      val = (float)std::atof(valStr.c_str());
      updateStalled(graphs, val);
    }

    return val;
  };

  std::string type() override { return "translation"; }

protected:
  bool quiet_{false};

  virtual float validateBG(const std::vector<Ptr<ExpressionGraph>>& /*graphs*/) override {
    return 0;
  }
};

// @TODO: combine with TranslationValidator (above) to avoid code duplication
class BleuValidator : public Validator<data::Corpus> {
private:
  bool detok_{false};

public:
  BleuValidator(std::vector<Ptr<Vocab>> vocabs, Ptr<Options> options, bool detok = false)
      : Validator(vocabs, options, false),
        detok_(detok),
        quiet_(options_->get<bool>("quiet-translation")) {
    builder_ = models::from_options(options_, models::usage::translation);

#ifdef USE_SENTENCEPIECE
    auto vocab = vocabs_.back();
    ABORT_IF(detok_ && vocab->type() != "SentencePieceVocab",
             "Detokenizing BLEU validator expects the target vocabulary to be SentencePieceVocab. "
             "Current vocabulary type is {}", vocab->type());
#else
    ABORT_IF(detok_,
             "Detokenizing BLEU validator expects the target vocabulary to be SentencePieceVocab. "
             "Marian has not been compiled with SentencePieceVocab support");
#endif

    createBatchGenerator(/*isTranslating=*/true);
  }

  virtual float validate(const std::vector<Ptr<ExpressionGraph>>& graphs) override {
    using namespace data;

    // Generate batches
    batchGenerator_->prepare(false);

    // Create scorer
    auto model = options_->get<std::string>("model");

    // @TODO: check if required - Temporary options for translation
    auto mopts = New<Options>();
    mopts->merge(options_);
    mopts->set("inference", true);

    std::vector<Ptr<Scorer>> scorers;
    for(auto graph : graphs) {
      auto builder = models::from_options(options_, models::usage::translation);
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

    timer::Timer timer;
    {
      auto printer = New<OutputPrinter>(options_, vocabs_.back());

      Ptr<OutputCollector> collector;
      if(options_->hasAndNotEmpty("valid-translation-output")) {
        auto fileName = options_->get<std::string>("valid-translation-output");
        collector = New<OutputCollector>(fileName); // for debugging
      }
      else {
        collector = New<OutputCollector>(/* null */); // don't print, but log
      }

      if(quiet_)
        collector->setPrintingStrategy(New<QuietPrinting>());
      else
        collector->setPrintingStrategy(New<GeometricPrinting>());

      threadPool_.reserve(graphs.size());

      size_t sentenceId = 0;
      TaskBarrier taskBarrier;
      for(auto batch : *batchGenerator_) {
        auto task = [=, &stats](size_t id) {
          thread_local Ptr<ExpressionGraph> graph;
          thread_local Ptr<Scorer> scorer;

          if(!graph) {
            graph = graphs[id % graphs.size()];
            scorer = scorers[id % graphs.size()];
          }

          auto search = New<BeamSearch>(options_,
                                        std::vector<Ptr<Scorer>>{scorer},
                                        vocabs_.back()->getEosId(),
                                        vocabs_.back()->getUnkId());
          auto histories = search->search(graph, batch);

          size_t no = 0;
          std::lock_guard<std::mutex> statsLock(mutex_);
          for(auto history : histories) {
            auto result = history->Top();
            const auto& words = std::get<0>(result);
            updateStats(stats, words, batch, no, vocabs_.back()->getEosId());

            std::stringstream best1;
            std::stringstream bestn;
            printer->print(history, best1, bestn);
            collector->Write((long)history->GetLineNum(),
                             best1.str(),
                             bestn.str(),
                             /*nbest=*/ false);
            no++;
          }
        };

        taskBarrier.push_back(threadPool_.enqueue(task, sentenceId));
        sentenceId++;
      }
      // ~TaskBarrier waits until all are done
    }

    if(!quiet_)
      LOG(info, "Total translation time: {:.5f}s", timer.elapsed());

    for(auto graph : graphs)
      graph->setInference(false);

    float val = calcBLEU(stats);
    updateStalled(graphs, val);

    return val;
  };

  std::string type() override { return detok_ ? "bleu-detok" : "bleu"; }

protected:
  bool quiet_{false};

  // Tokenizer function adapted from multi-bleu-detok.pl, corresponds to sacreBLEU.py
  std::string tokenize(const std::string& text) {
    std::string normText = text;

    // language-independent part:
    normText = regex::regex_replace(normText, regex::regex("<skipped>"), ""); // strip "skipped" tags
    normText = regex::regex_replace(normText, regex::regex("-\\n"), "");      // strip end-of-line hyphenation and join lines
    normText = regex::regex_replace(normText, regex::regex("\\n"), " ");      // join lines
    normText = regex::regex_replace(normText, regex::regex("&quot;"), "\"");  // convert SGML tag for quote to "
    normText = regex::regex_replace(normText, regex::regex("&amp;"), "&");    // convert SGML tag for ampersand to &
    normText = regex::regex_replace(normText, regex::regex("&lt;"), "<");     //convert SGML tag for less-than to >
    normText = regex::regex_replace(normText, regex::regex("&gt;"), ">");     //convert SGML tag for greater-than to <

    // language-dependent part (assuming Western languages):
    normText = " " + normText + " ";
    normText = regex::regex_replace(normText, regex::regex("([\\{-\\~\\[-\\` -\\&\\(-\\+\\:-\\@\\/])"), " $1 "); // tokenize punctuation
    normText = regex::regex_replace(normText, regex::regex("([^0-9])([\\.,])"), "$1 $2 "); // tokenize period and comma unless preceded by a digit
    normText = regex::regex_replace(normText, regex::regex("([\\.,])([^0-9])"), " $1 $2"); // tokenize period and comma unless followed by a digit
    normText = regex::regex_replace(normText, regex::regex("([0-9])(-)"), "$1 $2 ");       // tokenize dash when preceded by a digit
    normText = regex::regex_replace(normText, regex::regex("\\s+"), " "); // one space only between words
    normText = regex::regex_replace(normText, regex::regex("^\\s+"), ""); // no leading space
    normText = regex::regex_replace(normText, regex::regex("\\s+$"), ""); // no trailing space

    return normText;
  }

  std::vector<std::string> decode(const Words& words, bool addEOS = false) {
    auto vocab = vocabs_.back();
    auto tokens = utils::splitAny(tokenize(vocab->decode(words)), " ");
    if(addEOS)
      tokens.push_back("</s>");
    return tokens;
  }

  // Update document-wide sufficient statistics for BLEU with single sentence n-gram stats.
  template <typename T>
  void updateStats(std::vector<float>& stats,
                   const std::vector<T>& cand,
                   const std::vector<T>& ref) {

    std::map<std::vector<T>, size_t> rgrams;
    for(size_t i = 0; i < ref.size(); ++i) {
      // template deduction for std::min<T> seems to be weird under VS due to
      // macros in windows.h hence explicit type to avoid macro parsing.
      for(size_t l = 1; l <= std::min<size_t>(4ul, ref.size() - i); ++l) {
        std::vector<T> ngram(l);
        std::copy(ref.begin() + i, ref.begin() + i + l, ngram.begin());
        rgrams[ngram]++;
      }
    }

    std::map<std::vector<T>, size_t> tgrams;
    for(size_t i = 0; i < cand.size() - 1; ++i) {
      for(size_t l = 1; l <= std::min<size_t>(4ul, cand.size() - 1 - i); ++l) {
        std::vector<T> ngram(l);
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

  // Extract matching target reference from batch and pass on to update BLEU stats
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

    if(detok_)
      updateStats(stats, decode(cand, /*addEOS=*/ true), decode(ref));
    else
      updateStats(stats, cand, ref);
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

  virtual float validateBG(const std::vector<Ptr<ExpressionGraph>>& /*graphs*/) override {
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
    Ptr<Options> config);
}  // namespace marian
