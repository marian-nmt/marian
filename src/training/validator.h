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

  virtual float initScore();
  virtual void actAfterLoaded(TrainingState& state) override;
};

template <class DataSet, class BuilderType> // @TODO: BuilderType doesn't really serve a purpose here? Review and remove.
class Validator : public ValidatorBase {
public:
  Validator(std::vector<Ptr<Vocab>> vocabs, Ptr<Options> options, bool lowerIsBetter = true)
      : ValidatorBase(lowerIsBetter),
        vocabs_(vocabs),
        // options_ is a clone of global options, so it can be safely modified within the class
        options_(New<Options>(options->clone())) {
    // set options common for all validators
    options_->set("inference", true);
    options_->set("shuffle", "none"); // don't shuffle validation sets

    if(options_->has("valid-max-length")) {
      options_->set("max-length", options_->get<size_t>("valid-max-length"));
      options_->set("max-length-crop", true); // @TODO: make this configureable
    }
    if(options_->has("valid-mini-batch"))
      options_->set("mini-batch", options_->get<size_t>("valid-mini-batch"));
    options_->set("mini-batch-sort", "src");
    options_->set("maxi-batch", 10);
  }

  typedef typename DataSet::batch_ptr BatchPtr;

protected:
  // Create the BatchGenerator. Note that ScriptValidator does not use batchGenerator_.
  void createBatchGenerator(bool /*isTranslating*/) {
    // Create corpus
    auto validPaths = options_->get<std::vector<std::string>>("valid-sets");
    auto corpus = New<DataSet>(validPaths, vocabs_, options_);

    // Create batch generator
    batchGenerator_ = New<data::BatchGenerator<DataSet>>(corpus, options_);
  }
public:

  virtual float validate(const std::vector<Ptr<ExpressionGraph>>& graphs) override {
    for(auto graph : graphs)
      graph->setInference(true);

    batchGenerator_->prepare();

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
  Ptr<BuilderType> builder_; // @TODO: remove, this is not guaranteed to be state-free, hence not thread-safe, but we are using validators with multi-threading.
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
    std::string suffix = model.substr(model.size() - 4);
    ABORT_IF(suffix != ".npz" && suffix != ".bin", "Unknown model suffix {}", suffix);

    builder_->save(graphs[0], model + ".best-" + type() + suffix, true);
  }
};

class CrossEntropyValidator : public Validator<data::Corpus, models::ICriterionFunction> {
  using Validator::BatchPtr;

public:
  CrossEntropyValidator(std::vector<Ptr<Vocab>> vocabs, Ptr<Options> options);

  std::string type() override { return options_->get<std::string>("cost-type"); }

protected:
  virtual float validateBG(const std::vector<Ptr<ExpressionGraph>>& graphs) override;
};

// Used for validating with classifiers. Compute prediction accuracy versus ground truth for a set of classes
class AccuracyValidator : public Validator<data::Corpus, models::IModel> {
public:
  AccuracyValidator(std::vector<Ptr<Vocab>> vocabs, Ptr<Options> options);

  std::string type() override { return "accuracy"; }

protected:
  virtual float validateBG(const std::vector<Ptr<ExpressionGraph>>& graphs) override;
};

class BertAccuracyValidator : public Validator<data::Corpus, models::IModel> {
private:
  bool evalMaskedLM_{true};

public:
  BertAccuracyValidator(std::vector<Ptr<Vocab>> vocabs, Ptr<Options> options, bool evalMaskedLM);

  std::string type() override {
    if(evalMaskedLM_)
      return "bert-lm-accuracy";
    else
      return "bert-sentence-accuracy";
  }

protected:
  virtual float validateBG(const std::vector<Ptr<ExpressionGraph>>& graphs) override;
};


class ScriptValidator : public Validator<data::Corpus, models::IModel> {
public:
  ScriptValidator(std::vector<Ptr<Vocab>> vocabs, Ptr<Options> options);

  virtual float validate(const std::vector<Ptr<ExpressionGraph>>& graphs) override;

  std::string type() override { return "valid-script"; }

protected:
  virtual float validateBG(const std::vector<Ptr<ExpressionGraph>>& /*graphs*/) override {
    return 0;
  }
};

// validator that translates and computes BLEU (or any metric) with an external script
class TranslationValidator : public Validator<data::Corpus, models::IModel> {
public:
  TranslationValidator(std::vector<Ptr<Vocab>> vocabs, Ptr<Options> options);

  virtual float validate(const std::vector<Ptr<ExpressionGraph>>& graphs) override;

  std::string type() override { return "translation"; }

protected:
  bool quiet_{false};

  virtual float validateBG(const std::vector<Ptr<ExpressionGraph>>& /*graphs*/) override {
    return 0;
  }
};

// validator that translates and computes BLEU internally, with or without decoding
// @TODO: combine with TranslationValidator (above) to avoid code duplication
class BleuValidator : public Validator<data::Corpus, models::IModel> {
public:
  BleuValidator(std::vector<Ptr<Vocab>> vocabs, Ptr<Options> options, bool detok = false);

  virtual float validate(const std::vector<Ptr<ExpressionGraph>>& graphs) override;

  // @TODO: why do we return this string, but not pass it to the constructor?
  std::string type() override { return detok_ ? "bleu-detok" : "bleu"; }

protected:
  // Tokenizer function adapted from multi-bleu-detok.pl, corresponds to sacreBLEU.py
  static std::string tokenize(const std::string& text) {
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

  static std::string tokenizeContinuousScript(const std::string& sUTF8) {
    // We want BLEU-like scores that are comparable across different tokenization schemes.
    // For continuous scripts (Chinese, Japanese, Thai), we would need a language-specific
    // statistical word segmenter, which is outside the scope of Marian. As a practical
    // compromise, we segment continuous-script sequences into individual characters, while
    // leaving Western scripts as words. This way we can use the same settings for Western
    // languages, where Marian would report SacreBLEU scores, and Asian languages, where
    // scores are not standard but internally comparable across tokenization schemes.
    // @TODO: Check what sacrebleu.py is doing, and whether we can replicate that here faithfully.
    auto in = utils::utf8ToUnicodeString(sUTF8);
    auto out = in.substr(0, 0); // (out should be same type as in, don't want to bother with exact type)
    for (auto c : in) {
      auto isCS = utils::isContinuousScript(c);
      if (isCS) // surround continuous-script chars by spaces on each side
        out.push_back(' '); // (duplicate spaces are ignored when splitting later)
      out.push_back(c);
      if (isCS)
        out.push_back(' ');
    }
    return utils::utf8FromUnicodeString(out);
  }

  std::vector<std::string> decode(const Words& words, bool addEOS = false);

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
                   Word eos);

  float calcBLEU(const std::vector<float>& stats);

  virtual float validateBG(const std::vector<Ptr<ExpressionGraph>>& /*graphs*/) override {
    return 0;
  }

private:
  bool detok_;
  bool quiet_{ false };
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
std::vector<Ptr<ValidatorBase/*<data::Corpus>*/>> Validators(
    std::vector<Ptr<Vocab>> vocabs,
    Ptr<Options> config);
}  // namespace marian
