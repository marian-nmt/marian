#include "training/validator.h"

namespace marian {

std::vector<Ptr<ValidatorBase/*<data::Corpus>*/>> Validators(
    std::vector<Ptr<Vocab>> vocabs,
    Ptr<Options> config) {
  std::vector<Ptr<ValidatorBase/*<data::Corpus>*/>> validators;

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
    } else if(metric == "bleu" || metric == "bleu-detok" || metric == "bleu-segmented" || metric == "chrf") {
      auto validator = New<SacreBleuValidator>(vocabs, config, metric);
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
      ABORT("Unknown validation metric: {}", metric);
    }
  }

  return validators;
}


///////////////////////////////////////////////////////////////////////////////////////
float ValidatorBase::initScore() {
  return lowerIsBetter_ ? std::numeric_limits<float>::max() : std::numeric_limits<float>::lowest();
}

void ValidatorBase::actAfterLoaded(TrainingState& state) {
  if(state.validators[type()]) {
    lastBest_ = state.validators[type()]["last-best"].as<float>();
    stalled_ = state.validators[type()]["stalled"].as<size_t>();
  }
}

///////////////////////////////////////////////////////////////////////////////////////
CrossEntropyValidator::CrossEntropyValidator(std::vector<Ptr<Vocab>> vocabs, Ptr<Options> options)
    : Validator(vocabs, options) {
  createBatchGenerator(/*isTranslating=*/false);

  auto opts = options_->with("inference",
                             true,  // @TODO: check if required
                             "cost-type",
                             "ce-sum");
  // @TODO: remove, only used for saving?
  builder_ = models::createCriterionFunctionFromOptions(opts, models::usage::scoring);
}

float CrossEntropyValidator::validateBG(const std::vector<Ptr<ExpressionGraph>>& graphs) {
  auto ctype = options_->get<std::string>("cost-type");

  // @TODO: use with(...) everywhere, this will help with creating immutable options.
  // Make options const everywhere and get rid of "set"?
  auto opts = options_->with("inference", true, "cost-type", "ce-sum");

  StaticLoss loss;
  size_t samples = 0;
  std::deque<Ptr<ExpressionGraph>> graphQueue(graphs.begin(), graphs.end());

  auto task = [=, &loss, &samples, &graphQueue](BatchPtr batch) {
    thread_local Ptr<ExpressionGraph> graph;

    if(!graph) {
      std::unique_lock<std::mutex> lock(mutex_);
      ABORT_IF(graphQueue.empty(), "Asking for graph, but none left on queue");
      graph = graphQueue.front();
      graphQueue.pop_front();
    }

    auto builder = models::createCriterionFunctionFromOptions(options_, models::usage::scoring);

    builder->clear(graph);
    auto dynamicLoss = builder->build(graph, batch);
    graph->forward();

    std::unique_lock<std::mutex> lock(mutex_);
    loss += *dynamicLoss;
    samples += batch->size();
  };

  {
    threadPool_.reserve(graphs.size());
    TaskBarrier taskBarrier;
    for(auto batch : *batchGenerator_)
      taskBarrier.push_back(threadPool_.enqueue(task, batch));
    // ~TaskBarrier waits until all are done
  }

  if(ctype == "perplexity")
    return std::exp(loss.loss / loss.count);
  if(ctype == "ce-mean-words")
    return loss.loss / loss.count;
  if(ctype == "ce-sum")
    return loss.loss;
  else
    return loss.loss / samples;  // @TODO: back-compat, to be removed
}

///////////////////////////////////////////////////////////////////////////////////////
AccuracyValidator::AccuracyValidator(std::vector<Ptr<Vocab>> vocabs, Ptr<Options> options)
    : Validator(vocabs, options, /*lowerIsBetter=*/false) {
  createBatchGenerator(/*isTranslating=*/false);

  // @TODO: remove, only used for saving?
  builder_ = models::createModelFromOptions(options_, models::usage::raw);
}

float AccuracyValidator::validateBG(const std::vector<Ptr<ExpressionGraph>>& graphs) {
  size_t correct = 0;
  size_t totalLabels = 0;
  std::deque<Ptr<ExpressionGraph>> graphQueue(graphs.begin(), graphs.end());

  auto task = [=, &correct, &totalLabels, &graphQueue](BatchPtr batch) {
    thread_local Ptr<ExpressionGraph> graph;

    if(!graph) {
      std::unique_lock<std::mutex> lock(mutex_);
      ABORT_IF(graphQueue.empty(), "Asking for graph, but none left on queue");
      graph = graphQueue.front();
      graphQueue.pop_front();
    }

    auto builder = models::createModelFromOptions(options_, models::usage::raw);

    builder->clear(graph);
    Expr logits = builder->build(graph, batch).getLogits();
    graph->forward();

    std::vector<float> vLogits;
    logits->val()->get(vLogits);

    const auto& groundTruth = batch->back()->data();

    IndexType cols = logits->shape()[-1];

    size_t thisCorrect = 0;
    size_t thisLabels = groundTruth.size();

    for(int i = 0; i < thisLabels; ++i) {
      // CPU-side Argmax
      Word bestWord = Word::NONE;
      float bestValue = std::numeric_limits<float>::lowest();
      for(IndexType j = 0; j < cols; ++j) {
        float currValue = vLogits[i * cols + j];
        if(currValue > bestValue) {
          bestValue = currValue;
          bestWord = Word::fromWordIndex(j);
        }
      }
      thisCorrect += (size_t)(bestWord == groundTruth[i]);
    }

    std::unique_lock<std::mutex> lock(mutex_);
    totalLabels += thisLabels;
    correct += thisCorrect;
  };

  {
    threadPool_.reserve(graphs.size());

    TaskBarrier taskBarrier;
    for(auto batch : *batchGenerator_)
      taskBarrier.push_back(threadPool_.enqueue(task, batch));

    // ~TaskBarrier waits until all are done
  }

  return (float)correct / (float)totalLabels;
}

///////////////////////////////////////////////////////////////////////////////////////
BertAccuracyValidator::BertAccuracyValidator(std::vector<Ptr<Vocab>> vocabs,
                                           Ptr<Options> options,
                                           bool evalMaskedLM)
    : Validator(vocabs, options, /*lowerIsBetter=*/false), evalMaskedLM_(evalMaskedLM) {
  createBatchGenerator(/*isTranslating=*/false);
  // @TODO: remove, only used for saving?
  builder_ = models::createModelFromOptions(options_, models::usage::raw);
}

float BertAccuracyValidator::validateBG(const std::vector<Ptr<ExpressionGraph>>& graphs) {
  size_t correct = 0;
  size_t totalLabels = 0;
  size_t batchId = 0;
  std::deque<Ptr<ExpressionGraph>> graphQueue(graphs.begin(), graphs.end());

  auto task = [=, &correct, &totalLabels, &graphQueue](BatchPtr batch, size_t batchId) {
    thread_local Ptr<ExpressionGraph> graph;

    if(!graph) {
      std::unique_lock<std::mutex> lock(mutex_);
      ABORT_IF(graphQueue.empty(), "Asking for graph, but none left on queue");
      graph = graphQueue.front();
      graphQueue.pop_front();
    }

    auto builder = models::createModelFromOptions(options_, models::usage::raw);

    thread_local std::unique_ptr<std::mt19937> engine;
    if(!engine)
      engine.reset(new std::mt19937((unsigned int)(Config::seed + batchId)));

    auto bertBatch = New<data::BertBatch>(batch,
                                          *engine,
                                          options_->get<float>("bert-masking-fraction"),
                                          options_->get<std::string>("bert-mask-symbol"),
                                          options_->get<std::string>("bert-sep-symbol"),
                                          options_->get<std::string>("bert-class-symbol"),
                                          options_->get<int>("bert-type-vocab-size"));

    builder->clear(graph);
    auto classifierStates
        = std::dynamic_pointer_cast<BertEncoderClassifier>(builder)->apply(graph, bertBatch, true);
    graph->forward();

    auto maskedLMLogits = classifierStates[0]->getLogProbs();
    const auto& maskedLMLabels = bertBatch->bertMaskedWords();

    auto sentenceLogits = classifierStates[1]->getLogProbs();
    const auto& sentenceLabels = bertBatch->back()->data();

    auto count = [=, &correct, &totalLabels](Expr logits, const Words& labels) {
      IndexType cols = logits->shape()[-1];
      size_t thisCorrect = 0;
      size_t thisLabels = labels.size();

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
        thisCorrect += (size_t)(bestIndex == labels[i].toWordIndex());
      }

      std::unique_lock<std::mutex> lock(mutex_);
      totalLabels += thisLabels;
      correct += thisCorrect;
    };

    if(evalMaskedLM_)
      count(maskedLMLogits, maskedLMLabels);
    else
      count(sentenceLogits, sentenceLabels);
  };

  {
    threadPool_.reserve(graphs.size());
    TaskBarrier taskBarrier;
    for(auto batch : *batchGenerator_) {
      taskBarrier.push_back(threadPool_.enqueue(task, batch, batchId));
      batchId++;
    }
    // ~TaskBarrier waits until all are done
  }

  return (float)correct / (float)totalLabels;
}

///////////////////////////////////////////////////////////////////////////////////////
ScriptValidator::ScriptValidator(std::vector<Ptr<Vocab>> vocabs, Ptr<Options> options)
    : Validator(vocabs, options, false) {
  // @TODO: remove, only used for saving?
  builder_ = models::createModelFromOptions(options_, models::usage::raw);

  ABORT_IF(!options_->hasAndNotEmpty("valid-script-path"),
           "valid-script metric but no script given");
}

float ScriptValidator::validate(const std::vector<Ptr<ExpressionGraph>>& graphs,
                                Ptr<const TrainingState> /*ignored*/) {
  using namespace data;
  auto model = options_->get<std::string>("model");
  std::string suffix = model.substr(model.size() - 4);
  ABORT_IF(suffix != ".npz" && suffix != ".bin", "Unknown model suffix {}", suffix);

  builder_->save(graphs[0], model + ".dev" + suffix, true);

  auto valStr = utils::exec(options_->get<std::string>("valid-script-path"),
                            options_->get<std::vector<std::string>>("valid-script-args"));
  float val = (float)std::atof(valStr.c_str());
  updateStalled(graphs, val);

  return val;
}

///////////////////////////////////////////////////////////////////////////////////////
TranslationValidator::TranslationValidator(std::vector<Ptr<Vocab>> vocabs, Ptr<Options> options)
    : Validator(vocabs, options, false), quiet_(options_->get<bool>("quiet-translation")) {
  // @TODO: remove, only used for saving?
  builder_ = models::createModelFromOptions(options_, models::usage::translation);

  if(!options_->hasAndNotEmpty("valid-script-path"))
    LOG_VALID(warn, "No post-processing script given for validating translator");

  createBatchGenerator(/*isTranslating=*/true);
}

float TranslationValidator::validate(const std::vector<Ptr<ExpressionGraph>>& graphs,
                                     Ptr<const TrainingState> state) {
  using namespace data;

  // Generate batches
  batchGenerator_->prepare();

  // Create scorer
  auto model = options_->get<std::string>("model");

  std::vector<Ptr<Scorer>> scorers;
  for(auto graph : graphs) {
    auto builder = models::createModelFromOptions(options_, models::usage::translation);
    Ptr<Scorer> scorer = New<ScorerWrapper>(builder, "", 1.0f, model);
    scorers.push_back(scorer);  // @TODO: should this be done in the contructor?
  }

  // Set up output file
  std::string fileName;
  Ptr<io::TemporaryFile> tempFile;

  if(options_->hasAndNotEmpty("valid-translation-output")) {
    fileName = options_->get<std::string>("valid-translation-output");
    // fileName can be a template with fields for training state parameters:
    fileName = state->fillTemplate(fileName);
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
                         : New<OutputCollector>(tempFile->getFileName());

    if(quiet_)
      collector->setPrintingStrategy(New<QuietPrinting>());
    else
      collector->setPrintingStrategy(New<GeometricPrinting>());

    std::deque<Ptr<ExpressionGraph>> graphQueue(graphs.begin(), graphs.end());
    std::deque<Ptr<Scorer>> scorerQueue(scorers.begin(), scorers.end());
    auto task = [=, &graphQueue, &scorerQueue](BatchPtr batch) {
      thread_local Ptr<ExpressionGraph> graph;
      thread_local Ptr<Scorer> scorer;

      if(!graph) {
        std::unique_lock<std::mutex> lock(mutex_);
        ABORT_IF(graphQueue.empty(), "Asking for graph, but none left on queue");
        graph = graphQueue.front();
        graphQueue.pop_front();

        ABORT_IF(scorerQueue.empty(), "Asking for scorer, but none left on queue");
        scorer = scorerQueue.front();
        scorerQueue.pop_front();
      }

      auto search = New<BeamSearch>(options_, std::vector<Ptr<Scorer>>{scorer}, vocabs_.back());
      auto histories = search->search(graph, batch);

      for(auto history : histories) {
        std::stringstream best1;
        std::stringstream bestn;
        printer->print(history, best1, bestn);
        collector->Write(
            (long)history->getLineNum(), best1.str(), bestn.str(), options_->get<bool>("n-best"));
      }
    };

    threadPool_.reserve(graphs.size());
    TaskBarrier taskBarrier;
    for(auto batch : *batchGenerator_)
      taskBarrier.push_back(threadPool_.enqueue(task, batch));
    // ~TaskBarrier waits until all are done
  }

  if(!quiet_)
    LOG(info, "Total translation time: {:.5f}s", timer.elapsed());

  for(auto graph : graphs)
    graph->setInference(false);

  float val = 0.0f;

  // Run post-processing script if given
  if(options_->hasAndNotEmpty("valid-script-path")) {
    // auto command = options_->get<std::string>("valid-script-path") + " " + fileName;
    // auto valStr = utils::exec(command);
    auto valStr = utils::exec(options_->get<std::string>("valid-script-path"),
                              options_->get<std::vector<std::string>>("valid-script-args"),
                              fileName);
    val = (float)std::atof(valStr.c_str());
    updateStalled(graphs, val);
  }

  return val;
};

///////////////////////////////////////////////////////////////////////////////////////
SacreBleuValidator::SacreBleuValidator(std::vector<Ptr<Vocab>> vocabs, Ptr<Options> options, const std::string& metric)
    : Validator(vocabs, options, /*lowerIsBetter=*/false),
      metric_(metric),
      computeChrF_(metric == "chrf"),
      useWordIds_(metric == "bleu-segmented"),
      quiet_(options_->get<bool>("quiet-translation")) {

  ABORT_IF(computeChrF_ && useWordIds_, "Cannot compute ChrF on word ids"); // should not really happen, but let's check.

  if(computeChrF_) // according to SacreBLEU implementation this is the default for ChrF,
    order_ = 6;    // we compute stats over character ngrams up to length 6

  // @TODO: remove, only used for saving?
  builder_ = models::createModelFromOptions(options_, models::usage::translation);
  auto vocab = vocabs_.back();
  createBatchGenerator(/*isTranslating=*/true);
}

float SacreBleuValidator::validate(const std::vector<Ptr<ExpressionGraph>>& graphs,
                              Ptr<const TrainingState> state) {
  using namespace data;

  // Generate batches
  batchGenerator_->prepare();

  // Create scorer
  auto model = options_->get<std::string>("model");

  // @TODO: check if required - Temporary options for translation
  auto mopts = New<Options>();
  mopts->merge(options_);
  mopts->set("inference", true);

  std::vector<Ptr<Scorer>> scorers;
  for(auto graph : graphs) {
    auto builder = models::createModelFromOptions(options_, models::usage::translation);
    Ptr<Scorer> scorer = New<ScorerWrapper>(builder, "", 1.0f, model);
    scorers.push_back(scorer);
  }

  for(auto graph : graphs)
    graph->setInference(true);

  if(!quiet_)
    LOG(info, "Translating validation set...");

  // For BLEU
  // 0: 1-grams matched, 1: 1-grams cand total, 2: 1-grams ref total (used in ChrF)
  // ...,
  // n: reference length (used in BLEU)
  std::vector<float> stats(statsPerOrder * order_ + 1, 0.f);

  timer::Timer timer;
  {
    auto printer = New<OutputPrinter>(options_, vocabs_.back());

    Ptr<OutputCollector> collector;
    if(options_->hasAndNotEmpty("valid-translation-output")) {
      auto fileName = options_->get<std::string>("valid-translation-output");
      // fileName can be a template with fields for training state parameters:
      fileName = state->fillTemplate(fileName);
      collector = New<OutputCollector>(fileName);  // for debugging
    } else {
      collector = New<OutputCollector>(/* null */);  // don't print, but log
    }

    if(quiet_)
      collector->setPrintingStrategy(New<QuietPrinting>());
    else
      collector->setPrintingStrategy(New<GeometricPrinting>());

    std::deque<Ptr<ExpressionGraph>> graphQueue(graphs.begin(), graphs.end());
    std::deque<Ptr<Scorer>> scorerQueue(scorers.begin(), scorers.end());
    auto task = [=, &stats, &graphQueue, &scorerQueue](BatchPtr batch) {
      thread_local Ptr<ExpressionGraph> graph;
      thread_local Ptr<Scorer> scorer;

      if(!graph) {
        std::unique_lock<std::mutex> lock(mutex_);
        ABORT_IF(graphQueue.empty(), "Asking for graph, but none left on queue");
        graph = graphQueue.front();
        graphQueue.pop_front();

        ABORT_IF(scorerQueue.empty(), "Asking for scorer, but none left on queue");
        scorer = scorerQueue.front();
        scorerQueue.pop_front();
      }

      auto search = New<BeamSearch>(options_, std::vector<Ptr<Scorer>>{scorer}, vocabs_.back());
      auto histories = search->search(graph, batch);

      size_t no = 0;
      std::lock_guard<std::mutex> statsLock(mutex_);
      for(auto history : histories) {
        auto result = history->top();
        const auto& words = std::get<0>(result);
        updateStats(stats, words, batch, no);

        std::stringstream best1;
        std::stringstream bestn;
        printer->print(history, best1, bestn);
        collector->Write((long)history->getLineNum(),
                         best1.str(),
                         bestn.str(),
                         /*nbest=*/false);
        no++;
      }
    };

    threadPool_.reserve(graphs.size());
    TaskBarrier taskBarrier;
    for(auto batch : *batchGenerator_)
      taskBarrier.push_back(threadPool_.enqueue(task, batch));
    // ~TaskBarrier waits until all are done
  }

  if(!quiet_)
    LOG(info, "Total translation time: {:.5f}s", timer.elapsed());

  for(auto graph : graphs)
    graph->setInference(false);

  float val = computeChrF_ ? calcChrF(stats) : calcBLEU(stats);
  updateStalled(graphs, val);

  return val;
}

std::vector<std::string> SacreBleuValidator::decode(const Words& words, bool addEOS) {
  auto vocab = vocabs_.back();
  auto tokenString = vocab->surfaceForm(words);  // detokenize to surface form

  auto vocabType = vocab->type();
  if(vocabType == "FactoredVocab" || vocabType == "SentencePieceVocab") {
    LOG_VALID_ONCE(info, "Decoding validation set with {} for scoring", vocabType);
    tokenString = tokenize(tokenString); // tokenize according to SacreBLEU rules
    if(!computeChrF_) // for ChrF, we break into characters below, so no need to do this here
      tokenString = tokenizeContinuousScript(tokenString);  // CJT scripts only: further break into characters
  } else {
    LOG_VALID_ONCE(info, "{} keeps original segments for scoring", vocabType);
  }

  auto tokens = computeChrF_ ? splitIntoUnicodeChars(tokenString, /*removeWhiteSpace=*/true) // break into vector of unicode chars (as utf8 strings) for ChrF
                             : utils::splitAny(tokenString, " ", /*keepEmpty=*/false);       // or just split according to whitespace for BLEU

  if(addEOS)
    tokens.push_back("</s>");
  return tokens;
}

void SacreBleuValidator::updateStats(std::vector<float>& stats,
                                     const Words& cand,
                                     const Ptr<data::Batch> batch,
                                     size_t no) {
  auto vocab = vocabs_.back();

  auto corpusBatch = std::static_pointer_cast<data::CorpusBatch>(batch);
  auto subBatch = corpusBatch->back();

  size_t size = subBatch->batchSize();
  size_t width = subBatch->batchWidth();

  Words ref;  // fill ref
  for(size_t i = 0; i < width; ++i) {
    Word w = subBatch->data()[i * size + no];
    if(w == vocab->getEosId())
      break;
    if(w == vocab->getUnkId())
      LOG_VALID_ONCE(info, "References contain unknown word, metric scores may be inaccurate");
    ref.push_back(w);
  }

  LOG_VALID_ONCE(info, "First sentence's tokens as scored:");
  LOG_VALID_ONCE(info, "  Hyp: {}", utils::join(decode(cand, /*addEOS=*/false)));
  LOG_VALID_ONCE(info, "  Ref: {}", utils::join(decode(ref,  /*addEOS=*/false)));

  if(useWordIds_)
    updateStats(stats, cand, ref);
  else
    updateStats(stats, decode(cand, /*addEOS=*/false), decode(ref, /*addEOS=*/false));

}

// Re-implementation of BLEU metric from SacreBLEU
float SacreBleuValidator::calcBLEU(const std::vector<float>& stats) {
  float logbleu = 0;
  for(int i = 0; i < order_; ++i) {
    float commonNgrams     = stats[statsPerOrder * i + 0];
    float hypothesesNgrams = stats[statsPerOrder * i + 1];

    if(commonNgrams == 0.f)
      return 0.f;
    logbleu += std::log(commonNgrams) - std::log(hypothesesNgrams);
  }

  logbleu /= order_;

  float refLen = stats[statsPerOrder * order_];
  float hypUnigrams = stats[1];
  float brev_penalty = 1.f - std::max(refLen / hypUnigrams, 1.f);
  return std::exp(logbleu + brev_penalty) * 100.f;
}

// Re-implementation of ChrF metric from SacreBLEU, using standard parameters
float SacreBleuValidator::calcChrF(const std::vector<float>& stats) {
  float beta = 2.f;

  float avgPrecision    = 0.f;
  float avgRecall       = 0.f;
  size_t effectiveOrder = 0;

  for(size_t i = 0; i < order_; ++i) {
    float commonNgrams     = stats[statsPerOrder * i + 0];
    float hypothesesNgrams = stats[statsPerOrder * i + 1];
    float referencesNgrams = stats[statsPerOrder * i + 2];

    if(hypothesesNgrams > 0 && referencesNgrams > 0) {
        avgPrecision += commonNgrams / hypothesesNgrams;
        avgRecall    += commonNgrams / referencesNgrams;
        effectiveOrder++;
    }
  }

  if(effectiveOrder == 0)
      return 0.f;

  avgPrecision /= effectiveOrder;
  avgRecall    /= effectiveOrder;

  if(avgPrecision + avgRecall == 0.f)
    return 0.f;

  auto betaSquare = beta * beta;
  auto score = (1.f + betaSquare) * (avgPrecision * avgRecall) / ((betaSquare * avgPrecision) + avgRecall);
  return score * 100.f; // we multiply by 100 which is usually not done for ChrF, but this makes it more comparable to BLEU
}

}  // namespace marian
