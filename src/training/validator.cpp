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

float ScriptValidator::validate(const std::vector<Ptr<ExpressionGraph>>& graphs) {
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

float TranslationValidator::validate(const std::vector<Ptr<ExpressionGraph>>& graphs) {
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
BleuValidator::BleuValidator(std::vector<Ptr<Vocab>> vocabs, Ptr<Options> options, bool detok)
    : Validator(vocabs, options, false),
      detok_(detok),
      quiet_(options_->get<bool>("quiet-translation")) {
  // @TODO: remove, only used for saving?
  builder_ = models::createModelFromOptions(options_, models::usage::translation);

  // @TODO: replace bleu-detok by a separate parameter to enable (various forms of) detok
  auto vocab = vocabs_.back();
  ABORT_IF(detok_ && vocab->type() != "SentencePieceVocab" && vocab->type() != "FactoredVocab",
           "Detokenizing BLEU validator expects the target vocabulary to be SentencePieceVocab or "
           "FactoredVocab. "
           "Current vocabulary type is {}",
           vocab->type());

  createBatchGenerator(/*isTranslating=*/true);
}

float BleuValidator::validate(const std::vector<Ptr<ExpressionGraph>>& graphs) {
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
        updateStats(stats, words, batch, no, vocabs_.back()->getEosId());

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

  float val = calcBLEU(stats);
  updateStalled(graphs, val);

  return val;
}

std::vector<std::string> BleuValidator::decode(const Words& words, bool addEOS) {
  auto vocab = vocabs_.back();
  auto tokenString = vocab->surfaceForm(words);  // detokenize to surface form
  tokenString = tokenize(tokenString);           // tokenize according to SacreBLEU rules
  tokenString
      = tokenizeContinuousScript(tokenString);  // CJT scripts only: further break into characters
  auto tokens = utils::splitAny(tokenString, " ");
  if(addEOS)
    tokens.push_back("</s>");
  return tokens;
}

void BleuValidator::updateStats(std::vector<float>& stats,
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

  bool detok = detok_;
#if 1  // hack for now, to get this feature when running under Flo
  // Problem is that Flo pieces that pass 'bleu' do not know whether vocab is factored,
  // hence cannot select 'bleu-detok'.
  // @TODO: We agreed that we will replace bleu-detok by bleu with an additional
  // parameter to select the detokenization method, which will default to detok for
  // FactoredSegmenter, and no-op for base vocab.
  if(vocabs_.back()->type() == "FactoredVocab") {
    if(!quiet_)
      LOG_ONCE(info, "[valid] FactoredVocab implies using detokenized BLEU");
    detok = true;  // always use bleu-detok
  }
#endif
  if(detok) {  // log the first detokenized string
    LOG_ONCE(info, "[valid] First sentence's tokens after detokenization, as scored:");
    LOG_ONCE(info, "[valid]  Hyp: {}", utils::join(decode(cand, /*addEOS=*/true)));
    LOG_ONCE(info, "[valid]  Ref: {}", utils::join(decode(ref)));
  }
  if(detok)
    updateStats(stats, decode(cand, /*addEOS=*/true), decode(ref));
  else
    updateStats(stats, cand, ref);
}

float BleuValidator::calcBLEU(const std::vector<float>& stats) {
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

}  // namespace marian
