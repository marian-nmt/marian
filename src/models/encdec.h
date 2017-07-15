#pragma once

#include "marian.h"

namespace marian {

class EncoderBase {
protected:
  Ptr<Config> options_;
  std::string prefix_{"encoder"};
  bool inference_{false};

  virtual std::tuple<Expr, Expr> lookup(Expr srcEmbeddings,
                                        Ptr<data::CorpusBatch> batch,
                                        size_t index) {
    using namespace keywords;

    auto subBatch = (*batch)[index];

    int dimBatch = subBatch->batchSize();
    int dimEmb = srcEmbeddings->shape()[1];
    int dimWords = subBatch->batchWidth();

    auto graph = srcEmbeddings->graph();
    auto chosenEmbeddings = rows(srcEmbeddings, subBatch->indices());

    auto batchEmbeddings = reshape(chosenEmbeddings, {dimBatch, dimEmb, dimWords});
    auto batchMask = graph->constant({dimBatch, 1, dimWords},
                                     init = inits::from_vector(subBatch->mask()));

    return std::make_tuple(batchEmbeddings, batchMask);
  }

public:
  template <class... Args>
  EncoderBase(Ptr<Config> options, Args... args)
      : options_(options),
        prefix_(Get(keywords::prefix, "encoder", args...)),
        inference_(Get(keywords::inference, false, args...)) {}

  virtual Ptr<EncoderState> build(Ptr<ExpressionGraph>,
                                  Ptr<data::CorpusBatch>,
                                  size_t) = 0;

  template <typename T>
  T opt(const std::string& key) {
    return options_->get<T>(key);
  }
};

class DecoderBase {
protected:
  Ptr<Config> options_;
  std::string prefix_{"decoder"};

  bool inference_{false};

public:
  template <class... Args>
  DecoderBase(Ptr<Config> options, Args... args)
      : options_(options),
        prefix_(Get(keywords::prefix, "decoder", args...)),
        inference_(Get(keywords::inference, false, args...)) {}

  virtual Ptr<DecoderState> startState(Ptr<EncoderState> encState) = 0;
  virtual Ptr<DecoderState> step(Ptr<ExpressionGraph>, Ptr<DecoderState>) = 0;

  virtual std::tuple<Expr, Expr> groundTruth(Ptr<DecoderState> state,
                                             Ptr<ExpressionGraph> graph,
                                             Ptr<data::CorpusBatch> batch,
                                             size_t index) {
    using namespace keywords;

    int dimVoc = opt<std::vector<int>>("dim-vocabs").back();
    int dimEmb = opt<int>("dim-emb");

    auto yEmbFactory = embedding(graph)
                       ("prefix", prefix_ + "_Wemb")
                       ("dimVocab", dimVoc)
                       ("dimEmb", dimEmb);

    if(options_->has("embedding-fix-trg"))
      yEmbFactory
        ("fixed", opt<bool>("embedding-fix-trg"));

    if(options_->has("embedding-vectors")) {
      auto embFiles = opt<std::vector<std::string>>("embedding-vectors");
      yEmbFactory
        ("embFile", embFiles[index])
        ("normalization", opt<bool>("embedding-normalization"));
    }

    auto yEmb = yEmbFactory.construct();

    auto subBatch = (*batch)[index];
    int dimBatch = subBatch->batchSize();
    int dimWords = subBatch->batchWidth();

    auto chosenEmbeddings = rows(yEmb, subBatch->indices());

    auto y = reshape(chosenEmbeddings, {dimBatch, opt<int>("dim-emb"), dimWords});

    auto yMask = graph->constant({dimBatch, 1, dimWords},
                                 init = inits::from_vector(subBatch->mask()));

    auto yIdx = graph->constant({(int)subBatch->indices().size(), 1},
                                init = inits::from_vector(subBatch->indices()));

    auto yShifted = shift(y, {0, 0, 1, 0});

    state->setTargetEmbeddings(yShifted);

    return std::make_tuple(yMask, yIdx);
  }

  virtual void selectEmbeddings(Ptr<ExpressionGraph> graph,
                                Ptr<DecoderState> state,
                                const std::vector<size_t>& embIdx) {
    using namespace keywords;

    int dimTrgEmb = opt<int>("dim-emb");
    int dimTrgVoc = opt<std::vector<int>>("dim-vocabs").back();

    Expr selectedEmbs;
    if(embIdx.empty()) {
      selectedEmbs = graph->constant({1, dimTrgEmb},
                                     init = inits::zeros);
    } else {
      // embeddings are loaded from model during translation, no fixing required
      auto yEmb = embedding(graph)
                  ("prefix", prefix_ + "_Wemb")
                  ("dimVocab", dimTrgVoc)
                  ("dimEmb", dimTrgEmb)
                  .construct();
      selectedEmbs = rows(yEmb, embIdx);

      selectedEmbs
          = reshape(selectedEmbs, {1, dimTrgEmb, 1, (int)embIdx.size()});
    }
    state->setTargetEmbeddings(selectedEmbs);
  }

  virtual const std::vector<Expr> getAlignments() {
    return {};
  };

  template <typename T>
  T opt(const std::string& key) {
    return options_->get<T>(key);
  }
};

class EncoderDecoderBase {
public:
  virtual void load(Ptr<ExpressionGraph>, const std::string&) = 0;

  virtual void save(Ptr<ExpressionGraph>, const std::string&) = 0;

  virtual void save(Ptr<ExpressionGraph>, const std::string&, bool) = 0;

  virtual void selectEmbeddings(Ptr<ExpressionGraph> graph,
                                Ptr<DecoderState> state,
                                const std::vector<size_t>&) = 0;

  virtual Ptr<DecoderState> step(Ptr<ExpressionGraph> graph,
                                 Ptr<DecoderState>,
                                 const std::vector<size_t>&,
                                 const std::vector<size_t>&) = 0;

  virtual Ptr<DecoderState> step(Ptr<ExpressionGraph>, Ptr<DecoderState>) = 0;

  virtual Expr build(Ptr<ExpressionGraph> graph,
                     Ptr<data::CorpusBatch> batch,
                     bool clearGraph = true) = 0;

  virtual Ptr<EncoderBase> getEncoder() = 0;
  virtual Ptr<DecoderBase> getDecoder() = 0;
};

template <class Encoder, class Decoder>
class EncoderDecoder : public EncoderDecoderBase {
protected:
  Ptr<Config> options_;
  std::string prefix_;

  Ptr<EncoderBase> encoder_;
  Ptr<DecoderBase> decoder_;

  std::vector<size_t> batchIndices_;

  bool inference_{false};

public:
  typedef data::Corpus dataset_type;

  template <class... Args>
  EncoderDecoder(Ptr<Config> options, Args... args)
      : EncoderDecoder(options, {0, 1}, args...) {}

  template <class... Args>
  EncoderDecoder(Ptr<Config> options,
                 const std::vector<size_t>& batchIndices,
                 Args... args)
      : options_(options),
        batchIndices_(batchIndices),
        prefix_(Get(keywords::prefix, "", args...)),
        encoder_(New<Encoder>(
            options, keywords::prefix = prefix_ + "encoder", args...)),
        decoder_(New<Decoder>(
            options, keywords::prefix = prefix_ + "decoder", args...)),
        inference_(Get(keywords::inference, false, args...)) {}

  Ptr<EncoderBase> getEncoder() { return encoder_; }

  Ptr<DecoderBase> getDecoder() { return decoder_; }

  virtual void load(Ptr<ExpressionGraph> graph, const std::string& name) {
    graph->load(name);
  }

  virtual void save(Ptr<ExpressionGraph> graph,
                    const std::string& name,
                    bool saveTranslatorConfig) {
    // ignore config for now
    graph->save(name);
    options_->saveModelParameters(name);
  }

  virtual void save(Ptr<ExpressionGraph> graph, const std::string& name) {
    graph->save(name);
    options_->saveModelParameters(name);
  }

  virtual void clear(Ptr<ExpressionGraph> graph) {
    graph->clear();
    encoder_ = New<Encoder>(options_,
                            keywords::prefix = prefix_ + "encoder",
                            keywords::inference = inference_);

    decoder_ = New<Decoder>(options_,
                            keywords::prefix = prefix_ + "decoder",
                            keywords::inference = inference_);
  }

  virtual Ptr<DecoderState> startState(Ptr<ExpressionGraph> graph,
                                       Ptr<data::CorpusBatch> batch) {
    return decoder_->startState(
        encoder_->build(graph, batch, batchIndices_.front()));
  }

  virtual Ptr<DecoderState> step(Ptr<ExpressionGraph> graph,
                                 Ptr<DecoderState> state) {
    return decoder_->step(graph, state);
  }

  virtual Ptr<DecoderState> step(Ptr<ExpressionGraph> graph,
                                 Ptr<DecoderState> state,
                                 const std::vector<size_t>& hypIndices,
                                 const std::vector<size_t>& embIndices) {
    auto selectedState = hypIndices.empty() ? state : state->select(hypIndices);

    selectEmbeddings(graph, selectedState, embIndices);
    selectedState->setSingleStep(true);
    auto nextState = step(graph, selectedState);
    nextState->setProbs(logsoftmax(nextState->getProbs()));
    return nextState;
  }

  virtual void selectEmbeddings(Ptr<ExpressionGraph> graph,
                                Ptr<DecoderState> state,
                                const std::vector<size_t>& embIdx) {
    return decoder_->selectEmbeddings(graph, state, embIdx);
  }

  virtual Expr build(Ptr<ExpressionGraph> graph,
                     Ptr<data::CorpusBatch> batch,
                     bool clearGraph = true) {
    using namespace keywords;

    if(clearGraph)
      clear(graph);

    auto state = startState(graph, batch);

    Expr trgMask, trgIdx;
    std::tie(trgMask, trgIdx)
        = decoder_->groundTruth(state, graph, batch, batchIndices_.back());

    auto nextState = step(graph, state);

    auto cost = CrossEntropyCost(prefix_ + "cost")
                  (nextState->getProbs(), trgIdx, mask = trgMask);

    if(options_->has("guided-alignment") && !inference_) {
      auto alignments = decoder_->getAlignments();
      UTIL_THROW_IF2(alignments.empty(), "Model does not seem to support alignments");
      auto att = concatenate(alignments, axis = 3);
      return cost + guidedAlignmentCost(graph, batch, options_, att);
    } else {
      return cost;
    }
  }

  virtual Expr build(Ptr<ExpressionGraph> graph,
                     Ptr<data::Batch> batch,
                     bool clearGraph = true) {
    auto corpusBatch = std::static_pointer_cast<data::CorpusBatch>(batch);
    return build(graph, corpusBatch, clearGraph);
  }

  Ptr<data::BatchStats> collectStats(Ptr<ExpressionGraph> graph) {
    auto stats = New<data::BatchStats>();

    size_t step = 10;
    size_t maxLength = opt<size_t>("max-length");
    size_t numFiles = opt<std::vector<std::string>>("train-sets").size();
    for(size_t i = step; i <= maxLength; i += step) {
      size_t batchSize = step;
      std::vector<size_t> lengths(numFiles, i);
      bool fits = true;
      do {
        auto batch = data::CorpusBatch::fakeBatch(
            lengths, batchSize, options_->has("guided-alignment"));
        build(graph, batch);
        fits = graph->fits();
        if(fits)
          stats->add(batch);
        batchSize += step;
      } while(fits);
    }
    return stats;
  }

  template <typename T>
  T opt(const std::string& key) {
    return options_->get<T>(key);
  }

  virtual Expr buildToScore(Ptr<ExpressionGraph> graph,
                            Ptr<data::CorpusBatch> batch,
                            bool clearGraph = true) {
    using namespace keywords;

    if(clearGraph)
      clear(graph);
    auto state = startState(graph, batch);

    Expr trgMask, trgIdx;
    std::tie(trgMask, trgIdx)
        = decoder_->groundTruth(state, graph, batch, batchIndices_.back());

    auto nextState = step(graph, state);

    return -sum(cross_entropy(nextState->getProbs(), trgIdx) * trgMask, axis=2);
  }
};
}
