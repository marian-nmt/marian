#pragma once

#include "marian.h"

namespace marian {

class EncoderBase {
protected:
  Ptr<Options> options_;
  std::string prefix_{"encoder"};
  bool inference_{false};
  size_t batchIndex_{0};

  virtual std::tuple<Expr, Expr> lookup(Expr srcEmbeddings,
                                        Ptr<data::CorpusBatch> batch) {
    using namespace keywords;

    auto subBatch = (*batch)[batchIndex_];

    int dimBatch = subBatch->batchSize();
    int dimEmb = srcEmbeddings->shape()[1];
    int dimWords = subBatch->batchWidth();

    auto graph = srcEmbeddings->graph();
    auto chosenEmbeddings = rows(srcEmbeddings, subBatch->indices());

    auto batchEmbeddings
        = reshape(chosenEmbeddings, {dimBatch, dimEmb, dimWords});
    auto batchMask = graph->constant(
        {dimBatch, 1, dimWords}, init = inits::from_vector(subBatch->mask()));

    return std::make_tuple(batchEmbeddings, batchMask);
  }

public:
  EncoderBase(Ptr<Options> options)
      : options_(options),
        prefix_(options->get<std::string>("prefix", "encoder")),
        inference_(options->get<bool>("inference", false)),
        batchIndex_(options->get<size_t>("index", 0)) {}

  virtual Ptr<EncoderState> build(Ptr<ExpressionGraph>, Ptr<data::CorpusBatch>)
      = 0;

  template <typename T>
  T opt(const std::string& key) {
    return options_->get<T>(key);
  }

  virtual void clear() = 0;
};

class DecoderBase {
protected:
  Ptr<Options> options_;
  std::string prefix_{"decoder"};
  bool inference_{false};
  size_t batchIndex_{1};

public:
  DecoderBase(Ptr<Options> options)
      : options_(options),
        prefix_(options->get<std::string>("prefix", "decoder")),
        inference_(options->get<bool>("inference", false)),
        batchIndex_(options->get<size_t>("index", 1)) {}

  virtual Ptr<DecoderState> startState(Ptr<ExpressionGraph>,
                                       Ptr<data::CorpusBatch> batch,
                                       std::vector<Ptr<EncoderState>>&)
      = 0;

  virtual Ptr<DecoderState> step(Ptr<ExpressionGraph>, Ptr<DecoderState>) = 0;

  virtual std::tuple<Expr, Expr> groundTruth(Ptr<DecoderState> state,
                                             Ptr<ExpressionGraph> graph,
                                             Ptr<data::CorpusBatch> batch) {
    using namespace keywords;

    int dimVoc = opt<std::vector<int>>("dim-vocabs")[batchIndex_];
    int dimEmb = opt<int>("dim-emb");

    auto yEmbFactory = embedding(graph)  //
        ("dimVocab", dimVoc)             //
        ("dimEmb", dimEmb);

    if(opt<bool>("tied-embeddings-src") || opt<bool>("tied-embeddings-all"))
      yEmbFactory("prefix", "Wemb");
    else
      yEmbFactory("prefix", prefix_ + "_Wemb");

    if(options_->has("embedding-fix-trg"))
      yEmbFactory("fixed", opt<bool>("embedding-fix-trg"));

    if(options_->has("embedding-vectors")) {
      auto embFiles = opt<std::vector<std::string>>("embedding-vectors");
      yEmbFactory("embFile", embFiles[batchIndex_])  //
          ("normalization", opt<bool>("embedding-normalization"));
    }

    auto yEmb = yEmbFactory.construct();

    auto subBatch = (*batch)[batchIndex_];
    int dimBatch = subBatch->batchSize();
    int dimWords = subBatch->batchWidth();

    auto chosenEmbeddings = rows(yEmb, subBatch->indices());

    auto y
        = reshape(chosenEmbeddings, {dimBatch, opt<int>("dim-emb"), dimWords});

    auto yMask = graph->constant({dimBatch, 1, dimWords},
                                 init = inits::from_vector(subBatch->mask()));

    auto yIdx = graph->constant({(int)subBatch->indices().size(), 1},
                                init = inits::from_vector(subBatch->indices()));

    auto yShifted = shift(y, {0, 0, 1, 0});

    state->setTargetEmbeddings(yShifted);
    state->setTargetMask(yMask);

    return std::make_tuple(yMask, yIdx);
  }

  virtual void selectEmbeddings(Ptr<ExpressionGraph> graph,
                                Ptr<DecoderState> state,
                                const std::vector<size_t>& embIdx) {
    using namespace keywords;

    int dimTrgEmb = opt<int>("dim-emb");
    int dimTrgVoc = opt<std::vector<int>>("dim-vocabs")[batchIndex_];

    Expr selectedEmbs;
    if(embIdx.empty()) {
      selectedEmbs = graph->constant({1, dimTrgEmb}, init = inits::zeros);
    } else {
      // embeddings are loaded from model during translation, no fixing required
      auto yEmbFactory = embedding(graph)  //
          ("dimVocab", dimTrgVoc)          //
          ("dimEmb", dimTrgEmb);

      if(opt<bool>("tied-embeddings-src") || opt<bool>("tied-embeddings-all"))
        yEmbFactory("prefix", "Wemb");
      else
        yEmbFactory("prefix", prefix_ + "_Wemb");

      auto yEmb = yEmbFactory.construct();
      selectedEmbs = rows(yEmb, embIdx);

      selectedEmbs
          = reshape(selectedEmbs, {1, dimTrgEmb, 1, (int)embIdx.size()});
    }
    state->setTargetEmbeddings(selectedEmbs);
  }

  virtual const std::vector<Expr> getAlignments(int i = 0) { return {}; };

  template <typename T>
  T opt(const std::string& key) {
    return options_->get<T>(key);
  }

  virtual void clear() = 0;
};

class EncoderDecoderBase : public models::ModelBase {
public:
  virtual void selectEmbeddings(Ptr<ExpressionGraph> graph,
                                Ptr<DecoderState> state,
                                const std::vector<size_t>&)
      = 0;

  virtual Ptr<DecoderState> step(Ptr<ExpressionGraph> graph,
                                 Ptr<DecoderState>,
                                 const std::vector<size_t>&,
                                 const std::vector<size_t>&)
      = 0;

  virtual Ptr<DecoderState> step(Ptr<ExpressionGraph>, Ptr<DecoderState>) = 0;

  virtual std::vector<Ptr<EncoderBase>>& getEncoders() = 0;
  virtual std::vector<Ptr<DecoderBase>>& getDecoders() = 0;
};

class EncoderDecoder : public EncoderDecoderBase {
protected:
  Ptr<Options> options_;
  std::string prefix_;

  std::vector<Ptr<EncoderBase>> encoders_;
  std::vector<Ptr<DecoderBase>> decoders_;

  bool inference_{false};

  std::vector<std::string> modelFeatures_;

  void saveModelParameters(const std::string& name) {
    YAML::Node modelParams;
    for(auto& key : modelFeatures_)
      modelParams[key] = options_->getOptions()[key];

    if(options_->has("original-type"))
      modelParams["type"] = options_->getOptions()["original-type"];

    Config::AddYamlToNpz(modelParams, "special:model.yml", name);
  }

public:
  typedef data::Corpus dataset_type;

  EncoderDecoder(Ptr<Options> options)
      : options_(options),
        prefix_(options->get<std::string>("prefix", "")),
        inference_(options->get<bool>("inference", false)) {
    modelFeatures_ = {
        "type",
        "dim-vocabs",
        "dim-emb",
        "dim-rnn",
        "enc-cell",
        "enc-type",
        "enc-cell-depth",
        "enc-depth",
        "dec-depth",
        "dec-cell",
        "dec-cell-base-depth",
        "dec-cell-high-depth",
        "skip",
        "layer-normalization",
        "special-vocab",
        "tied-embeddings",
        "tied-embeddings-src",
        "tied-embeddings-all",
    };

    modelFeatures_.push_back("transformer-heads");
    modelFeatures_.push_back("transformer-dim-ffn");
    modelFeatures_.push_back("transformer-preprocess");
    modelFeatures_.push_back("transformer-postprocess");
    modelFeatures_.push_back("transformer-postprocess-emb");
  }

  std::vector<Ptr<EncoderBase>>& getEncoders() { return encoders_; }

  void push_back(Ptr<EncoderBase> encoder) { encoders_.push_back(encoder); }

  std::vector<Ptr<DecoderBase>>& getDecoders() { return decoders_; }

  void push_back(Ptr<DecoderBase> decoder) { decoders_.push_back(decoder); }

  virtual void load(Ptr<ExpressionGraph> graph, const std::string& name) {
    graph->load(name);
  }

  virtual void save(Ptr<ExpressionGraph> graph,
                    const std::string& name,
                    bool saveTranslatorConfig) {
    // ignore config for now
    graph->save(name);
    saveModelParameters(name);
  }

  virtual void save(Ptr<ExpressionGraph> graph, const std::string& name) {
    graph->save(name);
    saveModelParameters(name);
  }

  virtual void clear(Ptr<ExpressionGraph> graph) {
    graph->clear();

    for(auto& enc : encoders_)
      enc->clear();
    for(auto& dec : decoders_)
      dec->clear();
  }

  virtual Ptr<DecoderState> startState(Ptr<ExpressionGraph> graph,
                                       Ptr<data::CorpusBatch> batch) {
    std::vector<Ptr<EncoderState>> encoderStates;
    for(auto& encoder : encoders_)
      encoderStates.push_back(encoder->build(graph, batch));
    return decoders_[0]->startState(graph, batch, encoderStates);
  }

  virtual Ptr<DecoderState> step(Ptr<ExpressionGraph> graph,
                                 Ptr<DecoderState> state) {
    return decoders_[0]->step(graph, state);
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
    decoders_[0]->selectEmbeddings(graph, state, embIdx);
  }

  virtual Expr build(Ptr<ExpressionGraph> graph,
                     Ptr<data::CorpusBatch> batch,
                     bool clearGraph = true) {
    using namespace keywords;

    if(clearGraph)
      clear(graph);

    auto state = startState(graph, batch);

    Expr trgMask, trgIdx;
    std::tie(trgMask, trgIdx) = decoders_[0]->groundTruth(state, graph, batch);

    auto nextState = step(graph, state);

    std::string costType = opt<std::string>("cost-type");
    float ls = inference_ ? 0.f : opt<float>("label-smoothing");

    auto cost = Cost(nextState->getProbs(), trgIdx, trgMask, costType, ls);

    if(options_->has("guided-alignment") && !inference_) {
      auto alignments = decoders_[0]->getAlignments();
      UTIL_THROW_IF2(alignments.empty(),
                     "Model does not seem to support alignments");
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

  Ptr<data::BatchStats> collectStats(Ptr<ExpressionGraph> graph,
                                     size_t multiplier = 1) {
    auto stats = New<data::BatchStats>();

    size_t step = 10;
    size_t maxLength = opt<size_t>("max-length");

    maxLength = std::ceil(maxLength / (float)step) * step;

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
          stats->add(batch, multiplier);
        batchSize += step;
      } while(fits);
    }
    return stats;
  }

  template <typename T>
  T opt(const std::string& key) {
    return options_->get<T>(key);
  }

  template <typename T>
  void set(std::string key, T value) {
    options_->set(key, value);
  }
};
}
