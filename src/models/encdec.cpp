#include "encdec.h"

namespace marian {

void EncoderDecoder::saveModelParameters(const std::string& name) {
  Config::YamlNode modelParams;
  for(auto& key : modelFeatures_)
    modelParams[key] = options_->getOptions()[key];

  if(options_->has("original-type"))
    modelParams["type"] = options_->getOptions()["original-type"];

  modelParams["version"] = PROJECT_VERSION_FULL;

  Config::AddYamlToNpz(modelParams, "special:model.yml", name);
}

void EncoderDecoder::createDecoderConfig(const std::string& name) {
  Config::YamlNode decoder;
  decoder["models"] = std::vector<std::string>({name});
  decoder["vocabs"] = options_->get<std::vector<std::string>>("vocabs");
  decoder["normalize"] = opt<float>("normalize");
  decoder["beam-size"] = opt<size_t>("beam-size");

  decoder["mini-batch"] = opt<size_t>("valid-mini-batch");
  decoder["maxi-batch"] = opt<size_t>("valid-mini-batch") > 1 ? 100 : 1;
  decoder["maxi-batch-sort"]
      = opt<size_t>("valid-mini-batch") > 1 ? "trg" : "none";

  decoder["relative-paths"] = false;

  OutputFileStream out(name + ".decoder.yml");
  (std::ostream&)out << decoder;
}

EncoderDecoder::EncoderDecoder(Ptr<Options> options)
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
      "right-left",
      "special-vocab",
      "tied-embeddings",
      "tied-embeddings-src",
      "tied-embeddings-all",
  };

  modelFeatures_.push_back("transformer-heads");
  modelFeatures_.push_back("transformer-dim-ffn");
  modelFeatures_.push_back("transformer-ffn-activation");
  modelFeatures_.push_back("transformer-preprocess");
  modelFeatures_.push_back("transformer-postprocess");
  modelFeatures_.push_back("transformer-postprocess-emb");
}

void EncoderDecoder::load(Ptr<ExpressionGraph> graph,
                  const std::string& name,
                  bool markedReloaded) {
  graph->load(name, markedReloaded && !opt<bool>("ignore-model-config"));
}

void EncoderDecoder::save(Ptr<ExpressionGraph> graph,
                  const std::string& name,
                  bool saveTranslatorConfig) {
  // ignore config for now
  graph->save(name);
  saveModelParameters(name);

  if(saveTranslatorConfig)
    createDecoderConfig(name);
}

void EncoderDecoder::clear(Ptr<ExpressionGraph> graph) {
  graph->clear();

  for(auto& enc : encoders_)
    enc->clear();
  for(auto& dec : decoders_)
    dec->clear();
}

Ptr<DecoderState> EncoderDecoder::startState(Ptr<ExpressionGraph> graph,
                                     Ptr<data::CorpusBatch> batch) {
  std::vector<Ptr<EncoderState>> encoderStates;
  for(auto& encoder : encoders_)
    encoderStates.push_back(encoder->build(graph, batch));
  return decoders_[0]->startState(graph, batch, encoderStates);
}

Ptr<DecoderState> EncoderDecoder::step(Ptr<ExpressionGraph> graph,
                                       Ptr<DecoderState> state) {
  return decoders_[0]->step(graph, state);
}

Ptr<DecoderState> EncoderDecoder::step(Ptr<ExpressionGraph> graph,
                               Ptr<DecoderState> state,
                               const std::vector<size_t>& hypIndices,
                               const std::vector<size_t>& embIndices,
                               int dimBatch,
                               int beamSize) {
  auto selectedState
      = hypIndices.empty() ? state : state->select(hypIndices, beamSize);
  selectEmbeddings(graph, selectedState, embIndices, dimBatch, beamSize);
  selectedState->setSingleStep(true);
  auto nextState = step(graph, selectedState);
  nextState->setProbs(logsoftmax(nextState->getProbs()));
  return nextState;
}

void EncoderDecoder::selectEmbeddings(Ptr<ExpressionGraph> graph,
                              Ptr<DecoderState> state,
                              const std::vector<size_t>& embIdx,
                              int dimBatch,
                              int beamSize) {
  decoders_[0]->selectEmbeddings(graph, state, embIdx, dimBatch, beamSize);
}

Expr EncoderDecoder::build(Ptr<ExpressionGraph> graph,
                   Ptr<data::CorpusBatch> batch,
                   bool clearGraph) {
  using namespace keywords;

  if(clearGraph)
    clear(graph);

  auto state = startState(graph, batch);

  Expr trgMask, trgIdx;
  std::tie(trgMask, trgIdx) = decoders_[0]->groundTruth(state, graph, batch);

  auto nextState = step(graph, state);

  std::string costType = opt<std::string>("cost-type");
  float ls = inference_ ? 0.f : opt<float>("label-smoothing");

  Expr weights;
  bool sentenceWeighting = false;

  if(options_->has("data-weighting") && !inference_) {
    ABORT_IF(batch->getDataWeights().empty(),
             "Vector of weights is unexpectedly empty!");

    sentenceWeighting
        = options_->get<std::string>("data-weighting-type") == "sentence";
    int dimBatch = batch->size();
    int dimWords = sentenceWeighting ? 1 : batch->back()->batchWidth();

    weights = graph->constant({1, dimWords, dimBatch, 1},
                              inits::from_vector(batch->getDataWeights()));
  }

  auto cost
      = Cost(nextState->getProbs(), trgIdx, trgMask, costType, ls, weights);

  if(options_->has("guided-alignment") && !inference_) {
    auto alignments = decoders_[0]->getAlignments();
    ABORT_IF(alignments.empty(), "Model does not seem to support alignments");

    auto att = concatenate(alignments, axis = 3);
    return cost + guidedAlignmentCost(graph, batch, options_, att);
  } else {
    return cost;
  }
}

Expr EncoderDecoder::build(Ptr<ExpressionGraph> graph,
                   Ptr<data::Batch> batch,
                   bool clearGraph) {
  auto corpusBatch = std::static_pointer_cast<data::CorpusBatch>(batch);
  return build(graph, corpusBatch, clearGraph);
}

Ptr<data::BatchStats> EncoderDecoder::collectStats(Ptr<ExpressionGraph> graph,
                                   size_t multiplier) {
  auto stats = New<data::BatchStats>();

  size_t numFiles = opt<std::vector<std::string>>("train-sets").size();

  size_t first = opt<size_t>("mini-batch-fit-step");
  size_t step = opt<size_t>("mini-batch-fit-step");

  size_t maxLength = opt<size_t>("max-length");
  maxLength = std::ceil(maxLength / (float)step) * step;

  size_t maxBatch = 512;
  bool fits = true;
  while(fits) {
    std::vector<size_t> lengths(numFiles, first);
    auto batch = data::CorpusBatch::fakeBatch(lengths, maxBatch, options_);
    build(graph, batch);
    fits = graph->fits();
    if(fits)
      maxBatch *= 2;
  }

  for(size_t i = step; i <= maxLength; i += step) {
    size_t start = 1;
    size_t end = maxBatch;

    std::vector<size_t> lengths(numFiles, i);
    bool fits = true;

    do {
      size_t current = (start + end) / 2;
      // std::cerr << i << " " << current << std::endl;
      auto batch = data::CorpusBatch::fakeBatch(lengths, current, options_);
      build(graph, batch);
      fits = graph->fits();

      if(fits) {
        stats->add(batch, multiplier);
        start = current + 1;
      } else {
        end = current - 1;
      }
    } while(end - start > step);

    maxBatch = start;
  }
  return stats;
}

}
