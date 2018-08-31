#include "encoder_decoder.h"
#include "common/cli_helper.h"

namespace marian {

EncoderDecoder::EncoderDecoder(Ptr<Options> options)
    : options_(options),
      prefix_(options->get<std::string>("prefix", "")),
      inference_(options->get<bool>("inference", false)) {
  modelFeatures_ = {"type",
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
                    "tied-embeddings-all"};

  modelFeatures_.insert("transformer-heads");
  modelFeatures_.insert("transformer-no-projection");
  modelFeatures_.insert("transformer-dim-ffn");
  modelFeatures_.insert("transformer-ffn-depth");
  modelFeatures_.insert("transformer-ffn-activation");
  modelFeatures_.insert("transformer-dim-aan");
  modelFeatures_.insert("transformer-aan-depth");
  modelFeatures_.insert("transformer-aan-activation");
  modelFeatures_.insert("transformer-aan-nogate");
  modelFeatures_.insert("transformer-preprocess");
  modelFeatures_.insert("transformer-postprocess");
  modelFeatures_.insert("transformer-postprocess-emb");
  modelFeatures_.insert("transformer-decoder-autoreg");
  modelFeatures_.insert("transformer-tied-layers");
  modelFeatures_.insert("transformer-guided-alignment-layer");
}

std::vector<Ptr<EncoderBase>>& EncoderDecoder::getEncoders() {
  return encoders_;
}

void EncoderDecoder::push_back(Ptr<EncoderBase> encoder) {
  encoders_.push_back(encoder);
}

std::vector<Ptr<DecoderBase>>& EncoderDecoder::getDecoders() {
  return decoders_;
}

void EncoderDecoder::push_back(Ptr<DecoderBase> decoder) {
  decoders_.push_back(decoder);
}

void EncoderDecoder::createDecoderConfig(const std::string& name) {
  Config::YamlNode decoder;
  decoder["models"] = std::vector<std::string>({name});
  decoder["vocabs"] = options_->get<std::vector<std::string>>("vocabs");
  decoder["beam-size"] = opt<size_t>("beam-size");
  decoder["normalize"] = opt<float>("normalize");
  decoder["word-penalty"] = opt<float>("word-penalty");

  decoder["mini-batch"] = opt<size_t>("valid-mini-batch");
  decoder["maxi-batch"] = opt<size_t>("valid-mini-batch") > 1 ? 100 : 1;
  decoder["maxi-batch-sort"]
      = opt<size_t>("valid-mini-batch") > 1 ? "src" : "none";

  decoder["relative-paths"] = false;

  OutputFileStream out(name + ".decoder.yml");
  (std::ostream&)out << decoder;
}

Config::YamlNode EncoderDecoder::getModelParameters() {
  Config::YamlNode modelParams;
  for(auto& key : modelFeatures_)
    modelParams[key] = options_->getOptions()[key];

  if(options_->has("original-type"))
    modelParams["type"] = options_->getOptions()["original-type"];

  modelParams["version"] = PROJECT_VERSION_FULL;
  return modelParams;
}

std::string EncoderDecoder::getModelParametersAsString() {
  auto yaml = getModelParameters();
  YAML::Emitter out;
  cli::OutputYaml(yaml, out);
  return std::string(out.c_str());
}

void EncoderDecoder::load(Ptr<ExpressionGraph> graph,
                          const std::string& name,
                          bool markedReloaded) {
  graph->load(name, markedReloaded && !opt<bool>("ignore-model-config", false));
}

void EncoderDecoder::mmap(Ptr<ExpressionGraph> graph,
                          const void* ptr,
                          bool markedReloaded) {
  graph->mmap(ptr, markedReloaded && !opt<bool>("ignore-model-config", false));
}

void EncoderDecoder::save(Ptr<ExpressionGraph> graph,
                          const std::string& name,
                          bool saveTranslatorConfig) {
  // ignore config for now
  LOG(info, "Saving model weights and runtime parameters to {}", name);

  graph->save(name, getModelParametersAsString());

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

  // initialize shortlist here
  if(shortlistGenerator_) {
    auto shortlist = shortlistGenerator_->generate(batch);
    decoders_[0]->setShortlist(shortlist);
  }

  return decoders_[0]->startState(graph, batch, encoderStates);
}

Ptr<DecoderState> EncoderDecoder::step(Ptr<ExpressionGraph> graph,
                                       Ptr<DecoderState> state,
                                       const std::vector<size_t>& hypIndices, // [beamIndex * activeBatchSize + batchIndex]
                                       const std::vector<size_t>& embIndices, // [beamIndex * activeBatchSize + batchIndex]
                                       int dimBatch,
                                       int beamSize) {
  // create updated state that reflects reordering and dropping of hypotheses
  state = hypIndices.empty() ? state : state->select(hypIndices, beamSize);

  // Fill stte with embeddings based on last prediction
  decoders_[0]->embeddingsFromPrediction(
      graph, state, embIndices, dimBatch, beamSize);
  auto nextState = decoders_[0]->step(graph, state);

  return nextState;
}

Ptr<DecoderState> EncoderDecoder::stepAll(Ptr<ExpressionGraph> graph,
                                          Ptr<data::CorpusBatch> batch,
                                          bool clearGraph) {
  if(clearGraph)
    clear(graph);

  // Required first step, also intializes shortlist
  auto state = startState(graph, batch);

  // Fill state with embeddings from batch (ground truth)
  decoders_[0]->embeddingsFromBatch(graph, state, batch);
  auto nextState = decoders_[0]->step(graph, state);
  nextState->setTargetMask(state->getTargetMask());
  nextState->setTargetIndices(state->getTargetIndices());

  return nextState;
}

Expr EncoderDecoder::build(Ptr<ExpressionGraph> graph,
                           Ptr<data::CorpusBatch> batch,
                           bool clearGraph) {
  auto state = stepAll(graph, batch, clearGraph);

  // returns raw logits
  return state->getLogProbs();
}

Expr EncoderDecoder::build(Ptr<ExpressionGraph> graph,
                           Ptr<data::Batch> batch,
                           bool clearGraph) {
  auto corpusBatch = std::static_pointer_cast<data::CorpusBatch>(batch);
  return build(graph, corpusBatch, clearGraph);
}

}  // namespace marian
