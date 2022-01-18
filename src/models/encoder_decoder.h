#pragma once

#include "marian.h"

#include "models/decoder.h"
#include "models/encoder.h"
#include "models/model_base.h"
#include "models/states.h"

namespace marian {

class IEncoderDecoder : public models::IModel {
public:
  virtual ~IEncoderDecoder() {}

  virtual void load(Ptr<ExpressionGraph> graph,
                    const std::vector<io::Item>& items,
                    bool markedReloaded = true) = 0;

  virtual void load(Ptr<ExpressionGraph> graph,
                    const std::string& name,
                    bool markedReloaded = true) override
      = 0;

  virtual void mmap(Ptr<ExpressionGraph> graph,
                    const void* ptr,
                    bool markedReloaded = true)
      = 0;

  virtual void save(Ptr<ExpressionGraph> graph,
                    const std::string& name,
                    bool saveTranslatorConfig = false) override
      = 0;

  virtual void clear(Ptr<ExpressionGraph> graph) override = 0;

  virtual Logits build(Ptr<ExpressionGraph> graph,
                       Ptr<data::Batch> batch,
                       bool clearGraph = true) override = 0;

  virtual Logits build(Ptr<ExpressionGraph> graph,
                       Ptr<data::CorpusBatch> batch,
                       bool clearGraph = true) = 0;

  virtual Ptr<DecoderState> startState(Ptr<ExpressionGraph> graph,
                                       Ptr<data::CorpusBatch> batch) = 0;

  virtual Ptr<DecoderState> step(Ptr<ExpressionGraph> graph,
                                 Ptr<DecoderState> state,
                                 const std::vector<IndexType>& hypIndices,   // [beamIndex * activeBatchSize + batchIndex]
                                 const Words& words,                         // [beamIndex * activeBatchSize + batchIndex]
                                 const std::vector<IndexType>& batchIndices, // [batchIndex]
                                 int beamSize)
      = 0;

  virtual Ptr<Options> getOptions() = 0;

  virtual void setShortlistGenerator(
      Ptr<const data::ShortlistGenerator> shortlistGenerator)
      = 0;

  virtual Ptr<data::Shortlist> getShortlist() = 0;

  virtual data::SoftAlignment getAlignment() = 0;
};

class EncoderDecoder : public IEncoderDecoder, public LayerBase {
protected:
  Ptr<const data::ShortlistGenerator> shortlistGenerator_;

  const std::string prefix_;
  const bool inference_{ false };

  std::vector<Ptr<EncoderBase>> encoders_;
  std::vector<Ptr<DecoderBase>> decoders_;

  std::set<std::string> modelFeatures_;

  Config::YamlNode getModelParameters();
  std::string getModelParametersAsString();

  virtual void createDecoderConfig(const std::string& name);

public:
  typedef data::Corpus dataset_type;

  EncoderDecoder(Ptr<ExpressionGraph> graph, Ptr<Options> options);

  virtual Ptr<Options> getOptions() override { return options_; }

  std::vector<Ptr<EncoderBase>>& getEncoders();

  void push_back(Ptr<EncoderBase> encoder);

  std::vector<Ptr<DecoderBase>>& getDecoders();

  void push_back(Ptr<DecoderBase> decoder);

  virtual void load(Ptr<ExpressionGraph> graph,
                    const std::vector<io::Item>& items,
                    bool markedReloaded = true) override;

  virtual void load(Ptr<ExpressionGraph> graph,
                    const std::string& name,
                    bool markedReloaded = true) override;

  virtual void mmap(Ptr<ExpressionGraph> graph,
                    const void* ptr,
                    bool markedReloaded = true) override;

  virtual void save(Ptr<ExpressionGraph> graph,
                    const std::string& name,
                    bool saveTranslatorConfig = false) override;

  virtual void clear(Ptr<ExpressionGraph> graph) override;

  template <typename T>
  T opt(const std::string& key) {
    return options_->get<T>(key);
  }

  template <typename T>
  T opt(const std::string& key, const T& def) {
    return options_->get<T>(key, def);
  }

  template <typename T>
  void set(std::string key, T value) {
    options_->set(key, value);
  }

  virtual void setShortlistGenerator(
      Ptr<const data::ShortlistGenerator> shortlistGenerator) override {
    shortlistGenerator_ = shortlistGenerator;
  };

  virtual Ptr<data::Shortlist> getShortlist() override {
    return decoders_[0]->getShortlist();
  };

  // convert alignment tensors that live GPU-side into a CPU-side vector of vectors
  virtual data::SoftAlignment getAlignment() override {
    data::SoftAlignment softAlignments;
    auto alignments = decoders_[0]->getAlignments(); // [tgt index][beam depth, max src length, batch size, 1]
    for(auto alignment : alignments) { // [beam depth, max src length, batch size, 1]
      softAlignments.push_back({});
      alignment->val()->get(softAlignments.back());
    }
    return softAlignments; // [tgt index][beam depth * max src length * batch size]
  };

  /*********************************************************************/

  virtual Ptr<DecoderState> startState(Ptr<ExpressionGraph> graph,
                                       Ptr<data::CorpusBatch> batch) override;

  virtual Ptr<DecoderState> step(Ptr<ExpressionGraph> graph,
                                 Ptr<DecoderState> state,
                                 const std::vector<IndexType>& hypIndices,
                                 const Words& words,
                                 const std::vector<IndexType>& batchIndices,
                                 int beamSize) override;

  virtual Ptr<DecoderState> stepAll(Ptr<ExpressionGraph> graph,
                                    Ptr<data::CorpusBatch> batch,
                                    bool clearGraph = true);

  virtual Logits build(Ptr<ExpressionGraph> graph,
                       Ptr<data::CorpusBatch> batch,
                       bool clearGraph = true) override;

  virtual Logits build(Ptr<ExpressionGraph> graph,
                       Ptr<data::Batch> batch,
                       bool clearGraph = true) override;
};

}  // namespace marian
