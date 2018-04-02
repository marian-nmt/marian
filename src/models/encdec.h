#pragma once

#include "marian.h"

#include "layers/generic.h"
#include "layers/guided_alignment.h"
#include "model_base.h"
#include "states.h"
#include "encoder.h"
#include "decoder.h"

namespace marian {

class EncoderDecoderBase : public models::ModelBase {
public:
  virtual void selectEmbeddings(Ptr<ExpressionGraph> graph,
                                Ptr<DecoderState> state,
                                const std::vector<size_t>&,
                                int dimBatch,
                                int beamSize)
      = 0;

  virtual Ptr<DecoderState> step(Ptr<ExpressionGraph> graph,
                                 Ptr<DecoderState>,
                                 const std::vector<size_t>&,
                                 const std::vector<size_t>&,
                                 int dimBatch,
                                 int beamSize)
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

  void saveModelParameters(const std::string& name);

  virtual void createDecoderConfig(const std::string& name);

public:
  typedef data::Corpus dataset_type;

  EncoderDecoder(Ptr<Options> options);

  std::vector<Ptr<EncoderBase>>& getEncoders() { return encoders_; }

  void push_back(Ptr<EncoderBase> encoder) { encoders_.push_back(encoder); }

  std::vector<Ptr<DecoderBase>>& getDecoders() { return decoders_; }

  void push_back(Ptr<DecoderBase> decoder) { decoders_.push_back(decoder); }

  virtual void load(Ptr<ExpressionGraph> graph,
                    const std::string& name,
                    bool markedReloaded = true);

  virtual void save(Ptr<ExpressionGraph> graph,
                    const std::string& name,
                    bool saveTranslatorConfig = false);

  virtual void clear(Ptr<ExpressionGraph> graph);

  virtual Ptr<DecoderState> startState(Ptr<ExpressionGraph> graph,
                                       Ptr<data::CorpusBatch> batch);

  virtual Ptr<DecoderState> step(Ptr<ExpressionGraph> graph,
                                 Ptr<DecoderState> state);

  virtual Ptr<DecoderState> step(Ptr<ExpressionGraph> graph,
                                 Ptr<DecoderState> state,
                                 const std::vector<size_t>& hypIndices,
                                 const std::vector<size_t>& embIndices,
                                 int dimBatch,
                                 int beamSize);

  virtual void selectEmbeddings(Ptr<ExpressionGraph> graph,
                                Ptr<DecoderState> state,
                                const std::vector<size_t>& embIdx,
                                int dimBatch,
                                int beamSize);

  virtual Expr build(Ptr<ExpressionGraph> graph,
                     Ptr<data::CorpusBatch> batch,
                     bool clearGraph = true);

  virtual Expr build(Ptr<ExpressionGraph> graph,
                     Ptr<data::Batch> batch,
                     bool clearGraph = true);

  Ptr<data::BatchStats> collectStats(Ptr<ExpressionGraph> graph,
                                     size_t multiplier = 1);

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
