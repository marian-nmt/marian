#pragma once

#include "marian.h"

#include "model_base.h"
#include "states.h"
#include "encoder.h"
#include "decoder.h"

namespace marian {

class EncoderDecoderBase : public models::ModelBase {
public:
  virtual void load(Ptr<ExpressionGraph> graph,
                    const std::string& name,
                    bool markedReloaded = true) = 0;

  virtual void save(Ptr<ExpressionGraph> graph,
                    const std::string& name,
                    bool saveTranslatorConfig = false) = 0;

  virtual void clear(Ptr<ExpressionGraph> graph) = 0;

  virtual Expr build(Ptr<ExpressionGraph> graph,
                     Ptr<data::Batch> batch,
                     bool clearGraph = true) = 0;

  virtual Ptr<DecoderState> startState(Ptr<ExpressionGraph> graph,
                                       Ptr<data::CorpusBatch> batch) = 0;

  virtual Ptr<DecoderState> step(Ptr<ExpressionGraph> graph,
                                 Ptr<DecoderState> state,
                                 const std::vector<size_t>& hypIndices,
                                 const std::vector<size_t>& embIndices,
                                 int dimBatch,
                                 int beamSize) = 0;

  virtual Expr build(Ptr<ExpressionGraph> graph,
                     Ptr<data::CorpusBatch> batch,
                     bool clearGraph = true) = 0;

  virtual Ptr<Options> getOptions() = 0;

  virtual void setShortlistGenerator(Ptr<data::ShortlistGenerator> shortlistGenerator) = 0;

  virtual Ptr<data::Shortlist> getShortlist() = 0;
};

class EncoderDecoder : public EncoderDecoderBase {
protected:
  Ptr<Options> options_;
  Ptr<data::ShortlistGenerator> shortlistGenerator_;

  std::string prefix_;

  std::vector<Ptr<EncoderBase>> encoders_;
  std::vector<Ptr<DecoderBase>> decoders_;

  bool inference_{false};

  std::set<std::string> modelFeatures_;

  Config::YamlNode getModelParameters();
  void saveModelParameters(const std::string& name);

  virtual void createDecoderConfig(const std::string& name);

public:
  typedef data::Corpus dataset_type;

  EncoderDecoder(Ptr<Options> options);

  virtual Ptr<Options> getOptions() { return options_; }

  std::vector<Ptr<EncoderBase>>& getEncoders();

  void push_back(Ptr<EncoderBase> encoder);

  std::vector<Ptr<DecoderBase>>& getDecoders();

  void push_back(Ptr<DecoderBase> decoder);

  virtual void load(Ptr<ExpressionGraph> graph,
                    const std::string& name,
                    bool markedReloaded = true);

  virtual void save(Ptr<ExpressionGraph> graph,
                    const std::string& name,
                    bool saveTranslatorConfig = false);

  virtual void clear(Ptr<ExpressionGraph> graph);

  template <typename T>
  T opt(const std::string& key) {
    return options_->get<T>(key);
  }

  template <typename T>
  void set(std::string key, T value) {
    options_->set(key, value);
  }

  virtual void setShortlistGenerator(Ptr<data::ShortlistGenerator> shortlistGenerator) {
    shortlistGenerator_ = shortlistGenerator;
  };

  virtual Ptr<data::Shortlist> getShortlist() {
    return decoders_[0]->getShortlist();
  };

  /*********************************************************************/

  virtual Ptr<DecoderState> startState(Ptr<ExpressionGraph> graph,
                                       Ptr<data::CorpusBatch> batch);

  virtual Ptr<DecoderState> step(Ptr<ExpressionGraph> graph,
                                 Ptr<DecoderState> state,
                                 const std::vector<size_t>& hypIndices,
                                 const std::vector<size_t>& embIndices,
                                 int dimBatch,
                                 int beamSize);

  virtual Ptr<DecoderState> stepAll(Ptr<ExpressionGraph> graph,
                                    Ptr<data::CorpusBatch> batch,
                                    bool clearGraph = true);

  virtual Expr build(Ptr<ExpressionGraph> graph,
                     Ptr<data::CorpusBatch> batch,
                     bool clearGraph = true);

  virtual Expr build(Ptr<ExpressionGraph> graph,
                     Ptr<data::Batch> batch,
                     bool clearGraph = true);

};


}
