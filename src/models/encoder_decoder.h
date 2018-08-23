#pragma once

#include "marian.h"

#include "decoder.h"
#include "encoder.h"
#include "model_base.h"
#include "states.h"

namespace marian {

class EncoderDecoderBase : public models::ModelBase {
public:
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

  virtual Expr build(Ptr<ExpressionGraph> graph,
                     Ptr<data::Batch> batch,
                     bool clearGraph = true) override
      = 0;

  virtual Ptr<DecoderState> startState(Ptr<ExpressionGraph> graph,
                                       Ptr<data::CorpusBatch> batch)
      = 0;

  virtual Ptr<DecoderState> step(Ptr<ExpressionGraph> graph,
                                 Ptr<DecoderState> state,
                                 const std::vector<size_t>& hypIndices,
                                 const std::vector<size_t>& embIndices,
                                 int dimBatch,
                                 int beamSize)
      = 0;

  virtual Expr build(Ptr<ExpressionGraph> graph,
                     Ptr<data::CorpusBatch> batch,
                     bool clearGraph = true)
      = 0;

  virtual Ptr<Options> getOptions() = 0;

  virtual void setShortlistGenerator(
      Ptr<data::ShortlistGenerator> shortlistGenerator)
      = 0;

  virtual Ptr<data::Shortlist> getShortlist() = 0;

  virtual data::SoftAlignment getAlignment() = 0;
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
  std::string getModelParametersAsString();

  virtual void createDecoderConfig(const std::string& name);

public:
  typedef data::Corpus dataset_type;

  EncoderDecoder(Ptr<Options> options);

  virtual Ptr<Options> getOptions() override { return options_; }

  std::vector<Ptr<EncoderBase>>& getEncoders();

  void push_back(Ptr<EncoderBase> encoder);

  std::vector<Ptr<DecoderBase>>& getDecoders();

  void push_back(Ptr<DecoderBase> decoder);

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
      Ptr<data::ShortlistGenerator> shortlistGenerator) override {
    shortlistGenerator_ = shortlistGenerator;
  };

  virtual Ptr<data::Shortlist> getShortlist() override {
    return decoders_[0]->getShortlist();
  };

  virtual data::SoftAlignment getAlignment() override {
    data::SoftAlignment aligns;
    for(auto aln : decoders_[0]->getAlignments()) {
      aligns.push_back({});
      aln->val()->get(aligns.back());
    }
    return aligns;
  };

  /*********************************************************************/

  virtual Ptr<DecoderState> startState(Ptr<ExpressionGraph> graph,
                                       Ptr<data::CorpusBatch> batch) override;

  virtual Ptr<DecoderState> step(Ptr<ExpressionGraph> graph,
                                 Ptr<DecoderState> state,
                                 const std::vector<size_t>& hypIndices,
                                 const std::vector<size_t>& embIndices,
                                 int dimBatch,
                                 int beamSize) override;

  virtual Ptr<DecoderState> stepAll(Ptr<ExpressionGraph> graph,
                                    Ptr<data::CorpusBatch> batch,
                                    bool clearGraph = true);

  virtual Expr build(Ptr<ExpressionGraph> graph,
                     Ptr<data::CorpusBatch> batch,
                     bool clearGraph = true) override;

  virtual Expr build(Ptr<ExpressionGraph> graph,
                     Ptr<data::Batch> batch,
                     bool clearGraph = true) override;
};

}  // namespace marian
