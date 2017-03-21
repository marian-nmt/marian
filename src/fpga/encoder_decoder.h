#pragma once
#include "common/scorer.h"
#include "matrix.h"
#include "types.h"

namespace amunmt {
namespace FPGA {

class Weights;
class Encoder;
class Decoder;

class EncoderDecoder : public Scorer {
public:
  EncoderDecoder(const God &god,
           const std::string& name,
                 const YAML::Node& config,
                 size_t tab,
                 const Weights& model,
                 const cl_context &context,
                 const cl_device_id &device);

  virtual void Decode(const God &god, const State& in,
                     State& out, const std::vector<size_t>& beamSizes);

  virtual void BeginSentenceState(State& state, size_t batchSize=1);

  virtual void AssembleBeamState(const State& in,
                                 const Beam& beam,
                                 State& out);

  virtual void SetSource(const Sentences& sources);

  virtual void Filter(const std::vector<size_t>&);

  virtual State* NewState() const;

  virtual size_t GetVocabSize() const;

  virtual BaseMatrix& GetProbs();

protected:
  const Weights& model_;
  mblas::Matrix SourceContext_;

  std::unique_ptr<Encoder> encoder_;
  std::unique_ptr<Decoder> decoder_;

  const cl_context &context_;
};



}
}

