// -*- mode: c++; tab-width: 2; indent-tabs-mode: nil -*-
#include <iostream>

#include "common/god.h"
#include "common/sentences.h"

#include "encoder_decoder.h"
#include "gpu/mblas/matrix_functions.h"
#include "gpu/dl4mt/dl4mt.h"
#include "gpu/decoder/encoder_decoder_state.h"
#include "gpu/decoder/best_hyps.h"

using namespace std;

namespace amunmt {
namespace GPU {

EncoderDecoder::EncoderDecoder(
		const God &god,
		const std::string& name,
        const YAML::Node& config,
        size_t tab,
        const Weights& model)
  : Scorer(name, config, tab),
    model_(model),
    encoder_(new Encoder(model_)),
    decoder_(new Decoder(god, model_)),
    indices_(god.Get<size_t>("beam-size"))
{}

State* EncoderDecoder::NewState() const {
  return new EDState();
}

void EncoderDecoder::SetSource(const Sentences& source) {
  encoder_->GetContext(source, tab_, sourceContext_, batchMapping_);
  //cerr << "GPU sourceContext_=" << sourceContext_.Debug(1) << endl;
}

void EncoderDecoder::BeginSentenceState(State& state, size_t batchSize) {
  EDState& edState = state.get<EDState>();
  decoder_->EmptyState(edState.GetStates(), sourceContext_, batchSize, batchMapping_);

  decoder_->EmptyEmbedding(edState.GetEmbeddings(), batchSize);
}

void EncoderDecoder::Decode(const God &god, const State& in, State& out, const std::vector<uint>& beamSizes) {
  const EDState& edIn = in.get<EDState>();
  EDState& edOut = out.get<EDState>();

  decoder_->Decode(edOut.GetStates(),
                     edIn.GetStates(),
                     edIn.GetEmbeddings(),
                     sourceContext_,
                     batchMapping_,
                     beamSizes);
}

void EncoderDecoder::AssembleBeamState(const State& in,
                               const Beam& beam,
                               State& out) {
  std::vector<size_t> beamWords;
  std::vector<size_t> beamStateIds;
  for (const HypothesisPtr &h : beam) {
     beamWords.push_back(h->GetWord());
     beamStateIds.push_back(h->GetPrevStateIndex());
  }
  //cerr << "beamWords=" << Debug(beamWords, 2) << endl;
  //cerr << "beamStateIds=" << Debug(beamStateIds, 2) << endl;

  const EDState& edIn = in.get<EDState>();
  EDState& edOut = out.get<EDState>();
  indices_.resize(beamStateIds.size());
  thrust::host_vector<size_t> tmp = beamStateIds;

  mblas::copy(thrust::raw_pointer_cast(tmp.data()),
      beamStateIds.size(),
      thrust::raw_pointer_cast(indices_.data()),
      cudaMemcpyHostToDevice);
  //cerr << "indices_=" << mblas::Debug(indices_, 2) << endl;

  mblas::Assemble(edOut.GetStates(), edIn.GetStates(), indices_);
  //cerr << "edOut.GetStates()=" << edOut.GetStates().Debug(1) << endl;

  //cerr << "beamWords=" << Debug(beamWords, 2) << endl;
  decoder_->Lookup(edOut.GetEmbeddings(), beamWords);
  //cerr << "edOut.GetEmbeddings()=" << edOut.GetEmbeddings().Debug(1) << endl;
}

void EncoderDecoder::GetAttention(mblas::Matrix& Attention) {
  decoder_->GetAttention(Attention);
}

BaseMatrix& EncoderDecoder::GetProbs() {
  return decoder_->GetProbs();
}

mblas::Matrix& EncoderDecoder::GetAttention() {
  return decoder_->GetAttention();
}

size_t EncoderDecoder::GetVocabSize() const {
  return decoder_->GetVocabSize();
}

void EncoderDecoder::Filter(const std::vector<size_t>& filterIds) {
  decoder_->Filter(filterIds);
}

EncoderDecoder::~EncoderDecoder() {}

////////////////////////////////////////////
EncoderDecoderLoader::EncoderDecoderLoader(const std::string name,
                     const YAML::Node& config)
 : Loader(name, config) {}

void EncoderDecoderLoader::Load(const God &god) {
  std::string path = Get<std::string>("path");
  std::vector<size_t> devices = god.Get<std::vector<size_t>>("devices");

  size_t maxDeviceId = 0;
  for (size_t i = 0; i < devices.size(); ++i) {
    if (devices[i] > maxDeviceId) {
      maxDeviceId = devices[i];
    }
  }

  ThreadPool devicePool(devices.size());
  weights_.resize(maxDeviceId + 1);

  for(auto d : devices) {
    devicePool.enqueue([d, &path, this] {
        LOG(info, "Loading model {} onto gpu {}", path, d);
        HANDLE_ERROR(cudaSetDevice(d));
        weights_[d].reset(new Weights(path, d));
      });
  }
}

EncoderDecoderLoader::~EncoderDecoderLoader()
{
  for (size_t d = 0; d < weights_.size(); ++d) {
    const Weights *weights = weights_[d].get();
    if (weights) {
      HANDLE_ERROR(cudaSetDevice(d));
      weights_[d].reset(nullptr);
    }
  }
}

ScorerPtr EncoderDecoderLoader::NewScorer(const God &god, const DeviceInfo &deviceInfo) const {
  size_t d = deviceInfo.deviceId;

  HANDLE_ERROR(cudaSetDevice(d));
  size_t tab = Has("tab") ? Get<size_t>("tab") : 0;
  return ScorerPtr(new EncoderDecoder(god, name_, config_,
                                      tab, *weights_[d]));
}

BestHypsBasePtr EncoderDecoderLoader::GetBestHyps(const God &god) const {
  return BestHypsBasePtr(new GPU::BestHyps(god));
}

}
}

