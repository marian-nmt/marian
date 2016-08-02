#pragma once

#include <vector>
#include <yaml-cpp/yaml.h>
#include <iostream>

#include "matrix.h"
#include "scorer.h"
#include "loader.h"
#include "dl4mt.h"

#include "threadpool.h"

class MultiEncoderState : public State {
  public:
    mblas::Matrix& GetStates() {
      return states_;
    }

    mblas::Matrix& GetEmbeddings() {
      return embeddings_;
    }

    const mblas::Matrix& GetStates() const {
      return states_;
    }

    const mblas::Matrix& GetEmbeddings() const {
      return embeddings_;
    }

  private:
    mblas::Matrix states_;
    mblas::Matrix embeddings_;
};

class MultiEncoder : public Scorer {
  private:
    typedef MultiEncoderState EDState;

  public:
    MultiEncoder(const std::string& name,
                   const YAML::Node& config,
                   size_t numEncoders,
                   const Weights& model)
      : Scorer(name, config, 0),
        model_(model),
        SourceContexts_(numEncoders),
        decoder_(new Decoder(model_, numEncoders)) {
      for (size_t i = 0; i < numEncoders; ++i) {
        encoders_.emplace_back(new Encoder(model_, i));
      }
    }

    virtual void Score(const State& in,
                       Prob& prob,
                       State& out) {
      const EDState& edIn = in.get<EDState>();
      EDState& edOut = out.get<EDState>();

      decoder_->MakeStep(edOut.GetStates(), prob,
                         edIn.GetStates(), edIn.GetEmbeddings(),
                         SourceContexts_);
    }

    virtual State* NewState() {
      return new EDState();
    }

    virtual void BeginSentenceState(State& state) {
      EDState& edState = state.get<EDState>();
      decoder_->EmptyState(edState.GetStates(), SourceContexts_, 1);
      decoder_->EmptyEmbedding(edState.GetEmbeddings(), 1);
    }

    virtual void SetSource(const Sentence& source) {
      for (size_t i = 0; i < encoders_.size(); ++i) {
        encoders_[i]->GetContext(source.GetWords(i),
                                 SourceContexts_[i]);
      }
    }

    virtual void AssembleBeamState(const State& in,
                                   const Beam& beam,
                                   State& out) {
      std::vector<size_t> beamWords;
      std::vector<size_t> beamStateIds;
      for(auto h : beam) {
         beamWords.push_back(h->GetWord());
         beamStateIds.push_back(h->GetPrevStateIndex());
      }

      const EDState& edIn = in.get<EDState>();
      EDState& edOut = out.get<EDState>();

      mblas::Assemble(edOut.GetStates(),
                      edIn.GetStates(), beamStateIds);
      decoder_->Lookup(edOut.GetEmbeddings(), beamWords);
    }

    void GetAttention(mblas::Matrix& Attention, size_t idx) {
      decoder_->GetAttention(Attention, idx);
    }

    size_t GetVocabSize() const {
      return decoder_->GetVocabSize();
    }

  private:
    const Weights& model_;
    std::vector<std::unique_ptr<Encoder>> encoders_;
    std::unique_ptr<Decoder> decoder_;

    std::vector<mblas::Matrix> SourceContexts_;
    mblas::Matrix ConcatenatedSourceContext_;
};

class MultiEncoderLoader : public Loader {
  public:
    MultiEncoderLoader(const std::string name,
                         const YAML::Node& config)
     : Loader(name, config) {}

    virtual void Load() {
      std::string path = Get<std::string>("path");
      auto devices = God::Get<std::vector<size_t>>("devices");
      ThreadPool devicePool(devices.size());
      weights_.resize(devices.size());

      size_t i = 0;
      for(auto d : devices) {
        devicePool.enqueue([i, d, &path, this] {
          LOG(info) << "Loading model " << path << " onto gpu" << d;
          cudaSetDevice(d);
          weights_[i].reset(new Weights(path, d, God::Get<size_t>("encoders")));
        });
        ++i;
      }
    }

    virtual ScorerPtr NewScorer(size_t taskId) {
      size_t i = taskId % weights_.size();
      size_t d = weights_[i]->GetDevice();
      cudaSetDevice(d);
      size_t numEncoders = God::Get<size_t>("encoders");
      std::cerr << "NUM ENCODERS: " << numEncoders << std::endl;
      return ScorerPtr(new MultiEncoder(name_, config_,
                                          numEncoders, *weights_[i]));
    }

  private:
    std::vector<std::unique_ptr<Weights>> weights_;
};
