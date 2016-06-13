#pragma once

#include <vector>
#include <yaml-cpp/yaml.h>

#include "matrix.h"
#include "scorer.h"
#include "loader.h"
#include "dl4mt.h"

#include "threadpool.h"

class EncoderDecoderState : public State {
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

class EncoderDecoder : public Scorer {
  private:
    typedef EncoderDecoderState EDState;
    
  public:
    EncoderDecoder(const std::string& name,
                   const YAML::Node& config,
                   size_t tab,
                   const Weights& model)
    : Scorer(name, config, tab), model_(model),
      encoder_(new Encoder(model_)), decoder_(new Decoder(model_))
    {}
    
    virtual void Score(const State& in,
                       Prob& prob,
                       State& out) {
      const EDState& edIn = in.get<EDState>();
      EDState& edOut = out.get<EDState>();
      
      decoder_->MakeStep(edOut.GetStates(), prob,
                        edIn.GetStates(), edIn.GetEmbeddings(),
                        SourceContext_);
    }
    
    virtual State* NewState() {
      return new EDState(); 
    }
    
    virtual void BeginSentenceState(State& state) {
      EDState& edState = state.get<EDState>();
      decoder_->EmptyState(edState.GetStates(), SourceContext_, 1);
      decoder_->EmptyEmbedding(edState.GetEmbeddings(), 1);
    }

    virtual void SetSource(const Sentence& source) {
      encoder_->GetContext(source.GetWords(tab_),
                           SourceContext_);
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
    
    void GetAttention(mblas::Matrix& Attention) {
      decoder_->GetAttention(Attention);
    }
    
    size_t GetVocabSize() const {
      return decoder_->GetVocabSize();
    }
    
  private:
    const Weights& model_;
    std::unique_ptr<Encoder> encoder_;
    std::unique_ptr<Decoder> decoder_;
    
    mblas::Matrix SourceContext_;
};

class EncoderDecoderLoader : public Loader {
  public:
    EncoderDecoderLoader(const std::string name,
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
          LOG(info) << "Loading model " << path;
          weights_[i].reset(new Weights(path, d));
        });
        ++i;
      }
    }
  
    virtual ScorerPtr NewScorer(size_t taskId) {
      size_t i = taskId % weights_.size();
      size_t d = weights_[i]->GetDevice();
      size_t tab = Has("tab") ? Get<size_t>("tab") : 0;
      return ScorerPtr(new EncoderDecoder(name_, config_,
                                          tab, *weights_[i]));
    }
    
  private:
    std::vector<std::unique_ptr<Weights>> weights_;
};
