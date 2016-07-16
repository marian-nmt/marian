#pragma once

#include <vector>

#include "types.h"
#include "file_stream.h"
#include "scorer.h"
#include "matrix.h"
#include "decoder/sentence.h"

typedef std::vector<Word> SrcTrgMap;
typedef std::vector<float> Penalties;

class ApePenaltyState : public State {
  // Dummy, this scorer is stateless
};

class ApePenalty : public Scorer {
  private:
    const SrcTrgMap& srcTrgMap_;
    const Penalties& penalties_;

  public:
    ApePenalty(const std::string& name,
               const YAML::Node& config,
               size_t tab,
               const SrcTrgMap& srcTrgMap,
               const Penalties& penalties)
    : Scorer(name, config, tab), srcTrgMap_(srcTrgMap),
      penalties_(penalties)
    { }

    // @TODO: make this work on GPU
    virtual void SetSource(const Sentence& source) {
      const Words& words = source.GetWords(tab_);

      costs_.clear();
      costs_.resize(penalties_.size());
      std::copy(penalties_.begin(), penalties_.end(), costs_.begin());

      for(auto&& s : words) {
        Word t = srcTrgMap_[s];
        if(t != UNK && t < costs_.size())
          costs_[t] = 0.0;
      }
    }

    // @TODO: make this work on GPU
    virtual void Score(const State& in,
                       Prob& prob,
                       State& out) {
      size_t cols = prob.Cols();
      costs_.resize(cols, -1.0);
      for(size_t i = 0; i < prob.Rows(); ++i)
        std::copy(costs_.begin(), costs_.begin() + cols, prob.begin() + i * cols);
    }

    virtual State* NewState() {
      return new ApePenaltyState();
    }

    virtual void BeginSentenceState(State& state) { }

    virtual void AssembleBeamState(const State& in,
                                   const Beam& beam,
                                   State& out) { }

    virtual size_t GetVocabSize() const {
      UTIL_THROW2("Not correctly implemented");
    }

   void Filter(const std::vector<size_t>& filterIds) {}


  private:
    std::vector<float> costs_;
};

class ApePenaltyLoader : public Loader {
  public:
    ApePenaltyLoader(const std::string& name,
                     const YAML::Node& config)
     : Loader(name, config) {}

    virtual void Load() {
      size_t tab = Has("tab") ? Get<size_t>("tab") : 0;
      const Vocab& svcb = God::GetSourceVocab(tab);
      const Vocab& tvcb = God::GetTargetVocab();

      srcTrgMap_.resize(svcb.size(), UNK);
      for(Word s = 0; s < svcb.size(); ++s)
        srcTrgMap_[s] = tvcb[svcb[s]];

      penalties_.resize(tvcb.size(), -1.0);

      if(Has("path")) {
        LOG(info) << "Loading APE penalties from " << Get<std::string>("path");
        YAML::Node penalties = YAML::Load(InputFileStream(Get<std::string>("path")));
        for(auto&& pair : penalties) {
          std::string entry = pair.first.as<std::string>();
          float penalty = pair.second.as<float>();
          penalties_[tvcb[entry]] = -penalty;
        }
      }
    }

    virtual ScorerPtr NewScorer(size_t taskId) {
      size_t tab = Has("tab") ? Get<size_t>("tab") : 0;
      return ScorerPtr(new ApePenalty(name_, config_, tab,
                                      srcTrgMap_, penalties_));
    }

  private:
    SrcTrgMap srcTrgMap_;
    Penalties penalties_;
};
