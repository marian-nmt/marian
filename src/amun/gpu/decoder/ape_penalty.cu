#include "ape_penalty.h"
#include "common/god.h"
#include "common/vocab.h"
#include "gpu/types-gpu.h"
#include "gpu/decoder/best_hyps.h"

namespace GPU {

ApePenalty::ApePenalty(const std::string& name,
		   const YAML::Node& config,
		   size_t tab,
		   const SrcTrgMap& srcTrgMap,
		   const Penalties& penalties)
  : Scorer(name, config, tab), srcTrgMap_(srcTrgMap),
    penalties_(penalties)
{}

// @TODO: make this work on GPU
void ApePenalty::SetSource(const Sentence& source) {
  const Words& words = source.GetWords(tab_);

  costs_.clear();
  costs_.resize(penalties_.size());
  algo::copy(penalties_.begin(), penalties_.end(), costs_.begin());

  for(auto&& s : words) {
	Word t = srcTrgMap_[s];
	if(t != UNK && t < costs_.size())
	  costs_[t] = 0.0;
  }
}

// @TODO: make this work on GPU
void ApePenalty::Score(const State& in, State& out) {
  size_t cols = Probs_.Cols();
  costs_.resize(cols, -1.0);
  for(size_t i = 0; i < Probs_.dim(0); ++i) {
    algo::copy(costs_.begin(), costs_.begin() + cols, Probs_.begin() + i * cols);
  }
}

State* ApePenalty::NewState() {
  return new ApePenaltyState();
}

void ApePenalty::BeginSentenceState(State& state) { }

void ApePenalty::AssembleBeamState(const State& in,
							   const Beam& beam,
							   State& out) { }

size_t ApePenalty::GetVocabSize() const {
  UTIL_THROW2("Not correctly implemented");
}

BaseMatrix& ApePenalty::GetProbs() {
  return Probs_;
}

/////////////////////////////////////////////////////

ApePenaltyLoader::ApePenaltyLoader(const std::string& name,
                 const YAML::Node& config)
 : Loader(name, config) {}

void ApePenaltyLoader::Load() {
  size_t tab = Has("tab") ? Get<size_t>("tab") : 0;
  const Vocab& svcb = God::GetSourceVocab(tab);
  const Vocab& tvcb = God::GetTargetVocab();

  srcTrgMap_.resize(svcb.size(), UNK);
  for(Word s = 0; s < svcb.size(); ++s)
    srcTrgMap_[s] = tvcb[svcb[s]];

  penalties_.resize(tvcb.size(), -1.0);

  if(Has("path")) {
    LOG(info)->info("Loading APE penalties from {}",Get<std::string>("path"));
    YAML::Node penalties = YAML::Load(InputFileStream(Get<std::string>("path")));
    for(auto&& pair : penalties) {
      std::string entry = pair.first.as<std::string>();
      float penalty = pair.second.as<float>();
      penalties_[tvcb[entry]] = -penalty;
    }
  }
}

ScorerPtr ApePenaltyLoader::NewScorer(God &god, size_t taskId) {
  size_t tab = Has("tab") ? Get<size_t>("tab") : 0;
  return ScorerPtr(new ApePenalty(name_, config_, tab,
                                  srcTrgMap_, penalties_));
}

BestHypsBase *ApePenaltyLoader::GetBestHyps(God &god) {
  return new GPU::BestHyps();
}

}

