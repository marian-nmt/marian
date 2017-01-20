#pragma once

#include <vector>

#include "common/types.h"
#include "common/file_stream.h"
#include "common/scorer.h"
#include "common/base_best_hyps.h"
#include "common/loader.h"

#include "gpu/mblas/matrix.h"

namespace GPU {

typedef std::vector<Word> SrcTrgMap;
typedef std::vector<float> Penalties;

class ApePenaltyState : public State {
	  // Dummy, this scorer is stateless
  public:
    virtual std::string Debug() const {
      return "ApePenaltyState";
    }

};

class ApePenalty : public Scorer {
  public:
    ApePenalty(const std::string& name,
               const YAML::Node& config,
               size_t tab,
               const SrcTrgMap& srcTrgMap,
               const Penalties& penalties);

    // @TODO: make this work on GPU
    virtual void SetSource(const Sentence& source);

    // @TODO: make this work on GPU
    virtual void Score(const State& in, State& out);

    virtual State* NewState();

    virtual void BeginSentenceState(State& state);

    virtual void AssembleBeamState(const State& in,
                                   const Beam& beam,
                                   State& out);

    virtual size_t GetVocabSize() const;

	  void Filter(const std::vector<size_t>&) {}

    virtual BaseMatrix& GetProbs();

  private:
    std::vector<float> costs_;
    const SrcTrgMap& srcTrgMap_;
    mblas::Matrix Probs_;
    const Penalties& penalties_;

};

/////////////////////////////////////////////////////
class ApePenaltyLoader : public Loader {
  public:
    ApePenaltyLoader(const std::string& name,
                     const YAML::Node& config);

    virtual void Load();

    virtual ScorerPtr NewScorer(God &god, size_t taskId);
    virtual BestHypsBase *GetBestHyps(God &god);

  private:
    SrcTrgMap srcTrgMap_;
    Penalties penalties_;
};

}
