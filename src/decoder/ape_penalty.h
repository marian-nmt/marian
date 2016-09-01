#pragma once

#include <vector>

#include "types.h"
#include "file_stream.h"
#include "scorer.h"
#include "matrix.h"
#include "loader.h"

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
               const Penalties& penalties);
    
    // @TODO: make this work on GPU
    virtual void SetSource(const Sentence& source);
    
    // @TODO: make this work on GPU
    virtual void Score(const State& in,
    		mblas::BaseMatrix& prob,
    		State& out);
    
    virtual State* NewState();
    
    virtual void BeginSentenceState(State& state);
    
    virtual void AssembleBeamState(const State& in,
                                   const Beam& beam,
                                   State& out);
    
    virtual size_t GetVocabSize() const;
    
    virtual Prob *CreateMatrix();

  private:
    std::vector<float> costs_;
};

/////////////////////////////////////////////////////
class ApePenaltyLoader : public Loader {
  public:
    ApePenaltyLoader(const std::string& name,
                     const YAML::Node& config);
  
    virtual void Load();
  
    virtual ScorerPtr NewScorer(size_t taskId);
    
  private:
    SrcTrgMap srcTrgMap_;
    Penalties penalties_; 
};
