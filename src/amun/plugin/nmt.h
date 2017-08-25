#pragma once

#include <iostream>
#include <algorithm>
#include <vector>
#include <set>
#include <memory>

#include "common/god.h"
#include "common/scorer.h"
#include "common/sentence.h"
#include "gpu/mblas/matrix.h"
#include "gpu/decoder/encoder_decoder.h"
#include "neural_phrase.h"
#include "hypo_info.h"

namespace amunmt {

class Vocab;

class NeuralPhrase;

class MosesPlugin {
  public:
    MosesPlugin();
		~MosesPlugin();
		
    void SetDevice();
    size_t GetDevice();
    const amunmt::God &GetGod() const
    { return god_; }

    void initGod(const std::string& configPath);

    HypoState SetSource(const std::vector<size_t>& words);

    HypoStates Score(const AmunInputs &inputs);

  private:
    amunmt::God god_;
    
};

}
