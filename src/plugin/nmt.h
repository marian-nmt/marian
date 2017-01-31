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
		
    static size_t GetDevices(size_t = 1);
    void SetDevice();
    size_t GetDevice();
    const amunmt::God &GetGod() const
    { return god_; }

    void initGod(const std::string& configPath);

    AmunOutput SetSource(const std::vector<size_t>& words);

    AmunOutputs Score(const AmunInputs &inputs);

  private:
    amunmt::God god_;
    
};

}
