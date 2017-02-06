#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <cmath>
#include <boost/timer/timer.hpp>

#include "nmt.h"
#include "common/vocab.h"
#include "common/god.h"
#include "common/history.h"
#include "common/sentence.h"
#include "common/search.h"

namespace amunmt {

void MosesPlugin::initGod(const std::string& configPath) {
  std::string configs = "-c " + configPath;
  god_.Init(configs);
}

MosesPlugin::MosesPlugin()
{}

MosesPlugin::~MosesPlugin()
{
}

size_t MosesPlugin::GetDevices(size_t maxDevices) {
  int num_gpus = 0; // number of CUDA GPUs
  HANDLE_ERROR( cudaGetDeviceCount(&num_gpus));
  std::cerr << "Number of CUDA devices: " << num_gpus << std::endl;

  for (int i = 0; i < num_gpus; i++) {
      cudaDeviceProp dprop;
      HANDLE_ERROR( cudaGetDeviceProperties(&dprop, i));
      std::cerr << i << ": " << dprop.name << std::endl;
  }
  return (size_t)std::min(num_gpus, (int)maxDevices);
}

AmunOutput MosesPlugin::SetSource(const std::vector<size_t>& words) {
  AmunOutput ret;

  amunmt::Sentences sentences;
  sentences.push_back(SentencePtr(new Sentence(god_, 0, words)));

  // Encode
  Search &search = god_.GetSearch();
  size_t numScorers = search.GetScorers().size();

  std::shared_ptr<Histories> histories(new Histories(god_, sentences));

  size_t batchSize = sentences.size();
  Beam prevHyps(batchSize, HypothesisPtr(new Hypothesis()));

  States states = search.NewStates();

  search.PreProcess(god_, sentences, histories, prevHyps);
  search.Encode(sentences, states);

  // fill return info
  ret.states = states;
  ret.prevHyps = prevHyps;
  ret.score = 0;

  return ret;
}

AmunOutputs MosesPlugin::Score(const AmunInputs &inputs)
{
  AmunOutputs outputs(inputs.size());

  // TODO

  return outputs;
}

}
