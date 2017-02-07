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

HypoState MosesPlugin::SetSource(const std::vector<size_t>& words) {
  HypoState ret;

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

HypoStates MosesPlugin::Score(const HypoStates &inputs)
{
  HypoStates outputs(inputs.size());

  // TODO

  return outputs;
}

}
