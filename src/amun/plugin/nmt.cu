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

  ret.sentences.reset(new Sentences());
  ret.sentences->push_back(SentencePtr(new Sentence(god_, 0, words)));

  // Encode
  Search &search = god_.GetSearch();
  size_t numScorers = search.GetScorers().size();

  std::shared_ptr<Histories> histories(new Histories(god_, *ret.sentences));

  size_t batchSize = ret.sentences->size();
  Beam prevHyps(batchSize, HypothesisPtr(new Hypothesis()));

  States states = search.NewStates();

  search.PreProcess(god_, *ret.sentences, histories, prevHyps);
  search.Encode(*ret.sentences, states);

  // fill return info
  ret.states = states;
  ret.prevHyps = prevHyps;
  ret.score = 0;

  return ret;
}

HypoStates MosesPlugin::Score(const AmunInputs &inputs)
{
  HypoStates outputs(inputs.size());

  // TODO

  return outputs;
}

}
