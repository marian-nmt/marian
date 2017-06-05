#pragma once

#include "marian.h"
#include "translator/history.h"
#include "translator/scorers.h"

#include "translator/helpers.h"
#include "translator/nth_element.h"

namespace marian {

class BeamSearch {
private:
  Ptr<Config> options_;
  std::vector<Ptr<Scorer>> scorers_;
  size_t beamSize_;

public:
  template <class... Args>
  BeamSearch(Ptr<Config> options,
             const std::vector<Ptr<Scorer>>& scorers,
             Args... args)
      : options_(options),
        scorers_(scorers),
        beamSize_(options_->get<size_t>("beam-size")) {}

  Beam toHyps(const std::vector<uint> keys,
              const std::vector<float> costs,
              size_t vocabSize,
              const Beam& beam,
              std::vector<Ptr<ScorerState>>& states) {
    Beam newBeam;
    for(int i = 0; i < keys.size(); ++i) {
      int embIdx = keys[i] % vocabSize;
      int hypIdx = keys[i] / vocabSize;
      float cost = costs[i];

      std::vector<float> breakDown(states.size(), 0);
      beam[hypIdx]->GetCostBreakdown().resize(states.size(), 0);

      for(int j = 0; j < states.size(); ++j)
        breakDown[j] = states[j]->breakDown(keys[i])
                       + beam[hypIdx]->GetCostBreakdown()[j];

      auto hyp = New<Hypothesis>(beam[hypIdx], embIdx, hypIdx, cost);
      hyp->GetCostBreakdown() = breakDown;
      newBeam.push_back(hyp);
    }
    return newBeam;
  }

  Beam pruneBeam(const Beam& beam) {
    Beam newBeam;
    for(auto hyp : beam) {
      if(hyp->GetWord() > 0) {
        newBeam.push_back(hyp);
      }
    }
    return newBeam;
  }

  Ptr<History> search(Ptr<ExpressionGraph> graph,
                      Ptr<data::CorpusBatch> batch,
                      size_t sentenceId = 0) {
    auto history = New<History>(sentenceId, options_->get<bool>("normalize"));
    Beam beam(1, New<Hypothesis>());
    bool first = true;
    bool final = false;
    std::vector<size_t> beamSizes(1, beamSize_);
    auto nth = New<NthElement>(beamSize_, batch->size());
    history->Add(beam);

    std::vector<Ptr<ScorerState>> states;

    for(auto scorer : scorers_) {
      scorer->clear(graph);
    }

    for(auto scorer : scorers_) {
      states.push_back(scorer->startState(graph, batch));
    }

    do {
      //**********************************************************************
      // create constant containing previous costs for current beam
      std::vector<size_t> hypIndices;
      std::vector<size_t> embIndices;
      Expr prevCosts;
      if(first) {
        // no cost
        prevCosts = graph->constant({1, 1, 1, 1},
                                    keywords::init = inits::from_value(0));
      } else {
        std::vector<float> beamCosts;
        for(auto hyp : beam) {
          hypIndices.push_back(hyp->GetPrevStateIndex());
          embIndices.push_back(hyp->GetWord());
          beamCosts.push_back(hyp->GetCost());
        }
        prevCosts
            = graph->constant({1, 1, 1, (int)beamCosts.size()},
                              keywords::init = inits::from_vector(beamCosts));
      }

      //**********************************************************************
      // prepare costs for beam search
      auto totalCosts = prevCosts;

      for(int i = 0; i < scorers_.size(); ++i) {
        states[i] = scorers_[i]->step(graph, states[i], hypIndices, embIndices);
        totalCosts
            = totalCosts + scorers_[i]->getWeight() * states[i]->getProbs();
      }

      if(first)
        graph->forward();
      else
        graph->forwardNext();

      //**********************************************************************
      // suppress specific symbols if not at right positions
      if(!options_->get<bool>("allow-unk"))
        suppressUnk(totalCosts);
      for(auto state : states)
        state->blacklist(totalCosts, batch);

      //**********************************************************************
      // perform beam search and pruning
      std::vector<unsigned> outKeys;
      std::vector<float> outCosts;

      beamSizes[0] = first ? beamSize_ : beam.size();
      nth->getNBestList(beamSizes, totalCosts->val(), outCosts, outKeys, first);

      int dimTrgVoc = totalCosts->shape()[1];
      beam = toHyps(outKeys, outCosts, dimTrgVoc, beam, states);

      final = history->size() >= 3 * batch->words();
      history->Add(beam, final);
      beam = pruneBeam(beam);

      first = false;

    } while(!beam.empty() && !final);

    return history;
  }
};
}
