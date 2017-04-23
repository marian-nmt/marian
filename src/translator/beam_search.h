#pragma once

#include "marian.h"
#include "translator/nth_element.h"
#include "translator/helpers.h"
#include "translator/history.h"

namespace marian {

template <class Builder>
class BeamSearch {
  private:
    Ptr<Config> options_;
    Ptr<Builder> builder_;
    size_t beamSize_;
    cudaStream_t stream_{0};

  public:
    typedef Builder model_type;    
    
    template <class ...Args>
    BeamSearch(Ptr<Config> options, Args ...args)
     : options_(options),
       builder_(New<Builder>(options, keywords::inference=true, args...)),
       beamSize_(options_->get<size_t>("beam-size"))
    {}

    Beam toHyps(const std::vector<uint> keys,
                const std::vector<float> costs,
                size_t vocabSize,
                const Beam& beam) {
      Beam newBeam;
      for(int i = 0; i < keys.size(); ++i) {
        int embIdx = keys[i] % vocabSize;
        int hypIdx = keys[i] / vocabSize;
        float cost = costs[i];
          
        newBeam.push_back(
          New<Hypothesis>(beam[hypIdx], embIdx, hypIdx, cost));
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

    Ptr<DecoderState>
    step(Ptr<ExpressionGraph> graph,
         Ptr<DecoderState> state,
         size_t position) {
      
      builder_->selectEmbeddings(graph, state, {}, position);
      
      state->setSingleStep(true);
      auto nextState = builder_->step(state);
      
      nextState->setProbs(logsoftmax(nextState->getProbs()));
      return nextState;
    }
    
    Ptr<DecoderState>
    step(Ptr<ExpressionGraph> graph,
         Ptr<DecoderState> state,
         const Beam& beam,
         size_t position) {

      std::vector<size_t> hypIndeces;
      std::vector<size_t> embIndeces;
      std::vector<float> beamCosts;

      for(auto hyp : beam) {
        hypIndeces.push_back(hyp->GetPrevStateIndex());
        embIndeces.push_back(hyp->GetWord());
        beamCosts.push_back(hyp->GetCost());
      }

      Ptr<DecoderState> selectedState
        = hypIndeces.empty() ? state : state->select(hypIndeces);
      
      builder_->selectEmbeddings(graph, selectedState, embIndeces, position);
      
      selectedState->setSingleStep(true);
      
      auto nextState = builder_->step(selectedState);

      auto costs = graph->constant(keywords::shape={1, 1, 1, (int)beamCosts.size()},
                                   keywords::init=inits::from_vector(beamCosts));
      auto totalCosts = logsoftmax(nextState->getProbs()) + costs;
      
      nextState->setProbs(totalCosts);
      
      return nextState;
    }

    Ptr<History> search(Ptr<ExpressionGraph> graph,
                        Ptr<data::CorpusBatch> batch,
                        size_t sentenceId = 0) {

      builder_->clear(graph);
      auto startState = builder_->startState(graph, batch);
        
      auto history = New<History>(sentenceId, options_->get<bool>("normalize"));
      Beam beam(1, New<Hypothesis>());
      bool first = true;
      bool final = false;
      std::vector<size_t> beamSizes(1, beamSize_);
      auto nth = New<NthElement>(beamSize_, batch->size(), stream_);

      history->Add(beam);

      Ptr<DecoderState> state;
      do {

        if(first) {
          state = step(graph, startState, history->size() - 1);
          graph->forward();
        }
        else {
          state = step(graph, state, beam, history->size() - 1);
          beamSizes[0] = beam.size();
          graph->forwardNext();
        }

        size_t dimTrgVoc = state->getProbs()->shape()[1];

        std::vector<unsigned> outKeys;
        std::vector<float> outCosts;

        if(!options_->get<bool>("allow-unk"))
          suppressUnk(state->getProbs());
        
        nth->getNBestList(beamSizes, state->getProbs()->val(),
                          outCosts, outKeys, first);
        first = false;

        beam = toHyps(outKeys, outCosts, dimTrgVoc, beam);
        final = history->size() >= 3 * batch->words();
        history->Add(beam, final);
        beam = pruneBeam(beam);

      } while(!beam.empty() && !final);

      return history;
    }
};

}