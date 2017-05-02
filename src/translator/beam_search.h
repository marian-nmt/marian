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
                const Beam& beam,
                Ptr<DecoderState> state) {
      
      Beam newBeam;
      for(int i = 0; i < keys.size(); ++i) {
        int embIdx = keys[i] % vocabSize;
        int hypIdx = keys[i] / vocabSize;
        float cost = costs[i];
        
        auto breakDown = state->breakDown(keys[i]);
        
        int scores = breakDown.size();
        beam[hypIdx]->GetCostBreakdown().resize(scores, 0);
        
        std::vector<float> weights(1, 1);
        weights.resize(scores, 0);
        
        //std::vector<float> weights = options_->get<std::vector<float>>("weights");
        //weights.resize(scores, 0);
        
        //std::vector<float> weights = { 0.724739, -0.0248948, 0.121632, 0.00900556, -0.119728 };
        //std::vector<float> weights = { 0.476082, 0.0476081, 0.0141789, 0.200336, -0.261795 };
        
        float totalCost = 0;
        for(int i = 0; i < scores; ++i) {
          breakDown[i] += beam[hypIdx]->GetCostBreakdown()[i];
          totalCost += weights[i] * breakDown[i];
        }
        
        newBeam.push_back(
          New<Hypothesis>(beam[hypIdx], embIdx, hypIdx, totalCost));
        
        newBeam.back()->GetCostBreakdown() = breakDown;
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
     
      for(auto hyp : beam) {
        hypIndeces.push_back(hyp->GetPrevStateIndex());
        embIndeces.push_back(hyp->GetWord());
      }

      Ptr<DecoderState> selectedState
        = hypIndeces.empty() ? state : state->select(hypIndeces);
      
      builder_->selectEmbeddings(graph, selectedState, embIndeces, position);
      selectedState->setSingleStep(true);
      
      auto nextState = builder_->step(selectedState);
      nextState->setProbs(logsoftmax(nextState->getProbs()));
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

        Expr totalCosts;
        if(first) {
          state = step(graph, startState, history->size() - 1);
          auto probs = state->getProbs();
          
          auto costs = graph->constant(keywords::shape={1, 1, 1, 1},
                                       keywords::init=inits::from_value(0));
          totalCosts = probs + costs;
          graph->forward();
        }
        else {
          state = step(graph, state, beam, history->size() - 1);
          beamSizes[0] = beam.size();
        
          auto probs = state->getProbs();
          std::vector<float> beamCosts;
          for(auto hyp : beam)
            beamCosts.push_back(hyp->GetCost());
          auto costs = graph->constant(keywords::shape={1, 1, 1, (int)beamCosts.size()},
                                       keywords::init=inits::from_vector(beamCosts));
          totalCosts = probs + costs;
          
          graph->forwardNext();
        }

        
        std::vector<unsigned> outKeys;
        std::vector<float> outCosts;

        if(!options_->get<bool>("allow-unk"))
          suppressUnk(totalCosts);
        
        auto attState = std::dynamic_pointer_cast<DecoderStateHardAtt>(state);
        if(attState) {
          auto attentionIdx = attState->getAttentionIndices();
          int dimVoc = totalCosts->shape()[1];
          for(int i = 0; i < attentionIdx.size(); i++) {
            if(batch->front()->indeces()[attentionIdx[i]] != 0) {                
              totalCosts->val()->set(i * dimVoc + EOS_ID,
                                     std::numeric_limits<float>::lowest());
            }
            else {                
              totalCosts->val()->set(i * dimVoc + STP_ID,
                                     std::numeric_limits<float>::lowest());
            }
          }
        }
        
        nth->getNBestList(beamSizes, totalCosts->val(),
                          outCosts, outKeys, first);
        first = false;

        int dimTrgVoc = totalCosts->shape()[1];
        beam = toHyps(outKeys, outCosts, dimTrgVoc, beam, state);
        
        
        final = history->size() >= 3 * batch->words();
        history->Add(beam, final);
        beam = pruneBeam(beam);

      } while(!beam.empty() && !final);

      return history;
    }
};

}