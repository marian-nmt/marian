#pragma once

#include "models/encdec.h"
#include "models/s2s.h"
#include "models/hardatt.h"

namespace marian {

class DecoderStateHardAttCDI : public DecoderStateHardAtt {
  private:
    std::vector<size_t> embeddingIndices_;
  
  public:
    DecoderStateHardAttCDI(const std::vector<Expr> states,
                           Expr probs,
                           Ptr<EncoderState> encState,
                           const std::vector<size_t>& attentionIndices,
                           const std::vector<size_t>& embeddingIndices)
    : DecoderStateHardAtt(states, probs, encState, attentionIndices),
      embeddingIndices_(embeddingIndices) {}

    virtual void setTargetEmbeddingIndeces(const std::vector<size_t>& embIndices) {
      embeddingIndices_ = embIndices;
    }
    
    virtual std::vector<size_t>& getTargetEmbeddingIndices() {
      return embeddingIndices_;
    }
    
    virtual Ptr<DecoderState> select(const std::vector<size_t>& selIdx) {
      int numSelected = selIdx.size();
      int dimState = states_[0]->shape()[1];
      
      std::vector<Expr> selectedStates;
      for(auto state : states_) {
        selectedStates.push_back(
          reshape(rows(state, selIdx),
                  {1, dimState, 1, numSelected})
        );
      }
      
      std::vector<size_t> selectedAttentionIndices;
      std::vector<size_t> selectedEmbs;
      
      for(auto i : selIdx) {
        selectedAttentionIndices.push_back(attentionIndices_[i]);
        selectedEmbs.push_back(embeddingIndices_[i]);
      }
      
      return New<DecoderStateHardAttCDI>(selectedStates, probs_, encState_,
                                        selectedAttentionIndices, selectedEmbs);
    }
};

class DecoderHardAttCDI : public DecoderBase {
  private:
    Ptr<RNN<GRU>> rnnL1;
    Ptr<MLRNN<GRU>> rnnLn;
  
  public:

    template <class ...Args>
    DecoderHardAttCDI(Ptr<Config> options, Args ...args)
     : DecoderBase(options, args...) {}

    virtual Ptr<DecoderState> startState(Ptr<EncoderState> encState) {
      using namespace keywords;

      auto meanContext = weighted_average(encState->getContext(),
                                          encState->getMask(),
                                          axis=2);

      bool layerNorm = options_->get<bool>("layer-normalization");
      auto start = Dense("ff_state",
                         options_->get<int>("dim-rnn"),
                         activation=act::tanh,
                         normalize=layerNorm)(meanContext);
      
      std::vector<Expr> startStates(options_->get<size_t>("layers-dec"), start);
      return New<DecoderStateHardAttCDI>(startStates, nullptr, encState,
                                         std::vector<size_t>({0}),
                                         std::vector<size_t>({0}));
    }
     
    virtual Ptr<DecoderState> step(Ptr<DecoderState> state) {
      using namespace keywords;

      int dimTrgVoc = options_->get<std::vector<int>>("dim-vocabs").back();
      
      int dimTrgEmb = options_->get<int>("dim-emb")
                    + options_->get<int>("dim-pos");
      
                    
      int dimDecState = options_->get<int>("dim-rnn");
      bool layerNorm = options_->get<bool>("layer-normalization");
      bool skipDepth = options_->get<bool>("skip");
      size_t decoderLayers = options_->get<size_t>("layers-dec");

      float dropoutRnn = inference_ ? 0 : options_->get<float>("dropout-rnn");
      float dropoutTrg = inference_ ? 0 : options_->get<float>("dropout-trg");

      auto stateHardAtt = std::dynamic_pointer_cast<DecoderStateHardAttCDI>(state);
      
      auto trgEmbeddings = stateHardAtt->getTargetEmbeddings();
      auto graph = trgEmbeddings->graph();
      
      auto context = stateHardAtt->getEncoderState()->getContext();
      int dimContext = context->shape()[1];
      int dimSrcWords = context->shape()[2];
      
      int dimBatch = context->shape()[0];
      int dimTrgWords = trgEmbeddings->shape()[2];
            
      if(dropoutTrg) {
        auto trgWordDrop = graph->dropout(dropoutTrg, {dimBatch, 1, dimTrgWords});
        trgEmbeddings = dropout(trgEmbeddings, mask=trgWordDrop);
      }
      
      auto flatContext = reshape(context, {dimBatch * dimSrcWords, dimContext});
      auto attendedContext = rows(flatContext, stateHardAtt->getAttentionIndices());
      attendedContext = reshape(attendedContext, {dimBatch, dimContext, dimTrgWords});
      
      auto rnnInputs = concatenate({trgEmbeddings, attendedContext}, axis=1);
      int dimInput = rnnInputs->shape()[1];
      
      if(!rnnL1)
        rnnL1 = New<RNN<GRU>>(graph, "decoder",
                              dimInput, dimDecState,
                              dropout_prob=dropoutRnn,
                              normalize=layerNorm);

      auto stateL1 = (*rnnL1)(rnnInputs, stateHardAtt->getStates()[0]);
      
      std::vector<Expr> statesOut;
      statesOut.push_back(stateL1);

      Expr outputLn;
      if(decoderLayers > 1) {
        std::vector<Expr> statesIn;
        for(int i = 1; i < stateHardAtt->getStates().size(); ++i)
          statesIn.push_back(stateHardAtt->getStates()[i]);

        if(!rnnLn) 
          rnnLn = New<MLRNN<GRU>>(graph, "decoder",
                                  decoderLayers - 1,
                                  dimDecState, dimDecState,
                                  normalize=layerNorm,
                                  dropout_prob=dropoutRnn,
                                  skip=skipDepth,
                                  skip_first=skipDepth);
        
        std::vector<Expr> statesLn;
        std::tie(outputLn, statesLn) = (*rnnLn)(stateL1, statesIn);

        statesOut.insert(statesOut.end(),
                         statesLn.begin(), statesLn.end());
      }
      else {
        outputLn = stateL1;
      }

      //// 2-layer feedforward network for outputs and cost
      auto logitsL1 = Dense("ff_logit_l1", dimTrgEmb,
                            activation=act::tanh,
                            normalize=layerNorm)
                        (rnnInputs, outputLn);

      auto logitsOut = Dense("ff_logit_l2", dimTrgVoc)(logitsL1);    
      
      return New<DecoderStateHardAttCDI>(statesOut, logitsOut,
                                      stateHardAtt->getEncoderState(),
                                      stateHardAtt->getAttentionIndices(),
                                      stateHardAtt->getTargetEmbeddingIndices());
    }
    
    
    virtual std::tuple<Expr, Expr>
    groundTruth(Ptr<DecoderState> state,
                Ptr<ExpressionGraph> graph,
                Ptr<data::CorpusBatch> batch,
                size_t index) {
      using namespace keywords;

      
      // ***********************************************************************

      int dimVoc = options_->get<std::vector<int>>("dim-vocabs").back();
      int dimEmb = options_->get<int>("dim-emb");
      int dimAct = 16;
      
      auto srcBatch = batch->front();
      auto subBatch = (*batch)[index];
      int dimBatch = subBatch->batchSize();
      int dimWords = subBatch->batchWidth();
      
      std::vector<size_t> actionIndices;
      std::vector<size_t> transformedIndices;
      
      std::vector<size_t> attentionIndices(dimBatch, 0);
      std::vector<size_t> currentPos(dimBatch, 0);
      std::iota(currentPos.begin(), currentPos.end(), 0);

      for(int i = 0; i < dimWords - 1; ++i) {
        for(int j = 0; j < dimBatch; ++j) {
          size_t word = subBatch->indeces()[i * dimBatch + j];
          
          // copy from source and move attention position
          if(word == CPY_ID) {
            transformedIndices.push_back(srcBatch->indeces()[currentPos[j]]);
            actionIndices.push_back(2);
            currentPos[j] += dimBatch;
          }
          // reuse last word and move attention position
          else if(word == DEL_ID) {
            if(i == 0) {
              transformedIndices.push_back(0);
            }
            else {
              size_t prev = transformedIndices[(i - 1) * dimBatch + j];
              transformedIndices.push_back(prev);
            }
              
            actionIndices.push_back(3);
            currentPos[j] += dimBatch;
          }
          // insert target word
          else {
            transformedIndices.push_back(word);
            if(word == 0)
              actionIndices.push_back(0);
            else
              actionIndices.push_back(1);
          }
          
          attentionIndices.push_back(currentPos[j]);
        }
      }
      
      // ***********************************************************************
      
      //for(auto i : transformedIndices)
      //  std::cerr << i << " ";
      //std::cerr << std::endl;
      
      auto yEmb = Embedding("Wemb_dec", dimVoc, dimEmb)(graph);
      auto chosenEmbeddings = rows(yEmb, transformedIndices);
    
      auto actEmb = Embedding("Wact_dec", 4, dimAct)(graph);
      auto chosenActions = rows(actEmb, actionIndices);
      
      //batch->debug();
      //debug(chosenActions, "act");
      //debug(chosenEmbeddings, "emb");
      
      chosenEmbeddings = concatenate({chosenActions, chosenEmbeddings}, axis=1);
      dimEmb += dimAct;
      
      auto y = reshape(chosenEmbeddings, {dimBatch, dimEmb, dimWords});

      auto yMask = graph->constant(shape={dimBatch, 1, dimWords},
                                   init=inits::from_vector(subBatch->mask()));
          
      auto yIdx = graph->constant(shape={(int)subBatch->indeces().size(), 1},
                                  init=inits::from_vector(subBatch->indeces()));
    
      auto yShifted = shift(y, {0, 0, 1, 0});
      
      //state->setTargetEmbeddingsIndices ???
      
      state->setTargetEmbeddings(yShifted);
      
      
      // ***********************************************************************
      
      std::dynamic_pointer_cast<DecoderStateHardAttCDI>(state)->setAttentionIndices(attentionIndices);
            
      return std::make_tuple(yMask, yIdx);
    }
    
    virtual void selectEmbeddings(Ptr<ExpressionGraph> graph,
                                  Ptr<DecoderState> state,
                                  const std::vector<size_t>& embIdx,
                                  size_t position=0) {

      //************************************************************************
      
      using namespace keywords;
      
      int dimTrgEmb = options_->get<int>("dim-emb");
      int dimTrgVoc = options_->get<std::vector<int>>("dim-vocabs").back();
      int dimAct = 16;
    
      auto stateHardAtt = std::dynamic_pointer_cast<DecoderStateHardAttCDI>(state);
      int dimSrcWords = state->getEncoderState()->getContext()->shape()[2];

      
      Expr selectedEmbs;
      if(embIdx.empty()) {
        selectedEmbs = graph->constant(shape={1, dimTrgEmb + dimAct},
                                       init=inits::zeros);
        stateHardAtt->setAttentionIndices({0});
        stateHardAtt->setTargetEmbeddingIndeces({0});
      }
      else {
        
        std::vector<size_t> transformedIdx;
        std::vector<size_t> actionIdx;
        
        for(size_t i = 0; i < embIdx.size(); ++i) {
          if(embIdx[i] == CPY_ID) {
            size_t attIndex = stateHardAtt->getAttentionIndices()[i];
            transformedIdx.push_back(stateHardAtt->getSourceWords()[attIndex]);
            actionIdx.push_back(2);
          }
          else if(embIdx[i] == DEL_ID) {
            transformedIdx.push_back(stateHardAtt->getTargetEmbeddingIndices()[i]);
            actionIdx.push_back(3);
          }
          else {
            transformedIdx.push_back(embIdx[i]);
            if(embIdx[i] == 0)
              actionIdx.push_back(0);
            else
              actionIdx.push_back(1);
          }
          
          if(embIdx[i] == CPY_ID || embIdx[i] == DEL_ID) {
            stateHardAtt->getAttentionIndices()[i]++;
            if(stateHardAtt->getAttentionIndices()[i] >= dimSrcWords)
              stateHardAtt->getAttentionIndices()[i] = dimSrcWords - 1;
          }
          
        }        
          
        auto yEmb = Embedding("Wemb_dec", dimTrgVoc, dimTrgEmb)(graph);
        selectedEmbs = rows(yEmb, transformedIdx);
      
        auto actEmb = Embedding("Wact_dec", 4, dimAct)(graph);
        auto selectedActions = rows(actEmb, actionIdx);
        
        selectedEmbs = concatenate({selectedActions, selectedEmbs}, axis=1);
        selectedEmbs = reshape(selectedEmbs,
                               {1, dimTrgEmb + dimAct, 1, (int)transformedIdx.size()});
        
        stateHardAtt->setTargetEmbeddingIndeces(transformedIdx);
      }
      
      state->setTargetEmbeddings(selectedEmbs);
    }

};

typedef EncoderDecoder<EncoderS2S, DecoderHardAttCDI> HardAttCDI;



}
