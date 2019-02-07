#pragma once
#include <algorithm>

#include "marian.h"
#include "translator/history.h"
#include "translator/scorers.h"

#include "translator/helpers.h"
#include "translator/nth_element.h"

namespace marian {

class BeamSearch {
private:
  Ptr<Options> options_;
  std::vector<Ptr<Scorer>> scorers_;
  size_t beamSize_;
  Word trgEosId_ = (Word)-1;
  Word trgUnkId_ = (Word)-1;

public:
  BeamSearch(Ptr<Options> options,
             const std::vector<Ptr<Scorer>>& scorers,
             Word trgEosId,
             Word trgUnkId = -1)
      : options_(options),
        scorers_(scorers),
        beamSize_(options_->has("beam-size")
                      ? options_->get<size_t>("beam-size")
                      : 3),
        trgEosId_(trgEosId),
        trgUnkId_(trgUnkId) {}

  Beams toHyps(const std::vector<unsigned int> keys,
               const std::vector<float> pathScores,
               size_t vocabSize,
               const Beams& beams,
               const std::vector<Ptr<ScorerState>>& states,
               size_t beamSize,
               bool first,
               Ptr<data::CorpusBatch> batch) const {
    Beams newBeams(beams.size());

    std::vector<float> align;
    if(options_->hasAndNotEmpty("alignment"))
      // Use alignments from the first scorer, even if ensemble
      align = scorers_[0]->getAlignment();

    for(size_t i = 0; i < keys.size(); ++i) { // keys: [beamSize, ?] (flattened)
      // Keys contains indices to vocab items in the entire beam.
      // Values can be between 0 and beamSize * vocabSize.
      auto beamIdx = i / beamSize;

      if(newBeams[beamIdx].size() < beams[beamIdx].size()) {
        Word wordIdx = (Word)(keys[i] % vocabSize);
        // Retrieve short list for final softmax (based on words aligned
        // to source sentences). If short list has been set, map the indices
        // in the sub-selected vocabulary matrix back to their original positions.
        auto shortlist = scorers_[0]->getShortlist();
        if(shortlist)
          wordIdx = shortlist->reverseMap(wordIdx); // @TODO: should reverseMap accept a size_t or a Word?

        const auto& beam = beams[beamIdx];
        auto& newBeam = newBeams[beamIdx];

        const float pathScore = pathScores[i];

        // keys[i] = offset into row-major cube of dims [whatIsThis, beamSize, vocabSize]
        // deconstruct into individual indices
        const auto hypIdx = (IndexType)(keys[i] / vocabSize);
        const auto whatIsThis = (hypIdx / beamSize); // @TODO: is this batchIdx?
        size_t beamHypIdx = hypIdx % beamSize;

        auto hypIdxTrans = IndexType(whatIsThis + beamHypIdx * beams.size());
        if(first)
          hypIdxTrans = hypIdx;

        if(beamHypIdx >= (int)beam.size())  // @TODO: What is this condition? Cf. beamHypIdx = hypIdx % beamSize
          beamHypIdx = beamHypIdx % beam.size();

        if(first)
          beamHypIdx = 0;

        auto hyp = New<Hypothesis>(beam[beamHypIdx], wordIdx, hypIdxTrans, pathScore);

        // Set score breakdown for n-best lists
        if(options_->get<bool>("n-best")) {
          std::vector<float> breakDown(states.size(), 0);
          beam[beamHypIdx]->GetScoreBreakdown().resize(states.size(), 0);
          for(size_t j = 0; j < states.size(); ++j) {
            size_t key = wordIdx + hypIdxTrans * vocabSize;
            breakDown[j] = states[j]->breakDown(key)
                           + beam[beamHypIdx]->GetScoreBreakdown()[j];
          }
          hyp->GetScoreBreakdown() = breakDown;
        }

        // Set alignments
        if(!align.empty()) {
          hyp->SetAlignment(
              getAlignmentsForHypothesis(align, batch, (int)beamHypIdx, (int)beamIdx));
        }

        newBeam.push_back(hyp);
      }
    }
    return newBeams;
  }

  std::vector<float> getAlignmentsForHypothesis(
      const std::vector<float> alignAll,
      Ptr<data::CorpusBatch> batch,
      int beamHypIdx,
      int beamIdx) const {
    // Let's B be the beam size, N be the number of batched sentences,
    // and L the number of words in the longest sentence in the batch.
    // The alignment vector:
    //
    // if(first)
    //   * has length of N x L if it's the first beam
    //   * stores elements in the following order:
    //     beam1 = [word1-batch1, word1-batch2, ..., word2-batch1, ...]
    // else
    //   * has length of N x L x B
    //   * stores elements in the following order:
    //     beams = [beam1, beam2, ..., beam_n]
    //
    // The mask vector is always of length N x L and has 1/0s stored like
    // in a single beam, i.e.:
    //   * [word1-batch1, word1-batch2, ..., word2-batch1, ...]
    //
    size_t batchSize = batch->size();
    size_t batchWidth = batch->width() * batchSize;
    std::vector<float> align;

    for(size_t w = 0; w < batchWidth / batchSize; ++w) {
      size_t a = ((batchWidth * beamHypIdx) + beamIdx) + (batchSize * w);
      size_t m = a % batchWidth;
      if(batch->front()->mask()[m] != 0)
        align.emplace_back(alignAll[a]);
    }

    return align;
  }

  // remove all beam entries that have reached EOS
  Beams purgeBeams(const Beams& beams) {
    Beams newBeams;
    for(auto beam : beams) {
      Beam newBeam;
      for(auto hyp : beam) {
        if(hyp->getWord() != trgEosId_) {
          newBeam.push_back(hyp);
        }
      }
      newBeams.push_back(newBeam);
    }
    return newBeams;
  }

  //**********************************************************************
  // main decoding function
  Histories search(Ptr<ExpressionGraph> graph, Ptr<data::CorpusBatch> batch) {
    int dimBatch = (int)batch->size();

    Histories histories(dimBatch);
    for(int i = 0; i < dimBatch; ++i) {
      size_t sentId = batch->getSentenceIds()[i];
      histories[i] = New<History>(sentId,
                                  options_->get<float>("normalize"),
                                  options_->get<float>("word-penalty"));
    }

    size_t localBeamSize = beamSize_; // max over beam sizes of active sentence hypotheses

    auto getNBestList = createGetNBestListFn(localBeamSize, dimBatch, graph->getDeviceId());

    Beams beams(dimBatch);        // array [dimBatch] of array [localBeamSize] of Hypothesis
    for(auto& beam : beams)
      beam.resize(localBeamSize, New<Hypothesis>());

    for(int i = 0; i < dimBatch; ++i)
      histories[i]->add(beams[i], trgEosId_);

    std::vector<Ptr<ScorerState>> states;

    for(auto scorer : scorers_) {
      scorer->clear(graph);
    }

    for(auto scorer : scorers_) {
      states.push_back(scorer->startState(graph, batch));
    }

    // the decoder maintains the following state:
    //  - histories : array [dimBatch] of History
    //    with History : vector [t] of array [localBeamSize] of Hypothesis
    //    with Hypothesis : (last word, aggregate score, prev Hypothesis)
    //     - search grid
    //     - stores traceback information
    //     - gets added to in each output time step
    //     - the final version is the return value of this function
    //  - beams : array [dimBatch] of  array [localBeamSize] of Hypothesis
    //     - current output time step's set of active hypotheses, aka active search space
    //     - gets replaced at the end of each output time step
    //  - states[.] : ScorerState
    //     - NN state
    //     - one per scorer, e.g. 2 for ensemble of 2

    // main loop over output time steps
    for(bool first = true; ; first = false) {
      //**********************************************************************
      // create constant containing previous path scores for current beam
      // Also create mapping of hyp indices, which are not 1:1 if sentences complete.
      std::vector<IndexType> hypIndices; // [localBeamsize, 1, dimBatch, 1] (flattened) index of beam index that each of the new top N originated from
      std::vector<Word> prevWords;      // [localBeamsize, 1, dimBatch, 1] (flattened) predecessor word
      Expr prevPathScores;               // [localBeamSize, 1, dimBatch, 1], where the last axis broadcasts into vocab size when adding pathScores
      if(first) {
        // no scores yet
        prevPathScores = graph->constant({1, 1, 1, 1}, inits::from_value(0));
      } else {
        dimBatch = (int)batch->size();
        ABORT_IF(dimBatch != beams.size(), "Dimensions mismatch??");

        std::vector<float> beamScores;
        for(size_t beamIndex = 0; beamIndex < localBeamSize; ++beamIndex) {
          for(int batchIndex = 0; batchIndex < dimBatch; ++batchIndex) { // loop over batch entries (active sentences)
            auto& beam = beams[batchIndex];
            if(beamIndex < beam.size()) {
              auto hyp = beam[beamIndex];
              hypIndices.push_back((IndexType)hyp->getPrevStateIndex()); // backpointer
              prevWords .push_back(hyp->getWord());
              beamScores.push_back(hyp->getPathScore());
            } else {  // dummy hypothesis
              hypIndices.push_back(0);
              prevWords .push_back(Word{});  // (unused)
              beamScores.push_back(-9999);
            }
          }
        }

        prevPathScores = graph->constant({(int)localBeamSize, 1, dimBatch, 1},
                                    inits::from_vector(beamScores));
      }

      //**********************************************************************
      // prepare scores for beam search
      auto pathScores = prevPathScores;

      for(size_t i = 0; i < scorers_.size(); ++i) {
        // compute output probabilities for current output time step
        //  - uses hypIndices[index in beam, 1, batch index, 1] and embIndices[index in beam, 1, batch index, 1] to reorder hypotheses
        //  - returns new NN state for use in next output time step
        //  - returns vector of prediction probabilities over output vocab via newState
        auto newState = scorers_[i]->step(
            graph, states[i], hypIndices, prevWords, dimBatch, (int)localBeamSize);

        // expand all hypotheses, [localBeamSize, 1, dimBatch, 1] -> [localBeamSize, 1, dimBatch, dimVocab]
        pathScores = pathScores + scorers_[i]->getWeight() * newState->getLogProbs();

        // update state in-place for next output time step
        states[i] = newState;
      }

      // make beams continuous
      if(dimBatch > 1 && localBeamSize > 1)
        pathScores = transpose(pathScores, {2, 1, 0, 3}); // -> [dimBatch, 1, localBeamSize, dimVocab]

      if(first)
        graph->forward();
      else
        graph->forwardNext();

      //**********************************************************************
      // suppress specific symbols if not at right positions
      if(trgUnkId_ != -1 && options_->has("allow-unk")
         && !options_->get<bool>("allow-unk"))
        suppressWord(pathScores, trgUnkId_);
      for(auto state : states)
        state->blacklist(pathScores, batch);

      //**********************************************************************
      // perform beam search and pruning

      // find N best amongst the (localBeamSize * dimVocab) hypotheses
      const std::vector<size_t> beamSizes(dimBatch, localBeamSize);
      std::vector<unsigned int> outKeys;
      std::vector<float> outPathScores;
      getNBestList(beamSizes, pathScores->val(), outPathScores, outKeys, first);
      // outPathScores and outKeys contain pathScores and their original indices in N-best order

      // convert N-best sets to updated search space
      int dimTrgVoc = pathScores->shape()[-1];
      beams = toHyps(outKeys,
                     outPathScores,
                     dimTrgVoc,
                     beams,
                     states,
                     localBeamSize,
                     first,
                     batch);

      // remove all hyps that end in EOS
      auto purgedBeams = purgeBeams(beams); // @TODO: rename; this is not pruning

      // add updated search space to search grid for traceback
      bool maxLengthReached = false;
      for(int i = 0; i < dimBatch; ++i) {
        // if this batch entry has surviving hyps then add them to the traceback grid
        if(!beams[i].empty()) {
          if (histories[i]->size() >= options_->get<float>("max-length-factor") * batch->front()->batchWidth())
            maxLengthReached = true;
          histories[i]->add(beams[i], trgEosId_, purgedBeams[i].empty() || maxLengthReached);
        }
      }
      if (maxLengthReached) // early exit if max length limit was reached
        break;

      // this is the search space for the next output time step
      beams = purgedBeams;

      // determine beam size for next output time step, as max over still-active sentences
      // E.g. if all batch entries are down from beam 5 to no more than 4 surviving hyps, then
      // switch to beam of 4 for all. If all are done, then beam ends up being 0, and we are done.
      if(!first) {
        size_t maxBeam = 0;
        for(auto& beam : beams)
          if(beam.size() > maxBeam)
            maxBeam = beam.size();
        localBeamSize = maxBeam;
      }
      if (localBeamSize == 0) // done if all batch entries have reached EOS on all beam entries
        break;
    } // end of main loop over output tokens

    return histories; // [dimBatch][t][N best hyps]
  }
};
}  // namespace marian
