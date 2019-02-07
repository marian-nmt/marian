#pragma once
#include <algorithm>

#include "marian.h"
#include "translator/history.h"
#include "translator/scorers.h"
#include "data/factored_vocab.h"

#include "translator/helpers.h"
#include "translator/nth_element.h"

namespace marian {

class BeamSearch {
private:
  Ptr<Options> options_;
  std::vector<Ptr<Scorer>> scorers_;
  size_t beamSize_;
  Word trgEosId_{Word::NONE};
  Word trgUnkId_{Word::NONE};

public:
  BeamSearch(Ptr<Options> options,
             const std::vector<Ptr<Scorer>>& scorers,
             Word trgEosId,
             Word trgUnkId = Word::NONE)
      : options_(options),
        scorers_(scorers),
        beamSize_(options_->has("beam-size")
                      ? options_->get<size_t>("beam-size")
                      : 3),
        trgEosId_(trgEosId),
        trgUnkId_(trgUnkId) {}

  // combine new expandedPathScores and previous beams into new set of beams
  Beams toHyps(const std::vector<unsigned int> nBestKeys, // [dimBatch, beamSize] flattened -> ((batchIdx, beamHypIdx) flattened, word idx) flattened
               const std::vector<float> nBestPathScores,  // [dimBatch, beamSize] flattened
               const size_t vocabSize,
               const Beams& beams,
               const std::vector<Ptr<ScorerState /*const*/>>& states,
               const size_t beamSize,
               const bool first,
               Ptr<data::CorpusBatch /*const*/> batch) const {
    const auto dimBatch = beams.size();
    Beams newBeams(dimBatch);

    std::vector<float> align;
    if(options_->hasAndNotEmpty("alignment"))
      // Use alignments from the first scorer, even if ensemble
      align = scorers_[0]->getAlignment();

    for(size_t i = 0; i < nBestKeys.size(); ++i) { // [dimBatch, beamSize] flattened
      // Keys contains indices to vocab items in the entire beam.
      // Values can be between 0 and beamSize * vocabSize.
      const auto batchIdx = i / beamSize; // and i % beamSize is the beam hyp index
      const auto& beam = beams[batchIdx];
      auto& newBeam = newBeams[batchIdx];

      if(newBeam.size() < beam.size()) {
        const float pathScore = nBestPathScores[i];
        const auto  key       = nBestKeys[i]; // key = pathScore's location, as ((batchIdx, beamHypIdx) flattened, word idx) flattened

        // decompose key into individual indices
        WordIndex  wordIdx = (WordIndex)(key % vocabSize);
        const auto hypIdx  =            (key / vocabSize);
#if 1
        // further decompose hypIdx, taking into account that the very first entry had beam size 1
        // and compose a new hypIdx that assumes actual beamSize
        const auto keyBatchIdx   = hypIdx / (first ? 1 : beamSize);
        const auto keyBeamHypIdx = hypIdx % (first ? 1 : beamSize);
        const auto hypIdxTrans = keyBeamHypIdx * dimBatch + keyBatchIdx;
        ABORT_IF(keyBeamHypIdx >= (int)beam.size(), "Beam hyp index exceeds beam size??"); // @TODO: is this possible? Should be, but does not seem to trigger.
#else
        const auto keyBatchIdx = hypIdx / beamSize; // @REVIEW: is this actually keyBatchIdx?
        size_t keyBeamHypIdx = hypIdx % beamSize;

        auto hypIdxTrans = keyBatchIdx + keyBeamHypIdx * dimBatch;
        if(first)
          hypIdxTrans = hypIdx; // == keyBeamHypIdx + keyBatchIdx * beamSize? or was beamSize=1, and keyBeamHypIdx = 0?

        ABORT_IF(keyBeamHypIdx >= (int)beam.size(), "Beam hyp index exceeds beam size??");
        //if(keyBeamHypIdx >= (int)beam.size())  // @TODO: What is this condition? Cf. keyBeamHypIdx = hypIdx % beamSize; beamSize = max(beams[.].size())
        //  keyBeamHypIdx = keyBeamHypIdx % beam.size();

        if(first)
          keyBeamHypIdx = 0;
#endif
        // Retrieve short list for final softmax (based on words aligned
        // to source sentences). If short list has been set, map the indices
        // in the sub-selected vocabulary matrix back to their original positions.
        auto shortlist = scorers_[0]->getShortlist();
        if(shortlist)
          wordIdx = shortlist->reverseMap(wordIdx); // @TODO: should reverseMap accept a size_t or a Word?
        // now wordIdx is a regular Word again

        auto hyp = New<Hypothesis>(beam[keyBeamHypIdx], Word::fromWordIndex(wordIdx), (IndexType)hypIdxTrans, pathScore);

        // Set score breakdown for n-best lists
        if(options_->get<bool>("n-best")) {
          std::vector<float> breakDown(states.size(), 0);
          beam[keyBeamHypIdx]->getScoreBreakdown().resize(states.size(), 0); // @TODO: Why? Can we just guard the read-out below, then make it const? Or getScoreBreakdown(j)?
          for(size_t j = 0; j < states.size(); ++j) {
            size_t key1 = hypIdxTrans * vocabSize + wordIdx;
            breakDown[j] = states[j]->breakDown(key1) + beam[keyBeamHypIdx]->getScoreBreakdown()[j];
          }
          hyp->setScoreBreakdown(breakDown);
        }

        // Set alignments
        if(!align.empty()) {
          hyp->setAlignment(getAlignmentsForHypothesis(align, batch, (int)keyBeamHypIdx, (int)batchIdx));
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
    // @TODO: EOS id does not need to be stored in this object, since it is available from vocab()
    ABORT_IF(batch->back()->vocab()->getEosId() != trgEosId_, "Batch uses different EOS token than was passed to BeamSearch originally");

    auto factoredVocab = batch->back()->vocab()->tryAs<FactoredVocab>();
    size_t numFactors = factoredVocab ? factoredVocab->getNumGroups() : 1;
    numFactors;

    const int dimBatch = (int)batch->size();

    auto getNBestList = createGetNBestListFn(beamSize_, dimBatch, graph->getDeviceId());

    for(auto scorer : scorers_) {
      scorer->clear(graph);
    }

    Histories histories(dimBatch);
    for(int i = 0; i < dimBatch; ++i) {
      size_t sentId = batch->getSentenceIds()[i];
      histories[i] = New<History>(sentId,
                                  options_->get<float>("normalize"),
                                  options_->get<float>("word-penalty"));
    }

    // start states
    std::vector<Ptr<ScorerState>> states;
    for(auto scorer : scorers_) {
      states.push_back(scorer->startState(graph, batch));
    }

    Beams beams(dimBatch, Beam(beamSize_, New<Hypothesis>())); // array [dimBatch] of array [localBeamSize] of Hypothesis
    //Beams beams(dimBatch); // array [dimBatch] of array [localBeamSize] of Hypothesis
    //for(auto& beam : beams)
    //  beam.resize(beamSize_, New<Hypothesis>());

    for(int i = 0; i < dimBatch; ++i)
      histories[i]->add(beams[i], trgEosId_);

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
    //     - gets replaced at the end of each output time step

    // main loop over output time steps
    for (size_t t = 0; ; t++) {
      ABORT_IF(dimBatch != beams.size(), "Lost a batch entry??");
      // determine beam size for next output time step, as max over still-active sentences
      // E.g. if all batch entries are down from beam 5 to no more than 4 surviving hyps, then
      // switch to beam of 4 for all. If all are done, then beam ends up being 0, and we are done.
      size_t localBeamSize = 0; // @TODO: is there some std::algorithm for this?
      for(auto& beam : beams)
        if(beam.size() > localBeamSize)
          localBeamSize = beam.size();

      // done if all batch entries have reached EOS on all beam entries
      if (localBeamSize == 0)
        break;

      //**********************************************************************
      // create constant containing previous path scores for current beam
      // Also create mapping of hyp indices, which are not 1:1 if sentences complete.
      std::vector<IndexType> hypIndices; // [localBeamsize, 1, dimBatch, 1] (flattened) index of hyp that the new top N originated from
      std::vector<Word> prevWords;       // [localBeamsize, 1, dimBatch, 1] (flattened) predecessor word
      Expr prevPathScores;               // [localBeamSize, 1, dimBatch, 1], where the last axis broadcasts into vocab size when adding expandedPathScores
      if(t == 0) { // no scores yet
        prevPathScores = graph->constant({1, 1, 1, 1}, inits::from_value(0));
      } else {
        std::vector<float> prevScores;
        for(size_t beamIndex = 0; beamIndex < localBeamSize; ++beamIndex) {
          for(int batchIndex = 0; batchIndex < dimBatch; ++batchIndex) { // loop over batch entries (active sentences)
            auto& beam = beams[batchIndex];
            if(beamIndex < beam.size()) {
              auto hyp = beam[beamIndex];
              hypIndices.push_back((IndexType)hyp->getPrevStateIndex()); // index where to find prev hyp (beamHypIdx, batchIdx), =beamHypIdx * dimBatch + batchIdx
              prevWords .push_back(hyp->getWord());
              prevScores.push_back(hyp->getPathScore());
            } else {  // pad to localBeamSize (dummy hypothesis)
              hypIndices.push_back(0);
              prevWords.push_back(Word::ZERO);  // (unused)
              prevScores.push_back(-9999);
            }
          }
        }
        prevPathScores = graph->constant({(int)localBeamSize, 1, dimBatch, 1},
                                    inits::from_vector(prevScores));
      }

      //**********************************************************************
      // compute expanded path scores with word prediction probs from all scorers
      auto expandedPathScores = prevPathScores; // will become [localBeamSize, 1, dimBatch, dimVocab]
      for(size_t i = 0; i < scorers_.size(); ++i) {
        // compute output probabilities for current output time step
        //  - uses hypIndices[index in beam, 1, batch index, 1] to reorder hypotheses
        //  - adds prevWords [index in beam, 1, batch index, 1] to the decoder model's target history
        //  - performs one step of the decoder model
        //  - returns new NN state for use in next output time step
        //  - returns vector of prediction probabilities over output vocab via newState
        auto newState = scorers_[i]->step(
            graph, states[i], hypIndices, prevWords, dimBatch, (int)localBeamSize);

        // expand all hypotheses, [localBeamSize, 1, dimBatch, 1] -> [localBeamSize, 1, dimBatch, dimVocab]
        expandedPathScores = expandedPathScores + scorers_[i]->getWeight() * newState->getLogProbs().getLogits();

        // update state in-place for next output time step
        states[i] = newState;
      }

      // make beams continuous
      if(dimBatch > 1 && localBeamSize > 1)
        expandedPathScores = swapAxes(expandedPathScores, 0, 2); // -> [dimBatch, 1, localBeamSize, dimVocab]
      //  expandedPathScores = transpose(expandedPathScores, {2, 1, 0, 3}); // -> [dimBatch, 1, localBeamSize, dimVocab]

      // perform NN computation
      if(t == 0)
        graph->forward();
      else
        graph->forwardNext();

      //**********************************************************************
      // suppress specific symbols if not at right positions
      if(trgUnkId_ != Word::NONE && options_->has("allow-unk") && !options_->get<bool>("allow-unk"))
        suppressWord(expandedPathScores, trgUnkId_);
      for(auto state : states)
        state->blacklist(expandedPathScores, batch);

      //**********************************************************************
      // perform beam search

      // find N best amongst the (localBeamSize * dimVocab) hypotheses
      std::vector<unsigned int> nBestKeys; // [dimBatch, localBeamSize] flattened -> ((batchIdx, beamHypIdx) flattened, word idx) flattened
      std::vector<float> nBestPathScores;  // [dimBatch, localBeamSize] flattened
      getNBestList(/*beamSizes=*/std::vector<size_t>(dimBatch, localBeamSize), // output layout of (nBestPathScores, nBestKeys)  --@REVIEW: correct?
                   /*in*/ expandedPathScores->val(),                           // [dimBatch, 1, localBeamSize, dimVocab or dimShortlist]
                   /*out*/ nBestPathScores, /*out*/ nBestKeys,
                   /*first=*/t == 0); // @TODO: Why is this passed? To know that the beam size is 1 for first step, for flattened hyp index?
      // Now, nBestPathScores contain N-best expandedPathScores, and nBestKeys for each their original location (batchIdx, beamHypIdx, word).

      // combine N-best sets with existing search space (beams) to updated search space
      beams = toHyps(nBestKeys, nBestPathScores,
                     /*dimTrgVoc=*/expandedPathScores->shape()[-1],
                     beams,
                     states,           // used for keeping track of per-ensemble-member path score
                     localBeamSize,    // used in the encoding of the (batchIdx, beamHypIdx, word) tuples
                     /*first=*/t == 0, // used to indicate originating beamSize of 1
                     batch);

      // remove all hyps that end in EOS
      // The position of a hyp in the beam may change.
      const auto purgedBeams = purgeBeams(beams);

      // add updated search space (beams) to search grid (histories) for traceback
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
    } // end of main loop over output time steps

    return histories; // [dimBatch][t][N best hyps]
  }
};
}  // namespace marian
