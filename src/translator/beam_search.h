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

  static constexpr auto INVALID_PATH_SCORE = -9999;// .0f;

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
               Ptr<data::CorpusBatch /*const*/> batch, // for alignments only
               Ptr<FactoredVocab/*const*/> factoredVocab, size_t factorGroup) const {
    std::vector<float> align;
    if(options_->hasAndNotEmpty("alignment"))
      align = scorers_[0]->getAlignment(); // use alignments from the first scorer, even if ensemble

    const auto dimBatch = beams.size();
    Beams newBeams(dimBatch);

    for(size_t i = 0; i < nBestKeys.size(); ++i) { // [dimBatch, beamSize] flattened
      // Keys contains indices to vocab items in the entire beam.
      // Values can be between 0 and beamSize * vocabSize.
      const float pathScore = nBestPathScores[i];
      const auto  key       = nBestKeys[i]; // key = pathScore's tensor location, as (batchIdx, beamHypIdx, word idx) flattened

      // decompose key into individual indices (batchIdx, beamHypIdx, wordIdx)
      const auto wordIdx    = (WordIndex)(key % vocabSize);
      const auto beamHypIdx =            (key / vocabSize) % (first ? 1 : beamSize);
      const auto batchIdx   =            (key / vocabSize) / (first ? 1 : beamSize);
      //LOG(info, "key = (batch {}, beam {}, word {})", batchIdx, beamHypIdx, wordIdx);

      ABORT_IF(i / beamSize != batchIdx, "Inconsistent batchIdx value in key??");

      const auto& beam = beams[batchIdx];
      auto& newBeam = newBeams[batchIdx];

      if (newBeam.size() >= beam.size()) // @TODO: Why this condition? It does happen. Why?
        continue;

      if (pathScore <= INVALID_PATH_SCORE) // (dummy slot or word that cannot be expanded by current factor)
        continue;
      //if (beamHypIdx >= (int)beam.size() && pathScore <= INVALID_PATH_SCORE) // (dummy slot)
      //  continue;
      ABORT_IF(beamHypIdx >= (int)beam.size(), "Out of bounds beamHypIdx value {} in key?? word={}, batch={}, pathScore={}", beamHypIdx, wordIdx, batchIdx, pathScore);

      // map wordIdx to word
      auto prevHyp = beam[beamHypIdx];
      auto prevBeamHypIdx = beamHypIdx;
      Word word;
      // If short list has been set, then wordIdx is an index into the short-listed word set,
      // rather than the true word index.
      auto shortlist = scorers_[0]->getShortlist();
      if (shortlist)
        word = Word::fromWordIndex(shortlist->reverseMap(wordIdx));
      else if (factoredVocab) {
        // For factored decoding, the word is built over multiple decoding steps,
        // starting with the lemma, then adding factors one by one.
        if (factorGroup == 0) {
          word = factoredVocab->lemma2Word(wordIdx);
          //LOG(info, "new lemma {}={}", word.toWordIndex(), factoredVocab->word2string(word));
        }
        else {
          //LOG(info, "expand word {}={} with factor[{}] {}", beam[beamHypIdx]->getWord().toWordIndex(),
          //    factoredVocab->word2string(beam[beamHypIdx]->getWord()), factorGroup, wordIdx);
          word = beam[beamHypIdx]->getWord();
          ABORT_IF(!factoredVocab->canExpandFactoredWord(word, factorGroup), "A word without this factor snuck through to here??");
          //if (factoredVocab->canExpandFactoredWord(word, factorGroup))
            word = factoredVocab->expandFactoredWord(word, factorGroup, wordIdx);
          // @TODO: maybe factor the two above into a single function; for now, I want the extra checks
          //else
          //  continue; // skip if word does not have this factor
            //ABORT_IF(prevHyp->getPathScore() != pathScore, "Score changed despite factor not applicable??");
          prevBeamHypIdx = prevHyp->getPrevStateIndex();
          prevHyp = prevHyp->getPrevHyp(); // short-circuit the backpointer, so that the traceback doesnot contain partially factored words
        }
      }
      else
        word = Word::fromWordIndex(wordIdx);

      auto hyp = New<Hypothesis>(prevHyp, word, prevBeamHypIdx, pathScore);

      // Set score breakdown for n-best lists
      if(options_->get<bool>("n-best")) {
        std::vector<float> breakDown(states.size(), 0);
        beam[beamHypIdx]->getScoreBreakdown().resize(states.size(), 0); // @TODO: Why? Can we just guard the read-out below, then make it const? Or getScoreBreakdown(j)?
        for(size_t j = 0; j < states.size(); ++j) {
          size_t flattenedLogitIndex = (beamHypIdx * dimBatch + batchIdx) * vocabSize + wordIdx;  // (beam idx, batch idx, word idx); note: beam and batch are transposed, compared to 'key'
          breakDown[j] = states[j]->breakDown(flattenedLogitIndex) + beam[beamHypIdx]->getScoreBreakdown()[j];
          // @TODO: pass those 3 indices directly into breakDown (state knows the dimensions)
        }
        hyp->setScoreBreakdown(breakDown);
      }

      // Set alignments
      if(!align.empty()) {
        hyp->setAlignment(getAlignmentsForHypothesis(align, batch, (int)beamHypIdx, (int)batchIdx));
      }

      newBeam.push_back(hyp);
    }

    // also propagate factored hypotheses that do not get expanded in this step as they don't have this factor
    if (factorGroup > 0) {
      for (size_t batchIdx = 0; batchIdx < beams.size(); batchIdx++) {
        const auto& beam = beams[batchIdx];
        auto& newBeam = newBeams[batchIdx];
        for (const auto& beamHyp : beam) {
          auto word = beamHyp->getWord();
          //LOG(info, "Checking {}", factoredVocab->word2string(word));
          if (factoredVocab->canExpandFactoredWord(word, factorGroup)) // handled above
            continue;
          //LOG(info, "Forwarded {}", factoredVocab->word2string(word));
          newBeam.push_back(beamHyp);
        }
        if (newBeam.size() > beamSize) {
          //LOG(info, "Size {}, sorting...", newBeam.size());
          std::nth_element(newBeam.begin(), newBeam.begin() + beamSize, newBeam.end(), [](Ptr<Hypothesis> a, Ptr<Hypothesis> b) {
            return a->getPathScore() > b->getPathScore(); // (sort highest score first)
          });
          //LOG(info, "Size {}, sorted...", newBeam.size());
          newBeam.resize(beamSize); // @TODO: needed?
        }
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
#if 0   // use '1' here to disable factored decoding, e.g. for comparisons
    factoredVocab.reset();
#endif
    size_t numFactorGroups = factoredVocab ? factoredVocab->getNumGroups() : 1;
    if (numFactorGroups == 1) // if no factors then reset
      factoredVocab.reset();

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
    //  - beams : array [dimBatch] of  array [localBeamSize] of Hypothesis
    //     - current output time step's set of active hypotheses, aka active search space
    //     - gets replaced at the end of each output time step
    //  - states[.] : ScorerState
    //     - NN state
    //     - one per scorer, e.g. 2 for ensemble of 2
    //     - gets replaced in each output time step
    // and it forms the following return value
    //  - histories : array [dimBatch] of History
    //    with History : vector [t] of array [localBeamSize] of Hypothesis
    //    with Hypothesis : (last word, aggregate score, prev Hypothesis)
    //     - search grid
    //     - stores traceback information
    //     - gets added to in each output time step
    //     - the final version is the return value of this function

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

      for (size_t factorGroup = 0; factorGroup < numFactorGroups; factorGroup++) {
      // BEGIN FOR factorGroup = 0 .. numFactorGroups-1
      // @TODO: use an explicit nested loop here for factors
      // for factored vocabs, we do one factor at a time, but without updating the scorer for secondary factors
      //auto factorGroup = t % numFactorGroups;

      //**********************************************************************
      // create constant containing previous path scores for current beam
      // Also create mapping of hyp indices, for reordering the decoder-state tensors.
      std::vector<IndexType> hypIndices; // [localBeamsize, 1, dimBatch, 1] (flattened) tensor index ((beamHypIdx, batchIdx), flattened) of prev hyp that a hyp originated from
      std::vector<Word> prevWords;       // [localBeamsize, 1, dimBatch, 1] (flattened) word that a hyp ended in, for advancing the decoder-model's history
      Expr prevPathScores;               // [localBeamSize, 1, dimBatch, 1], path score that a hyp ended in (last axis will broadcast into vocab size when adding expandedPathScores)
      std::vector<float> prevScores; // @TODO: remove here again
      if(t == 0 && factorGroup == 0) { // no scores yet
        prevPathScores = graph->constant({1, 1, 1, 1}, inits::from_value(0));
      } else {
        //std::vector<float> prevScores;
        for(size_t beamHypIdx = 0; beamHypIdx < localBeamSize; ++beamHypIdx) {
          for(int batchIdx = 0; batchIdx < dimBatch; ++batchIdx) { // loop over batch entries (active sentences)
            auto& beam = beams[batchIdx];
            if(beamHypIdx < beam.size()) {
              auto hyp = beam[beamHypIdx];
              hypIndices.push_back((IndexType)(hyp->getPrevStateIndex() * dimBatch + batchIdx)); // (beamHypIdx, batchIdx), flattened, for index_select() operation
              prevWords .push_back(hyp->getWord());
              prevScores.push_back((factoredVocab && !factoredVocab->canExpandFactoredWord(hyp->getWord(), factorGroup)) ? INVALID_PATH_SCORE : hyp->getPathScore());
            } else {  // pad to localBeamSize (dummy hypothesis)
              hypIndices.push_back(0);
              prevWords.push_back(trgEosId_);  // (unused, but must be valid)
              prevScores.push_back(INVALID_PATH_SCORE);
            }
          }
        }
        prevPathScores = graph->constant({(int)localBeamSize, 1, dimBatch, 1}, inits::from_vector(prevScores));
      }

      //**********************************************************************
      // compute expanded path scores with word prediction probs from all scorers
      auto expandedPathScores = prevPathScores; // will become [localBeamSize, 1, dimBatch, dimVocab]
      Expr logProbs;
      for(size_t i = 0; i < scorers_.size(); ++i) {
        if (factorGroup == 0) {
          // compute output probabilities for current output time step
          //  - uses hypIndices[index in beam, 1, batch index, 1] to reorder scorer state to reflect the top-N in beams[][]
          //  - adds prevWords [index in beam, 1, batch index, 1] to the scorer's target history
          //  - performs one step of the scorer
          //  - returns new NN state for use in next output time step
          //  - returns vector of prediction probabilities over output vocab via newState
          // update state in-place for next output time step
          //if (t > 0) for (size_t kk = 0; kk < prevWords.size(); kk++)
          //  LOG(info, "prevWords[{},{}]={} -> {}", t/numFactorGroups, factorGroup,
          //      factoredVocab ? factoredVocab->word2string(prevWords[kk]) : (*batch->back()->vocab())[prevWords[kk]],
          //      prevScores[kk]);
          states[i] = scorers_[i]->step(graph, states[i], hypIndices, prevWords, dimBatch, (int)localBeamSize);
          if (numFactorGroups == 1)
            logProbs = states[i]->getLogProbs().getLogits(); // [localBeamSize, 1, dimBatch, dimVocab]
          else
            logProbs = states[i]->getLogProbs().getFactoredLogits(factorGroup); // [localBeamSize, 1, dimBatch, dimVocab]
          //logProbs->debug("logProbs");
        }
        else {
          // add secondary factors
          // For those, we don't update the decoder-model state in any way.
          // Instead, we just keep expanding with the factors.
          // Considerations:
          //  - not all scores should get a factor
          //    We need a [localBeamSize, 1, dimBatch, 1] tensor that knows whether a factor is applicable
          //    by considering the lemma at each (beamHypIdx, batchIdx). prevWords is already in the right order.
          //  - factors are incorporated one step at a time; so we will have temporary Word entries
          //    in hyps with some factors set to FACTOR_NOT_SPECIFIED.
          logProbs = states[i]->getLogProbs().getFactoredLogits(factorGroup, hypIndices, localBeamSize); // [localBeamSize, 1, dimBatch, dimVocab]
          // Note: pathScores was set to INVALID_PATH_SCORE if this path cannot expanded by this factor; toHyps() handles that special case on the side
          //for (size_t kk = 0; kk < prevWords.size(); kk++)
          //  LOG(info, "prevWords[{},{}]={} -> {}", t/numFactorGroups, factorGroup, factoredVocab->word2string(prevWords[kk]), prevScores[kk]);
          //auto factorMaskVector = states[i]->getLogProbs().getFactorMasks(prevWords, factorGroup);
          //for (auto& m : factorMaskVector)
          //  m = m ? 0.f : INVALID_PATH_SCORE; // block hyps that do not have the factor; these are short-circuited directly
          //auto logFactorMasks = graph->constant({(int)localBeamSize, 1, dimBatch, 1}, inits::from_vector(factorMaskVector));
          //logProbs = logProbs + logFactorMasks; // those hyps that don't have a factor get multiplied with 0
        }
        // expand all hypotheses, [localBeamSize, 1, dimBatch, 1] -> [localBeamSize, 1, dimBatch, dimVocab]
        expandedPathScores = expandedPathScores + scorers_[i]->getWeight() * logProbs;
        //logProbs->debug("logProbs");
      }

      // make beams continuous
      if(dimBatch > 1 && localBeamSize > 1)
        expandedPathScores = swapAxes(expandedPathScores, 0, 2); // -> [dimBatch, 1, localBeamSize, dimVocab]
      else // (avoid copy if we can)
        expandedPathScores = reshape(expandedPathScores, {dimBatch, 1, (int)localBeamSize, expandedPathScores->shape()[-1]}); // -> [dimBatch, 1, localBeamSize, dimVocab]
      //  expandedPathScores = transpose(expandedPathScores, {2, 1, 0, 3}); // -> [dimBatch, 1, localBeamSize, dimVocab]
      //expandedPathScores->debug("expandedPathScores");

      // perform NN computation
      if(t == 0 && factorGroup == 0)
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
      std::vector<unsigned int> nBestKeys; // [dimBatch, localBeamSize] flattened -> (batchIdx, beamHypIdx, word idx) flattened
      std::vector<float> nBestPathScores;  // [dimBatch, localBeamSize] flattened
      // @TODO: getNBestList() API is redundant; input dimensions are known from expandedPathScores(); but no way to specify target N different from input N
      getNBestList(/*beamSizes=*/std::vector<size_t>(dimBatch, localBeamSize), // output layout of (nBestPathScores, nBestKeys)  --@REVIEW: correct?
                   /*in*/ expandedPathScores->val(),                           // [dimBatch, 1, localBeamSize, dimVocab or dimShortlist]
                   /*out*/ nBestPathScores, /*out*/ nBestKeys,
                   /*first=*/t == 0 && factorGroup == 0); // @TODO: Why is this passed? To know that the beam size is 1 for first step, for flattened hyp index?
      // Now, nBestPathScores contain N-best expandedPathScores, and nBestKeys for each their original location (batchIdx, beamHypIdx, word).

      // combine N-best sets with existing search space (beams) to updated search space
      beams = toHyps(nBestKeys, nBestPathScores,
                     /*dimTrgVoc=*/expandedPathScores->shape()[-1],
                     beams,
                     states,           // used for keeping track of per-ensemble-member path score
                     localBeamSize,    // used in the encoding of the (batchIdx, beamHypIdx, word) tuples
                     /*first=*/t == 0 && factorGroup == 0, // used to indicate originating beamSize of 1
                     batch, factoredVocab, factorGroup);
      } // END FOR factorGroup = 0 .. numFactorGroups-1

      // remove all hyps that end in EOS
      // The position of a hyp in the beam may change.
      const auto purgedNewBeams = purgeBeams(beams);

      // add updated search space (beams) to our return value
      bool maxLengthReached = false;
      for(int i = 0; i < dimBatch; ++i) {
        // if this batch entry has surviving hyps then add them to the traceback grid
        if(!beams[i].empty()) {
          if (histories[i]->size() >= options_->get<float>("max-length-factor") * batch->front()->batchWidth())
            maxLengthReached = true;
          histories[i]->add(beams[i], trgEosId_, purgedNewBeams[i].empty() || maxLengthReached);
        }
      }
      if (maxLengthReached) // early exit if max length limit was reached
        break;

      // this is the search space for the next output time step
      beams = purgedNewBeams;
    } // end of main loop over output time steps

    return histories; // [dimBatch][t][N best hyps]
  }
};
}  // namespace marian
