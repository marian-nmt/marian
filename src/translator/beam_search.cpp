#include "translator/beam_search.h"

#include "data/factored_vocab.h"
#include "translator/helpers.h"
#include "translator/nth_element.h"
#include "data/shortlist.h"
#include "common/utils.h"

namespace marian {

// combine new expandedPathScores and previous beams into new set of beams
Beams BeamSearch::toHyps(const std::vector<unsigned int>& nBestKeys, // [currentDimBatch, beamSize] flattened -> ((batchIdx, beamHypIdx) flattened, word idx) flattened
                         const std::vector<float>& nBestPathScores,  // [currentDimBatch, beamSize] flattened
                         const size_t nBestBeamSize, // for interpretation of nBestKeys
                         const size_t vocabSize,     // ditto.
                         const Beams& beams,
                         const std::vector<Ptr<ScorerState /*const*/>>& states,
                         Ptr<data::CorpusBatch /*const*/> batch, // for alignments only
                         Ptr<FactoredVocab/*const*/> factoredVocab, size_t factorGroup,
                         const std::vector<bool>& dropBatchEntries, // [origDimBatch] - empty source batch entries are marked with true, should be cleared after first use.
                         const std::vector<IndexType>& batchIdxMap) const { // [origBatchIdx -> currentBatchIdx]
  std::vector<float> align; // collects alignment information from the last executed time step
  if(options_->hasAndNotEmpty("alignment") && factorGroup == 0)
    align = scorers_[0]->getAlignment(); // [beam depth * max src length * current batch size] -> P(s|t); use alignments from the first scorer, even if ensemble,

  const auto origDimBatch = beams.size(); // see function search for definition of origDimBatch and currentDimBatch etc.
  Beams newBeams(origDimBatch);           // return value of this function goes here. There are always origDimBatch beams.

  // create a reverse batchMap to obtain original batchIdx in the starting batch size
  // and calculate the current batch size based on non-empty beams
  std::vector<IndexType> reverseBatchIdxMap; // empty if not purging batch entries
  size_t currentDimBatch = beams.size();
  if(PURGE_BATCH) {
    reverseBatchIdxMap.resize(batchIdxMap.size()); // adjust size if doing batch purging.
    currentDimBatch = 0;
    for(int i = 0; i < batchIdxMap.size(); ++i) {
      reverseBatchIdxMap[batchIdxMap[i]] = i; // reverse batch index mapping, multiple occurences get overwritten with the last one,
                                              // which is expected due to down-shifting
      if(!beams[i].empty())
        currentDimBatch++;
    }
  }

  for(size_t i = 0; i < nBestKeys.size(); ++i) { // [currentDimBatch, beamSize] flattened
    // Keys encode batchIdx, beamHypIdx, and word index in the entire beam.
    // They can be between 0 and (vocabSize * nBestBeamSize * batchSize)-1.
    // (beamHypIdx refers to the GPU tensors, *not* the beams[] array; they are not the same in case of purging)
    const auto  key = nBestKeys[i];
    
    // decompose key into individual indices (batchIdx, beamHypIdx, wordIdx)
    const auto beamHypIdx      = (key / vocabSize) % nBestBeamSize;
    const auto currentBatchIdx = (key / vocabSize) / nBestBeamSize;
    const auto origBatchIdx    = reverseBatchIdxMap.empty() ? currentBatchIdx : reverseBatchIdxMap[currentBatchIdx]; // map currentBatchIdx back into original position within starting maximal batch size, required to find correct beam
    bool dropHyp = !dropBatchEntries.empty() && dropBatchEntries[origBatchIdx] && factorGroup == 0;
    
    WordIndex wordIdx;
    if(dropHyp) { // if we force=drop the hypothesis, assign EOS, otherwise the expected word id.
      if(factoredVocab) { // when using factoredVocab, extract the EOS lemma index from the word id, we predicting factors one by one here, hence lemma only
        std::vector<size_t> eosFactors;
        factoredVocab->word2factors(factoredVocab->getEosId(), eosFactors);
        wordIdx = (WordIndex)eosFactors[0];
      } else { // without factoredVocab lemma index and word index are the same. Safe cruising. 
        wordIdx = trgVocab_->getEosId().toWordIndex();
      }
    } else { // we are not dropping anything, just assign the normal index
      wordIdx = (WordIndex)(key % vocabSize);
    }

    // @TODO: We currently assign a log probability of 0 to all beam entries of the dropped batch entry, instead it might be a good idea to use
    // the per Hyp pathScore without the current expansion (a bit hard to obtain). 
    // For the case where we drop empty inputs, 0 is fine. For other use cases like a forced stop, the penultimate pathScore might be better. 
    // For the empty hyp this would naturally result in 0, too. 
    const float pathScore = dropHyp ? 0.f : nBestPathScores[i]; // 0 (Prob = 1, maximum score) if dropped or expanded path score for (batchIdx, beamHypIdx, word)

    const auto& beam = beams[origBatchIdx];
    auto& newBeam = newBeams[origBatchIdx]; // extended hypotheses are going to be placed in this new beam

    if(newBeam.size() >= beam.size()) // getNBestList() generates N for all batch entries incl. those that already have a narrower beam
      continue;
    if(pathScore == INVALID_PATH_SCORE) // (dummy slot or word that cannot be expanded by current factor)
      continue;
    
    ABORT_IF(pathScore < INVALID_PATH_SCORE, "Actual pathScore ({}) is lower than INVALID_PATH_SCORE ({})??", pathScore, INVALID_PATH_SCORE); // This should not happen in valid situations. Currently the only smaller value would be -inf (effect of overflow in summation?)
    ABORT_IF(beamHypIdx >= beam.size(), "Out of bounds beamHypIdx??"); // effectively this is equivalent to ABORT_IF(beams[origBatchIdx].empty(), ...)

    // map wordIdx to word
    auto prevBeamHypIdx = beamHypIdx; // back pointer
    auto prevHyp = beam[prevBeamHypIdx];
    Word word;
    // If short list has been set, then wordIdx is an index into the short-listed word set,
    // rather than the true word index.
    auto shortlist = scorers_[0]->getShortlist();
    if (factoredVocab) {
      // For factored decoding, the word is built over multiple decoding steps,
      // starting with the lemma, then adding factors one by one.
      if (factorGroup == 0) {
        word = factoredVocab->lemma2Word(shortlist ? shortlist->reverseMap((int) prevBeamHypIdx, (int) currentBatchIdx, wordIdx) : wordIdx);
        std::vector<size_t> factorIndices; factoredVocab->word2factors(word, factorIndices);
        //LOG(info, "{} + {} ({}) -> {} -> {}",
        //    factoredVocab->decode(prevHyp->tracebackWords()),
        //    factoredVocab->word2string(word), factorIndices[0], prevHyp->getPathScore(), pathScore);
      }
      else {
        //LOG(info, "{} |{} ({}) = {} ({}) -> {} -> {}",
        //    factoredVocab->decodeForDiagnostics(beam[beamHypIdx]->tracebackWords()),
        //    factoredVocab->getFactorGroupPrefix(factorGroup), factorGroup,
        //    factoredVocab->getFactorName(factorGroup, wordIdx), wordIdx,
        //    prevHyp->getPathScore(), pathScore);
        word = beam[beamHypIdx]->getWord();
        ABORT_IF(!factoredVocab->canExpandFactoredWord(word, factorGroup),
                  "A word without this factor snuck through to here??");
        word = factoredVocab->expandFactoredWord(word, factorGroup, wordIdx);
        prevBeamHypIdx = prevHyp->getPrevStateIndex();
        prevHyp = prevHyp->getPrevHyp(); // short-circuit the backpointer, so that the traceback does not contain partially factored words
      }
    }
    else if (shortlist)
      word = Word::fromWordIndex(shortlist->reverseMap((int) prevBeamHypIdx, (int) currentBatchIdx, wordIdx));
    else
      word = Word::fromWordIndex(wordIdx);

    auto hyp = Hypothesis::New(prevHyp, word, prevBeamHypIdx, pathScore);

    // Set score breakdown for n-best lists
    if(options_->get<bool>("n-best")) {
      auto breakDown = beam[beamHypIdx]->getScoreBreakdown();
      ABORT_IF(factoredVocab && factorGroup > 0 && !factoredVocab->canExpandFactoredWord(word, factorGroup),
               "A word without this factor snuck through to here??");
      breakDown.resize(states.size(), 0); // at start, this is empty, so this will set the initial score to 0
      for(size_t j = 0; j < states.size(); ++j) {
        auto lval = states[j]->getLogProbs().getFactoredLogitsTensor(factorGroup); // [maxBeamSize, 1, currentDimBatch, dimFactorVocab]
        // The flatting happens based on actual (current) batch size and batch index computed with batch-pruning as we are looking into the pruned tensor
        size_t flattenedLogitIndex = (beamHypIdx * currentDimBatch + currentBatchIdx) * vocabSize + wordIdx;  // (beam idx, batch idx, word idx); note: beam and batch are transposed, compared to 'key'

        // @TODO: use a function on shape() to index, or new method val->at({i1, i2, i3, i4}) with broadcasting
        ABORT_IF(lval->shape() != Shape({(int)nBestBeamSize, 1, (int)currentDimBatch, (int)vocabSize}) &&
                 (beamHypIdx == 0 && lval->shape() != Shape({1, 1, (int)currentDimBatch, (int)vocabSize})),
                 "Unexpected shape of logits?? {} != {}", lval->shape(), Shape({(int)nBestBeamSize, 1, (int)currentDimBatch, (int)vocabSize}));

        breakDown[j] += lval->get(flattenedLogitIndex);
      }
      hyp->setScoreBreakdown(breakDown);
    }

    // Set alignments
    if(!align.empty())
      hyp->setAlignment(getAlignmentsForHypothesis(align, batch, (int)beamHypIdx, (int)currentBatchIdx, (int)origBatchIdx, (int)currentDimBatch));
    else // not first factor: just copy
      hyp->setAlignment(beam[beamHypIdx]->getAlignment());

    newBeam.push_back(hyp);
  }

  // if factored vocab and this is not the first factor, we need to
  // also propagate factored hypotheses that do not get expanded in this step because they don't have this factor
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
      if (newBeam.size() > beam.size()) {
        //LOG(info, "Size {}, sorting...", newBeam.size());
        std::nth_element(newBeam.begin(), newBeam.begin() + beam.size(), newBeam.end(), [](Hypothesis::PtrType a, Hypothesis::PtrType b) {
          return a->getPathScore() > b->getPathScore(); // (sort highest score first)
        });
        //LOG(info, "Size {}, sorted...", newBeam.size());
        newBeam.resize(beam.size());
      }
    }
  }
  return newBeams;
}

std::vector<float> BeamSearch::getAlignmentsForHypothesis( // -> P(s|t) for current t and given beam and batch dim
    const std::vector<float> alignAll, // [beam depth, max src length, batch size, 1], flattened vector of all attention probablities
    Ptr<data::CorpusBatch> batch,
    int beamHypIdx,
    int currentBatchIdx,
    int origBatchIdx,
    int currentDimBatch) const {
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

  size_t origDimBatch = batch->size();  // number of sentences in batch
  size_t batchWidth   = batch->width(); // max src length

  // loop over words of batch entry 'currentBatchIdx' and beam entry 'beamHypIdx'
  std::vector<float> align;
  for(size_t srcPos = 0; srcPos < batchWidth; ++srcPos) { // loop over source positions
    // We are looking into the probabilites from an actual tensor, hence we need to use currentDimBatch and currentBatchIdx.
    size_t currentAttIdx = (batchWidth * beamHypIdx + srcPos) * currentDimBatch + currentBatchIdx; // = flatten [beam index, s, batch index, 0]

    // We are looking into the mask from the orginal batch, hence we need to use origDmBatch and origBatchIdx.
    size_t origAttIdx  = (batchWidth * beamHypIdx + srcPos) * origDimBatch + origBatchIdx;; // = flatten [beam index, s, batch index, 0]
    size_t origMaskIdx = origAttIdx % (batchWidth * origDimBatch); // == batchIdx + (batchSize * srcPos) = flatten [0, s, batch index, 0]

    // If the original position is not masked out used the corresponding current attention score.
    if(batch->front()->mask()[origMaskIdx] != 0)
      align.emplace_back(alignAll[currentAttIdx]);
  }
  return align;
}

// remove all beam entries that have reached EOS
Beams BeamSearch::purgeBeams(const Beams& beams, /*in/out=*/std::vector<IndexType>& batchIdxMap) {
  const auto trgEosId = trgVocab_->getEosId();
  Beams newBeams;
  size_t beamIdx = 0; // beam index
  for(auto beam : beams) {
    Beam newBeam; // a beam of surviving hyps
    for(auto hyp : beam)
      if(hyp->getWord() != trgEosId) // if this hyp is not finished,
        newBeam.push_back(hyp);      // move over to beam of surviving hyps

    if(PURGE_BATCH)
      if(newBeam.empty() && !beam.empty()) {      // previous beam had hyps, but all were finished in this step, newBeam will now stay empty
        for(size_t i = beamIdx + 1; i < beams.size(); ++i) // for all entries above this beam
          batchIdxMap[i] = batchIdxMap[i] - 1;  // make them look at one batch index below, as the current entry will be removed from the batch.
    }

    newBeams.push_back(newBeam);
    beamIdx++; // move to next beam index
  }
  return newBeams;
}

//**********************************************************************
// main decoding function
Histories BeamSearch::search(Ptr<ExpressionGraph> graph, Ptr<data::CorpusBatch> batch) {
  auto factoredVocab = trgVocab_->tryAs<FactoredVocab>();
  size_t numFactorGroups = factoredVocab ? factoredVocab->getNumGroups() : 1;
  if (numFactorGroups == 1) // if no factors then we didn't need this object in the first place
    factoredVocab.reset();

  // We will use the prefix "origBatch..." whenever we refer to batch dimensions of the original batch. These do not change during search.
  // We will use the prefix "currentBatch.." whenever we refer to batch dimension that can change due to batch-pruning.
  const int origDimBatch = (int)batch->size();
  const auto trgEosId = trgVocab_->getEosId();

  auto getNBestList = createGetNBestListFn(beamSize_, origDimBatch, graph->getDeviceId());

  for(auto scorer : scorers_) {
    scorer->clear(graph);
  }

  Histories histories(origDimBatch);
  for(int i = 0; i < origDimBatch; ++i) {
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

  // create one beam per batch entry with sentence-start hypothesis
  Beams beams(origDimBatch, Beam(beamSize_, Hypothesis::New())); // array [origDimBatch] of array [maxBeamSize] of Hypothesis, keeps full size through search.
                                                                 // batch purging is determined from an empty sub-beam.
  std::vector<IndexType> batchIdxMap(origDimBatch); // Record at which batch entry a beam is looking.
                                                    // By default that corresponds to position in array,
                                                    // but shifts in the course of removing batch entries when they are finished.

  const std::vector<bool> emptyBatchEntries; // used for recording if there are empty input batch entries
  for(int origBatchIdx = 0; origBatchIdx < origDimBatch; ++origBatchIdx) {
    batchIdxMap[origBatchIdx] = origBatchIdx; // map to same position on initialization
    auto& beam = beams[origBatchIdx];
    histories[origBatchIdx]->add(beam, trgEosId); // add beams with start-hypotheses to traceback grid

    // Mark batch entries that consist only of source <EOS> i.e. these are empty lines. They will be forced to EOS and purged from batch
    const auto& srcEosId = batch->front()->vocab()->getEosId();
    const_cast<std::vector<bool>&>(emptyBatchEntries).push_back(batch->front()->data()[origBatchIdx] == srcEosId); // const_cast during construction
  }

  Expr suppressedWordIndices;
  bool suppressUnk     = !options_->get<bool>("allow-unk", false);
  bool suppressSpecial = !options_->get<bool>("allow-special", false);
  if (suppressUnk || suppressSpecial) { // do we need to suppress unk or special?
    std::vector<WordIndex> suppressed = trgVocab_->suppressedIndices(suppressUnk, suppressSpecial);

    auto shortlist = scorers_[0]->getShortlist(); // first shortlist is generally ok, @TODO: make sure they are the same across scorers?
    if(shortlist) // check if suppressed words are allowed by the shortlist, if not, remove
      suppressed.erase(std::remove_if(suppressed.begin(), 
                                      suppressed.end(), 
                                      [&](WordIndex i) { 
                                        return shortlist->tryForwardMap(i) == data::Shortlist::npos;
                                      }),
                       suppressed.end());
    
    if(!suppressed.empty())
      suppressedWordIndices = graph->indices(suppressed);
  }

  // the decoding process updates the following state information in each output time step:
  //  - beams: array [origDimBatch] of array [maxBeamSize] of Hypothesis
  //     - current output time step's set of active hypotheses, aka active search space
  //  - states[.]: ScorerState
  //     - NN state; one per scorer, e.g. 2 for ensemble of 2
  // and it forms the following return value
  //  - histories: array [origDimBatch] of History
  //    with History: vector [t] of array [maxBeamSize] of Hypothesis
  //    with Hypothesis: (last word, aggregate score, prev Hypothesis)

  IndexType currentDimBatch = origDimBatch;
  auto prevBatchIdxMap = batchIdxMap; // [origBatchIdx -> currentBatchIdx] but shifted by one time step
  // main loop over output time steps
  for (size_t t = 0; ; t++) {
    //std::cerr << "\nstep=" << t << std::endl;
    ABORT_IF(origDimBatch != beams.size(), "Lost a batch entry??");
    // determine beam size for next output time step, as max over still-active sentences
    // E.g. if all batch entries are down from beam 5 to no more than 4 surviving hyps, then
    // switch to beam of 4 for all. If all are done, then beam ends up being 0, and we are done.
    size_t maxBeamSize = 0; // @TODO: is there some std::algorithm for this?
    for(auto& beam : beams)
      if(beam.size() > maxBeamSize)
        maxBeamSize = beam.size();

    // done if all batch entries have reached EOS on all beam entries
    if (maxBeamSize == 0)
      break;

    for (size_t factorGroup = 0; factorGroup < numFactorGroups; factorGroup++) {
      // for factored vocabs, we do one factor at a time, but without updating the scorer for secondary factors

      //**********************************************************************
      // create constant containing previous path scores for current beam
      // Also create mapping of hyp indices, for reordering the decoder-state tensors.
      std::vector<IndexType> batchIndices;    // [1,           1, currentDimBatch, 1] indices of currently used batch indices with regard to current, actual tensors
      std::vector<IndexType> hypIndices;      // [maxBeamSize, 1, currentDimBatch, 1] (flattened) tensor index ((beamHypIdx, batchIdx), flattened) of prev hyp that a hyp originated from
      std::vector<Word> prevWords;            // [maxBeamSize, 1, currentDimBatch, 1] (flattened) word that a hyp ended in, for advancing the decoder-model's history
      Expr prevPathScores;                    // [maxBeamSize, 1, currentDimBatch, 1], path score that a hyp ended in (last axis will broadcast into vocab size when adding expandedPathScores)

      bool anyCanExpand = false; // stays false if all hyps are invalid factor expansions
      if(t == 0 && factorGroup == 0) { // no scores yet
        prevPathScores = graph->constant({1, 1, 1, 1}, inits::fromValue(0));
        anyCanExpand = true;

        // at the beginning all batch entries are used
        batchIndices.resize(origDimBatch);
        std::iota(batchIndices.begin(), batchIndices.end(), 0);
      } else {
        if(factorGroup == 0)                                                              // only factorGroup==0 can subselect neural state
          for(int currentBatchIdx = 0; currentBatchIdx < beams.size(); ++currentBatchIdx) // loop over batch entries (active sentences)
            if(!beams[currentBatchIdx].empty() || !PURGE_BATCH)                           // for each beam check
              batchIndices.push_back(prevBatchIdxMap[currentBatchIdx]);                   // which batch entries were active in previous step

        std::vector<float> prevScores;
        for(size_t beamHypIdx = 0; beamHypIdx < maxBeamSize; ++beamHypIdx) { // loop over globally maximal beam-size (maxBeamSize)
          for(int origBatchIdx = 0; origBatchIdx < origDimBatch; ++origBatchIdx) { // loop over all batch entries (active and inactive)
            auto& beam = beams[origBatchIdx];
            if(beamHypIdx < beam.size()) {
              auto hyp = beam[beamHypIdx];
              auto word = hyp->getWord();
              auto canExpand = (!factoredVocab || factoredVocab->canExpandFactoredWord(hyp->getWord(), factorGroup));
              //LOG(info, "[{}, {}] Can expand {} with {} -> {}", batchIdx, beamHypIdx, (*batch->back()->vocab())[hyp->getWord()], factorGroup, canExpand);
              anyCanExpand |= canExpand;

              auto currentBatchIdx = origBatchIdx;
              if(PURGE_BATCH) {
                if(factorGroup == 0)
                  currentBatchIdx = prevBatchIdxMap[origBatchIdx]; // subselection may happen for factorGroup == 0
                else
                  currentBatchIdx = batchIdxMap[origBatchIdx];     // no subselection happens for factorGroup > 0,
                                                                   // but we treat it like a next step, since a step
                                                                   // happened for factorGroup == 0
              }

              auto hypIndex = (IndexType)(hyp->getPrevStateIndex() * currentDimBatch + currentBatchIdx); // (beamHypIdx, batchIdx), flattened, for index_select() operation

              hypIndices.push_back(hypIndex); // (beamHypIdx, batchIdx), flattened as said above.
              prevWords .push_back(word);
              prevScores.push_back(canExpand ? hyp->getPathScore() : INVALID_PATH_SCORE);
            } else {  // pad to maxBeamSize (dummy hypothesis)
              if(!PURGE_BATCH || !beam.empty()) { // but only if we are not pruning and the beam is not deactivated yet
                hypIndices.push_back(0);
                prevWords.push_back(trgEosId);    // (unused, but must be valid)
                prevScores.push_back((float)INVALID_PATH_SCORE);
              }
            }
          }
        }
        if(factorGroup == 0)
          currentDimBatch = (IndexType) batchIndices.size(); // keep batch size constant for all factor groups in a time step
        prevPathScores = graph->constant({(int)maxBeamSize, 1, (int)currentDimBatch, 1}, inits::fromVector(prevScores));
      }
      if (!anyCanExpand) // all words cannot expand this factor: skip
        continue;

      //**********************************************************************
      // compute expanded path scores with word prediction probs from all scorers
      auto expandedPathScores = prevPathScores; // will become [maxBeamSize, 1, currDimBatch, dimVocab]
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
          states[i] = scorers_[i]->step(graph, states[i], hypIndices, prevWords, batchIndices, (int)maxBeamSize);
          if (numFactorGroups == 1) { // @TODO: this branch can go away
            logProbs = states[i]->getLogProbs().getLogits(); // [maxBeamSize, 1, currentDimBatch, dimVocab]
          } else {
            auto shortlist = scorers_[i]->getShortlist();
            logProbs = states[i]->getLogProbs().getFactoredLogits(factorGroup, shortlist); // [maxBeamSize, 1, currentDimBatch, dimVocab]
          }
        }
        else {
          // add secondary factors
          // For those, we don't update the decoder-model state in any way.
          // Instead, we just keep expanding with the factors.
          // We will have temporary Word entries in hyps with some factors set to FACTOR_NOT_SPECIFIED.
          // For some lemmas, a factor is not applicable. For those, the factor score is the same (zero)
          // for all factor values. This would thus unnecessarily pollute the beam with identical copies,
          // and push out other hypotheses. Hence, we exclude those here by setting the path score to
          // INVALID_PATH_SCORE. Instead, toHyps() explicitly propagates those hyps by simply copying the
          // previous hypothesis.
          logProbs = states[i]->getLogProbs().getFactoredLogits(factorGroup, /*shortlist=*/ nullptr, hypIndices, maxBeamSize); // [maxBeamSize, 1, currentDimBatch, dimVocab]
        }
        // expand all hypotheses, [maxBeamSize, 1, currentDimBatch, 1] -> [maxBeamSize, 1, currentDimBatch, dimVocab]
        expandedPathScores = expandedPathScores + scorers_[i]->getWeight() * logProbs;
      }

      // make beams continuous
      expandedPathScores = swapAxes(expandedPathScores, 0, 2); // -> [currentDimBatch, 1, maxBeamSize, dimVocab]

      // perform NN computation
      if(t == 0 && factorGroup == 0)
        graph->forward();
      else
        graph->forwardNext();

      //**********************************************************************
      // suppress specific symbols if not at right positions
      if(suppressedWordIndices && factorGroup == 0)
        suppressWords(expandedPathScores, suppressedWordIndices);

      //**********************************************************************
      // perform beam search

      // find N best amongst the (maxBeamSize * dimVocab) hypotheses
      std::vector<unsigned int> nBestKeys; // [currentDimBatch, maxBeamSize] flattened -> (batchIdx, beamHypIdx, word idx) flattened
      std::vector<float> nBestPathScores;  // [currentDimBatch, maxBeamSize] flattened
      getNBestList(/*in*/   expandedPathScores->val(),   // [currentDimBatch, 1, maxBeamSize, dimVocab or dimShortlist]
                  /*N=*/    maxBeamSize,                 // desired beam size
                  /*out*/   nBestPathScores,
                   /*out*/  nBestKeys,
                  /*first=*/t == 0 && factorGroup == 0); // @TODO: this is only used for checking presently, and should be removed altogether
      // Now, nBestPathScores contain N-best expandedPathScores for each batch and beam,
      // and nBestKeys for each their original location (batchIdx, beamHypIdx, word).

      // combine N-best sets with existing search space (beams) to updated search space
      beams = toHyps(nBestKeys, nBestPathScores,
                     /*nBestBeamSize*/expandedPathScores->shape()[-2], // used for interpretation of keys
                     /*vocabSize=*/expandedPathScores->shape()[-1],    // used for interpretation of keys
                     beams,
                     states,            // used for keeping track of per-ensemble-member path score
                     batch,             // only used for propagating alignment info
                     factoredVocab, factorGroup,
                     emptyBatchEntries, // [origDimBatch] - empty source batch entries are marked with true
                     batchIdxMap);      // used to create a reverse batch index map to recover original batch indices for this step
    } // END FOR factorGroup = 0 .. numFactorGroups-1

    prevBatchIdxMap = batchIdxMap; // save current batchIdx map to be used in next step; we are then going to look one step back

    // remove all hyps that end in EOS
    // The position of a hyp in the beam may change.
    // in/out = shifts the batch index map if a beam gets fully purged
    const auto purgedNewBeams = purgeBeams(beams, /*in/out=*/batchIdxMap);

    // add updated search space (beams) to our return value
    bool maxLengthReached = false;
    for(int batchIdx = 0; batchIdx < origDimBatch; ++batchIdx) {
      // if this batch entry has surviving hyps then add them to the traceback grid
      if(!beams[batchIdx].empty()) { // if the beam is not empty expand the history object associated with the beam
        if (histories[batchIdx]->size() >= options_->get<float>("max-length-factor") * batch->front()->batchWidth())
          maxLengthReached = true;
        histories[batchIdx]->add(beams[batchIdx], trgEosId, purgedNewBeams[batchIdx].empty() || maxLengthReached);
      }
    }
    if (maxLengthReached) // early exit if max length limit was reached
      break;

    // this is the search space for the next output time step
    beams = purgedNewBeams;
  } // end of main loop over output time steps

  return histories; // [origDimBatch][t][N best hyps]
}

}  // namespace marian
