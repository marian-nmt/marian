#include "common/search.h"

#include <boost/timer/timer.hpp>

#include "common/god.h"
#include "common/history.h"
#include "common/filter.h"
#include "common/base_matrix.h"

using namespace std;

Search::Search(size_t threadId)
  : scorers_(God::Summon().GetScorers(threadId)),
    bestHyps_(God::Summon().GetBestHyps(threadId)) {
}


size_t Search::MakeFilter(const std::set<Word>& srcWords, size_t vocabSize) {
  filterIndices_ = God::Summon().GetFilter().GetFilteredVocab(srcWords, vocabSize);
  for (size_t i = 0; i < scorers_.size(); i++) {
      scorers_[i]->Filter(filterIndices_);
  }
  return filterIndices_.size();
}

void Search::InitScorers(const Sentences& sentences, States& states, States& nextStates) {
  for (size_t i = 0; i < scorers_.size(); i++) {
    Scorer &scorer = *scorers_[i];
    scorer.SetSource(sentences);

    states[i].reset(scorer.NewState());
    nextStates[i].reset(scorer.NewState());

    scorer.BeginSentenceState(*states[i], sentences.size());
  }
}

boost::shared_ptr<Histories> Search::Decode(const Sentences& sentences) {
  boost::timer::cpu_timer timer;

  boost::shared_ptr<Histories> ret(new Histories(sentences));

  size_t batchSize = sentences.size();
  std::vector<size_t> beamSizes(batchSize, 1);

  Beam prevHyps(batchSize, HypothesisPtr(new Hypothesis()));
  for (size_t i = 0; i < ret->size(); ++i) {
    History &history = *ret->at(i).get();
    history.Add(prevHyps);
  }

  States states(scorers_.size());
  States nextStates(scorers_.size());

  size_t vocabSize = scorers_[0]->GetVocabSize();

  bool filter = God::Summon().Get<std::vector<std::string>>("softmax-filter").size();
  if (filter) {
    std::set<Word> srcWords;
    for (size_t i = 0; i < sentences.size(); ++i) {
      const Sentence &sentence = *sentences.at(i);
      for (const auto& srcWord : sentence.GetWords()) {
        srcWords.insert(srcWord);
      }
    }
    vocabSize = MakeFilter(srcWords, vocabSize);
  }

  size_t maxLength = 0;
  for (size_t i = 0; i < sentences.size(); ++i) {
    const Sentence &sentence = *sentences.at(i);
    maxLength = std::max(maxLength, sentence.GetWords().size());
  }

  InitScorers(sentences, states, nextStates);

  for (size_t decoderStep = 0; decoderStep < 3 * maxLength; ++decoderStep) {
    for (size_t i = 0; i < scorers_.size(); i++) {
      Scorer &scorer = *scorers_[i];
      State &state = *states[i];
      State &nextState = *nextStates[i];

      scorer.Score(state, nextState, beamSizes);
    }

    if (decoderStep == 0) {
      for (auto& beamSize : beamSizes) {
        beamSize = God::Summon().Get<size_t>("beam-size");
      }
    }
    Beams beams(batchSize);
    bool returnAlignment = God::Summon().Get<bool>("return-alignment");

    bestHyps_(beams, prevHyps, beamSizes, scorers_, filterIndices_, returnAlignment);

    for (size_t i = 0; i < batchSize; ++i) {
      if (!beams[i].empty()) {
        ret->at(i)->Add(beams[i], ret->at(i)->size() == 3 * sentences.at(i)->GetWords().size());
      }
    }

    Beam survivors;
    for (size_t batchID = 0; batchID < batchSize; ++batchID) {
      for (auto& h : beams[batchID]) {
        if (h->GetWord() != EOS) {
          survivors.push_back(h);
        } else {
          --beamSizes[batchID];
        }
      }
    }

    if (survivors.size() == 0) {
      break;
    }

    for (size_t i = 0; i < scorers_.size(); i++) {
      scorers_[i]->AssembleBeamState(*nextStates[i], survivors, *states[i]);
    }

    prevHyps.swap(survivors);
  }

  LOG(progress) << "Batch " << sentences.taskCounter << "." << sentences.bunchId
                << ": Search took " << timer.format(3, "%ws");

  for (auto scorer : scorers_) {
	  scorer->CleanUpAfterSentence();
  }

  return ret;
}
