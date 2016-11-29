#include "common/search.h"

#include <boost/timer/timer.hpp>

#include "common/god.h"
#include "common/history.h"
#include "common/filter.h"
#include "common/base_matrix.h"

using namespace std;

Search::Search(size_t threadId)
  : scorers_(God::GetScorers(threadId)),
    BestHyps_(God::GetBestHyps(threadId)) {
}


size_t Search::MakeFilter(const Words& srcWords, size_t vocabSize) {
  filterIndices_ = God::GetFilter().GetFilteredVocab(srcWords, vocabSize);
  for (size_t i = 0; i < scorers_.size(); i++) {
      scorers_[i]->Filter(filterIndices_);
  }
  return filterIndices_.size();
}

Histories Search::Decode(const Sentences& sentences) {
  boost::timer::cpu_timer timer;

  size_t batchSize = sentences.size();
  std::vector<size_t> beamSizes(batchSize, 1);

  // @TODO Future: in order to do batch sentence decoding
  // it should be enough to keep track of hypotheses in
  // separate History objects.

  Histories histories(batchSize);
  Beam prevHyps(batchSize, HypothesisPtr(new Hypothesis()));
  for (auto& history : histories) {
    history.Add(prevHyps);
  }

  States states(scorers_.size());
  States nextStates(scorers_.size());

  size_t vocabSize = scorers_[0]->GetVocabSize();

  bool filter = God::Get<std::vector<std::string>>("softmax-filter").size();
  if (filter) {
    vocabSize = MakeFilter(sentences[0].GetWords(), vocabSize);
  }



  size_t maxLength = 0;
  for (const auto& sentence : sentences) {
    maxLength = std::max(maxLength, sentence.GetWords().size());
  }

  for (size_t decoderStep = 0; decoderStep < 3 * maxLength; ++decoderStep) {
    if (decoderStep == 0) {
      for (size_t i = 0; i < scorers_.size(); i++) {
        Scorer &scorer = *scorers_[i];
        scorer.SetSource(sentences);

        states[i].reset(scorer.NewState());
        nextStates[i].reset(scorer.NewState());

        scorer.BeginSentenceState(*states[i], batchSize);
      }
    }

    for (size_t i = 0; i < scorers_.size(); i++) {
      Scorer &scorer = *scorers_[i];
      State &state = *states[i];
      State &nextState = *nextStates[i];

      scorer.Score(state, nextState, beamSizes);
      if (decoderStep == 0) {
        for (auto& beamSize : beamSizes) {
          beamSize = God::Get<size_t>("beam-size");
        }
      }
    }

    std::vector<Beam> beams(batchSize);

    bool returnAlignment = God::Get<bool>("return-alignment");

    BestHyps_(beams, prevHyps, beamSizes, scorers_, filterIndices_, returnAlignment);
    for (size_t i = 0; i < batchSize; ++i) {
      if (!beams[i].empty()) {
        histories[i].Add(beams[i], histories[i].size() == 3 * sentences[i].GetWords().size());
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

  LOG(progress) << "Line " << sentences[0].GetLine()
                << ": Search took " << timer.format(3, "%ws");

  for (auto scorer : scorers_) {
	  scorer->CleanUpAfterSentence();
  }

  return histories;
}
