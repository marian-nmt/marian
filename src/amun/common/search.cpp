#include <boost/timer/timer.hpp>
#include "common/search.h"
#include "common/sentences.h"
#include "common/god.h"
#include "common/history.h"
#include "common/filter.h"
#include "common/base_matrix.h"

using namespace std;

namespace amunmt {

Search::Search(const God &god)
{
  deviceInfo_ = god.GetNextDevice();
  scorers_ = god.GetScorers(deviceInfo_);
  bestHyps_ = god.GetBestHyps(deviceInfo_);
}

Search::~Search()
{
#ifdef CUDA
  if (deviceInfo_.deviceType == GPUDevice) {
    cudaSetDevice(deviceInfo_.deviceId);
  }
#endif
}
  
States Search::NewStates() const
{
  size_t numScorers = scorers_.size();

  States states(numScorers);
  for (size_t i = 0; i < numScorers; i++) {
    Scorer &scorer = *scorers_[i];
    states[i].reset(scorer.NewState());
  }

  return states;
}

size_t Search::MakeFilter(const God &god, const std::set<Word>& srcWords, size_t vocabSize) {
  filterIndices_ = god.GetFilter().GetFilteredVocab(srcWords, vocabSize);
  for (size_t i = 0; i < scorers_.size(); i++) {
      scorers_[i]->Filter(filterIndices_);
  }
  return filterIndices_.size();
}

void Search::Encode(const Sentences& sentences, States& states) {
  for (size_t i = 0; i < scorers_.size(); i++) {
    Scorer &scorer = *scorers_[i];
    scorer.SetSource(sentences);

    scorer.BeginSentenceState(*states[i], sentences.size());
  }
}

void Search::Decode(
		const God &god,
		const Sentences& sentences,
		States &states,
		std::shared_ptr<Histories> &histories,
		Beam &prevHyps)
{
  States nextStates = NewStates();

  size_t batchSize = sentences.size();
  std::vector<size_t> beamSizes(batchSize, 1);

  for (size_t decoderStep = 0; decoderStep < 3 * sentences.GetMaxLength(); ++decoderStep) {
	  bool hasSurvivors = Decode(
			  god,
			  sentences,
			  states,
			  histories,
			  prevHyps,
			  decoderStep,
			  nextStates,
			  beamSizes
	  	  	  );

	    if (!hasSurvivors) {
	    	break;
	    }
  }
}

bool Search::Decode(
		const God &god,
		const Sentences& sentences,
		States &states,
		std::shared_ptr<Histories> &histories,
		Beam &prevHyps,
		size_t decoderStep,
		States &nextStates,
		std::vector<size_t> &beamSizes
		)
{
   size_t batchSize = sentences.size();

    for (size_t i = 0; i < scorers_.size(); i++) {
      Scorer &scorer = *scorers_[i];
      const State &state = *states[i];
      State &nextState = *nextStates[i];

      scorer.Decode(god, state, nextState, beamSizes);
    }

    if (decoderStep == 0) {
      for (auto& beamSize : beamSizes) {
      beamSize = god.Get<size_t>("beam-size");
      }
    }

    Beams beams(batchSize);
    Beam survivors;

    bool hasSurvivors = CalcBeam(
    		god,
    		prevHyps,
    		beams,
    		beamSizes,
    		histories,
    		sentences,
    		survivors,
    		states,
    		nextStates
    		);

    if (!hasSurvivors) {
    	return false;
    }

    return true;
}

bool Search::CalcBeam(
		const God &god,
		Beam &prevHyps,
		Beams &beams,
		std::vector<size_t> &beamSizes,
		std::shared_ptr<Histories> &histories,
		const Sentences& sentences,
		Beam &survivors,
		States &states,
		States &nextStates
		)
{
    bool returnAlignment = god.Get<bool>("return-alignment");
    size_t batchSize = sentences.size();

    bestHyps_->CalcBeam(god, prevHyps, scorers_, filterIndices_, returnAlignment, beams, beamSizes);

    for (size_t i = 0; i < batchSize; ++i) {
      if (!beams[i].empty()) {
        histories->at(i)->Add(beams[i], histories->at(i)->size() == 3 * sentences.at(i)->GetWords().size());
      }
    }

    for (size_t batchID = 0; batchID < batchSize; ++batchID) {
      for (auto& h : beams[batchID]) {
        if (h->GetWord() != EOS_ID) {
          survivors.push_back(h);
        } else {
          --beamSizes[batchID];
        }
      }
    }

    if (survivors.size() == 0) {
      return false;
    }

    for (size_t i = 0; i < scorers_.size(); i++) {
      scorers_[i]->AssembleBeamState(*nextStates[i], survivors, *states[i]);
    }

    prevHyps.swap(survivors);

    return true;
}

std::shared_ptr<Histories> Search::Process(const God &god, const Sentences& sentences) {
  boost::timer::cpu_timer timer;

  std::shared_ptr<Histories> histories(new Histories(god, sentences));

  size_t batchSize = sentences.size();
  size_t numScorers = scorers_.size();

  Beam prevHyps(batchSize, HypothesisPtr(new Hypothesis()));

  States states = NewStates();

  // calc
  PreProcess(god, sentences, histories, prevHyps);
  Encode(sentences, states);
  Decode(god, sentences, states, histories, prevHyps);
  PostProcess();

  LOG(progress, "Search took {}", timer.format(3, "%ws"));
  return histories;
}

void Search::PreProcess(
		const God &god,
		const Sentences& sentences,
		std::shared_ptr<Histories> &histories,
		Beam &prevHyps)
{
  size_t vocabSize = scorers_[0]->GetVocabSize();

  for (size_t i = 0; i < histories->size(); ++i) {
    History &history = *histories->at(i).get();
    history.Add(prevHyps);
  }

  bool filter = god.Get<std::vector<std::string>>("softmax-filter").size();
  if (filter) {
    std::set<Word> srcWords;
    for (size_t i = 0; i < sentences.size(); ++i) {
      const Sentence &sentence = *sentences.at(i);
      for (const auto& srcWord : sentence.GetWords()) {
        srcWords.insert(srcWord);
      }
    }
    vocabSize = MakeFilter(god, srcWords, vocabSize);
  }

}

void Search::PostProcess()
{
  for (auto scorer : scorers_) {
	  scorer->CleanUpAfterSentence();
  }
}


}

