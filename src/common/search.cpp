#include <boost/timer/timer.hpp>
#include "search.h"
#include "common/base_matrix.h"

Search::Search(size_t threadId)
: scorers_(God::GetScorers(threadId)) {}

History Search::Decode(const Sentence& sentence) {
  boost::timer::cpu_timer timer;

  size_t beamSize = God::Get<size_t>("beam-size");
  bool normalize = God::Get<bool>("normalize");
  size_t vocabSize = scorers_[0]->GetVocabSize();

  // @TODO Future: in order to do batch sentence decoding
  // it should be enough to keep track of hypotheses in
  // separate History objects.

  History history;

  Beam prevHyps = { HypothesisPtr(new Hypothesis()) };
  history.Add(prevHyps);

  States states(scorers_.size());
  States nextStates(scorers_.size());
  BaseMatrices probs(scorers_.size());

  for(size_t i = 0; i < scorers_.size(); i++) {
	Scorer &scorer = *scorers_[i];
	scorer.SetSource(sentence);

	states[i].reset(scorer.NewState());
	nextStates[i].reset(scorer.NewState());

	scorer.BeginSentenceState(*states[i]);

	probs[i] = scorer.CreateMatrix();
  }

  const size_t maxLength = sentence.GetWords().size() * 3;
  do {
	for(size_t i = 0; i < scorers_.size(); i++) {
		Scorer &scorer = *scorers_[i];
		BaseMatrix &prob = *probs[i];

		prob.Resize(beamSize, vocabSize);
		scorer.Score(*states[i], prob, *nextStates[i]);
	}

	// Looking at attention vectors
	//mblas::Matrix A;
	//std::static_pointer_cast<EncoderDecoder>(scorers_[0])->GetAttention(A);
	//mblas::debug1(A, 0, sentence.GetWords().size());

	Beam hyps;

	assert(probs.size());
	const BaseMatrix &firstMatrix = *probs[0];

	firstMatrix.BestHyps(hyps, prevHyps, probs, beamSize, history, scorers_, filterIndices_);
	history.Add(hyps, history.size() == maxLength);

	Beam survivors;
	for(auto h : hyps) {
	  if(h->GetWord() != EOS) {
		survivors.push_back(h);
	  }
	}
	beamSize = survivors.size();
	if(beamSize == 0) {
	  break;
	}

	for(size_t i = 0; i < scorers_.size(); i++) {
	  scorers_[i]->AssembleBeamState(*nextStates[i], survivors, *states[i]);
	}

	prevHyps.swap(survivors);

  } while(history.size() <= maxLength);

  LOG(progress) << "Line " << sentence.GetLine()
	<< ": Search took " << timer.format(3, "%ws");

  for(size_t i = 0; i < scorers_.size(); i++) {
	  Scorer &scorer = *scorers_[i];
	  scorer.CleanUpAfterSentence();

	  BaseMatrix *prob = probs[i];
	  delete prob;
  }

  return history;
}


