#include <boost/timer/timer.hpp>
#include "common/search.h"
#include "common/sentences.h"
#include "common/god.h"
#include "common/history.h"
#include "common/histories.h"
#include "common/filter.h"
#include "common/base_tensor.h"

#ifdef CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

using namespace std;

namespace amunmt {

Search::Search(const God &god)
  : deviceInfo_(god.GetNextDevice()),
    scorers_(god.GetScorers(deviceInfo_)),
    filter_(god.GetFilter()),
    maxBeamSize_(god.Get<unsigned>("beam-size")),
    maxLengthMult_(god.Get<float>("max-length-multiple")),
    normalizeScore_(god.Get<bool>("normalize")),
    bestHyps_(god.GetBestHyps(deviceInfo_))
{
  //activeCount_.resize(god.Get<unsigned>("mini-batch") + 1, 0);
  BEGIN_TIMER_CPU("Search");
}


Search::~Search()
{
  PAUSE_TIMER_CPU("Search");
  //BatchStats();
#ifdef CUDA
  if (deviceInfo_.deviceType == GPUDevice) {
    cudaSetDevice(deviceInfo_.deviceId);
  }
#endif

  if (timers.size()) {
    boost::timer::nanosecond_type encDecWall = timers["Search"].elapsed().wall;

    cerr << "timers:" << endl;
    for (auto iter = timers.begin(); iter != timers.end(); ++iter) {
      const boost::timer::cpu_timer &timer = iter->second;
      boost::timer::cpu_times t = timer.elapsed();
      boost::timer::nanosecond_type wallTime = t.wall;

      int percent = (float) wallTime / (float) encDecWall * 100.0f;

      cerr << iter->first << " ";

      for (int i = 0; i < ((int)35 - (int)iter->first.size()); ++i) {
        cerr << " ";
      }

      cerr << timer.format(2, "%w") << " (" << percent << ")" << endl;
    }
  }

}

void Search::CleanAfterTranslation()
{
  for (auto scorer : scorers_) {
    scorer->CleanAfterTranslation();
  }
}

std::shared_ptr<Histories> Search::Translate(const Sentences& sentences) {
  boost::timer::cpu_timer timer;

  if (filter_) {
    FilterTargetVocab(sentences);
  }


  States states = Encode(sentences);
  States nextStates = NewStates();
  std::vector<unsigned> beamSizes(sentences.size(), 1);

  std::shared_ptr<Histories> histories(new Histories(sentences, normalizeScore_, maxLengthMult_));
  Beam prevHyps = histories->GetFirstHyps();

  for (unsigned decoderStep = 0; decoderStep < maxLengthMult_ * (float) sentences.GetMaxLength(); ++decoderStep) {
    //boost::timer::cpu_timer timerStep;
    //timerStep.start();

    for (unsigned i = 0; i < scorers_.size(); i++) {
      scorers_[i]->Decode(*states[i], *nextStates[i], beamSizes);
    }

    if (decoderStep == 0) {
      for (auto& beamSize : beamSizes) {
        beamSize = maxBeamSize_;
      }
    }
    //cerr << "beamSizes=" << Debug(beamSizes, 1) << endl;

    bool hasSurvivors = CalcBeam(histories, beamSizes, prevHyps, states, nextStates, decoderStep);
    if (!hasSurvivors) {
      break;
    }

    //timerStep.stop();
    //cerr << "decoderStep=" << decoderStep << " " << timerStep.format(4, "%w") << endl;
    //cerr << "states0=" << states[0]->Debug(0) << endl;
    //cerr << "beamSizes=" << beamSizes.size() << " " << histories->NumActive() << endl;
    //++activeCount_[histories->NumActive()];
  }

  CleanAfterTranslation();

  LOG(progress)->info("Search took {}", timer.format(3, "%ws"));
  return histories;
}

States Search::Encode(const Sentences& sentences) {
  States states;
  for (auto& scorer : scorers_) {
    scorer->Encode(sentences);
    auto state = scorer->NewState();
    scorer->BeginSentenceState(*state, sentences.size());
    states.emplace_back(state);
  }
  return states;
}

bool Search::CalcBeam(
    std::shared_ptr<Histories>& histories,
    std::vector<unsigned>& beamSizes,
    Beam& prevHyps,
    States& states,
    States& nextStates,
    unsigned decoderStep)
{
    unsigned batchSize = beamSizes.size();
    Beams beams(batchSize);
    bestHyps_->CalcBeam(prevHyps, scorers_, filterIndices_, beams, beamSizes);
    histories->Add(beams);

    //cerr << "batchSize=" << batchSize << endl;
    histories->SetActive(false);
    Beam survivors;
    for (unsigned batchId = 0; batchId < batchSize; ++batchId) {
      const History &hist = *histories->at(batchId);
      unsigned maxLength = hist.GetMaxLength();

      //cerr << "beamSizes[batchId]=" << batchId << " " << beamSizes[batchId] << " " << maxLength << endl;
      for (auto& h : beams[batchId]) {
        if (decoderStep < maxLength && h->GetWord() != EOS_ID) {
          survivors.push_back(h);

          histories->SetActive(batchId, true);
        } else {
          --beamSizes[batchId];
        }
      }
    }

    if (survivors.size() == 0) {
      return false;
    }

    for (unsigned i = 0; i < scorers_.size(); i++) {
      scorers_[i]->AssembleBeamState(*nextStates[i], survivors, *states[i]);
    }

    //cerr << "survivors=" << survivors.size() << endl;
    prevHyps.swap(survivors);
    return true;
}


States Search::NewStates() const {
  States states;
  for (auto& scorer : scorers_) {
    states.emplace_back(scorer->NewState());
  }
  return states;
}

void Search::FilterTargetVocab(const Sentences& sentences) {
  unsigned vocabSize = scorers_[0]->GetVocabSize();
  std::set<Word> srcWords;
  for (unsigned i = 0; i < sentences.size(); ++i) {
    const Sentence& sentence = sentences.Get(i);
    for (const auto& srcWord : sentence.GetWords()) {
      srcWords.insert(srcWord);
    }
  }

  filterIndices_ = filter_->GetFilteredVocab(srcWords, vocabSize);
  for (auto& scorer : scorers_) {
    scorer->Filter(filterIndices_);
  }
}
/*
void Search::BatchStats()
{
  unsigned sum = 0;
  for (size_t i = 0; i < activeCount_.size(); ++i) {
    sum += activeCount_[i];
  }

  cerr << "batches: ";
  for (size_t i = 0; i < activeCount_.size(); ++i) {
      cerr << ((float) activeCount_[i] / (float) sum) << " ";
  }
  cerr << endl;
}
*/

}

