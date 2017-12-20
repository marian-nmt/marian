#include <boost/timer/timer.hpp>
#include "common/search.h"
#include "common/sentences.h"
#include "common/god.h"
#include "common/history.h"
#include "common/filter.h"
#include "common/base_matrix.h"

#ifdef CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

using namespace std;

namespace amunmt {

Search::Search(const God &god)
  : deviceInfo_(god.GetNextDevice()),
    scorers_(god.GetScorers(deviceInfo_, *this)),
    filter_(god.GetFilter()),
    normalizeScore_(god.Get<bool>("normalize")),
    bestHyps_(god.GetBestHyps(deviceInfo_))
{}


Search::~Search() {
  //cerr << "~Search1" << endl;
#ifdef CUDA
  if (deviceInfo_.deviceType == GPUDevice) {
    cudaSetDevice(deviceInfo_.deviceId);
  }
#endif
  scorers_.clear();
  //cerr << "~Search2" << endl;
}

void Search::Translate(SentencesPtr sentences)
{
  assert(scorers_.size() == 1);
//  scorers_[0]->Translate(sentences);
  scorers_[0]->Encode(sentences);
}


void Search::FilterTargetVocab(const Sentences& sentences) {
  size_t vocabSize = scorers_[0]->GetVocabSize();
  std::set<Word> srcWords;
  for (size_t i = 0; i < sentences.size(); ++i) {
    const Sentence& sentence = *sentences.Get(i);
    for (const auto& srcWord : sentence.GetWords()) {
      srcWords.insert(srcWord);
    }
  }

  filterIndices_ = filter_->GetFilteredVocab(srcWords, vocabSize);
  for (auto& scorer : scorers_) {
    scorer->Filter(filterIndices_);
  }
}



}

