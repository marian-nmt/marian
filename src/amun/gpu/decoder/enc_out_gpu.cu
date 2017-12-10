#include <vector>
#include "enc_out_gpu.h"
#include "gpu/mblas/matrix_functions.h"

using namespace std;

namespace amunmt {
namespace GPU {

EncOutGPU::EncOutGPU(SentencesPtr sentences)
:EncOut(sentences)
{
  size_t tab = 0;
  size_t maxSentenceLength = sentences->GetMaxLength();

  //cerr << "1dMapping=" << mblas::Debug(dMapping, 2) << endl;
  std::vector<uint> hSentenceLengths(sentences->size());

  for (size_t i = 0; i < sentences->size(); ++i) {
    const Sentence &sentence = sentences->Get(i);
    hSentenceLengths[i] = sentence.GetWords(tab).size();
  }

  sentenceLengths_.NewSize(sentences->size(), 1, 1, 1);
  mblas::copy(thrust::raw_pointer_cast(hSentenceLengths.data()),
              hSentenceLengths.size(),
              sentenceLengths_.data(),
              cudaMemcpyHostToDevice);

  //cerr << "sentenceLengths_=" << sentenceLengths_.Debug(2) << endl;
  //cerr << "sentencesMask_=" << sentencesMask_.Debug(2) << endl;
}

}
}

