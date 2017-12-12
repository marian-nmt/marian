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

  h_sentenceLengths_.resize(sentences->size());

  for (size_t i = 0; i < sentences->size(); ++i) {
    h_sentenceLengths_[i] = sentences->Get(i).GetWords(tab).size();
  }

  sentenceLengths_.newSize(sentences->size());
  mblas::copy(h_sentenceLengths_.data(),
              h_sentenceLengths_.size(),
              sentenceLengths_.data(),
              cudaMemcpyHostToDevice);

  //cerr << "sentenceLengths_=" << sentenceLengths_.Debug(2) << endl;
}

}
}

