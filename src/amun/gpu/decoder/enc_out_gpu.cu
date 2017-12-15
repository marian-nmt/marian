#include <vector>
#include "enc_out_gpu.h"
#include "gpu/mblas/matrix_functions.h"

using namespace std;

namespace amunmt {
namespace GPU {

EncOutGPU::EncOutGPU(SentencesPtr sentences)
:EncOut(sentences)
,sentenceLengths_(sentences->size())
{
  mblas::copy(h_sentenceLengths_.data(),
              h_sentenceLengths_.size(),
              sentenceLengths_.data(),
              cudaMemcpyHostToDevice);
}

EncOutGPU::~EncOutGPU()
{
  /*
  cerr << "sentenceLengths_="
      << &sentenceLengths_ << " "
      << sentenceLengths_.size()
      << endl;
  */
  //cerr << "~EncOutGPU" << endl;
}

}
}

