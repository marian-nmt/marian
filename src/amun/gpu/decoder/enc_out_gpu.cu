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


  //cerr << "sentenceLengths_=" << sentenceLengths_.Debug(2) << endl;
  //cerr << "sentencesMask_=" << sentencesMask_.Debug(2) << endl;
}

}
}

