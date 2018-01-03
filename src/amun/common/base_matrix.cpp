#include <sstream>
#include "base_matrix.h"

using namespace std;

namespace amunmt {

unsigned BaseMatrix::size() const {
  unsigned ret = dim(0);
  for (unsigned i = 1; i < SHAPE_SIZE; ++i) {
    ret *= dim(i);
  }

  return ret;
}

std::string BaseMatrix::Debug(unsigned detailed) const
{
  std::stringstream strm;
  strm << dim(0) << "x" << dim(1) << "x" << dim(2) << "x" << dim(3) << "=" << size();
  //strm << " " << this;
  return strm.str();
}

}

