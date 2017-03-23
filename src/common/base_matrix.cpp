#include <sstream>
#include "base_matrix.h"

namespace amunmt {

size_t BaseMatrix::size() const {
  size_t ret = dim(0);
  for (size_t i = 1; i < SHAPE_SIZE; ++i) {
    ret *= dim(i);
  }

  return ret;
}

std::string BaseMatrix::Debug(bool detailed) const
{
  std::stringstream strm;
  strm << dim(0) << "x" << dim(1) << "x" << dim(2) << "x" << dim(3);
  return strm.str();
}

}

