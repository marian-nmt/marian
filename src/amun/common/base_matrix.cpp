#include <sstream>
#include "base_matrix.h"

namespace amunmt {

std::string BaseMatrix::Debug(bool detailed) const
{
  std::stringstream strm;
  strm << dim(0) << "x" << dim(1) << "x" << dim(2) << "x" << dim(3);
  return strm.str();
}

}

