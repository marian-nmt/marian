#include <sstream>
#include "base_matrix.h"

namespace amunmt {

std::string BaseMatrix::Debug(bool detailed) const
{
  std::stringstream strm;
  strm << Rows() << "x" << Cols();
  return strm.str();
}

}

