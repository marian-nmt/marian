#include <sstream>
#include "base_matrix.h"

namespace amunmt {

std::string BaseMatrix::Debug() const
{
  std::stringstream strm;
  strm << Rows() << "x" << Cols();
  return strm.str();
}

}

