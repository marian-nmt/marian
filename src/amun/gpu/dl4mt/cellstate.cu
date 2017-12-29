#include <sstream>
#include "cellstate.h"

using namespace std;

namespace amunmt {
namespace GPU {

std::string CellState::Debug(size_t verbosity) const
{
  std::stringstream strm;

  strm << "output=" << output->Debug(0) << " cell=" << cell->Debug(0);

  return strm.str();
}

}
}
