#include <sstream>
#include "cellstate.h"

using namespace std;

namespace amunmt {
namespace GPU {

std::string CellState::Debug(unsigned verbosity) const
{
  std::stringstream strm;

  strm << "output=" << output->Debug(verbosity) << " cell=" << cell->Debug(verbosity);

  return strm.str();
}

}
}
