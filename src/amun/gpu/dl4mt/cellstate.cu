#include <sstream>
#include "cellstate.h"

using namespace std;

namespace amunmt {
namespace GPU {

std::string CellState::Debug(unsigned verbosity) const
{
  stringstream strm;
  strm << "output=" << output->Debug(verbosity);
  strm << " cell=" << (cell == NULL ? "NULL" : cell->Debug(verbosity));
  return strm.str();

}

}
}
