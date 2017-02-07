#include <sstream>
#include "hypo_info.h"

using namespace std;

namespace amunmt {

std::string AmunOutput::Debug() const
{
  stringstream strm;
 /*
  strm << " words=" << words.size()
      << " lastWord=" << lastWord
      << " prevStates=" << prevStates.size()
      << " nextStates=" << nextStates.size()
      << " score=" << score;
 */
  return strm.str();
}

}

