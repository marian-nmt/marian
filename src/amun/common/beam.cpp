#include <sstream>
#include "beam.h"

using namespace std;

namespace amunmt {


std::string Debug(const Beam &vec, unsigned verbosity)
{
  std::stringstream strm;

  strm << "size=" << vec.size();

  if (verbosity) {
    for (unsigned i = 0; i < vec.size(); ++i) {
      const HypothesisPtr &hypo = vec[i];
      strm << " " << hypo->GetWord();
    }
  }

  return strm.str();
}

std::string Debug(const Beams &vec, unsigned verbosity)
{
  std::stringstream strm;

  strm << "size=" << vec.size();

  if (verbosity) {
    for (unsigned i = 0; i < vec.size(); ++i) {
      const Beam &beam = vec[i];
      strm << endl << "\t" << Debug(beam, verbosity);
    }
  }

  return strm.str();
}

}

