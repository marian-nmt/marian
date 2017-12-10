#include "history.h"
#include "sentences.h"

using namespace std;

namespace amunmt {

History::History(size_t lineNo, bool normalizeScore, size_t maxLength)
  : normalize_(normalizeScore),
    lineNo_(lineNo),
   maxLength_(maxLength)
{
  Add({HypothesisPtr(new Hypothesis())});
}

}

