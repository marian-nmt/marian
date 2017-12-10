#include "history.h"
#include "sentences.h"

using namespace std;

namespace amunmt {

History::History(const Sentence &sentence, bool normalizeScore, size_t maxLength)
  : normalize_(normalizeScore),
    lineNo_(sentence.GetLineNum()),
   maxLength_(maxLength)
{
  Add({HypothesisPtr(new Hypothesis(sentence))});
}

}

