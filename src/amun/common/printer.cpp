#include "printer.h"

using namespace std;

namespace amunmt {

std::string GetNematusAlignmentString(const HypothesisPtr& hypothesis, std::string best, std::string source, size_t linenum) {
  std::vector<SoftAlignment> aligns;
  HypothesisPtr last = hypothesis;
  while (last->GetPrevHyp().get() != nullptr) {
    aligns.push_back(*(last->GetAlignment(0)));
    last = last->GetPrevHyp();
  }
  //<Sentence Number> ||| <Translation> ||| 0 ||| <Source> ||| <Source word count> <Translation word count>
  std::stringstream firstline;
  int srcspaces = std::count_if(source.begin(), source.end(), [](unsigned char c){ return std::isspace(c); });
  
  firstline << linenum << " ||| " << best << " ||| " << hypothesis->GetCost() / aligns.size() * -1 
  << " ||| " << source << " ||| " << srcspaces+2 << " " << aligns.size();

  std::stringstream alignString;
  for (auto it = aligns.rbegin(); it != aligns.rend(); ++it) {
    alignString << "\n";
    for (size_t i = 0; i < srcspaces+2; ++i) {
      if (i>0) alignString << " ";
      alignString << (*it)[i];
    }
  }
  alignString << "\n";

  return firstline.str() + alignString.str();
}

}

