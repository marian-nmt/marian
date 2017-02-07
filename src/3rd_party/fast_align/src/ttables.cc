#include "src/ttables.h"

#include <string>
#include <fstream>

#include "src/corpus.h"

using namespace std;

void TTable::DeserializeLogProbsFromText(istream* in, Dict& d) {
  int c = 0;
  string e;
  string f;
  double p;
  while(*in) {
    (*in) >> e >> f >> p;
    if (e.empty()) break;
    ++c;
    unsigned ie = d.Convert(e);
    if (ie >= static_cast<int>(ttable.size())) ttable.resize(ie + 1);
    ttable[ie][d.Convert(f)] = exp(p);
  }
  cerr << "Loaded " << c << " translation parameters.\n";
}

