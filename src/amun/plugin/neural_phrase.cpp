#include <iostream>
#include <sstream>
#include "neural_phrase.h"

using namespace std;

namespace amunmt {

std::string NeuralPhrase::Debug() const
{
  stringstream strm;
  for (size_t i = 0; i < words.size(); ++i) {
    strm << words[i] << " ";
  }
  return strm.str(); 
}

}


