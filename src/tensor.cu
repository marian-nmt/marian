#include <fstream>
#include "tensor.h"

using namespace std;

namespace marian {

void Tensor::Load(const std::string &path)
{
  fstream strm;
  strm.open(path.c_str());

  string line;
  while ( getline (strm, line) )
  {
	cerr << line << '\n';
  }
  strm.close();

}

}

