#include <fstream>
#include "tensor.h"

using namespace std;

namespace marian {

inline std::vector<std::string> Tokenize(const std::string& str,
    const std::string& delimiters = " \t")
{
  std::vector<std::string> tokens;
  // Skip delimiters at beginning.
  std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
  // Find first "non-delimiter".
  std::string::size_type pos     = str.find_first_of(delimiters, lastPos);

  while (std::string::npos != pos || std::string::npos != lastPos) {
    // Found a token, add it to the vector.
    tokens.push_back(str.substr(lastPos, pos - lastPos));
    // Skip delimiters.  Note the "not_of"
    lastPos = str.find_first_not_of(delimiters, pos);
    // Find next "non-delimiter"
    pos = str.find_first_of(delimiters, lastPos);
  }

  return tokens;
}

//! convert string to variable of type T. Used to reading floats, int etc from files
template<typename T>
T Scan(const std::string &input)
{
  std::stringstream stream(input);
  T ret;
  stream >> ret;
  return ret;
}

//! convert vectors of string to vectors of type T variables
template<typename T>
inline std::vector<T> Scan(const std::vector< std::string > &input)
{
  std::vector<T> output(input.size());
  for (size_t i = 0 ; i < input.size() ; i++) {
    output[i] = Scan<T>( input[i] );
  }
  return output;
}

//! tokenise input string to vector of type T
template<typename T>
inline std::vector<T> Tokenize( const std::string &input
                                , const std::string& delimiters = " \t")
{
  std::vector<std::string> stringVector = Tokenize(input, delimiters);
  return Scan<T>( stringVector );
}


void Tensor::Load(const std::string &path)
{
  size_t totSize = std::accumulate(pimpl_->shape().begin(), pimpl_->shape().end(),
		  1, std::multiplies<int>());
  cerr << "totSize=" << totSize << endl;
  std::vector<float> hostData(totSize);

  fstream strm;
  strm.open(path.c_str());

  string line;
  size_t ind = 0;
  while ( getline (strm, line) )
  {
	cerr << line << '\n';
	vector<Float> toks = Tokenize<Float>(line);
	for (size_t i = 0; i < toks.size(); ++i) {
		hostData[ind] = toks[i];
	}

	++ind;
  }
  strm.close();


}

}

