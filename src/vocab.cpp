#include <limits>
#include "vocab.h"

using namespace std;

////////////////////////////////////////////////////////
inline std::vector<std::string> Tokenize(const std::string& str,
    const std::string& delimiters = " \t")
{
  std::vector<std::string> tokens;
  // Skip delimiters at beginning.
  std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
  // Find first "non-delimiter".
  std::string::size_type pos = str.find_first_of(delimiters, lastPos);

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
////////////////////////////////////////////////////////
size_t Vocab::GetUNK() const
{
	return std::numeric_limits<size_t>::max();
}

size_t Vocab::GetPad() const
{
	return std::numeric_limits<size_t>::max() - 1;
}

size_t Vocab::GetEOS() const
{
	return std::numeric_limits<size_t>::max() - 2;
}


size_t Vocab::GetOrCreate(const std::string &word)
{
	size_t id;
	Coll::const_iterator iter = coll_.find(word);
	if (iter == coll_.end()) {
		id = coll_.size();
		coll_[word] = id;
	}
	else {
		id = iter->second;
	}
	return id;
}

std::vector<size_t> Vocab::ProcessSentence(const std::string &sentence)
{
	vector<string> toks = Tokenize(sentence);
	vector<size_t> ret(toks.size());

	for (size_t i = 0; i < toks.size(); ++i) {
		size_t id = GetOrCreate(toks[i]);
		ret[i] = id;
	}

	return ret;
}
