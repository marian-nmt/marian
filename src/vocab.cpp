// This file is part of the Marian toolkit.
// Marian is copyright (c) 2016 Marcin Junczys-Dowmunt.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

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

size_t Vocab::Get(const std::string &word) const
{
	Coll::const_iterator iter = coll_.find(word);
	return iter->second;
}

std::vector<size_t> Vocab::ProcessSentence(const std::string &sentence)
{
	vector<string> toks = Tokenize(sentence);
	vector<size_t> ret(toks.size());

	for (size_t i = 0; i < toks.size(); ++i) {
		size_t id = GetOrCreate(toks[i]);
		ret[i] = id;
	}
  ret.push_back(GetEOS()); // Append EOS token.

	return ret;
}
