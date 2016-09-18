#pragma once

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

#include <unordered_map>
#include <string>
#include <vector>

class Vocab
{
public:
  Vocab() {
    GetOrCreate("__UNK__");
    GetOrCreate("__PAD__");
    GetOrCreate("__EOS__");
  }
  virtual ~Vocab() {}

public:
        size_t Size() const { return coll_.size(); }
        size_t Get(const std::string &word) const;
        size_t GetOrCreate(const std::string &word);
	std::vector<size_t> ProcessSentence(const std::string &sentence);

	size_t GetUNK() const { return Get("__UNK__"); }
	size_t GetPAD() const { return Get("__PAD__"); }
	size_t GetEOS() const { return Get("__EOS__"); }
protected:
	typedef std::unordered_map<std::string, size_t> Coll;
	Coll coll_;
};

