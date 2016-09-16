#pragma once

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

