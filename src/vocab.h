#pragma once

#include <unordered_map>
#include <string>
#include <vector>

class Vocab
{
public:
	size_t GetOrCreate(const std::string &word);
	std::vector<size_t> ProcessSentence(const std::string &sentence);

	size_t GetUNK() const;
	size_t GetPad() const;
	size_t GetEOS() const;
protected:
	typedef std::unordered_map<std::string, size_t> Coll;
	Coll coll_;
};

