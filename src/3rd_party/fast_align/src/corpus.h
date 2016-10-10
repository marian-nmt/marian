#ifndef CPYPDICT_H_
#define CPYPDICT_H_

#include <string>
#include <iostream>
#include <cassert>
#include <fstream>
#include <vector>
#include <set>
#include "src/hashtables.h"
#include "src/port.h"

class Dict {
 public:
  Dict() : b0_("<bad0>") {
    words_.reserve(1000);
  }

  inline unsigned max() const { return words_.size(); }

  static bool is_ws(char x) {
    return (x == ' ' || x == '\t');
  }

  inline void ConvertWhitespaceDelimitedLine(const std::string& line, std::vector<unsigned>* out) {
    size_t cur = 0;
    size_t last = 0;
    int state = 0;
    out->clear();
    while(cur < line.size()) {
      if (is_ws(line[cur++])) {
        if (state == 0) continue;
        out->push_back(Convert(line.substr(last, cur - last - 1)));
        state = 0;
      } else {
        if (state == 1) continue;
        last = cur - 1;
        state = 1;
      }
    }
    if (state == 1)
      out->push_back(Convert(line.substr(last, cur - last)));
  }

  inline unsigned Convert(const std::string& word, bool frozen = false) {
    MAP_TYPE::iterator i = d_.find(word);
    if (i == d_.end()) {
      if (frozen)
        return 0;
      words_.push_back(word);
      d_[word] = words_.size();
      return words_.size();
    } else {
      return i->second;
    }
  }

  inline const std::string& Convert(const unsigned id) const {
    if (id == 0) return b0_;
    return words_[id-1];
  }
 private:
  std::string b0_;
  std::vector<std::string> words_;
  MAP_TYPE d_;
};

inline void ReadFromFile(const std::string& filename,
                  Dict* d,
                  std::vector<std::vector<unsigned> >* src,
                  std::set<unsigned>* src_vocab) {
  src->clear();
  std::cerr << "Reading from " << filename << std::endl;
  std::ifstream in(filename.c_str());
  assert(in);
  std::string line;
  int lc = 0;
  while(getline(in, line)) {
    ++lc;
    src->push_back(std::vector<unsigned>());
    d->ConvertWhitespaceDelimitedLine(line, &src->back());
    for (unsigned i = 0; i < src->back().size(); ++i) src_vocab->insert(src->back()[i]);
  }
}

#endif
