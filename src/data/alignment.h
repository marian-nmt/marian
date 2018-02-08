#pragma once

#include "data/corpus_base.h"

namespace marian {
namespace data {

class WordAlignment {
private:
  typedef std::pair<int, int> Point;
  typedef std::vector<Point> Alignment;

  std::vector<Alignment> data_;

public:
  WordAlignment(const std::string& fname) {
    InputFileStream aStream(fname);
    std::string line;
    size_t c = 0;

    LOG(info, "[data] Loading word alignment from {}", fname);

    while(std::getline((std::istream&)aStream, line)) {
      data_.emplace_back();
      std::vector<std::string> atok = split(line, " -");
      for(size_t i = 0; i < atok.size(); i += 2)
        data_.back().emplace_back(std::stoi(atok[i]), std::stoi(atok[i + 1]));
      c++;
    }

    LOG(info, "[data] Done");
  }

  std::vector<std::string> split(const std::string& input,
                                 const std::string& chars) {
    std::vector<std::string> output;
    boost::split(output, input, boost::is_any_of(chars));
    return output;
  }

  void guidedAlignment(Ptr<CorpusBatch> batch) {
    int srcWords = batch->front()->batchWidth();
    int trgWords = batch->back()->batchWidth();
    int dimBatch = batch->getSentenceIds().size();
    std::vector<float> guided(dimBatch * srcWords * trgWords, 0.f);

    for(int b = 0; b < dimBatch; ++b) {
      auto& alignment = data_[batch->getSentenceIds()[b]];
      for(auto& p : alignment) {
        int sid, tid;
        std::tie(sid, tid) = p;

        size_t idx = b + sid * dimBatch + tid * srcWords * dimBatch;
        guided[idx] = 1.f;
      }
    }
    batch->setGuidedAlignment(guided);
  }
};
}
}
