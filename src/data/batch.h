#pragma once

#include <vector>

namespace marian {
namespace data {

class Batch {
public:
  virtual size_t size() const = 0;
  virtual size_t words() const { return 0; };

  const std::vector<size_t>& getSentenceIds() const { return sentenceIds_; }
  void setSentenceIds(const std::vector<size_t>& ids) { sentenceIds_ = ids; }

protected:
  std::vector<size_t> sentenceIds_;
};
}
}
