#pragma once

#include <vector>

#include "common/options.h"
#include "common/utils.h"
#include "data/alignment.h"
#include "data/vocab.h"
#include "translator/history.h"
#include "translator/hypothesis.h"

namespace marian {

class OutputPrinter {
public:
  OutputPrinter(Ptr<Options> options, Ptr<Vocab> vocab)
      : vocab_(vocab),
        reverse_(options->get<bool>("right-left")),
        nbest_(options->get<bool>("n-best", false)
                   ? options->get<size_t>("beam-size")
                   : 0),
        alignment_(options->get<std::string>("alignment", "")),
        alignmentThreshold_(getAlignmentThreshold(alignment_)) {}

  template <class OStream>
  void print(Ptr<History> history, OStream& best1, OStream& bestn) {
    const auto& nbl = history->nBest(nbest_);

    for(size_t i = 0; i < nbl.size(); ++i) {
      const auto& result = nbl[i];
      const auto& hypo = std::get<1>(result);
      auto words = std::get<0>(result);

      if(reverse_)
        std::reverse(words.begin(), words.end());

      std::string translation = vocab_->decode(words);
      bestn << history->getLineNum() << " ||| " << translation;

      if(!alignment_.empty())
        bestn << " ||| " << getAlignment(hypo);

      bestn << " |||";
      if(hypo->getScoreBreakdown().empty()) {
        bestn << " F0=" << hypo->getPathScore();
      } else {
        for(size_t j = 0; j < hypo->getScoreBreakdown().size(); ++j) {
          bestn << " F" << j << "= " << hypo->getScoreBreakdown()[j];
        }
      }

      float realScore = std::get<2>(result);
      bestn << " ||| " << realScore;

      if(i < nbl.size() - 1)
        bestn << std::endl;
      else
        bestn << std::flush;
    }

    auto result = history->top();
    auto words = std::get<0>(result);

    if(reverse_)
      std::reverse(words.begin(), words.end());

    std::string translation = vocab_->decode(words);

    best1 << translation;
    if(!alignment_.empty()) {
      const auto& hypo = std::get<1>(result);
      best1 << " ||| " << getAlignment(hypo);
    }

    best1 << std::flush;
  }

private:
  Ptr<Vocab> vocab_;
  bool reverse_{false};
  size_t nbest_{0};
  std::string alignment_;
  float alignmentThreshold_{0.f};

  std::string getAlignment(const Hypothesis::PtrType& hyp);

  float getAlignmentThreshold(const std::string& str) {
    try {
      return std::max(std::stof(str), 0.f);
    } catch(...) {
      return 0.f;
    }
  }
};
}  // namespace marian
