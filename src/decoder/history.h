#pragma once

#include "hypothesis.h"

class History {
  private:
    struct HypothesisCoord {
      bool operator<(const HypothesisCoord& hc) const {
        return cost < hc.cost;
      }
      
      size_t i;
      size_t j;
      float cost;
    };
  
  public:
    void Add(const Beam& beam, bool last = false) {
      for(size_t j = 0; j < beam.size(); ++j)
        if(beam[j].GetWord() == EOS || last)
          topHyps_.push({ history_.size(), j, beam[j].GetCost() });
      history_.push_back(beam);
    }
    
    size_t size() const {
      return history_.size();
    }
    
    NBestList NBest(size_t n) const {
      NBestList nbest;
      auto topHypsCopy = topHyps_;
      while(nbest.size() < n && !topHypsCopy.empty()) {
        auto bestHypCoord = topHypsCopy.top();
        topHypsCopy.pop();
        
        size_t start = bestHypCoord.i;
        size_t j  = bestHypCoord.j;
        
        Sentence targetWords;
        for(int i = start; i >= 0; i--) {
          auto& bestHyp = history_[i][j];
          targetWords.push_back(bestHyp.GetWord());
          j = bestHyp.GetPrevStateIndex();
        }
      
        std::reverse(targetWords.begin(), targetWords.end());
        nbest.emplace_back(targetWords, history_[bestHypCoord.i][bestHypCoord.j]);
      }
      return nbest;
    }
    
    Result Top() const {
      return NBest(1)[0];
    }
    
  private:
    std::vector<Beam> history_;
    mutable std::priority_queue<HypothesisCoord> topHyps_;
      
};
