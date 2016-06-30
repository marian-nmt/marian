#pragma once

#include <queue>
#include <boost/pool/object_pool.hpp>

#include "god.h"
#include "hypothesis.h"

class History {
  friend std::ostream& operator<<(std::ostream &out, const History &obj);

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
    History()
     : pool_(new boost::object_pool<Hypothesis>()),
       normalize_(God::Get<bool>("normalize"))
    {}
    
    History(History &&h)
     : pool_(h.pool_),
       history_(std::move(h.history_)),
       topHyps_(std::move(h.topHyps_)),
       normalize_(h.normalize_)
    {}
    
    template <class ...Args>
    Hypothesis* NewHypothesis(Args&& ...args) {
      return pool_->construct(std::make_tuple(args...));
    }
        
    void Add(const Beam& beam, bool last = false) {
      if(beam.back()->GetPrevHyp() != nullptr) {
        for(size_t j = 0; j < beam.size(); ++j)
          if(beam[j]->GetWord() == EOS || last) {
            float cost = normalize_ ? beam[j]->GetCost() / history_.size() : beam[j]->GetCost();
            topHyps_.push({ history_.size(), j, cost });
          }
      }
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
        
        Words targetWords;
        HypothesisPtr bestHyp = history_[start][j];
        while(bestHyp->GetPrevHyp() != nullptr) {
          //std::cerr << "bestHyp=" << *bestHyp << std::endl;
          targetWords.push_back(bestHyp->GetWord());
          bestHyp = bestHyp->GetPrevHyp();
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
    std::shared_ptr<boost::object_pool<Hypothesis>> pool_;

    std::vector<Beam> history_;
    std::priority_queue<HypothesisCoord> topHyps_;
    bool normalize_;  
};
