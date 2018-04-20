#pragma once

#include <vector>
#include <memory>
#include <functional>
#include <boost/timer/timer.hpp>
#include <boost/chrono.hpp>

namespace marian {

class AutoTunerRecorder {
public:
  virtual void start(size_t hash) = 0;
  virtual void stop(size_t hash, bool) = 0;
};

template <typename Return, typename ...Args>
class AutoTuner : public AutoTunerRecorder {
private:
  typedef std::function<Return(Args...)> Algorithm;

  const size_t max = 100;

  UPtr<boost::timer::cpu_timer> timer_;

  struct HashedAlgorithm {
    size_t hash;
    Algorithm algorithm;
  };

  struct Stat {
    double time;
    size_t runs;
  };

  std::unordered_map<size_t, Stat> stats_;
  std::unordered_map<size_t, size_t> done_;

  std::vector<HashedAlgorithm> algorithms_;

  size_t choose() {
    size_t best = 0;
    double bestTime = std::numeric_limits<double>::max();

    for(size_t i = 0; i < algorithms_.size(); ++i) {
      auto doneIt = done_.find(algorithms_[i].hash);
      if(doneIt != done_.end())
        return doneIt->second;

      auto it = stats_.find(algorithms_[i].hash);
      if(it != stats_.end()) {
        auto& stat = it->second;

        // collect more stats
        if(stat.runs < max)
          return i;

        if(stat.time < bestTime) {
            bestTime = stat.time;
            best = i;
        }
      }
      else {
        // collect more stats
        return i;
      }
    }

    for(auto& a : algorithms_)
      done_[a.hash] = best;

    return best;
  }

public:

    void insert(const HashedAlgorithm& ha) {
      algorithms_.push_back(ha);
    }

    void clear() {
      algorithms_.clear();
    }

    Return run(Args ...args) {
      return algorithms_[choose()].algorithm(args...);
    }

    void start(size_t hash) {
      if(!timer_ && done_.count(hash) == 0)
        timer_.reset(new boost::timer::cpu_timer());
    }

    void stop(size_t hash, bool stop) {
      if(stop && done_.count(hash) == 0) {
        timer_->stop();

        typedef boost::chrono::duration<double> sec;
        sec seconds = boost::chrono::nanoseconds(timer_->elapsed().user);

        auto it = stats_.find(hash);
        if(it != stats_.end()) {
          if(it->second.runs < max) {
              it->second.time += seconds.count();
              it->second.runs += 1;
          }
        }
        else {
          stats_.emplace(hash, Stat({seconds.count(), 1}));
        }

        timer_.reset(nullptr);
      }
    }
};

}
