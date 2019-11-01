#pragma once

#include "common/timer.h"

#include <chrono>
#include <functional>
#include <memory>
#include <vector>

namespace marian {

class AutoTunerRecorder {
public:
  virtual void start(size_t hash) = 0;
  virtual void stop(size_t hash, bool) = 0;
};

template <typename Return, typename... Args>
class AutoTuner : public AutoTunerRecorder {
private:
  typedef std::function<Return(Args...)> Algorithm;

  // When the autotuner decides the fastest algorithm for a specific tensor operation (e.g. GEMM),
  // the autotuner runs each algorithm at least this 'collectStatMax' number of times and
  // collects the statistics.
  const size_t collectStatMax = 50;
  UPtr<timer::CPUTimer> timer_;

  // This structure holds a hash key an algorithm function (e.g. int16, packed gemm, mkl gemm)
  // for a specific operation size
  // hash: a unique hash key for each operation size
  //      (e.g. m, n, k, transpose A, transpose B, bias size for GEMM)
  // algorithm: a function that holds an algorithm
  struct HashedAlgorithm {
    size_t hash;
    Algorithm algorithm;
  };

  // This structure represents the collected statistics.
  // time: total accumulated time of this operator execution with the given algorithm
  // runs: total time this algorithm was executed
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
        if(stat.runs < collectStatMax)
          return i;

        if(stat.time < bestTime) {
          bestTime = stat.time;
          best = i;
        }
      } else {
        // collect more stats
        return i;
      }
    }

    for(auto& a : algorithms_)
      done_[a.hash] = best;

    return best;
  }

public:
  void insert(const HashedAlgorithm& ha) { algorithms_.push_back(ha); }

  void clear() { algorithms_.clear(); }

  Return run(Args... args) { return algorithms_[choose()].algorithm(args...); }

  void start(size_t hash) override {
    if(!timer_ && done_.count(hash) == 0)
      timer_.reset(new timer::CPUTimer());
  }

  void stop(size_t hash, bool stop) override {
    if(stop && done_.count(hash) == 0) {
      timer_->stop();

      auto seconds = timer_->elapsed();

      auto it = stats_.find(hash);
      if(it != stats_.end()) {
        if(it->second.runs < collectStatMax) {
          it->second.time += seconds;
          it->second.runs += 1;
        }
      } else {
        stats_.emplace(hash, Stat({seconds, 1}));
      }

      timer_.reset(nullptr);
    }
  }
};

}  // namespace marian
