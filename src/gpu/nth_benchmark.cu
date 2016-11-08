
#include <cmath>
#include <memory>
#include <sstream>
#include <random>
#include <algorithm>

#include <boost/timer/timer.hpp>
#include "gpu/mblas/matrix.h"
#include "gpu/nth_element.h"

using namespace thrust::placeholders;

struct ProbCompare {
  ProbCompare(const float* data) : data_(data) {}

  __host__ __device__
  bool operator()(const unsigned a, const unsigned b) {
    return data_[a] > data_[b];
  }

  const float* data_;
};

int main(int argc, char** argv) {

  size_t beamSize = 5;
  size_t vocSize = 30000;
  size_t tests = 1000;

  std::vector<float> rands(beamSize * vocSize);
  std::random_device rnd_device;
  std::mt19937 mersenne_engine(rnd_device());
  std::uniform_real_distribution<float> dist(-100, -4);

  auto gen = std::bind(dist, mersenne_engine);
  std::generate(std::begin(rands), std::end(rands), gen);

  rands[10000 + 30000 * 0] = -.1;
  rands[10001 + 30000 * 1] = -.1;
  rands[10002 + 30000 * 2] = -.1;
  rands[10003 + 30000 * 3] = -.1;
  rands[10004 + 30000 * 4] = -.1;

  DeviceVector<float> ProbsOrig(beamSize * vocSize);
  thrust::copy(rands.begin(), rands.end(), ProbsOrig.begin());

  DeviceVector<float> Probs(beamSize * vocSize);

  DeviceVector<unsigned> keys(Probs.size());
  HostVector<unsigned> bestKeys(beamSize);
  HostVector<float> bestCosts(beamSize);

  while(0) {
    boost::timer::cpu_timer timer;
    for(int i = 0; i < tests; ++i) {
      thrust::copy(ProbsOrig.begin(), ProbsOrig.end(), Probs.begin());

      thrust::sequence(keys.begin(), keys.end());
      thrust::nth_element(keys.begin(), keys.begin() + beamSize, keys.end(),
                          ProbCompare(thrust::raw_pointer_cast(Probs.data())));

      //for(int i = 0; i < beamSize; ++i) {
      //  bestKeys[i] = keys[i];
      //  bestCosts[i] = Probs[keys[i]];
      //}
    }
    std::cerr << "Search took " << timer.format(3, "%ws");
  }

  {
    boost::timer::cpu_timer timer;
    for(int i = 0; i < tests; ++i) {
      thrust::copy(ProbsOrig.begin(), ProbsOrig.end(), Probs.begin());

      for(size_t j = 0; j < beamSize; ++j) {
        DeviceVector<float>::iterator iter =
          algo::max_element(Probs.begin(), Probs.end());
        bestKeys[j] = iter - Probs.begin();
        bestCosts[j] = *iter;
        *iter = std::numeric_limits<float>::lowest();
      }
      algo::copy(bestKeys.begin(), bestKeys.end(), keys.begin());
    }
    std::cerr << "Search took " << timer.format(3, "%ws") << std::endl;
  }

  {
    boost::timer::cpu_timer timer;
    for(int i = 0; i < tests; ++i) {
      thrust::copy(ProbsOrig.begin(), ProbsOrig.end(), Probs.begin());

      thrust::sequence(keys.begin(), keys.end());
      thrust::sort_by_key(Probs.begin(), Probs.end(),
                          keys.begin(), algo::greater<float>());

      algo::copy_n(keys.begin(), beamSize, bestKeys.begin());
      algo::copy_n(Probs.begin(), beamSize, bestCosts.begin());
    }
    std::cerr << "Search took " << timer.format(3, "%ws") << std::endl;
  }
}
