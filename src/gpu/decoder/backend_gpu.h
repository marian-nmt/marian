#pragma once

#include "matrix.h"

namespace GPU {

class BackendGPU /* : public Backend */ {
  public:
    template <typename T> 
    using DeviceVector = thrust::device_vector<T>;
    
    template <typename T> 
    using HostVector = thrust::host_vector<T>;
    
    class Payload : public PayloadBase {
      public:
        Payload() {}
        
        mblas::Matrix& operator*() {
          return matrix_;
        }
        
      private:    
        mblas::Matrix matrix_;
    };
    
    
    static void PartialSortByKey(Payload& probs,
                            HostVector<unsigned>& bestKeys,
                            HostVector<float>& bestCosts) {
      size_t beamSize = bestKeys.size();
      if(beamSize < 10) {
        for(size_t i = 0; i < beamSize; ++i) {
          DeviceVector<float>::iterator iter =
            thrust::max_element((*probs).begin(), (*probs).end());
          bestKeys[i] = iter - (*probs).begin();
          bestCosts[i] = *iter;
          *iter = std::numeric_limits<float>::lowest();
        }
      }
      else {
        DeviceVector<unsigned> keys((*probs).size());
        thrust::sequence(keys.begin(), keys.end());
        thrust::sort_by_key((*probs).begin(), (*probs).end(),
                            keys.begin(), thrust::greater<float>());
      
        thrust::copy_n(keys.begin(), beamSize, bestKeys.begin());
        thrust::copy_n((*probs).begin(), beamSize, bestCosts.begin());
      }
    }
    
    template <class It1, class It2>
    static void copy(It1 begin, It1 end, It2 out) {
      thrust::copy(begin, end, out);
    }
};

}

