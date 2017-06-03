#pragma once

#include <set>
#include <deque>

#include "common/definitions.h"
#include "tensors/tensor.h"

namespace marian {

class AllocationException : public std::exception {
  public:
    virtual const char* what() const throw() {
      return "Memory re-allocation attempted";
    }
};

class TensorAllocator {
  private:
    const size_t CHUNK  = 512;
    const size_t MBYTE  = 1024 * 1024;
    const size_t FLOATS = CHUNK * MBYTE / sizeof(float);

    DeviceGPU device_;

    typedef std::pair<size_t, float*> Gap;
    std::set<Gap> gaps_;
    Gap lastGap_;

    bool throw_{false};

    std::deque<Tensor> allocated_;

    void reset(Tensor t, float* start) {
      t->reset(start);
    }

    void resetAllocated(float* oldStart) {
      gaps_.clear();
      size_t prev = 0;
      for(auto&& t : allocated_) {
        size_t dist = t->data() - oldStart;
        reset(t, device_.data() + dist);
        if(dist > prev)
          gaps_.emplace(dist - prev, device_.data() + prev);
        prev = dist + t->size();
      }

      if(allocated_.empty()) {
        lastGap_ = { device_.capacity(), device_.data() };
      }
      else {
        size_t used = allocated_.back()->data() - device_.data() + allocated_.back()->size();
        size_t gap = device_.capacity() - used;
        float* addr = allocated_.back()->data() + allocated_.back()->size();
        lastGap_ = { gap, addr };
      }

      gaps_.insert(lastGap_);
    }

    auto getGap(Shape shape) -> decltype(gaps_.begin()) {
      auto it = std::lower_bound(gaps_.begin(), gaps_.end(),
                                 std::make_pair((size_t)shape.elements(), (float*)0));
      return it;
    }

    auto checkSpace(Shape shape) -> decltype(gaps_.begin()) {
      auto gapIt = getGap(shape);
      if(gapIt == gaps_.end()) {
        if(throw_)
          throw AllocationException();
        size_t incr = device_.capacity() - lastGap_.first + shape.elements();
        reserve(device_.capacity() + incr);
        gapIt = gaps_.find(lastGap_);
      }
      return gapIt;
    }

  public:

    TensorAllocator(size_t device)
     : device_(device) {
      lastGap_ = { device_.capacity(), device_.data() };
      gaps_.insert(lastGap_);
    }

    ~TensorAllocator() {
      clear();
    }

    void throwAtReallocation(bool throwRealloc) {
      throw_ = throwRealloc;
    }

    void reserve(size_t elements = 0) {
      float mult = elements / FLOATS + 1;
      LOG(memory, "Extending reserved space to {} MB (device {})",
        mult * CHUNK, device_.getDevice());

      size_t old = device_.capacity();
      float* oldStart = device_.data();
      device_.reserve(mult * FLOATS);
      resetAllocated(oldStart);
    }

    void reserveExact(size_t elements = 0) {
      size_t mbytes = (elements * sizeof(float)) / MBYTE;
      LOG(memory, "Reserving space for {} floats ({} MB, device {})",
        elements, mbytes, device_.getDevice());

      size_t old = device_.capacity();
      float* oldStart = device_.data();
      device_.reserve(elements);
      resetAllocated(oldStart);
    }

    void clear() {
      gaps_.clear();
      lastGap_ = { device_.capacity(), device_.data() };
      gaps_.insert(lastGap_);
      allocated_.clear();
    }

    void allocate(Tensor &t, Shape shape) {
      if(!t || t->shape() != shape) {
        auto it = checkSpace(shape);
        float* start = it->second;
        t.reset(new TensorBase(start, shape, device_.getDevice()));
        allocated_.push_back(t);
        if(it->first > t->size())
          gaps_.emplace(it->first - t->size(), it->second + t->size());
        gaps_.erase(it);
      }
    }

    void free(Tensor& t) {
      auto it = allocated_.rbegin();
      while(it != allocated_.rend()) {
        if(*it == t) {
          Gap gap = { t->size(), t->data() };
          allocated_.erase(std::next(it).base());

          auto it2 = gaps_.begin();
          std::vector<decltype(it2)> adjacent;
          while(it2 != gaps_.end()) {
            if(it2->second + it2->first  == gap.second) {
              gap = { gap.first + it2->first, it2->second };
              adjacent.push_back(it2);
            }
            if(gap.second + gap.first == it2->second) {
              gap = { gap.first + it2->first, gap.second };
              adjacent.push_back(it2);
            }
            it2++;
          }
          for(auto&& a : adjacent)
            gaps_.erase(a);
          gaps_.insert(gap);
          break;
        }
        it++;
      }
      t.reset();
    }

    Tensor asTensor() {
      float* start = device_.data();
      return Tensor(new TensorBase(start, {1, (int)size()}, device_.getDevice()));
    }

    size_t capacity() {
      return device_.capacity();
    }

    size_t size() {
      float* start = device_.data();
      float* end = start;
      if(!allocated_.empty())
        end = allocated_.back()->data() + allocated_.back()->size();

      return end - start;
    }
};

}
