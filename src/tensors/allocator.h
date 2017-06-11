#pragma once

#include <cstdint>
#include <deque>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <vector>

namespace marian {

template <class T>
using Ptr = std::shared_ptr<T>;

template <class T>
using UPtr = std::unique_ptr<T>;

template <class T>
using Weak = std::weak_ptr<T>;

template <class T, typename... Args>
Ptr<T> New(Args&&... args) {
  return Ptr<T>(new T(std::forward<Args>(args)...));
}

template <class T>
Ptr<T> New(Ptr<T> p) {
  return Ptr<T>(p);
}

class AllocationException : public std::exception {
public:
  virtual const char* what() const throw() {
    return "Memory re-allocation attempted";
  }
};

class Gap {
  private:
    uint8_t* data_;
    size_t size_;

  public:
    Gap(uint8_t* data, size_t size)
      : data_(data), size_(size) {}

    uint8_t* data() const { return data_; }
    uint8_t* data() { return data_; }

    size_t size() const { return size_; }

    bool operator<(const Gap& mp) const {
      return (size_ < mp.size()) || (size_ == mp.size() && data_ < mp.data());
    }

    bool operator==(const Gap& mp) const {
      return data_ == mp.data() && size_ == mp.size();
    }

    bool adjacent(const Gap& mp) const {
      return data_ + size_ == mp.data() || mp.data() + mp.size() == data_;
    }

    friend Gap operator+(const Gap& mp1, const Gap& mp2) {
      return Gap(mp1.data(), mp1.size() + mp2.size());
    }

    Gap combine(const Gap& mp) const {
      if(mp < *this)
        return mp + *this;
      else
        return *this + mp;
    }

    Gap rest(size_t offset) const {
      return Gap(data_ + offset, size_ - offset);
    }
};

class AllocatorBase : public std::enable_shared_from_this<AllocatorBase> {
  public:
    virtual void free(uint8_t*, size_t) = 0;
};

class MemoryPiece {
  private:
    uint8_t* data_;
    size_t size_;
    Ptr<AllocatorBase> allocator_;

  public:
    MemoryPiece(uint8_t* data, size_t size, Ptr<AllocatorBase> allocator)
      : data_(data), size_(size), allocator_(allocator) {}

    ~MemoryPiece() {
      free();
    }

    uint8_t* data() const { return data_; }
    uint8_t* data() { return data_; }
    size_t size() const { return size_; }

    void set(uint8_t* data, size_t size) {
      data_ = data;
      size_ = size;
    }

    void setPtr(uint8_t* data) {
      data_ = data;
    }

    void free() {
      if(data_) {
        allocator_->free(data_, size_);
        data_ = nullptr;
        size_ = 0;
      }
    }

    Ptr<AllocatorBase> allocator() { return allocator_; }
};

template <class Device>
class Allocator : public AllocatorBase {
  private:
    Device device_;
    size_t available_;
    size_t step_;

    std::set<Gap> gaps_;
    std::unordered_map<uint8_t*, Weak<MemoryPiece>> allocated_;

    void grow(size_t add) {
      uint8_t* oldData = device_.data();
      size_t oldSize = device_.size();

      std::cerr << oldSize + add << std::endl;
      device_.reserve(oldSize + add);

      std::set<Gap> oldGaps;
      gaps_.swap(oldGaps);

      for(auto gap : oldGaps)
        gaps_.insert(Gap(device_.data() + std::distance(oldData, gap.data()), gap.size()));
      insertGap(Gap(device_.data() + oldSize, add));

      std::unordered_map<uint8_t*, Weak<MemoryPiece>> oldAllocated;
      allocated_.swap(oldAllocated);
      for(auto it : oldAllocated) {
        uint8_t* newPtr = device_.data() + std::distance(oldData, it.first);
        allocated_[newPtr] = oldAllocated[it.first];
        allocated_[newPtr].lock()->setPtr(newPtr);
      }
    }

    Gap getGap(size_t size) {
      auto it = std::lower_bound(gaps_.begin(), gaps_.end(),
                                 Gap(nullptr, size));
      while(it == gaps_.end()) {
        grow(step_);
        it = std::lower_bound(gaps_.begin(), gaps_.end(),
                              Gap(nullptr, size));
      }

      //if(it == gaps_.end()) {
      //  throw AllocationException();
      //}

      available_ -= it->size();
      return *it;
    }

    void insertGap(Gap gap, bool consolidate = true) {
      available_ += gap.size();
      if(consolidate) {
        auto it = gaps_.begin();
        std::vector<decltype(it)> adjacent;
        while(it != gaps_.end()) {
          if(gap.adjacent(*it)) {
            gap = gap.combine(*it);
            adjacent.push_back(it);
          }
          it++;
        }
        for(auto&& a : adjacent)
          gaps_.erase(a);
      }
      gaps_.insert(gap);
    }

  public:

    Allocator(size_t deviceNo, size_t bytes, size_t step)
    : device_(deviceNo), step_(step), available_(0) {
      reserve(bytes);
    }

    void reserve(size_t bytes) {
      device_.reserve(bytes);
      clear();
    }

    Ptr<MemoryPiece> alloc(size_t bytes) {
      Gap gap = getGap(bytes);

      gaps_.erase(gap);
      if(gap.size() > bytes) {
        insertGap(gap.rest(bytes), false);
      }

      auto ptr = gap.data();
      auto mp = New<MemoryPiece>(ptr, bytes, shared_from_this());
      allocated_[ptr] = mp;

      return mp;
    }

    void free(uint8_t* ptr, size_t size) {
      auto it = allocated_.find(ptr);
      if(it != allocated_.end()) {
        allocated_.erase(ptr);
        insertGap(Gap(ptr, size), true);
      }
    }

    void clear() {
      available_ = 0;
      gaps_.clear();
      allocated_.clear();
      insertGap({device_.data(), device_.size()}, false);
    }

    size_t size() { return device_.size(); }

    size_t available() { return available_; }
};

}
