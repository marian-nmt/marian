#pragma once

#include <cstdint>
#include <deque>
#include <memory>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/definitions.h"
#include "tensors/device.h"
#include "tensors/memory_piece.h"
#include "tensors/types.h"

namespace marian {

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
  Gap(uint8_t* data, size_t size) : data_(data), size_(size) {}

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

  friend std::ostream& operator<<(std::ostream& out, const Gap& gap) {
    out << "gap - ptr: " << std::hex << (size_t)gap.data() << std::dec
        << " size: " << gap.size();
    return out;
  }

  Gap combine(const Gap& mp) const {
    if(mp.data() < this->data())
      return mp + *this;
    else
      return *this + mp;
  }

  Gap rest(size_t offset) const { return Gap(data_ + offset, size_ - offset); }
};

class Allocator {
private:
  Ptr<Device> device_;
  size_t available_{0};
  size_t step_{128 * 1024 * 1024};
  size_t alignment_{256};
  bool throw_{false};

  std::set<Gap> gaps_;
  std::unordered_map<uint8_t*, Ptr<MemoryPiece>> allocated_;

  size_t align(size_t size) {
    return ceil(size / (float)alignment_) * alignment_;
  }

  void grow(size_t add) {
    add = align(add);
    uint8_t* oldData = device_->data();
    size_t oldSize = device_->size();

    device_->reserve(oldSize + add);

    std::set<Gap> oldGaps;
    gaps_.swap(oldGaps);

    for(auto gap : oldGaps)
      gaps_.insert(Gap(device_->data() + std::distance(oldData, gap.data()),
                       gap.size()));
    insertGap(Gap(device_->data() + oldSize, add));

    std::unordered_map<uint8_t*, Ptr<MemoryPiece>> oldAllocated;
    allocated_.swap(oldAllocated);
    for(auto it : oldAllocated) {
      uint8_t* newPtr = device_->data() + std::distance(oldData, it.first);
      allocated_[newPtr] = oldAllocated[it.first];
      allocated_[newPtr]->setPtr(newPtr);
    }
  }

  Gap getGap(size_t size) {
    size = align(size);
    auto it = std::lower_bound(gaps_.begin(), gaps_.end(), Gap(nullptr, size));

    if(throw_ && it == gaps_.end()) {
      throw AllocationException();
    }

    while(it == gaps_.end()) {
      grow(step_);
      it = std::lower_bound(gaps_.begin(), gaps_.end(), Gap(nullptr, size));
    }

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
  Allocator(DeviceId deviceId,
            size_t bytes,
            size_t step,
            size_t alignment = 256)
      : device_(DispatchDevice(deviceId, alignment)),
        step_(step),
        available_(0),
        alignment_(alignment) {
    reserve(bytes);
  }

  void throwAtReallocation(bool throwRealloc) { throw_ = throwRealloc; }

  void reserve(size_t bytes) {
    bytes = align(bytes);
    if(bytes > 0)
      device_->reserve(bytes);
    clear();
  }

  template <typename T>
  size_t capacity(size_t num) {
    return align(num * sizeof(T));
  }

  size_t capacity(size_t num, Type type) {
    return align(num * sizeOf(type));
  }


  Ptr<MemoryPiece> alloc(size_t num, Type type) {
    return alloc(num * sizeOf(type));
  }


  template <typename T>
  Ptr<MemoryPiece> alloc(size_t num) {
    return alloc(capacity<T>(num));
  }

  Ptr<MemoryPiece> alloc(size_t bytes) {
    bytes = align(bytes);
    Gap gap = getGap(bytes);

    gaps_.erase(gap);
    if(gap.size() > bytes) {
      insertGap(gap.rest(bytes), false);
    }

    auto ptr = gap.data();
    auto mp = New<MemoryPiece>(ptr, bytes);
    allocated_[ptr] = mp;
    return mp;
  }

  bool free(uint8_t* ptr, size_t bytes) {
    bytes = align(bytes);

    ABORT_IF(ptr == 0, "Double free?");

    if(!ptr)
      return false;

    auto it = allocated_.find(ptr);
    if(it != allocated_.end()) {
      allocated_.erase(ptr);
      insertGap(Gap(ptr, bytes), true);
      return true;
    }
    return false;
  }

  bool free(Ptr<MemoryPiece> mp) {
    if(free(mp->data(), mp->size())) {
      mp->set(nullptr, 0);
      return true;
    }
    return false;
  }

  void clear() {
    available_ = 0;
    gaps_.clear();
    allocated_.clear();
    insertGap({device_->data(), device_->size()}, false);
  }

  Ptr<MemoryPiece> memory() {
    return New<MemoryPiece>(device_->data(), device_->size());
  }

  size_t size() { return device_->size(); }

  size_t available() { return available_; }

  DeviceId getDevice() { return device_->getDevice(); }
};
}
