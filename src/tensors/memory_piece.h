#pragma once

namespace marian {

class MemoryPiece {
  private:
    uint8_t* data_;
    size_t size_;

  public:
    MemoryPiece(uint8_t* data, size_t size)
      : data_(data), size_(size) {}

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

    friend std::ostream& operator<<(std::ostream& out, const MemoryPiece mp) {
      out << "MemoryPiece - ptr: " << std::hex << (size_t)mp.data()
        << std::dec << " size: " << mp.size();
      return out;
    }
};

}