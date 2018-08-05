#pragma once

#include "common/definitions.h"
#include "common/file_stream.h"
#include "common/definitions.h"
#include "common/types.h"

#include <string>

namespace marian {

struct Header {
  size_t nameLength;
  size_t type;
  size_t shapeLength;
  size_t dataLength;
};

struct BinaryTensor {
  std::string name;
  Shape shape;
  Type type;
  const void* data;
};

class Binary {
private:
  std::vector<BinaryTensor> tensors_;
  const void* current_;

  template <typename T>
  const T* get(size_t num = 1) {
    const T* ptr = (const T*)current_;
    current_ = (const T*)current_ + num;
    return ptr;
  }

public:
  Binary(const void* ptr)
    : current_(ptr) {

    size_t numHeaders = *get<size_t>();
    const Header* headers = get<Header>(numHeaders);

    tensors_.resize(numHeaders);
    for(int i = 0; i < numHeaders; ++i) {
      tensors_[i].type = (Type)headers[i].type;
      tensors_[i].name = get<char>(headers[i].nameLength);
    }

    for(int i = 0; i < numHeaders; ++i) {
      size_t len = headers[i].shapeLength;
      tensors_[i].shape.resize(len);
      const int* arr = get<int>(len);
      std::copy(arr, arr + len, tensors_[i].shape.begin());
    }

    // move by offset bytes
    size_t offset = *get<size_t>();
    get<char>(offset);

    for(int i = 0; i < numHeaders; ++i)
      tensors_[i].data = get<char>(headers[i].dataLength);

  }

  auto begin() -> decltype(tensors_.begin()) {
    return tensors_.begin();
  }

  auto end() -> decltype(tensors_.end()) {
    return tensors_.end();
  }
};

class Binarizer {
private:
  UPtr<OutputFileStream> out_;
  size_t pos_;

  typedef std::tuple<std::string, Tensor> NamedTensor;
  std::vector<NamedTensor> tensors;

public:
  Binarizer(const std::string& name)
    : out_{new OutputFileStream(name)}, pos_(0) {}

  void add(const std::string& pName, const Tensor& tensor) {
    tensors.emplace_back(pName, tensor);
  }

  template <typename T>
  void write(T* data, size_t num = 1) {
    ((std::ostream&)*out_).write((char*)data, num * sizeof(T));
    pos_ += num * sizeof(T);
  }

  void save() {

    std::vector<Header> headers;
    for(const auto& nt : tensors) {
      std::string name; Tensor tensor;
      std::tie(name, tensor) = nt;

      //headers.push_back(Header{name.size() + 1,
      //                         (size_t)tensor->type(),
      //                         tensor->shape().size(),
      //                         tensor->memory()->size()});
    }

    size_t headerSize = headers.size();
    write(&headerSize);
    write(headers.data(), headers.size());

    //// Write out all names
    //for(const auto& nt : tensors) {
    //  const std::string& name = std::get<0>(nt);
    //  write(name.data(), name.size() + 1);
    //}
    //
    //// Write out all shapes
    //for(const auto& nt : tensors) {
    //  const Shape shape = std::get<1>(nt)->shape();
    //  write(shape.data(), shape.size());
    //}
    //
    //// align to next 256-byte boundary
    //size_t nextpos = ((pos_ + sizeof(size_t)) / 256 + 1) * 256;
    //size_t offset = nextpos - pos_ - sizeof(size_t);
    //
    //write(&offset);
    //for(size_t i = 0; i < offset; i++) {
    //   char dummy = 0;
    //   write(&dummy);
    //}
    //
    //// Write out all values
    //for(const auto& nt : tensors) {
    //  const Tensor tensor = std::get<1>(nt);
    //  ABORT_IF(tensor->getDevice().type != DeviceType::cpu, "Only CPU models can be binarized");
    //  write(tensor->memory()->data(), tensor->memory()->size());
    //}

  }

};

}
