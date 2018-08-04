#pragma once

#include "common/definitions.h"
#include "common/file_stream.h"

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

  template <typename T>
  const T* get(const void*& ptr, size_t num = 1) {
    const T* obj = (const T*)ptr;
    ptr = (const T*)ptr + num;
    return obj;
  }

public:
  Binary(const void* ptr) {
    size_t numHeaders = *get<size_t>(ptr);
    const Header* headers = get<Header>(ptr, numHeaders);

    tensors_.resize(numHeaders);
    for(int i = 0; i < numHeaders; ++i)
      tensors_[i].name = get<char>(ptr, headers[i].nameLength);

    for(int i = 0; i < numHeaders; ++i) {
      size_t len = headers[i].shapeLength;
      tensors_[i].shape.resize(len);
      const int* arr = get<int>(ptr, len);
      std::copy(arr, arr + len, tensors_[i].shape.begin());
    }

    // move by offset bytes
    get<char>(ptr, *get<size_t>(ptr));
    
    for(int i = 0; i < numHeaders; ++i)
      tensors_[i].data = get<char>(ptr, headers[i].dataLength);
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

  typedef std::tuple<std::string, Tensor> NamedTensor;
  std::vector<NamedTensor> tensors;

public:
  Binarizer(const std::string& name)
    : out_{new OutputFileStream(name)} {}

  void add(const std::string& pName, const Tensor& tensor) {
    tensors.emplace_back(pName, tensor);
  }

  void saveTensor(const Tensor& tensor) {
    ABORT_IF(tensor->getDevice().type != DeviceType::cpu, "Only CPU models can be binarized");
    ((std::ostream&)*out_).write((char*)tensor->memory()->data(), tensor->memory()->size());
  }

  void save() {
    std::vector<Header> headers;
    for(const auto& nt : tensors) {
      std::string name;
      Tensor tensor;

      std::tie(name, tensor) = nt;
      headers.push_back(Header{name.size(),
                               (size_t)tensor->type(), 
                               tensor->shape().size(),
                               // will be aligned to usable byte boundaries
                               tensor->memory()->size()});
    }

    size_t pos = 0;

    size_t headerSize = headers.size();
    ((std::ostream&)*out_).write((char*)&headerSize, sizeof(headerSize));
    pos += sizeof(headerSize);

    ((std::ostream&)*out_).write((char*)headers.data(), sizeof(Header) * headers.size());
    pos += sizeof(Header) * headers.size();

    // Write out all names
    for(const auto& nt : tensors) {
      const std::string& name = std::get<0>(nt);
      // take \0 into account, hence + 1
      ((std::ostream&)*out_).write((char*)name.data(), name.size() + 1);
      pos += name.size() + 1;
    }

    // Write out all shapes
    for(const auto& nt : tensors) {
      const Shape shape = std::get<1>(nt)->shape();
      ((std::ostream&)*out_).write((char*)shape.data(), shape.size() * sizeof(int));
      pos += shape.size() * sizeof(int);
    }

    // align to next 256-byte boundary
    size_t nextpos = ((pos + sizeof(size_t)) / 256 + 1) * 256;
    size_t offset = nextpos - pos - sizeof(size_t);

    ((std::ostream&)*out_).write((char*)&offset, sizeof(offset));
    for(size_t i = 0; i < offset; i++) {
       char dummy = 0;
       ((std::ostream&)*out_).write(&dummy, 1);
    }

    // Write out all values
    // @TODO: handle types!
    for(const auto& nt : tensors) {
      const Tensor tensor = std::get<1>(nt);
      saveTensor(tensor);
    }

  }

};

}