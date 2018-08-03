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

public:
  Binary(const void* ptr) {
    const void* current = ptr;
    size_t headerSize = *(const size_t*)current;
  
    tensors_.resize(headerSize);

    current = (const size_t*)current + 1;

    const Header* beg = (const Header*)current;
    const Header* end = (const Header*)current + headerSize;

    current = end;
    for(const Header* it = beg; it != end; it++) {
      int i = std::distance(beg, it);
      tensors_[i].name = (const char*)current;
      current = (const char*)current + it->nameLength + 1;
    }

    for(const Header* it = beg; it != end; it++) {
      int i = std::distance(beg, it);
      Shape shape;
      tensors_[i].shape.resize(it->shapeLength);
      std::copy((const int*)current,
                (const int*)current + it->shapeLength,
                tensors_[i].shape.begin());
      current = (const int*)current + it->shapeLength;
    }

    size_t offset = *(size_t*)current;
    current = (size_t*)current + 1;
    current = (char*)current + offset;

    for(const Header* it = beg; it != end; it++) {
      int i = std::distance(beg, it);
      tensors_[i].data = current;
      current = (const char*)current + it->dataLength;
    }
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