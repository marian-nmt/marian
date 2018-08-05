#pragma once

#include "common/definitions.h"
#include "common/file_stream.h"
#include "common/definitions.h"
#include "common/types.h"
#include "common/io_item.h"

#include <string>

// Increase this if binary format changes
#define BINARY_FILE_VERSION 1

namespace marian {
namespace io {

namespace binary {

struct Header {
  size_t nameLength;
  size_t type;
  size_t shapeLength;
  size_t dataLength;
};

template <typename T>
const T* get(const void*& current, size_t num = 1) {
  const T* ptr = (const T*)current;
  current = (const T*)current + num;
  return ptr;
}

static void loadItems(const void* current,
                      std::vector<io::Item>& items,
                      bool mapped = false) {

  size_t binaryFileVersion = *get<size_t>(current);
  ABORT_IF(binaryFileVersion != BINARY_FILE_VERSION,
           "Binary file versions do not match: {} (file) != {} (expected)",
           binaryFileVersion,
           BINARY_FILE_VERSION);

  size_t numHeaders = *get<size_t>(current);
  const Header* headers = get<Header>(current, numHeaders);

  items.resize(numHeaders);
  for(int i = 0; i < numHeaders; ++i) {
    items[i].type = (Type)headers[i].type;
    items[i].name = get<char>(current, headers[i].nameLength);
    items[i].mapped = mapped;
  }

  for(int i = 0; i < numHeaders; ++i) {
    size_t len = headers[i].shapeLength;
    items[i].shape.resize(len);
    const int* arr = get<int>(current, len);
    std::copy(arr, arr + len, items[i].shape.begin());
  }

  // move by offset bytes
  size_t offset = *get<size_t>(current);
  get<char>(current, offset);

  for(int i = 0; i < numHeaders; ++i) {
    if(items[i].mapped) {
      items[i].ptr = get<char>(current, headers[i].dataLength);
    } else {
      size_t len = headers[i].dataLength;
      const char* ptr = get<char>(current, len);
      items[i].bytes.resize(len);
      std::copy(ptr, ptr + len, items[i].bytes.begin());
    }
  }
}

static void loadItems(const std::string& fName,
                      std::vector<io::Item>& items) {

  // Read file into buffer
  size_t fileSize = boost::filesystem::file_size(fName);
  char* ptr = new char[fileSize];
  InputFileStream in(fName);
  ((std::istream&)in).read(ptr, fileSize);

  // Load items from buffer without mapping
  loadItems(ptr, items, false);

  // Delete buffer
  delete[] ptr;
}

static io::Item getItem(const std::string& fName,
                        const std::string& vName) {

  std::vector<io::Item> items;
  loadItems(fName, items);

  for(auto& item : items)
    if(item.name == vName)
      return item;

  return io::Item();
}


template <typename T>
static void write(OutputFileStream& out, size_t& pos, T* data, size_t num = 1) {
  ((std::ostream&)out).write((char*)data, num * sizeof(T));
  pos += num * sizeof(T);
}

static void saveItems(const std::string fName,
                      const std::vector<io::Item>& items) {
  OutputFileStream out(fName);
  size_t pos = 0;

  size_t binaryFileVersion = BINARY_FILE_VERSION;
  write(out, pos, &binaryFileVersion);

  std::vector<Header> headers;
  for(const auto& item : items) {
    headers.push_back(Header{item.name.size() + 1,
                             (size_t)item.type,
                             item.shape.size(),
                             item.size()});
  }

  size_t headerSize = headers.size();
  write(out, pos, &headerSize);
  write(out, pos, headers.data(), headers.size());

  // Write out all names
  for(const auto& item : items) {
    write(out, pos, item.name.data(), item.name.size() + 1);
  }
  // Write out all shapes
  for(const auto& item : items) {
    write(out, pos, item.shape.data(), item.shape.size());
  }

  // align to next 256-byte boundary
  size_t nextpos = ((pos + sizeof(size_t)) / 256 + 1) * 256;
  size_t offset = nextpos - pos - sizeof(size_t);

  write(out, pos, &offset);
  for(size_t i = 0; i < offset; i++) {
    char padding = 0;
    write(out, pos, &padding);
  }

  // Write out all values
  for(const auto& item : items) {
    write(out, pos, item.data(), item.size());
  }
}

}
}
}
