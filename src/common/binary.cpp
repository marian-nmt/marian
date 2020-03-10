#include "common/binary.h"
#include "common/definitions.h"
#include "common/file_stream.h"
#include "common/io_item.h"
#include "common/types.h"

#include <string>

namespace marian {
namespace io {

namespace binary {

struct Header {
  size_t nameLength;
  size_t type;
  size_t shapeLength;
  size_t dataLength;
};

// cast current void pointer to T pointer and move forward by num elements 
template <typename T>
const T* get(const void*& current, size_t num = 1) {
  const T* ptr = (const T*)current;
  current = (const T*)current + num;
  return ptr;
}

void loadItems(const void* current, std::vector<io::Item>& items, bool mapped) {
  size_t binaryFileVersion = *get<size_t>(current);
  ABORT_IF(binaryFileVersion != BINARY_FILE_VERSION,
           "Binary file versions do not match: {} (file) != {} (expected)",
           binaryFileVersion,
           BINARY_FILE_VERSION);

  size_t numHeaders = *get<size_t>(current); // number of item headers that follow
  const Header* headers = get<Header>(current, numHeaders); // read that many headers

  // prepopulate items with meta data from headers
  items.resize(numHeaders);
  for(int i = 0; i < numHeaders; ++i) {
    items[i].type = (Type)headers[i].type;
    items[i].name = get<char>(current, headers[i].nameLength);
    items[i].mapped = mapped;
  }

  // read in actual shape and data
  for(int i = 0; i < numHeaders; ++i) {
    size_t len = headers[i].shapeLength;
    items[i].shape.resize(len); 
    const int* arr = get<int>(current, len); // read shape
    std::copy(arr, arr + len, items[i].shape.begin()); // copy to Item::shape 
  }

  // move by offset bytes, aligned to 256-bytes boundary
  size_t offset = *get<size_t>(current);
  get<char>(current, offset);

  for(int i = 0; i < numHeaders; ++i) {
    if(items[i].mapped) { // memory-mapped, hence only set pointer
      items[i].ptr = get<char>(current, headers[i].dataLength);
    } else { // reading into item data
      size_t len = headers[i].dataLength;
      items[i].bytes.resize(len);
      const char* ptr = get<char>(current, len);
      std::copy(ptr, ptr + len, items[i].bytes.begin());
    }
  }
}

void loadItems(const std::string& fileName, std::vector<io::Item>& items) {
  // Read file into buffer
  size_t fileSize = filesystem::fileSize(fileName);
  std::vector<char> buf(fileSize);
// @TODO: check this again:
#if 1 // for some reason, the #else branch fails with "file not found" in the *read* operation (open succeeds)
  FILE *f = fopen(fileName.c_str(), "rb");
  ABORT_IF(f == nullptr, "Error {} ('{}') opening file '{}'", errno, strerror(errno), fileName);
  auto rc = fread(buf.data(), sizeof(*buf.data()), buf.size(), f);
  ABORT_IF(rc != buf.size(), "Error {} ('{}') reading file '{}'", errno, strerror(errno), fileName);
  fclose(f);
#else
  io::InputFileStream in(fileName);
  in.read(buf.data(), buf.size());
#endif

  // Load items from buffer without mapping
  loadItems(buf.data(), items, false);
}

io::Item getItem(const void* current, const std::string& varName) {
  std::vector<io::Item> items;
  loadItems(current, items);

  for(auto& item : items)
    if(item.name == varName)
      return item;

  return io::Item();
}

io::Item getItem(const std::string& fileName, const std::string& varName) {
  std::vector<io::Item> items;
  loadItems(fileName, items);

  for(auto& item : items)
    if(item.name == varName)
      return item;

  return io::Item();
}

void saveItems(const std::string& fileName,
               const std::vector<io::Item>& items) {
  io::OutputFileStream out(fileName);
  size_t pos = 0;

  size_t binaryFileVersion = BINARY_FILE_VERSION;
  pos += out.write(&binaryFileVersion);

  std::vector<Header> headers;
  for(const auto& item : items) {
    headers.push_back(Header{item.name.size() + 1,
                             (size_t)item.type,
                             item.shape.size(),
                             item.bytes.size()}); // binary item size with padding, will be 256-byte-aligned
  }

  size_t headerSize = headers.size();
  pos += out.write(&headerSize);
  pos += out.write(headers.data(), headers.size());

  // Write out all names
  for(const auto& item : items) {
    pos += out.write(item.name.data(), item.name.size() + 1);
  }
  // Write out all shapes
  for(const auto& item : items) {
    pos += out.write(item.shape.data(), item.shape.size());
  }

  // align to next 256-byte boundary
  size_t nextpos = ((pos + sizeof(size_t)) / 256 + 1) * 256;
  size_t offset = nextpos - pos - sizeof(size_t);

  pos += out.write(&offset);
  for(size_t i = 0; i < offset; i++) {
    char padding = 0;
    pos += out.write(&padding);
  }

  // Write out all values
  for(const auto& item : items)
    pos += out.write(item.data(), item.bytes.size()); // writes out data with padding, keeps 256-byte boundary. 
                                                      // Amazingly this is binary-compatible with V1 and aligned and 
                                                      // non-aligned models can be read with the same procedure.
                                                      // No version-bump required. Gets 5-8% of speed back when mmapped.
}

}  // namespace binary
}  // namespace io
}  // namespace marian
