#include "common/binary.h"
#include "common/definitions.h"
#include "common/file_stream.h"
#include "common/io_item.h"
#include "common/types.h"
#include "tensors/cpu/integer_common.h"

#include <string>

namespace marian {
namespace io {

namespace binary {

struct Header {
  uint64_t nameLength;
  uint64_t type;
  uint64_t shapeLength;
  uint64_t dataLength;
};

// cast current void pointer to T pointer and move forward by num elements 
template <typename T>
const T* get(const void*& current, uint64_t num = 1) {
  const T* ptr = (const T*)current;
  current = (const T*)current + num;
  return ptr;
}

void loadItems(const void* current, std::vector<io::Item>& items, bool mapped) {
  uint64_t binaryFileVersion = *get<uint64_t>(current);
  ABORT_IF(binaryFileVersion != BINARY_FILE_VERSION,
           "Binary file versions do not match: {} (file) != {} (expected)",
           binaryFileVersion,
           BINARY_FILE_VERSION);

  uint64_t numHeaders = *get<uint64_t>(current); // number of item headers that follow
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
    uint64_t len = headers[i].shapeLength;
    items[i].shape.resize(len); 
    const int* arr = get<int>(current, len); // read shape
    std::copy(arr, arr + len, items[i].shape.begin()); // copy to Item::shape 
  }

  // move by offset bytes, aligned to 256-bytes boundary
  uint64_t offset = *get<uint64_t>(current);
  get<char>(current, offset);

  for(int i = 0; i < numHeaders; ++i) {
    // For intgemm AVX512 and AVX512VNNI have the same arangement, but the VNNI algorithm is faster.
    // Change the type to the fastest one supported.
    if (items[i].type == Type::intgemm8avx512) {
      items[i].type = cpu::integer::getIntgemmType(Type::intgemm8);
    }
    if(items[i].mapped) { // memory-mapped, hence only set pointer
      // @TOOD: verify this actually works for the hardware-specific ones like intgemm8avx2
      ABORT_IF(items[i].type == Type::intgemm8 || items[i].type == Type::intgemm16, "mmap format not supported for hardware non-specific intgemm matrices");
      items[i].ptr = get<char>(current, headers[i].dataLength);
    } else { // reading into item data
      uint64_t len = headers[i].dataLength;
      items[i].bytes.resize(len);
      const char* ptr = get<char>(current, len);
      // Intgemm8/16 matrices in binary model are just quantized, however they also need to be reordered
      // Reordering depends on the architecture (SSE/AVX2/AVX512) so we read in the quantized matrices and
      // then reorder them before adding them as a parameter in the graph.
      if (matchType<intgemm8>(items[i].type)) {
        items[i].type = cpu::integer::getIntgemmType(Type::intgemm8);
        cpu::integer::prepareAndTransposeB<Type::intgemm8>(items[i], ptr);
      } else if (matchType<intgemm16>(items[i].type)) {
        items[i].type = cpu::integer::getIntgemmType(Type::intgemm16);
        cpu::integer::prepareAndTransposeB<Type::intgemm16>(items[i], ptr);
      } else {
        std::copy(ptr, ptr + len, items[i].bytes.begin());
      }
    }
  }
}

void loadItems(const std::string& fileName, std::vector<io::Item>& items) {
  // Read file into buffer
  uint64_t fileSize = filesystem::fileSize(fileName);
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
  uint64_t pos = 0;

  uint64_t binaryFileVersion = BINARY_FILE_VERSION;
  pos += out.write(&binaryFileVersion);

  std::vector<Header> headers;
  for(const auto& item : items) {
    headers.push_back(Header{item.name.size() + 1,
                             (uint64_t)item.type,
                             item.shape.size(),
                             item.bytes.size()}); // binary item size with padding, will be 256-byte-aligned
  }

  uint64_t headerSize = headers.size();
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
  uint64_t nextpos = ((pos + sizeof(uint64_t)) / 256 + 1) * 256;
  uint64_t offset = nextpos - pos - sizeof(uint64_t);

  pos += out.write(&offset);
  for(uint64_t i = 0; i < offset; i++) {
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
