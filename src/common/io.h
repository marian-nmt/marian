#pragma once

#include "3rd_party/yaml-cpp/yaml.h"
#include "common/io_item.h"

#include <string>
#include <vector>

// interface for handling model files in marian, both *.npz files and
// *.bin files have the same way of accessing them and are identified
// by suffixes (*.npz or *.bin).

// Files with the *.bin suffix are supposed to be memory-mappable for
// CPU decoding.

namespace marian {
namespace io {

bool isNpz(const std::string& fileName);
bool isBin(const std::string& fileName);

void getYamlFromModel(YAML::Node& yaml, const std::string& varName, const std::string& fileName);
void getYamlFromModel(YAML::Node& yaml, const std::string& varName, const void* ptr);
void getYamlFromModel(YAML::Node& yaml, const std::string& varName, const std::vector<Item>& items);

void addMetaToItems(const std::string& meta,
                    const std::string& varName,
                    std::vector<io::Item>& items);

std::vector<Item> loadItems(const std::string& fileName);
std::vector<Item> loadItems(const void* ptr);

std::vector<Item> mmapItems(const void* ptr);

void saveItems(const std::string& fileName, const std::vector<Item>& items);

/**
 * Creates a flat io::Item from a given std::vector so that it can be saved in a npz file 
 * or Marian's native binary format with the given name.
 */
template <typename T>
Item fromVector(const std::vector<T>& vec, const std::string& name) {
  Item item;
  item.name = std::move(name);
  item.shape = Shape({1, (int)vec.size()}); // @TODO: review if this should be {1, size} or rather just {size}
  item.type = typeId<T>();
  item.bytes.resize(vec.size() * sizeOf(item.type));
  std::copy((char*)vec.data(), (char*)(vec.data() + vec.size()), item.bytes.begin());
  return item;
}

}  // namespace io
}  // namespace marian
