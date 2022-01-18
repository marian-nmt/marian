#include "common/io.h"

#include "3rd_party/cnpy/cnpy.h"
#include "common/shape.h"
#include "common/types.h"

#include "common/binary.h"
#include "common/io_item.h"

namespace marian {
namespace io {

bool isNpz(const std::string& fileName) {
  return fileName.size() >= 4
         && fileName.substr(fileName.length() - 4) == ".npz";
}

bool isBin(const std::string& fileName) {
  return fileName.size() >= 4
         && fileName.substr(fileName.length() - 4) == ".bin";
}

void getYamlFromNpz(YAML::Node& yaml,
                    const std::string& varName,
                    const std::string& fileName) {
  auto item = cnpy::npz_load(fileName, varName);
  if(item->size() > 0)
    yaml = YAML::Load(item->data());
}

void getYamlFromBin(YAML::Node& yaml,
                    const std::string& varName,
                    const std::string& fileName) {
  auto item = binary::getItem(fileName, varName);
  if(item.size() > 0)
    yaml = YAML::Load(item.data());
}

void getYamlFromModel(YAML::Node& yaml,
                      const std::string& varName,
                      const std::string& fileName) {
  if(io::isNpz(fileName)) {
    io::getYamlFromNpz(yaml, varName, fileName);
  } else if(io::isBin(fileName)) {
    io::getYamlFromBin(yaml, varName, fileName);
  } else {
    ABORT("Unknown model file format for file {}", fileName);
  }
}

void getYamlFromModel(YAML::Node& yaml,
                      const std::string& varName,
                      const void* ptr) {
  auto item = binary::getItem(ptr, varName);
  if(item.size() > 0)
    yaml = YAML::Load(item.data());
}

// Load YAML from item
void getYamlFromModel(YAML::Node& yaml,
                      const std::string& varName,
                      const std::vector<Item>& items) {
    for(auto& item : items) {
      if(item.name == varName) {
        yaml = YAML::Load(item.data());
        return;
      }
    }
}

void addMetaToItems(const std::string& meta,
                    const std::string& varName,
                    std::vector<io::Item>& items) {
  Item item;
  item.name = varName;

  // increase size by 1 to add \0
  item.shape = Shape({(int)meta.size() + 1});

  item.bytes.resize(item.shape.elements());
  std::copy(meta.begin(), meta.end(), item.bytes.begin());
  // set string terminator
  item.bytes.back() = '\0';

  item.type = Type::int8;

  items.push_back(item);
}

void loadItemsFromNpz(const std::string& fileName, std::vector<Item>& items) {
  auto numpy = cnpy::npz_load(fileName);
  for(auto it : numpy) {
    Shape shape;
    shape.resize(it.second->shape.size());
    for(size_t i = 0; i < it.second->shape.size(); ++i)
      shape.set(i, (size_t)it.second->shape[i]);

    Item item;
    item.name = it.first;
    item.shape = shape;

    char npzType = it.second->type;
    int wordSize = it.second->word_size;
    if     (npzType == 'f' && wordSize == 2) item.type = Type::float16;
    else if(npzType == 'f' && wordSize == 4) item.type = Type::float32;
    else if(npzType == 'f' && wordSize == 8) item.type = Type::float64;
    else if(npzType == 'i' && wordSize == 1) item.type = Type::int8;
    else if(npzType == 'i' && wordSize == 2) item.type = Type::int16;
    else if(npzType == 'i' && wordSize == 4) item.type = Type::int32;
    else if(npzType == 'i' && wordSize == 8) item.type = Type::uint64;
    else if(npzType == 'u' && wordSize == 1) item.type = Type::uint8;
    else if(npzType == 'u' && wordSize == 2) item.type = Type::uint16;
    else if(npzType == 'u' && wordSize == 4) item.type = Type::uint32;
    else if(npzType == 'u' && wordSize == 8) item.type = Type::uint64;
    else ABORT("Numpy item '{}' type '{}' with size {} not supported", it.first, npzType, wordSize);

    item.bytes.swap(it.second->bytes);
    items.emplace_back(std::move(item));
  }
}

std::vector<Item> loadItems(const std::string& fileName) {
  std::vector<Item> items;
  if(isNpz(fileName)) {
    loadItemsFromNpz(fileName, items);
  } else if(isBin(fileName)) {
    binary::loadItems(fileName, items);
  } else {
    ABORT("Unknown model file format for file {}", fileName);
  }

  return items;
}

std::vector<Item> loadItems(const void* ptr) {
  std::vector<Item> items;
  binary::loadItems(ptr, items, false);
  return items;
}

std::vector<Item> mmapItems(const void* ptr) {
  std::vector<Item> items;
  binary::loadItems(ptr, items, true);
  return items;
}

// @TODO: make cnpy and our wrapper talk to each other in terms of types
// or implement our own saving routines for npz based on npy, probably better.
void saveItemsNpz(const std::string& fileName, const std::vector<Item>& items) {
  std::vector<cnpy::NpzItem> npzItems;
  for(auto& item : items) {
    std::vector<unsigned int> shape(item.shape.begin(), item.shape.end());
    char type;

    if     (item.type == Type::float16) type = cnpy::map_type(typeid(float)); // becomes 'f', correct size is given below
    else if(item.type == Type::float32) type = cnpy::map_type(typeid(float));
    else if(item.type == Type::float64) type = cnpy::map_type(typeid(double));
    else if(item.type == Type::int8)    type = cnpy::map_type(typeid(int8_t));
    else if(item.type == Type::int16)   type = cnpy::map_type(typeid(int16_t));
    else if(item.type == Type::int32)   type = cnpy::map_type(typeid(int32_t));
    else if(item.type == Type::int64)   type = cnpy::map_type(typeid(int64_t));
    else if(item.type == Type::uint8)   type = cnpy::map_type(typeid(uint8_t));
    else if(item.type == Type::uint16)  type = cnpy::map_type(typeid(uint16_t));
    else if(item.type == Type::uint32)  type = cnpy::map_type(typeid(uint32_t));
    else if(item.type == Type::uint64)  type = cnpy::map_type(typeid(uint64_t));
    else ABORT("Other types ({}) not supported", item.type);
      
    npzItems.emplace_back(item.name, item.bytes, shape, type, sizeOf(item.type));
  }
  cnpy::npz_save(fileName, npzItems);
}

void saveItems(const std::string& fileName, const std::vector<Item>& items) {
  if(isNpz(fileName)) {
    saveItemsNpz(fileName, items);
  } else if(isBin(fileName)) {
    binary::saveItems(fileName, items);
  } else {
    ABORT("Unknown file format for file {}", fileName);
  }
}

}  // namespace io
}  // namespace marian
