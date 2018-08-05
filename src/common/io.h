#pragma once

#include "3rd_party/cnpy/cnpy.h"
#include "3rd_party/yaml-cpp/yaml.h"
#include "common/shape.h"
#include "common/types.h"

//#include "common/binary.h"

#include <string>

namespace marian {

namespace io {

struct Item {
  std::vector<char> bytes;
  char* ptr{0};
  bool mapped{false};

  std::string name;
  Shape shape;
  Type type{Type::float32};

  char* data() {
    if(mapped)
      return ptr;
    else
      return bytes.data();
  }

  size_t size() {
    return shape.elements() * sizeOf(type);
  }
};

static bool isNpz(const std::string& name) {
  return name.size() >= 4 && name.substr(name.length() - 4) == ".npz";
}

static bool isBin(const std::string& name) {
  return name.size() >= 4 && name.substr(name.length() - 4) == ".bin";
}

static void getYamlFromNpz(YAML::Node& yaml,
                           const std::string& varName,
                           const std::string& fName) {
  yaml = YAML::Load(cnpy::npz_load(fName, varName)->data());
}

static void getYamlFromBin(YAML::Node& yaml,
                           const std::string& varName,
                           const std::string& fName) {
  ABORT("Not implemented");
  //yaml = YAML::Load(cnpy::npz_load(fName, varName)->data());
}

static void getYamlFromModel(YAML::Node& yaml,
                             const std::string& varName,
                             const std::string& fName) {
  if(io::isNpz(fName)) {
    io::getYamlFromNpz(yaml, varName, fName);
  }
  else if(io::isBin(fName)) {
    io::getYamlFromBin(yaml, varName, fName);
  }
  else {
    ABORT("Unknown model file format for file {}", fName);
  }
}

// helper to serialize a YAML::Node to a Yaml string in a 0-terminated character
// vector
static std::vector<char> asYamlCharVector(const YAML::Node node) {
  YAML::Emitter out;
  OutputYaml(node, out);
  return std::vector<char>(out.c_str(), out.c_str() + strlen(out.c_str()) + 1);
}

static void addMetaToNpz(const std::string& meta,
                         const std::string& varName,
                         const std::string& fName) {
  // YAML::Node's Yaml representation is saved as a 0-terminated char vector to
  // the NPZ file
  unsigned int shape = meta.size();
  cnpy::npz_save(fName, varName, meta.data(), &shape, 1, "a");
}

static void addMetaToBin(const std::string& meta,
                         const std::string& varName,
                         const std::string& fName) {
  // YAML::Node's Yaml representation is saved as a 0-terminated char vector to
  // the NPZ file
  unsigned int shape = meta.size();
  ABORT("Not implemented");
}


// same as AddYamlToNpz() but adds to an in-memory NpzItem vector instead
static void addMetaToItems(const std::string& meta,
                           const std::string& varName,
                           std::vector<io::Item>& items) {
  Item item;

  item.name = varName;

  // increase size by 1 to add \0
  item.shape = Shape({(int)meta.size() + 1});

  item.bytes.resize(item.shape[0]);
  std::copy(meta.begin(), meta.end() + item.shape[0], item.bytes.begin());

  item.type = Type::int8;

  items.push_back(item);
}

static void loadItemsFromNpz(const std::string& fName, std::vector<Item>& items) {
    auto numpy = cnpy::npz_load(fName);
    for(auto it : numpy) {

      Shape shape;
      if(it.second->shape.size() == 1) {
        shape.resize(2);
        shape.set(0, 1);
        shape.set(1, it.second->shape[0]);
      } else {
        shape.resize(it.second->shape.size());
        for(size_t i = 0; i < it.second->shape.size(); ++i)
          shape.set(i, it.second->shape[i]);
      }

      Item item;
      item.name = it.first;
      item.shape = shape;
      item.bytes.swap(it.second->bytes);

      items.emplace_back(std::move(item));
    }
}

static std::vector<Item> loadItems(const std::string& fName) {
  std::vector<Item> items;
  if(isNpz(fName)) {
    loadItemsFromNpz(fName, items);
  }
  else if(isBin(fName)) {
    ABORT("Not implemented");
  }
  else {
    ABORT("Unknown model file format for file {}", fName);
  }

  return items;
}

// @TODO: make cnpy and our wrapper talk to each other in terms of types
// or implement our own saving routines for npz based on npy, probably better.
static void saveItemsNpz(const std::string& fname, const std::vector<Item>& items) {
  std::vector<cnpy::NpzItem> npzItems;
  for(auto& item : items) {
    std::vector<unsigned int> shape(item.shape.begin(), item.shape.end());
    if(item.type == Type::float32)
      npzItems.push_back(cnpy::NpzItem(item.name,
                                       item.bytes,
                                       shape,
                                       cnpy::map_type(typeid(float)),
                                       sizeOf(Type::float32)));
    else if(item.type == Type::int8) {
      npzItems.push_back(cnpy::NpzItem(item.name,
                                       item.bytes,
                                       shape,
                                       cnpy::map_type(typeid(char)),
                                       sizeOf(Type::int8)));
    }
    else {
      ABORT("Type currently not supported");
    }
  }
  cnpy::npz_save(fname, npzItems);
}

static void saveItemsBin(const std::string& fname, const std::vector<Item>& items) {
    ABORT("Not implemented");
}

static void saveItems(const std::string& fname, const std::vector<Item>& items) {
  if(isNpz(fname)) {
    saveItemsNpz(fname, items);
  }
  else if(isBin(fname)) {
    saveItemsBin(fname, items);
  }
  else {
    ABORT("Unknown file format for file {}", fname);
  }
}

}
}
