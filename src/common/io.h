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

void getYamlFromModel(YAML::Node& yaml,
                      const std::string& varName,
                      const std::string& fName);

void addMetaToItems(const std::string& meta,
                    const std::string& varName,
                    std::vector<io::Item>& items);

std::vector<Item> loadItems(const std::string& fName);

std::vector<Item> loadItems(const void* ptr);

std::vector<Item> mmapItems(const void* ptr);

void saveItems(const std::string& fname, const std::vector<Item>& items);

}
}
