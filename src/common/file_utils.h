#pragma once

#include <string>
#include <vector>

#include "common/file_stream.h"

namespace marian {
namespace fileutils {

void cut(const std::string& tsvIn,
         Ptr<io::TemporaryFile> tsvOut,
         const std::vector<size_t>& fields,
         size_t numFields,
         const std::string& sep = "\t");

}  // namespace utils
}  // namespace marian
