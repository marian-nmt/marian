#include "common/file_utils.h"
#include "common/utils.h"

namespace marian {
namespace fileutils {

void cut(const std::string& tsvIn,
         Ptr<io::TemporaryFile> tsvOut,
         const std::vector<size_t>& fields,
         size_t numFields,
         const std::string& sep /*= "\t"*/) {
  std::vector<std::string> tsvFields(numFields);
  std::string line;
  io::InputFileStream ioIn(tsvIn);
  while(getline(ioIn, line)) {
    tsvFields.clear();
    utils::splitTsv(line, tsvFields, numFields);  // split tab-separated fields
    for(size_t i = 0; i < fields.size(); ++i) {
      *tsvOut << tsvFields[fields[i]];
      if(i < fields.size() - 1)
        *tsvOut << sep;  // concatenating fields with the custom separator
    }
    *tsvOut << std::endl;
  }
};

}  // namespace fileutils
}  // namespace marian
