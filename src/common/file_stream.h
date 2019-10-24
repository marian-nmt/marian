#pragma once

#include <fstream>
#include "common/definitions.h"
#include "common/filesystem.h"
#include "common/logging.h"

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-override"
#endif
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4101)
#endif
#include "3rd_party/zstr/zstr.hpp"
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <iostream>
#include <memory>
#include <vector>

#ifdef _MSC_VER
#include <fcntl.h>
#include <io.h>
#include <stdlib.h>
#endif

namespace marian {
namespace io {

//////////////////////////////////////////////////////////////////////////////////////////////
class InputFileStream : public std::istream {
public:
  explicit InputFileStream(const std::string& file);
  virtual ~InputFileStream();

  bool empty();
  void setbufsize(size_t size) const;
  std::string getFileName() const;

protected:
  marian::filesystem::Path file_;
  std::streambuf* streamBuf_;
};

std::istream& getline(std::istream& in, std::string& line);

  //////////////////////////////////////////////////////////////////////////////////////////////
class OutputFileStream : public std::ostream {
public:
  explicit OutputFileStream(const std::string& file);
  virtual ~OutputFileStream();
  
  template <typename T>
  size_t write(const T* ptr, size_t num = 1) {
    this->write((char*)ptr, num * sizeof(T));
    // fail() seems to be correct here. Failure to write should abort.
    ABORT_IF(fail(), "Error writing to file '{}'", file_.string());
    return num * sizeof(T);
  }

protected:
  marian::filesystem::Path file_;
  std::streambuf* streamBuf1_;
  std::streambuf* streamBuf2_;
};

///////////////////////////////////////////////////////////////////////////////////////////////
class TemporaryFile : public OutputFileStream {
public:
  TemporaryFile(const std::string& base = "/tmp/", bool earlyUnlink = true);
  virtual ~TemporaryFile();

  UPtr<InputFileStream> getInputStream();
  std::string getFileName() const;

protected:
  bool unlink_;
  UPtr<InputFileStream> inSteam_;

  std::string CreateFileName(const std::string& base) const;

};

}  // namespace io
}  // namespace marian
