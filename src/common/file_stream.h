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

#ifdef __GNUC__  // not supported; maybe we just need to increment a standard flag in gcc/cmake?
namespace std {
template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
}  // namespace std
#endif

#ifdef _MSC_VER
#include <fcntl.h>
#include <io.h>
#include <stdlib.h>
#endif

namespace marian {
namespace io {

///////////////////////////////////////////////////////////////////////////////////////////////
class TemporaryFile2 {
private:
  int fd_{-1};
  bool unlink_;
  std::string name_;

#ifndef _MSC_VER
  int mkstemp_and_unlink(char* tmpl);
#endif

  int MakeTemp(const std::string& base);

  void NormalizeTempPrefix(std::string& base);

public:
  TemporaryFile2(const std::string base = "/tmp/", bool earlyUnlink = true);

  ~TemporaryFile2();

  int getFileDescriptor() { return fd_; }

  std::string getFileName() { return name_; }
};

///////////////////////////////////////////////////////////////////////////////////////////////
class TemporaryFileNew : public std::fstream {
public:
  TemporaryFileNew(const std::string& base = "/tmp/", bool earlyUnlink = true);
  std::string getFileName();
  virtual ~TemporaryFileNew();

protected:
  std::string name_;
  bool unlink_;

  void NormalizeTempPrefix(std::string& base);
  void MakeTemp(const std::string& base);
};

//////////////////////////////////////////////////////////////////////////////////////////////
class InputFileStreamNew : public std::istream {
public:
  explicit InputFileStreamNew(const std::string& file);
  virtual ~InputFileStreamNew();

  bool empty();
  void setbufsize(size_t size) const;

protected:
  marian::filesystem::Path file_;
  std::streambuf* streamBuf_;
};

//////////////////////////////////////////////////////////////////////////////////////////////
class OutputFileStreamNew : public std::ostream {
public:
  explicit OutputFileStreamNew(const std::string& file);
  virtual ~OutputFileStreamNew();
  
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

}  // namespace io
}  // namespace marian
