#pragma once

#include <fstream>
#include <iostream>
#include <memory>
#include "common/definitions.h"
#include "common/filesystem.h"
#include "common/logging.h"

// Even when compiling with clang, __GNUC__ may be defined, so
// we need to add some extra checks to avoid compile errors with
// respect to -Wsuggest-override.
#ifdef __GNUC__
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wunused-value"
#  if defined(__has_warning)
#    if __has_warning("-Wsuggest-override")
#      pragma GCC diagnostic ignored "-Wsuggest-override"
#    endif
#  else
#    pragma GCC diagnostic ignored "-Wsuggest-override"
#  endif
#endif

#ifdef _MSC_VER
#pragma warning(push) // 4101: 'identifier' : unreferenced local variable. One parameter variable in zstr.hpp is not used.
#pragma warning(disable : 4101)
#endif
#include "3rd_party/zstr/zstr.hpp"
#ifdef _MSC_VER
#pragma warning(pop)
#endif
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

namespace marian {
namespace io {

//////////////////////////////////////////////////////////////////////////////////////////////
class InputFileStream : public std::istream {
public:
  explicit InputFileStream(const std::string& file);
  virtual ~InputFileStream();

  bool empty();
  void setbufsize(size_t size);
  std::string getFileName() const;

protected:
  marian::filesystem::Path file_;
  std::unique_ptr<std::streambuf> streamBuf1_;  // main streambuf
  std::unique_ptr<std::streambuf> streamBuf2_;  // in case of a .gz file
  FILE* pipe_{};                                // in case of pipe syntax
  std::vector<char> readBuf_;
};

std::istream& getline(std::istream& in, std::string& line);

//////////////////////////////////////////////////////////////////////////////////////////////
class OutputFileStream : public std::ostream {
public:
  explicit OutputFileStream(const std::string& file);
  virtual ~OutputFileStream();

  std::string getFileName() const;

  template <typename T>
  size_t write(const T* ptr, size_t num = 1) {
    std::ostream::write((char*)ptr, num * sizeof(T));
    // fail() seems to be correct here. Failure to write should abort.
    ABORT_IF(fail(), "Error writing to file '{}'", file_.string());
    return num * sizeof(T);
  }

protected:
  explicit OutputFileStream();  // for temp file

  marian::filesystem::Path file_;
  std::unique_ptr<std::streambuf> streamBuf1_;
  std::unique_ptr<std::streambuf> streamBuf2_;
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

  void NormalizeTempPrefix(std::string& base) const;
  void MakeTemp(const std::string& base);

};

}  // namespace io
}  // namespace marian
