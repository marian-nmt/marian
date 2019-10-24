#pragma once

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

class TemporaryFile {
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
  TemporaryFile(const std::string base = "/tmp/", bool earlyUnlink = true);

  ~TemporaryFile();

  int getFileDescriptor() { return fd_; }

  std::string getFileName() { return name_; }
};

// A streambuf to read from a file descriptor.
class ReadFDBuf : public std::streambuf {
public:
  // Does not take ownership of file.
  explicit ReadFDBuf(int fd, std::size_t buffer_size = 4096);

private:
  int_type underflow() override;

  // If the putback goes below the buffer, try to seek backwards.
  int_type pbackfail(int c = EOF) override;

  // TODO: override setbuf.

  // Read some amount into [Begin(), End()), returning the amount read.
  ssize_t Read();

  char* Begin() { return &mem_.front(); }
  char* End() { return &mem_.back() + 1; }

  int fd_;
  std::vector<char> mem_;

  ReadFDBuf(const ReadFDBuf&) = delete;
  ReadFDBuf& operator=(const ReadFDBuf&) = delete;
};

// A streambuf to Write to a file descriptor.
class WriteFDBuf : public std::streambuf {
public:
  explicit WriteFDBuf(int fd, std::size_t buffer_size = 4096);

  ~WriteFDBuf();

private:
  int_type overflow(int c = EOF) override;

  // Write everything in the buffer to the file.
  int sync() override;

  // Write part of the buffer, returning the amount written.
  ssize_t WriteSome(const char* from, const char* to);

  char* Begin() { return &mem_.front(); }
  char* End() { return &mem_.back() + 1; }

  int fd_;
  std::vector<char> mem_;

  WriteFDBuf(const WriteFDBuf&) = delete;
  WriteFDBuf& operator=(const WriteFDBuf&) = delete;
};

// lseek(fd, 0, SEEK_SET) but with error checking.
void RewindFile(int fd);

class InputFileStream {
public:
  explicit InputFileStream(const std::string& file);

  explicit InputFileStream(TemporaryFile& tempfile);

  explicit InputFileStream(std::istream& strm) : istream_(new std::istream(strm.rdbuf())) {}

  operator std::istream&() { return *istream_; }

  operator bool() { return (bool)*istream_; }

  bool bad() const { return istream_->bad(); }

  bool fail() const { return istream_->fail(); }

  char widen(char c) { return istream_->widen(c); }

  std::string path() { return file_.string(); }

  bool empty() { return istream_->peek() == std::ifstream::traits_type::eof(); }

  void setbufsize(size_t size) const;

  template <typename T>
  friend InputFileStream& operator>>(InputFileStream& stream, T& t) {
    *stream.istream_ >> t;
    // bad() seems to be correct here. Should not abort on EOF.
    ABORT_IF(stream.bad(),
             "Error {} ({}) reading from file '{}'",
             errno,
             strerror(errno),
             stream.path());
    return stream;
  }

  template <typename T>
  size_t read(T* ptr, size_t num = 1) {
    istream_->read((char*)ptr, num * sizeof(T));
    // fail() seems to be correct here. Failure to read should abort.
    ABORT_IF(fail(), "Error {} ({}) reading from file '{}'", errno, strerror(errno), path());
    return num * sizeof(T);
  }

private:
  marian::filesystem::Path file_;
  std::unique_ptr<std::istream> istream_;

  mutable std::vector<char> readBuf_;  // for setbuf()
  std::unique_ptr<ReadFDBuf> temporary_reader_;
};

// wrapper around std::getline() that handles Windows input files with extra CR
// chars at the line end
static inline InputFileStream& getline(InputFileStream& in, std::string& line)
{
  std::getline((std::istream&)in, line);
  // bad() seems to be correct here. Should not abort on EOF.
  ABORT_IF(in.bad(), "Error reading from file '{}'", in.path());
  // strip terminal CR if present
  if(in && !line.empty() && line.back() == in.widen('\r'))
    line.pop_back();
  return in;
}

// wrapper around std::getline() that handles Windows input files with extra CR
// chars at the line end
// To be pedantic, shouldn't this require delim == '\n' to consume a '\r'?
static inline InputFileStream& getline(InputFileStream& in, std::string& line, char delim) {
  std::getline((std::istream&)in, line, delim);
  // bad() seems to be correct here. Should not abort on EOF.
  ABORT_IF(in.bad(), "Error reading from file '{}'", in.path());
  // strip terminal CR if present
  if(in && !line.empty() && line.back() == in.widen('\r'))
    line.pop_back();
  return in;
}

class OutputFileStream {
public:
  OutputFileStream(const std::string& file);

  OutputFileStream(TemporaryFile& tempfile);

  OutputFileStream(std::ostream& strm) { ostream_ = std::make_unique<std::ostream>(strm.rdbuf()); }

  operator std::ostream&() { return *ostream_; }

  operator bool() { return (bool)*ostream_; }

  bool bad() const { return ostream_->bad(); }

  bool fail() const { return ostream_->fail(); }

  template <typename T>
  friend OutputFileStream& operator<<(OutputFileStream& stream, const T& t) {
    *stream.ostream_ << t;
    // fail() seems to be correct here. Failure to write should abort.
    ABORT_IF(stream.fail(), "Error writing to file '{}'", stream.path());
    return stream;
  }

  // handle things like std::endl which is actually a function not a value
  friend OutputFileStream& operator<<(OutputFileStream& stream,
                                      std::ostream& (*var)(std::ostream&)) {
    *stream.ostream_ << var;
    // fail() seems to be correct here. Failure to write should abort.
    ABORT_IF(stream.fail(), "Error writing to file '{}'", stream.path());
    return stream;
  }

  template <typename T>
  size_t write(const T* ptr, size_t num = 1) {
    ostream_->write((char*)ptr, num * sizeof(T));
    // fail() seems to be correct here. Failure to write should abort.
    ABORT_IF(fail(), "Error writing to file '{}'", path());
    return num * sizeof(T);
  }

  std::string path() { return file_.string(); }

private:
  marian::filesystem::Path file_;
  std::unique_ptr<std::ostream> ostream_;

  std::unique_ptr<WriteFDBuf> temporary_writer_;
};

}  // namespace io
}  // namespace marian
