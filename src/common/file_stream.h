#pragma once

// @TODO: this file still contains lots of stuff from boost::filesystem and boost::iostreams,
// this has to be figured out.

#include "common/filesystem.h"

#include <boost/filesystem/fstream.hpp>
#include <boost/iostreams/device/file_descriptor.hpp>
#pragma warning(push)
#pragma warning(disable: 4458) // declaration of 'traits_type' hides class member
#pragma warning(disable: 4456) // declaration of 'c' hides previous local declaration
#pragma warning(disable: 4244) // conversion from 'int' to 'char', possible loss of data
#pragma warning(disable: 4706) // assignment within conditional expression
#include <boost/iostreams/filter/gzip.hpp>
#pragma warning(pop)
#include <boost/iostreams/filtering_stream.hpp>
#include <iostream>
#include "common/logging.h"

#ifdef _MSC_VER

#include <fcntl.h>
#include <io.h>
#include <stdlib.h>

#endif

namespace marian {
namespace io {

class TemporaryFile {
private:
  int fd_;
  bool unlink_;
  std::string name_;

#ifndef _MSC_VER
  int mkstemp_and_unlink(char* tmpl) {
    int ret = mkstemp(tmpl);
    if(unlink_ && ret != -1) {
      ABORT_IF(unlink(tmpl), "Error while deleting '{}'", tmpl);
    }
    return ret;
  }
#endif


  int MakeTemp(const std::string& base) {
#ifdef _MSC_VER
    char* name = tempnam(base.c_str(), "marian.");
    ABORT_IF(name == NULL,
      "Error while making a temporary based on '{}'",
      base);

    int oflag = _O_RDWR | _O_CREAT | _O_EXCL;
    if (unlink_) oflag |= _O_TEMPORARY;

    int ret = open(name, oflag, _S_IREAD | _S_IWRITE);
    ABORT_IF(ret == -1,
      "Error while making a temporary based on '{}'",
      base);

    name_ = name;
    free(name);

    return ret;
#else
    std::string name(base);
    name += "marian.XXXXXX";
    name.push_back(0);
    int ret;
    ABORT_IF(-1 == (ret = mkstemp_and_unlink(&name[0])),
      "Error while making a temporary based on '{}'",
      base);
    name_ = name;
    return ret;
#endif
  }

  void NormalizeTempPrefix(std::string& base) {
    if(base.empty())
      return;

#ifdef _MSC_VER
    if(base.substr(0,4) == "/tmp")
      base = getenv("TMP");
#else
    if(base[base.size() - 1] == '/')
      return;
    struct stat sb;
    // It's fine for it to not exist.
    if(stat(base.c_str(), &sb) == - 1)
      return;
    if(S_ISDIR(sb.st_mode))
      base += '/';
#endif
  }

public:
  TemporaryFile(const std::string base = "/tmp/", bool earlyUnlink = true)
      : unlink_(earlyUnlink) {
    std::string baseTemp(base);
    NormalizeTempPrefix(baseTemp);
    fd_ = MakeTemp(baseTemp);
  }

  ~TemporaryFile() {
#ifdef _MSC_VER
    if (fd_ == -1)
      return;

    if(close(fd_)) {
      std::cerr << "Could not close file " << fd_ << std::endl;
      std::abort();
    }

    if(!unlink_) {
      ABORT_IF(remove(name_.c_str()), "Error while deleting '{}'", name_);
    }
#else
    if(fd_ != -1 && !unlink_) {
      ABORT_IF(unlink(name_.c_str()), "Error while deleting '{}'", name_);
    }
    if(fd_ != -1 && close(fd_)) {
      std::cerr << "Could not close file " << fd_ << std::endl;
      std::abort();
    }
#endif
  }

  int getFileDescriptor() { return fd_; }

  std::string getFileName() { return name_; }
};

class InputFileStream {
public:
  InputFileStream(const std::string& file) : file_(file), ifstream_(file_.getBoost()) {
    ABORT_IF(
        !marian::filesystem::exists(file_), "File '{}' does not exist", file);

    if(file_.extension() == marian::filesystem::Path(std::string(".gz")))
      istream_.push(boost::iostreams::gzip_decompressor());
    istream_.push(ifstream_);
  }

  InputFileStream(TemporaryFile& tempfile)
      : fds_(tempfile.getFileDescriptor(), boost::iostreams::never_close_handle) {
    lseek(tempfile.getFileDescriptor(), 0, SEEK_SET);
    istream_.push(fds_, 1024);
  }

  InputFileStream(std::istream& strm) { istream_.push(strm, 0); }

  operator std::istream&() { return istream_; }

  operator bool() { return (bool)istream_; }

  bool bad() const {
    return istream_.bad();
  }

  bool fail() const {
    return istream_.fail();
  }

  char widen(char c) {
    return istream_.widen(c);
  }

  bool isOpen() const {
    return ifstream_.is_open();
  }

  std::string path() { return file_.string(); }

  bool empty() { return ifstream_.peek() == std::ifstream::traits_type::eof(); }

  template <typename T>
  friend InputFileStream& operator>>(InputFileStream& stream, T& t) {
    stream.istream_ >> t;
    // bad() seems to be correct here. Should not abort on EOF.
    ABORT_IF(stream.bad(),
             "Error reading from file '{}'",
             stream.path());
    return stream;
  }

  template <typename T>
  size_t read(T* ptr, size_t num = 1) {
    istream_.read((char*)ptr, num * sizeof(T));
    // fail() seems to be correct here. Failure to read should abort.
    ABORT_IF(fail(),
             "Error reading from file '{}'",
             path());
    return num * sizeof(T);
  }

private:
  marian::filesystem::Path file_;
  boost::filesystem::ifstream ifstream_;
  boost::iostreams::file_descriptor_source fds_;
  boost::iostreams::filtering_istream istream_;
};

// wrapper around std::getline() that handles Windows input files with extra CR
// chars at the line end
static InputFileStream& getline(InputFileStream& in, std::string& line) {
  std::getline((std::istream&)in, line);
  // bad() seems to be correct here. Should not abort on EOF.
  ABORT_IF(in.bad(),
           "Error reading from file '{}'",
           in.path());
  // strip terminal CR if present
  if(in && !line.empty() && line.back() == in.widen('\r'))
    line.pop_back();
  return in;
}

// wrapper around std::getline() that handles Windows input files with extra CR
// chars at the line end
static InputFileStream& getline(InputFileStream& in, std::string& line, char delim) {
  std::getline((std::istream&)in, line, delim);
  // bad() seems to be correct here. Should not abort on EOF.
  ABORT_IF(in.bad(),
           "Error reading from file '{}'",
           in.path());
  // strip terminal CR if present
  if(in && !line.empty() && line.back() == in.widen('\r'))
    line.pop_back();
  return in;
}

class OutputFileStream {
public:
  OutputFileStream(const std::string& file) : file_(file), ofstream_(file_.getBoost()) {
    ABORT_IF(
        !marian::filesystem::exists(file_), "File '{}' does not exist", file);

    if(file_.extension() == marian::filesystem::Path(std::string(".gz")))
      ostream_.push(boost::iostreams::gzip_compressor());
    ostream_.push(ofstream_);
  }

  OutputFileStream(TemporaryFile& tempfile)
      : fds_(tempfile.getFileDescriptor(), boost::iostreams::never_close_handle) {
    lseek(tempfile.getFileDescriptor(), 0, SEEK_SET);
    ostream_.push(fds_, 1024);
  }

  OutputFileStream(std::ostream& strm) { ostream_.push(strm, 0); }

  operator std::ostream&() { return ostream_; }

  operator bool() { return (bool)ostream_; }

  bool bad() const {
    return ostream_.bad();
  }

  bool fail() const {
    return ostream_.fail();
  }

  template <typename T>
  friend OutputFileStream& operator<<(OutputFileStream& stream, const T& t) {
    stream.ostream_ << t;
    // fail() seems to be correct here. Failure to write should abort.
    ABORT_IF(stream.fail(),
             "Error writing to file '{}'",
             stream.path());
    return stream;
  }

  // handle things like std::endl which is actually a function not a value
  friend OutputFileStream& operator<<(OutputFileStream& stream, std::ostream& (*var)(std::ostream&)) {
    stream.ostream_ << var;
    // fail() seems to be correct here. Failure to write should abort.
    ABORT_IF(stream.fail(),
             "Error writing to file '{}'",
             stream.path());
    return stream;
  }

  template <typename T>
  size_t write(const T* ptr, size_t num = 1) {
    ostream_.write((char*)ptr, num * sizeof(T));
    // fail() seems to be correct here. Failure to write should abort.
    ABORT_IF(fail(),
             "Error writing to file '{}'",
             path());
    return num * sizeof(T);
  }

  std::string path() { return file_.string(); }

private:
  marian::filesystem::Path file_;
  boost::filesystem::ofstream ofstream_;
  boost::iostreams::file_descriptor_sink fds_;
  boost::iostreams::filtering_ostream ostream_;
};

}
}
