#pragma once

#include "common/filesystem.h"
#include "common/logging.h"

#pragma warning(push)
#pragma warning(disable: 4458) // declaration of 'traits_type' hides class member
#pragma warning(disable: 4456) // declaration of 'c' hides previous local declaration
#pragma warning(disable: 4244) // conversion from 'int' to 'char', possible loss of data
#pragma warning(disable: 4706) // assignment within conditional expression
#include "3rd_party/zstr/zstr.hpp"
#pragma warning(pop)

#include <iostream>

#ifdef _WIN32
#include <io.h>
#include <sys/stat.h>
#include <sys/types.h>
#else
#include <ext/stdio_filebuf.h>
#endif

class TemporaryFile {
private:
  int fd_;
  bool unlink_;
  std::string name_;

#ifndef _WIN32
  std::unique_ptr<__gnu_cxx::stdio_filebuf<char>> buf_;
#endif


  int mkstemp_and_unlink(char* tmpl) {
#ifdef _WIN32
    ABORT_IF(true, "mkstemp not available in Windows");
    int ret = -1;
#else
    int ret = mkstemp(tmpl);
#endif
    if(unlink_ && ret != -1) {
      ABORT_IF(unlink(tmpl), "Error while deleting '{}'", tmpl);
    }
    return ret;
  }

  int MakeTemp(const std::string& base) {
    std::string name(base);
    name += "marian.XXXXXX";
    name.push_back(0);
    int ret;
    ABORT_IF(-1 == (ret = mkstemp_and_unlink(&name[0])),
             "Error while making a temporary based on '{}'",
             base);
    name_ = name;
    return ret;
  }

  void NormalizeTempPrefix(std::string& base) {
    if(base.empty())
      return;
    if(base[base.size() - 1] == '/')
      return;
    struct stat sb;
    // It's fine for it to not exist.
    if(-1 == stat(base.c_str(), &sb))
      return;
#ifdef _WIN32
#define S_ISDIR(m) (((m)&S_IFMT) == S_IFDIR)  // TODO: unify this
#endif
    if(S_ISDIR(sb.st_mode))
      base += '/';
  }

public:
  TemporaryFile(const std::string base = "/tmp/", bool earlyUnlink = true)
      : unlink_(earlyUnlink) {
    std::string baseTemp(base);
    NormalizeTempPrefix(baseTemp);
    fd_ = MakeTemp(baseTemp);

#ifndef _WIN32
    buf_.reset(new __gnu_cxx::stdio_filebuf<char>(fd_, std::ios::in|std::ios::out));
#endif
  }

  ~TemporaryFile() {
    if(fd_ != -1 && !unlink_) {
      ABORT_IF(unlink(name_.c_str()), "Error while deleting '{}'", name_);
    }
    if(fd_ != -1 && close(fd_)) {
      ABORT("Could not close file {}", fd_ );
    }
  }

  void seek(size_t pos) {
    lseek(fd_, pos, SEEK_SET);
  }

  int getFileDescriptor() { return fd_; }

  std::streambuf* rdbuf() { buf_.get(); }

  std::string getFileName() { return name_; }
};

class InputFileStream {
private:
  std::unique_ptr<std::istream> istream_;
  marian::filesystem::Path file_;

public:
  InputFileStream(const std::string& file)
  : file_(file) {
    ABORT_IF(!marian::filesystem::exists(file_),"File '{}' does not exist", file);

    if(file_.extension() == marian::filesystem::Path(std::string(".gz")))
      istream_.reset(new zstr::ifstream(file_));
    else
      istream_.reset(new std::ifstream(file_));
  }

  InputFileStream(TemporaryFile& tempfile) {
    tempfile.seek(0);
    istream_.reset(new std::istream(tempfile.rdbuf()));
  }

  InputFileStream(std::istream& strm) {
    istream_.reset(new std::istream(strm.rdbuf()));
  }


  operator std::istream&() { return *istream_; }

  operator bool() { return (bool)*istream_; }

  template <typename T>
  friend InputFileStream& operator>>(InputFileStream& stream, T& t) {
    *(stream.istream_) >> t;
    return stream;
  }

  template <typename T>
  size_t read(T* ptr, size_t num = 1) {
    istream_->read((char*)ptr, num * sizeof(T));
    return num * sizeof(T);
  }

  std::string path() { return file_.string(); }

  bool empty() { return istream_->peek() == std::ifstream::traits_type::eof(); }

};

class OutputFileStream {
private:
  std::unique_ptr<std::ostream> ostream_;
  marian::filesystem::Path file_;

public:
  OutputFileStream(const std::string& file) : file_(file) {
    if(file_.extension() == marian::filesystem::Path(std::string(".gz")))
      ostream_.reset(new zstr::ofstream(file_));
    else
      ostream_.reset(new std::ofstream(file_));
  }

  OutputFileStream(TemporaryFile& tempfile) {
    tempfile.seek(0);
    ostream_.reset(new std::ostream(tempfile.rdbuf()));
  }

  OutputFileStream(std::ostream& strm) {
    ostream_.reset(new std::ostream(strm.rdbuf()));
  }

  operator std::ostream&() { return *ostream_; }

  operator bool() { return (bool)*ostream_; }

  template <typename T>
  friend OutputFileStream& operator<<(OutputFileStream& stream, const T& t) {
    *(stream.ostream_) << t;
    return stream;
  }

  template <typename T>
  size_t write(const T* ptr, size_t num = 1) {
    ostream_->write((char*)ptr, num * sizeof(T));
    return num * sizeof(T);
  }

  std::string path() { return file_.string(); }
};
