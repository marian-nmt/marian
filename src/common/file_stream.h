#pragma once

// @TODO: this file still contains lots of stuff from boost::filesystem and boost::iostreams,
// this has to be figured out.

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
  }

  ~TemporaryFile() {
    if(fd_ != -1 && !unlink_) {
      ABORT_IF(unlink(name_.c_str()), "Error while deleting '{}'", name_);
    }
    if(fd_ != -1 && close(fd_)) {
      std::cerr << "Could not close file " << fd_ << std::endl;
      std::abort();
    }
  }

  int getFileDescriptor() { return fd_; }

  std::string getFileName() { return name_; }
};

class InputFileStream {
private:
  std::unique_ptr<std::istream> istream_;
  marian::filesystem::Path file_;

#ifndef _WIN32
    std::unique_ptr<__gnu_cxx::stdio_filebuf<char>> filebuf_;
#endif

public:
  InputFileStream(const std::string& file)
  : file_(file) {
    ABORT_IF(!marian::filesystem::exists(file_),"File '{}' does not exist", file);
    istream_.reset(new zstr::ifstream(file_));
  }

  InputFileStream(TemporaryFile& tempfile) {
    lseek(tempfile.getFileDescriptor(), 0, SEEK_SET);

  // @TODO: this is non-standard, add more alternatives
  // this SO answer describes a number of alternatives for different compilers, checking g++ for now.
  // https://stackoverflow.com/questions/2746168/how-to-construct-a-c-fstream-from-a-posix-file-descriptor
  #ifndef _WIN32
    filebuf_.reset(new __gnu_cxx::stdio_filebuf<char>(tempfile.getFileDescriptor(), std::ios::in));
    istream_.reset(new zstr::istream(filebuf_.get()));
  #endif
  }

  InputFileStream(std::istream& strm) {
    istream_.reset(new zstr::istream(strm));
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

#ifndef _WIN32
    std::unique_ptr<__gnu_cxx::stdio_filebuf<char>> filebuf_;
#endif

public:
  OutputFileStream(const std::string& file) : file_(file) {
    ABORT_IF(!marian::filesystem::exists(file_), "File '{}' does not exist", file);
    ostream_.reset(new zstr::ofstream(file_));
  }

  OutputFileStream(TemporaryFile& tempfile) {
    lseek(tempfile.getFileDescriptor(), 0, SEEK_SET);

  // @TODO: this is non-standard, add more alternatives
  // this SO answer describes a number of alternatives for different compilers, checking g++ for now.
  // https://stackoverflow.com/questions/2746168/how-to-construct-a-c-fstream-from-a-posix-file-descriptor
  #ifndef _WIN32
    filebuf_.reset(new __gnu_cxx::stdio_filebuf<char>(tempfile.getFileDescriptor(), std::ios::out));
    ostream_.reset(new std::ostream(filebuf_.get()));
  #endif
  }

  OutputFileStream(std::ostream& strm) {
    ostream_.reset(new zstr::ostream(strm));
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
