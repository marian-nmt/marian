#pragma once

#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/iostreams/device/file_descriptor.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <iostream>

#include <sys/stat.h>

#include "exception.h"

namespace io = boost::iostreams;

class TemporaryFile {
private:
  int fd_;

  int mkstemp_and_unlink(char* tmpl) {
    int ret = mkstemp(tmpl);
    if(ret != -1) {
      UTIL_THROW_IF2(unlink(tmpl), "while deleting " << tmpl);
    }
    return ret;
  }

  int MakeTemp(const std::string& base) {
    std::string name(base);
    name += "XXXXXX";
    name.push_back(0);
    int ret;
    UTIL_THROW_IF2(-1 == (ret = mkstemp_and_unlink(&name[0])),
                   "while making a temporary based on " << base);
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
    if(S_ISDIR(sb.st_mode))
      base += '/';
  }

public:
  TemporaryFile(const std::string base = "/tmp/") {
    std::string baseTemp(base);
    NormalizeTempPrefix(baseTemp);
    fd_ = MakeTemp(baseTemp);
  }

  ~TemporaryFile() {
    if(fd_ != -1 && close(fd_)) {
      std::cerr << "Could not close file " << fd_ << std::endl;
      std::abort();
    }
  }

  int getFileDescriptor() { return fd_; }
};

class InputFileStream {
public:
  InputFileStream(const std::string& file) : file_(file), ifstream_(file_) {
    UTIL_THROW_IF2(!boost::filesystem::exists(file_),
                   "File " << file << " does not exist");

    if(file_.extension() == ".gz")
      istream_.push(io::gzip_decompressor());
    istream_.push(ifstream_);
  }

  InputFileStream(TemporaryFile& tempfile)
      : fds_(tempfile.getFileDescriptor(), io::never_close_handle) {
    lseek(tempfile.getFileDescriptor(), 0, SEEK_SET);
    istream_.push(fds_, 1024);
  }

  InputFileStream(std::istream& strm) { istream_.push(strm, 0); }

  operator std::istream&() { return istream_; }

  operator bool() { return (bool)istream_; }

  template <typename T>
  friend InputFileStream& operator>>(InputFileStream& stream, T& t) {
    stream.istream_ >> t;
    return stream;
  }

  std::string path() { return file_.string(); }

  bool empty() {
    return ifstream_.peek() == std::ifstream::traits_type::eof();
  }

private:
  boost::filesystem::path file_;
  boost::filesystem::ifstream ifstream_;
  io::file_descriptor_source fds_;
  io::filtering_istream istream_;
};

class OutputFileStream {
public:
  OutputFileStream(const std::string& file) : file_(file), ofstream_(file_) {
    UTIL_THROW_IF2(!boost::filesystem::exists(file_),
                   "File " << file << " does not exist");

    if(file_.extension() == ".gz")
      ostream_.push(io::gzip_compressor());
    ostream_.push(ofstream_);
  }

  OutputFileStream(TemporaryFile& tempfile)
      : fds_(tempfile.getFileDescriptor(), io::never_close_handle) {
    lseek(tempfile.getFileDescriptor(), 0, SEEK_SET);
    ostream_.push(fds_, 1024);
  }

  OutputFileStream(std::ostream& strm) { ostream_.push(strm, 0); }

  operator std::ostream&() { return ostream_; }

  operator bool() { return (bool)ostream_; }

  template <typename T>
  friend OutputFileStream& operator<<(OutputFileStream& stream, const T& t) {
    stream.ostream_ << t;
    return stream;
  }

  std::string path() { return file_.string(); }

private:
  boost::filesystem::path file_;
  boost::filesystem::ofstream ofstream_;
  io::file_descriptor_sink fds_;
  io::filtering_ostream ostream_;
};
