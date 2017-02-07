#pragma once

#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <iostream>

#include "exception.h"

class InputFileStream {
  public:
    InputFileStream(const std::string& file)
     : file_(file), ifstream_(file_)
    {
      UTIL_THROW_IF2(!boost::filesystem::exists(file_),
                     "File " << file << " does not exist");

      if(file_.extension() == ".gz")
        istream_.push(boost::iostreams::gzip_decompressor());
      istream_.push(ifstream_);
    }

    InputFileStream(std::istream& strm)
    {
      istream_.push(strm, 0);
    }

    operator std::istream& () {
      return istream_;
    }

    operator bool () {
      return (bool)istream_;
    }

    template <typename T>
    friend InputFileStream& operator>>(InputFileStream& stream, T& t) {
      stream.istream_ >> t;
      return stream;
    }

    std::string path() {
      return file_.string();
    }

  private:
    boost::filesystem::path file_;
    boost::filesystem::ifstream ifstream_;
    boost::iostreams::filtering_istream istream_;
};

class OutputFileStream {
  public:
    OutputFileStream(const std::string& file)
     : file_(file), ofstream_(file_)
    {
      UTIL_THROW_IF2(!boost::filesystem::exists(file_),
                     "File " << file << " does not exist");

      if(file_.extension() == ".gz")
        ostream_.push(boost::iostreams::gzip_compressor());
      ostream_.push(ofstream_);
    }

    OutputFileStream(std::ostream& strm)
    {
      ostream_.push(strm, 0);
    }

    operator std::ostream& () {
      return ostream_;
    }

    operator bool () {
      return (bool)ostream_;
    }

    template <typename T>
    friend OutputFileStream& operator<<(OutputFileStream& stream, const T& t) {
      stream.ostream_ << t;
      return stream;
    }

    std::string path() {
      return file_.string();
    }

  private:
    boost::filesystem::path file_;
    boost::filesystem::ofstream ofstream_;
    boost::iostreams::filtering_ostream ostream_;
};

