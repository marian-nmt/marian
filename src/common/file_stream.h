#pragma once

#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <iostream>

#include "exception.h"

namespace amunmt {

class InputFileStream {
  public:
    InputFileStream(const std::string& file)
     : file_(file), ifstream_(file_)
    {
      amunmt_UTIL_THROW_IF2(!boost::filesystem::exists(file_),
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

  private:
    boost::filesystem::path file_;
    boost::filesystem::ifstream ifstream_;
    boost::iostreams::filtering_istream istream_;
};

}

