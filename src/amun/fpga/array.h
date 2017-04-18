#pragma once
#include <sstream>
#include <vector>
#include <cassert>
#include "types-fpga.h"
#include "matrix_functions.h"

namespace amunmt {
namespace FPGA {

template<typename T>
class Array
{
public:
  Array(const OpenCLInfo &openCLInfo)
  :openCLInfo_(openCLInfo)
  ,size_(0)
  {
  }

  Array(const OpenCLInfo &openCLInfo, size_t size)
  :openCLInfo_(openCLInfo)
  ,size_(size)
  {
    cl_int err;
    mem_ = clCreateBuffer(openCLInfo.context,  CL_MEM_READ_WRITE,  sizeof(T) * size, NULL, &err);
    CheckError(err);
  }

  Array(const OpenCLInfo &openCLInfo, size_t size, const T &value)
  :Array(openCLInfo, size)
  {
    CheckError( clEnqueueFillBuffer(openCLInfo.commands, mem_, &value, sizeof(T), 0, size_ * sizeof(T), 0, NULL, NULL) );
    CheckError( clFinish(openCLInfo.commands) );
  }

  Array(const OpenCLInfo &openCLInfo, const std::vector<T> &vec)
  :openCLInfo_(openCLInfo)
  ,size_(vec.size())
  {
    cl_int err;
    mem_ = clCreateBuffer(openCLInfo.context,  CL_MEM_COPY_HOST_PTR,  sizeof(T) * size_, (void*) vec.data(), &err);
    CheckError(err);

  }

  ~Array()
  {
    //CheckError( clReleaseMemObject(mem_) );
  }

  size_t size() const
  { return size_; }

  cl_mem &data()
  { return mem_;  }

  const cl_mem &data() const
  { return mem_;  }

  const OpenCLInfo &GetOpenCLInfo() const
  { return openCLInfo_; }

  void Swap(Array &other)
  {
    assert(&openCLInfo_ == &other.openCLInfo_);
    std::swap(size_, other.size_);
    std::swap(mem_, other.mem_);
  }

  void Fill(const T &val)
  {
    CheckError( clEnqueueFillBuffer(openCLInfo_.commands, mem_, &val, sizeof(T), 0, size() * sizeof(T), 0, NULL, NULL) );
    CheckError( clFinish(openCLInfo_.commands) );
  }

  void Fill(const std::vector<T> &vec)
  {
    CheckError( clEnqueueFillBuffer(openCLInfo_.commands, mem_, vec.data(), sizeof(T), 0, vec.size() * sizeof(T), 0, NULL, NULL) );
    CheckError( clFinish(openCLInfo_.commands) );
  }

  virtual std::string Debug(size_t verbosity = 1) const
  {
    std::stringstream strm;
    strm << mem_ << " size=" << size_;

    if (verbosity) {
      float sum = mblas::SumSizet(openCLInfo_, mem_, size_);
      strm << " sum=" << sum << std::flush;
    }

    if (verbosity == 2) {
      T results[size_];
      CheckError( clEnqueueReadBuffer( openCLInfo_.commands, mem_, CL_TRUE, 0, sizeof(T) * size_, &results, 0, NULL, NULL ) );

      for (size_t i = 0; i < size_; ++i) {
        strm << " " << results[i];
      }
    }

    return strm.str();
  }

protected:
  const OpenCLInfo &openCLInfo_;

  size_t size_;
  cl_mem mem_;

};

}
}

