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

  virtual std::string Debug(bool detailed = false) const
  {
    std::stringstream strm;
    strm << size_ << " " << mem_;

    if (detailed) {
      float sum = mblas::SumSizet(mem_, size_, openCLInfo_);
      strm << " sum=" << sum << std::flush;
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

