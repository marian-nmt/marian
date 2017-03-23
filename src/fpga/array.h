#pragma once
#include <sstream>
#include <vector>
#include "types-fpga.h"

namespace amunmt {
namespace FPGA {

template<typename T>
class Array
{
public:
  Array(const cl_context &context, const cl_device_id &device, size_t size)
  :context_(context)
  ,device_(device)
  ,size_(size)
  {
    cl_int err;
    mem_ = clCreateBuffer(context_,  CL_MEM_READ_WRITE,  sizeof(T) * size, NULL, &err);
    CheckError(err);
  }

  Array(const cl_context &context, const cl_device_id &device, const std::vector<T> &vec)
  :context_(context)
  ,device_(device)
  ,size_(vec.size())
  {
    cl_int err;
    mem_ = clCreateBuffer(context_,  CL_MEM_COPY_HOST_PTR,  sizeof(T) * size_, (void*) vec.data(), &err);
    CheckError(err);
  }

  ~Array()
  {
    CheckError( clReleaseMemObject(mem_) );
  }

  size_t size() const
  { return size_; }

  cl_mem &data()
  { return mem_;  }

  const cl_mem &data() const
  { return mem_;  }

  virtual std::string Debug(bool detailed = false) const
  {
    std::stringstream strm;
    strm << size_ << " " << mem_;

    if (detailed) {
      float sum = Sum();
      strm << " sum=" << sum << std::flush;
    }

    return strm.str();

  }

protected:
  const cl_context &context_;
  const cl_device_id &device_;

  size_t size_;
  cl_mem mem_;

  float Sum() const
  {

  }

};



}
}

