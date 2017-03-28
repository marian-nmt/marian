#pragma once

#include <vector>
#include <stddef.h>
#include "types-fpga.h"

namespace amunmt {
namespace FPGA {

template<typename T>
class Array;

namespace mblas {

class Matrix;

template<typename T>
T Sum(const std::vector<T> &vec)
{
  T ret = T();
  for (size_t i = 0; i < vec.size(); ++i) {
    ret += vec[i];
  }
  return ret;
}

float Sum(
    const cl_mem &mem,
    size_t size,
    const cl_context &context,
    const cl_device_id &device);

unsigned int SumSizet(
    const cl_mem &mem,
    size_t size,
    const cl_context &context,
    const cl_device_id &device);

Matrix& CopyRows(
	     const cl_context &context,
		 const cl_device_id &device,
		 Matrix& Out,
		 const Matrix& In,
		 const Array<unsigned int>& indices);

Matrix& Assemble(
		const cl_context &context,
		const cl_device_id &device,
		Matrix& Out,
		 const Matrix& In,
		 const Array<unsigned int>& indices);

}
}
}

