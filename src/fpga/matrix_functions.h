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

size_t SumSizet(
    const cl_mem &mem,
    size_t size,
    const cl_context &context,
    const cl_device_id &device);

Matrix& CopyRows(
	     const cl_context &context,
		 const cl_device_id &device,
		 Matrix& Out,
		 const Matrix& In,
		 const cl_mem &dev,
		 size_t numPairs);

Matrix& Assemble(
		const cl_context &context,
		const cl_device_id &device,
		Matrix& Out,
		 const Matrix& In,
		 const Array<size_t>& indeces);

}
}
}

