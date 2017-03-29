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
    uint size,
    const OpenCLInfo &openCLInfo);

unsigned int SumSizet(
    const cl_mem &mem,
    uint size,
    const cl_context &context,
    const cl_device_id &device);

Matrix& CopyRows(
    const OpenCLInfo &openCLInfo,
		 Matrix& Out,
		 const Matrix& In,
		 const Array<uint>& indices);

Matrix& Assemble(
    const OpenCLInfo &openCLInfo,
		Matrix& Out,
		 const Matrix& In,
		 const Array<uint>& indices);

void Fill(
    const OpenCLInfo &openCLInfo,
    Matrix& In,
    float value=0.0f);

Matrix& Prod(Matrix& C, const Matrix& A, const Matrix& B,
             bool transA = false, bool transB = false);



} // namespace mblas {
} // namespace FPGA {
} // namespace amunmt {

