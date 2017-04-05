#pragma once

#include <vector>
#include <stddef.h>
#include "types-fpga.h"

namespace amunmt {
namespace FPGA {

struct Weights;

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
    const OpenCLInfo &openCLInfo);

Matrix& CopyRows(
		 Matrix& Out,
		 const Matrix& In,
		 const Array<uint>& indices);

Matrix& Assemble(
		Matrix& Out,
		 const Matrix& In,
		 const Array<uint>& indices);

void Fill(
    Matrix& In,
    float value=0.0f);

Matrix& Transpose(Matrix& Out, const Matrix& In);

Matrix& Transpose(Matrix& Out);

Matrix& Concat(Matrix& Out, const Matrix& In);

Matrix& Prod(Matrix& C, const Matrix& A, const Matrix& B,
             bool transA = false, bool transB = false);

inline void Normalization(Matrix& out, const Matrix& in, const Matrix& alpha, float eps)
{}

void ElementwiseOps(mblas::Matrix& NextState,
                    const mblas::Matrix& State,
                    const mblas::Matrix& RUH,
                    const mblas::Matrix& Temp,
                    const mblas::Matrix& B,
                    const mblas::Matrix& Bx1,
                    const mblas::Matrix& Bx2,
                    const uint &rows,
                    const uint &cols);

Matrix& BroadcastVecAdd(Matrix& Out, const Matrix& In);

Matrix& ElementLogit(Matrix& Out, const Matrix& In);

Matrix& ElementTanh(Matrix& Out, const Matrix& In1, const Matrix& In2);

Matrix& Slice(Matrix& Out,
              const Matrix& In,
              uint n, uint dim);

} // namespace mblas {
} // namespace FPGA {
} // namespace amunmt {

