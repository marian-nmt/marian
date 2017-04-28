#pragma once
#include <sstream>
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

float SumFloat(
    const OpenCLInfo &openCLInfo,
    const cl_mem &mem,
    uint size
    );

uint SumUInt(
    const OpenCLInfo &openCLInfo,
    const cl_mem &mem,
    uint size
    );

Matrix& Copy(Matrix& Out, const Matrix& In);

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
    float value);

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

Matrix& ElementLogit(Matrix& Out, const Matrix& In);

Matrix& ElementTanh(Matrix& Out, const Matrix& In1, const Matrix& In2);

Matrix& ElementTanh2(Matrix& Out, const Matrix& In1, const Matrix& In2);

Matrix& ElementWhatever(Matrix& Out, const Matrix& In1, const Matrix& In2);

Matrix& ElementAddWeighted(Matrix& Out, float weight, const Matrix& In);

Matrix& BroadcastVecAdd(Matrix& Out, const Matrix& In);

Matrix& BroadcastVecTanh(Matrix& Out, const Matrix& In);

Matrix& BroadcastTanh(Matrix& Out, const Matrix& In, const Array<int>& batchMapping, size_t srcSize);

Matrix& BroadcastVecColumnAddWeighted(Matrix& Out, float weight, const Array<float>& In);


Matrix& Slice(Matrix& Out,
              const Matrix& In,
              uint n, uint dim);

void PasteRows(Matrix& Out, const Matrix& In, const size_t rowNo, size_t colNo=0, size_t sparse=1);

void MapMatrix(Matrix& state, const Array<int>& mapping, size_t i);

void Mean(Matrix& Out, const Matrix& In, const Array<int>& mapping);

Matrix& Softmax(Matrix& Out, const Array<int>& batchIds, const Array<int>& srcMapping,size_t srcSize);

Matrix& LogSoftmax(Matrix& Out);

void WeightedMean(Matrix& Out,const Matrix& Weights, const Matrix& In, const Array<int>& mapping);

void SetColumn(Matrix& In, int noColumn, float value);

void MaxElement(
    Array<float> &d_out,
    const Array<int> &d_ind,
    mblas::Matrix &d_in,
    int numBatches,
    const Array<int> &batchFirstElementIdxs);
//float* d_out, int* d_ind, float* d_in, int numBatches, int* batchFirstElementIdxs

void NthElement(
    Array<float>& d_out,
    Array<unsigned> &d_ind,
    const mblas::Matrix &Probs,
    const Array<uint> &beamSizes,
    size_t maxBatchSize,
    const Array<uint> &d_cummulatedBeamSizes,
    const Array<uint> &d_batchFirstElementIdxs);


} // namespace mblas {
} // namespace FPGA {
} // namespace amunmt {

