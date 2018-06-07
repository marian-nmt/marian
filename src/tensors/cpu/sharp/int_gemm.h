#pragma once

#include "tensors/tensor.h"
#include "tensors/tensor_allocator.h"
#include "tensors/tensor_operators.h"

#include <cassert>
#include <cstddef>
#include <emmintrin.h>
#include <immintrin.h>
#include <tmmintrin.h>
#include <xmmintrin.h>

namespace marian {
namespace cpu {
namespace int16 {

const int BITS = 10;

#ifdef __AVX512F__
void AVX_Quantize16(const float *input, int16_t *output, float quant_mult, std::size_t size);
void AVX_Quantize8(const float *input, int8_t *output, float quant_mult, std::size_t size);
void AVX_MatrixMult16(const __m512i * A, const __m512i * B, float * C, float unquant_mult, int num_A_rows, int num_B_rows, int width);
void AVX_MatrixMult8(const __m512i * A, const __m512i * B, float * C, float unquant_mult, int num_A_rows, int num_B_rows, int width);
#endif
void SSE_Quantize16(const float * input, __m128i * output, float quant_mult, int num_rows, int width);
void SSE_MatrixMult16(const __m128i * A, const __m128i * B, float * C, float unquant_mult, int num_A_rows, int num_B_rows, int width);

static inline void Quantize16(marian::Tensor out,
                            const marian::Tensor in,
                            float clipValue) {

    float quant_mult = pow(2.0, (float)BITS);
#ifdef __AVX512F__
    AVX_Quantize16(in->data(), out->data<int16_t>(), quant_mult, in->shape().elements());
#else
    int num_rows = in->shape().elements() / in->shape()[-1];
    int width = in->shape()[-1];
    SSE_Quantize16(in->data(), out->data<__m128i>(), quant_mult, num_rows, width);
#endif
}

static inline void Quantize8(marian::Tensor out,
                            const marian::Tensor in,
                            float clipValue) {
#ifdef __AVX512F__
    float quant_mult = 127.0 / clipValue;
    AVX_Quantize8(in->data(), out->data<int8_t>(), quant_mult, in->shape().elements());
#else
    ABORT("8-bit is currently only AVX512");
#endif
}

// This operates on floats after processing so doesn't care about int8_t vs int16_t.
static void AddBias(marian::Tensor C, const marian::Tensor Bias) {
    float* y = C->data();
    const float* x = C->data();
    const float* bias = Bias->data();

    int m = C->shape().elements() / C->shape()[-1];
    int n = C->shape()[-1];
#ifdef __AVX512F__
    int n16 = n & ~15;
#else
    int n4 = (n / 4) * 4;
#endif

    for(int j = 0; j < m; ++j) {
        int i = 0;
#ifdef __AVX512F__
        for (; i < n16; i += 16) {
            __m512 ai = _mm512_loadu_ps(x + j * n + i);
            __m512 bi = _mm512_loadu_ps(bias + i);
            __m512 yi = _mm512_add_ps(ai, bi);
            _mm512_storeu_ps(y + j * n + i, yi);
        }
#else
        for (; i < n4; i += 4) {
            __m128 ai = _mm_loadu_ps(x + j * n + i);
            __m128 bi = _mm_loadu_ps(bias + i);
            __m128 yi = _mm_add_ps(ai, bi);
            _mm_storeu_ps(y + j * n + i, yi);
        }
#endif
        for (; i < n; i++) {
            y[j * n + i] = x[j * n + i] + bias[i];
        }
    }
}

static void ProdInt16(marian::Tensor C,
                      const marian::Tensor A,
                      const marian::Tensor B,
                      float scale) {
    ABORT_IF(scale != 1, "Scale other than 1 not supported");

    // @TODO: make this a parameter
    float quant_mult = pow(2.0, (float)BITS);

    // If we quantize to n bits and then multiple the values together, the result will be quantized to n^2 bits.
    // So we must divide by 1.0/(n^2) to get back the original value.
    float unquant_mult = 1.0 / (quant_mult * quant_mult);

    float* fC = C->data();
    int num_A_rows = A->shape().elements() / A->shape()[-1];
    int num_B_rows = B->shape().elements() / B->shape()[-1];
    int width = B->shape()[-1];
#ifdef __AVX512F__
    AVX_MatrixMult16(A->data<__m512i>(), B->data<__m512i>(), fC, unquant_mult, num_A_rows, num_B_rows, width);
#else
    SSE_MatrixMult16(A->data<__m128i>(), B->data<__m128i>(), fC, unquant_mult, num_A_rows, num_B_rows, width);
#endif
}

static void ProdInt8(marian::Tensor C,
                     const marian::Tensor A,
                     const marian::Tensor B,
                     float scale,
                     float clipValue) {
#ifdef __AVX512F__
    // This would be easy...
    ABORT_IF(scale != 1, "Scale other than 1 not supported");
    float quant_mult = 127.0 / clipValue;
    float unquant_mult = 1.0 / (quant_mult * quant_mult);

    float* fC = C->data();
    int num_A_rows = A->shape().elements() / A->shape()[-1];
    int num_B_rows = B->shape().elements() / B->shape()[-1];
    int width = B->shape()[-1];
    AVX_MatrixMult8(A->data<__m512i>(), B->data<__m512i>(), fC, unquant_mult, num_A_rows, num_B_rows, width);
#else
    ABORT("8-bit is currently only AVX512");
#endif

}

}}} // namespaces
