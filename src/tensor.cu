// This file is part of the Marian toolkit.
// Marian is copyright (c) 2016 Marcin Junczys-Dowmunt.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <fstream>
#include "tensor.h"

namespace marian {

///////////////////////////////////////////////////////////////////
__global__
void gIncr(float *d, size_t ind, float delta) {
  d[ind] += delta;
}


__global__
void gSum(float *d, size_t size, float &total) {
  total = 0;
  for (size_t i = 0; i < size; ++i) {
    total += d[i];
  }
}

///////////////////////////////////////////////////////////////////

void Tensor::set(const std::vector<float>& data)
{
	UTIL_THROW_IF2(!pimpl_, "Tensor has not been allocated");
	pimpl_->set(data.begin(), data.end());
}

void Tensor::set(const std::vector<float>::const_iterator &begin, const std::vector<float>::const_iterator &end)
{
	UTIL_THROW_IF2(!pimpl_, "Tensor has not been allocated");
	pimpl_->set(begin, end);
}

Tensor& operator<<(Tensor& t, const std::vector<float> &vec) {
  t.set(vec);
  return t;
}

std::vector<float>& operator<<(std::vector<float> &vec, const Tensor& t) {
  t.get(vec);
  return vec;
}

void Tensor::incr(size_t ind, Tensor::value_type delta) {
  value_type *d = data();
  gIncr<<<1,1>>>(d, ind, delta);
  cudaDeviceSynchronize();
}

Tensor::value_type Tensor::sum() {
  float *d_a;
  const unsigned int bytes = sizeof(value_type);
  cudaMalloc((value_type**)&d_a, bytes);

  value_type *d = data();
  gSum<<<1,1>>>(d, size(), *d_a);
  cudaDeviceSynchronize();

  value_type h_a;
  cudaMemcpy(&h_a, d_a, bytes, cudaMemcpyDeviceToHost);
  cudaFree(d_a);

  return h_a;
}

void Tensor::sum(Tensor &out, size_t ind) {
  float *d_a = out.data() + ind;

  value_type *d = data();
  gSum<<<1,1>>>(d, size(), *d_a);
  cudaDeviceSynchronize();
}

}
