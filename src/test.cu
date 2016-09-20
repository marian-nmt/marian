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
#include <boost/timer/timer.hpp>

#include "marian.h"
#include "mnist.h"
#include "vocab.h"
#include "tensor_operators.h"

using namespace marian;
using namespace keywords;

template <class Functor>
__global__ void tgElement(Functor functor, TensorView t, int rows, int cols) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int i = bid + blockIdx.x;
    if(i < rows) {
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int j = tid + threadIdx.x;
        if(j < cols)
          t(i, j) = functor(i, j);
      }
    }
  }
}

template <class Functor>
void tElement(Functor functor, Tensor t) {

  
  int m = t.shape()[0];
  int n = t.shape()[1];
  
  int blocks  = std::min(MAX_BLOCKS, m);
  int threads = std::min(MAX_THREADS, n);
  tgElement<<<blocks, threads>>>(functor, TensorView(t), m, n);
  cudaStreamSynchronize(0);
}

int main(int argc, char** argv) {
  ExpressionGraph g;

  //Tensor a({1000, 1000}, 3);
  //Tensor b({1, 1}, 2);
  //
  //TensorView ta(a);
  //TensorView tb(b);
  //
  //boost::timer::cpu_timer timer;
  //
  //
  //auto f = _1 + _2;
  //auto pp1 = [=] __device__ (int i, int j) mutable -> float {
  //  return f(ta(i, j), tb(i, j));  
  //};
  //
  //auto pp2 = [=] __device__ (int i, int j) mutable -> float {
  //  return f(pp1(i, j), tb(i, j));  
  //};
  //
  //for(int i = 0; i < 1000; ++i)
  //  tElement(pp2, a);  

    
//  std::cerr << timer.format(5, "%ws") << std::endl;
  return 0;
}
