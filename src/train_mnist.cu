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

#include "marian.h"
#include "mnist.h"
#include "optimizers.h"

int main(int argc, char** argv) {
  const size_t IMAGE_SIZE = 784;
  const size_t LABEL_SIZE = 10;
  int numofdata;

  std::vector<float> trainImages = datasets::mnist::ReadImages("../examples/mnist/t10k-images-idx3-ubyte", numofdata, IMAGE_SIZE);
  std::vector<float> trainLabels = datasets::mnist::ReadLabels("../examples/mnist/t10k-labels-idx1-ubyte", numofdata, LABEL_SIZE);

  using namespace marian;
  using namespace keywords;

  ExpressionGraph g;
  
  Expr x = named(g.input(shape={whatevs, IMAGE_SIZE}), "x");
  Expr y = named(g.input(shape={whatevs, LABEL_SIZE}), "y");

  Expr w = named(g.param(shape={IMAGE_SIZE, LABEL_SIZE}), "w");
  Expr b = named(g.param(shape={1, LABEL_SIZE}), "b");

  auto scores = dot(x, w) + b;
  auto lr = softmax(scores);
  auto cost = named(-mean(sum(y * log(lr), axis=1), axis=0), "cost");
  std::cerr << "lr=" << lr.Debug() << std::endl;

  Adagrad opt;
  opt(g, 300);
  
  return 0;
}
