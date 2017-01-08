#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cudnn.h>
#include <unistd.h>

#include <boost/timer/timer.hpp>

#include "tensors/tensor_allocator.h"
#include "tensors/tensor_gpu.h"

#include "tensor_operators.h"
#include "param_initializers.h"

using namespace marian;

template <class F>
void testForward(F f, size_t l,
                 const Shape& shape,
                 const std::string& desc) {

    auto ta = newTensorAllocator<DeviceGPU>();

    Tensor in, out;
    ta->allocate(in, shape);
    ta->allocate(out, shape);

    uniform(-5, 5)(in);

    std::cout << desc << ": " << std::flush;
    boost::timer::cpu_timer timer;
    for(int i = 0; i < l; ++i) {
      f(out, in);
      if(i % 100 == 0)
        std::cout << "." << std::flush;
    }
    std::cout << timer.format(5, "%ws") << std::endl;
}

template <class F>
void testBackward(F f, size_t l,
                  const Shape& shape,
                  const std::string& desc) {

    auto ta = newTensorAllocator<DeviceGPU>();

    Tensor in, adj, grad;
    ta->allocate(in, shape);
    ta->allocate(adj, shape);
    adj->set(1);
    ta->allocate(grad, shape);

    uniform(-5, 5)(in);

    std::cout << desc << ": " << std::flush;
    boost::timer::cpu_timer timer;
    for(int i = 0; i < l; ++i) {
      f(grad, adj, in);
      if(i % 100 == 0)
        std::cout << "." << std::flush;
    }
    std::cout << timer.format(5, "%ws") << std::endl;
}

int main() {
    int l = 1000;

    std::vector<Shape> shapes = {
        {1000, 1000},
        {80 * 25, 85000},
        {85000, 80},
    };

    for(auto& shape : shapes) {
        std::cout << "Testing shape: " << shape[0] << "x" << shape[1] << std::endl << std::endl;

        std::cout << "Softmax forward" << std::endl;
        testForward(CudnnSoftmax, l, shape, "CuDNN ");
        //testForward(Softmax, l, shape, "Marian");
        std::cout << std::endl;

        std::cout << "LogSoftmax forward" << std::endl;
        testForward(CudnnLogSoftmax, l, shape, "CuDNN ");
        testForward(LogSoftmax, l, shape, "Marian");
        std::cout << std::endl;

        std::cout << "Softmax backward" << std::endl;
        testBackward(CudnnSoftmaxGrad, l, shape, "CuDNN ");
        testBackward(SoftmaxGrad, l, shape, "Marian");
        std::cout << std::endl;

        std::cout << "Log-softmax backward" << std::endl;
        testBackward(CudnnLogSoftmaxGrad, l, shape, "CuDNN ");
        testBackward(LogSoftmaxGrad, l, shape, "Marian");
        std::cout << std::endl;
    }
    return 0;
}
