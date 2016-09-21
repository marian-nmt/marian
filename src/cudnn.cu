#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cudnn.h>

#include <boost/timer/timer.hpp>

#include "tensor.h"
#include "tensor_operators.h"
#include "param_initializers.h"

using namespace marian;

int main() {
    int d = 4;
    
    Tensor in({d, d});
    Tensor out({d, d});
    Tensor adj({d, d}, 1);
    
    auto f = uniform(-5, 5);
    f(in);
    
    std::cerr << in.Debug() << std::endl;
    
    {
        boost::timer::cpu_timer timer;
        for(int i = 0; i < 1; ++i) {
          Tensor grad({d, d});
          CudnnLogSoftmax(out, in);
          CudnnLogSoftmaxGrad(grad, adj, in);
          std::cerr << in.Debug() << std::endl;
          std::cerr << adj.Debug() << std::endl;
          std::cerr << grad.Debug() << std::endl;
        }
        std::cerr << timer.format(5, "%ws") << std::endl;
    }
    
    {
        boost::timer::cpu_timer timer;
        for(int i = 0; i < 1; ++i) {
          Tensor grad({d, d});
          CudnnLogSoftmax(out, in);
          LogSoftmaxGrad(grad, adj, in);
          std::cerr << in.Debug() << std::endl;
          std::cerr << adj.Debug() << std::endl;
          std::cerr << grad.Debug() << std::endl;
        }
        std::cerr << timer.format(5, "%ws") << std::endl;
    }

    return 0;
}