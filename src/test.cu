#include <iostream>
#include <ctime>
#include <vector>
#include <algorithm>
#include <random>
#include <boost/timer/timer.hpp>

#include "marian.h"
#include "operators.h"

using namespace marian;

int main(int argc, char** argv) {
    boost::timer::auto_cpu_timer t;   
    
    Var x = input("X", Tensor({4, 2}));
    Var y = input("Y", Tensor({4, 2}));
    
    std::vector<float> vx = {
        0, 0,
        0, 1,
        1, 0,
        1, 1
    };
    
    std::vector<float> vy = {
        1, 0,
        1, 0,
        0, 1,
        1, 0
    };
    
    thrust::copy(vx.begin(), vx.end(), x.val().begin());
    thrust::copy(vy.begin(), vy.end(), y.val().begin());
    
    Var w0 = forsave("W0", uniform(Tensor({2, 2})));
    Var b0 = forsave("b0", uniform(Tensor({1, 2})));
    
    Var w1 = forsave("W1", uniform(Tensor({2, 2})));
    Var b1 = forsave("b1", uniform(Tensor({1, 2})));
    
    std::vector<Var> params = { w0, w1, b0, b1 };
    
    Var ry = sigma(dot(x, w0) + b0);
    ry = softmax(dot(ry, w1) + b1, Axis::axis1);
    Var cost = -mean(sum(y * log(ry), Axis::axis1), Axis::axis0); 
    
    float alpha = 0.1;
    for(size_t i = 0; i < 30000; ++i) {
      cost.forward();
      
      if(i % 100 == 0) {
        for(size_t j = 0; j < 4; ++j) {
          std::cerr << ry.val()[j*2] << std::endl;
        }
        std::cerr << i << " ct: " << cost.val()[0] << std::endl;
        //  alpha = alpha * 0.9;
      }
      
      cost.backward();
      for(auto p : params) {
        //std::cerr << p.grad()[0] << std::endl;
        auto update =
            _1 -= alpha * _2;
            
        Element(update, p.val(), p.grad());
      }
    }
    
    return 0;
}