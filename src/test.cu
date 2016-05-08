#include <iostream>
#include <ctime>
#include <vector>
#include <algorithm>
#include <random>
#include <boost/timer/timer.hpp>
#include <typeinfo>
#include <typeindex>
#include <unordered_map>

#include <boost/any.hpp>

#include "marian.h"
#include "operators.h"
#include "keywords.h"

using namespace marian;


int main(int argc, char** argv) {

    using namespace keywords;
    
    auto layer = demo(300, prefix="test_");
    
    //auto x = input("X", shape={1, 768});
    //auto y = input("Y", shape={1, 10});
    //
    //auto l = x;
    //for(auto n : { 300, 200, 100, 50, 20 })
    //    l = dense(n, l, activation=tanh);
    //    
    //auto w = param("W", init=orthogonal, shape={20, 10});
    //auto b = param("b", init=orthogonal, shape={1, 10});
    //l = sigmoid(dot(w, l) + b);
    //
    //auto lp = dense(10, l, activation=softmax(axis=1));
    //auto cost = -mean(sum(y * log(lp), axis=1));

    
    //auto x1 = input(k::name="x0", k::shape={1,100});
    //auto x2 = input(k::name="x1", k::shape={1,100});
    //auto y = output(k::name="y", k::shape={1,10});
    //
    //auto l1 = dense(100,
    //                k::name="layer1",
    //                k::input={x1, x2},
    //                k::activation=sigmoid,
    //                k::init_w=orthogonal,
    //                k::init_b=uniform(-0.1,0.1)
    //                k::merge=concat);
    //auto l2 = dense(100, k::input=l1, k::name="charlie"
    //                k::activation=tanh);
    //auto lout = dense(10, k::input=l2,
    //                k::activation=softmax);
    //
    //auto cost = -mean(sum(y * log(lout), k::axis=1));
    //
    //auto w = cost["charlie_w"];
    //auto b = cost["layer1_b"];
    //
    //auto opt = optimizer(cost,
    //                     k::method=adadelta);
    //
    //Tensor X(k::shape={60, 768}, k::init=mnist(""));
    //Tensor Y(k::shape={60, 10}, k::init=mnist(""));
    //
    //float c = opt.fit_batch({X1, X2}, Y, k::logger=logger);
    //
    //Tensor xTrain
    //    (shape, {60000, 784})
    //    (init, mnist("train.ubyte"));
    //
    //Tensor yTrain
    //    (shape, {60000, 10})
    //    (init, mnist("train.ubyte", true));
    //
    //Tensor xBatch = slice(xTrain, {0, 50, 5});
    //
    //Var x = input("X");
    //Var y = input("Y");
    //
    //ry = dense(input=x, size=200, activation=tanh,
    //           init_w=orthogonal, init_b=uniform(-0.1. 0.1));
    //
    //ry = dense(ry)(size, 100)(activation, tanh);
    //ry = dense(ry)(size, 10)(activation, softmax);
    //
    //Var cost = -mean(y * log(ry) + (1 - y) * log(1 - ry)); 
    //
    //boost::timer::auto_cpu_timer t;   
    //float eta = 0.01;
    //for(size_t i = 0; i < 2000; ++i) {
    //  cost.forward();
    //  
    //  if(i % 200 == 0) {
    //    for(size_t j = 0; j < 4; ++j) {
    //      std::cerr << ry.val()[j] << std::endl;
    //    }
    //    std::cerr << i << " ct: " << cost.val()[0] << std::endl;
    //  }
    //  
    //  cost.backward();
    //  for(auto p : params) {
    //    auto update =
    //        _1 -= eta * _2;
    //    Element(update, p.val(), p.grad());
    //  }
    //}
    
    return 0;
}