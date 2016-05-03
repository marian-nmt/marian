#include <iostream>
#include <ctime>

#include "mad.h"

mad::Var layer(size_t max) {
    
    using namespace mad;
    
    Var x0 = 1, x1 = 2, x2 = 3;
    Var y = 0.0;
    for(int i = 0; i < max; i++) {
        Var xi = i;
        y = y + x0 + log(x2) + x1;
        for(int j = 0; j < i; ++j) {
            y = y + xi;
        }
    }
    
    return y;
}

int main(int argc, char** argv) {
    srand(time(NULL));
    
    using namespace mad;
    
    Var y1 = layer(10);
    Var y2 = layer(5);
    
    Var y = y1 + log(y2);
    
    set_zero_all_adjoints();
    y.calc_gradients();
    
    std::cerr << "y1 = " << y1.val() << std::endl;
    std::cerr << "y2 = " << y2.val() << std::endl;
    std::cerr << "y = " << y.val() << std::endl;
    
    std::cerr << "dy/dy1 = " << y1.grad() << std::endl;
    std::cerr << "dy/dy2 = " << y2.grad() << std::endl;
    
}