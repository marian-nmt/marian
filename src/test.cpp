#include <iostream>
#include <ctime>

#include "mad.h"

int main(int argc, char** argv) {
    
    using namespace mad;
    {
        srand(time(NULL));
        size_t max = rand() % 20 + 1;
        
        Var x0 = 1, x1 = 2, x2 = 3;
        
        Var y = 0.0;
        for(int i = 0; i < max; i++)
            y = y + x0 + log(x2) + x1;
        
        std::vector<Var> x = { x0, x1, x2 };
           
        set_zero_all_adjoints();
        y.grad();
        
        std::cerr << "y = " << y.val() << std::endl;
        for(int i = 0; i < x.size(); ++i)
            std::cerr << "dy/dx_" << i << " = " << x[i].adj() << std::endl;
    }
}