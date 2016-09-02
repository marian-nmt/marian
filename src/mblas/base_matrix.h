#pragma once
#include <vector>

namespace mblas {

class BaseMatrix;
typedef std::vector<BaseMatrix*> BaseMatrices;

class BaseMatrix {
  public:
    //virtual ~BaseMatrix() {}
    
    virtual size_t Rows() const = 0;
    virtual size_t Cols() const = 0;
    virtual void Resize(size_t rows, size_t cols) = 0;

};

}

