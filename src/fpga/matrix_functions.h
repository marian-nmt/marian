#include <vector>
#include <stddef.h>

namespace amunmt {
namespace FPGA {
namespace mblas {

class Matrix;

Matrix& CopyRows(Matrix& Out,
                 const Matrix& In,
                 const size_t* dev,
                 size_t numPairs);

Matrix& Assemble(Matrix& Out,
                 const Matrix& In,
                 const std::vector<size_t>& indeces);

}
}
}

