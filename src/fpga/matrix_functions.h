#include <vector>
#include <stddef.h>
#include "array.h"

namespace amunmt {
namespace FPGA {
namespace mblas {

class Matrix;

Matrix& CopyRows(Matrix& Out,
                 const Matrix& In,
                 const cl_mem &dev,
                 size_t numPairs);

Matrix& Assemble(Matrix& Out,
                 const Matrix& In,
                 const Array<size_t>& indeces);

}
}
}

