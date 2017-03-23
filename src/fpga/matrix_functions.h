#include <vector>
#include <stddef.h>
#include "array.h"

namespace amunmt {
namespace FPGA {
namespace mblas {

class Matrix;

float Sum(
    const cl_mem &mem,
    size_t size,
    const cl_context &context,
    const cl_device_id &device);

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

