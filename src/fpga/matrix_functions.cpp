#include "matrix_functions.h"
#include "matrix.h"

namespace amunmt {
namespace FPGA {
namespace mblas {

Matrix& CopyRows(Matrix& Out,
                 const Matrix& In,
                 const cl_mem &dev,
                 size_t numPairs)
{

}

Matrix& Assemble(Matrix& Out,
                 const Matrix& In,
                 const Array<size_t>& indeces)
{
  Out.Resize(indeces.size(), In.dim(1));
  CopyRows(Out, In, indeces.data(), indeces.size());
  return Out;
}

}
}
}

