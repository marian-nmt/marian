#include "matrix_functions.h"
#include "matrix.h"

namespace amunmt {
namespace FPGA {
namespace mblas {

Matrix& CopyRows(Matrix& Out,
                 const Matrix& In,
                 const size_t* dev,
                 size_t numPairs)
{

}

Matrix& Assemble(Matrix& Out,
                 const Matrix& In,
                 const std::vector<size_t>& indeces)
{
  Out.Resize(indeces.size(), In.Cols());
  //CopyRows(Out, In, thrust::raw_pointer_cast(indeces.data()), indeces.size());
  return Out;
}

}
}
}

