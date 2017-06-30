#include "common/exception.h"
#include "npz_converter.h"
#include "matrix_functions.h"

using namespace std;

namespace amunmt {
namespace FPGA {

NpzConverter::NpzConverter(const std::string& file)
  : model_(cnpy::npz_load(file))
{
  cerr << "file=" << file << endl;
}

mblas::Matrix NpzConverter::GetMatrix(
    const OpenCLInfo &openCLInfo,
    const std::string& key,
    bool transpose
    ) const
{
  mblas::Matrix matrix(openCLInfo);
  //cerr << "key1=" << key << " " << matrix.Debug(1) << endl;

  cnpy::npz_t::const_iterator it = model_.find(key);
  if(it != model_.end()) {
    //cerr << key << " found" << endl;
    const cnpy::NpyArray &array = it->second;
    NpyMatrixWrapper np(array);

    matrix.Resize(np.size1(), np.size2());
    matrix.Set(np.data());

    if (transpose) {
      mblas::Transpose(matrix);
    }
  }
  else {
    //cerr << key << " NOT found" << endl;
  }
  //cerr << "key2=" << key << " " << matrix.Debug(1) << endl;

  return std::move(matrix);
}


}
}

