#include "npz_converter.h"
#include "common/exception.h"

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
  cnpy::npz_t::const_iterator it = model_.find(key);
  if(it != model_.end()) {
    const cnpy::NpyArray &array = it->second;
    NpyMatrixWrapper np(array);

    mblas::Matrix matrix(openCLInfo, np.size1(), np.size2(), np.data());

    if (transpose) {
      // TODO
    }

    //cerr << "key=" << key << " " << matrix.Debug(1) << endl;

    return std::move(matrix);
  }
  else {
    amunmt_UTIL_THROW2("Missing " << key);
  }
}


}
}

