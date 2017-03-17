#include "npz_converter.h"
#include "common/exception.h"

using namespace std;

namespace amunmt {
namespace FPGA {

mblas::Matrix NpzConverter::GetMatrix(
    const cl_context &context,
    const std::string& key,
    bool transpose
    ) const
{
  cnpy::npz_t::const_iterator it = model_.find(key);
  if(it != model_.end()) {
    const cnpy::NpyArray &array = it->second;
    NpyMatrixWrapper np(array);

    mblas::Matrix matrix(context, np.size1(), np.size2(), np.data());

    if (transpose) {
      // TODO
    }

    return std::move(matrix);
  }
  else {
    amunmt_UTIL_THROW2("Missing " << key);
  }
}


}
}

