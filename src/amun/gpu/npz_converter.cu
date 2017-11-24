#include "npz_converter.h"
#include "common/exception.h"
#include "mblas/matrix_functions.h"

namespace amunmt {
namespace GPU {

NpzConverter::NpzConverter(const std::string& file)
  : model_(cnpy::npz_load(file)),
    destructed_(false)
{
}

NpzConverter::~NpzConverter() {
  if(!destructed_)
    model_.destruct();
}

void NpzConverter::Destruct() {
  model_.destruct();
  destructed_ = true;
}

template<typename T>
T Debug(const T *data, size_t size)
{
  T sum = 0;
  for (size_t i = 0; i < size; ++i) {
    sum += data[i];
  }
  return sum;
}

std::shared_ptr<mblas::Matrix> NpzConverter::get(const std::string& key, bool mandatory, bool transpose) const
{
  //mblas::TestMemCpy();

  std::shared_ptr<mblas::Matrix> ret;
  auto it = model_.find(key);
  if(it != model_.end()) {
    NpyMatrixWrapper np(it->second);
    size_t size = np.size();

    HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
    std::cerr << "np=" << np.data() << " " << size << " " << Debug(np.data(), size) << std::endl;

    mblas::TestMemCpy(size, np.data());

    mblas::Matrix *matrix = new mblas::Matrix(np.size1(), np.size2(), 1, 1);
    mblas::copy(np.data(), size, matrix->data(), cudaMemcpyHostToDevice);
    HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));

    std::cerr << key << "=" << matrix->Debug(1) << std::endl;

    std::vector<float> h_vec2(size);
    mblas::copy(matrix->data(), size, h_vec2.data(), cudaMemcpyDeviceToHost);
    std::cerr << "h_vec2=";
    float sum = 0;
    for (size_t i = 0; i < size; ++i) {
      //cerr << h_vec2[i] << " ";
      sum += h_vec2[i];
    }
    std::cerr << sum;

    std::cerr << std::endl;

    if (transpose) {
      mblas::Transpose(*matrix);
    }

    ret.reset(matrix);
  }
  else if (mandatory) {
    std::cerr << "Error: Matrix not found:" << key << std::endl;
    //amunmt_UTIL_THROW2(strm.str()); //  << key << std::endl
    abort();
  }
  else {
    mblas::Matrix *matrix = new mblas::Matrix();
    ret.reset(matrix);
  }

  //std::cerr << "key=" << key << " " << matrix.Debug(1) << std::endl;
  return ret;
}

std::shared_ptr<mblas::Matrix> NpzConverter::getFirstOfMany(const std::vector<std::pair<std::string, bool>> keys, bool mandatory) const
{
  std::shared_ptr<mblas::Matrix> ret;
  for (auto key : keys) {
    auto it = model_.find(key.first);
    if(it != model_.end()) {
      NpyMatrixWrapper np(it->second);
      mblas::Matrix *matrix = new mblas::Matrix(np.size1(), np.size2(), 1, 1);
      mblas::copy(np.data(), np.size(), matrix->data(), cudaMemcpyHostToDevice);

      if (key.second) {
        mblas::Transpose(*matrix);
      }
      ret.reset(matrix);
      return ret;
    }
  }

  if (mandatory) {
    std::cerr << "Error: Matrix not found:" << keys[0].first << std::endl;
    //amunmt_UTIL_THROW2(strm.str()); //  << key << std::endl
    abort();
  }
  else {
    std::cerr << "Optional matrix not found, continuing: " << keys[0].first << std::endl;
  }

  return ret;

}


}
}
