#include "npz_converter.h"



NpzConverter::NpzConverter(const std::string& file)
  : model_(cnpy::npz_load(file)),
    destructed_(false) {
  }

NpzConverter::~NpzConverter() {
  if(!destructed_)
    model_.destruct();
}

void NpzConverter::Destruct() {
  model_.destruct();
  destructed_ = true;
}

mblas::Matrix NpzConverter::operator[](const std::string& key) const {
  typedef blaze::CustomMatrix<float, blaze::unaligned,
    blaze::unpadded, blaze::rowMajor> BlazeWrapper;
  mblas::Matrix matrix;
  auto it = model_.find(key);
  if(it != model_.end()) {
    NpyMatrixWrapper np(it->second);
    matrix = BlazeWrapper(np.data(), np.size1(), np.size2());
  }
  else {
    std::cerr << "Missing " << key << std::endl;
  }
  return std::move(matrix);
}

mblas::Matrix NpzConverter::operator()(const std::string& key, bool transpose) const {
  mblas::Matrix matrix = (*this)[key];
  mblas::Trans(matrix);
  return std::move(matrix);
}
