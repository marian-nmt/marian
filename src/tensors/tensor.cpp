#include "tensors/tensor.h"
#include "tensors/tensor_operators.h"
#include "common/io.h"

namespace marian {

template <typename T>
std::string TensorBase::debug(int precision, int dispCols) {
  // values
  size_t totSize = shape_.elements();
  std::vector<T> values(totSize);

  get(values);

  std::stringstream strm;
  assert(shape_.size());
  strm << shape_;
  strm << " type=" << type_;
  strm << " device=" << backend_->getDeviceId();
  strm << " ptr=" << (size_t)memory_->data();
  strm << " bytes=" << memory_->size();
  strm << std::endl;

  int colWidth  = precision + 4;

  if(isFloat(type_))
    strm << std::fixed << std::setprecision(precision) << std::setfill(' ');
  else
    strm << std::fixed << std::setprecision(0) << std::setfill(' ');

  double maxv = std::numeric_limits<double>::lowest();
  double minv = std::numeric_limits<double>::max();
  double l2Sum = 0.0;
  for(int i = 0; i < values.size(); ++i) {
    if((double)values[i] > maxv) maxv = (double)values[i];
    if((double)values[i] < minv) minv = (double)values[i];
    l2Sum += (double)values[i] * (double)values[i];
  }
  strm << "min: " << minv << " max: " << maxv << " l2-norm: " << sqrt(l2Sum) << std::endl;

  for(int i = 0; i < values.size(); ++i) {
    std::vector<int> dims;
    shape().dims(i, dims);

    bool disp = true;
    for(int j = 0; j < dims.size(); ++j)
      disp = disp && (dims[j] < dispCols || dims[j] >= shape()[j] - dispCols);

    if(disp) {
      if(dims.back() == 0) {
        bool par = true;
        std::vector<std::string> p;
        for(int j = (int)dims.size() - 1; j >= 0; --j) {
          if(dims[j] != 0)
            par = false;

          p.push_back(par ? "[" : " ");
        }
        for(auto it = p.rbegin(); it != p.rend(); ++it)
          strm << *it;
        strm << " ";
      }

      strm << std::setw(colWidth);
      if(isFloat(type_)) {
        strm << (double)values[i];
      } else if(isSignedInt(type_)) {
        strm << (int64_t)values[i];
      } else {
        strm << (uint64_t)values[i];
      }
      strm << " ";

      if(dims.back() + 1 == shape().back()) {
        for(int j = (int)dims.size() - 1; j >= 0; --j) {
          if(dims[j] + 1 != shape()[j])
            break;
          strm << "]";
        }
        strm << std::endl;
      }

      bool prev = true;
      for(int j = (int)dims.size() - 1; j >= 0; --j) {
        if(j < (int)dims.size() - 1)
          prev = prev && dims[j + 1] + 1 == shape()[j + 1];
        if(prev && dims[j] + 1 == dispCols && shape()[j] > 2 * dispCols) {
          if(j < (int)dims.size() - 1)
            for(int k = 0; k <= j; ++k)
              strm << " ";
          strm << "... ";
          if(j < (int)dims.size() - 1)
            strm << std::endl;
          break;
        }
      }
    }
  }
  strm << std::endl;
  return strm.str();
}

template std::string TensorBase::debug<float16>(int, int);
template std::string TensorBase::debug<float  >(int, int);
template std::string TensorBase::debug<double >(int, int);

template std::string TensorBase::debug<uint8_t >(int, int);
template std::string TensorBase::debug<uint16_t>(int, int);
template std::string TensorBase::debug<uint32_t>(int, int);
template std::string TensorBase::debug<uint64_t>(int, int);

template std::string TensorBase::debug<int8_t >(int, int);
template std::string TensorBase::debug<int16_t>(int, int);
template std::string TensorBase::debug<int32_t>(int, int);
template std::string TensorBase::debug<int64_t>(int, int);

// fill an io::item with data from a Tensor, used for saving 
// and other IO operations.
void TensorBase::get(io::Item& item, const std::string& name) {
  item.name  = name;
  item.shape = shape_;
  item.type  = type_;

  item.bytes.resize(memory_->size());
  copy(backend_,
       memory_->data<char>(),
       memory_->data<char>() + memory_->size(),
       item.bytes.data());
}

void TensorBase::set(const io::Item& item) {
  ABORT_IF(item.type != type_, "Tensor type {} and item type {} do not match", type_, item.type);
  ABORT_IF(item.shape != shape_, "Tensor shape {} and item shape {} do not match", shape_, item.shape);
  ABORT_IF(item.bytes.size() > memory_->size(), "Item data size {} too large for memory {}", item.bytes.size(), memory_->size());
  copy(backend_,
       item.bytes.data(),
       item.bytes.data() + item.bytes.size(),
       memory_->data<char>());
}

size_t TensorBase::hash() {
  io::Item temp;
  size_t seed = 0;
  get(temp, "temp");
  for(auto c : temp.bytes)
    util::hash_combine(seed, c);
  return seed;
}

}  // namespace marian

