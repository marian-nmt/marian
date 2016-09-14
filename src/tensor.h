#pragma once

#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <numeric>
#include <sstream>

#include "definitions.h"
#include "exception.h"
#include "thrust_functions.h"

namespace marian {

inline std::string Debug(const Shape &shape)
{
	std::stringstream strm;
	strm << shape[0];
	assert(shape.size());
	for (size_t i = 1; i < shape.size(); ++i) {
		strm << "x" << shape[i];
	}
	return strm.str();
}

inline size_t GetTotalSize(const Shape &shape)
{
	size_t ret = std::accumulate(shape.begin(), shape.end(),
			  1, std::multiplies<int>());
	return ret;
}

template<class Float>
class TensorImpl {
  private:
    Shape shape_;
    thrust::device_vector<Float> data_;
    size_t tno_;
    static size_t tensorCounter;

  public:
    typedef Float value_type;

    TensorImpl(const Shape& shape, value_type value = 0)
    : shape_(shape), tno_(tensorCounter++)
    {

      // @TODO:
      UTIL_THROW_IF2(shape_.size() != 2,
                     "For now, only 2D Tensors, will be fixed later.");

      UTIL_THROW_IF2(shape_.size() < 1 || shape_.size() > 4,
                     "Wrong number of dimensions: " << shape_.size());

      std::cerr << "Allocating : " << shape[0] << " " << shape[1] << std::endl;

      int size = GetTotalSize(shape_);
      data_.resize(size, value);
    }

    TensorImpl(const TensorImpl&) = delete;
    TensorImpl(TensorImpl&&) = delete;

   value_type operator[](size_t i) const {
      return data_[i];
    }

    auto begin() -> decltype( data_.begin() ) {
      return data_.begin();
    }

    auto begin() const -> decltype( data_.begin() ) {
      return data_.begin();
    }

    auto end() -> decltype( data_.end() ) {
      return data_.end();
    }

    auto end() const -> decltype( data_.end() ) {
      return data_.end();
    }

    const Shape& shape() const {
        return shape_;
    }

    size_t size() const {
      return data_.size();
    }

    value_type* data() {
      return thrust::raw_pointer_cast(data_.data());
    }

    size_t id() const {
      return tno_;
    }

    void set(value_type value) {
      thrust::fill(data_.begin(), data_.end(), value);
    }

    void set(const std::vector<float>::const_iterator &begin, const std::vector<float>::const_iterator &end) {
	  thrust::copy(begin, end, data_.begin());
    }

    void get(std::vector<float>::iterator out) {
	  thrust::copy(data_.begin(), data_.end(), out);      
    }
    
    std::string Debug() const
    {
    	std::stringstream strm;
    	assert(shape_.size());
    	strm << "shape=" << marian::Debug(shape_) << std::endl;

    	// values
    	size_t totSize = GetTotalSize(shape());
    	std::vector<Float> values(totSize);
		thrust::copy(data_.begin(), data_.end(), values.begin());

		size_t ind = 0;
		for (size_t i = 0; i < shape()[0]; ++i) {
			for (size_t j = 0; j < shape()[1]; ++j) {
				strm << values[ind] << " ";
				++ind;
			}
			strm << std::endl;
		}
    	return strm.str();
    }
};

template <typename Type>
size_t TensorImpl<Type>::tensorCounter = 0;

class Tensor {
  private:
    std::shared_ptr<TensorImpl<Float>> pimpl_;

  public:
    typedef TensorImpl<Float>::value_type value_type;

    Tensor() {}
    Tensor(const Shape& shape, value_type value = 0) {
      allocate(shape, value);
    }

    ~Tensor() {}

    void allocate(const Shape& shape, value_type value = 0) {
      if(!pimpl_)
        pimpl_.reset(new TensorImpl<Float>(shape, value));
    }

    value_type operator[](size_t i) const {
      return (*pimpl_)[i];
    }

    size_t size() const {
      return pimpl_->size();
    }

    value_type* data() {
      return pimpl_->data();
    }

    const value_type* data() const {
      return pimpl_->data();
    }

    auto begin() -> decltype( pimpl_->begin() ) {
      return pimpl_->begin();
    }

    auto begin() const -> decltype( pimpl_->begin() ) {
      return pimpl_->begin();
    }

    auto end() -> decltype( pimpl_->begin() ) {
      return pimpl_->begin();
    }

    auto end() const -> decltype( pimpl_->begin() ) {
      return pimpl_->begin();
    }

    const Shape& shape() const {
      return pimpl_->shape();
    }

    void set(value_type value) {
      pimpl_->set(value);
    }

    size_t id() const {
      return pimpl_->id();
    }

    operator bool() {
      return pimpl_ != nullptr;
    }

    std::string Debug() const
    {
    	return pimpl_->Debug();
    }

    void Print() const {
      for (int i = 0; i < size(); ++i) {
        std::cerr << (*this)[i] << " ";
      }
      std::cerr << std::endl;
    }

    //void Load(const std::string &path);
    void set(const std::vector<float>& data);
    void set(const std::vector<float>::const_iterator &begin, const std::vector<float>::const_iterator &end);

    void get(std::vector<float>::iterator out) {
      pimpl_->get(out);
    }
    
    void get(std::vector<float> &vout) {
      pimpl_->get(vout.begin());
    }
};

}
