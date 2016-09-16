#pragma once
/* Copyright (C) 
 * 2016 - MLAMU & friends
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 * 
 */

#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <numeric>
#include <sstream>

#include "definitions.h"
#include "exception.h"
#include "thrust_functions.h"

namespace marian {

/**
 * @brief Debug shape by printing it. 
 *
 * @param shape Shape of Tensor.
 *
 * @return String of shape.
 */
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

/**
 * @brief Calculate the vector size based on Tensor shape. 
 *
 * @param shape Shape of Tensor.
 *
 * @return Size of Tensor vector.
 */
inline size_t GetTotalSize(const Shape &shape)
{
	size_t ret = std::accumulate(shape.begin(), shape.end(),
			  1, std::multiplies<int>());
	return ret;
}

/**
 * @brief This class manages the Tensor on the GPU. 
 *
 * @tparam Float Data type.
 */
template<class Float>
class TensorImpl {
  private:
    Shape shape_; /*!< Dimenions of Tensor */
    thrust::device_vector<Float> data_; /*< Vector of data that Tensor is managing on GPU. */
    size_t tno_; /*< Tensor number */
    static size_t tensorCounter; /*< Static counter of created Tensors */

  public:
    typedef Float value_type; /*< Tensor value type */

    /**
     * @brief Constructor
     *
     * @param shape Shape of Tensor.
     * @param value Value to fill Tensor's vector with.
     */
    TensorImpl(const Shape& shape, value_type value = 0)
    : shape_(shape), tno_(tensorCounter++)
    {

      // @TODO:
      UTIL_THROW_IF2(shape_.size() != 2,
                     "For now, only 2D Tensors, will be fixed later.");

      UTIL_THROW_IF2(shape_.size() < 1 || shape_.size() > 4,
                     "Wrong number of dimensions: " << shape_.size());

      int size = GetTotalSize(shape_);
      data_.resize(size, value);
    }

    TensorImpl(const TensorImpl&) = delete;
    TensorImpl(TensorImpl&&) = delete;

    /**
     * @brief Get the i-th element of Tensor vector.
     *
     * @param i Index.
     *
     * @return Value of Tensor vector indexed with i.
     */
   value_type operator[](size_t i) const {
      return data_[i];
    }

   /**
    * @brief Get begin iterator of Tensor's vector.
    *
    * @return Vector begin iterator.
    */
    auto begin() -> decltype( data_.begin() ) {
      return data_.begin();
    }

   /**
    * @brief Get begin iterator of Tensor's vector (const).
    *
    * @return Vector begin iterator (const)
    */
    auto begin() const -> decltype( data_.begin() ) {
      return data_.begin();
    }

   /**
    * @brief Get end iterator of Tensor's vector.
    *
    * @return Vector end iterator
    */
    auto end() -> decltype( data_.end() ) {
      return data_.end();
    }

   /**
    * @brief Get end iterator of Tensor's vector (const).
    *
    * @return Vector end iterator (const)
    */
    auto end() const -> decltype( data_.end() ) {
      return data_.end();
    }

    /**
     * @brief Get Tensor's shape (const)
     *
     * @return Shape of Tensor
     */
    const Shape& shape() const {
        return shape_;
    }

    /**
     * @brief Get size of Tensor's vector.
     *
     * @return Length of Tensor's vector.
     */
    size_t size() const {
      return data_.size();
    }

    /**
     * @brief Cast data from Tensor's GPU to value_type.
     *
     * @return Pointer of value_type array.
     */
    value_type* data() {
      return thrust::raw_pointer_cast(data_.data());
    }

    /**
     * @brief Get Tensor id (number).
     *
     * @return Tensor id.
     */
    size_t id() const {
      return tno_;
    }

    /**
     * @brief Fill Tensor's vector with specified value on the GPU.
     *
     * @param value Value to fill vector with.
     */
    void set(value_type value) {
      thrust::fill(data_.begin(), data_.end(), value);
    }

    /**
     * @brief Set Tensor's vector to values of specified vector by copying it to GPU.
     *
     * @param begin Begin iterator of a vector.
     * @param end End iterator of a vector.
     */
    void set(const std::vector<float>::const_iterator &begin, const std::vector<float>::const_iterator &end) {
	  thrust::copy(begin, end, data_.begin());
    }

    /**
     * @brief Copy Tensor's vector from GPU to vector variable on CPU.
     *
     * @param out Vector to copy data to.
     */
    void get(std::vector<float>::iterator out) {
	  thrust::copy(data_.begin(), data_.end(), out);      
    }
    
    /**
     * @brief Debug function.
     *
     * @return Vector in string form.
     */
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

/**
 * @brief Class that communicates with GPU's Tensor.
 */
class Tensor {
  private:
    std::shared_ptr<TensorImpl<Float>> pimpl_; /*< Pointer to Tensor working on GPU */

  public:
    typedef TensorImpl<Float>::value_type value_type; /*< Get value type of GPU's Tensor data */

    /**
     * @brief Default constructor
     */
    Tensor() {}

    /**
     * @brief Constructor that allocates memory.
     *
     * @param shape Shape of Tensor. 
     * @param value Value to fill Tensor's vector with.
     */
    Tensor(const Shape& shape, value_type value = 0) {
      allocate(shape, value);
    }

    /**
     * @brief Default destructor
     */
    ~Tensor() {}

    /**
     * @brief Allocate memory if Tensor doesn't exist on GPU. Otherwise, do nothing.
     *
     * @param shape Shape of Tensor.
     * @param value Value to fill Tensor's vector with.
     */
    void allocate(const Shape& shape, value_type value = 0) {
      if(!pimpl_)
        pimpl_.reset(new TensorImpl<Float>(shape, value));
    }

    /**
     * @brief Get i-th element of GPU Tensor vector (const).
     *
     * @param i Index.
     *
     * @return Value of specified element of Tensor.
     */
    value_type operator[](size_t i) const {
      return (*pimpl_)[i];
    }

    /**
     * @brief Get size of GPU Tensor's vector.
     *
     * @return Size of Tensor vector.
     */
    size_t size() const {
      return pimpl_->size();
    }

    /**
     * @brief Return pointer to GPU Tensor's data.
     *
     * @return Pointer to GPU Tensor's data.
     */
    value_type* data() {
      return pimpl_->data();
    }

    /**
     * @brief Return pointer to GPU Tensor's data (const).
     *
     * @return Pointer to GPU Tensor's data.
     */
    const value_type* data() const {
      return pimpl_->data();
    }

   /**
    * @brief Get begin iterator of GPU Tensor's vector.
    *
    * @return Vector begin iterator.
    */
    auto begin() -> decltype( pimpl_->begin() ) {
      return pimpl_->begin();
    }

   /**
    * @brief Get begin iterator of GPU Tensor's vector (const).
    *
    * @return Vector begin iterator (const)
    */
    auto begin() const -> decltype( pimpl_->begin() ) {
      return pimpl_->begin();
    }

   /**
    * @brief Get end iterator of Tensor's vector.
    *
    * @return Vector end iterator
    */
    auto end() -> decltype( pimpl_->end() ) {
      return pimpl_->end();
    }

   /**
    * @brief Get end iterator of Tensor's vector (const).
    *
    * @return Vector end iterator (const)
    */
    auto end() const -> decltype( pimpl_->end() ) {
      return pimpl_->end();
    }

    /**
     * @brief Get GPU Tensor's shape.
     *
     * @return Tensor's shape.
     */
    const Shape& shape() const {
      return pimpl_->shape();
    }

    /**
     * @brief Fill GPU Tensor's vector with specified value.
     *
     * @param value Value to fill Tensor with.
     */
    void set(value_type value) {
      pimpl_->set(value);
    }

    /**
     * @brief Get GPU Tensor id (number).
     *
     * @return Tensor id.
     */
    size_t id() const {
      return pimpl_->id();
    }

    /**
     * @brief Check if Tensor is allocated.
     *
     * @return True or False
     */
    operator bool() {
      return pimpl_ != nullptr;
    }

    /**
     * @brief Run Debug on GPU Tensor.
     *
     * @return String of Tensor's data.
     */
    std::string Debug() const
    {
    	return pimpl_->Debug();
    }

    /**
     * @brief Print Tensor data on CPU (?) (const). 
     */
    void Print() const {
      for (int i = 0; i < size(); ++i) {
        std::cerr << (*this)[i] << " ";
      }
      std::cerr << std::endl;
    }

    //void Load(const std::string &path);
    
    /**
     * @brief Set GPU Tensor's vector to values of specified vector.
     *
     * @param data Vector copied to GPU.
     */
    void set(const std::vector<float>& data);
    /**
     * @brief Fill GPU Tensor's vector using values from the specified vector.
     *
     * @param begin Begin iterator of vector being copied.
     * @param end End iterator of vector being copied.
     */
    void set(const std::vector<float>::const_iterator &begin, const std::vector<float>::const_iterator &end);

    /**
     * @brief Copy Tensor's vector from GPU to vector variable on CPU (const).
     *
     * @param out Vector iterator used in copying.
     */
    void get(std::vector<float>::iterator out) const {
      pimpl_->get(out);
    }
    
    /**
     * @brief Copy Tensor's vector from GPU to vector variable on CPU.
     *
     * @param out Vector to copy data to.
     */
    void get(std::vector<float> &vout) const {
      vout.resize(size());
      pimpl_->get(vout.begin());
    }
};

/**
 * @brief Operator to set data on Tensor using vector.
 *
 * @param t Tensor.
 * @param vec Vector used to set data in Tensor.
 *
 * @return Tensor with assigned data.
 */
Tensor& operator<<(Tensor& t, const std::vector<float> &vec);

/**
 * @brief Operator to get data from Tensor to vector.
 *
 * @param vec Vector to save copied data.
 * @param t Tensor to copy data from.
 *
 * @return Vector with copied data.
 */
std::vector<float>& operator<<(std::vector<float> &vec, const Tensor& t);

}
