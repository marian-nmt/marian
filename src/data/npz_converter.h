#pragma once

// This file is part of the Marian toolkit.

//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "cnpy/cnpy.h"
#include "tensor.h"  //XXX Marcin, is this include actually needed? It appears to not be used.

/**
 * @brief Loads model data stored in a npz file.
 *
 * Use of this class enables such data to later be stored in standard Marian
 * data structures.
 *
 * Note: this class makes use of the 3rd-party class <code>npy</code>.
 */
class NpzConverter {
  // Private inner classes of the NpzConverter class
private:
  /**
   * Wraps npy data such that the underlying matrix shape and
   *    matrix data are made accessible.
   */
  class NpyMatrixWrapper {
  public:
    /**
     * Constructs a wrapper around an underlying npy data structure,
     *    enabling the underlying data to be accessed as a matrix.
     *
     * @param npy the underlying data
     */
    NpyMatrixWrapper(const cnpy::NpyArray& npy) : npy_(npy) {}

    /**
     * Returns the total number of elements in the underlying matrix.
     *
     * @return the total number of elements in the underlying matrix
     */
    size_t size() const { return size1() * size2(); }

    /**
     * Returns a pointer to the raw data that underlies the matrix.
     *
     * @return a pointer to the raw data that underlies the matrix
     */
    float* data() const { return (float*)npy_.data; }

    /**
     * Given the index (i, j) of a matrix element,
     *   this operator returns the float value from the underlying npz data
     *   that is stored in the matrix.
     *
     * XXX: Marcin, is the following correct? Or do I have the row/column labels
     * swapped?
     *
     * @param i Index of a column in the matrix
     * @param j Index of a row in the matrix
     *
     * @return the float value stored at column i, row j of the matrix
     */
    float operator()(size_t i, size_t j) const {
      return ((float*)npy_.data)[i * size2() + j];
    }

    /**
     * Returns the number of columns in the matrix.
     *
     * XXX: Marcin, is this following correct? Or do I have the row/column
     * labels swapped?
     *
     * @return the number of columns in the matrix
     */
    size_t size1() const { return npy_.shape[0]; }

    /**
     * Returns the number of rows in the matrix.
     *
     * XXX: Marcin, is this following correct? Or do I have the row/column
     * labels swapped?
     *
     * @return the number of rows in the matrix
     */
    size_t size2() const {
      if(npy_.shape.size() == 1)
        return 1;
      else
        return npy_.shape[1];
    }

  private:
    /** Instance of the underlying (3rd party) data structure. */
    const cnpy::NpyArray& npy_;

  };  // End of NpyMatrixWrapper class

  // Public methods of the NpzConverter class
public:
  /**
   * Constructs an object that reads npz data from a file.
   *
   * @param file Path to file containing npz data
   */
  NpzConverter(const std::string& file)
      : model_(cnpy::npz_load(file)), destructed_(false) {}

  /**
   * Destructs the model that underlies this NpzConverter object,
   *    if that data has not already been destructed.
   */
  ~NpzConverter() {
    if(!destructed_)
      model_.destruct();
  }

  /**
   * Destructs the model that underlies this NpzConverter object,
   *    and marks that data as having been destructed.
   */
  void Destruct() {
    model_.destruct();
    destructed_ = true;
  }

  /**
   * Loads data corresponding to a search key into the provided vector.
   *
   * @param key Search key                                    XXX Marcin, what
   * type of thing is "key"? What are we searching for here?
   * @param data Container into which data will be loaded     XXX Lane, is there
   * a way in Doxygen to mark and inout variable?
   * @param shape Shape object into which the number of rows and columns of the
   * vectors will be stored
   */
  void Load(const std::string& key,
            std::vector<float>& data,
            marian::Shape& shape) const {
    auto it = model_.find(key);
    if(it != model_.end()) {
      NpyMatrixWrapper np(it->second);
      data.clear();
      data.resize(np.size());
      std::copy(np.data(), np.data() + np.size(), data.begin());

      shape = {(int)np.size1(), (int)np.size2()};

    } else {
      std::cerr << "Missing " << key << std::endl;
    }
  }

  // Private member data of the NpzConverter class
private:
  /** Underlying npz data */
  cnpy::npz_t model_;

  /** Indicates whether the underlying data has been destructed. */
  bool destructed_;

};  // End of NpzConverter class
