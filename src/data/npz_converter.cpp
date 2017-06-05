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

#include "npz_converter.h"

NpzConverter::NpzConverter(const std::string& file)
    : model_(cnpy::npz_load(file)), destructed_(false) {}

NpzConverter::~NpzConverter() {
  if(!destructed_)
    model_.destruct();
}

void NpzConverter::Destruct() {
  model_.destruct();
  destructed_ = true;
}

/** TODO: Marcin, what does this function do? Why isn't it a method? */
mblas::Matrix NpzConverter::operator[](const std::string& key) const {
  typedef blaze::
      CustomMatrix<float, blaze::unaligned, blaze::unpadded, blaze::rowMajor>
          BlazeWrapper;
  mblas::Matrix matrix;
  auto it = model_.find(key);
  if(it != model_.end()) {
    NpyMatrixWrapper np(it->second);
    matrix = BlazeWrapper(np.data(), np.size1(), np.size2());
  } else {
    std::cerr << "Missing " << key << std::endl;
  }
  return std::move(matrix);
}

mblas::Matrix NpzConverter::operator()(const std::string& key,
                                       bool transpose) const {
  mblas::Matrix matrix = (*this)[key];
  mblas::Trans(matrix);
  return std::move(matrix);
}
