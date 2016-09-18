//#pragma once

// This file is part of the Marian toolkit.
// Marian is copyright (c) 2016 Marcin Junczys-Dowmunt.
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

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>

namespace datasets {
namespace mnist {

typedef unsigned char uchar;


const size_t IMAGE_MAGIC_NUMBER = 2051;
const size_t LABEL_MAGIC_NUMBER = 2049;

auto reverseInt = [](int i) {
  unsigned char c1, c2, c3, c4;
  c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
  return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
};

std::vector<float> ReadImages(const std::string& full_path, int& number_of_images, int imgSize) {
  std::ifstream file(full_path);

  if (! file.is_open())
    throw std::runtime_error("Cannot open file `" + full_path + "`!");

  int magic_number = 0;
  file.read((char *)&magic_number, sizeof(magic_number));
  magic_number = reverseInt(magic_number);

  if (magic_number != IMAGE_MAGIC_NUMBER)
    throw std::runtime_error("Invalid MNIST image file!");

  int n_rows = 0;
  int n_cols = 0;
  file.read((char *)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
  file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
  file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

  assert(n_rows * n_cols == imgSize);

  int n = number_of_images * imgSize;
  std::vector<float> _dataset(n);

  for (int i = 0; i < n; i++) {
    unsigned char pixel = 0;
    file.read((char*)&pixel, sizeof(pixel));
    _dataset[i] = pixel / 255.0f;
  }
  return _dataset;
}

std::vector<float> ReadLabels(const std::string& full_path, int& number_of_labels, int label_size) {
  std::ifstream file(full_path);

  if (! file.is_open())
    throw std::runtime_error("Cannot open file `" + full_path + "`!");

  int magic_number = 0;
  file.read((char *)&magic_number, sizeof(magic_number));
  magic_number = reverseInt(magic_number);

  if (magic_number != LABEL_MAGIC_NUMBER)
    throw std::runtime_error("Invalid MNIST label file!");

  file.read((char *)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);

  int n = number_of_labels * label_size;
  std::vector<float> _dataset(n, 0.0f);

  for (int i = 0; i < number_of_labels; i++) {
    unsigned char label;
    file.read((char*)&label, 1);
    _dataset[(i * 10) + (int)(label)] = 1.0f;
  }

  return _dataset;
}

} // namespace mnist
} // namespace datasets


//int main(int argc, const char *argv[]) {
  //int numImg = 0;
  //auto images = datasets::mnist::ReadImages("../examples/mnist/t10k-images-idx3-ubyte", numImg);
  //auto labels = datasets::mnist::ReadLabels("../examples/mnist/t10k-labels-idx1-ubyte", numImg);

  //std::cout << "Number of images: " << numImg << std::endl;

  //for (int i = 0; i < 3; i++) {
    //for (int j = 0; j < datasets::mnist::IMAGE_SIZE; j++) {
      //std::cout << images[(i * datasets::mnist::IMAGE_SIZE) + j] << ",";
    //}
    //std::cout << "\nlabels= ";
    //for (int k = 0; k < 10; k++) {
      //std::cout << labels[(i * 10) + k] << ",";
    //}
    //std::cout << std::endl;
  //}
  //return 0;
//}
