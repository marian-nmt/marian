//#pragma once

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>

namespace datasets {
namespace mnist {

typedef unsigned char uchar;

const size_t IMAGE_SIZE = 784;
const size_t LABEL_SIZE = 10;

const size_t IMAGE_MAGIC_NUMBER = 2051;
const size_t LABEL_MAGIC_NUMBER = 2049;

auto reverseInt = [](int i) {
  unsigned char c1, c2, c3, c4;
  c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
  return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
};

std::vector<float> ReadImages(const std::string& full_path, int& number_of_images) {
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

  assert(n_rows * n_cols == IMAGE_SIZE);

  int n = number_of_images * IMAGE_SIZE;
  std::vector<float> _dataset(n);

  for (int i = 0; i < n; i++) {
    unsigned char pixel = 0;
    file.read((char*)&pixel, sizeof(pixel));
    _dataset[i] = pixel / 255.0f;
  }
  return _dataset;
}

std::vector<float> ReadLabels(const std::string& full_path, int& number_of_labels) {
  std::ifstream file(full_path);

  if (! file.is_open())
    throw std::runtime_error("Cannot open file `" + full_path + "`!");

  int magic_number = 0;
  file.read((char *)&magic_number, sizeof(magic_number));
  magic_number = reverseInt(magic_number);

  if (magic_number != LABEL_MAGIC_NUMBER)
    throw std::runtime_error("Invalid MNIST label file!");

  file.read((char *)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);

  int n = number_of_labels * LABEL_SIZE;
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
