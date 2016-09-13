//#pragma once

#include <string>
#include <iostream>
#include <fstream>
#include <vector>

namespace datasets {
namespace mnist {

typedef unsigned char uchar;

auto reverseInt = [](int i) {
  unsigned char c1, c2, c3, c4;
  c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
  return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
};

std::vector<float> ReadImages(const std::string& full_path, int& number_of_images, int& image_size) {
  std::ifstream file(full_path);

  if (! file.is_open())
    throw std::runtime_error("Cannot open file `" + full_path + "`!");

  int magic_number = 0, n_rows = 0, n_cols = 0;

  file.read((char *)&magic_number, sizeof(magic_number));
  magic_number = reverseInt(magic_number);

  if (magic_number != 2051)
    throw std::runtime_error("Invalid MNIST image file!");

  file.read((char *)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
  file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
  file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

  image_size = n_rows * n_cols;
  int n = number_of_images * image_size;
  std::vector<float> _dataset(n);
  unsigned char pixel = 0;

  for (int i = 0; i < n; i++) {
    file.read((char*)&pixel, sizeof(pixel));
    _dataset[i] = pixel / 255.0f;
  }
  return _dataset;
}

std::vector<int> ReadLabels(const std::string& full_path) {
  std::ifstream file(full_path);

  if (! file.is_open())
    throw std::runtime_error("Cannot open file `" + full_path + "`!");

  int magic_number = 0;
  file.read((char *)&magic_number, sizeof(magic_number));
  magic_number = reverseInt(magic_number);

  if (magic_number != 2049)
    throw std::runtime_error("Invalid MNIST label file!");

  int number_of_labels = 0;
  file.read((char *)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);

  std::vector<int> _dataset(number_of_labels);
  for (int i = 0; i < number_of_labels; i++) {
    file.read((char*)&_dataset[i], 1);
  }

  return _dataset;
}

} // namespace mnist
} // namespace datasets


//int main(int argc, const char *argv[]) {
  //int numImg, imgSize;
  //auto images = datasets::mnist::ReadImages("../examples/mnist/t10k-images-idx3-ubyte", numImg, imgSize);
  //auto labels = datasets::mnist::ReadLabels("../examples/mnist/t10k-labels-idx1-ubyte");

  //std::cout
    //<< "Number of images: " << numImg << std::endl
    //<< "Image size: " << imgSize << std::endl;

  //for (int i = 0; i < 3; i++) {
    //for (int j = 0; j < imgSize; j++) {
      //std::cout << images[(i * imgSize) + j] << ",";
    //}
    //std::cout << " label=" << (int)labels[i] << std::endl;
  //}
  //return 0;
//}
