#include <iostream>
#include <string>
#include "bpe.h"

int main(int argc, char *argv[])
{
  if (argc < 1) {
    std::cout << "No BPE codes. Exit." << std::endl;
  }
  BPE bpe(argv[1]);
  std::string line;
  while (std::getline(std::cin, line)) {
    auto tokens = bpe.Segment(line);
    for (size_t i = 0; i < tokens.size(); ++i) {
      std::cout << tokens[i];
      if (i == tokens.size() - 1) {
        std::cout << std::endl;
      } else {
        std::cout << " ";
      }
    }
  }
  return 0;
}
