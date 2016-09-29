#include <iostream>
#include <string>
#include "common/processor/bpe.h"

#include "common/utf8.h"

int main(int argc, char *argv[])
{
  if (argc < 1) {
    std::cout << "No BPE codes. Exit." << std::endl;
  }
  BPE bpe(argv[1]);
  std::string line;
  std::ios_base::sync_with_stdio(false);
  while (std::getline(std::cin, line)) {
    if (line.empty()) std::cout << std::endl;
    bpe.PrintSegment(line);
  }
  return 0;
}
