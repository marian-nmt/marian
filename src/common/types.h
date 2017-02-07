#pragma once

#include <cstdlib>
#include <cstdint>
#include <vector>
#include <iostream>

namespace amunmt {

typedef size_t Word;
typedef std::vector<Word> Words;

const Word EOS = 0;
const Word UNK = 1;

enum DeviceType
{
	CPUDevice = 7,
	GPUDevice = 11
};

struct DeviceInfo
{
  friend std::ostream& operator<<(std::ostream& out, const DeviceInfo& obj);

  DeviceType deviceType;
  size_t threadInd;
  size_t deviceId;
};

}

