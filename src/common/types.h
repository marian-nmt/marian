#pragma once

#include <cstdlib>
#include <cstdint>
#include <vector>

typedef size_t Word;
typedef std::vector<Word> Words;

const Word EOS = 0;
const Word UNK = 1;

enum DeviceType
{
	CPUDevice,
	GPUDevice
};

struct DeviceInfo
{
  DeviceType deviceType;
  size_t threadInd;
  size_t deviceInd;
};
