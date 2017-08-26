#pragma once

#include <cstdlib>
#include <cstdint>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <boost/timer/timer.hpp>

namespace amunmt {

typedef size_t Word;
typedef std::vector<Word> Words;

const Word EOS_ID = 0;
const Word UNK_ID = 1;

const std::string EOS_STR = "</s>";
const std::string UNK_STR = "<unk>";

enum DeviceType
{
	CPUDevice = 7,
	GPUDevice = 11,
	FPGADevice = 13
};

struct DeviceInfo
{
  friend std::ostream& operator<<(std::ostream& out, const DeviceInfo& obj);

  DeviceType deviceType;
  size_t threadInd;
  size_t deviceId;
};

/////////////////////////////////////////////////////////////////////////////////////
extern std::unordered_map<std::string, boost::timer::cpu_timer> timers;

#define BEGIN_TIMER(str) {}
#define PAUSE_TIMER(str) {}
//#define BEGIN_TIMER(str) { HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream())); timers[str].resume(); }
//#define PAUSE_TIMER(str) { HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream())); \
							timers[str].stop(); }
}

