#pragma once
#include <string>
#include "types-fpga.h"

namespace amunmt {
namespace FPGA {

cl_context CreateContext(
    size_t maxDevices,
    cl_device_id *devices,
    cl_uint &numDevices);

cl_kernel CreateKernel(const std::string &filePath, const std::string &kernelName, const OpenCLInfo &openCLInfo);
cl_command_queue CreateCommandQueue(const OpenCLInfo &openCLInfo);




}
}
