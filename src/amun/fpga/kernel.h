#pragma once
#include <string>
#include "types-fpga.h"

namespace amunmt {
namespace FPGA {

cl_context CreateContext(
    size_t maxDevices,
    cl_device_id *devices,
    cl_uint &numDevices);

void CreateProgram(OpenCLInfo &openCLInfo, const std::string &filePath);

cl_kernel CreateKernel(const std::string &kernelName, const OpenCLInfo &openCLInfo);
cl_command_queue CreateCommandQueue(const OpenCLInfo &openCLInfo);

unsigned char *loadBinaryFile(const char *file_name, size_t *size);



}
}
