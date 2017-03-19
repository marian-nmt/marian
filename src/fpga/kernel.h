#pragma once
#include <string>
#include "types.h"

namespace amunmt {
namespace FPGA {

cl_context CreateContext(
    size_t maxDevices,
    cl_device_id *devices,
    cl_uint &numDevices);

cl_kernel CreateKernel(const std::string &filePath, const cl_context &context, const cl_device_id &device);
cl_command_queue CreateCommandQueue(const cl_context &context, const cl_device_id &device);

void HelloWorld(
    cl_kernel &kernel,
    const cl_context &context,
    const cl_device_id &device,
    const cl_command_queue &commands,
    size_t dataSize);



}
}
