#pragma once
#include <string>
#include "types.h"

namespace amunmt {
namespace FPGA {

cl_kernel CreateKernel(const std::string &filePath, const cl_context &context, const cl_device_id &device);
cl_command_queue CreateCommandQueue(const cl_context &context, const cl_device_id &device);
void ExecuteKernel(cl_kernel &kernel, const cl_context &context, const cl_device_id &device, cl_command_queue &commands);

void HelloWorld(cl_kernel &kernel, const cl_context &context, const cl_device_id &device, cl_command_queue &commands);


}
}
