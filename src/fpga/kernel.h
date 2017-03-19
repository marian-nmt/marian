#pragma once
#include <string>
#include "types.h"

namespace amunmt {
namespace FPGA {

cl_kernel CreateKernel(const std::string &filePath, const cl_context &context, const cl_device_id &device);
void ReleaseKernel(cl_kernel &kernel);

int ExecuteKernel(cl_kernel &kernel, const cl_context &context, const cl_device_id &device);
//int HelloWorld2();

}
}
