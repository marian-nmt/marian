#pragma once
#include <string>
#include "types.h"

namespace amunmt {
namespace FPGA {

int ExecuteKernel(const std::string &filePath, const cl_context &context, const cl_device_id &device);
//int HelloWorld2();

}
}
