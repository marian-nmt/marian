#pragma once
#include "types.h"

namespace amunmt {
namespace FPGA {

void HelloWorld(
    cl_kernel &kernel,
    const cl_context &context,
    const cl_device_id &device,
    const cl_command_queue &commands,
    size_t dataSize);

}
}

