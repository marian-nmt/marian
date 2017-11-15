#pragma once
#include "types-fpga.h"

namespace amunmt {
namespace FPGA {

void HelloWorld(
    cl_kernel &kernel,
    const OpenCLInfo &openCLInfo,
    const cl_command_queue &commands,
    size_t dataSize);

}
}

