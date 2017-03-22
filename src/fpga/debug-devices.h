#pragma once
#include "types-fpga.h"

void DebugDevicesInfo(cl_platform_id id);
void DebugDevicesInfo(cl_device_id *devices, cl_uint numDevices);
void DebugDeviceInfo(cl_device_id id);


