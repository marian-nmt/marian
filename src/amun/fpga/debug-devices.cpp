#include "debug-devices.h"

using namespace std;

void DebugDevicesInfo(cl_platform_id id)
{
  cl_device_id devices[100];
  cl_uint numDevices = 0;
  CheckError( clGetDeviceIDs(id, CL_DEVICE_TYPE_GPU, 100, devices, &numDevices) );
  DebugDevicesInfo(devices, numDevices);
}

void DebugDevicesInfo(cl_device_id *devices, cl_uint numDevices)
{
  cerr << "numDevices=" << numDevices << endl;

  for (int i=0; i<numDevices; i++)
  {
    DebugDeviceInfo(devices[i]);
  }
}


void DebugDeviceInfo(cl_device_id id)
{
  char buffer[10240];
  cl_uint buf_uint;
  cl_ulong buf_ulong;
  cerr << id << ":";

  CheckError( clGetDeviceInfo(id, CL_DEVICE_NAME, sizeof(buffer), buffer, NULL) );
  cerr << " extension=" << buffer;

  CheckError( clGetDeviceInfo(id, CL_DEVICE_VENDOR, sizeof(buffer), buffer, NULL) );
  cerr << " vendor=" << buffer;

  CheckError( clGetDeviceInfo(id, CL_DEVICE_VERSION, sizeof(buffer), buffer, NULL) );
  cerr << " version=" << buffer;

  CheckError( clGetDeviceInfo(id, CL_DRIVER_VERSION, sizeof(buffer), buffer, NULL) );
  cerr << " driver version=" << buffer;

  CheckError( clGetDeviceInfo(id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(buf_uint), &buf_uint, NULL) );
  cerr << " compute units=" << buf_uint;

  CheckError( clGetDeviceInfo(id, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(buf_uint), &buf_uint, NULL) );
  cerr << " clock freq=" << buf_uint;

  CheckError( clGetDeviceInfo(id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(buf_ulong), &buf_ulong, NULL) );
  cerr << " global mem=" << buf_ulong;

  cerr << endl;

}

