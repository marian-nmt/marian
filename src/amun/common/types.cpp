#include "types.h"

using namespace std;

namespace amunmt {

std::ostream& operator<<(std::ostream& out, const DeviceInfo& obj)
{
  out << obj.deviceType << " t=" << obj.threadInd << " d=" << obj.deviceId;
  return out;
}

}

