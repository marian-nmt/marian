#include "filesystem.h"

#ifndef _MSC_VER
// don't include these on Windows:
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace marian {
namespace filesystem {

#ifdef _MSC_VER
// Pretend that Windows knows no named pipes. It does, by the way, but
// they seem to be different from pipes on Unix / Linux. See
// https://docs.microsoft.com/en-us/windows/win32/ipc/named-pipes
bool is_fifo(char const* /*path*/) {
  return false;
}
#else
bool is_fifo(char const* path) {
  struct stat buf;
  stat(path, &buf);
  return S_ISFIFO(buf.st_mode);
}
#endif

bool is_fifo(std::string const& path) {
  return is_fifo(path.c_str());
}

} // end of namespace marian::filesystem
} // end of namespace marian
