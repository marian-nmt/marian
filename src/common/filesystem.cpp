#include "filesystem.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

namespace marian {
namespace filesystem {

bool is_fifo(char const* path) {
  struct stat buf;
  stat(path, &buf);
  return S_ISFIFO(buf.st_mode);
}

} // end of namespace marian::filesystem
} // end of namespace marian
