#include "common/file_stream.h"

#include <streambuf>
#include <string>
#include <vector>
#ifdef _MSC_VER
#include <windows.h>
#include <io.h>
#else
#include <sys/types.h>
#include <unistd.h>
#endif

namespace marian {
namespace io {

// Get error strings out of errno.
namespace {
#ifdef __GNUC__
const char *HandleStrerror(int ret, const char *buf) __attribute__ ((unused));
const char *HandleStrerror(const char *ret, const char * /*buf*/) __attribute__ ((unused));
#endif
// At least one of these functions will not be called.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#endif
// The XOPEN version.
const char *HandleStrerror(int ret, const char *buf) {
  if (!ret) return buf;
  return NULL;
}

// The GNU version.
const char *HandleStrerror(const char *ret, const char * /*buf*/) {
  return ret;
}
#ifdef __clang__
#pragma clang diagnostic pop
#endif

std::string StrError() {
  char buf[200];
  buf[0] = 0;
#if defined(sun) || defined(_WIN32) || defined(_WIN64)
  return strerror(errno);
#else
  return HandleStrerror(strerror_r(errno, buf, 200), buf);
#endif
}
} // namespace

void RewindFile(int fd) {
  ABORT_IF((off_t)-1 == lseek(fd, 0, SEEK_SET), "lseek to 0 failed in fd {}", fd);
}

ReadFDBuf::ReadFDBuf(int fd, std::size_t buffer_size)
: fd_(fd), mem_(buffer_size) {
  setg(End(), End(), End());
}

ReadFDBuf::int_type ReadFDBuf::underflow() {
  if (gptr() == egptr()) {
    // Reached end, refill
    ssize_t got = Read();
    if (!got) {
      // End of file
      return traits_type::eof();
    }
    setg(Begin(), Begin(), Begin() + got);
  }
  return traits_type::to_int_type(*gptr());
}

// If the putback goes below the buffer, try to seek backwards.
ReadFDBuf::int_type ReadFDBuf::pbackfail(int c) {
  /* "It is unspecified whether the content of the controlled input
   * sequence is modified if the function succeeds and c does not match the
   * character at that position."
   * -- http://www.cplusplus.com/reference/streambuf/streambuf/pbackfail/
   */
  if (gptr() > Begin()) {
    setg(Begin(), gptr() - 1, End());
  } else {
    if (lseek(fd_, -1, SEEK_CUR) == -1) {
      return EOF;
    }
    ssize_t got = Read();
    if (!got) {
      // This happens if the file was truncated underneath us.
      return traits_type::eof();
    }
    setg(Begin(), Begin(), Begin() + got);
  }
  return traits_type::to_int_type(*gptr());
}

// Read some amount into [Begin(), End()), returning the amount read.
ssize_t ReadFDBuf::Read() {
  ssize_t got;
  // Loop to keep reading if EINTR happens.
  // This way the program is robust to Ctrl+Z then backgrounding.
  do {
    errno = 0;
    got =
#ifdef _MSC_VER
      _read
#else
      read
#endif
      (fd_, Begin(), End() - Begin());
  } while (got == -1 && errno == EINTR);
  ABORT_IF(got < 0, "Error reading fd {}: {}", fd_, StrError());
  return got;
}

WriteFDBuf::WriteFDBuf(int fd, std::size_t buffer_size)
: fd_(fd), mem_(buffer_size) {
  setp(End(), End());
}

WriteFDBuf::~WriteFDBuf() { sync(); }

WriteFDBuf::int_type WriteFDBuf::overflow(int c) {
  if (c == EOF) {
    // Apparently overflow(EOF) means sync().
    sync();
    return c;
  }
  if (pptr() == epptr()) {
    // Out of buffer.  Write and reset.
    sync();
    setp(Begin(), End());
  }
  // Put character on the end.
  *pptr() = traits_type::to_char_type(c);
  pbump(1);
  return c;
}

// Write everything in the buffer to the file.
int WriteFDBuf::sync() {
  const char *from = pbase();
  const char *to = pptr();
  while (from != to) {
    from += WriteSome(from, to);
  }
  // We die on all failures.
  return 0;
}

// Write part of the buffer, returning the amount written.
ssize_t WriteFDBuf::WriteSome(const char *from, const char *to) {
  ssize_t put = 0;
  do {
    errno = 0;
    put =
#ifdef _MSC_VER
      _write
#else
      write
#endif
      (fd_, from, to - from);
  } while (put == -1 && errno == EINTR);
  ABORT_IF(put < 0, "Error writing to fd {}: {}", fd_, StrError());
  return put;
}

} // namespace io
} // namespace marian
