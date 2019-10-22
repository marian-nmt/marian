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

///////////////////////////////////////////////////////////////////////////////////////////////
#ifndef _MSC_VER
int TemporaryFile2::mkstemp_and_unlink(char *tmpl) {
  int ret = mkstemp(tmpl);
  if(unlink_ && ret != -1) {
    ABORT_IF(unlink(tmpl), "Error while deleting '{}'", tmpl);
  }
  return ret;
}
#endif

int TemporaryFile2::MakeTemp(const std::string &base) {
#ifdef _MSC_VER
  char *name = tempnam(base.c_str(), "marian.");
  ABORT_IF(name == NULL, "Error while making a temporary based on '{}'", base);

  int oflag = _O_RDWR | _O_CREAT | _O_EXCL;
  if(unlink_)
    oflag |= _O_TEMPORARY;

  int ret = open(name, oflag, _S_IREAD | _S_IWRITE);
  ABORT_IF(ret == -1, "Error while making a temporary based on '{}'", base);

  name_ = name;
  free(name);

  return ret;
#else
  std::string name(base);
  name += "marian.XXXXXX";
  name.push_back(0);
  int ret;
  ABORT_IF(-1 == (ret = mkstemp_and_unlink(&name[0])),
           "Error while making a temporary based on '{}'",
           base);
  name_ = name;
  return ret;
#endif
}

void TemporaryFile2::NormalizeTempPrefix(std::string &base) {
  if(base.empty())
    return;

#ifdef _MSC_VER
  if(base.substr(0, 4) == "/tmp")
    base = getenv("TMP");
#else
  if(base[base.size() - 1] == '/')
    return;
  struct stat sb;
  // It's fine for it to not exist.
  if(stat(base.c_str(), &sb) == -1)
    return;
  if(S_ISDIR(sb.st_mode))
    base += '/';
#endif
}

TemporaryFile2::TemporaryFile2(const std::string base, bool earlyUnlink) : unlink_(earlyUnlink) {
  std::string baseTemp(base);
  NormalizeTempPrefix(baseTemp);
  fd_ = MakeTemp(baseTemp);
}

TemporaryFile2::~TemporaryFile2() {
#ifdef _MSC_VER
  if(fd_ == -1)
    return;

  if(close(fd_)) {
    std::cerr << "Could not close file " << fd_ << std::endl;
    std::abort();
  }

  if(!unlink_) {
    ABORT_IF(remove(name_.c_str()), "Error while deleting '{}'", name_);
  }
#else
  if(fd_ != -1 && !unlink_) {
    ABORT_IF(unlink(name_.c_str()), "Error while deleting '{}'", name_);
  }
  if(fd_ != -1 && close(fd_)) {
    std::cerr << "Could not close file " << fd_ << std::endl;
    std::abort();
  }
#endif
}

///////////////////////////////////////////////////////////////////////////////////////////////
InputFileStream::InputFileStream(const std::string &file) : file_(file) {
  ABORT_IF(!marian::filesystem::exists(file_), "File '{}' does not exist", file);

  if(file_.extension() == marian::filesystem::Path(".gz"))
    istream_ = std::make_unique<zstr::ifstream>(file_.string());
  else
    istream_ = std::make_unique<std::ifstream>(file_.string());
  ABORT_IF(fail(), "Error {} ({}) opening file '{}'", errno, strerror(errno), path());

  std::cerr << "InputFileStreamOld1 created" << std::endl;
}

InputFileStream::InputFileStream(TemporaryFile2 &tempfile) {
  RewindFile(tempfile.getFileDescriptor());
  temporary_reader_.reset(new ReadFDBuf(tempfile.getFileDescriptor()));
  istream_.reset(new std::istream(temporary_reader_.get()));
  std::cerr << "InputFileStreamOld2 created" << std::endl;
}

InputFileStream::InputFileStream(std::istream &strm) : istream_(new std::istream(strm.rdbuf())) {
  std::cerr << "InputFileStreamOld3 created" << std::endl;
}

void InputFileStream::setbufsize(size_t size) const {
  istream_->rdbuf()->pubsetbuf(0, 0);
  readBuf_.resize(size);
  istream_->rdbuf()->pubsetbuf(readBuf_.data(), readBuf_.size());
}

///////////////////////////////////////////////////////////////////////////////////////////////
OutputFileStream::OutputFileStream(const std::string &file) : file_(file) {
  if(file_.extension() == marian::filesystem::Path(".gz"))
    ostream_ = std::make_unique<zstr::ofstream>(file_.string());
  else
    ostream_ = std::make_unique<std::ofstream>(file_.string());
  ABORT_IF(!marian::filesystem::exists(file_), "File '{}' could not be opened", file);
}

OutputFileStream::OutputFileStream(TemporaryFile2 &tempfile) {
  RewindFile(tempfile.getFileDescriptor());
  temporary_writer_.reset(new WriteFDBuf(tempfile.getFileDescriptor()));
  ostream_.reset(new std::ostream(temporary_writer_.get()));
}

///////////////////////////////////////////////////////////////////////////////////////////////
TemporaryFileNew::TemporaryFileNew(const std::string &base) {
  MakeTemp(base);
}

int TemporaryFileNew::MakeTemp(const std::string &base) {
#ifdef _MSC_VER
  char *name = tempnam(base.c_str(), "marian.");
  ABORT_IF(name == NULL, "Error while making a temporary based on '{}'", base);

  int oflag = _O_RDWR | _O_CREAT | _O_EXCL | _O_TEMPORARY;
  std::fstream::open(name, oflag, _S_IREAD | _S_IWRITE);
  // ABORT_IF(ret == -1, "Error while making a temporary based on '{}'", base);

  name_ = name;
  free(name);
#else
  std::string name(base);
  name += "marian.XXXXXX";
  name.push_back(0);
  int ret;
  ABORT_IF(-1 == (ret = mkstemp_and_unlink(&name[0])),
           "Error while making a temporary based on '{}'",
           base);
  name_ = name;
  return ret;
#endif
}

#ifndef _MSC_VER
int TemporaryFileNew::mkstemp_and_unlink(char *tmpl) {
  int ret = mkstemp(tmpl);
  if(ret != -1) {
    ABORT_IF(unlink(tmpl), "Error while deleting '{}'", tmpl);
  }
  return ret;
}
#endif

///////////////////////////////////////////////////////////////////////////////////////////////
InputFileStreamNew::InputFileStreamNew(const std::string &file)
    : std::istream(NULL), file_(file), streamBuf_(NULL) {
  ABORT_IF(!marian::filesystem::exists(file_), "File '{}' does not exist", file);

  std::filebuf *fileBuf = new std::filebuf();
  streamBuf_ = fileBuf->open(file.c_str(), std::ios::in);
  if(!streamBuf_) {
    ABORT("File can't be read", file);
  }

  if(file_.extension() == marian::filesystem::Path(".gz")) {
    streamBuf_ = new zstr::istreambuf(streamBuf_);
  }

  this->init(streamBuf_);

  std::cerr << "InputFileStreamNew created" << std::endl;
}

InputFileStreamNew::~InputFileStreamNew() {
  delete streamBuf_;
}

///////////////////////////////////////////////////////////////////////////////////////////////
OutputFileStreamNew::OutputFileStreamNew(const std::string &file)
    : zstr::ofstream(file), file_(file) {
  ABORT_IF(!marian::filesystem::exists(file_), "File '{}' does not exist", file);
}

} // namespace io
} // namespace marian
