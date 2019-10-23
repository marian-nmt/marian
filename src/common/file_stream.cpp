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

  std::cerr << "TemporaryFile2 created" << std::endl;
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
TemporaryFileNew::TemporaryFileNew(const std::string &base, bool earlyUnlink)
    : unlink_(earlyUnlink) {
  std::string baseTemp(base);
  NormalizeTempPrefix(baseTemp);
  MakeTemp(baseTemp);

  std::cerr << "TemporaryFileNew created" << std::endl;
}

TemporaryFileNew::~TemporaryFileNew() {
  if(!unlink_) {
    ABORT_IF(remove(name_.c_str()), "Error while deleting '{}'", name_);
  }
}

void TemporaryFileNew::NormalizeTempPrefix(std::string &base) {
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

void TemporaryFileNew::MakeTemp(const std::string &base) {
  char *name = tempnam(base.c_str(), "marian.");
  ABORT_IF(name == NULL, "Error while making a temporary based on '{}'", base);

#ifdef _MSC_VER
  int oflag = _O_RDWR | _O_CREAT | _O_EXCL | _O_TEMPORARY;
  std::fstream::open(name, oflag, _S_IREAD | _S_IWRITE);
  ABORT_IF(errno, "Error while making a temporary based on '{}'", base);
#else
  std::fstream::open(name, std::fstream::in | std::fstream::out);
  ABORT_IF(errno, "Error while making a temporary based on '{}'", base);

  ABORT_IF(remove(name), "Error while deleting '{}'", name);
#endif

  name_ = name;
  free(name);
}


std::string TemporaryFileNew::getFileName() {
  return name_;
}

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

bool InputFileStreamNew::empty() {
  return this->peek() == std::ifstream::traits_type::eof();
}

void InputFileStreamNew::setbufsize(size_t size) const {
  // do nothing. Is this needed?
}

///////////////////////////////////////////////////////////////////////////////////////////////
OutputFileStreamNew::OutputFileStreamNew(const std::string &file)
    : std::ostream(NULL), file_(file), streamBuf1_(NULL), streamBuf2_(NULL) {
  std::filebuf *fileBuf = new std::filebuf();
  streamBuf1_ = fileBuf->open(file.c_str(), std::ios::out | std::ios_base::binary);
  if(!streamBuf1_) {
    std::cerr << "File can't be opened" << file << std::endl;
  }

  if(file_.extension() == marian::filesystem::Path(".gz")) {
    streamBuf2_ = new zstr::ostreambuf(streamBuf1_);
    this->init(streamBuf2_);
  } else {
    this->init(streamBuf1_);
  }

  std::cerr << "OutputFileStreamNew created" << std::endl;
}

OutputFileStreamNew::~OutputFileStreamNew() {
  this->flush();
  delete streamBuf2_;
  delete streamBuf1_;
}

} // namespace io
} // namespace marian
