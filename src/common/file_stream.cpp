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
InputFileStream::InputFileStream(const std::string &file)
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
}

InputFileStream::~InputFileStream() {
  delete streamBuf_;
}

bool InputFileStream::empty() {
  return this->peek() == std::ifstream::traits_type::eof();
}

void InputFileStream::setbufsize(size_t size) const {
  // do nothing. Is this needed?
}

std::string InputFileStream::getFileName() const {
  return file_.string();
}

// wrapper around std::getline() that handles Windows input files with extra CR
// chars at the line end
std::istream &getline(std::istream &in, std::string &line) {
  std::getline(in, line);
  // bad() seems to be correct here. Should not abort on EOF.
  ABORT_IF(in.bad(), "Error reading from stream");
  // strip terminal CR if present
  if(in && !line.empty() && line.back() == in.widen('\r'))
    line.pop_back();
  return in;
}
///////////////////////////////////////////////////////////////////////////////////////////////
OutputFileStream::OutputFileStream(const std::string &file)
    : std::ostream(NULL), file_(file), streamBuf1_(NULL), streamBuf2_(NULL) {
  std::filebuf *fileBuf = new std::filebuf();
  streamBuf1_ = fileBuf->open(file.c_str(), std::ios::out | std::ios_base::binary);
  ABORT_IF(!streamBuf1_, "File can't be opened", file);

  if(file_.extension() == marian::filesystem::Path(".gz")) {
    streamBuf2_ = new zstr::ostreambuf(streamBuf1_);
    this->init(streamBuf2_);
  } else {
    this->init(streamBuf1_);
  }
}

OutputFileStream::~OutputFileStream() {
  this->flush();
  delete streamBuf2_;
  delete streamBuf1_;
}

///////////////////////////////////////////////////////////////////////////////////////////////
TemporaryFile::TemporaryFile(const std::string &base, bool earlyUnlink)
    : OutputFileStream(CreateFileName(base)), unlink_(earlyUnlink) {
  inSteam_ = std::make_unique<io::InputFileStream>(file_.string());
  if (unlink_) {
    ABORT_IF(remove(file_.string().c_str()), "Error while deleting '{}'", file_.string());
  }
}

TemporaryFile::~TemporaryFile() {
  if(!unlink_) {
    ABORT_IF(remove(file_.string().c_str()), "Error while deleting '{}'", file_.string());
  }
}

std::string TemporaryFile::CreateFileName(const std::string &base) const {
  // NormalizeTempPrefix
  std::string ret = base;
  if(!base.empty()) {
#ifdef _MSC_VER
    if(ret.substr(0, 4) == "/tmp") {
      ret = getenv("TMP");
    }
#else
    if(ret[ret.size() - 1] != '/') {
      struct stat sb;
      // It's fine for it to not exist.
      if(stat(ret.c_str(), &sb) != -1) {
        if(S_ISDIR(sb.st_mode)) {
          ret += '/';
        }
      }
    }
#endif
  }

  char *name = tempnam(ret.c_str(), "marian.");
  ABORT_IF(name == NULL, "Error while making a temporary based on '{}'", base);
  ret = name;
  free(name);
  return ret;
}

UPtr<InputFileStream> TemporaryFile::getInputStream() {
  return std::move(inSteam_);
}

std::string TemporaryFile::getFileName() const {
  return file_.string();
}

} // namespace io
} // namespace marian
