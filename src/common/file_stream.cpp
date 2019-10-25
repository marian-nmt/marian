#include "common/file_stream.h"

#include <streambuf>
#include <string>
#include <vector>
#ifdef _MSC_VER
#include <io.h>
#include <windows.h>
#else
#include <sys/types.h>
#include <unistd.h>
#endif

namespace marian {
namespace io {

// Get error strings out of errno.
namespace {
#ifdef __GNUC__
const char *HandleStrerror(int ret, const char *buf) __attribute__((unused));
const char *HandleStrerror(const char *ret, const char * /*buf*/) __attribute__((unused));
#endif
// At least one of these functions will not be called.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#endif
// The XOPEN version.
const char *HandleStrerror(int ret, const char *buf) {
  if(!ret)
    return buf;
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
}  // namespace

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

OutputFileStream::OutputFileStream()
    : std::ostream(NULL), streamBuf1_(NULL), streamBuf2_(NULL) {}

OutputFileStream::~OutputFileStream() {
  this->flush();
  delete streamBuf2_;
  delete streamBuf1_;
}

///////////////////////////////////////////////////////////////////////////////////////////////
TemporaryFile::TemporaryFile(const std::string &base, bool earlyUnlink)
    : OutputFileStream(), unlink_(earlyUnlink) {
  std::string baseTemp(base);
  NormalizeTempPrefix(baseTemp);
  MakeTemp(baseTemp);

  inSteam_ = UPtr<io::InputFileStream>(new io::InputFileStream(file_.string()));
  if(unlink_) {
    ABORT_IF(remove(file_.string().c_str()), "Error while deleting '{}'", file_.string());
  }
}

TemporaryFile::~TemporaryFile() {
  if(!unlink_) {
    ABORT_IF(remove(file_.string().c_str()), "Error while deleting '{}'", file_.string());
  }
}

void TemporaryFile::NormalizeTempPrefix(std::string &base) {
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
void TemporaryFile::MakeTemp(const std::string &base) {
#ifdef _MSC_VER
  char *name = tempnam(base.c_str(), "marian.");
  ABORT_IF(name == NULL, "Error while making a temporary based on '{}'", base);

  int oflag = _O_RDWR | _O_CREAT | _O_EXCL;
  if(unlink_)
    oflag |= _O_TEMPORARY;

  int fd = open(name, oflag, _S_IREAD | _S_IWRITE);
  ABORT_IF(fd == -1, "Error while making a temporary based on '{}'", base);

#else
  // create temp file
  std::string name(base);
  name += "marian.XXXXXX";
  name.push_back(0);
  int fd = mkstemp(&name[0]);
  ABORT_IF(fd == -1, "Error creating temp file {}", name);

  file_ = name;
#endif

  // open again with c++
  std::filebuf *fileBuf = new std::filebuf();
  streamBuf1_ = fileBuf->open(name, std::ios::out | std::ios_base::binary);
  ABORT_IF(!streamBuf1_, "File can't be temp opened", name);

  this->init(streamBuf1_);

  // close original file descriptor
  ABORT_IF(close(fd), "Can't close file descriptor", name);

#ifdef _MSC_VER
  free(name);
#endif
}

UPtr<InputFileStream> TemporaryFile::getInputStream() {
  return std::move(inSteam_);
}

std::string TemporaryFile::getFileName() const {
  return file_.string();
}

}  // namespace io
}  // namespace marian
