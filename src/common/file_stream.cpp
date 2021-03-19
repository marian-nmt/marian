#include "common/file_stream.h"
#include "common/utils.h"

#include <streambuf>
#include <string>
#include <vector>
#include <cstdio>
#ifdef _MSC_VER
#include <io.h>
#include <windows.h>
#include <fcntl.h>
#include <stdlib.h>
#else
#include <sys/types.h>
#include <unistd.h>
#endif

namespace marian {
namespace io {

///////////////////////////////////////////////////////////////////////////////////////////////
InputFileStream::InputFileStream(const std::string &file)
    : std::istream(NULL) {
  // the special syntax "command |" starts command in a sh shell and reads out its result
  if (marian::utils::endsWith(file, "|")) {
#ifdef __unix__
    auto command = file.substr(0, file.size() - 1);
    // open as a pipe
    pipe_ = popen(command.c_str(), "r");
    ABORT_IF(!pipe_, "Command failed to execute ({}): {}", errno, command);
    // there is no official way to construct a filebuf from a FILE* or fd, so we use /proc/{pid}/fd/{fd}
    // For now, this only works on Linux. There are similar workarounds for Windows.
    file_ = "/proc/" + std::to_string(getpid()) + "/fd/" + std::to_string(fileno(pipe_));
#else
    ABORT("Pipe syntax not supported in this build of Marian: {}", file);
#endif
  } else {
    ABORT_IF(!marian::filesystem::exists(file), "File '{}' does not exist", file);
    file_ = file;
  }
  streamBuf1_.reset(new std::filebuf());
  auto ret = static_cast<std::filebuf*>(streamBuf1_.get())->open(file_.string().c_str(), std::ios::in | std::ios::binary);
  ABORT_IF(!ret, "Error opening file ({}): {}", errno, file_.string());
  ABORT_IF(ret != streamBuf1_.get(), "Return value is not equal to streambuf pointer, that is weird");

  // insert .gz decompression
  if(marian::utils::endsWith(file, ".gz")) {
    streamBuf2_ = std::move(streamBuf1_);
    streamBuf1_.reset(new zstr::istreambuf(streamBuf2_.get()));
  }

  // initialize the underlying istream
  this->init(streamBuf1_.get());
}

InputFileStream::~InputFileStream() {
#ifdef __unix__  // (pipe syntax is only supported on UNIX-like OS)
  if (pipe_)
    pclose(pipe_);  // non-NULL if pipe syntax was used
#endif
}

bool InputFileStream::empty() {
  return this->peek() == std::ifstream::traits_type::eof();
}

void InputFileStream::setbufsize(size_t size) {
  rdbuf()->pubsetbuf(0, 0);
  readBuf_.resize(size);
  rdbuf()->pubsetbuf(readBuf_.data(), readBuf_.size());
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
    : std::ostream(NULL), file_(file) {
  streamBuf1_.reset(new std::filebuf());
  auto ret = static_cast<std::filebuf*>(streamBuf1_.get())->open(file.c_str(), std::ios::out | std::ios_base::binary);
  ABORT_IF(!ret, "File cannot be opened", file);
  ABORT_IF(ret != streamBuf1_.get(), "Return value is not equal to streambuf pointer, that is weird");

  if(file_.extension() == marian::filesystem::Path(".gz")) {
    streamBuf2_.reset(new zstr::ostreambuf(streamBuf1_.get()));
    this->init(streamBuf2_.get());
  } else {
    this->init(streamBuf1_.get());
  }
}

OutputFileStream::OutputFileStream()
    : std::ostream(NULL) {}

OutputFileStream::~OutputFileStream() {
  this->flush();
}

std::string OutputFileStream::getFileName() const {
  return file_.string();
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
  if(!unlink_)
    // We do not check for errors here as this is the destructor and we cannot really fix an error anyway.
    remove(file_.string().c_str()), "Error while deleting '{}'", file_.string();
}

void TemporaryFile::NormalizeTempPrefix(std::string &base) const {
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

  file_ = std::string(name);
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
  streamBuf1_.reset(new std::filebuf());
  auto ret = static_cast<std::filebuf*>(streamBuf1_.get())->open(name, std::ios::out | std::ios_base::binary);
  ABORT_IF(!streamBuf1_, "File {} cannot be temp opened", name);
  ABORT_IF(ret != streamBuf1_.get(), "Return value ({}) is not equal to streambuf pointer ({}), that is weird.", (size_t)ret, (size_t)streamBuf1_.get());

  this->init(streamBuf1_.get());

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
