#include "../include/temp.hpp"
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <ctime>

#if defined(_PATHIE_UNIX)
#include <sys/types.h>
#include <unistd.h>
#elif defined(_WIN32)
#include <Windows.h>
#else
#error Unsupported system
#endif

using namespace Pathie;

static std::string generate_random_filename(const std::string& namepart)
{
  std::stringstream name;
  name << namepart << "-" << time(NULL) << rand();

#if defined(_PATHIE_UNIX)
  name << getpid();
#elif defined(_WIN32)
  name << GetCurrentProcessId();
#else
#error Unsupported system
#endif

  return name.str();
}

/**
 * TempEntry is not meant to be instanciated on itself. This
 * constructor does the common work between the Tempdir and Tempfile
 * classes, namely it generates a temporary filename that is not
 * currently in use.
 *
 * \param namepart
 * A string that will be included verbatim into the basename
 * of the created directory.
 *
 * \remark The generated path name is of form
 * `<namepart>-<currenttime><random><pid>`. However, future releases
 * may change this format, so do not rely on it.
 */
TempEntry::TempEntry(std::string namepart)
  : m_keep(false)
{
  do {
    m_path = Path::temp_dir() / generate_random_filename(namepart);
  } while (m_path.exists());
}

/**
 * Destructor.
 */
TempEntry::~TempEntry()
{
  //
}

/**
 * Returns the absolute path to the temporary entry
 * that was created by the constructor.
 */
Path TempEntry::path() const
{
  return m_path;
}

/**
 * Call this function if you do not want the destructor to delete
 * the created temporary entry. You can still expressly delete
 * the temporary entry by calling remove().
 *
 * \param k
 * If true (default), the destructor will not delete the temporary entry.
 * If false, the destructor will delete the temporary entry.
 */
void TempEntry::keep(bool k)
{
  m_keep = k;
}

/**
 * Returns the keep status; see keep().
 */
bool TempEntry::is_kept() const
{
  return m_keep;
}

/**
 * Constructs an instance of this class. A temporary directory
 * is created that will be recursively removed when the object
 * is deleted.
 *
 * \param namepart
 * A string that will be included verbatim into the basename
 * of the created directory.
 *
 * \returns The newly created instance.
 *
 * \remark There is a small timespan between the generation of the
 * temporary path name and the creation of the directory in which it
 * is theoretically possible for another process to create an entry
 * that conflicts with the generated name. However, since the
 * generated name includes a random number, the process identifier,
 * and the number of seconds since epoch as well as the given
 * `namepart`, the chance of an accidental collision is very low.
 * Even a malicious attacker would have to guess the random number, so
 * if your `srand()` seed is chosen properly and your C standard
 * library is properly impelemented, this risk is again very low.
 */
Tempdir::Tempdir(std::string namepart)
  : TempEntry(namepart)
{
  m_path.mktree();
}

/**
 * Destructor, removes the temporary entry unless keep() has been called.
 * Does nothing if the temporary file does not exist anymore for whatever
 * reason.
 */
Tempdir::~Tempdir()
{
  if (!m_keep)
    remove();
}

/**
 * Recursively removes the temporary directory. This method
 * ignores what was set with keep(), i.e., it *always* deletes
 * the temporary directory if you call it. This method does
 * nothing if the directory does not exist anymore for whatever
 * reason.
 */
void Tempdir::remove() const
{
  if (m_path.exists())
    m_path.rmtree();
}

/**
 * Constructs an instance of this class. A temporary file
 * is created that will be recursively removed when the object
 * is deleted.
 *
 * \param namepart
 * A string that will be included verbatim into the basename
 * of the created filename.
 *
 * \returns The newly created instance.
 *
 * \remark There is a small timespan between the generation of the
 * temporary path name and the creation of the file in which it
 * is theoretically possible for another process to create an entry
 * that conflicts with the generated name. However, since the
 * generated name includes a random number, the process identifier,
 * and the number of seconds since epoch as well as the given
 * `namepart`, the chance of an accidental collision is very low.
 * Even a malicious attacker would have to guess the random number, so
 * if your `srand()` seed is chosen properly and your C standard
 * library is properly impelemented, this risk is again very low.
 */
Tempfile::Tempfile(std::string namepart)
  : TempEntry(namepart)
{
  m_path.touch();
}

/**
 * Destructor, removes the temporary file unless keep() has been called.
 * Does nothing if the temporary directory does not exist anymore for whatever
 * reason.
 */
Tempfile::~Tempfile()
{
  if (!m_keep)
    remove();
}

/**
 * Removes the temporary file. This method
 * ignores what was set with keep(), i.e., it *always* deletes
 * the temporary file if you call it. This method does nothing
 * if the file does not exist anymore for whatever
 * reason.
 */
void Tempfile::remove() const
{
  if (m_path.exists())
    m_path.unlink();
}
