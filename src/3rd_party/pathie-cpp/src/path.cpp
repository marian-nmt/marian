/* -*- coding: utf-8 -*-
 * This file is part of Pathie.
 *
 * Copyright © 2015, 2017 Marvin Gülker
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "../include/path.hpp"
#include "../include/pathie.hpp"
#include "../include/errors.hpp"

#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdexcept>
#include <errno.h>

#if defined(_WIN32)
#include <windows.h>
#include <winioctl.h>
#include <direct.h>
#include <shlobj.h>
#include <shlwapi.h>
//#include <ntifs.h> // Currently not in msys2

// @TODO: This is a hack to make it compile under Windows, check if this is safe.
#define F_OK    0

#elif defined(_PATHIE_UNIX)
#include <unistd.h>
#include <limits.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/param.h> // defines "BSD" macro on BSD systems
#include <pwd.h>
#include <glob.h>
#include <fnmatch.h>

#else
#error Unsupported system.
#endif

#ifdef BSD
#include <sys/time.h>
#include <sys/sysctl.h>
#endif

using namespace Pathie;
using namespace std;

Path::localpathtype Path::c_localdefault = LOCALPATH_LOCAL;

/**
 * The default constructor. It does **not** create an empty
 * path, but a path whose value is ".", i.e. the current
 * working directory as a relative path (see also pwd()).
 */
Path::Path()
{
  m_path = ".";
}

/**
 * Copies contents from path to a new instance.
 *
 * \param[in] path The Path instance to copy.
 */
Path::Path(const Path& path)
{
  m_path = path.m_path;
}

/**
 * This constructs a path from a given std::string.
 *
 * \param path String to construct from. Must be encoded in UTF-8.
 *
 * \returns a new instance of class Path.
 */
Path::Path(std::string path)
{
  m_path = path;
  sanitize();
}

/**
 * Constructs a Path instance from a list of path components.
 * This is the inverse of the burst() method.
 *
 * \param[in] components List of components to join.
 *
 * \returns A new instance.
 */
Path::Path(const std::vector<Path>& components)
{
  m_path = components.front().m_path;

  if (components.size() > 1) {
    // Ensure that for both absolute and relative path we end in
    // a slash for appending below
    if (m_path[0] != '/') {
      m_path += "/";
    }

    std::vector<Path>::const_iterator iter;
    for(iter=components.begin()+1; iter != components.end(); iter++) { // first element has already been taken care of above
      m_path += (*iter).m_path + "/";
    }

    // Trailing slash is unwanted, remove it
    m_path = m_path.substr(0, m_path.length()-1);
  }
}

/**
 * Sanitizes the path. It:
 *
 * 1. Replaces any backslashes with forward slashes (read Windows).
 * 2. Replaces all double forward slashes with single forward slashes
 * 3. Delates a trailing slash, if any.
 */
void Path::sanitize()
{
  bool isWindowsUNCPath = m_path.size() >= 2 && (m_path[0] == '\\' && m_path[1] == '\\'); // UNC path

  // Replace any backslashes \ with forward slashes /.
  size_t cur = string::npos;
  while ((cur = m_path.find("\\")) != string::npos) { // assignment intended
    m_path.replace(cur, 1, "/");
  }

  // Replace all double slashes // with a single one
  // [fseide] except for the first position, which would be a Windows UNC path
  cur = string::npos;
  while ((cur = m_path.find("//", isWindowsUNCPath ? 1 : 0)) != string::npos) { // assignment intended
    m_path.replace(cur, 2, "/");
  }

  // Remove trailing slash if any (except for the filesystem root)
  long len = (long)m_path.length();
#if defined(_PATHIE_UNIX)
  if (len > 1 && m_path[len - 1] == '/')
    m_path = m_path.substr(0, len - 1);
#elif defined(_WIN32)
  if (len > 1) { // / is root of current drive, "x" is the relative path "./x"
    // Check if X:/foo/bar
    if (len > 3 && m_path[len - 1] == '/') { // More than 3 chars cannot be root
      m_path = m_path.substr(0, len - 1);
    }
    else { // Only drive root?
      if (m_path[1] == ':') {
        // Here m_path must be a drive root. The colon ":" is not allowed in paths on Windows except as the 2nd char to denote the drive letter
        if (len == 2) { // Whoa -- "X:" misses leading / for drive root, append it
          m_path.append("/");
        }
        else if (len == 3 && m_path[2] != '/') { // Whoa -- "X:f" misses leading / for root directory, insert it
          m_path.insert(2, "/");
        }
        // else length is 3 with a slash, i.e. "X:/". This is fine and shall not be touched.
      }
      else { // not a drive root, delete trailing / if any
        if (m_path[len - 1] == '/') {
          m_path = m_path.substr(0, len - 1);
        }
      }
    }
  }
#else
#error Unsupported system
#endif
}

/** \name Conversion methods
 *
 * Convert a path to other objects.
 */
///@{

/**
 * Returns a copy of the underlying `std::string`. This is always
 * encoded in UTF-8, regardless of the operating system.
 *
 * \see native() utf8_str()
 */
std::string Path::str() const
{
  return m_path;
}

/**
 * This method does the same as str(). It exists to make code using
 * the UTF-8 variant more readable, because one tends to forget
 * whether str() returns the native or the UTF-8 variant.
 *
 * \see native() str()
 */
std::string Path::utf8_str() const
{
  return m_path;
}

#if defined(_PATHIE_UNIX)
std::string Path::native() const
{
  return utf8_to_filename(m_path);
}

#elif defined(_WIN32)
/**
 * Returns the path in the platform’s native format. Note
 * that this method returns a `std::string` on UNIX,
 * whereas it returns a `std::wstring` on Windows.
 *
 * On Windows, the returned string also uses exclusively backslashes
 * instead of forward slashes. It is encoded in UTF-16LE.
 *
 * On UNIX, the returned string is in the encoding dictated by the locale
 * ($LANG and $LC_ALL variables).
 */
std::wstring Path::native() const
{
  std::string dup(m_path);

  size_t pos = 0;
  while((pos = dup.find("/", pos)) != std::string::npos) { // Single = intended
    dup.replace(pos, 1, "\\");
  }

  return utf8_to_utf16(dup);
}
#else
#error Unsupported system.
#endif

///@}


/** \name Path decomposition
 *
 * Retrieve the parts of the path you want.
 */
///@{

/**
 * Returns the path’s basename, i.e. the last component
 * of the path, including the file excention.
 *
 * For example, "/foo/bar.txt" has a basename of "bar.txt",
 * and "/foo/bar" has a basename of "bar".
 *
 * \returns a new Path instance with only the basename.
 *
 * \see dirname()
 */
Path Path::basename() const
{
  if (m_path == ".")
    return Path(".");
  else if (m_path == "..")
    return Path("..");
  else if (is_root())
    return Path(m_path);

  size_t pos = 0;
  if ((pos = m_path.rfind("/")) != string::npos) // Single = intended
    return Path(m_path.substr(pos + 1));
  else
    return Path(m_path);
}

/**
 * Returns the path’s dirname, i.e. all components of the
 * path except for the basename component (see basename()).
 *
 * For example, "/foo/bar/baz.txt" has a dirname of "/foo/bar",
 * and "/foo/bar/baz" has a dirname of "/foo/bar".
 *
 * \returns a new Path instance with only the dirname.
 *
 * \see basename() parent()
 */
Path Path::dirname() const
{
  if (m_path == ".")
    return Path(".");
  else if (m_path == "..")
    return Path(".");
  else if (is_root())
    return Path(m_path);

  size_t pos = 0;
  if ((pos = m_path.rfind("/")) != string::npos) { // Single = intended
    if (pos == 0) { // /usr
      return root();
    }
#ifdef _WIN32
    else if (pos == 1 && m_path[1] == ':') { // X:/foo
      return root();
    }
#endif
    else { // regular/path or /regular/path
      return Path(m_path.substr(0, pos));
    }
  }
  else // single relative directory
    return Path(".");
}

/**
 * This is a convenience method that allows you to retrieve
 * both the dirname() and the basename() in one call.
 *
 * \param[out] dname Receives the dirname() value.
 * \param[out] bname Receives the basename() value.
 */
void Path::split(Path& dname, Path& bname) const
{
  dname = dirname();
  bname = basename();
}

/**
 * This method returns the file extension of the path,
 * if possible; otherwise it returns an empty string.
 * Filenames that consist entirely of a "file extension",
 * i.e. ".txt" or "/foo/.txt" will return an empty string.
 */
std::string Path::extension() const
{
  if (m_path == ".")
    return "";
  else if (m_path == "..")
    return "";

  size_t pos = 0;
  if ((pos = m_path.rfind(".")) != string::npos) { // assignment intended
    if (pos == 0 || pos == m_path.length() - 1) // .foo and foo.
      return "";
    else {
      if (m_path[pos - 1] == '/') // foo/.txt
	return "";
      else
	return m_path.substr(pos);
    }
  }
  else
    return "";
}

/**
 * This is the same as dirname() and is provided only for convenience.
 *
 * \see dirname()
 */
Path Path::parent() const
{
  return dirname();
}

/**
 * Returns the number of components in the path string, or
 * in different words, counts the slashes and adds one for
 * the last element, except if the path is just the root
 * (see is_root()).
 *
 * The return value of this method minus one is the last
 * possible index for operator[].
 */
size_t Path::component_count() const
{
  if (is_root())
    return 1;

  size_t result = 0;
  size_t pos = 0;
  while ((pos = m_path.find("/", pos)) != string::npos) { // Assignment intended
    result++;
    pos++;
  }

  return ++result;
}

/**
 * Returns the filesystem root for this path. On UNIX,
 * this will always return /, but on Windows it will
 * return X:/ if the referenced path is an absolute path
 * with drive letter, and / if the referenced path is
 * a relative path or an absolute path on the current
 * drive.
 */
Path Path::root() const
{
#if defined(_PATHIE_UNIX)
  return Path("/");
#elif defined(_WIN32)
  // Check if we have an absolute path with drive,
  // otherwise return the root for the current drive.
  if (m_path[1] == ':') // Colon is on Windows only allowed here to denote a preceeding drive letter => absolute path
    return Path(m_path.substr(0, 3));
  else
    return Path("/");
#else
#error Unsupported system.
#endif
}

/**
 * This method splits up the paths into its separate components,
 * i.e. it splits it up at every /, except for the leading / of
 * an absolute path, which is considered a component on its own
 * and is thus the first element of a bursted absolute path.
 *
 * \param descend (`false`) If this is true, keeps the parent paths when bursting.
 *
 * \returns A vector of Path instances, where each instance
 * corresponds to one component of the Path.
 *
 * Example:
 *
 * ~~~~~~~~~~~~~~~~~~~~ c++
 * Path p("/tmp/foo/bar");
 * p.burst(); // => /, tmp, foo, bar
 * p.burst(true); // => /, /tmp, /tmp/foo, /tmp/foo/bar
 * ~~~~~~~~~~~~~~~~~~~~
 */
std::vector<Path> Path::burst(bool descend /* = false */) const
{
  size_t pos = 0;
  size_t lastpos = 0;
  std::vector<Path> results;
  std::string prefix;

  // Take care of leading / of absolute paths
  if (m_path[0] == '/') {
    results.push_back(Path("/"));
    prefix.append("/");

    // Adjust pos so we don’t find the initial /
    pos++;
    lastpos++;
  }

  while((pos = m_path.find("/", pos)) != string::npos) {
    std::string component = m_path.substr(lastpos, pos - lastpos);

    if (descend) {
      results.push_back(Path(prefix + component));
      prefix.append(component);
      prefix.append("/");
    }
    else {
      results.push_back(Path(component));
    }

    lastpos = pos + 1;
    pos++;
  }

  std::string lastcomponent = m_path.substr(lastpos);

  if (descend)
    results.push_back(Path(prefix + lastcomponent)); // Note no trailing /
  else
    results.push_back(Path(lastcomponent));

  return results;
}

///@}

/** \name Path expansion
 *
 * Expand paths to a more fuller version without shortcuts.
 */

///@{

/**
 * This method, removes all occurences of . and .. from the path,
 * leaving a clean filesystem path.
 *
 * Note that neither an absolute path is created, nor
 * are shortcuts other than . and .. expanded.
 *
 * This method does not access file filesystem, and thus does not
 * know about symbolic links. Therefore, if the path contains symlinks,
 * the result may not be the way you expect it. Use real() if
 * you need to resolve all your symbolic links in the path.
 *
 * For example, if you have a directory `/tmp/foo`, which contains a
 * symbolic link `bar` that points to `/tmp/bar`, then a path of
 * `/tmp/foo/bar/..` will be prune()d to `/tmp/foo`, although the
 * canonically correct result is `/tmp`. The latter is what you will
 * get if you use real().
 *
 * \returns A new string with . and .. removed.
 *
 * \see expand() real()
 */
Path Path::prune() const
{
  std::string newpath(m_path); // copy
  size_t pos = 0;
  while((pos = newpath.find("/.", pos)) != string::npos) { // assignment intended
    if (newpath.substr(pos, 3) == "/..") {

      // Weird path like /..foo or foo/..bar, which are NOT relative paths
      if (newpath.length() > pos + 3 && newpath[pos + 3] != '/') {
        // Do not reset `pos' -- this has to stay. Advance to the next char.
        pos++;
        continue;
      }

      if (pos == 0) {
        // /.. at beginning of string, replace with root / (/ on Windows is root on current drive)
        newpath.erase(pos, 3);

        // Whoops -- the entire string was just "/.."
        if (newpath.empty()) {
          newpath.append("/");
        }
      }
#ifdef _WIN32
      // Cater for paths with drive X:/ on Windows
      else if (pos == 2 && newpath[1] == ':') { // ":" is on Windows only allowed at pos 1, where it signifies the preceding char is a drive letter
        // X:/. or X:/.. at beginning of string
        if(newpath.length() > 4 && newpath[4] == '.') { // X:/..
          // Prevent special case "X:/..foo", which is directory "..foo" under the root
          if (newpath.length() <= 5 || newpath[5] != '/') {
            // X:/.. or X:/../foo/bar at beginning of string, replace with drive root
            newpath.erase(pos, 3);
          }
        }
        else { // X:/./foo/bar X:/..foo
          // Prevent special case "X:/.foo", which is directory ".foo" under the root
          if (newpath.length() <= 4 || newpath[4] != '/') {
            // X:/. or X:/./foo/bar at beginning of string, replace with drive root
            newpath.erase(pos, 2);
          }
        }

        if (newpath.length() == 2) {
          // Whoops -- the entire string was just "X:/.." or "X:/."
          newpath.append("/");
        }
      }
#endif
      else {
        size_t pos2 = 0;
        if ((pos2 = newpath.rfind("/", pos - 1)) != string::npos) { // assignment intended
          // Remove parent directory.
          newpath.erase(pos2, pos - pos2 + 3);
        }
        else { // ../ for relative path (as in foo/../baz.txt)
          newpath.erase(0, pos + 4);
        }
      }
    }
    else { // Single /.

      // Weird path like /..foo or foo/..bar, which are NOT relative paths
      if (newpath.length() > pos + 2 && newpath[pos + 2] != '/') {
        // Do not reset `pos' -- this has to stay. Advance to the next char.
        pos++;
        continue;
      }

      newpath.erase(pos, 2);

      // Whoops -- the entire string was just "/."
      if (newpath.empty()) {
        newpath.append("/");
      }
    }

    // Reset as we have modified the string and might need to go again over it
    pos = 0;
  }

  /* If we are empty now, the original string was a one-element
   * relative path with .. appended. We cannot know what to set
   * without referring to pwd(), which is external access and
   * forbidden for this method. So instead, we do the one sane thing
   * and just use ".". */
  if (newpath.empty())
    newpath = ".";

  return Path(newpath);
}

/**
 * \note Under specific circumstances (see below), this method
 * accesses the file system.
 *
 * This method creates an absolute path by use of prune(), but
 * additionally expands any expandable strings. If one of the
 * following substitution sequences are encountered, it will be
 * replaced accordingly.
 *
 * "~" is expanded to the user’s home directory, see home().
 *
 * \returns a new instance with everything expanded.
 *
 * \remark This method uses prune() to expand ".." entries, therefore
 * it will not consider symbolic links when resolving those. Use
 * real() if you need to do that.
 *
 * \see prune() real()
 */
Path Path::expand() const
{
  Path path(*this); // copy

  if (m_path[0] != '~')
    path = path.absolute();

  std::string str = path.str();
  if (str[0] == '~') {
    Path homepath = home();

    if (str[1] == '/' || str.length() == 1) {
      // User home requested
      str.replace(0, 1, homepath.m_path);
    }

    path = Path(str);
  }

  return path.prune();
}

/**
 * \note This method acceses the filesystem.
 *
 * This is the bruteforce method for determing the real path
 * of the entry in question on the filesystem. It looks on
 * each single component of the path, checks if it is a
 * symbolic link, and if so, resolves it.
 *
 * This method supports symbolic link resolving only on UNIX.
 *
 * It still does not consider hardlinks, mountpoints, and junctions,
 * though. However, a hardlink is a real second valid name for an
 * object; in contrast to a symbolic link, if one hardlink gets
 * removed, the other one stays still valid. If you remove the file a
 * symbolic link points to, the link breaks. Thus, it is not even
 * possible to determine which of two hardlinks to a file is the
 * "primary" one. Mountpoints and junctions (junctions are on Windows
 * what mountpoints are on UNIX) behave similar with respect to
 * entire directory hierarchies.
 *
 * \see expand() prune()
 */
Path Path::real() const
{
#if defined(_PATHIE_UNIX)
  std::string nstr = native();
  char path[PATH_MAX];
  if (!realpath(nstr.c_str(), path))
    throw(Pathie::ErrnoError(errno));

  return Path(filename_to_utf8(path));
#elif defined(_WIN32)
  // On Windows there sadly is no easy way to do this. We can
  // only determine if a given path is a symlink and resolve it...
  // Instructions taken from: http://msdn.microsoft.com/en-us/library/windows/desktop/aa363940%28v=vs.85%29.aspx
  std::vector<Path> components = burst();
  unsigned int pos = 0;

  while (pos < components.size()) {
    // Build path consisting of all elements upto our position pointer
    Path reduced_path(components.front());
    if (components.size() - pos > 1) {
      for (unsigned int i=1; i <= pos; i++) { // i=0 is already in the initialization above
        reduced_path = reduced_path.join(components[i]);
      }
    }

    // If that’s a symlink, resolve it and replace our path until
    // the symlink with the symlink’s target.
    /*std::wstring reduced_path_utf16 = utf8_to_utf16(reduced_path.m_path);
    if (is_ntfs_symlink(reduced_path_utf16.c_str())) {
      wchar_t* target_utf16 = read_ntfs_symlink(reduced_path_utf16.c_str());
      Path target(utf16_to_utf8(target_utf16));
      std::vector<Path> target_components = target.burst();
      free(target_utf16);

      // Replace all components up to pos with the symlink target
      components.erase(components.begin(), components.begin() + pos);
      std::vector<Path> temp(components);
      components.clear();
      for(auto iter=target_components.begin(); iter != target_components.end(); iter++)
        components.push_back(*iter);
      for(auto iter=temp.begin(); iter != temp.end(); iter++)
        components.push_back(*iter);
    }
    else {*/
      // Note a symlink can point to another symlink, so we can only
      // advance to the next element if this element has been tested
      // for not being a symlink.
      pos++;
      //}
  }

  // BUild a new path from the now resolved components
  Path result(components.front());
  if (components.size() > 1) {
    for(std::vector<Path>::const_iterator iter=components.begin();
    		iter != components.end(); iter++) {
      result = result.join(*iter);
    }
  }

  return result;
#else
#error Unsupported system.
#endif
}

// Msys2 does currently not have ntifs.h windows header, which
// is required for reading NTFS symlinks.
#if 0
//#ifdef __WIN32
/*
 * Checking if a file is a symlink under Windows is insane.
 * See http://msdn.microsoft.com/en-us/library/windows/desktop/aa363940%28v=vs.85%29.aspx
 * for the detailed instructions by Microsoft on how to do
 * that.
 */
bool Path::is_ntfs_symlink(const wchar_t* path) const
{
  // First we need to obtain the file attributes.
  DWORD attrs = GetFileAttributesW(path);
  if (attrs == INVALID_FILE_ATTRIBUTES) {
    DWORD err = GetLastError();
    throw(Pathie::WindowsError(err));
  }

  /* These file attributes must contain the REPARSE_POINT attribute
   * that mark the file as being symlink, junction, or similar.
   * Actually, reparse points can contain many more custom data, but
   * we are not intersted in those. */
  if (attrs & FILE_ATTRIBUTE_REPARSE_POINT) {
    // Now we have to retrieve a special attributes handle from the file.
    WIN32_FIND_DATAW finddata;
    HANDLE findhandle = FindFirstFileW(path, &finddata);
    if (findhandle == INVALID_HANDLE_VALUE) {
      DWORD err = GetLastError();
      throw(Pathie::WindowsError(err));
    }
    FindClose(findhandle);

    // These extended attributes contain the SYMLINK tag if this file
    // is a symlink.
    if (finddata.dwReserved0 & IO_REPARSE_TAG_SYMLINK)
      return true;

    // Junction or so, we do not resolve that
    return false;
  }

  // Regular file
  return false;
}

/*
 * Reading the link target also is insanely hard.
 * The process is documented at http://msdn.microsoft.com/en-us/library/windows/desktop/aa365503%28v=vs.85%29.aspx
 * in general. The key function is DeviceIoControl(), documented
 * at http://msdn.microsoft.com/en-us/library/windows/desktop/aa363216%28v=vs.85%29.aspx
 * .
 *
 * This function does not check if `path` is a symlink, but assumes it.
 * It will exhibit unexpactable behaviour if this assumption is wrong.
 *
 * The returned pointer must be freed by you.
 */
wchar_t* Path::read_ntfs_symlink(const wchar_t* path) const
{
  // We have to open the file (directories are files on Windows also) first.
  HANDLE filehandle = CreateFileW(path, GENERIC_READ, 0, NULL, OPEN_EXISTING, FILE_FLAG_OPEN_REPARSE_POINT, NULL);
  if (filehandle == INVALID_HANDLE_VALUE) {
    DWORD err = GetLastError();
    throw(Pathie::WindowsError(err));
  }

  // This infamous structure is documented here: http://msdn.microsoft.com/en-us/library/ff552012.aspx
  unsigned long reparsebufsize = REPARSE_GUID_DATA_BUFFER_HEADER_SIZE; // According to docs this is the minimum size
  REPARSE_DATA_BUFFER* p_reparse_data = NULL;
  while (true) {
    reparsebufsize += 4096; // Do you have a better guess?
    p_reparse_data = (REPARSE_DATA_BUFFER*) realloc(p_reparse_data, reparsebufsize);
    memset(p_reparse_data, '\0', reparsebufsize);

    DWORD bytecount = 0;
    // Obtain the reparse tag. FSCTL_GET_REPARSE_POINT is documented here: http://msdn.microsoft.com/en-us/library/windows/desktop/aa364571(v=vs.85).aspx
    if (DeviceIoControl(filehandle, FSCTL_GET_REPARSE_POINT, NULL, 0, p_reparse_data, reparsebufsize, &bytecount, NULL) == 0) {
      DWORD errsav = GetLastError();
      if (errsav == ERROR_INSUFFICIENT_BUFFER) { // buffer was to small, try again
        continue;
      }
      else {
        throw(Pathie::WindowsError(errsav));
      }
    }
    else { // success
      break;
    }
  }

  // See also http://msdn.microsoft.com/en-us/library/windows/desktop/aa365511(v=vs.85).aspx
  // And this one: http://www.codeproject.com/Articles/21202/Reparse-Points-in-Vista
  if (p_reparse_data->ReparseTag == IO_REPARSE_TAG_SYMLINK) {
    wchar_t* subsname  = (wchar_t*) malloc(p_reparse_data->SymbolicLinkReparseBuffer.SubstituteNameLength + 2); // UTF-16 NUL
    wchar_t* printname = (wchar_t*) malloc(p_reparse_data->SymbolicLinkReparseBuffer.PrintNameLength + 2); // UTF-16 NUL

    memset(subsname,  '\0', p_reparse_data->SymbolicLinkReparseBuffer.SubstituteNameLength + 2);
    memset(printname, '\0', p_reparse_data->SymbolicLinkReparseBuffer.PrintNameLength + 2);

    wcsncpy(subsname,  &p_reparse_data->SymbolicLinkReparseBuffer.PathBuffer[p_reparse_data->SymbolicLinkReparseBuffer.SubstituteNameOffset], p_reparse_data->SymbolicLinkReparseBuffer.SubstituteNameLength / sizeof(WCHAR));
    wcsncpy(printname, &p_reparse_data->SymbolicLinkReparseBuffer.PathBuffer[p_reparse_data->SymbolicLinkReparseBuffer.PrintNameOffset], p_reparse_data->SymbolicLinkReparseBuffer.PrintNameLength / sizeof(WCHAR));

    // Actually, it appears the subsname has no real usecase...
    free(subsname);
    free(p_reparse_data);
    CloseHandle(filehandle);
    return printname;
  }
  else {
    return NULL;
  }
}
#endif

///@}

/** \name Special files and directories
 *
 * Files and directories with a special meaning that did not
 * fit in the other groups.
 */
///@{

/**
 * Determines the current process working directory and returns
 * it as an absolute path. Contains a leading drive letter on
 * Windows.
 */
Path Path::pwd()
{
#if defined(_PATHIE_UNIX)
  char cwd[PATH_MAX];
  if (getcwd(cwd, PATH_MAX) != NULL)
    return Path(filename_to_utf8(cwd));
  else
    throw(std::runtime_error("Failed to retrieve current working directory."));
#elif defined(_WIN32)
  wchar_t cwd[MAX_PATH];
  if (GetCurrentDirectoryW(MAX_PATH, cwd) == 0)
    throw(std::runtime_error("Failed to retrieve current working directory."));
  else
    return Path(utf16_to_utf8(std::wstring(cwd)));
#else
#error Unsupported platform.
#endif
}

/**
 * \note On Linux, this method accesses the `/proc` filesystem.
 *
 * This method returns the full absolute path to the currently running
 * executable.
 */
Path Path::exe()
{
#if defined(__linux__) || defined(__APPLE__)
  char buf[PATH_MAX];
  ssize_t size = ::readlink("/proc/self/exe", buf, PATH_MAX);

  if (size < 0)
    throw(Pathie::ErrnoError(errno));

  return Path(filename_to_utf8(std::string(buf, size)));
#elif defined(BSD)
  // BSD does not have /proc mounted by default. However, using raw syscalls,
  // we can figure out what would have been in /proc/curproc/file. See
  // sysctl(3) for the management info base identifiers that are used here.
  int mib[4];
  char buf[PATH_MAX];
  size_t bufsize = PATH_MAX;
  mib[0] = CTL_KERN;
  mib[1] = KERN_PROC;
  mib[2] = KERN_PROC_PATHNAME;
  mib[3] = -1; // According to sysctl(3), -1 means the current process.

  if (sysctl(mib, 4, buf, &bufsize, NULL, 0) != 0) // Note this changes `bufsize' to the number of chars copied
    throw(Pathie::ErrnoError(errno));

  return Path(filename_to_utf8(std::string(buf, bufsize - 1))); // Exclude terminating NUL
#elif defined(_WIN32)
  wchar_t buf[MAX_PATH];
  if (GetModuleFileNameW(NULL, buf, MAX_PATH) == 0) {
    DWORD err = GetLastError();
    throw(Pathie::WindowsError(err));
  }

  std::string str = utf16_to_utf8(buf);
  return Path(str);
#else
#error Unsupported platform.
#endif
}

/**
 * This method returns the current user’s home directory. On UNIX
 * systems, the $HOME environment variable is consulted, whereas
 * on Windows the Windows API is queried for the directory.
 *
 * It will throw std::runtime_error if $HOME is not defined on
 * UNIX.
 */
Path Path::home()
{
#if defined(_PATHIE_UNIX)
  char* homedir = getenv("HOME");
  if (homedir)
    return Path(filename_to_utf8(homedir));
  else
    throw(std::runtime_error("$HOME not defined."));
#elif defined(_WIN32)
  /* TODO: Switch to KNOWNFOLDERID system as explained
   * on http://msdn.microsoft.com/en-us/library/windows/desktop/bb762494%28v=vs.85%29.aspx
   * and http://msdn.microsoft.com/en-us/library/windows/desktop/bb762181%28v=vs.85%29.aspx
   *. Howevever, MinGW does currently (September 2014) not have
   * the new KNOWNFOLDERID declarations.
   */

  wchar_t homedir[MAX_PATH];
  if (SHGetFolderPathW(NULL, CSIDL_PROFILE, NULL, SHGFP_TYPE_CURRENT, homedir) != S_OK)
    throw(std::runtime_error("Home directory not defined."));

  return Path(utf16_to_utf8(homedir));
#else
#error Unsupported system.
#endif
}

///@}

/** \name Handling of absolute and relative paths
 *
 * Converting relative paths to absolute ones and vice-versa.
 */
///@{

/**
 * Builds an absolute path from the referenced path by
 * prefixing it with a `base` path, which defaults to
 * the current working directory. If the referenced path
 * is absolute already, nothing is done and a copy of the
 * referenced path is returned.
 *
 * \param[in] base Base path. Default is the return value of Path::pwd().
 *
 * \returns A new instance that is absolute.
 *
 * \see relative()
 */
Path Path::absolute(const Path& base /* = Path::pwd() */) const
{
  if (is_absolute())
    return Path(m_path);
  else
    return base.join(m_path);
}

/**
 * The referenced path has to to be absolute; by doing pure string
 * manipulation (read: no symlinks), it will then be determined how to
 * go from the (also absolute) `base` path to the referenced path. The
 * result is a relative path, which will be returned by this method.
 *
 * On Windows, this method will throw an std::invalid_argument if the `base`
 * is on a different drive than the referenced path. If either the referenced
 * or the passed path is relative, std::invalid_argument will also be thrown.
 *
 * \param base Base path from which to start. Must also be absolute.
 *
 * \returns A new instance as a relative path.
 *
 * Example:
 *
 * ~~~~~~~~~~~~~~~~~~~~ c++
 * Path p1("/tmp/foo/bar/baz");
 * Path p2("/tmp/xxx/yyy");
 *
 * p1.relative(p2); // => ../../foo/bar/baz
 * p2.relative(p1); // => ../../../xxx/yyy
 * ~~~~~~~~~~~~~~~~~~~~
 *
 * \remark Both the referenced path and the `base` argument
 * are prune()d before they are worked with.
 *
 * \see absolute()
 */
Path Path::relative(Path base) const
{
  if (is_relative())
    throw(std::invalid_argument("Referenced path must be absolute."));
  if (base.is_relative())
    throw(std::invalid_argument("Argument path must be absolute."));

  // Wipe all ".." and ".", this would break the below algorithm
  base = base.prune();
  Path refpath = prune();

  // Shortcut for equal paths
  if (base.m_path == refpath.m_path)
    return Path(".");

  // Shortcut if base is the root
  if (base.is_root()) {
#if defined(_PATHIE_UNIX)
    return Path(refpath.m_path.substr(1)); // Skip leading /
#elif defined(_WIN32)
    return Path(refpath.m_path.substr(root().m_path.length())); // Skip leading / or X:/
#else
#error Unsupported system.
#endif
  }

  size_t pos = 0;
  size_t baselength = base.m_path.length();
  size_t reflength  = refpath.m_path.length();
  while (true) {
    if (pos >= baselength)
      break;
    else if (pos >= reflength)
      break;
    else if (base.m_path[pos] != refpath.m_path[pos])
      break;
    else
      pos++;
  }
  // pos now points to the last character in which both strings were equal

  // For each component in base that is not part of refpath, add a "..".
  std::string resultstr;
  Path basepart(base.m_path.substr(pos));
  for(size_t i=0; i < basepart.component_count(); i++)
    resultstr.append("../");

  // Now append the part of refpath that is not part of base to the result.
  resultstr.append(refpath.m_path.substr(pos));

  // Done.
  return Path(resultstr);
}

/**
 * Checks if this is an absolute path, i.e. one that
 * starts with a / on all systems or with X:/
 * only on Windows, where `X` is a drive letter.
 *
 * Note that / on Windows is the root of the current drive
 * and hence also an absolute path.
 */
bool Path::is_absolute() const
{
#if defined(_PATHIE_UNIX)
  return m_path[0] == '/';
#elif defined(_WIN32)
  // / is root on current drive
  if (m_path[0] == '/')
    return true;

  return m_path[1] == ':'; // This is the only position where : is allowed on windows, and if it is there, the path is absolute with a drive letter (X:/)
#else
#error Unsupported system.
#endif
}

/**
 * The inverse of is_absolute().
 */
bool Path::is_relative() const
{
  return !is_absolute();
}

/**
 * Checks if this path is a filesystem root. On UNIX, this
 * is the case if the path consists solely of one slash, on
 * Windows this is the case if the path looks like this:
 * "<letter>:/".
 */
bool Path::is_root() const
{
#if defined(_PATHIE_UNIX)
  return m_path.length() == 1 && m_path[0] == '/';
#elif defined(_WIN32)
  // / on Windows is root on current drive
  if (m_path.length() == 1 && m_path[0] == '/')
    return true;

  // X:/ is root including drive letter
  return m_path.length() == 3 && m_path[1] == ':';
#else
#error Unsupported platform.
#endif
}

///@}

/** \name In-place substitution
 *
 * These methods change the underlying path string.
 */
///@{

void Path::assign(std::string str)
{
  m_path = str;
}

void Path::swap(Path& path) throw()
{
  m_path.swap(path.m_path);
}

///@}

/** \name File attributes
 *
 * Functions that work on file attributes like timestamps.
 */
///@{

#if defined(_PATHIE_UNIX)
struct stat* Path::stat() const
{
  struct stat* s = (struct stat*) malloc(sizeof(struct stat));
  std::string nstr = native();

  if (::stat(nstr.c_str(), s) < 0)
    throw(Pathie::ErrnoError(errno));

  return s;
}
#elif defined(_WIN32)
/**
 * \note This method accesses the file system.
 *
 * Returns a pointer to a C `stat` struct that describes the
 * given file. You have to free() the pointer manually yourself.
 *
 * \returns A `struct stat` pointer on UNIX, and a `struct _stat`
 * pointer on Windows.
 */
struct _stat* Path::stat() const
{
  struct _stat* s = (struct _stat*) malloc(sizeof(struct _stat));
  std::wstring utf16 = utf8_to_utf16(m_path);

  if (_wstat(utf16.c_str(), s) < 0)
    throw(Pathie::ErrnoError(errno));

  return s;
}
#else
#error Unsupported system.
#endif

/**
 * \note This method accesses the file system.
 *
 * Returns the file size.
 */
long Path::size() const
{
#if defined(_PATHIE_UNIX)
  struct stat s;
  std::string nstr = native();

  if (::stat(nstr.c_str(), &s) < 0)
    throw(Pathie::ErrnoError(errno));
#elif defined(_WIN32)
  struct _stat s;
  std::wstring utf16 = utf8_to_utf16(m_path);

  if (_wstat(utf16.c_str(), &s) < 0)
    throw(Pathie::ErrnoError(errno));
#else
#error Unsupported system.
#endif

  return s.st_size;
}

/**
 * \note This method accesses the file system.
 *
 * Returns the file’s last access time. The value is not
 * really reliable.
 */
time_t Path::atime() const
{
#if defined(_PATHIE_UNIX)
  struct stat s;
  std::string nstr = native();

  if (::stat(nstr.c_str(), &s) < 0)
    throw(Pathie::ErrnoError(errno));
#elif defined(_WIN32)
  struct _stat s;
  std::wstring utf16 = utf8_to_utf16(m_path);

  if (_wstat(utf16.c_str(), &s) < 0)
    throw(Pathie::ErrnoError(errno));
#else
#error Unsupported system.
#endif

  return s.st_atime;
}

/**
 * \note This method accesses the file system.
 *
 * Returns the file’s last modification time.
 */
time_t Path::mtime() const
{
#if defined(_PATHIE_UNIX)
  struct stat s;
  std::string nstr = native();

  if (::stat(nstr.c_str(), &s) < 0)
    throw(Pathie::ErrnoError(errno));
#elif defined(_WIN32)
  struct _stat s;
  std::wstring utf16 = utf8_to_utf16(m_path);

  if (_wstat(utf16.c_str(), &s) < 0)
    throw(Pathie::ErrnoError(errno));
#else
#error Unsupported system.
#endif

  return s.st_mtime;
}

/**
 * \note This method accesses the file system.
 *
 * Returns the file’s creation time.
 */
time_t Path::ctime() const
{
#if defined(_PATHIE_UNIX)
  struct stat s;
  std::string nstr = native();

  if (::stat(nstr.c_str(), &s) < 0)
    throw(Pathie::ErrnoError(errno));
#elif defined(_WIN32)
  struct _stat s;
  std::wstring utf16 = utf8_to_utf16(m_path);

  if (_wstat(utf16.c_str(), &s) < 0)
    throw(Pathie::ErrnoError(errno));
#else
#error Unsupported system.
#endif

  return s.st_ctime;
}

///@}

/** \name Path traversal
 *
 * What’s in this directory?
 */
///@{

/**
 * Returns an entry_iterator instance you can use to iterate
 * the entries in a directory. Note that the list somewhere
 * always includes the "." (current directory) and ".."
 * (parent directory) entries.
 */
entry_iterator Path::begin_entries() const
{
  return entry_iterator(this);
}

/**
 * Returns the terminal iterator you test for in order to
 * find out whether the iteration is complete.
 */
entry_iterator Path::end_entries() const
{
  return entry_iterator();
}

/**
 * \note This method accesses the file system.
 *
 * This method assumes the path is a directory and returns
 * a list of all entries in it. The items in the list follow
 * the order of the items on the file system, i.e. for most
 * applications they are to be considered unsorted.
 *
 * \see children()
 */
std::vector<Path> Path::entries() const
{
  std::vector<Path> results;
  for(entry_iterator iter=begin_entries(); iter != end_entries(); iter++) {
    results.push_back(*iter);
  }

  return results;
}

/**
 * \note This method accesses the file system.
 *
 * This method assumes the path is a directory and returns
 * a list of all its children. Children are all entries
 * in the directory *except* for the entries for the directory
 * itself and its parent directory.
 *
 * Or for short, this method is the same as children() except
 * the return value does not include the "." and ".." entries.
 *
 * \see entries()
 */
std::vector<Path> Path::children() const
{
  std::vector<Path> results;
  for(entry_iterator iter=begin_entries(); iter != end_entries(); iter++) {
    if (*iter != Path(".") && *iter != Path(".."))
      results.push_back(*iter);
  }

  return results;
}

/**
 * \note This method accesses the file system.
 *
 * Recursively traverse the directory structure below the referenced
 * path. Each entry will be passed to the callback while traversing
 * from top to bottom. If the entry passed is a directory, you can return
 * true if you want to traverse that directory down or false if you
 * don't want to. If the entry passed is not a directory, the
 * callback's return value is ignored.
 *
 * The callback will never be passed "." and ".." entries. All paths
 * passed to the callback retain the full prefix, i.e. if you
 * have this structure:
 *
 * ~~~~~~~~~~~~~~~~
 * foo
 *   bar/
 *     baz.txt
 * ~~~~~~~~~~~~~~~~
 *
 * Then find() will give you these paths in this order: `foo`,
 * `foo/bar`, and `foo/bar/baz.txt`, rather than just the sole
 * basename (which you can still obtain by calling basename() on the
 * argument).
 *
 * \param cb Callback that takes the currently examined path.
 *
 * \remark Do not assume any order for the paths you receive,
 * except that you will be given a directory entry before you
 * are given its child entries.
 */
void Path::find(bool (*cb)(const Path& entry)) const
{
  for(entry_iterator iter=begin_entries(); iter != end_entries(); iter++) {
    // Skip . and ..
    if (iter->str() != "." && iter->str() != "..") {
      Path path = join(*iter);
      if (cb(path) && path.is_directory()) {
        path.find(cb);
      }
    }
  }
}

///@}

/** \name Path status information
 *
 * Query information on the path.
 */
///@{


/**
 * \note This method acceses the filesystem.
 *
 * Checks if the file exists. Note that if you don’t have
 * sufficient rights for the check on the given path, this
 * method will throw an exception.
 */
bool Path::exists() const
{
#if defined(_PATHIE_UNIX)
  std::string nstr = native();

  if (access(nstr.c_str(), F_OK) == -1) {
    int errsav = errno;
    if (errsav == ENOENT) {
      return false;
    }
    else {
      throw(Pathie::ErrnoError(errsav));
    }
  }
  else
    return true;
#elif defined(_WIN32)
  std::wstring utf16 = utf8_to_utf16(m_path);
  if (_waccess(utf16.c_str(), F_OK) == -1) {
    int errsav = errno;
    if (errsav == ENOENT) {
      return false;
    }
    else {
      throw(Pathie::ErrnoError(errsav));
    }
  }
  else
    return true;
#else
#error Unsupported system.
#endif
}

/**
 * \note This method acceses the filesystem.
 *
 * Checks if this file is a symbolic link; also
 * works with NTFS symlinks on Windows. Returns false
 * rather than erroring out if the referenced path does
 * not exist.
 */
bool Path::is_symlink() const
{
#if defined(_PATHIE_UNIX)
  struct stat s;
  std::string nstr = native();

  if (lstat(nstr.c_str(), &s) < 0) {
    int errsav = errno;

    if (errsav == ENOENT)
      return false;
    else
      throw(Pathie::ErrnoError(errsav));
  }

  if (S_ISLNK(s.st_mode))
    return true;
  else
    return false;
#elif defined(_WIN32)
  if (!exists())
    return false;

  return false;
  // ntifs.h is currently not included in msys2
  //std::wstring path = utf8_to_utf16(m_path);
  //return is_ntfs_symlink(path.c_str());
#else
#error Unsupported system.
#endif
}

/**
 * \note This method acceses the filesystem.
 *
 * Checks if this is a directory. Returns false if the
 * referenced path does not exist rather than erroring out.
 */
bool Path::is_directory() const
{
#if defined(_PATHIE_UNIX)
  struct stat s;
  std::string nstr = native();

  if (::stat(nstr.c_str(), &s) < 0) {
    int errsav = errno;

    // "Not found" means it isn’t a directory.
    if (errsav == ENOENT)
      return false;
    else
      throw(Pathie::ErrnoError(errsav));
  }

  if (S_ISDIR(s.st_mode))
    return true;
  else
    return false;
#elif defined(_WIN32)
  struct _stat s;
  std::wstring utf16 = utf8_to_utf16(m_path);
  if (_wstat(utf16.c_str(), &s) < 0) {
    int errsav = errno;

    if (errsav == ENOENT)
      return false;
    else
      throw(Pathie::ErrnoError(errsav));
  }

  return (s.st_mode & S_IFDIR) != 0;
#else
#error Unsupported system.
#endif
}

/**
 * \note This method accesses the filesystem.
 *
 * Checks if this is a file. Returns false
 * if the referenced path does not exist rather
 * than erroring out.
 */
bool Path::is_file() const
{
#if defined(_PATHIE_UNIX)
  struct stat s;
  std::string nstr = native();

  if (::stat(nstr.c_str(), &s) < 0) {
    int errsav = errno;

    if (errsav == ENOENT)
      return false;
    else
      throw(Pathie::ErrnoError(errsav));
  }

  if (S_ISREG(s.st_mode))
    return true;
  else
    return false;
#elif defined(_WIN32)
  struct _stat s;
  std::wstring utf16 = utf8_to_utf16(m_path);
  if (_wstat(utf16.c_str(), &s) < 0) {
    int errsav = errno;

    if (errsav == ENOENT)
      return false;
    else
      throw(Pathie::ErrnoError(errno));
  }

  return (s.st_mode & S_IFREG) != 0;
#else
#error Unsupported system.
#endif
}

///@}

/** \name Utility methods
 *
 * These methods operate on the file or directory referenced
 * by the path.
 */
/// @{

/**
 * \note This method writes to the filesystem.
 *
 * Creates the referenced directory non-recursively,
 * i.e. parent directories are not created. Trying
 * to create a directory below a nonexistant directory
 * will result in an ErrnoError exception.
 *
 * \remark UNIX note: The directory is created with RWX permissions
 * for everyone, but filtered by your current `umask` before applied
 * to disk.
 *
 * \see mktree()
 */
void Path::mkdir() const
{
#if defined(_PATHIE_UNIX)
  std::string nstr = native();

  if (::mkdir(nstr.c_str(), S_IRWXU | S_IRWXG | S_IRWXO) < 0)
    throw(Pathie::ErrnoError(errno));
#elif defined(_WIN32)
  std::wstring utf16 = utf8_to_utf16(m_path);

  if (_wmkdir(utf16.c_str()) < 0)
    throw(Pathie::ErrnoError(errno));
#else
#error Unsupported system.
#endif
}

/**
 * \note This method writes to the filesystem.
 *
 * Deletes the referenced directory, which is required
 * to be empty, if not, an ErrnoError will be thrown.
 *
 * This cannot be used to delete a file rather than a
 * directory.
 *
 * \see remove() unlink()
 */
void Path::rmdir() const
{
#if defined(_PATHIE_UNIX)
  std::string nstr = native();

  if (::rmdir(nstr.c_str()) < 0)
    throw(Pathie::ErrnoError(errno));
#elif defined(_WIN32)
  std::wstring utf16 = utf8_to_utf16(m_path);
  if (_wrmdir(utf16.c_str()) < 0)
    throw(Pathie::ErrnoError(errno));
#else
#error Unsupported system.
#endif
}

/**
 * \note This method writes to the filesystem.
 *
 * Deletes the referenced file. This cannot be used to
 * delete a directory rather than a file.
 *
 * \see remove() rmdir()
 */
void Path::unlink() const
{
#if defined(_PATHIE_UNIX)
  std::string nstr = native();
  if (::unlink(nstr.c_str()) < 0)
    throw(Pathie::ErrnoError(errno));
#elif defined(_WIN32)
  std::wstring utf16 = utf8_to_utf16(m_path);
  if (_wunlink(utf16.c_str()) < 0)
    throw(Pathie::ErrnoError(errno));
#else
#error Unsupported system.
#endif
}

/**
 * \note This method writes to the filesystem.
 *
 * Delete this path, regardless of whether it is a file
 * or an empty directory. This method can’t be used to
 * delete a directory that isn’t empty.
 *
 * \see rmdir() unlink()
 */
void Path::remove() const
{
#if defined(_PATHIE_UNIX)
  std::string nstr = native();

  if (::remove(nstr.c_str()) < 0)
    throw(Pathie::ErrnoError(errno));
#elif defined(_WIN32)
  std::wstring utf16 = utf8_to_utf16(m_path);
  bool result = false;

  /* On Windows, `_wremove()` doesn’t work on directories. This
   * function uses the apropriate native Win32API function
   * calls accordingly therefore. */
  if (is_directory())
    result = RemoveDirectoryW(utf16.c_str()) != 0;
  else
    result = DeleteFileW(utf16.c_str()) != 0;

  if (!result) {
    DWORD err = GetLastError();
    throw(Pathie::WindowsError(err));
  }

#else
#error Unsupported system.
#endif
}

/**
 * \note This method writes to the file system.
 *
 * This method provides a functionality akin to the UNIX `mkdir -p`
 * command, i.e. it creates the referenced directory, and if necessary,
 * also creates all parent directories. Note this method does not
 * throw an ErrnoError if the referenced directory already exists;
 * it just does nothing.
 *
 * \see mkdir()
 */
void Path::mktree() const
{
  // Root is required to exist
  if (is_root())
    return;

  if (!is_directory()) {
    Path p = parent();

    if (!p.is_directory()) {
      p.mktree();
    }

    mkdir();
  }

}

/**
 * \note This method accesses the filesystem.
 *
 * Open the referenced path as a file with the given mode.
 * Refer to your preferred C documentation for the value
 * of the `mode` parameter.
 *
 * As with all methods of this library, Unicode filenames
 * are handled properly on both UNIX and Windows by transcoding
 * to UTF-16LE on Windows. Therefore, on UNIX the file
 * is opened using `fopen()`, and on Windows it is opened
 * using `_wfopen()`. Thanksfully, as an exception
 * to Microsoft’s wchar-them-all rule, it is possible to close
 * a file that is opened with `_wfopen()` by means of the
 * regular `fclose()` function, which saves me from implementing
 * a wrapper around the C `FILE*` pointer to abstract the problem.
 *
 * In contrast to original `fopen()`, this method throws an
 * ErrnoError exception if the call fails, i.e. if `fopen()`
 * returns NULL. As a result, this method will _never_ return
 * a NULL pointer.
 *
 * Here’s an example of how to use this method (with error checking
 * ommited):
 *
 * ~~~~~~~~~~~~~~~~~ c++
 * Path p("Unicöde file.txt");
 * FILE* p_file = p.fopen("w");
 * fwrite("A", 1, 1, p_file);
 * fclose(p_file);
 * ~~~~~~~~~~~~~~~~~
 *
 * This will create a file named "Unicöde.txt" both on UNIX and Windows.
 *
 * \param[in] mode File open mode as per the C `fopen()` documentation.
 *
 * \remark Don’t forget you have to close the file using `fclose()`, which
 * works, as explained, both on UNIX and Windows. `fclose()` is
 * not wrapped by this library, use your C libraries’ implementation
 * directly.
 *
 * \remark The file’s actual _contents_ are not affected in any way
 * by this method. They are outside the scope of this library; note
 * however that with regard to line endings you might want to consider
 * the "b" mode modifier for binary files.
 *
 * \see [Microsoft’s documentation on `fopen()` and `_wfopen()`](http://msdn.microsoft.com/en-us/library/yeby3zcb.aspx)
 */
FILE* Path::fopen(const char* mode) const
{
#if defined(_PATHIE_UNIX)
  std::string nstr = native();

  FILE* ptr = ::fopen(nstr.c_str(), mode);
  if (ptr)
    return ptr;
  else
    throw(Pathie::ErrnoError(errno));
#elif defined(_WIN32)
  std::wstring utf16_path = utf8_to_utf16(m_path);
  std::wstring utf16_mode = utf8_to_utf16(mode);
  FILE* ptr = _wfopen(utf16_path.c_str(), utf16_mode.c_str());

  if (ptr)
    return ptr;
  else
    throw(Pathie::ErrnoError(errno));
#else
#error Unsupported system.
#endif
}

/**
 * \note This method writes to the filesystem.
 *
 * Sets the file’s modification and access times to the
 * current time. If the file does not yet exist, it is created.
 *
 * This is akin to the UNIX `touch` command.
 */
void Path::touch() const
{
#if defined(BSD) // FreeBSD didn’t have futimens() yet as of testing (december 2014)
  FILE* p_file = Path::fopen("a");
  if (futimes(fileno(p_file), NULL) < 0) {
    fclose(p_file);
    throw(Pathie::ErrnoError(errno));
  }

  fclose(p_file);
#elif defined(_PATHIE_UNIX)
  FILE* p_file = Path::fopen("a");
  // futimens() is considered the modern variant of doing this
  // (at least according to utimes(2) on my Linux system).
  if (futimens(fileno(p_file), NULL) < 0) {
    fclose(p_file);
    throw(Pathie::ErrnoError(errno));
  }

  fclose(p_file);
#elif defined(_WIN32)
  // Create file if it does not exist yet
  if (!exists()) {
    FILE* p_file = Path::fopen("a");
    fclose(p_file);
  }

  SYSTEMTIME currenttime;
  GetSystemTime(&currenttime);

  FILETIME newtime;
  if (SystemTimeToFileTime(&currenttime, &newtime) == 0) {
    DWORD err = GetLastError();
    throw(Pathie::WindowsError(err));
  }

  std::wstring utf16 = utf8_to_utf16(m_path);
  HANDLE filehandle = CreateFileW(utf16.c_str(), FILE_WRITE_ATTRIBUTES, 0, NULL, OPEN_EXISTING, 0, NULL);
  if (filehandle == INVALID_HANDLE_VALUE) {
    DWORD err = GetLastError();
    throw(Pathie::WindowsError(err));
  }

  if (SetFileTime(filehandle, NULL, &newtime, &newtime) == 0) {
    int errsav = GetLastError();
    CloseHandle(filehandle);
    throw(Pathie::WindowsError(errsav));
  }

  CloseHandle(filehandle);
#else
#error Unsupported system.
#endif
}

/**
 * \note This method writes to the filesystem.
 *
 * This method, which is akin to the UNIX "rm -r" command, removes
 * the entire referenced directory hierarchy recursively, including
 * any files and directories contained therein.
 */
void Path::rmtree() const
{
  if (is_directory()) {
    std::vector<Path> kids = children();

    for(std::vector<Path>::iterator iter=kids.begin(); iter != kids.end(); iter++) {
       join(*iter).rmtree();
    }

    rmdir();
  }
  else { // file or similar
    unlink();
  }
}

/**
 * \note This method writes to the filesystem.
 *
 * This method makes the referenced file a symbolic link
 * to the path passed as an argument. On Windows, an
 * NTFS symlink is created.
 *
 * \remark On Windows, this function requires that the process holds
 * the `SE_CREATE_SYMBOLIC_LINK_NAME` privilege or it will fail with a
 * WindowsError exception whose error code is 1314
 * (`ERROR_PRIVILEGE_NOT_HELD`).
 */
void Path::make_symlink(const Path& target) const
{
#if defined(_PATHIE_UNIX)
  std::string target_nstr = target.native();
  std::string nstr = native();

  if (symlink(target_nstr.c_str(), nstr.c_str()) < 0)
    throw(Pathie::ErrnoError(errno));
#elif defined(_WIN32)
  std::wstring source = utf8_to_utf16(m_path);
  std::wstring target2 = utf8_to_utf16(target.m_path);

  DWORD flags = 0;
  if (target.is_directory())
    flags = SYMBOLIC_LINK_FLAG_DIRECTORY;

  if (CreateSymbolicLinkW(source.c_str(), target2.c_str(), flags) == 0) {
    DWORD err = GetLastError();
    throw(Pathie::WindowsError(err));
  }
#else
#error Unsupported system.
#endif
}

/**
 * \note This method accesses the file system.
 *
 * Treats the referened path as a symlink and reads in its target,
 * returning it as a new Path intance. Supports NTFS symlinks.
 */
Path Path::readlink() const
{
#if defined(_PATHIE_UNIX)
  std::string nstr = native();
  char buf[PATH_MAX];
  memset(buf, '\0', PATH_MAX);

  ssize_t count = ::readlink(nstr.c_str(), buf, PATH_MAX);
  if (count < 0)
    throw(Pathie::ErrnoError(errno));

  return Path(filename_to_utf8(std::string(buf, count)));
#elif defined(_WIN32)
  std::wstring utf16_path = utf8_to_utf16(m_path);

  throw(std::runtime_error("NTFS symlinks currently not supported."));

  // ntifs.h currently not included in msys2.h
  //if (!is_ntfs_symlink(utf16_path.c_str()))
  //  throw(std::runtime_error("Not an NTFS symlink."));
  //
  //wchar_t* utf16_target = NULL;
  //utf16_target = read_ntfs_symlink(utf16_path.c_str());
  //
  //Path result(utf16_to_utf8(utf16_target));
  //free(utf16_target);
  //
  //return result;
#else
#error Unsupported system.
#endif
}

/**
 * \note This method writes to the file system.
 *
 * Renames a file to another name without involving file streams.
 *
 * \param[in] newname The new name of the file.
 */
void Path::rename(Path& newname) const
{
#if defined(_PATHIE_UNIX)
  std::string nstr = native();
  std::string newname_nstr = newname.native();

  if (::rename(nstr.c_str(), newname_nstr.c_str()) != 0)
    throw Pathie::ErrnoError(errno);
#elif defined(_WIN32)
  std::wstring utf16_oldname = utf8_to_utf16(m_path);
  std::wstring utf16_newname = utf8_to_utf16(newname.m_path);

  if (_wrename(utf16_oldname.c_str(), utf16_newname.c_str()) != 0)
    throw Pathie::ErrnoError(errno);
#else
#error Unsupported system.
#endif
}

///@}

/** \name Operators
 *
 * C++ operators.
 */
///@{

Path& Path::operator=(const Path& path)
{
  // Self-assignment
  if (this == &path)
    return *this;

  m_path = path.m_path;
  return *this;
}

Path& Path::operator=(const std::string& str)
{
  m_path = str;
  return *this;
}

/**
 * Compares two Path instances. Two paths are considered equal
 * if their underlying path std::strings are equal.
 */
bool Path::operator==(const Path& other) const
{
  return m_path == other.m_path;
}

/**
 * Compares two Path instances. Two paths are considered inequal
 * if their underlying path std::strings are inequal.
 */
bool Path::operator!=(const Path& other) const
{
  return m_path != other.m_path;
}

/**
 * Compares two Path instances. The referenced path is
 * considered smaller than `other` if the underlying path
 * std::string of the referenced path is smaller than the
 * one of `other`.
 */
bool Path::operator<(const Path& other) const
{
  return m_path < other.m_path;
}

/**
 * Compares two Path instances. The referenced path is
 * considered greater than `other` if the underlying path
 * std::string of the referenced path is greater than the
 * one of `other`.
 */
bool Path::operator>(const Path& other) const
{
  return m_path > other.m_path;
}

/**
 * Compares two Path instances. The referenced path is
 * considered smaller than or equal to `other` if the underlying path
 * std::string of the referenced path is smaller than or equal to the
 * one of `other`.
 */
bool Path::operator<=(const Path& other) const
{
  return m_path <= other.m_path;
}

/**
 * Compares two Path instances. The referenced path is
 * considered greater than or equal to `other` if the underlying path
 * std::string of the referenced path is greater than or equal to the
 * one of `other`.
 */
bool Path::operator>=(const Path& other) const
{
  return m_path >= other.m_path;
}

/**
 * This method allows you to access a specific component in the
 * path string. The first component has the index 0; for an
 * absolute path, it will be the / entry.
 *
 * If you specify an index that is beyond the end of the path,
 * an std::out_of_range exception will be thrown.
 *
 * \param index Index of the component to retrieve.
 *
 * \see component_count()
 *
 * \remark This operator loops over the path string internally
 * each time you request an element. If you want to index the
 * path consecutively, you might consider using burst(), which
 * can be more performant as it only loops once over the path
 * string.
 */
Path Path::operator[](size_t index) const
{
  // Absolute path index 0 needs special treatment
  if (index == 0 && m_path[0] == '/')
    return Path("/");

  size_t pos     = 0;
  size_t lastpos = 0;
  size_t i       = 0;
  while ((pos = m_path.find("/", pos)) != string::npos) { // Assignment intended
    if (i == index)
      return Path(m_path.substr(lastpos, pos - lastpos));

    lastpos = pos + 1;
    pos++;
    i++;
  }

  // Last element requested
  if (index == i)
    return Path(m_path.substr(lastpos));

  // Out of range
  throw(std::out_of_range("Index out of path range"));
}

/**
 * Appends a /, then the new component, then calls expand(), and
 * finally returns a new Path instance.
 *
 * \param path New component.
 *
 * \returns New Path instance.
 */
Path Path::operator/(Path path) const
{
  return join(path);
}

/**
 * Appends a /, then the new component, and
 * finally returns a new Path instance.
 *
 * \param str New component.
 *
 * \returns New Path instance.
 */
Path Path::operator/(std::string str) const
{
  return join(str);
}

/**
 * Appends a / followed by the new component `path` onto this
 * instance and returns this instance.
 *
 * \param path New component.
 *
 * \returns The receiver.
 */
Path& Path::operator/=(Path path)
{
  *this = join(path);
  return *this;
}

/**
 * Appends a / followed by the new component `path` onto this
 * instance and returns this instance.
 *
 * \param str New component.
 *
 * \returns The receiver.
 */
Path& Path::operator/=(std::string str)
{
  *this = join(str);
  return *this;
}

/**
 * Allows you to insert Pathie::Path instances into `std::cout`.
 *
 * ~~~~~~~~~~ c++
 * Pathie::Path p("foo/bar");
 * std::cout << p << std::endl;
 * ~~~~~~~~~~
 */
std::ostream& operator<<(std::ostream& stream, const Path& p)
{
  return stream << p.str();
}

///@}

#ifdef _PATHIE_UNIX
/*
 * Returns the XDG directory for the given environment variable,
 * if defined, otherwise returns home() with `defaultpath`
 * appended.
 *
 * See http://standards.freedesktop.org/basedir-spec/basedir-spec-latest.html
 * for values.
 */
Path Path::get_xdg_dir(const std::string& envvarname, const std::string& defaultpath)
{
  std::string env_nstr = utf8_to_filename(envvarname); // environment is encoded the same as the filenames
  char* env_value = getenv(env_nstr.c_str());
  if (env_value)
    return Path(filename_to_utf8(env_value));

  return Path::home().join(defaultpath);
}

std::vector<Path> Path::get_xdg_dirlist(const std::string& envvarname, const std::string& defaultlist)
{
  std::string env_nstr = utf8_to_filename(envvarname); // environment is encoded the same as the filenames
  char* env_value = getenv(env_nstr.c_str());
  std::string envstr;
  if (env_value && strcmp(env_value, "") != 0)
    envstr = filename_to_utf8(env_value); // Encode entire env string to UTF-8
  else
    envstr = defaultlist;

  size_t pos = 0;
  size_t lastpos = 0;
  std::vector<Path> results;
  while ((pos = envstr.find(":")) != string::npos) {
    results.push_back(Path(envstr.substr(lastpos, pos))); // envstr is already UTF-8

    lastpos = pos + 1;
    pos++;
  }

  results.push_back(envstr.substr(lastpos));

  return results;
}

std::string Path::get_xdg_userdir_setting(const std::string& setting)
{
  // XDG user-dirs spec recommends (only) checking for $XDG_CONFIG_HOME/user-dirs.dirs,
  // the files under $XDG_CONFIG_DIRS are not to consider.
  Path userconfig = Path::config_dir().join("user-dirs.dirs");

  if (userconfig.is_file()) {
    FILE* p_file = userconfig.fopen("r");

    char line[256];
    char buf[256];
    bool found = false;
    while (!feof(p_file)) {
      memset(line, 0, 256);
      memset(buf, 0, 256);

      fgets(line, 256, p_file);

      // Ignore comments and empty lines
      if (line[0] == '#' || line[0] == '\n')
        continue;

      // Extract the setting name from the line, e.g. "DOWNLOAD" for
      // "XDG_DOWNLOAD_DIR=...".
      strncpy(buf, line + 4, setting.length()); // +4 for "XDG_"
      if (strcmp(buf, setting.c_str()) == 0) {
        found = true;
        break;
      }
    }

    fclose(p_file);

    // Error out if not found
    if (!found) {
      std::string msg = "Unknown XDG directory '";
      msg += setting + "' requested.";
      throw(std::runtime_error(msg));
    }

    // OK, we have found the correct setting. Extract the value now.
    // »XDG_DOWNLOAD_DIR="$HOME/Downloads"«
    char* start = strchr(line, '"') + 1; // Exclude " itself
    size_t len  = strcspn(start, "\"");

    if (!start) // Malformed
      throw(std::runtime_error("Malformed XDG config file (quote mismatch/missing quotes)!"));

    memset(buf, 0, 256);
    strncpy(buf, start, len);
    // buf now contains the part between the quotes followed by NUL bytes

    char result[PATH_MAX];
    memset(result, 0, PATH_MAX);

    // Replace $HOME with env value of $HOME
    start = strstr(buf, "$HOME");
    if (start) { // Contains $HOME
      char* homestr = getenv("HOME");
      if (!homestr)
        throw(std::runtime_error("$HOME not set!"));

      // Stuff before $HOME
      strncpy(result, buf, ((char*)start) - ((char*)buf)); // Compiler does not allow doing pointer arithmetics with char[], but with char* ??? They should be the same...
      // $HOME replacement
      strcpy(result + strlen(result), homestr);
      // Suff after $HOME ($HOME is exactly 5 chars long)
      strcpy(result + strlen(result), start + 5);
    }
    else { // No $HOME included. Copy everything verbosely.
      strcpy(result, buf);
    }

    // result now holds the final result with lots of NUL bytes at the end.
    return std::string(result);
  }

  // No XDG configuration on this system, use $HOME.
  return Path::home().str();
}
#endif

/** \name Program data directories
 *
 * Directories containing program data other than files the
 * user works with (e.g. configuration files).
 */
///@{

/**
 * Returns the directory for application- and user-specific permanent
 * data.
 *
 * On UNIX, this returns $XDG_DATA_HOME, defaulting to ~/.local/share.
 *
 * On Windows, this returns the roaming appdata folder, which defaults
 * to `C:/Users/username/AppData/Roaming`.
 */
Path Path::data_dir()
{
#if defined(_PATHIE_UNIX)
  return get_xdg_dir("XDG_DATA_HOME", ".local/share");
#elif defined(_WIN32)
  wchar_t dir[MAX_PATH];
  HRESULT result = SHGetFolderPathW(NULL, CSIDL_APPDATA, NULL, SHGFP_TYPE_CURRENT, dir);
  if (result != S_OK)
    throw(Pathie::WindowsHresultError(result));

  return Path(utf16_to_utf8(dir));
#else
#error Unsupported system.
#endif
}

/**
 * \warning This method may behave unexpectedly on Windows; see below.
 *
 * Returns the directory for application- and user-specific configuration
 * files.
 *
 * On UNIX, this returns $XDG_CONFIG_HOME, defaulting to ~/.config.
 *
 * Windows does not have a notion of a directory for configuration
 * files, hence some return value for this method had to be chosen. I
 * think it is best to not clutter a user’s home directory with config
 * files, and [this stackoverflow thread](https://stackoverflow.com/questions/2243895/location-to-put-user-configuration-files-in-windows)
 * suggests to place the files in the data_dir(). That however yields
 * the problem of possible name clashes when you want to name a file
 * the same in data_dir() and config_dir(). It is not an option to
 * fall back to the "LocalSettings" directory instead, because 99% of
 * the applications written are "roaming" applications rather than
 * "local" ones, and any use of the "LocalSettings" directory
 * (available via cache_dir()) must be a specific decision of the
 * programmer therefore. The decision was made that this method on
 * Windows should return the same as data_dir() without a specific
 * encforcing reason, but, as said, some decision needed to be
 * made. As a consequence, you have to be careful to not accidentally
 * place equally named files in data_dir() and config_dir() as they
 * would conflict.
 *
 * I want to point out that on Windows, configuration files are rather
 * unusual. The normal way to save configuration settings on Windows
 * is use of the Windows Registry, which is beyond the scope of a
 * path manipulation library like Pathie.
 */
Path Path::config_dir()
{
#if defined(_PATHIE_UNIX)
  return get_xdg_dir("XDG_CONFIG_HOME", ".config");
#elif defined(_WIN32)
  return data_dir();
#else
#error Unsupported system.
#endif
}

/**
 * Returns the directory for application- and user-specific cache files, i.e.
 * files that, when deleted, do not impact the application apart from resetting
 * it to some default values. A typical example for cache data is saving the
 * folder where the user last opened a file, so that when he starts the application
 * the next time and wants to open a file, is directly taken to the directory
 * where he last picked a file from. Positions of windows could also be saved
 * here, allowing application windows to be placed exactly where they were
 * when the application was closed last time. In short, store the unimportant
 * stuff here and be prepared the data is gone on application startup.
 *
 * On UNIX, this returns $XDG_CACHE_HOME, defaulting to ~/.cache.
 *
 * On Windows, this method returns the LOCAL_APPDATA folder, which means that
 * in corporate setups using Windows roaming the data will not be available
 * if you log in on another machine (which is expected, cf. the directory
 * saving example above, which would break if this was saved into the roaming
 * folder). This defaults to `C:/Users/username/AppData/Local`.
 */
Path Path::cache_dir()
{
#if defined(_PATHIE_UNIX)
  return get_xdg_dir("XDG_CACHE_HOME", ".cache");
#elif defined(_WIN32)
  wchar_t dir[MAX_PATH];
  HRESULT result = SHGetFolderPathW(NULL, CSIDL_LOCAL_APPDATA, NULL, SHGFP_TYPE_CURRENT, dir);
  if (result != S_OK)
    throw(Pathie::WindowsHresultError(result));

  return Path(utf16_to_utf8(dir));
#else
#error Unsupported system.
#endif
}

/**
 * Returns the directory for application- and user-specific volatile
 * runtime data, i.e. data that WILL be deleted once the user logs
 * off.
 *
 * On UNIX, this returns $XDG_RUNTIME_DIR. That environment variable is
 * required to be defined as per the XDG standard, and if it isn’t, this
 * method prints a warning to the standard error stream and uses the
 * value of Path::temp_dir() instead.
 *
 * On Windows, the return value of this method is equivalent to that
 * of temp_dir() always.
 */
Path Path::runtime_dir()
{
#if defined(_PATHIE_UNIX)
  std::string nstr = utf8_to_filename("XDG_RUNTIME_DIR"); // environment is encoded the same as paths
  char* env_value = getenv(nstr.c_str());
  if (env_value)
    return Path(filename_to_utf8(env_value));

  Path tmp = Path::temp_dir();
  std::cerr << "(pathie XDG) WARNING: XDG_RUNTIME_DIR not defined in environment. Falling back to '" << tmp.str() << "'." << std::endl;

  return tmp;
#elif defined(_WIN32)
  return temp_dir();
#else
#error Unsupported system.
#endif
}

/**
 * Returns the root directory for temporary directories, i.e.
 * directories which are expected to vanish when the application
 * closes. Do not assume that anything you created in this
 * directory still exists after your application exited and is
 * restarted.
 *
 * \returns Path instance for temporary directory.
 *
 * \remark On UNIX, this function honours the value of the
 * environment variable $TMPDIR. If that is not defined, the standard
 * "/tmp" location will be returned. On Windows, GetTempPath() is
 * called to retrieve the path, which in turn honours the environment
 * variables $TMP, $TEMP, and $USERPROFILE (in that order); if all
 * of them are undefined, a Windows-predefined path is returned,
 * which defaults to `C:/Users/username/AppData/Local/Temp`.
 *
 * \see mktmpdir(3), [GetTempPath()](http://msdn.microsoft.com/en-us/library/windows/desktop/aa364992%28v=vs.85%29.aspx)
 */
Path Path::temp_dir()
{
#if defined(_PATHIE_UNIX)
  std::string nstr = utf8_to_filename("TMPDIR"); // environment is encoded the same as paths
  char* env_value = NULL;

  if ((env_value = getenv(nstr.c_str()))) // Single = intended
    return Path(filename_to_utf8(env_value));


  return Path("/tmp"); // As per the Filesystem Hierarchy Standard.
#elif defined(_WIN32)
  wchar_t buf[MAX_PATH +1]; // See http://msdn.microsoft.com/en-us/library/windows/desktop/aa364992%28v=vs.85%29.aspx for the +1
  DWORD count = GetTempPathW(MAX_PATH + 1, buf);

  if (count == 0) {
    DWORD err = GetLastError();
    throw(Pathie::WindowsError(err));
  }

  std::wstring utf16(buf, count);
  return utf16_to_utf8(utf16);
#else
#error Unsupported system.
#endif
}

///@}

/**
 * Create a temporary directory (with permissions set to
 * 0700 on UNIX). The directory is guaranteed to be empty, and
 * it is your responsibility to recursively remove the
 * directory on program exit (or earlier).
 *
 * \param[in] name (`"tmpd"`) This will be used as part of
 * the name of the directory, _not_ as the entire name.
 *
 * \returns Path instance for the new temporary directory.
 *
 * \remark Parts of the random name are generated with the
 * C rand() function, so you might want to call srand()
 * before using this function in order to seed the random
 * number generator with a useful value.
 */
Path Path::mktmpdir(const std::string& name /* = "tmpd" */)
{
  Path tmp = Path::temp_dir() / Path(make_tempname(name));
  tmp.mkdir();

#ifdef _PATHIE_UNIX
  std::string nstr = tmp.native();
  chmod(nstr.c_str(), S_IRWXU); // Silently ignore failure of setting file permissions
#endif
  // TODO: How to do that on Windows?

  return tmp;
}

// Constructs a filename that tries to be unique.
std::string Path::make_tempname(const std::string& namepart)
{
  time_t now;
  struct tm* p_nowinfo = NULL;
  time(&now);
  p_nowinfo = localtime(&now);

  char buf[16]; // 15 + NUL
  memset(buf, '\0', 16);
  strftime(buf, 16, "%Y%m%d-%H%M%S", p_nowinfo);
  std::string timepart(buf, 15);

#if defined(_PATHIE_UNIX)
  std::stringstream ss;
  ss << getpid();
  std::string pidpart = ss.str();
#elif defined(_WIN32)
  std::stringstream ss;
  ss << GetCurrentProcessId();
  std::string pidpart = ss.str();
#else
#error Unsupported system.
#endif

  memset(buf, '\0', 16);
  short i;
  for(i=0; i < 16; i++)
    buf[i] = 97 + rand() % 26; // Random char between a and z

  std::string randompart(buf, 15);

  return namepart + "_" + timepart + pidpart + randompart;
}

#if defined(_PATHIE_UNIX)
/**
 * \note Only available on UNIX. Accesses the file system.
 *
 * Returns $XDG_DATA_DIRS as per the XDG specification.
 * If that is not set, returns a vector of paths for
 * /usr/local/share and /usr/share.
 */
std::vector<Path> Path::data_dirs()
{
  return get_xdg_dirlist("XDG_DATA_DIRS", "/usr/local/share/:/usr/share/");
}

/**
 * \note Only available on UNIX. Accesses the file system.
 *
 * Returns $XDG_CONFIG_DIRS as per the XDG specification.
 * If that is not set, returns a vector of paths for
 * /etc/xdg (i.e. a one-element vector).
 */
std::vector<Path> Path::config_dirs()
{
  return get_xdg_dirlist("XDG_CONFIG_DIRS", "/etc/xdg");
}
#endif

/** \name User data directories
 *
 * Directories that contain user data like music or text files
 * the user works with.
 */
///@{

/**
 * \note On UNIX, this method accesses the file system.
 *
 * Retrieves the directory of the user’s desktop. Generally,
 * any files placed in this directory will appear on the
 * user’s desktop view (the area shown when no windows
 * are open).
 *
 * On UNIX, this is $XDG_DESKTOP_DIR, defaulting to `~/Desktop`.
 * Note you likely will receive a localised version (like “Schreibtisch”
 * on a German Linux).
 *
 * On Windows, the default is `C:/Users/username/Desktop` or a localised
 * version.
 */
Path Path::desktop_dir()
{
#if defined(_PATHIE_UNIX)
  return Path(get_xdg_userdir_setting("DESKTOP"));
#elif defined(_WIN32)
  wchar_t dir[MAX_PATH];
  HRESULT result = SHGetFolderPathW(NULL, CSIDL_DESKTOPDIRECTORY, NULL, SHGFP_TYPE_CURRENT, dir);
  if (result != S_OK)
    throw(Pathie::WindowsHresultError(result));

  return Path(utf16_to_utf8(dir));
#else
#error Unsupported system.
#endif
}

/**
 * \note On UNIX, this method accesses the file system.
 *
 * Retrieves the directory for the user’s documents. This is
 * not the place for your data files, savegames, or configuration
 * files -- it is meant only for textual and other documents you can
 * access with an office or similar program. See data_dir() for a directory
 * you can store your data into.
 *
 * On UNIX, this is $XDG_DOCUMENTS_DIR, defaulting to `~/Documents`.
 * Note you likely will receive a localised version (like "Dokumente"
 * on a German Linux).
 *
 * On Windows, the default is `C:/Users/username/Documents` or a localised
 * version.
 */
Path Path::documents_dir()
{
#if defined(_PATHIE_UNIX)
  return Path(get_xdg_userdir_setting("DOCUMENTS"));
#elif defined(_WIN32)
  wchar_t dir[MAX_PATH];
  HRESULT result = SHGetFolderPathW(NULL, CSIDL_PERSONAL, NULL, SHGFP_TYPE_CURRENT, dir);
  if (result != S_OK)
    throw(Pathie::WindowsHresultError(result));

  return Path(utf16_to_utf8(dir));
#else
#error Unsupported system.
#endif
}

/**
 * \note On UNIX, this method accesses the file system.
 *
 * Retrieves the user’s download directory. Unfortunately, this function
 * is currently unsupported under Windows, because MinGW has not yet
 * adapted the necessary win32api changes.
 *
 * On UNIX, this is $XDG_DOWNLOAD_DIR, defaulting to `~/Downloads`.
 * Note you likely will receive a localised version.
 */
Path Path::download_dir()
{
#if defined(_PATHIE_UNIX)
  return Path(get_xdg_userdir_setting("DOWNLOAD"));
#elif defined(_WIN32)
  // Not available via CSIDL, must use the newer KNOWNFOLDERID system,
  // which is not supported by MinGW yet.
  throw(std::runtime_error("KNOWNFOLDERID is not supported by MinGW yet, can't retrieve this directory."));
#else
#error Unsupported system.
#endif
}

/**
 * \note On UNIX, this method accesses the file system.
 *
 * Retrieves the user’s music directory.
 *
 * On UNIX, this is $XDG_MUSIC_DIR, defaulting to `~/Music`.
 * Note you likely will receive a localised version (like "Musik"
 * on a German Linux).
 *
 * On Windows, this defaults to `C:/users/username/Music` or a localised
 * version.
 */
Path Path::music_dir()
{
#if defined(_PATHIE_UNIX)
  return Path(get_xdg_userdir_setting("MUSIC"));
#elif defined(_WIN32)
  wchar_t dir[MAX_PATH];
  HRESULT result = SHGetFolderPathW(NULL, CSIDL_MYMUSIC, NULL, SHGFP_TYPE_CURRENT, dir);
  if (result != S_OK)
    throw(Pathie::WindowsHresultError(result));

  return Path(utf16_to_utf8(dir));
#else
#error Unsupported system.
#endif
}

/**
 * \note On UNIX, this method accesses the file system.
 *
 * Retrieves the user’s pictures directory.
 *
 * On UNIX, this is $XDG_PICTURES_DIR, defaulting to `~/Pictures`.
 * Note you likely will receive a localised version (like "Bilder"
 * on a German Linux).
 *
 * On Windows, this defaults to `C:/users/username/Pictures` or a
 * localised version.
 */
Path Path::pictures_dir()
{
#if defined(_PATHIE_UNIX)
  return Path(get_xdg_userdir_setting("PICTURES"));
#elif defined(_WIN32)
  wchar_t dir[MAX_PATH];
  HRESULT result = SHGetFolderPathW(NULL, CSIDL_MYPICTURES, NULL, SHGFP_TYPE_CURRENT, dir);
  if (result != S_OK)
    throw(Pathie::WindowsHresultError(result));

  return Path(utf16_to_utf8(dir));
#else
#error Unsupported system.
#endif
}

/**
 * \note On UNIX, this method accesses the file system.
 *
 * Retrieves the user’s publicshare directory. This directory may
 * be exposed to network access on the local network, though this
 * is not required.
 *
 * On UNIX, this is $XDG_PUBLICSHARE_DIR, defaulting to `~/Public`.
 * Note you likely will receive a localised version (like "Öffentlich"
 * on a German Linux).
 *
 * On Windows, this defaults to `C:/users/username/AppData/Roaming/Microsoft/Windows/Network Shortcuts`.
 */
Path Path::publicshare_dir()
{
#if defined(_PATHIE_UNIX)
  return Path(get_xdg_userdir_setting("PUBLICSHARE"));
#elif defined(_WIN32)
  wchar_t dir[MAX_PATH];
  HRESULT result = SHGetFolderPathW(NULL, CSIDL_NETHOOD, NULL, SHGFP_TYPE_CURRENT, dir);
  if (result != S_OK)
    throw(Pathie::WindowsHresultError(result));

  return Path(utf16_to_utf8(dir));
#else
#error Unsupported system.
#endif
}

/**
 * \note On UNIX, this method accesses the file system.
 *
 * Retrieves the user’s directory for document templates. The files
 * in this directory can generally be accessed by right-clicking
 * in the user’s favourite file manager and selecting "new" followed
 * by the desired file. The file will then be copied from the templates
 * directory into the directory the user works in at the moment.
 *
 * On UNIX, this is $XDG_TEMPLATES_DIR, defaulting to `~/Templates`.
 * Note you likely will receive a localised version (like "Vorlagen"
 * on a German Linux).
 *
 * On Windows, this defaults to `C:/users/username/AppData/Roaming/Microsoft/Windows/Templates`.
 */
Path Path::templates_dir()
{
#if defined(_PATHIE_UNIX)
  return Path(get_xdg_userdir_setting("TEMPLATES"));
#elif defined(_WIN32)
  wchar_t dir[MAX_PATH];
  HRESULT result = SHGetFolderPathW(NULL, CSIDL_TEMPLATES, NULL, SHGFP_TYPE_CURRENT, dir);
  if (result != S_OK)
    throw(Pathie::WindowsHresultError(result));

  return Path(utf16_to_utf8(dir));
#else
#error Unsupported system.
#endif
}

/**
 * \note On UNIX, this method accesses the file system.
 *
 * Retrieves the user’s directory for videos.
 *
 * On UNIX, this is $XDG_VIDEOS_DIR, defaulting to `~/Videos`
 * or a localised version.
 *
 * On Windows, this defaults to `C:/users/username/Videos` or a
 * localised version.
 */
Path Path::videos_dir()
{
#if defined(_PATHIE_UNIX)
  return Path(get_xdg_userdir_setting("VIDEOS"));
#elif defined(_WIN32)
  wchar_t dir[MAX_PATH];
  HRESULT result = SHGetFolderPathW(NULL, CSIDL_MYVIDEO, NULL, SHGFP_TYPE_CURRENT, dir);
  if (result != S_OK)
    throw(Pathie::WindowsHresultError(result));

  return Path(utf16_to_utf8(dir));
#else
#error Unsupported system.
#endif
}

/**
 * \note On UNIX, this method accesses the file system.
 *
 * Retrieves the user’s path for application starters. On UNIX,
 * this will return a directory (typically `~/.local/share/applications`)
 * where you can store XDG `.desktop` files in so they get picked up
 * by the desktop environment’s application menu for that user. On Windows,
 * the user’s startmenu folder is returned, and any files and directories
 * you add there will show up in the user’s startmenu.
 *
 * \remark On Windows, this is not the global startmenu folder, but the
 * user’s specific ones. Other users will not have the entries you put
 * here in their startmenu.
 */
Path Path::appentries_dir()
{
#if defined(_PATHIE_UNIX)
  return data_dir().join("applications");
#elif defined(_WIN32)
  wchar_t dir[MAX_PATH];
  HRESULT result = SHGetFolderPathW(NULL, CSIDL_STARTMENU, NULL, SHGFP_TYPE_CURRENT, dir);
  if (result != S_OK)
    throw(Pathie::WindowsHresultError(result));

  return Path(utf16_to_utf8(dir));
#else
#error Unsupported system.
#endif
}

///@}

/** \name Global data directories
 *
 * Directories that contain data either unrelated to users at all,
 * or applicable to all users at once. Be careful to read the
 * Windows notes in the documentation of these methods, as Windows
 * only supplies are much smaller set of system directories than UNIX.
 */
///@{

/**
 * Retrieves the global directory for application starters. On UNIX,
 * any XDG `.desktop` files you place there should show up in any user’s
 * desktop environment’s menu, and on Windows, anything you place there
 * should show up in any user’s startmenu.
 *
 * \param local (true) If true, this method returns the location
 * under the `/usr/local` hierarchy, otherwise it returns the
 * location under the `/usr` hierarchy. This parameter has no meaning
 * on Windows and is ignored.
 */
Path Path::global_appentries_dir(localpathtype local)
{
#if defined(_PATHIE_UNIX)
  if (local == Path::LOCALPATH_LOCAL || (local == Path::LOCALPATH_DEFAULT && get_global_dir_default() == Path::LOCALPATH_LOCAL))
    return Path("/usr/local/share/applications");
  else
    return Path("/usr/share/applications");
#elif defined(_WIN32)
  wchar_t dir[MAX_PATH];
  HRESULT result = SHGetFolderPathW(NULL, CSIDL_COMMON_STARTMENU, NULL, SHGFP_TYPE_CURRENT, dir);
  if (result != S_OK)
    throw(Pathie::WindowsHresultError(result));

  return Path(utf16_to_utf8(dir));
#else
#error Unsupported system.
#endif
  local; // make compiler happy
}

/**
 * Retrieves the directory for immutable application data that isn’t user-specific,
 * i.e. which shall be available to all users using the system.
 *
 * On UNIX, this is `/usr/share`. On Windows, this is `C:/Windows/system32`.
 * On Windows, beware conflicts with files of the same name in
 * global_config_dir()!
 *
 * \param local (true) If true, this method returns the location
 * under the `/usr/local` hierarchy, otherwise it returns the
 * location under the `/usr` hierarchy. This parameter has no meaning
 * under Windows and is ignored.
 */
Path Path::global_immutable_data_dir(localpathtype local)
{
#if defined(_PATHIE_UNIX)
  if (local == Path::LOCALPATH_LOCAL || (local == Path::LOCALPATH_DEFAULT && get_global_dir_default() == Path::LOCALPATH_LOCAL))
    return Path("/usr/local/share");
  else
    return Path("/usr/share");
#elif defined(_WIN32)
  wchar_t dir[MAX_PATH];
  HRESULT result = SHGetFolderPathW(NULL, CSIDL_SYSTEM, NULL, SHGFP_TYPE_CURRENT, dir);
  if (result != S_OK)
    throw(Pathie::WindowsHresultError(result));

  return Path(utf16_to_utf8(dir));
#else
#error Unsupported system.
#endif
  local; // make compiler happy
}

/**
 * Retrieves the directory for mutable application data that isn’t user-specific,
 * i.e. which shall be available to all users using the system.
 *
 * On UNIX, this is `/var/lib`. On Windows, this is the Application Data folder
 * for the "All Users" account. On Windows, this is equivalent to global_cache_dir(),
 * so beware file name conflicts on Windows!
 *
 * \param local (true) If true, this method returns the location
 * under the `/var/local` hierarchy, otherwise it returns the
 * location under the `/var` hierarchy. This parameter has no meaning
 * under Windows and is ignored.
 */
Path Path::global_mutable_data_dir(localpathtype local)
{
#if defined(_PATHIE_UNIX)
  if (local == Path::LOCALPATH_LOCAL || (local == Path::LOCALPATH_DEFAULT && get_global_dir_default() == Path::LOCALPATH_LOCAL))
    return Path("/var/local/lib");
  else
    return Path("/var/lib");
#elif defined (_WIN32)
  wchar_t dir[MAX_PATH];
  HRESULT result = SHGetFolderPathW(NULL, CSIDL_COMMON_APPDATA, NULL, SHGFP_TYPE_CURRENT, dir);
  if (result != S_OK)
    throw(Pathie::WindowsHresultError(result));

  return Path(utf16_to_utf8(dir));
#else
#error Unsupported system
#endif
  local; // make compiler happy
}

/**
 * Retrieves the directory for global cache data, i.e. data, which
 * is not essential to the program and can be reconstructed if it
 * gets lost.
 *
 * On UNIX, this returns `/var/cache`. Windows does not have a notion
 * of such a directory, hence the value is equal to the return value
 * of global_mutable_data_dir(). Therefore: On Windows, beware conflicts if you
 * use files of the same name in global_mutable_data_dir() and
 * global_cache_dir()!
 *
 * \param local (true) If true, returns the cache directory for locally installed
 * programs, which is `/var/local/cache`. This parameter has no effect under
 * systems other than UNIX.
 */
Path Path::global_cache_dir(localpathtype local)
{
#if defined(_PATHIE_UNIX)
 if (local == Path::LOCALPATH_LOCAL || (local == Path::LOCALPATH_DEFAULT && get_global_dir_default() == Path::LOCALPATH_LOCAL))
    return Path("/var/local/cache");
  else
    return Path("/var/cache");
#elif defined(_WIN32)
  return global_mutable_data_dir();
#else
#error Unsupported system.
#endif
  local; // make compiler happy
}

/**
 * \note On UNIX, this method accesses the filesystem.
 *
 * Returns the directory for volatile information that will be deleted
 * on system shutdown.
 *
 * On UNIX, this returns `/run` if it exists, otherwise `/var/run`.
 * Windows does not have a notion of such a directory; as a replacement,
 * `C:/Temp` is returned.
 *
 * \param local (true) If true, returns the equivalent directory for
 * `/run` for locally installed programs, which is `/var/local/run`. This
 * parameter has no effect on systems other than UNIX.
 */
Path Path::global_runtime_dir(localpathtype local)
{
#if defined(_PATHIE_UNIX)
  if (local == Path::LOCALPATH_LOCAL || (local == Path::LOCALPATH_DEFAULT && get_global_dir_default() == Path::LOCALPATH_LOCAL))
    return Path("/var/local/run");

  Path run("/run");
  if (run.exists())
    return run;
  else
    return Path("/var/run");
#elif defined(_WIN32)
  return Path("C:/Temp");
#else
#error Unsupported system.
#endif
  local; // make compiler happy
}

/**
 * Returns the global directory for configuration files.
 *
 * On UNIX, this is `/etc`. Windows does not really have a notion
 * for configuration directories. This method returns the Windows
 * system folder for that purpose, typically `C:/Windows/system32`;
 * this is equivalent to global_immutable_data_dir(), so be careful
 * when you place files of the same name in global_config_dir()!
 *
 * \param local (true) If true, returns the global configuration
 * directory for locally installed programs instead, which is
 * `/usr/local/etc`.
 */
Path Path::global_config_dir(localpathtype local)
{
#if defined(_PATHIE_UNIX)
  if (local == Path::LOCALPATH_LOCAL || (local == Path::LOCALPATH_DEFAULT && get_global_dir_default() == Path::LOCALPATH_LOCAL))
    return Path("/usr/local/etc");
  else
    return Path("/etc");

#elif defined(_WIN32)
  wchar_t dir[MAX_PATH];
  HRESULT result = SHGetFolderPathW(NULL, CSIDL_SYSTEM, NULL, SHGFP_TYPE_CURRENT, dir);
  if (result != S_OK)
    throw(Pathie::WindowsHresultError(result));

  return Path(utf16_to_utf8(dir));
#else
#error Unsupported system.
#endif
  local; // make compiler happy
}

/**
 * Retrieves the global directory for self-contained applications, i.e.
 * applications that require a directory structure different from the
 * Filesystem Hierarchy Standard (FHS). Such programs are an exception
 * under UNIX, but are the regular case on Windows. The programs placed
 * in this directory are intended to be available to all users using the
 * system.
 *
 * Under UNIX, this method returns the `/opt` directory. On Windows,
 * it returns the Program Files directory (typically `C:\Program Files`).
 *
 * \note On UNIX, the FHS mandates that programs installed under
 * `/opt` do not use the usual directories for variable information
 * returned by global_mutable_data_dir() and global_cache_dir(), but
 * instead use `/var/opt`.
 */
Path Path::global_programs_dir()
{
#if defined(_PATHIE_UNIX)
  return Path("/opt");
#elif defined(_WIN32)
  wchar_t dir[MAX_PATH];
  HRESULT result = SHGetFolderPathW(NULL, CSIDL_PROGRAM_FILES, NULL, SHGFP_TYPE_CURRENT, dir);
  if (result != S_OK)
    throw(Pathie::WindowsHresultError(result));

  return Path(utf16_to_utf8(dir));

#else
#error Unsupported system.
#endif
}

///@}

/** \name Miscellaneous static functions
 *
 * Other functions that didn’t fit somewhere else.
 */
///@{

/// \note This method accesses the filesystem.
///
/// Uses a shell-like glob pattern on the current working directory.
/// Typically available patterns include "*" for a string of
/// arbitrary length and "?" for a string of length one.
///
/// Refer to glob(7) for glob patterns available on UNIX.
/// Refer to [MSDN](http://msdn.microsoft.com/en-us/library/windows/desktop/aa364418%28v=vs.85%29.aspx)
/// for glob patterns available on Windows.
///
/// Windows does not support recursive patterns like
///
/// \verbatim **/* \endverbatim
///
/// or
///
/// \verbatim foo/*/bar \endverbatim
///
/// . This will result in a Pathie::WindowsError exception
/// with Windows error code 123 (“invalid filename”). For cross-platform
/// recursive matching, you can try to combine find() and fnmatch().
///
/// \param[in] pattern Glob pattern.
/// \param flags (`0`) Globbing flags. Refer to glob(3) for
/// possible values; the parameter is ignored on Windows.
///
/// \returns A vector of Path instances that matched the glob
/// pattern.
///
/// \remark Glob patterns on UNIX are generally much more powerful than
/// those on Windows. Be careful when using anything apart from "*" and "?"
/// patterns on Windows.
///
/// \see dglob() fnmatch()
///
std::vector<Path> Path::glob(const std::string& pattern, int flags /* = 0 */)
{
#if defined(_PATHIE_UNIX)
  std::string nstr = utf8_to_filename(pattern);
  glob_t globinfo;
  int result = ::glob(nstr.c_str(), flags, NULL, &globinfo);

  if (result == GLOB_NOMATCH) {
    return std::vector<Path>(); // Empty vector
  }
  else if (result == 0) {
    std::vector<Path> result;

    for(size_t i=0; i < globinfo.gl_pathc; i++) {
      result.push_back(Path(filename_to_utf8(globinfo.gl_pathv[i])));
    }

    globfree(&globinfo);
    return result;
  }
  else {
    throw(GlobError(result));
  }
#elif defined(_WIN32)
  std::vector<Path> results;
  std::wstring utf16_pattern = utf8_to_utf16(pattern);

  /* Windows’ FindFirstFile()/FindNextFile() returns bare file names.
   * However, to ensure output similar to the UNIX version, we prepend
   * the pattern’s stem if a slash / is found in the pattern; FindFirstFile()/
   * FindNextFile() don’t support recursive matching anyway, so this is safe. */
  std::string stem;
  size_t pos = 0;
  if ((pos = pattern.rfind("/")) != string::npos) // Single = intended
    stem = pattern.substr(0, pos + 1); // Trailing / included

  // Prepare
  HANDLE filehandle = INVALID_HANDLE_VALUE;
  WIN32_FIND_DATAW finddata;
  memset(&finddata, '\0', sizeof(WIN32_FIND_DATA));

  // Try finding the first file
  filehandle = FindFirstFileW(utf16_pattern.c_str(), &finddata);

  // Check if some error happened
  if (filehandle == INVALID_HANDLE_VALUE) {
    DWORD errval = GetLastError();
    if (errval == ERROR_FILE_NOT_FOUND) // According to docs, this means no matching files were found. Return empty list.
      return results;
    else if (errval != ERROR_SUCCESS)
      throw Pathie::WindowsError(errval);
  }

  // All well, save this one...
  results.push_back(Path(stem + utf16_to_utf8(finddata.cFileName)));

  // ...and continue.
  while (FindNextFileW(filehandle, &finddata)) {
    results.push_back(Path(stem + utf16_to_utf8(finddata.cFileName)));
  }

  DWORD errval = GetLastError();
  FindClose(filehandle);

  if (errval != ERROR_NO_MORE_FILES)
    throw(Pathie::WindowsError(errval));

  return results;
#else
#error Unsupported system.
#endif
  flags; // make compiler happy
}

///@}

/** \name Miscellaneous member functions
 *
 * Methods that didn’t fit anywhere else.
 */

///@{

/**
 * This method tests whether the referenced path matches the
 * given pattern under the rules of the local glob-matching
 * function. Note this method does _not_ access the filesystem,
 * hence there is no guarantee that the referenced path exists.
 *
 * \param[in] pattern The pattern to match.
 * \param flags Any flags. This parameter is ignored on Windows,
 * for UNIX refer to the fnmatch(3) manpage.
 *
 * \returns Whether the path matches the pattern.
 *
 * \remark On Windows, this method uses the [PathMatchSpec()](http://msdn.microsoft.com/en-us/library/bb773727%28VS.85%29.aspx)
 * function; on UNIX, it uses fnmatch(3).
 *
 * \remark Windows’s `PathMatchSpec()` function does not support
 * recursive matching patterns, while the UNIX fnmatch(8), relying
 * on glob(7), does.
 *
 * \remark Glob patterns on UNIX are generally much more powerful than
 * those on Windows. Be careful when using anything apart from "*" and "?"
 * patterns on Windows.
 *
 * \see glob() dglob()
 */
bool Path::fnmatch(const std::string& pattern, int flags /* = 0 */) const
{
#if defined(_PATHIE_UNIX)
  std::string nstr = native();
  std::string pattern_nstr = utf8_to_filename(pattern);
  return ::fnmatch(pattern_nstr.c_str(), nstr.c_str(), flags) == 0;
#elif defined(_WIN32)
  std::wstring utf16path = utf8_to_utf16(m_path);
  std::wstring utf16pattern = utf8_to_utf16(pattern);
  return PathMatchSpecW(utf16path.c_str(), utf16pattern.c_str()) != 0;
#else
#error Unsupported system.
#endif
  flags; // make compiler happy
}

/**
 * \note This method acceses the filesystem.
 *
 * Like glob(), but prepends the referenced path to the glob
 * pattern.
 *
 * \see glob() fnmatch()
 */
std::vector<Path> Path::dglob(const std::string& pattern, int flags /* = 0 */) const
{
  return glob(m_path + "/" + pattern, flags);
}

/**
 * Appends a /, then the new component, and
 * finally returns a new Path instance.
 *
 * \param path New component.
 *
 * \returns New Path instance.
 */
Path Path::join(Path path) const
{
  Path p(m_path + "/" + path.str());
  return p;
}

/**
 * Appends a /, then the new component, and
 * finally returns a new Path instance.
 *
 * \param str New component.
 *
 * \returns New Path instance.
 */
Path Path::join(std::string str) const
{
  Path path(m_path + "/" + str);
  return path;
}

/**
 * Replaces the current extension with the given new extension
 * and returns the result. If the referenced path doesn’t have
 * a file extension currently, the new extension is appended.
 *
 * \param new_extension The new extension. If the leading point
 * is missing, it will automatically be prepended.
 *
 * \returns The new Path instance.
 */
Path Path::sub_ext(std::string new_extension) const
{
  // If the point is missing, add it to the beginning.
  if (new_extension.find(".") == string::npos)
    new_extension.insert(0, ".");

  std::string old_extension = extension();
  if (old_extension.empty()) {
    return Path(m_path + new_extension);
  }
  else {
    size_t pos = m_path.find(old_extension);
    return Path(m_path.substr(0, pos) + new_extension);
  }
}

///@}
