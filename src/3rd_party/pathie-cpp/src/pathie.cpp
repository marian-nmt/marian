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

#include "../include/pathie.hpp"
#include "../include/errors.hpp"

#if defined(_WIN32)
#include <windows.h>

/**
 * Converts a UTF-16LE string into UTF-8. Only available
 * on Windows.
 */
std::string Pathie::utf16_to_utf8(std::wstring str)
{
  int size = WideCharToMultiByte(CP_UTF8, 0, str.c_str(), (int)str.length(), NULL, 0, NULL, NULL);

  char* utf8 = (char*) malloc(size); // sizeof(char) = 1 per ANSI C standard.
  memset(utf8, 0, size);

  size = WideCharToMultiByte(CP_UTF8, 0, str.c_str(), (int)str.length(), utf8, size,  NULL, NULL);

  if (size == 0)
    throw(Pathie::WindowsError(GetLastError()));

  std::string utf8str(utf8, size);
  free(utf8);

  return utf8str;
}

/**
 * Converts a UTF-8 string into UTF-16LE. Only available
 * on Windows.
 */
std::wstring Pathie::utf8_to_utf16(std::string str)
{
  int count = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.length(), NULL, 0);

  wchar_t* utf16 = (wchar_t*) malloc(count * sizeof(wchar_t));
  memset(utf16, 0, count * sizeof(wchar_t));

  count = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.length(), utf16, count);

  if (count == 0)
    throw(Pathie::WindowsError(GetLastError()));

  std::wstring utf16str(utf16, count);
  free(utf16);

  return utf16str;
}
#endif

#ifdef _PATHIE_UNIX
#include <cstring>
#include <cstdlib>
#include <errno.h>
#include <iconv.h>
#include <langinfo.h>
#include <sys/param.h> // defines "BSD" macro on BSD systems

/* iconv() function family is available on every POSIX-conformant
 * system. In POSIX.1-2008, it’s specified in the "System Interfaces"
 * section.
 *
 * nl_langinfo() is also specified by POSIX, though I’ve found no evidence
 * that iconv() is required to understand the encoding output by nl_langinfo(CODESET).
 * From checking on Linux and FreeBSD, this however seems very likely, so we have
 * to assume that this always is the case.
 */

/**
 * This function converts the given string from the given source encoding
 * to another given target encoding and returns the result as a std::string.
 *
 * \param[in] from_encoding Convert from this encoding.
 * \param[in] to_encoding Convert into this encoding.
 * \param[in] string The string to convert.
 *
 * \returns The converted string.
 *
 * \remark See the output of the `iconv --list` command for a list of
 * supported encodings.
 */
std::string Pathie::convert_encodings(const char* from_encoding, const char* to_encoding, const std::string& string)
{
  size_t input_length = string.length();

  // We need a C string working copy that isn’t const
  char* copy = (char*) malloc(input_length + 1); // Terminating NUL
  strcpy(copy, string.c_str());

  // Set up the encoding converter
  iconv_t converter    = iconv_open(to_encoding, from_encoding);
  size_t outbytes_left = 0;
  size_t inbytes_left  = input_length;

  if (converter == (iconv_t) -1)
    throw Pathie::ErrnoError(errno);

  /* There is no way to know how much space iconv() will need. So we keep
   * allocating more and more memory as needed. `current_size' keeps track
   * of how large our memory blob is currently. `outbuf' is the pointer to
   * that memory blob. */
  size_t current_size = input_length + 1; // NUL
  char* outbuf        = NULL;
  char* inbuf         = copy; // Copy the pointer

  int errsav = 0;
  outbytes_left = current_size;
  while(true) {
    outbuf         = (char*) realloc(outbuf - (current_size - outbytes_left), current_size + 10);
    current_size  += 10;
    outbytes_left += 10;

    errno  = 0;
    errsav = 0;

#if defined(BSD) && ! defined(__APPLE__) //Since MacOS evolved from BSD, it is captured here but the iconv on macos behaves differently
    // What the heck. FreeBSD violates POSIX.1-2008: it declares iconv()
    // differently than mandated by POSIX: http://pubs.opengroup.org/onlinepubs/9699919799/functions/iconv.html
    // (it declares a `const' where it must not be).
    iconv(converter, const_cast<const char**>(&inbuf), &inbytes_left, &outbuf, &outbytes_left); // sets outbytes_left to 0 or very low values if not enough space (E2BIG)
#else
    iconv(converter, &inbuf, &inbytes_left, &outbuf, &outbytes_left); // sets outbytes_left to 0 or very low values if not enough space (E2BIG)
#endif
    errsav = errno;

    if (errsav != E2BIG) {
      break;
    }
  }

  iconv_close(converter);
  free(copy);

  size_t count = current_size - outbytes_left;
  outbuf -= count; // iconv() advances the pointer!

  if (errsav != 0) {
    free(outbuf);
    throw(Pathie::ErrnoError(errsav));
  }

  std::string result(outbuf, count);
  free(outbuf);

  return result;
}

/**
 * Converts the given UTF-8 string into the native filename encoding.
 */
std::string Pathie::utf8_to_filename(const std::string& utf8)
{
  bool fs_encoding_is_utf8 = false;
  char* fsencoding = NULL;
#if defined(__APPLE__) || defined(PATHIE_ASSUME_UTF8_ON_UNIX)
  fs_encoding_is_utf8 = true;
#else
  fsencoding = nl_langinfo(CODESET);
  fs_encoding_is_utf8 = (strcmp(fsencoding, "UTF-8") == 0);
#endif

  // Skip the expensive convert_encodings() call if the filesystem
  // encoding already is UTF-8.
  if (fs_encoding_is_utf8) {
    return std::string(utf8);
  }

  return convert_encodings("UTF-8", fsencoding, utf8);
}

/**
 * Converts the given string in native filesystem encoding to
 * UTF-8.
 */
std::string Pathie::filename_to_utf8(const std::string& native_filename)
{
  bool fs_encoding_is_utf8 = false;
  char* fsencoding = NULL;
#if defined(__APPLE__) || defined(PATHIE_ASSUME_UTF8_ON_UNIX)
  fs_encoding_is_utf8 = true;
#else
  fsencoding = nl_langinfo(CODESET);
  fs_encoding_is_utf8 = (strcmp(fsencoding, "UTF-8") == 0);
#endif

  // Skip the expensive convert_encodings() call if the filesystem
  // encoding already is UTF-8.
  if (fs_encoding_is_utf8) {
    return std::string(native_filename);
  }

  return convert_encodings(fsencoding, "UTF-8", native_filename);
}
#endif
