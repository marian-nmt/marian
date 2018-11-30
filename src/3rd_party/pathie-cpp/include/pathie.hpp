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

#ifndef PATHIE_PATHIE_HPP
#define PATHIE_PATHIE_HPP
#if __cplusplus < 199711L
#error Pathie requires C++98 support. Please use an option such as -std=c++98 to enable it.
#endif

#if !defined(_PATHIE_UNIX) && (defined(unix) || defined(__unix__) || defined(__unix) || defined(__APPLE__) || defined(BSD))
#define _PATHIE_UNIX
#endif

#include <string>

/// Namespace for this library.
namespace Pathie {

  /// Returns the version number is MAJOR.MINOR.TINY.
  std::string version();

  /**
   * Returns the Git commit this was build from.
   * Empty string if build without Git.
   */
  std::string gitrevision();

#ifdef _WIN32
  std::string utf16_to_utf8(std::wstring);
  std::wstring utf8_to_utf16(std::string);
#endif

#ifdef _PATHIE_UNIX
  std::string utf8_to_filename(const std::string& utf8);
  std::string filename_to_utf8(const std::string& native_filename);
  std::string convert_encodings(const char* from_encoding, const char* to_encoding, const std::string& string);
#endif

}

#endif
