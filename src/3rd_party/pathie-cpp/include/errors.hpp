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

#ifndef PATHIE_ERRORS_HPP
#define PATHIE_ERRORS_HPP
#include <exception>
#include <string>
#include <cstdlib>

/* DWORD is typedef'ed from unsigned long, see
 * <https://msdn.microsoft.com/en-us/library/cc230318.aspx>
 * HRESULT is typedef'ed from LONG, which in turn is a typedef
 * of long, see <https://msdn.microsoft.com/en-us/library/cc230330.aspx>.
 * I spell the types out here in this header to avoid having to
 * include windows.h, which might interfer with programmes using
 * pathie that want to include windows.h on itself. */

#include "pathie.hpp"

namespace Pathie {

  /// Base class for all exceptions in this library.
  class PathieError: public std::exception {
  public:
    PathieError(); ///< Constructs a new instance.
    PathieError(std::string message); ///< Contructs a new instance with the given what() message.
    virtual ~PathieError() throw();

    virtual const char* what() const throw(); ///< The error message.
  protected:
    std::string m_pathie_errmsg; ///< The error message given in the constructor.
  };


  /// This exception is thrown when a call to a C/system function results
  /// in `errno` being set.
  class ErrnoError: public PathieError {
  public:
    ErrnoError(int val); ///< Constructs a new instance from the given `errno` value.
    virtual ~ErrnoError() throw();

    inline int get_val(){return m_val;} ///< The `errno` value.
  private:
    int m_val;
  };

#ifdef _WIN32

  /// This exception is thrown only on Windows, when a call to the Win32API
  /// fails.
  /// The "unsigned long" type here is actually DWORD (which is it a
  /// typedef of in Win32).
  class WindowsError: public PathieError {
  public:
    WindowsError(unsigned long val); ///< Constructs a new instance from the given GetLastError() value.
    virtual ~WindowsError() throw();

    inline int get_val(){return m_val;} ///< The GetLastError() value.
  private:
    unsigned long m_val;
  };

  /// Similar to WindowsError, this exception is thrown when a HANDLE function
  /// from the Win32API fails.
  /// The "long" type here is actually HRESULT (which it is a typedef of in Win32).
  class WindowsHresultError: public PathieError {
  public:
    WindowsHresultError(long value); ///< Constructs a new instance from the given handle function result.
    virtual ~WindowsHresultError() throw();

    inline long get_val(){return m_val;} ///< The handle function result.
  private:
    int m_val;
  };
#endif

#ifdef _PATHIE_UNIX

  /// This exception is thrown only on UNIX, when a call to the POSIX glob(3)
  /// function fails.
  class GlobError: public PathieError {
  public:
    GlobError(int val); ///< Contructs a new instance from the given glob(3) error code.
    virtual ~GlobError() throw();

    inline int get_val(){return m_val;} ///< The glob(3) error code.
  private:
    int m_val;
  };
#endif

}
#endif
