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

#include "../include/errors.hpp"

#include <cerrno>
#include <cstring>
#include <sstream>

#if defined(_WIN32)
#include <windows.h>
#elif defined(_PATHIE_UNIX)
#include <glob.h>
#endif

using namespace Pathie;

PathieError::PathieError()
{
  m_pathie_errmsg = "Unknown pathie exception.";
}

PathieError::PathieError(std::string message)
{
  m_pathie_errmsg = message;
}

PathieError::~PathieError() throw()
{
  //
}

const char* PathieError::what() const throw()
{
  return m_pathie_errmsg.c_str();
}

ErrnoError::ErrnoError(int val)
{
  std::stringstream ss;
  ss << val;

  m_val = val;
  m_pathie_errmsg = "Errno " + ss.str() + ": " + strerror(val);
}

ErrnoError::~ErrnoError() throw()
{
  //
}

#ifdef _WIN32
WindowsError::WindowsError(DWORD val)
{
  std::stringstream ss;
  ss << val;

  wchar_t* buf = NULL;
  FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                 NULL,
                 val,
                 LANG_USER_DEFAULT,
                 (wchar_t*) &buf, // What a weird API.
                 0,
                 NULL);

  m_val = val;
  m_pathie_errmsg = std::string("Windows Error Code ") + ss.str() + ": " + utf16_to_utf8(buf);

  LocalFree(buf);
}

WindowsError::~WindowsError() throw()
{
  //
}

WindowsHresultError::WindowsHresultError(HRESULT val)
{
  std::stringstream ss;
  ss << val;

  m_val = val;
  m_pathie_errmsg = std::string("Windows HRESULT Error Code :") + ss.str();
}

WindowsHresultError::~WindowsHresultError() throw()
{
  //
}

#endif

#ifdef _PATHIE_UNIX
GlobError::GlobError(int val)
{
  std::stringstream ss;
  ss << val;

  m_val = val;

  m_pathie_errmsg = "Glob error code " + ss.str() + ": ";

  switch(val) {
  case GLOB_NOSPACE:
    m_pathie_errmsg += "GLOB_NOSPACE";
    break;
  case GLOB_ABORTED:
    m_pathie_errmsg += "GLOB_ABORTED";
    break;
  case GLOB_NOMATCH:
    m_pathie_errmsg += "GLOB_NOMATCH";
    break;
  default:
    m_pathie_errmsg += "Unknown glob error";
    break;
  }
}

GlobError::~GlobError() throw()
{
  //
}
#endif
