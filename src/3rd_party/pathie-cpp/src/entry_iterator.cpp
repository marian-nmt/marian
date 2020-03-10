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

#include "../include/entry_iterator.hpp"
#include "../include/path.hpp"
#include "../include/errors.hpp"

#if defined(__unix__) || defined(__APPLE__)
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <stdexcept>
#elif defined(_WIN32)
#include <Windows.h>
#else
#error Unsupported system
#endif

using namespace Pathie;

/**
 * The default constructor always constructs the terminal
 * iterator, i.e. the one you want to test for if you want
 * to know whether an iteration has completed.
 */
entry_iterator::entry_iterator()
  : mp_directory(NULL),
    mp_cur(NULL),
    mp_cur_path(new Path())
{
}

/**
 * Construct an iterator that reads the entries in the given directory.
 */
entry_iterator::entry_iterator(const Path* p_directory)
  : mp_directory(p_directory),
    mp_cur(NULL),
    mp_cur_path(new Path())
{
  open_native_handle();
}

/**
 * Destructor. Closes the open native handle, if it is open.
 */
entry_iterator::~entry_iterator()
{
  close_native_handle();

  if (mp_cur_path)
    delete mp_cur_path;

  // `mp_directory' is NOT deleted, because this class does not own it!
}

/**
 * Opens the native handle to the directory and reads the first
 * entry from the directory.
 */
void entry_iterator::open_native_handle()
{
#if defined(_PATHIE_UNIX)
  std::string nstr = mp_directory->native();
  mp_cur = opendir(nstr.c_str());

  if (mp_cur) {
    struct dirent* p_dirent = readdir(static_cast<DIR*>(mp_cur));
    *mp_cur_path = filename_to_utf8(p_dirent->d_name);
  }
  else {
    throw(Pathie::ErrnoError(errno));
  }
#elif defined(_WIN32)
  std::wstring utf16 = utf8_to_utf16(mp_directory->str() + "/*");
  WIN32_FIND_DATAW finddata;

  mp_cur = FindFirstFileW(utf16.c_str(), &finddata);
  if (static_cast<HANDLE>(mp_cur) == INVALID_HANDLE_VALUE) {
    DWORD err = GetLastError();
    mp_cur = NULL;
    throw(Pathie::WindowsError(err));
  }
  else {
    *mp_cur_path = utf16_to_utf8(finddata.cFileName);
  }
#else
#error Unsupported system
#endif
}

/// Helper function for closing the native handle.
void entry_iterator::close_native_handle()
{
  if (!mp_cur)
    return;

#if defined(_PATHIE_UNIX)
  closedir(static_cast<DIR*>(mp_cur));
#elif defined(_WIN32)
  FindClose(static_cast<HANDLE>(mp_cur));
#endif

  // Reset member variables
  *mp_cur_path = Path();
  mp_cur = NULL;
}

/**
 * Increment operator. Calling this advances the iterator by one,
 * thus pointing it to the next entry. If the end is reached,
 * the iterator will compare equal to the return value of the
 * default constructor, and dereferencing it yields an undefined
 * result.
 *
 * \remark Note that this operator does *not* return the old value
 * the iterator had, simply because that would mean copying the
 * receiver first, and copying instances of this class is not
 * possible. Thus, *do not rely* on the return value of this
 * method.
 */
entry_iterator& entry_iterator::operator++(int)
{
  if (mp_cur) {
#if defined(_PATHIE_UNIX)
    struct dirent* p_dirent = readdir(static_cast<DIR*>(mp_cur));
    if (p_dirent) {
      *mp_cur_path = filename_to_utf8(p_dirent->d_name);
    }
    else {
      close_native_handle();
    }
#elif defined(_WIN32)
    WIN32_FIND_DATAW finddata;
    if (FindNextFileW(static_cast<HANDLE>(mp_cur), &finddata)) {
      *mp_cur_path = utf16_to_utf8(finddata.cFileName);
    }
    else {
      close_native_handle();
    }
#else
#error Unsupported system
#endif
  }
  else { // Finished already
    throw(std::range_error("Tried to advance a finished entry_iterator!"));
  }

  return *this;
}

/// Same as the other operator++().
entry_iterator& entry_iterator::operator++()
{
  return (operator++(0));
}

/**
 * Derefence operator. Returns the entry the iterator currently
 * points at.
 */
const Path& entry_iterator::operator*() const
{
  return *mp_cur_path;
}

/**
 * Resets this iterator to start again on the path given.
 */
entry_iterator& entry_iterator::operator=(const Path* p_directory)
{
  close_native_handle();
  mp_directory = p_directory;
  open_native_handle();
  return *this;
}

/**
 * Boolean operator. In comparisons, this iterator is true if
 * it has not yet finished, false otherwise.
 */
entry_iterator::operator bool() const
{
  return !!mp_directory;
}

/**
 * Equality test. Two instances of this class are equal if:
 *
 * 1. If `other` is a terminal iterator as created by the parameterless
 *    constructor: if the receiver has finished iterating the directory.
 * 2. If `other` is not a terminal iterator as described: if both
 *    iterators refer to the same top directory and their current
 *    native handle is the same and in the same state (hint: this
 *    is not going to happen under normal circumstances).
 */
bool entry_iterator::operator==(const entry_iterator& other) const
{
  if (other.mp_directory == NULL) {
    /* `mp_directory' is only null for the terminal iterator, that is,
     * a test for the terminal iterator was requested. An entry_iterator
     * is terminated when `mp_cur' is null, so that's what is returned
     * in reality when a test with the terminal iterator is
     * requested. */
    return !mp_cur;
  }
  else {
    return mp_directory == other.mp_directory && mp_cur == other.mp_cur;
  }
}

/// Inverse of operator==().
bool entry_iterator::operator!=(const entry_iterator& other) const
{
  return !(*this == other);
}

/**
 * Derefence operator. Returns the entry the iterator currently
 * points at.
 */
const Path* entry_iterator::operator->() const
{
  return mp_cur_path;
}

/// "Copy" constructor -- see class docs for more info.
entry_iterator::entry_iterator(const entry_iterator& other)
  : mp_directory(other.mp_directory),
    mp_cur(other.mp_cur),
    mp_cur_path(other.mp_cur_path)
{
  entry_iterator& e = const_cast<entry_iterator&>(other);
  e.mp_directory    = NULL;
  e.mp_cur          = NULL;
  e.mp_cur_path     = new Path();
}

/// "Copy" assignment -- see class docs for more info.
entry_iterator& entry_iterator::operator=(const entry_iterator& other)
{
  mp_directory      = other.mp_directory;
  mp_cur            = other.mp_cur;
  mp_cur_path       = other.mp_cur_path;

  entry_iterator& e = const_cast<entry_iterator&>(other);
  e.mp_directory    = NULL;
  e.mp_cur          = NULL;
  e.mp_cur_path     = new Path();

  return *this;
}

