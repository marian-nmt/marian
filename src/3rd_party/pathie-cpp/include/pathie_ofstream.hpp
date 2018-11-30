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

#ifndef PATHIE_OFSTREAM_HPP
#define PATHIE_OFSTREAM_HPP

#if defined(_WIN32) && defined(__GNUC__)
#include <ostream>
#include <ext/stdio_filebuf.h>
#else
#include <fstream>
#endif

#include "path.hpp"

namespace Pathie {

#if defined(_PATHIE_UNIX)
  class ofstream: public std::ofstream {
  public:
    ofstream();
    ofstream(char* path, std::ios_base::openmode = std::ios_base::out);
    ofstream(std::string path, std::ios_base::openmode = std::ios_base::out);
    ofstream(Pathie::Path path, std::ios_base::openmode = std::ios_base::out);

    void open(const char* filename, ios_base::openmode mode = ios_base::out | ios_base::trunc);
    void open(const std::string& filename, ios_base::openmode mode = ios_base::out | ios_base::trunc);
    void open(const Pathie::Path& filename, ios_base::openmode mode = ios_base::out | ios_base::trunc);
  };
#elif defined (_WIN32)
#  if defined(__GNUC__)
  /**
   * \brief Output stream for UTF-8-encoded filenames.
   *
   * Unicode filenames with C++ are horrible, and this is why the Pathie library
   * was written in the first sense. However, working with paths may be nice,
   * but what does this mean for you if you cannot actually open the file
   * whose path you have been manipulating? On UNIX, the `std::ofstream` class
   * will work just as expected if you pass it a UTF-8 unicode filename and it
   * will open exactly the path you specified. Windows however uses UTF-16LE
   * as the encoding for pathnames, and the same code that runs on UNIX will
   * produce garbage filenames on Windows. Take this as an example:
   *
   * ~~~~~~~~~~~~~~~~~ c++
   * std::ofstream file("Bärenstark.txt");
   * file << "Some content" << std::endl;
   * file.close();
   * ~~~~~~~~~~~~~~~~~
   *
   * The file will appear as expected on UNIX, but on Windows it will have
   * a garbage filename because Windows interprets filenames based on the
   * `char` type as in the local encoding (Windows-1252 on a Western European
   * Windows system). You have to use filenames based on `wchar_t` on Windows
   * to get the desired effect. This, however, doesn’t work neither:
   *
   * ~~~~~~~~~~~~~~~~~ c++
   * std::ofstream file(L"Bärenstark.txt");
   * file << "Some content" << std::endl;
   * file.close()
   * ~~~~~~~~~~~~~~~~~
   *
   * That is, it works on the Microsoft Visual C++ Compiler (MSVC). The reason
   * for this is that the ISO C++ standard does not specify a constructor
   * that takes filenames based on `wchar_t`, but only on `char`, which Windows
   * interpretes as described above. That’s a nice proof of how Windows tries
   * to be inherently different from all other modern OSes in this world, and
   * how it makes simple tasks a pain if you want cross-platform behaviour.
   * GCC on Windows, as distributed by the MinGW project, does not support the
   * nonstandard contructor. As it stands, you **cannot** create Unicode files
   * via the standard C++ interface with MinGW GCC. There is, however, a special
   * function in the Windows API called `_wfopen()` that lets you at least open
   * a file via a `fopen()`-like C API. Thankfully GCC provides a (also nonstandard)
   * measure to create a filebuffer (this is what is used by the C++ streams
   * under the hood to access the files) from a C `FILE*`. This class wraps
   * that GNU C++ extension (`gnu_cxx::stdio_filebuf`) on Windows, as well as it wraps
   * the standard stream API on other platforms. It therefore unites the different
   * access methods under a single uniform interface that allows you to
   * create Unicode filenames regardless of the platform you run on.
   *
   * Let’s revisit the previous example, now with Pathie’s streams:
   *
   * ~~~~~~~~~~~~~~~~~ c++
   * Pathie::ofstream file("Bärenstark.txt");
   * file << "Some content" << std::endl;
   * file.close()
   * ~~~~~~~~~~~~~~~~~
   *
   * The `Pathie::ofstream` constructor takes a UTF-8 string and does the
   * necessary conversion to UTF-16, uses `_wfopen()` under the hood to access
   * the file, and then wraps a C++ stream around the already opened file
   * descriptor. On platforms other than MiNGW Windows, the `Pathie::ofstream` class
   * will just delegate to the standard `std::ofstream` class. As a bonus,
   * if you compile with MSVC the nonstandard constructor described above
   * is used.
   *
   * Of course, there’s also a constructor that will make it work directly
   * with instances of Pathie::Path:
   *
   * ~~~~~~~~~~~~~~~~~ c++
   * Pathie::Path p("Bärenstark.txt");
   * Pathie::ofstream file(p);
   * file << "Some content" << std::endl;
   * file.close()
   * ~~~~~~~~~~~~~~~~~
   *
   * That is, you can stay with UTF-8 `char`-based strings (like `std::string`)
   * for anything you use. Ain’t that great?
   *
   * \warning On Windows, this class tries to behave as similar as the standard
   * `std::ofstream` as possible. Due to the file descriptor magic it does under
   * the hood, however, there is a little difference: If you construct an
   * instance of this class without associating it immediately with a filename
   * (the constructor without arguments), using any methods apart from `is_open()`
   * (which is specifically implemented for that purpose) that use the underlying
   * filebuffer will result in segmentation faults, because the filebuffer has
   * not yet been constructed (the area where it will be constructed into is
   * full of NUL bytes if you wonder).
   *
   * \note Please refer to your preferred C++ STL documentation for the
   * `std::ofstream` class for general usage of C++ file streams.
   */
  class ofstream: public std::basic_ostream<char, std::char_traits<char> >
  {
  public:
    typedef char char_type;                          ///< Type used inside the stream.
    typedef std::char_traits<char> traits_type;      ///< Traits type
    typedef typename traits_type::int_type int_type; ///< Int type
    typedef typename traits_type::pos_type pos_type; ///< pos type
    typedef typename traits_type::off_type off_type; ///< offset type

    ofstream();
    explicit ofstream(const char* filename, ios_base::openmode mode = ios_base::out|ios_base::trunc);
    explicit ofstream(const std::string& filename, ios_base::openmode mode = ios_base::out|ios_base::trunc);
    explicit ofstream(const Pathie::Path& filename, ios_base::openmode mode = ios_base::out|ios_base::trunc);
    ~ofstream();

    __gnu_cxx::stdio_filebuf<char>* rdbuf() const;
    bool is_open() const; // C++11 mandates const this, C++98 hadn’t that
    void open(const char* filename, ios_base::openmode mode = ios_base::out | ios_base::trunc);
    void open(const std::string& filename, ios_base::openmode mode = ios_base::out | ios_base::trunc);
    void open(const Pathie::Path& filename, ios_base::openmode mode = ios_base::out | ios_base::trunc);
    void close();

  private:
    FILE* mp_file;
    __gnu_cxx::stdio_filebuf<char>* mp_filebuffer;
    bool m_buffer_allocated;
  };

#  elif defined(_MSC_VER)
    class ofstream: public std::ofstream {
    public:
      ofstream();
      ofstream(char* path, std::ios_base::openmode = std::ios_base::out);
      ofstream(std::string path, std::ios_base::openmode = std::ios_base::out);
      ofstream(Pathie::Path path, std::ios_base::openmode = std::ios_base::out);
    };
#  else
#    error Unsupported compiler: do not know how to open C++ stream on Unicode file.
#  endif
#else
#  error Unsupported system.
#endif

}
#endif
