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

#ifndef PATHIE_IFSTREAM_HPP
#define PATHIE_IFSTREAM_HPP
#include <fstream>

#if defined(_WIN32) && defined(__GNUC__)
#include <ext/stdio_filebuf.h>
#endif

#include "path.hpp"

namespace Pathie {

#if defined(_PATHIE_UNIX)
  class ifstream: public std::ifstream {
  public:
    ifstream();
    ifstream(char* path, std::ios_base::openmode = std::ios_base::in);
    ifstream(std::string path, std::ios_base::openmode = std::ios_base::in);
    ifstream(Pathie::Path path, std::ios_base::openmode = std::ios_base::in);

    void open(const char* filename, ios_base::openmode mode = ios_base::in);
    void open(const std::string& filename, ios_base::openmode mode = ios_base::in);
    void open(const Pathie::Path& filename, ios_base::openmode mode = ios_base::in);
  };

#elif defined (_WIN32)
#  if defined(__GNUC__)
  /**
   * \brief Input stream for UTF-8-encoded filenames.
   *
   * This class implements an interface like `std::ifstream` that works
   * with Unicode paths regardless of the platform. Please refer to
   * the documentation of Pathie::ofstream for more information on
   * rationale and usage; this class works the same way as Pathie::ofstream,
   * just for input rather than output file streams.
   */
  class ifstream: public std::basic_istream<char, std::char_traits<char> >
  {
  public:
    typedef char char_type;                          ///< Type used inside the stream.
    typedef std::char_traits<char> traits_type;      ///< Traits type
    typedef typename traits_type::int_type int_type; ///< Int type
    typedef typename traits_type::pos_type pos_type; ///< pos type
    typedef typename traits_type::off_type off_type; ///< offset type

    ifstream();
    explicit ifstream(const char* filename, ios_base::openmode mode = ios_base::in);
    explicit ifstream(const std::string& filename, ios_base::openmode mode = ios_base::in);
    explicit ifstream(const Pathie::Path& filename, ios_base::openmode mode = ios_base::in);
    ~ifstream();

    __gnu_cxx::stdio_filebuf<char>* rdbuf() const;
    bool is_open() const; // C++11 mandates const this, C++98 hadn’t that
    void open(const char* filename, ios_base::openmode mode = ios_base::in);
    void open(const std::string& filename, ios_base::openmode mode = ios_base::in);
    void open(const Pathie::Path& filename, ios_base::openmode mode = ios_base::in);
    void close();

  private:
    FILE* mp_file;
    __gnu_cxx::stdio_filebuf<char>* mp_filebuffer;
    bool m_buffer_allocated;
  };

#  elif defined(_MSC_VER)
    class ifstream: public std::ifstream {
    public:
      ifstream();
      ifstream(char* path, std::ios_base::openmode = std::ios_base::in);
      ifstream(std::string path, std::ios_base::openmode = std::ios_base::in);
      ifstream(Pathie::Path path, std::ios_base::openmode = std::ios_base::in);
    };

#  else
#    error Unsupported compiler: do not know how to open C++ stream on Unicode file.
#  endif
#else
#  error Unsupported system.
#endif

}
#endif
