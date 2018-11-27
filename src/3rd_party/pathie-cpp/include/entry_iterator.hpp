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

#ifndef PATHIE_ENTRY_ITERATOR_HPP
#define PATHIE_ENTRY_ITERATOR_HPP
#include <iterator>

namespace Pathie {

  class Path;

  /**
   * An iterator class for reading the entries in a directory.
   * Note that the entries of a directory always include the
   * "." (current directory) and ".." (parent directory) entries
   * unresolved, and that the order in which the entries in the
   * directory are returned is undefined (actually, the order
   * depends on the filesystem used).
   *
   * The iterators of this class are always const. You cannot change
   * the values referenced.
   *
   * It is unspecified behaviour what happens if a directory entry is
   * added or removed to/from the directory while you are iterating
   * it. Thus, keep iterations short in time.
   *
   * Instances of this class wrap an ephemeral handle like for example
   * a directory descriptor on Linux. This handle is not copiable,
   * which should normally mean that instances of this class cannot be
   * copied. However, the `std::iterator` interface mandates that
   * iterator instances are copiable (see "Requirements" here:
   * <http://en.cppreference.com/w/cpp/concept/Iterator>) and in fact
   * the language copies iterators all the time if you use them for
   * example in a for loop. Consequently, this class implements the
   * copy constructor and the copy assignment. However, these operations
   * do *not* actually copy the instance, but instead *move* the content
   * from the source instance to the target instance. The source intance
   * is afterwards unusable and looks like a finished iterator. The
   * `const` qualifiers in the copy operations are explicitely casted
   * away inside the functions to allow this, so they don't mean anything
   * for them. This works fairly nice for the ordinary use case (where
   * the language creates implicit copies), but the API may look as if
   * copying instances is allowed. It is not. *Do not copy* instances of
   * this class even though it looks as if it's possible. Implicit
   * copies automatically done by C++ as in for loops are okay, but
   * that's it. That is, you *can* do this:
   *
   * ~~~~{.cpp}
   *   entry_iterator iter;
   *   for(iter=my_path.begin_entries(); iter != my_path.end_entries(); iter++) {
   *     // Work with iter...
   *   }
   * ~~~~
   *
   * But you *cannot* do this:
   *
   * ~~~~{.cpp}
   * entry_iterator iter=my_path.begin_entries();
   * entry_iterator iter2(iter);
   * ~~~~
   *
   * This example does compile, but `iter` will be unusable after
   * `iter2` has been constructed.
   */
  class entry_iterator: public std::iterator<std::input_iterator_tag, Path, int>
  {
  public:
    entry_iterator();
    entry_iterator(const Path* p_top);
    ~entry_iterator();
    entry_iterator& operator=(const Path* p_top); // Restart assignment
    operator bool() const;
    bool operator==(const entry_iterator& other) const;
    bool operator!=(const entry_iterator& other) const;
    entry_iterator& operator++(int);
    entry_iterator& operator++();
    const Path& operator*() const;
    const Path* operator->() const;

    // "Copy" operations that really move the content, see class docs
    entry_iterator(const entry_iterator& other);
    entry_iterator& operator=(const entry_iterator& other);
  private:
    void open_native_handle();
    void close_native_handle();

    const Path* mp_directory; ///< Path requested to read from.
    void* mp_cur; ///< Native handle to the opened directory.
    Path* mp_cur_path; ///< Path instance of the path pointed to by mp_cur (only a pointer to allow forward-declaration of Path).
  };
}

#endif /* PATHIE_ENTRY_ITERATOR_HPP */
