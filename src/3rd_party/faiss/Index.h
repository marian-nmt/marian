/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#ifndef FAISS_INDEX_H
#define FAISS_INDEX_H

#include "utils/misc.h"
#include <cstdio>
#include <typeinfo>
#include <string>
#include <sstream>

#define FAISS_VERSION_MAJOR 1
#define FAISS_VERSION_MINOR 6
#define FAISS_VERSION_PATCH 3

/**
 * @namespace faiss
 *
 * Throughout the library, vectors are provided as float * pointers.
 * Most algorithms can be optimized when several vectors are processed
 * (added/searched) together in a batch. In this case, they are passed
 * in as a matrix. When n vectors of size d are provided as float * x,
 * component j of vector i is
 *
 *   x[ i * d + j ]
 *
 * where 0 <= i < n and 0 <= j < d. In other words, matrices are
 * always compact. When specifying the size of the matrix, we call it
 * an n*d matrix, which implies a row-major storage.
 */


namespace faiss {

/** Abstract structure for an index, supports adding vectors and searching them.
 *
 * All vectors provided at add or search time are 32-bit float arrays,
 * although the internal representation may vary.
 */
struct Index {
    using idx_t = int64_t;  ///< all indices are this type
    using component_t = float;
    using distance_t = float;
};

}


#endif
