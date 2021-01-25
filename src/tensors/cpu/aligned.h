#pragma once

#include "common/definitions.h"
#include <stdlib.h>
#ifdef _WIN32
#include <malloc.h>
#endif

namespace marian {
namespace cpu {
namespace {

// allocate function for tensor reserve() below. 
// Alignment is needed because we use AVX512 and AVX2 vectors. We should fail if we can't allocate aligned memory.

#ifdef _WIN32
void *genericMalloc(size_t alignment, size_t size) {
  void *ret = _aligned_malloc(size, alignment);
  ABORT_IF(!ret, "Failed to allocate memory on CPU");
  return ret;
}
void genericFree(void *ptr) {
  _aligned_free(ptr);
}
#else
// Linux and OS X.  There is no fallback to malloc because we need it to be aligned.
void *genericMalloc(size_t alignment, size_t size) {
  // On macos, aligned_alloc is available only on c++17
  // Furthermore, it requires that the memory requested is an exact multiple of the alignment, otherwise it fails.
  // posix_memalign is available both Mac (Since 2016) and Linux and in both gcc and clang
  void *result;
  // Error could be detected by return value or just remaining nullptr.
  ABORT_IF(posix_memalign(&result, alignment, size), "Failed to allocate memory on CPU");
  return result;
}
void genericFree(void *ptr) {
  free(ptr);
}
#endif

}
} // namespace cpu
} // namespace marian
