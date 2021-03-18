#pragma once

#include <inttypes.h>

#ifdef QUICKSAND_WINDOWS_BUILD
#define PI32 "d"
#define PI64 "lld"
#define PU32 "u"
#define PU64 "llu"
#else
#define PI32 PRId32
#define PI64 PRId64
#define PU32 PRIu32
#define PU64 PRIu64
#endif

