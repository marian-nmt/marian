#pragma once

// Do NOT include this file directly except in special circumstances.
// (E.g., you want to define macros which call these but don't want to include Logger.h everywhere).
// Normally you should include logging/Logger.h

#define LOG_WRITE(format, ...) do {\
    abort(); \
} while (0)

#define LOG_WRITE_STRING(str) do {\
    abort(); \
} while (0)

#define LOG_ERROR(format, ...) do {\
    abort(); \
} while (0)

#define LOG_ERROR_AND_THROW(format, ...) do {\
    abort(); \
} while (0)

#define DECODING_LOGIC_ERROR(format, ...) do {\
    abort(); \
} while (0)
