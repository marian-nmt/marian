//=================================================================================================
/*!
//  \file blaze/util/logging/LogLevel.h
//  \brief Header file for the logging levels
//
//  Copyright (C) 2013 Klaus Iglberger - All Rights Reserved
//
//  This file is part of the Blaze library. You can redistribute it and/or modify it under
//  the terms of the New (Revised) BSD License. Redistribution and use in source and binary
//  forms, with or without modification, are permitted provided that the following conditions
//  are met:
//
//  1. Redistributions of source code must retain the above copyright notice, this list of
//     conditions and the following disclaimer.
//  2. Redistributions in binary form must reproduce the above copyright notice, this list
//     of conditions and the following disclaimer in the documentation and/or other materials
//     provided with the distribution.
//  3. Neither the names of the Blaze development group nor the names of its contributors
//     may be used to endorse or promote products derived from this software without specific
//     prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
//  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
//  OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
//  SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
//  TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
//  BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
//  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
//  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
//  DAMAGE.
*/
//=================================================================================================

#ifndef _BLAZE_UTIL_LOGGING_LOGLEVEL_H_
#define _BLAZE_UTIL_LOGGING_LOGLEVEL_H_


namespace blaze {

namespace logging {

//=================================================================================================
//
//  LOGGING LEVELS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Logging levels.
// \ingroup logging
//
// The LogLevel type enumeration represents the type of the global logging level. It defines
// all possible levels for the logging functionality. Depending on the setting of the global
// logging level (see blaze::logLevel), more or less information will be written to the log
// file(s). The following logging levels are available:
//
//  - \a inactive: Completely deactivates the logging functionality, i.e., no log file(s) will
//                 be written. Since this setting can immensely complicate error correction, it
//                 is not recommended to use this setting!
//  - \a error   : Only (severe) errors are written to the log file(s).
//  - \a warning : Extends the \a error setting by warning messages (default).
//  - \a info    : Extends the \a warning setting by additional informative messages.
//  - \a progress: Extends the \a info setting by progress information.
//  - \a debug   : Extends the \a progress setting by debug information.
//  - \a detail  : Extends the \a debug setting by very fine grained detail information.
//
// \a inactive plays a special role in the way that it switches off the logging functionality
// completely, i.e., no log file(s) will be created. The highest logging level is \a error,
// which exclusively writes severe errors to the log file(s). The lowest logging level is
// \a detail, which can create a tremendous amount of logging information. Note that each
// logging level comprises all higher logging levels. For instance, \a progress will also
// print all errors and warning to the log file(s).
*/
enum LogLevel
{
   inactive = 0,  //!< Log level for no logging.
   error    = 1,  //!< Log level for (sever) errors.
   warning  = 2,  //!< Log level for warnings.
   info     = 3,  //!< Log level for high-level information.
   progress = 4,  //!< Log level for progress information.
   debug    = 5,  //!< Log level for debug information.
   detail   = 6   //!< Log level for detail information.
};
//*************************************************************************************************

} // namespace logging

} // namespace blaze

#endif
