//=================================================================================================
/*!
//  \file blaze/config/Logging.h
//  \brief Configuration of the logging functionality
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


namespace blaze {

namespace logging {

//*************************************************************************************************
/*!\brief Setting of the logging level.
// \ingroup config
//
// This value specifies the logging level of the Blaze logging functionality. Depending on
// this setting, more or less informations will be written to the log file(s). The following
// logging levels can be selected:
//
//  - \a inactive: Completely deactives the logging functionality, i.e., no log file(s) will be
//                 written. Since this setting can immensely complicate error correction, it is
//                 not recommended to use this setting!
//  - \a error   : Only (severe) errors are written to the log file(s).
//  - \a warning : Extends the \a error setting by warning messages.
//  - \a info    : Extends the \a warning setting by additional informative messages (default).
//  - \a progress: Extends the \a info setting by progress informations.
//  - \a debug   : Extends the \a progress setting by debug information.
//  - \a detail  : Extends the \a debug setting by very fine grained detail information.
*/
constexpr LogLevel loglevel = info;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Adding an additional spacing line between two log messages.
// \ingroup config
//
// This setting gives the opportunity to add an additional spacing line between two log messages
// to improve readability of log files. If set to \a true, each log message will be appended with
// an additional empty line. If set to \a false, no line will be appended.
*/
constexpr bool spacing = false;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Compilation switch for function traces.
// \ingroup config
//
// This compilation switch triggers the use of function traces. In case the switch is set to
// 1, function traces via the BLAZE_FUNCTION_TRACE are enabled. Note however, that enabling
// function traces creates a dependency to the compiled Blaze library, i.e. it will be
// necessary to link the Blaze library to the executable. This is also true in case only
// template functionality is used!
//
// Possible settings for the function trace switch:
//  - Deactivated: \b 0 (default)
//  - Activated  : \b 1
*/
#define BLAZE_USE_FUNCTION_TRACES 0
//*************************************************************************************************

} // namespace logging

} // namespace blaze
