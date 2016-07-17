//=================================================================================================
/*!
//  \file blaze/util/logging/InfoSection.h
//  \brief Header file for the log info section
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

#ifndef _BLAZE_UTIL_LOGGING_INFOSECTION_H_
#define _BLAZE_UTIL_LOGGING_INFOSECTION_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/util/logging/LogSection.h>


namespace blaze {

namespace logging {

//=================================================================================================
//
//  BLAZE_LOG_INFO_SECTION MACRO
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Logging section for information messages.
// \ingroup logging
//
// This macro starts a log section for information messages. These messages are written to the
// log file(s) in case the blaze::loglevel has been set to \a info or higher. The following example
// demonstrates how this log section is used:

   \code
   int main( int argc, char** argv )
   {
      // Initialization of the MPI system (for MPI parallel simulations)
      // The MPI system must be initialized before any logging functionality may be used. In
      // case it was not called before the first log section it is assumed that the simulation
      // does not run in parallel. Thus in MPI-parallel simulations it is strongly recommended
      // to make MPI_Init() the very first call of the main function.
      MPI_Init( &argc, &argv );

      // ...

      // Log section for information messages
      // This section is only executed in case the logging level is at least 'info'. The
      // macro parameter specifies the name of the log handle (in this example 'log') that
      // can be used as a stream to log any kind of streamable information.
      BLAZE_LOG_INFO_SECTION( log ) {
         log << " Only printed within an active BLAZE_LOG_INFO_SECTION!\n";
      }

      // ...

      // Finalizing the MPI system (for MPI parallel simulations)
      // The MPI system must be finalized after the last pe functionality has been used. It
      // is recommended to make MPI_Finalize() the very last call of the main function.
      MPI_Finalize();
   }
   \endcode

// Note that uncaught exceptions emitted from the blaze::BLAZE_LOG_INFO_SECTION might result
// in lost and/or unlogged information!
*/
#define BLAZE_LOG_INFO_SECTION( NAME ) \
   if( blaze::logging::loglevel >= blaze::logging::info ) \
      if( blaze::logging::LogSection NAME = blaze::logging::info )
//*************************************************************************************************

} // namespace logging

} // namespace blaze

#endif
