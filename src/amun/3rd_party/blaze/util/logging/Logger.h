//=================================================================================================
/*!
//  \file blaze/util/logging/Logger.h
//  \brief Header file for the Logger class
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

#ifndef _BLAZE_UTIL_LOGGING_LOGGER_H_
#define _BLAZE_UTIL_LOGGING_LOGGER_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <fstream>
#include <mutex>
#include <blaze/util/singleton/Singleton.h>
#include <blaze/util/SystemClock.h>


namespace blaze {

namespace logging {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Implementation of a logger class.
// \ingroup logging
//
// The Logger class represents the core of the logging functionality. It is responsible for
// commiting logging messages immediately to the according log file(s). The logger works for
// both serial as well as MPI parallel environments. In case of a non-MPI-parallel simulation
// the Logger creates the log file 'blaze.log', which contains all logging information from all
// logging levels. In case of a MPI parallel simulation, each process creates his own individual
// log file called 'blazeX.log', where 'X' is replaced by the according rank the process has in
// the MPI_COMM_WORLD communicator.\n
// Note that the log file(s) are only created in case any logging information is created. This
// might for instance result in only a small number of log file(s) in MPI parallel simulations
// when only some of the processes encounter errors/warnings/etc.\n
// Note that the logging functionality may not be used before MPI_Init() has been finished. In
// consequence, this means that no global data that is initialized before the main() function
// may contain any use of the logging functionality!
*/
class Logger : private Singleton<Logger,SystemClock>
{
 private:
   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   explicit Logger();
   //@}
   //**********************************************************************************************

 public:
   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~Logger();
   //@}
   //**********************************************************************************************

 private:
   //**Logging functions***************************************************************************
   /*!\name Logging functions */
   //@{
   template< typename Type > void log( const Type& message );
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   void openLogFile();
   //@}
   //**********************************************************************************************

   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   std::mutex    mutex_;  //!< Synchronization mutex for thread-parallel logging.
   std::ofstream log_;    //!< The log file.
   //@}
   //**********************************************************************************************

   //**Friend declarations*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   friend class FunctionTrace;
   friend class LogSection;
   BLAZE_BEFRIEND_SINGLETON;
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  LOGGING FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Writes the log message to the log file.
//
// \param message The log message to be logged.
// \return void
//
// This function immediately commits the log message to the log file. The first call to this
// function will create the log file.
*/
template< typename Type >  // Type of the log message
void Logger::log( const Type& message )
{
   std::lock_guard<std::mutex> lock( mutex_ );
   if( !log_.is_open() )
      openLogFile();
   log_ << message;
   log_.flush();
}
//*************************************************************************************************

} // namespace logging

} // namespace blaze

#endif
