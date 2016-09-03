//=================================================================================================
/*!
//  \file blaze/util/logging/LogSection.h
//  \brief Header file for the LogSection class
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

#ifndef _BLAZE_UTIL_LOGGING_LOGSECTION_H_
#define _BLAZE_UTIL_LOGGING_LOGSECTION_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <new>
#include <sstream>
#include <blaze/system/Logging.h>
#include <blaze/util/logging/LogLevel.h>


namespace blaze {

namespace logging {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Logging section for (non-)MPI-parallel environments.
// \ingroup logging
//
// The LogSection class is an auxiliary helper class for all logging section macros. It is
// implemented as a wrapper around the Logger class and is responsible for the atomicity of
// the logging operations and for formatting any message that is written into the log file(s).
*/
class LogSection
{
 public:
   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
          LogSection( LogLevel level );
   inline LogSection( const LogSection& ls );
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~LogSection();
   //@}
   //**********************************************************************************************

   //**Conversion operators************************************************************************
   /*!\name Conversion operators */
   //@{
   inline operator bool() const;
   //@}
   //**********************************************************************************************

   //**Logging functions***************************************************************************
   /*!\name Logging functions */
   //@{
   template< typename Type > inline void log   ( const Type& message );
                                    void commit();
   //@}
   //**********************************************************************************************

   //**Forbidden operations************************************************************************
   /*!\name Forbidden operations */
   //@{
   LogSection& operator=( const LogSection& ) = delete;

   void* operator new  ( std::size_t ) = delete;
   void* operator new[]( std::size_t ) = delete;
   void* operator new  ( std::size_t, const std::nothrow_t& ) noexcept = delete;
   void* operator new[]( std::size_t, const std::nothrow_t& ) noexcept = delete;

   void operator delete  ( void* ) noexcept = delete;
   void operator delete[]( void* ) noexcept = delete;
   void operator delete  ( void*, const std::nothrow_t& ) noexcept = delete;
   void operator delete[]( void*, const std::nothrow_t& ) noexcept = delete;
   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   LogLevel          level_;    //!< The logging level of the log section.
   std::stringstream message_;  //!< Intermediate buffer for log messages.
   //@}
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  CONSTRUCTORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief The copy constructor for LogSection.
//
// \param ls The log section to be copied.
//
// The copy constructor is explicitly defined in order to enable its use in the log sections
// despite the non-copyable stringstream member variable.
*/
inline LogSection::LogSection( const LogSection& ls )
   : level_( ls.level_ )  // The logging level of the log section
{}
//*************************************************************************************************




//=================================================================================================
//
//  CONVERSION OPERATOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Conversion operator to \a bool.
//
// The conversion operator returns \a true to indicate that the logging section is active.
*/
inline LogSection::operator bool() const
{
   return true;
}
//*************************************************************************************************




//=================================================================================================
//
//  LOGGING FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Logs the given message to the log file.
//
// \param message The log message to be logged.
// \return void
*/
template< typename Type >  // Type of the log message
inline void LogSection::log( const Type& message )
{
   message_ << message;
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\name LogSection operators */
//@{
template< typename Type >
inline LogSection& operator<<( LogSection& logsection, const Type& message );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Global output operator for the LogSection class.
// \ingroup logging
//
// \param logsection Reference to the log section.
// \param message Reference to the log message.
// \return Reference to the log section.
*/
template< typename Type >  // Type of the log message
inline LogSection& operator<<( LogSection& logsection, const Type& message )
{
   logsection.log( message );
   return logsection;
}
//*************************************************************************************************

} // namespace logging

} // namespace blaze

#endif
