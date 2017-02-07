//=================================================================================================
/*!
//  \file blaze/util/Time.h
//  \brief Header file for time functions
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

#ifndef _BLAZE_UTIL_TIME_H_
#define _BLAZE_UTIL_TIME_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#if defined(_MSC_VER)
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#  include <winsock.h>
#  include <time.h>
#  include <sys/timeb.h>
#else
#  include <sys/resource.h>
#  include <sys/time.h>
#  include <sys/types.h>
#endif
#include <ctime>
#include <string>


namespace blaze {

//=================================================================================================
//
//  TIME FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\name Time functions */
//@{
inline std::string getDate();
inline std::string getTime();
inline double      getWcTime();
inline double      getCpuTime();
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a formated date string in the form YYYY-MM-DD
// \ingroup util
//
// \return Formated date string
*/
inline std::string getDate()
{
   std::time_t t;
   std::tm* localTime;
   char c[50];

   std::time( &t );
   localTime = std::localtime( &t );
   std::strftime( c, 50, "%Y-%m-%d", localTime );

   return std::string( c );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Creating a formated time and date string
// \ingroup util
//
// \return Formated time and date string in the format WEEKDAY DAY.MONTH YEAR, HOUR:MINUTES
*/
inline std::string getTime()
{
   std::time_t t;
   std::tm* localTime;
   char c[50];

   std::time( &t );
   localTime = std::localtime( &t );
   std::strftime( c, 50, "%A, %d.%B %Y, %H:%M", localTime );

   return std::string( c );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the current wall clock time in seconds.
// \ingroup util
//
// \return The current wall clock time in seconds.
*/
inline double getWcTime()
{
#ifdef WIN32
   struct _timeb timeptr;
   _ftime( &timeptr );
   return ( static_cast<double>( timeptr.time ) + static_cast<double>( timeptr.millitm )/1E3 );
#else
   struct timeval tp;
   gettimeofday( &tp, nullptr );
   return ( static_cast<double>( tp.tv_sec ) + static_cast<double>( tp.tv_usec )/1E6 );
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the current CPU time in seconds.
// \ingroup util
//
// \return The current CPU time in seconds.
*/
inline double getCpuTime()
{
#ifdef WIN32
   FILETIME CreateTime, ExitTime, KernelTime, UserTime;
   SYSTEMTIME SysTime;

   if( GetProcessTimes( GetCurrentProcess(), &CreateTime, &ExitTime, &KernelTime, &UserTime ) != TRUE ) {
      return 0.0;
   }
   else {
      FileTimeToSystemTime( &UserTime, &SysTime );
      return ( static_cast<double>( SysTime.wSecond ) + static_cast<double>( SysTime.wMilliseconds )/1E3 );
   }
#else
   struct rusage ruse;
   getrusage( RUSAGE_SELF, &ruse );
   return ( static_cast<double>( ruse.ru_utime.tv_sec ) + static_cast<double>( ruse.ru_utime.tv_usec )/1E6 );
#endif
}
//*************************************************************************************************

} // namespace blaze

#endif
