//=================================================================================================
/*!
//  \file blaze/util/SystemClock.h
//  \brief Header file for the SystemClock class
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

#ifndef _BLAZE_UTIL_SYSTEMCLOCK_H_
#define _BLAZE_UTIL_SYSTEMCLOCK_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <ctime>
#include <blaze/util/singleton/Singleton.h>
#include <blaze/util/SystemClockID.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief System clock of the Blaze library.
// \ingroup util
//
// The SystemClock class represents the system clock of the Blaze library. The system clock
// is the central timing functionality that can be used to query for the start time of the
// process, the current timestamp and the elapsed time since the start of the process. The
// following example demonstrates how the single system clock instance is acquired via the
// theSystemClock() functcion and how the system clock can be used:

   \code
   // The single system clock instance is accessible via the theSystemClock() function
   SystemClockID systemClock = theSystemClock();

   time_t start   = systemClock->start();    // Querying the start time of the process
   time_t current = systemClock->current();  // Querying the current timestamp
   time_t elapsed = systemClock->elapsed();  // Querying the elapsed time
   \endcode
*/
class SystemClock : private Singleton<SystemClock>
{
 private:
   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   explicit SystemClock();
   //@}
   //**********************************************************************************************

 public:
   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~SystemClock();
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   inline time_t start  () const;
   inline time_t now    () const;
   inline time_t elapsed() const;
   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   static time_t start_;  //!< Timestamp for the start of the process.
   //@}
   //**********************************************************************************************

   //**Friend declarations*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   friend SystemClockID theSystemClock();
   BLAZE_BEFRIEND_SINGLETON;
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  SYSTEM CLOCK SETUP FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\name System clock setup functions */
//@{
inline SystemClockID theSystemClock();
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns a handle to the Blaze system clock.
// \ingroup util
//
// \return Handle to the active system clock.
*/
inline SystemClockID theSystemClock()
{
   return SystemClock::instance();
}
//*************************************************************************************************




//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Returns the timestamp for the start of the process.
//
// \return Timestamp for the start of the process.
*/
inline time_t SystemClock::start() const
{
   return start_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the current timestamp.
//
// \return The current timestamp.
*/
inline time_t SystemClock::now() const
{
   return time( nullptr );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the elapsed time since the start of the process (in seconds).
//
// \return Elapsed time since the start of the process (in seconds).
*/
inline time_t SystemClock::elapsed() const
{
   return std::time( nullptr ) - start_;
}
//*************************************************************************************************

} // namespace blaze

#endif
