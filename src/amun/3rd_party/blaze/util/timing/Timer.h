//=================================================================================================
/*!
//  \file blaze/util/timing/Timer.h
//  \brief Progress timer for time and performance measurements
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

#ifndef _BLAZE_UTIL_TIMING_TIMER_H_
#define _BLAZE_UTIL_TIMING_TIMER_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <limits>
#include <blaze/util/Time.h>
#include <blaze/util/Types.h>


namespace blaze {

namespace timing {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Progress timer for time and performance measurements.
// \ingroup timing
//
// The Timer class offers timing & benchmarking functionality for all kinds of applications.
// The following example code demonstrates the use of the WcTimer class, which combines the
// Timer class template with the WcPolicy for wall clock time measurements, for a single time
// measurement:

   \code
   // Creating a new wall clock timer immediately starts a new time measurement
   WcTimer timer;

   ...  // Programm or code fragment to be measured

   // Stopping the time measurement
   timer.end();

   // Evaluation of the measured time
   double time = timer.last();
   \endcode

// The timer class additionally offers the functionality to start several time measurments in
// order to evaluate minimal, maximal or average times. The next example demonstrates a possible
// setup for such a series of time measurements:

   \code
   // Creating a new wall clock timer
   WcTimer timer;

   ...  // Additional setup code

   // Starting 10 wall clock time measurements
   for( unsigned int i=0; i<10; ++i ) {
      timer.start();
      ...  // Programm or code fragment to be measured
      timer.end();
   }

   // After the measurements, the desired timing results can be calculated, as for instance the
   // average wall clock time
   double average = timer.average();
   \endcode
*/
template< typename TP >  // Timing policy
class Timer
{
 public:
   //**Type definitions****************************************************************************
   typedef TP  TimingPolicy;  //!< Timing policy of the Timer.
   //**********************************************************************************************

   //**Constructor*********************************************************************************
   /*!\name Constructor */
   //@{
   explicit inline Timer();
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   // No explicitly declared destructor.
   //**********************************************************************************************

   //**Timing functions****************************************************************************
   /*!\name Timing functions */
   //@{
   inline void start();
   inline void end  ();
   inline void reset();
   //@}
   //**********************************************************************************************

   //**Get functions*******************************************************************************
   /*!\name Get functions */
   //@{
   inline size_t getCounter() const;
   //@}
   //**********************************************************************************************

   //**Time evaluation functions*******************************************************************
   /*!\name Time evaluation functions */
   //@{
   inline double total()   const;
   inline double average() const;
   inline double min()     const;
   inline double max()     const;
   inline double last()    const;
   //@}
   //**********************************************************************************************

 private:
   size_t counter_;  //!< Number of performed time measurements.
   double start_;    //!< Start of the current time measurement.
   double end_;      //!< End of the current time measurement.
   double time_;     //!< The total elapsed time of all measurements.
   double min_;      //!< The minimal time of all measurements.
   double max_;      //!< The maximal time of all measurements.
   double last_;     //!< The last measured time.
};
//*************************************************************************************************




//=================================================================================================
//
//  CONSTRUCTOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constructor of the Timer class.
//
// The creation of a new timer immediately starts a new time measurement. It is possible to
// either restart the time measurement at a specific point of time or to continue the time
// measurement and to end it via the end() function.
*/
template< typename TP >  // Timing policy
inline Timer<TP>::Timer()
   : counter_( 0   )
   , start_  ( 0.0 )
   , end_    ( 0.0 )
   , time_   ( 0.0 )
   , min_    ( std::numeric_limits<double>::max() )
   , max_    ( 0.0 )
   , last_   ( 0.0 )
{
   // Starting the time measurement
   start();
}
//*************************************************************************************************




//=================================================================================================
//
//  TIMING FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Starting a single time measurement.
//
// \return void
//
// This function starts a single time measurement.
*/
template< typename TP >  // Timing policy
inline void Timer<TP>::start()
{
   // Starting the time measurement and calculating a time stamp
   start_ = TimingPolicy::getTimestamp();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Ending a single time measurement.
//
// \return void
//
// This function ends the currently running time measurement and performs the necessary
// statistical calculations.
*/
template< typename TP >  // Timing policy
inline void Timer<TP>::end()
{
   // Stopping the time measurement and calculating a time stamp
   end_ = TimingPolicy::getTimestamp();

   // Increasing the counter
   ++counter_;

   // Calculating the wallclock and CPU time
   const double diff( end_ - start_ );

   // Average time measurement
   time_ += diff;

   // Minimum time measurement
   if( diff < min_ ) min_ = diff;

   // Maximum time measurement
   if( diff  > max_ ) max_ = diff;

   // Last time measurement
   last_ = diff;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Resetting the timer.
//
// \return void
//
// This function completely resets the timer and all information on the performed time
// measurements. In order to start a new time measurement, the start() function has to
// be used.
*/
template< typename TP >  // Timing policy
inline void Timer<TP>::reset()
{
   counter_ = 0;
   start_   = 0.0;
   end_     = 0.0;
   time_    = 0.0;
   min_     = std::numeric_limits<double>::max();
   max_     = 0.0;
   last_    = 0.0;
}
//*************************************************************************************************




//=================================================================================================
//
//  GET FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Returns the total number of time measurements performed by this timer.
//
// \return The number of performed time measurements.
*/
template< typename TP >  // Timing policy
inline size_t Timer<TP>::getCounter() const
{
   return counter_;
}
//*************************************************************************************************




//=================================================================================================
//
//  TIME EVALUATION FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Returns the total elapsed time of all performed time measurements.
//
// \return The total elapsed time of all time measurements.
*/
template< typename TP >  // Timing policy
inline double Timer<TP>::total() const
{
   return time_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the average time of all performed time measurements.
//
// \return The average time.
*/
template< typename TP >  // Timing policy
inline double Timer<TP>::average() const
{
   return time_ / counter_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the minimal time of all performed time measurements.
//
// \return The minimal time.
*/
template< typename TP >  // Timing policy
inline double Timer<TP>::min() const
{
   return min_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the maximal time of all performed time measurements.
//
// \return The maximal time.
*/
template< typename TP >  // Timing policy
inline double Timer<TP>::max() const
{
   return max_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the last measured time.
//
// \return The last measured time.
*/
template< typename TP >  // Timing policy
inline double Timer<TP>::last() const
{
   return last_;
}
//*************************************************************************************************

} // timing

} // blaze

#endif
