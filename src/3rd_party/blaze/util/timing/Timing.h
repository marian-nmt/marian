//=================================================================================================
/*!
//  \file blaze/util/timing/Timing.h
//  \brief Timing module documentation
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

#ifndef _BLAZE_UTIL_TIMING_TIMING_H_
#define _BLAZE_UTIL_TIMING_TIMING_H_


namespace blaze {

//=================================================================================================
//
//  DOXYGEN DOCUMENTATION
//
//=================================================================================================

//*************************************************************************************************
//! Namespace for the time measurement module.
namespace timing {}
//*************************************************************************************************


//*************************************************************************************************
/*!\defgroup timing Time measurement
// \ingroup util
//
// \image html clock.png
// \image latex clock.eps "Timing submodule" width=200pt
//
// The timing submodule offers the necessary functionality for timing and benchmarking purposes.
// The central element of the timing module is the Timer class. Depending on a chosen timing
// policy, this class offers the possibility to measure both single times and time series. In
// order to make time measurement as easy as possible, the Blaze library offers the two classes
// WcTimer and CpuTimer (both using the Timer class) to measure both wall clock and CPU time.
// The following example gives an impression on how time measurement for a single time works
// with the the Blaze library. Note that in this example the WcTimer could be easily replaced
// with the CpuTimer if instead of the wall clock time the CPU time was to be measured.

   \code
   // Creating a new wall clock timer immediately starts a new time measurement
   WcTimer timer;

   ...  // Programm or code fragment to be measured

   // Stopping the time measurement
   timer.end();

   // Evaluation of the measured time
   double time = timer.last();
   \endcode

// As already mentioned, it is also possible to start several time measurements with a single
// timer to evaluate for instance the minimal, the maximal or the average time of a specific
// task. The next example demonstrates a possible setup for such a series of time measurements:

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
//*************************************************************************************************

} // namespace blaze

#endif
