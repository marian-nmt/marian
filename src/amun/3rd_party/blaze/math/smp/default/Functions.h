//=================================================================================================
/*!
//  \file blaze/math/smp/default/Functions.h
//  \brief Header file for the default SMP utility functions
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

#ifndef _BLAZE_MATH_SMP_DEFAULT_FUNCTIONS_H_
#define _BLAZE_MATH_SMP_DEFAULT_FUNCTIONS_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/system/Inline.h>
#include <blaze/system/SMP.h>
#include <blaze/util/StaticAssert.h>
#include <blaze/util/Types.h>
#include <blaze/util/Unused.h>


namespace blaze {

//=================================================================================================
//
//  SMP UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\name SMP utility functions */
//@{
BLAZE_ALWAYS_INLINE size_t getNumThreads  ();
BLAZE_ALWAYS_INLINE void   setNumThreads  ( size_t number );
BLAZE_ALWAYS_INLINE void   shutDownThreads();
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the number of threads used for thread parallel operations.
// \ingroup smp
//
// \return The number of threads used for thread parallel operations.
//
// Via this function the number of threads used for thread parallel operations can be queried.
// Note that in case no parallelization is active the function will always return 1.
*/
BLAZE_ALWAYS_INLINE size_t getNumThreads()
{
   return 1UL;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Sets the number of threads to be used for thread parallel operations.
// \ingroup smp
//
// \param number The given number of threads \f$[1..\infty)\f$.
// \return void
// \exception std::invalid_argument Invalid number of threads.
//
// Via this function the maximum number of threads for thread parallel operations can be specified.
// Note that the given \a number must be in the range \f$[1..\infty)\f$. In case an invalid
// number of threads is specified, a \a std::invalid_argument exception is thrown. Also note that
// in case no parallelization is active, the function has no effect.
*/
BLAZE_ALWAYS_INLINE void setNumThreads( size_t number )
{
   UNUSED_PARAMETER( number );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Provides a reliable shutdown of C++11 threads for Visual Studio compilers.
// \ingroup smp
//
// \return void
//
// There is a known issue in Visual Studio 2012 and 2013 that may cause C++11 threads to hang
// if their destructor is executed after the \c main() function:
//
//    http://connect.microsoft.com/VisualStudio/feedback/details/747145
//
// This function, which has only an effect for Visual Studio compilers, provides a reliable way
// to circumvent this problem. If called directly before the end of the \c main() function it
// blocks until all threads have been destroyed:

   \code
   int main()
   {
      // ... Using the C++11 thread parallelization of Blaze

      shutDownThreads();
   }
   \endcode
*/
BLAZE_ALWAYS_INLINE void shutDownThreads()
{}
//*************************************************************************************************




//=================================================================================================
//
//  COMPILE TIME CONSTRAINT
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
namespace {

BLAZE_STATIC_ASSERT( !BLAZE_OPENMP_PARALLEL_MODE      );
BLAZE_STATIC_ASSERT( !BLAZE_CPP_THREADS_PARALLEL_MODE );

}
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
