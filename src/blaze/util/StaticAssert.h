//=================================================================================================
/*!
//  \file blaze/util/StaticAssert.h
//  \brief Compile time assertion
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

#ifndef _BLAZE_UTIL_STATICASSERT_H_
#define _BLAZE_UTIL_STATICASSERT_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/util/Suffix.h>


namespace blaze {

//=================================================================================================
//
//  COMPILE TIME ASSERTION
//
//=================================================================================================

//*************************************************************************************************
/*!\defgroup static_assert Compile time assertion
// \ingroup assert
//
// Static assertion offers the possibility to stop the compilation process if a specific compile
// time condition is not met. The blaze::BLAZE_STATIC_ASSERT and blaze::BLAZE_STATIC_ASSERT_MSG
// macros can be used to check a constant expression at compile time. If the expression evaluates
// to \a false, a compilation error is generated that stops the compilation process. If the
// expression (hopefully) evaluates to \a true, the compilation process is not aborted and the
// static check leaves neither code nor data and is therefore not affecting the performance.\n
//
// Both static assertion macros can be used wherever a standard typedef statement can be declared,
// i.e. in namespace scope, in class scope and in function scope. The following examples illustrate
// the use of the static assertion macros: The element type of the rotation matrix is checked at
// compile time and restricted to be of floating point type.

   \code
   #include <blaze/util/StaticAssert.h>
   #include <blaze/util/typetraits/FloatingPoint.h>

   template< typename T >
   class RotationMatrix {
      ...
      BLAZE_STATIC_ASSERT( IsFloatingPoint<T>::value );
      // ... or ...
      BLAZE_STATIC_ASSERT_MSG( IsFloatingPoint<T>::value, "Given type is not a floating point type" );
      ...
   };
   \endcode

// The static assertion implementation is based on the C++11 \c static_assert declaration. Thus
// the error message contains the violated compile time condition and directly refers to the
// violated static assertion. The following examples show a possible error message from the GNU
// g++ compiler:

   \code
   error: static assertion failed: Compile time condition violated
      static_assert( expr, "Compile time condition violated" )
      ^
   note: in expansion of macro 'BLAZE_STATIC_ASSERT'
      BLAZE_STATIC_ASSERT( IsFloatingPoint<T>::value );

   error: static assertion failed: Given type is not a floating point type
      static_assert( expr, msg )
      ^
   note: in expansion of macro 'BLAZE_STATIC_ASSERT_MSG'
      BLAZE_STATIC_ASSERT_MSG( IsFloatingPoint<T>::value, "Given type is not a floating point type" );
   \endcode
*/
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Compile time assertion macro.
// \ingroup static_assert
//
// In case of an invalid compile time expression, a compilation error is created.
*/
#define BLAZE_STATIC_ASSERT(expr) \
   static_assert( expr, "Compile time condition violated" )
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Compile time assertion macro.
// \ingroup static_assert
//
// In case of an invalid compile time expression, a compilation error is created.
*/
#define BLAZE_STATIC_ASSERT_MSG(expr,msg) \
   static_assert( expr, msg )
//*************************************************************************************************

} // namespace blaze

#endif
