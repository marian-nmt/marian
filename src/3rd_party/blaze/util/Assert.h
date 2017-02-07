//=================================================================================================
/*!
//  \file blaze/util/Assert.h
//  \brief Header file for run time assertion macros
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

#ifndef _BLAZE_UTIL_ASSERT_H_
#define _BLAZE_UTIL_ASSERT_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <cassert>
#include <blaze/system/Assertion.h>


namespace blaze {

//=================================================================================================
//
//  RUN TIME ASSERTION
//
//=================================================================================================

//*************************************************************************************************
/*!\defgroup assert Assertions
// \ingroup util
*/
/*!\defgroup runtime_assert Run time assertions
// \ingroup assert
*/
/*!\brief Assertion helper function.
// \ingroup runtime_assert
//
// The ASSERT_MESSAGE function is a small helper function to assist in printing an informative
// message in case an assert fires. This function builds on the ideas of Matthew Wilson, who
// directly combines a C-string error message with the run time expression (Imperfect C++,
// ISBN: 0321228774):

   \code
   assert( ... &&  "Error message" );
   assert( ... || !"Error message" );
   \endcode

// However, both approaches fail to compile without warning on certain compilers. Therefore
// this inline function is used instead of the direct approaches, which circumvents all compiler
// warnings:

   \code
   assert( ... || ASSERT_MESSAGE( "Error message" ) );
   \endcode
*/
inline bool ASSERT_MESSAGE( const char* /*msg*/ )
{
   return false;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Run time assertion macro for internal checks.
// \ingroup runtime_assert
//
// In case of an invalid run time expression, the program execution is terminated.\n
// The BLAZE_INTERNAL_ASSERT macro can be disabled by setting the \a BLAZE_USER_ASSERTION
// flag to zero or by defining \a NDEBUG during the compilation.
*/
#if BLAZE_INTERNAL_ASSERTION
#  define BLAZE_INTERNAL_ASSERT(expr,msg) assert( ( expr ) || blaze::ASSERT_MESSAGE( msg ) )
#else
#  define BLAZE_INTERNAL_ASSERT(expr,msg)
#endif
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Run time assertion macro for user checks.
// \ingroup runtime_assert
//
// In case of an invalid run time expression, the program execution is terminated.\n
// The BLAZE_USER_ASSERT macro can be disabled by setting the \a BLAZE_USER_ASSERT flag
// to zero or by defining \a NDEBUG during the compilation.
*/
#if BLAZE_USER_ASSERTION
#  define BLAZE_USER_ASSERT(expr,msg) assert( ( expr ) || blaze::ASSERT_MESSAGE( msg ) )
#else
#  define BLAZE_USER_ASSERT(expr,msg)
#endif
//*************************************************************************************************

} // namespace blaze

#endif
