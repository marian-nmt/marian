//=================================================================================================
/*!
//  \file blaze/math/functors/InvSqrt.h
//  \brief Header file for the InvSqrt functor
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

#ifndef _BLAZE_MATH_FUNCTORS_INVSQRT_H_
#define _BLAZE_MATH_FUNCTORS_INVSQRT_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/constraints/SIMDPack.h>
#include <blaze/math/shims/InvSqrt.h>
#include <blaze/math/simd/InvSqrt.h>
#include <blaze/math/typetraits/HasSIMDInvSqrt.h>
#include <blaze/system/Inline.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Generic wrapper for the invsqrt() function.
// \ingroup functors
*/
struct InvSqrt
{
   //**********************************************************************************************
   /*!\brief Returns the result of the invsqrt() function for the given object/value.
   //
   // \param a The given object/value.
   // \return The result of the invsqrt() function for the given object/value.
   */
   template< typename T >
   BLAZE_ALWAYS_INLINE auto operator()( const T& a ) const -> decltype( invsqrt( a ) )
   {
      return invsqrt( a );
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether SIMD is enabled for the specified data type \a T.
   //
   // \return \a true in case SIMD is enabled for the data type \a T, \a false if not.
   */
   template< typename T >
   static constexpr bool simdEnabled() { return HasSIMDInvSqrt<T>::value; }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns the result of the invsqrt() function for the given SIMD vector.
   //
   // \param a The given SIMD vector.
   // \return The result of the invsqrt() function for the given SIMD vector.
   */
   template< typename T >
   BLAZE_ALWAYS_INLINE auto load( const T& a ) const -> decltype( invsqrt( a ) )
   {
      BLAZE_CONSTRAINT_MUST_BE_SIMD_PACK( T );
      return invsqrt( a );
   }
   //**********************************************************************************************
};
//*************************************************************************************************

} // namespace blaze

#endif
