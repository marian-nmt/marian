//=================================================================================================
/*!
//  \file blaze/math/shims/InvSqrt.h
//  \brief Header file for the invsqrt shim
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

#ifndef _BLAZE_MATH_SHIMS_INVSQRT_H_
#define _BLAZE_MATH_SHIMS_INVSQRT_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/shims/Invert.h>
#include <blaze/math/shims/Sqrt.h>
#include <blaze/util/Assert.h>
#include <blaze/util/Complex.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/typetraits/IsBuiltin.h>


namespace blaze {

//=================================================================================================
//
//  INVSQRT SHIM
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Returns the inverse square root of the given built-in value.
// \ingroup math_shims
//
// \param a The given built-in value \f$[0..\infty)\f$.
// \return The inverse square root of the given value.
//
// \note The given value must be in the range \f$[0..\infty)\f$. The validity of the value is
// only checked by an user assert.
*/
template< typename T, typename = EnableIf_< IsBuiltin<T> > >
inline auto invsqrt( T a ) noexcept -> decltype( inv( sqrt( a ) ) )
{
   BLAZE_USER_ASSERT( a > T(0), "Invalid built-in value detected" );

   return inv( sqrt( a ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the inverse square root of the given complex number.
// \ingroup math_shims
//
// \param a The given complex number.
// \return The inverse square root of the given complex number.
//
// \note The given complex number must not be zero. The validity of the value is only checked by
// an user assert.
*/
template< typename T, typename = EnableIf_< IsBuiltin<T> > >
inline auto invsqrt( const complex<T>& a ) noexcept -> decltype( inv( sqrt( a ) ) )
{
   BLAZE_USER_ASSERT( abs( a ) != T(0), "Invalid complex value detected" );

   return inv( sqrt( a ) );
}
//*************************************************************************************************

} // namespace blaze

#endif
