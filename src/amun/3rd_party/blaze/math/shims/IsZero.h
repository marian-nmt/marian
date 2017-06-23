//=================================================================================================
/*!
//  \file blaze/math/shims/IsZero.h
//  \brief Header file for the isZero shim
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

#ifndef _BLAZE_MATH_SHIMS_ISZERO_H_
#define _BLAZE_MATH_SHIMS_ISZERO_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/shims/IsDefault.h>
#include <blaze/system/Inline.h>
#include <blaze/util/DisableIf.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/typetraits/IsNumeric.h>
#include <blaze/util/Unused.h>


namespace blaze {

//=================================================================================================
//
//  ISZERO SHIM
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Returns whether the given value/object represents the numeric value 0.
// \ingroup math_shims
//
// \param v The value to be tested.
// \return \a true in case the given value is 0, \a false otherwise.
//
// The \a isZero shim provides an abstract interface for testing a value/object of any type
// whether it represents the numeric value 0. In case the value/object is 0, the function
// returns \a true, otherwise it returns \a false.

   \code
   const int i1 = 0;                // isZero( i1 ) returns true
   double    d1 = 0.0;              // isZero( d1 ) returns true
   complex<double> c1( 0.0, 0.0 );  // isZero( c1 ) returns true

   const int i2 = 1;                // isZero( i2 ) returns false
   double    d2 = -1.0;             // isZero( d2 ) returns false
   complex<double> c2( 1.0, 0.0 );  // isZero( c2 ) returns false
   \endcode
*/
template< typename Type >
BLAZE_ALWAYS_INLINE EnableIf_< IsNumeric<Type>, bool > isZero( const Type& v ) noexcept
{
   return isDefault( v );
}
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Overload of the \a isZero function for non-numeric data types.
// \ingroup math_shims
//
// \param v The value/object to be tested.
// \return \a false.
*/
template< typename Type >
BLAZE_ALWAYS_INLINE DisableIf_< IsNumeric<Type>, bool > isZero( const Type& v ) noexcept
{
   UNUSED_PARAMETER( v );

   return false;
}
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
