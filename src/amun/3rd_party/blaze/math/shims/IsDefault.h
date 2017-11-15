//=================================================================================================
/*!
//  \file blaze/math/shims/IsDefault.h
//  \brief Header file for the isDefault shim
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

#ifndef _BLAZE_MATH_SHIMS_ISDEFAULT_H_
#define _BLAZE_MATH_SHIMS_ISDEFAULT_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <cmath>
#include <blaze/math/Accuracy.h>
#include <blaze/system/Inline.h>
#include <blaze/util/Complex.h>
#include <blaze/util/typetraits/IsBuiltin.h>


namespace blaze {

//=================================================================================================
//
//  ISDEFAULT SHIM
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Returns whether the given value/object is in default state.
// \ingroup math_shims
//
// \param v The value/object to be tested for its default state.
// \return \a true in case the given value/object is in its default state, \a false otherwise.
//
// The \a isDefault shim represents an abstract interface for testing a value/object whether
// it is in its default state or not. In case the value/object is in its default state, the
// function returns \a true, otherwise it returns \a false. For built-in data types, the
// function returns \a true in case the current value is zero.

   \code
   const int i = 0;          // isDefault( i ) returns true
   double    d = 2.0;        // isDefault( d ) returns false
   Vec3      v1;             // isDefault( v1 ) returns true
   Vec3      v2( 0, 0, 0 );  // isDefault( v2 ) returns true since (0,0,0) is the default state
   Vec3      v3( 1, 2, 3 );  // isDefault( v3 ) returns false
   \endcode
*/
template< typename Type >
BLAZE_ALWAYS_INLINE bool isDefault( const Type& v ) noexcept( IsBuiltin<Type>::value )
{
   return v == Type();
}
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the given single precision floating point value is zero.
// \ingroup math_shims
//
// \param v The single precision floating point value to be tested for zero.
// \return \a true in case the given value is zero, \a false otherwise.
//
// This overload of the \a isDefault shim tests whether the given single precision floating point
// value is exactly zero or within an epsilon range to zero. In case the value is zero or close
// to zero the function returns \a true, otherwise it returns \a false.
*/
BLAZE_ALWAYS_INLINE bool isDefault( float v ) noexcept
{
   return std::fabs( v ) <= accuracy;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the given double precision floating point value is zero.
// \ingroup math_shims
//
// \param v The double precision floating point value to be tested for zero.
// \return \a true in case the given value is zero, \a false otherwise.
//
// This overload of the \a isDefault shim tests whether the given double precision floating point
// value is exactly zero or within an epsilon range to zero. In case the value is zero or close
// to zero the function returns \a true, otherwise it returns \a false.
*/
BLAZE_ALWAYS_INLINE bool isDefault( double v ) noexcept
{
   return std::fabs( v ) <= accuracy;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the given extended precision floating point value is zero.
// \ingroup math_shims
//
// \param v The extended precision floating point value to be tested for zero.
// \return \a true in case the given value is zero, \a false otherwise.
//
// This overload of the \a isDefault shim tests whether the given extended precision floating
// point value is exactly zero or within an epsilon range to zero. In case the value is zero or
// close to zero the function returns \a true, otherwise it returns \a false.
*/
BLAZE_ALWAYS_INLINE bool isDefault( long double v ) noexcept
{
   return std::fabs( v ) <= accuracy;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the given complex number is zero.
// \ingroup math_shims
//
// \param v The complex number to be tested for zero.
// \return \a true in case the given value is zero, \a false otherwise.
//
// This overload of the \a isDefault shim tests whether both the real and the imaginary part of
// the given complex number are exactly zero or within an epsilon range to zero. In case the both
// parts are zero or close to zero the function returns \a true, otherwise it returns \a false.
*/
template< typename T >
BLAZE_ALWAYS_INLINE bool isDefault( const complex<T>& v ) noexcept( IsBuiltin<T>::value )
{
   return isDefault( real( v ) ) && isDefault( imag( v ) );
}
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
