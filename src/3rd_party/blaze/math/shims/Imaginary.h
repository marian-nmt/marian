//=================================================================================================
/*!
//  \file blaze/math/shims/Imaginary.h
//  \brief Header file for the imaginary shim
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

#ifndef _BLAZE_MATH_SHIMS_IMAGINARY_H_
#define _BLAZE_MATH_SHIMS_IMAGINARY_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/system/Inline.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/typetraits/IsBuiltin.h>
#include <blaze/util/Unused.h>


namespace blaze {

//=================================================================================================
//
//  IMAGINARY SHIM
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Computing the imaginary part of the given value/object.
// \ingroup math_shims
//
// \param a The given value/object.
// \return The imaginary part of the given value.
//
// The \a imag shim represents an abstract interface for the computation of the imaginary part
// of any given data type. In case the given value is of complex type the function returns the
// imaginary part:

   \code
   const blaze::complex<double> a( 3.0, -2.0 );
   const double b( imag( a ) );  // Results in -2.0
   \endcode

// Values of built-in data type are considered complex numbers with an imaginary part of 0. Thus
// the returned value is 0:

   \code
   const double a( -3.0 );
   const double b( imag( a ) );  // Results in 0.0
   \endcode
*/
template< typename T >
BLAZE_ALWAYS_INLINE EnableIf_< IsBuiltin<T>, T > imag( T a ) noexcept
{
   UNUSED_PARAMETER( a );

   return T(0);
}
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
