//=================================================================================================
/*!
//  \file blaze/math/shims/IsNaN.h
//  \brief Header file for the isnan shim
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

#ifndef _BLAZE_MATH_SHIMS_ISNAN_H_
#define _BLAZE_MATH_SHIMS_ISNAN_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/system/Inline.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/typetraits/IsNumeric.h>


//*************************************************************************************************
// Macro undefinition
//*************************************************************************************************

#ifdef isnan
#  undef isnan
#endif


namespace blaze {

//=================================================================================================
//
//  ISNAN SHIM
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Platform independent implementation of the C99 \a isnan function.
// \ingroup math_shims
//
// \param a Value to be checked.
// \return \a true if \a a is not a number (NaN), \a false otherwise.
//
// This function provides a platform independent check for NaN values. In contrast to the \a isnan
// function from the C standard, which is only supporting all floating point types, this function
// can be used for all numeric data types (i.e. all integral, floating point, and complex data
// types). The function returns \a true in case the given value is not a number (NaN). In all
// other cases the function returns \a false.
*/
template< typename T >
BLAZE_ALWAYS_INLINE EnableIf_< IsNumeric<T>, bool > isnan( T a ) noexcept
{
   return a != a;
}
//*************************************************************************************************

} // namespace blaze

#endif
