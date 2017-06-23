//=================================================================================================
/*!
//  \file blaze/math/simd/DerivedTypes.h
//  \brief Header file for the derived SIMD types
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

#ifndef _BLAZE_MATH_SIMD_DERIVEDTYPES_H_
#define _BLAZE_MATH_SIMD_DERIVEDTYPES_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/simd/SIMDTrait.h>
#include <blaze/system/Vectorization.h>


namespace blaze {

//=================================================================================================
//
//  DERIVED SIMD TYPES
//
//=================================================================================================

//*************************************************************************************************
/*!\brief The SIMD data type for 'char'.
// \ingroup simd
*/
typedef SIMDTrait<char>::Type  SIMDchar;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief The SIMD data type for 'signed char'.
// \ingroup simd
*/
typedef SIMDTrait<signed char>::Type  SIMDschar;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief The SIMD data type for 'unsigned char'.
// \ingroup simd
*/
typedef SIMDTrait<unsigned char>::Type  SIMDuchar;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief The SIMD data type for 'wchar_t'.
// \ingroup simd
*/
typedef SIMDTrait<wchar_t>::Type  SIMDwchar;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief The SIMD data type for 'complex<char>'.
// \ingroup simd
*/
typedef SIMDTrait< complex<char> >::Type  SIMDcchar;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief The SIMD data type for 'complex<signed char>'.
// \ingroup simd
*/
typedef SIMDTrait< complex<signed char> >::Type  SIMDcschar;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief The SIMD data type for 'complex<unsigned char>'.
// \ingroup simd
*/
typedef SIMDTrait< complex<unsigned char> >::Type  SIMDcuchar;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief The SIMD data type for 'complex<wchar_t>'.
// \ingroup simd
*/
typedef SIMDTrait< complex<wchar_t> >::Type  SIMDcwchar;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief The SIMD data type for 'short'.
// \ingroup simd
*/
typedef SIMDTrait<short>::Type  SIMDshort;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief The SIMD data type for 'unsigned short'.
// \ingroup simd
*/
typedef SIMDTrait<unsigned short>::Type  SIMDushort;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief The SIMD data type for 'complex<short>'.
// \ingroup simd
*/
typedef SIMDTrait< complex<short> >::Type  SIMDcshort;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief The SIMD data type for 'complex<unsigned short>'.
// \ingroup simd
*/
typedef SIMDTrait< complex<unsigned short> >::Type  SIMDcushort;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief The SIMD data type for 'int'.
// \ingroup simd
*/
typedef SIMDTrait<int>::Type  SIMDint;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief The SIMD data type for 'unsigned int'.
// \ingroup simd
*/
typedef SIMDTrait<unsigned int>::Type  SIMDuint;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief The SIMD data type for 'complex<int>'.
// \ingroup simd
*/
typedef SIMDTrait< complex<int> >::Type  SIMDcint;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief The SIMD data type for 'complex<unsigned int>'.
// \ingroup simd
*/
typedef SIMDTrait< complex<unsigned int> >::Type  SIMDcuint;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief The SIMD data type for 'long int'.
// \ingroup simd
*/
typedef SIMDTrait<long>::Type  SIMDlong;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief The SIMD data type for 'unsigned long int'.
// \ingroup simd
*/
typedef SIMDTrait<unsigned long>::Type  SIMDulong;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief The SIMD data type for 'complex<long int>'.
// \ingroup simd
*/
typedef SIMDTrait< complex<long> >::Type  SIMDclong;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief The SIMD data type for 'complex<unsigned long int>'.
// \ingroup simd
*/
typedef SIMDTrait< complex<unsigned long> >::Type  SIMDculong;
//*************************************************************************************************

} // namespace blaze

#endif
