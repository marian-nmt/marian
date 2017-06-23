//=================================================================================================
/*!
//  \file blaze/math/simd/Setzero.h
//  \brief Header file for the SIMD setzero functionality
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

#ifndef _BLAZE_MATH_SIMD_SETZERO_H_
#define _BLAZE_MATH_SIMD_SETZERO_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/simd/BasicTypes.h>
#include <blaze/system/Inline.h>
#include <blaze/system/Vectorization.h>


namespace blaze {

//=================================================================================================
//
//  8-BIT INTEGRAL SIMD TYPES
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Setting an integral SIMD type with 8-bit data values to zero.
// \ingroup simd
//
// \param value The value to be set to zero.
// \return void
*/
template< typename T >  // Type of the SIMD element
BLAZE_ALWAYS_INLINE void setzero( SIMDi8<T>& value ) noexcept
{
#if BLAZE_AVX2_MODE
   (~value).value = _mm256_setzero_si256();
#elif BLAZE_SSE2_MODE
   (~value).value = _mm_setzero_si128();
#else
   (~value).value = 0;
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Setting an integral SIMD type with 8-bit complex values to zero.
// \ingroup simd
//
// \param value The value to be set to zero.
// \return void
*/
template< typename T >  // Type of the SIMD element
BLAZE_ALWAYS_INLINE void setzero( SIMDci8<T>& value ) noexcept
{
#if BLAZE_AVX2_MODE
   (~value).value = _mm256_setzero_si256();
#elif BLAZE_SSE2_MODE
   (~value).value = _mm_setzero_si128();
#else
   (~value).value = 0;
#endif
}
//*************************************************************************************************




//=================================================================================================
//
//  16-BIT INTEGRAL SIMD TYPES
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Setting an integral SIMD type with 16-bit data values to zero.
// \ingroup simd
//
// \param value The value to be set to zero.
// \return void
*/
template< typename T >  // Type of the SIMD element
BLAZE_ALWAYS_INLINE void setzero( SIMDi16<T>& value ) noexcept
{
#if BLAZE_AVX2_MODE
   (~value).value = _mm256_setzero_si256();
#elif BLAZE_SSE2_MODE
   (~value).value = _mm_setzero_si128();
#else
   (~value).value = 0;
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Setting an integral SIMD type with 16-bit complex values to zero.
// \ingroup simd
//
// \param value The value to be set to zero.
// \return void
*/
template< typename T >  // Type of the SIMD element
BLAZE_ALWAYS_INLINE void setzero( SIMDci16<T>& value ) noexcept
{
#if BLAZE_AVX2_MODE
   (~value).value = _mm256_setzero_si256();
#elif BLAZE_SSE2_MODE
   (~value).value = _mm_setzero_si128();
#else
   (~value).value = 0;
#endif
}
//*************************************************************************************************




//=================================================================================================
//
//  32-BIT INTEGRAL SIMD TYPES
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Setting an integral SIMD type with 32-bit data values to zero.
// \ingroup simd
//
// \param value The value to be set to zero.
// \return void
*/
template< typename T >  // Type of the SIMD element
BLAZE_ALWAYS_INLINE void setzero( SIMDi32<T>& value ) noexcept
{
#if BLAZE_MIC_MODE
   (~value).value = _mm512_setzero_epi32();
#elif BLAZE_AVX2_MODE
   (~value).value = _mm256_setzero_si256();
#elif BLAZE_SSE2_MODE
   (~value).value = _mm_setzero_si128();
#else
   (~value).value = 0;
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Setting an integral SIMD type with 32-bit complex values to zero.
// \ingroup simd
//
// \param value The value to be set to zero.
// \return void
*/
template< typename T >  // Type of the SIMD element
BLAZE_ALWAYS_INLINE void setzero( SIMDci32<T>& value ) noexcept
{
#if BLAZE_MIC_MODE
   (~value).value = _mm512_setzero_epi32();
#elif BLAZE_AVX2_MODE
   (~value).value = _mm256_setzero_si256();
#elif BLAZE_SSE2_MODE
   (~value).value = _mm_setzero_si128();
#else
   (~value).value = 0;
#endif
}
//*************************************************************************************************




//=================================================================================================
//
//  64-BIT INTEGRAL SIMD TYPES
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Setting an integral SIMD type with 64-bit data values to zero.
// \ingroup simd
//
// \param value The value to be set to zero.
// \return void
*/
template< typename T >  // Type of the SIMD element
BLAZE_ALWAYS_INLINE void setzero( SIMDi64<T>& value ) noexcept
{
#if BLAZE_MIC_MODE
   (~value).value = _mm512_setzero_epi32();
#elif BLAZE_AVX2_MODE
   (~value).value = _mm256_setzero_si256();
#elif BLAZE_SSE2_MODE
   (~value).value = _mm_setzero_si128();
#else
   (~value).value = 0;
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Setting an integral SIMD type with 64-bit complex values to zero.
// \ingroup simd
//
// \param value The value to be set to zero.
// \return void
*/
template< typename T >  // Type of the SIMD element
BLAZE_ALWAYS_INLINE void setzero( SIMDci64<T>& value ) noexcept
{
#if BLAZE_MIC_MODE
   (~value).value = _mm512_setzero_epi32();
#elif BLAZE_AVX2_MODE
   (~value).value = _mm256_setzero_si256();
#elif BLAZE_SSE2_MODE
   (~value).value = _mm_setzero_si128();
#else
   (~value).value = 0;
#endif
}
//*************************************************************************************************




//=================================================================================================
//
//  32-BIT FLOATING POINT SIMD TYPES
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Setting a floating point SIMD type with 32-bit single precision data values to zero.
// \ingroup simd
//
// \param value The value to be set to zero.
// \return void
*/
BLAZE_ALWAYS_INLINE void setzero( SIMDfloat& value ) noexcept
{
#if BLAZE_MIC_MODE
   value.value = _mm512_setzero_ps();
#elif BLAZE_AVX_MODE
   value.value = _mm256_setzero_ps();
#elif BLAZE_SSE_MODE
   value.value = _mm_setzero_ps();
#else
   value.value = 0.0F;
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Setting a floating point SIMD type with 32-bit single precision complex values to zero.
// \ingroup simd
//
// \param value The value to be set to zero.
// \return void
*/
BLAZE_ALWAYS_INLINE void setzero( SIMDcfloat& value ) noexcept
{
#if BLAZE_MIC_MODE
   value.value = _mm512_setzero_ps();
#elif BLAZE_AVX_MODE
   value.value = _mm256_setzero_ps();
#elif BLAZE_SSE_MODE
   value.value = _mm_setzero_ps();
#else
   value.value = 0.0F;
#endif
}
//*************************************************************************************************




//=================================================================================================
//
//  64-BIT FLOATING POINT SIMD TYPES
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Setting a floating point SIMD type with 64-bit double precision data values to zero.
// \ingroup simd
//
// \param value The value to be set to zero.
// \return void
*/
BLAZE_ALWAYS_INLINE void setzero( SIMDdouble& value ) noexcept
{
#if BLAZE_MIC_MODE
   value.value = _mm512_setzero_pd();
#elif BLAZE_AVX_MODE
   value.value = _mm256_setzero_pd();
#elif BLAZE_SSE2_MODE
   value.value = _mm_setzero_pd();
#else
   value.value = 0.0;
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Setting a floating point SIMD type with 32-bit double precision complex values to zero.
// \ingroup simd
//
// \param value The value to be set to zero.
// \return void
*/
BLAZE_ALWAYS_INLINE void setzero( SIMDcdouble& value ) noexcept
{
#if BLAZE_MIC_MODE
   value.value = _mm512_setzero_pd();
#elif BLAZE_AVX_MODE
   value.value = _mm256_setzero_pd();
#elif BLAZE_SSE2_MODE
   value.value = _mm_setzero_pd();
#else
   value.value = 0.0;
#endif
}
//*************************************************************************************************

} // namespace blaze

#endif
