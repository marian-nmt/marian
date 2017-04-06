//=================================================================================================
/*!
//  \file blaze/math/simd/Reduction.h
//  \brief Header file for the SIMD reduction functionality
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

#ifndef _BLAZE_MATH_SIMD_REDUCTION_H_
#define _BLAZE_MATH_SIMD_REDUCTION_H_


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
/*!\brief Returns the sum of all elements in the 8-bit integral complex SIMD vector.
// \ingroup simd
//
// \param a The vector to be sumed up.
// \return The sum of all vector elements.
*/
BLAZE_ALWAYS_INLINE const complex<int8_t> sum( const SIMDcint8& a ) noexcept
{
#if BLAZE_AVX2_MODE
   return complex<int8_t>( a[0] + a[1] + a[ 2] + a[ 3] + a[ 4] + a[ 5] + a[ 6] + a[ 7] +
                           a[8] + a[9] + a[10] + a[11] + a[12] + a[13] + a[14] + a[15] );
#elif BLAZE_SSE2_MODE
   return complex<int8_t>( a[0] + a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7] );
#else
   return a.value;
#endif
}
//*************************************************************************************************




//=================================================================================================
//
//  16-BIT INTEGRAL SIMD TYPES
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Returns the sum of all elements in the 16-bit integral SIMD vector.
// \ingroup simd
//
// \param a The vector to be sumed up.
// \return The sum of all vector elements.
*/
BLAZE_ALWAYS_INLINE int16_t sum( const SIMDint16& a ) noexcept
{
#if BLAZE_AVX2_MODE
   const __m256i b( _mm256_hadd_epi16( a.value, a.value ) );
   const __m256i c( _mm256_hadd_epi16( b, b ) );
   const __m256i d( _mm256_hadd_epi16( c, c ) );
   const __m128i e = _mm_add_epi16( _mm256_extracti128_si256( d, 1 )
                                  , _mm256_castsi256_si128( d ) );
   return _mm_extract_epi16( e, 0 );
#elif BLAZE_SSSE3_MODE
   const __m128i b( _mm_hadd_epi16( a.value, a.value ) );
   const __m128i c( _mm_hadd_epi16( b, b ) );
   const __m128i d( _mm_hadd_epi16( c, c ) );
   return _mm_extract_epi16( d, 0 );
#elif BLAZE_SSE2_MODE
   return a[0] + a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7];
#else
   return a.value;
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the sum of all elements in the 16-bit integral complex SIMD vector.
// \ingroup simd
//
// \param a The vector to be sumed up.
// \return The sum of all vector elements.
*/
BLAZE_ALWAYS_INLINE const complex<int16_t> sum( const SIMDcint16& a ) noexcept
{
#if BLAZE_AVX2_MODE
   return complex<int16_t>( a[0] + a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7] );
#elif BLAZE_SSE2_MODE
   return complex<int16_t>( a[0] + a[1] + a[2] + a[3] );
#else
   return a.value;
#endif
}
//*************************************************************************************************




//=================================================================================================
//
//  32-BIT INTEGRAL SIMD TYPES
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Returns the sum of all elements in the 32-bit integral SIMD vector.
// \ingroup simd
//
// \param a The vector to be sumed up.
// \return The sum of all vector elements.
*/
BLAZE_ALWAYS_INLINE int32_t sum( const SIMDint32& a ) noexcept
{
#if BLAZE_MIC_MODE
   return _mm512_reduce_add_epi32( a.value );
#elif BLAZE_AVX2_MODE
   const __m256i b( _mm256_hadd_epi32( a.value, a.value ) );
   const __m256i c( _mm256_hadd_epi32( b, b ) );
   const __m128i d = _mm_add_epi32( _mm256_extracti128_si256( c, 1 )
                                  , _mm256_castsi256_si128( c ) );
   return _mm_extract_epi32( d, 0 );
#elif BLAZE_SSSE3_MODE
   const __m128i b( _mm_hadd_epi32( a.value, a.value ) );
   const __m128i c( _mm_hadd_epi32( b, b ) );
   return _mm_cvtsi128_si32( c );
#elif BLAZE_SSE2_MODE
   return a[0] + a[1] + a[2] + a[3];
#else
   return a.value;
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the sum of all elements in the 32-bit integral complex SIMD vector.
// \ingroup simd
//
// \param a The vector to be sumed up.
// \return The sum of all vector elements.
*/
BLAZE_ALWAYS_INLINE const complex<int32_t> sum( const SIMDcint32& a ) noexcept
{
#if BLAZE_MIC_MODE
   return complex<int32_t>( a[0] + a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7] );
#elif BLAZE_AVX2_MODE
   return complex<int32_t>( a[0] + a[1] + a[2] + a[3] );
#elif BLAZE_SSE2_MODE
   return complex<int32_t>( a[0] + a[1] );
#else
   return a.value;
#endif
}
//*************************************************************************************************




//=================================================================================================
//
//  64-BIT INTEGRAL SIMD TYPES
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Returns the sum of all elements in the 64-bit integral SIMD vector.
// \ingroup simd
//
// \param a The vector to be sumed up.
// \return The sum of all vector elements.
*/
BLAZE_ALWAYS_INLINE int64_t sum( const SIMDint64& a ) noexcept
{
#if BLAZE_MIC_MODE
   return _mm512_reduce_add_epi64( a.value );
#elif BLAZE_AVX2_MODE
   return a[0] + a[1] + a[2] + a[3];
#elif BLAZE_SSE2_MODE
   return a[0] + a[1];
#else
   return a.value;
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the sum of all elements in the 64-bit integral complex SIMD vector.
// \ingroup simd
//
// \param a The vector to be sumed up.
// \return The sum of all vector elements.
*/
BLAZE_ALWAYS_INLINE const complex<int64_t> sum( const SIMDcint64& a ) noexcept
{
#if BLAZE_MIC_MODE
   return complex<int64_t>( a[0] + a[1] + a[2] + a[3] );
#elif BLAZE_AVX2_MODE
   return complex<int64_t>( a[0] + a[1] );
#elif BLAZE_SSE2_MODE
   return a[0];
#else
   return a.value;
#endif
}
//*************************************************************************************************




//=================================================================================================
//
//  32-BIT FLOATING POINT SIMD TYPES
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Returns the sum of all elements in the single precision floating point SIMD vector.
// \ingroup simd
//
// \param a The vector to be sumed up.
// \return The sum of all vector elements.
*/
BLAZE_ALWAYS_INLINE float sum( const SIMDfloat& a ) noexcept
{
#if BLAZE_MIC_MODE
   return _mm512_reduce_add_ps( a.value );
#elif BLAZE_AVX_MODE
   const __m256 b( _mm256_hadd_ps( a.value, a.value ) );
   const __m256 c( _mm256_hadd_ps( b, b ) );
   const __m128 d = _mm_add_ps( _mm256_extractf128_ps( c, 1 ), _mm256_castps256_ps128( c ) );
   return _mm_cvtss_f32( d );
#elif BLAZE_SSE3_MODE
   const __m128 b( _mm_hadd_ps( a.value, a.value ) );
   const __m128 c( _mm_hadd_ps( b, b ) );
   return _mm_cvtss_f32( c );
#elif BLAZE_SSE_MODE
   return a[0] + a[1] + a[2] + a[3];
#else
   return a.value;
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the sum of all elements in the single precision complex SIMD vector.
// \ingroup simd
//
// \param a The vector to be sumed up.
// \return The sum of all vector elements.
*/
BLAZE_ALWAYS_INLINE const complex<float> sum( const SIMDcfloat& a ) noexcept
{
#if BLAZE_MIC_MODE
   return complex<float>( a[0] + a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7] );
#elif BLAZE_AVX_MODE
   return complex<float>( a[0] + a[1] + a[2] + a[3] );
#elif BLAZE_SSE_MODE
   return complex<float>( a[0] + a[1] );
#else
   return a.value;
#endif
}
//*************************************************************************************************




//=================================================================================================
//
//  64-BIT FLOATING POINT SIMD TYPES
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Returns the sum of all elements in the double precision floating point SIMD vector.
// \ingroup simd
//
// \param a The vector to be sumed up.
// \return The sum of all vector elements.
*/
BLAZE_ALWAYS_INLINE double sum( const SIMDdouble& a ) noexcept
{
#if BLAZE_MIC_MODE
   return _mm512_reduce_add_pd( a.value );
#elif BLAZE_AVX_MODE
   const __m256d b( _mm256_hadd_pd( a.value, a.value ) );
   const __m128d c = _mm_add_pd( _mm256_extractf128_pd( b, 1 ), _mm256_castpd256_pd128( b ) );
   return _mm_cvtsd_f64( c );
#elif BLAZE_SSE3_MODE
   const __m128d b( _mm_hadd_pd( a.value, a.value ) );
   return _mm_cvtsd_f64( b );
#elif BLAZE_SSE2_MODE
   return a[0] + a[1];
#else
   return a.value;
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the sum of all elements in the double precision complex SIMD vector.
// \ingroup simd
//
// \param a The vector to be sumed up.
// \return The sum of all vector elements.
*/
BLAZE_ALWAYS_INLINE const complex<double> sum( const SIMDcdouble& a ) noexcept
{
#if BLAZE_MIC_MODE
   return complex<double>( a[0] + a[1] + a[2] + a[3] );
#elif BLAZE_AVX_MODE
   return complex<double>( a[0] + a[1] );
#elif BLAZE_SSE2_MODE
   return a[0];
#else
   return a.value;
#endif
}
//*************************************************************************************************

} // namespace blaze

#endif
