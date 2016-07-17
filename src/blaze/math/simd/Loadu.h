//=================================================================================================
/*!
//  \file blaze/math/simd/Loadu.h
//  \brief Header file for the SIMD unaligned load functionality
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

#ifndef _BLAZE_MATH_SIMD_LOADU_H_
#define _BLAZE_MATH_SIMD_LOADU_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/simd/BasicTypes.h>
#include <blaze/system/Inline.h>
#include <blaze/system/Vectorization.h>
#include <blaze/util/Complex.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/mpl/And.h>
#include <blaze/util/mpl/If.h>
#include <blaze/util/StaticAssert.h>
#include <blaze/util/typetraits/HasSize.h>
#include <blaze/util/typetraits/IsIntegral.h>
#include <blaze/util/typetraits/IsSigned.h>


namespace blaze {

//=================================================================================================
//
//  8-BIT INTEGRAL SIMD TYPES
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Loads a vector of 1-byte integral values.
// \ingroup simd
//
// \param address The first integral value to be loaded.
// \return The loaded vector of integral values.
//
// This function loads a vector of 1-byte integral values. In contrast to the according \c loada()
// function, the given address is not required to be properly aligned.
*/
template< typename T >  // Type of the integral value
BLAZE_ALWAYS_INLINE const EnableIf_< And< IsIntegral<T>, HasSize<T,1UL> >
                                   , If_< IsSigned<T>, SIMDint8, SIMDuint8 > >
   loadu( const T* address ) noexcept
{
#if BLAZE_AVX2_MODE
   return _mm256_loadu_si256( reinterpret_cast<const __m256i*>( address ) );
#elif BLAZE_SSE2_MODE
   return _mm_loadu_si128( reinterpret_cast<const __m128i*>( address ) );
#else
   return *address;
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Loads a vector of 1-byte integral complex values.
// \ingroup simd
//
// \param address The first integral complex value to be loaded.
// \return The loaded vector of integral complex values.
//
// This function loads a vector of 1-byte integral complex values. In contrast to the according
// \c loada() function, the given address is not required to be properly aligned.
*/
template< typename T >  // Type of the integral value
BLAZE_ALWAYS_INLINE const EnableIf_< And< IsIntegral<T>, HasSize<T,1UL> >
                                   , If_< IsSigned<T>, SIMDcint8, SIMDcuint8 > >
   loadu( const complex<T>* address ) noexcept
{
   BLAZE_STATIC_ASSERT( sizeof( complex<T> ) == 2UL*sizeof( T ) );

#if BLAZE_AVX2_MODE
   return _mm256_loadu_si256( reinterpret_cast<const __m256i*>( address ) );
#elif BLAZE_SSE2_MODE
   return _mm_loadu_si128( reinterpret_cast<const __m128i*>( address ) );
#else
   return *address;
#endif
}
//*************************************************************************************************




//=================================================================================================
//
//  16-BIT INTEGRAL SIMD TYPES
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Loads a vector of 2-byte integral values.
// \ingroup simd
//
// \param address The first integral value to be loaded.
// \return The loaded vector of integral values.
//
// This function loads a vector of 2-byte integral values. In contrast to the according \c loada()
// function, the given address is not required to be properly aligned.
*/
template< typename T >  // Type of the integral value
BLAZE_ALWAYS_INLINE const EnableIf_< And< IsIntegral<T>, HasSize<T,2UL> >
                                   , If_< IsSigned<T>, SIMDint16, SIMDuint16 > >
   loadu( const T* address ) noexcept
{
#if BLAZE_AVX2_MODE
   return _mm256_loadu_si256( reinterpret_cast<const __m256i*>( address ) );
#elif BLAZE_SSE2_MODE
   return _mm_loadu_si128( reinterpret_cast<const __m128i*>( address ) );
#else
   return *address;
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Loads a vector of 2-byte integral complex values.
// \ingroup simd
//
// \param address The first integral complex value to be loaded.
// \return The loaded vector of integral complex values.
//
// This function loads a vector of 2-byte integral complex values. In contrast to the according
// \c loada() function, the given address is not required to be properly aligned.
*/
template< typename T >  // Type of the integral value
BLAZE_ALWAYS_INLINE const EnableIf_< And< IsIntegral<T>, HasSize<T,2UL> >
                                   , If_< IsSigned<T>, SIMDcint16, SIMDcuint16 > >
   loadu( const complex<T>* address ) noexcept
{
   BLAZE_STATIC_ASSERT( sizeof( complex<T> ) == 2UL*sizeof( T ) );

#if BLAZE_AVX2_MODE
   return _mm256_loadu_si256( reinterpret_cast<const __m256i*>( address ) );
#elif BLAZE_SSE2_MODE
   return _mm_loadu_si128( reinterpret_cast<const __m128i*>( address ) );
#else
   return *address;
#endif
}
//*************************************************************************************************




//=================================================================================================
//
//  32-BIT INTEGRAL SIMD TYPES
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Loads a vector of 4-byte integral values.
// \ingroup simd
//
// \param address The first integral value to be loaded.
// \return The loaded vector of integral values.
//
// This function loads a vector of 4-byte integral values. In contrast to the according \c loada()
// function, the given address is not required to be properly aligned.
*/
template< typename T >  // Type of the integral value
BLAZE_ALWAYS_INLINE const EnableIf_< And< IsIntegral<T>, HasSize<T,4UL> >
                                   , If_< IsSigned<T>, SIMDint32, SIMDuint32 > >
   loadu( const T* address ) noexcept
{
#if BLAZE_MIC_MODE
   __m512i v1 = _mm512_setzero_epi32();
   v1 = _mm512_loadunpacklo_epi32( v1, address );
   v1 = _mm512_loadunpackhi_epi32( v1, address+16UL );
   return v1;
#elif BLAZE_AVX2_MODE
   return _mm256_loadu_si256( reinterpret_cast<const __m256i*>( address ) );
#elif BLAZE_SSE2_MODE
   return _mm_loadu_si128( reinterpret_cast<const __m128i*>( address ) );
#else
   return *address;
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Loads a vector of 4-byte integral complex values.
// \ingroup simd
//
// \param address The first integral complex value to be loaded.
// \return The loaded vector of integral complex values.
//
// This function loads a vector of 4-byte integral complex values. In contrast to the according
// \c loada() function, the given address is not required to be properly aligned.
*/
template< typename T >  // Type of the integral value
BLAZE_ALWAYS_INLINE const EnableIf_< And< IsIntegral<T>, HasSize<T,4UL> >
                                   , If_< IsSigned<T>, SIMDcint32, SIMDcuint32 > >
   loadu( const complex<T>* address ) noexcept
{
   BLAZE_STATIC_ASSERT( sizeof( complex<T> ) == 2UL*sizeof( T ) );

#if BLAZE_MIC_MODE
   __m512i v1 = _mm512_setzero_epi32();
   v1 = _mm512_loadunpacklo_epi32( v1, address );
   v1 = _mm512_loadunpackhi_epi32( v1, address+8UL );
   return v1;
#elif BLAZE_AVX2_MODE
   return _mm256_loadu_si256( reinterpret_cast<const __m256i*>( address ) );
#elif BLAZE_SSE2_MODE
   return _mm_loadu_si128( reinterpret_cast<const __m128i*>( address ) );
#else
   return *address;
#endif
}
//*************************************************************************************************




//=================================================================================================
//
//  64-BIT INTEGRAL SIMD TYPES
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Loads a vector of 8-byte integral values.
// \ingroup simd
//
// \param address The first integral value to be loaded.
// \return The loaded vector of integral values.
//
// This function loads a vector of 8-byte integral values. In contrast to the according \c loada()
// function, the given address is not required to be properly aligned.
*/
template< typename T >  // Type of the integral value
BLAZE_ALWAYS_INLINE const EnableIf_< And< IsIntegral<T>, HasSize<T,8UL> >
                                   , If_< IsSigned<T>, SIMDint64, SIMDuint64 > >
   loadu( const T* address ) noexcept
{
#if BLAZE_MIC_MODE
   __m512i v1 = _mm512_setzero_epi32();
   v1 = _mm512_loadunpacklo_epi64( v1, address );
   v1 = _mm512_loadunpackhi_epi64( v1, address+8UL );
   return v1;
#elif BLAZE_AVX2_MODE
   return _mm256_loadu_si256( reinterpret_cast<const __m256i*>( address ) );
#elif BLAZE_SSE2_MODE
   return _mm_loadu_si128( reinterpret_cast<const __m128i*>( address ) );
#else
   return *address;
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Loads a vector of 8-byte integral complex values.
// \ingroup simd
//
// \param address The first integral complex value to be loaded.
// \return The loaded vector of integral complex values.
//
// This function loads a vector of 8-byte integral complex values. In contrast to the according
// \c loada() function, the given address is not required to be properly aligned.
*/
template< typename T >  // Type of the integral value
BLAZE_ALWAYS_INLINE const EnableIf_< And< IsIntegral<T>, HasSize<T,8UL> >
                                   , If_< IsSigned<T>, SIMDcint64, SIMDcuint64 > >
   loadu( const complex<T>* address ) noexcept
{
   BLAZE_STATIC_ASSERT( sizeof( complex<T> ) == 2UL*sizeof( T ) );

#if BLAZE_MIC_MODE
   __m512i v1 = _mm512_setzero_epi32();
   v1 = _mm512_loadunpacklo_epi64( v1, address );
   v1 = _mm512_loadunpackhi_epi64( v1, address+4UL );
   return v1;
#elif BLAZE_AVX2_MODE
   return _mm256_loadu_si256( reinterpret_cast<const __m256i*>( address ) );
#elif BLAZE_SSE2_MODE
   return _mm_loadu_si128( reinterpret_cast<const __m128i*>( address ) );
#else
   return If_< IsSigned<T>, SIMDcint64, SIMDcuint64 >( *address );
#endif
}
//*************************************************************************************************




//=================================================================================================
//
//  32-BIT FLOATING POINT SIMD TYPES
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Loads a vector of 'float' values.
// \ingroup simd
//
// \param address The first 'float' value to be loaded.
// \return The loaded vector of 'float' values.
//
// This function loads a vector of 'float' values. In contrast to the according \c loada()
// function, the given address is not required to be properly aligned.
*/
BLAZE_ALWAYS_INLINE const SIMDfloat loadu( const float* address ) noexcept
{
#if BLAZE_MIC_MODE
   __m512 v1 = _mm512_setzero_ps();
   v1 = _mm512_loadunpacklo_ps( v1, address );
   v1 = _mm512_loadunpackhi_ps( v1, address+16UL );
   return v1;
#elif BLAZE_AVX_MODE
   return _mm256_loadu_ps( address );
#elif BLAZE_SSE_MODE
   return _mm_loadu_ps( address );
#else
   return *address;
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Loads a vector of 'complex<float>' values.
// \ingroup simd
//
// \param address The first 'complex<float>' value to be loaded.
// \return The loaded vector of 'complex<float>' values.
//
// This function loads a vector of 'complex<float>' values. In contrast to the according \c loada()
// function, the given address is not required to be properly aligned.
*/
BLAZE_ALWAYS_INLINE const SIMDcfloat loadu( const complex<float>* address ) noexcept
{
   BLAZE_STATIC_ASSERT( sizeof( complex<float> ) == 2UL*sizeof( float ) );

#if BLAZE_MIC_MODE
   __m512 v1 = _mm512_setzero_ps();
   v1 = _mm512_loadunpacklo_ps( v1, reinterpret_cast<const float*>( address     ) );
   v1 = _mm512_loadunpackhi_ps( v1, reinterpret_cast<const float*>( address+8UL ) );
   return v1;
#elif BLAZE_AVX_MODE
   return _mm256_loadu_ps( reinterpret_cast<const float*>( address ) );
#elif BLAZE_SSE_MODE
   return _mm_loadu_ps( reinterpret_cast<const float*>( address ) );
#else
   return *address;
#endif
}
//*************************************************************************************************




//=================================================================================================
//
//  64-BIT FLOATING POINT SIMD TYPES
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Loads a vector of 'double' values.
// \ingroup simd
//
// \param address The first 'double' value to be loaded.
// \return The loaded vector of 'double' values.
//
// This function loads a vector of 'double' values. In contrast to the according \c loada()
// function, the given address is not required to be properly aligned.
*/
BLAZE_ALWAYS_INLINE const SIMDdouble loadu( const double* address ) noexcept
{
#if BLAZE_MIC_MODE
   __m512d v1 = _mm512_setzero_pd();
   v1 = _mm512_loadunpacklo_pd( v1, address );
   v1 = _mm512_loadunpackhi_pd( v1, address+8UL );
   return v1;
#elif BLAZE_AVX_MODE
   return _mm256_loadu_pd( address );
#elif BLAZE_SSE2_MODE
   return _mm_loadu_pd( address );
#else
   return *address;
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Loads a vector of 'complex<double>' values.
// \ingroup simd
//
// \param address The first 'complex<double>' value to be loaded.
// \return The loaded vector of 'complex<double>' values.
//
// This function loads a vector of 'complex<double>' values. In contrast to the according
// \c loada() function, the given address is not required to be properly aligned.
*/
BLAZE_ALWAYS_INLINE const SIMDcdouble loadu( const complex<double>* address ) noexcept
{
   BLAZE_STATIC_ASSERT( sizeof( complex<double> ) == 2UL*sizeof( double ) );

#if BLAZE_MIC_MODE
   __m512d v1 = _mm512_setzero_pd();
   v1 = _mm512_loadunpacklo_pd( v1, reinterpret_cast<const double*>( address     ) );
   v1 = _mm512_loadunpackhi_pd( v1, reinterpret_cast<const double*>( address+4UL ) );
   return v1;
#elif BLAZE_AVX_MODE
   return _mm256_loadu_pd( reinterpret_cast<const double*>( address ) );
#elif BLAZE_SSE2_MODE
   return _mm_loadu_pd( reinterpret_cast<const double*>( address ) );
#else
   return *address;
#endif
}
//*************************************************************************************************

} // namespace blaze

#endif
