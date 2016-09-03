//=================================================================================================
/*!
//  \file blaze/math/simd/Division.h
//  \brief Header file for the SIMD division functionality
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

#ifndef _BLAZE_MATH_SIMD_DIVISION_H_
#define _BLAZE_MATH_SIMD_DIVISION_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/simd/BasicTypes.h>
#include <blaze/system/Inline.h>
#include <blaze/system/Vectorization.h>


namespace blaze {

//=================================================================================================
//
//  32-BIT INTEGRAL SIMD TYPES
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Division of two vectors of 32-bit signed integral SIMD values.
// \ingroup simd
//
// \param a The left-hand side SIMD operand.
// \param b The right-hand side SIMD operand.
// \return The result of the division.
//
// This operation is only available for AVX-512.
*/
BLAZE_ALWAYS_INLINE const SIMDint32
   operator/( const SIMDint32& a, const SIMDint32& b ) noexcept
#if BLAZE_MIC_MODE
{
   return _mm512_div_epi32( a.value, b.value );
}
#else
= delete;
#endif
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Scaling of a vector of 32-bit signed integral complex SIMD values.
// \ingroup simd
//
// \param a The left-hand side complex values to be scaled.
// \param b The right-hand side scalars.
// \return The result of the division.
//
// This operation is only available for AVX-512.
*/
BLAZE_ALWAYS_INLINE const SIMDcint32
   operator/( const SIMDcint32& a, const SIMDint32& b ) noexcept
#if BLAZE_MIC_MODE
{
   return _mm512_div_epi32( a.value, b.value );
}
#else
= delete;
#endif
//*************************************************************************************************




//=================================================================================================
//
//  64-BIT INTEGRAL SIMD TYPES
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Division of two vectors of 64-bit signed integral SIMD values.
// \ingroup simd
//
// \param a The left-hand side SIMD operand.
// \param b The right-hand side SIMD operand.
// \return The result of the division.
//
// This operation is only available for AVX-512.
*/
BLAZE_ALWAYS_INLINE const SIMDint64
   operator/( const SIMDint64& a, const SIMDint64& b ) noexcept
#if BLAZE_MIC_MODE
{
   return _mm512_div_epi64( a.value, b.value );
}
#else
= delete;
#endif
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Scaling of a vector of 64-bit signed integral complex SIMD values.
// \ingroup simd
//
// \param a The left-hand side complex values to be scaled.
// \param b The right-hand side scalars.
// \return The result of the division.
//
// This operation is only available for AVX-512.
*/
BLAZE_ALWAYS_INLINE const SIMDcint64
   operator/( const SIMDcint64& a, const SIMDint64& b ) noexcept
#if BLAZE_MIC_MODE
{
   return _mm512_div_epi64( (~a).value, (~b).value );
}
#else
= delete;
#endif
//*************************************************************************************************




//=================================================================================================
//
//  32-BIT FLOATING POINT SIMD TYPES
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Division of two vectors of single precision floating point SIMD values.
// \ingroup simd
//
// \param a The left-hand side SIMD operand.
// \param b The right-hand side SIMD operand.
// \return The result of the division.
//
// This operation is only available for SSE, AVX, and AVX-512.
*/
template< typename T1    // Type of the left-hand side operand
        , typename T2 >  // Type of the right-hand side operand
BLAZE_ALWAYS_INLINE const SIMDfloat
   operator/( const SIMDf32<T1>& a, const SIMDf32<T2>& b ) noexcept
#if BLAZE_MIC_MODE
{
   return _mm512_div_ps( (~a).eval().value, (~b).eval().value );
}
#elif BLAZE_AVX_MODE
{
   return _mm256_div_ps( (~a).eval().value, (~b).eval().value );
}
#elif BLAZE_SSE_MODE
{
   return _mm_div_ps( (~a).eval().value, (~b).eval().value );
}
#else
= delete;
#endif
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Scaling of a vector of single precision floating point values complex SIMD values.
// \ingroup simd
//
// \param a The left-hand side complex values to be scaled.
// \param b The right-hand side scalars.
// \return The result of the division.
//
// This operation is only available for SSE, AVX, and AVX-512.
*/
BLAZE_ALWAYS_INLINE const SIMDcfloat
   operator/( const SIMDcfloat& a, const SIMDfloat& b ) noexcept
#if BLAZE_MIC_MODE
{
   return _mm512_div_ps( a.value, b.value );
}
#elif BLAZE_AVX_MODE
{
   return _mm256_div_ps( a.value, b.value );
}
#elif BLAZE_SSE_MODE
{
   return _mm_div_ps( a.value, b.value );
}
#else
= delete;
#endif
//*************************************************************************************************




//=================================================================================================
//
//  64-BIT FLOATING POINT SIMD TYPES
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Division of two vectors of double precision floating point SIMD values.
// \ingroup simd
//
// \param a The left-hand side SIMD operand.
// \param b The right-hand side SIMD operand.
// \return The result of the division.
//
// This operation is only available for SSE, AVX, and AVX-512.
*/
template< typename T1    // Type of the left-hand side operand
        , typename T2 >  // Type of the right-hand side operand
BLAZE_ALWAYS_INLINE const SIMDdouble
   operator/( const SIMDf64<T1>& a, const SIMDf64<T2>& b ) noexcept
#if BLAZE_MIC_MODE
{
   return _mm512_div_pd( (~a).eval().value, (~b).eval().value );
}
#elif BLAZE_AVX_MODE
{
   return _mm256_div_pd( (~a).eval().value, (~b).eval().value );
}
#elif BLAZE_SSE2_MODE
{
   return _mm_div_pd( (~a).eval().value, (~b).eval().value );
}
#else
= delete;
#endif
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Scaling of a vector of double precision floating point values complex SIMD values.
// \ingroup simd
//
// \param a The left-hand side complex values to be scaled.
// \param b The right-hand side scalars.
// \return The result of the division.
//
// This operation is only available for SSE, AVX, and AVX-512.
*/
BLAZE_ALWAYS_INLINE const SIMDcdouble
   operator/( const SIMDcdouble& a, const SIMDdouble& b ) noexcept
#if BLAZE_MIC_MODE
{
   return _mm512_div_pd( a.value, b.value );
}
#elif BLAZE_AVX_MODE
{
   return _mm256_div_pd( a.value, b.value );
}
#elif BLAZE_SSE2_MODE
{
   return _mm_div_pd( a.value, b.value );
}
#else
= delete;
#endif
//*************************************************************************************************

} // namespace blaze

#endif
