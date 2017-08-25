//=================================================================================================
/*!
//  \file blaze/math/simd/Multiplication.h
//  \brief Header file for the SIMD multiplication functionality
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

#ifndef _BLAZE_MATH_SIMD_MULTIPLICATION_H_
#define _BLAZE_MATH_SIMD_MULTIPLICATION_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/simd/BasicTypes.h>
#include <blaze/system/Inline.h>
#include <blaze/system/Vectorization.h>


namespace blaze {

//=================================================================================================
//
//  16-BIT INTEGRAL SIMD TYPES
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Multiplication of two vectors of 16-bit integral SIMD values of the same type.
// \ingroup simd
//
// \param a The left-hand side SIMD operand.
// \param b The right-hand side SIMD operand.
// \return The result of the multiplication.
//
// This operation is only available for SSE2 and AVX2.
*/
template< typename T >  // Type of both operands
BLAZE_ALWAYS_INLINE const T
   operator*( const SIMDi16<T>& a, const SIMDi16<T>& b ) noexcept
#if BLAZE_AVX2_MODE
{
   return _mm256_mullo_epi16( (~a).value, (~b).value );
}
#elif BLAZE_SSE2_MODE
{
   return _mm_mullo_epi16( (~a).value, (~b).value );
}
#else
= delete;
#endif
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Multiplication of two vectors of 16-bit integral SIMD values of different type.
// \ingroup simd
//
// \param a The left-hand side SIMD operand.
// \param b The right-hand side SIMD operand.
// \return The result of the multiplication.
//
// This operation is only available for SSE2 and AVX2.
*/
template< typename T1    // Type of the left-hand side operand
        , typename T2 >  // Type of the right-hand side operand
BLAZE_ALWAYS_INLINE const SIMDuint16
   operator*( const SIMDi16<T1>& a, const SIMDi16<T2>& b ) noexcept
#if BLAZE_AVX2_MODE
{
   return _mm256_mullo_epi16( (~a).value, (~b).value );
}
#elif BLAZE_SSE2_MODE
{
   return _mm_mullo_epi16( (~a).value, (~b).value );
}
#else
= delete;
#endif
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Scaling of a vector of 16-bit signed integral complex SIMD values.
// \ingroup simd
//
// \param a The left-hand side complex values to be scaled.
// \param b The right-hand side scalars.
// \return The result of the scaling operation.
//
// This operation is only available for SSE2 and AVX2.
*/
BLAZE_ALWAYS_INLINE const SIMDcint16
   operator*( const SIMDcint16& a, const SIMDint16& b ) noexcept
#if BLAZE_AVX2_MODE
{
   return _mm256_mullo_epi16( (~a).value, (~b).value );
}
#elif BLAZE_SSE2_MODE
{
   return _mm_mullo_epi16( (~a).value, (~b).value );
}
#else
= delete;
#endif
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Scaling of a vector of 16-bit unsigned integral complex SIMD values.
// \ingroup simd
//
// \param a The left-hand side complex values to be scaled.
// \param b The right-hand side scalars.
// \return The result of the scaling operation.
//
// This operation is only available for SSE2 and AVX2.
*/
BLAZE_ALWAYS_INLINE const SIMDcuint16
   operator*( const SIMDcuint16& a, const SIMDuint16& b ) noexcept
#if BLAZE_AVX2_MODE
{
   return _mm256_mullo_epi16( (~a).value, (~b).value );
}
#elif BLAZE_SSE2_MODE
{
   return _mm_mullo_epi16( (~a).value, (~b).value );
}
#else
= delete;
#endif
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Scaling of a vector of 16-bit signed integral complex SIMD values.
// \ingroup simd
//
// \param a The left-hand side scalars.
// \param b The right-hand side complex values to be scaled.
// \return The result of the scaling operation.
//
// This operation is only available for SSE2 and AVX2.
*/
BLAZE_ALWAYS_INLINE const SIMDcint16
   operator*( const SIMDint16& a, const SIMDcint16& b ) noexcept
#if BLAZE_AVX2_MODE
{
   return _mm256_mullo_epi16( (~a).value, (~b).value );
}
#elif BLAZE_SSE2_MODE
{
   return _mm_mullo_epi16( (~a).value, (~b).value );
}
#else
= delete;
#endif
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Scaling of a vector of 16-bit unsigned integral complex SIMD values.
// \ingroup simd
//
// \param a The left-hand side scalars.
// \param b The right-hand side complex values to be scaled.
// \return The result of the scaling operation.
//
// This operation is only available for SSE2 and AVX2.
*/
BLAZE_ALWAYS_INLINE const SIMDcuint16
   operator*( const SIMDuint16& a, const SIMDcuint16& b ) noexcept
#if BLAZE_AVX2_MODE
{
   return _mm256_mullo_epi16( (~a).value, (~b).value );
}
#elif BLAZE_SSE2_MODE
{
   return _mm_mullo_epi16( (~a).value, (~b).value );
}
#else
= delete;
#endif
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Multiplication of two vectors of 16-bit integral complex SIMD values.
// \ingroup simd
//
// \param a The left-hand side SIMD operand.
// \param b The right-hand side SIMD operand.
// \return The result of the multiplication.
//
// This operation is only available for SSE2 and AVX2.
*/
template< typename T >  // Type of both operands
BLAZE_ALWAYS_INLINE const T
   operator*( const SIMDci16<T>& a, const SIMDci16<T>& b ) noexcept
#if BLAZE_AVX2_MODE
{
   __m256i x, y, z;
   const __m256i neg( _mm256_set_epi16( 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1 ) );

   x = _mm256_shufflelo_epi16( (~a).value, 0xA0 );
   x = _mm256_shufflehi_epi16( x, 0xA0 );
   z = _mm256_mullo_epi16( x, (~b).value );
   x = _mm256_shufflelo_epi16( (~a).value, 0xF5 );
   x = _mm256_shufflehi_epi16( x, 0xF5 );
   y = _mm256_shufflelo_epi16( (~b).value, 0xB1 );
   y = _mm256_shufflehi_epi16( y, 0xB1 );
   y = _mm256_mullo_epi16( x, y );
   y = _mm256_mullo_epi16( y, neg );
   return _mm256_add_epi16( z, y );
}
#elif BLAZE_SSE2_MODE
{
   __m128i x, y, z;
   const __m128i neg( _mm_set_epi16( 1, -1, 1, -1, 1, -1, 1, -1 ) );

   x = _mm_shufflelo_epi16( (~a).value, 0xA0 );
   x = _mm_shufflehi_epi16( x, 0xA0 );
   z = _mm_mullo_epi16( x, (~b).value );
   x = _mm_shufflelo_epi16( (~a).value, 0xF5 );
   x = _mm_shufflehi_epi16( x, 0xF5 );
   y = _mm_shufflelo_epi16( (~b).value, 0xB1 );
   y = _mm_shufflehi_epi16( y, 0xB1 );
   y = _mm_mullo_epi16( x, y );
   y = _mm_mullo_epi16( y, neg );
   return _mm_add_epi16( z, y );
}
#else
= delete;
#endif
//*************************************************************************************************




//=================================================================================================
//
//  32-BIT INTEGRAL SIMD TYPES
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Multiplication of two vectors of 32-bit integral SIMD values of the same type.
// \ingroup simd
//
// \param a The left-hand side SIMD operand.
// \param b The right-hand side SIMD operand.
// \return The result of the multiplication.
//
// This operation is only available for SSE4, AVX2, and AVX-512.
*/
template< typename T >  // Type of both operands
BLAZE_ALWAYS_INLINE const T
   operator*( const SIMDi32<T>& a, const SIMDi32<T>& b ) noexcept
#if BLAZE_MIC_MODE
{
   return _mm512_mullo_epi32( (~a).value, (~b).value );
}
#elif BLAZE_AVX2_MODE
{
   return _mm256_mullo_epi32( (~a).value, (~b).value );
}
#elif BLAZE_SSE4_MODE
{
   return _mm_mullo_epi32( (~a).value, (~b).value );
}
#else
= delete;
#endif
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Multiplication of two vectors of 32-bit integral SIMD values of different type.
// \ingroup simd
//
// \param a The left-hand side SIMD operand.
// \param b The right-hand side SIMD operand.
// \return The result of the multiplication.
//
// This operation is only available for SSE4, AVX2, and AVX-512.
*/
template< typename T1    // Type of the left-hand side operand
        , typename T2 >  // Type of the right-hand side operand
BLAZE_ALWAYS_INLINE const SIMDuint32
   operator*( const SIMDi32<T1>& a, const SIMDi32<T2>& b ) noexcept
#if BLAZE_MIC_MODE
{
   return _mm512_mullo_epi32( (~a).value, (~b).value );
}
#elif BLAZE_AVX2_MODE
{
   return _mm256_mullo_epi32( (~a).value, (~b).value );
}
#elif BLAZE_SSE4_MODE
{
   return _mm_mullo_epi32( (~a).value, (~b).value );
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
// \return The result of the scaling operation.
//
// This operation is only available for SSE4, AVX2, and AVX-512.
*/
BLAZE_ALWAYS_INLINE const SIMDcint32
   operator*( const SIMDcint32& a, const SIMDint32& b ) noexcept
#if BLAZE_MIC_MODE
{
   return _mm512_mullo_epi32( (~a).value, (~b).value );
}
#elif BLAZE_AVX2_MODE
{
   return _mm256_mullo_epi32( (~a).value, (~b).value );
}
#elif BLAZE_SSE4_MODE
{
   return _mm_mullo_epi32( (~a).value, (~b).value );
}
#else
= delete;
#endif
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Scaling of a vector of 32-bit unsigned integral complex SIMD values.
// \ingroup simd
//
// \param a The left-hand side complex values to be scaled.
// \param b The right-hand side scalars.
// \return The result of the scaling operation.
//
// This operation is only available for SSE4, AVX2, and AVX-512.
*/
BLAZE_ALWAYS_INLINE const SIMDcuint32
   operator*( const SIMDcuint32& a, const SIMDuint32& b ) noexcept
#if BLAZE_MIC_MODE
{
   return _mm512_mullo_epi32( (~a).value, (~b).value );
}
#elif BLAZE_AVX2_MODE
{
   return _mm256_mullo_epi32( (~a).value, (~b).value );
}
#elif BLAZE_SSE4_MODE
{
   return _mm_mullo_epi32( (~a).value, (~b).value );
}
#else
= delete;
#endif
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Scaling of a vector of 32-bit signed integral complex SIMD values.
// \ingroup simd
//
// \param a The left-hand side scalars.
// \param b The right-hand side complex values to be scaled.
// \return The result of the scaling operation.
//
// This operation is only available for SSE4, AVX2, and AVX-512.
*/
template< typename T1    // Type of the left-hand side operand
        , typename T2 >  // Type of the right-hand side operand
BLAZE_ALWAYS_INLINE const SIMDcint32
   operator*( const SIMDint32& a, const SIMDcint32& b ) noexcept
#if BLAZE_MIC_MODE
{
   return _mm512_mullo_epi32( (~a).value, (~b).value );
}
#elif BLAZE_AVX2_MODE
{
   return _mm256_mullo_epi32( (~a).value, (~b).value );
}
#elif BLAZE_SSE4_MODE
{
   return _mm_mullo_epi32( (~a).value, (~b).value );
}
#else
= delete;
#endif
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Scaling of a vector of 32-bit unsigned integral complex SIMD values.
// \ingroup simd
//
// \param a The left-hand side scalars.
// \param b The right-hand side complex values to be scaled.
// \return The result of the scaling operation.
//
// This operation is only available for SSE4, AVX2, and AVX-512.
*/
template< typename T1    // Type of the left-hand side operand
        , typename T2 >  // Type of the right-hand side operand
BLAZE_ALWAYS_INLINE const SIMDcuint32
   operator*( const SIMDuint32& a, const SIMDcuint32& b ) noexcept
#if BLAZE_MIC_MODE
{
   return _mm512_mullo_epi32( (~a).value, (~b).value );
}
#elif BLAZE_AVX2_MODE
{
   return _mm256_mullo_epi32( (~a).value, (~b).value );
}
#elif BLAZE_SSE4_MODE
{
   return _mm_mullo_epi32( (~a).value, (~b).value );
}
#else
= delete;
#endif
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Multiplication of two vectors of 32-bit integral complex SIMD values.
// \ingroup simd
//
// \param a The left-hand side SIMD operand.
// \param b The right-hand side SIMD operand.
// \return The result of the multiplication.
//
// This operation is only available for SSE4, AVX2, and AVX-512.
*/
template< typename T >  // Type of both operands
BLAZE_ALWAYS_INLINE const T
   operator*( const SIMDci32<T>& a, const SIMDci32<T>& b ) noexcept
#if BLAZE_AVX2_MODE
{
   __m256i x, y, z;
   const __m256i neg( _mm256_set_epi32( 1, -1, 1, -1, 1, -1, 1, -1 ) );

   x = _mm256_shuffle_epi32( (~a).value, 0xA0 );
   z = _mm256_mullo_epi32( x, (~b).value );
   x = _mm256_shuffle_epi32( (~a).value, 0xF5 );
   y = _mm256_shuffle_epi32( (~b).value, 0xB1 );
   y = _mm256_mullo_epi32( x, y );
   y = _mm256_mullo_epi32( y, neg );
   return _mm256_add_epi32( z, y );
}
#elif BLAZE_SSE4_MODE
{
   __m128i x, y, z;
   const __m128i neg( _mm_set_epi32( 1, -1, 1, -1 ) );

   x = _mm_shuffle_epi32( (~a).value, 0xA0 );
   z = _mm_mullo_epi32( x, (~b).value );
   x = _mm_shuffle_epi32( (~a).value, 0xF5 );
   y = _mm_shuffle_epi32( (~b).value, 0xB1 );
   y = _mm_mullo_epi32( x, y );
   y = _mm_mullo_epi32( y, neg );
   return _mm_add_epi32( z, y );
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
/*!\brief Expression object for 32-bit floating point multiplication operations.
// \ingroup simd
//
// The SIMDf32MultExpr class represents the compile time expression for 32-bit floating point
// multiplication operations.
*/
template< typename T1    // Type of the left-hand side operand
        , typename T2 >  // Type of the right-hand side operand
struct SIMDf32MultExpr : public SIMDf32< SIMDf32MultExpr<T1,T2> >
{
   //**Type definitions****************************************************************************
   using This     = SIMDf32MultExpr<T1,T2>;  //!< Type of this SIMDf32MultExpr instance.
   using BaseType = SIMDf32<This>;           //!< Base type of this SIMDf32MultExpr instance.
   //**********************************************************************************************

   //**Constructor*********************************************************************************
   /*!\brief Constructor for the SIMDf32MultExpr class.
   //
   // \param a The left-hand side operand for the multiplication.
   // \param b The right-hand side operand for the multiplication.
   */
   explicit BLAZE_ALWAYS_INLINE SIMDf32MultExpr( const T1& a, const T2& b )
      : a_( a )  // The left-hand side operand for the multiplication
      , b_( b )  // The right-hand side operand for the multiplication
   {}
   //**********************************************************************************************

   //**Evaluation function*************************************************************************
   /*!\brief Evaluation of the expression object.
   //
   // \return The resulting packed 32-bit floating point value.
   */
   BLAZE_ALWAYS_INLINE const SIMDfloat eval() const noexcept
#if BLAZE_MIC_MODE
   {
      return _mm512_mul_ps( a_.eval().value, b_.eval().value );
   }
#elif BLAZE_AVX_MODE
   {
      return _mm256_mul_ps( a_.eval().value, b_.eval().value );
   }
#elif BLAZE_SSE_MODE
   {
      return _mm_mul_ps( a_.eval().value, b_.eval().value );
   }
#else
   = delete;
#endif
   //**********************************************************************************************

   //**Member variables****************************************************************************
   const T1 a_;  //!< The left-hand side operand for the multiplication.
   const T2 b_;  //!< The right-hand side operand for the multiplication.
   //**********************************************************************************************
};
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Multiplication of two vectors of single precision floating point SIMD values.
// \ingroup simd
//
// \param a The left-hand side SIMD operand.
// \param b The right-hand side SIMD operand.
// \return The result of the multiplication.
//
// This operation is only available for SSE, AVX, and AVX-512.
*/
template< typename T1    // Type of the left-hand side operand
        , typename T2 >  // Type of the right-hand side operand
BLAZE_ALWAYS_INLINE const SIMDf32MultExpr<T1,T2>
   operator*( const SIMDf32<T1>& a, const SIMDf32<T2>& b ) noexcept
{
   return SIMDf32MultExpr<T1,T2>( ~a, ~b );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Scaling of a vector of single precision complex SIMD values.
// \ingroup simd
//
// \param a The left-hand side complex values to be scaled.
// \param b The right-hand side scalars.
// \return The result of the scaling operation.
//
// This operation is only available for SSE, AVX, and AVX-512.
*/
BLAZE_ALWAYS_INLINE const SIMDcfloat
   operator*( const SIMDcfloat& a, const SIMDfloat& b ) noexcept
#if BLAZE_MIC_MODE
{
   return _mm512_mul_ps( a.value, b.value );
}
#elif BLAZE_AVX_MODE
{
   return _mm256_mul_ps( a.value, b.value );
}
#elif BLAZE_SSE_MODE
{
   return _mm_mul_ps( a.value, b.value );
}
#else
= delete;
#endif
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Scaling of a vector of single precision complex SIMD values.
// \ingroup simd
//
// \param a The left-hand side scalars.
// \param b The right-hand side complex values to be scaled.
// \return The result of the scaling operation.
//
// This operation is only available for SSE, AVX, and AVX-512.
*/
BLAZE_ALWAYS_INLINE const SIMDcfloat
   operator*( const SIMDfloat& a, const SIMDcfloat& b ) noexcept
#if BLAZE_MIC_MODE
{
   return _mm512_mul_ps( a.value, b.value );
}
#elif BLAZE_AVX_MODE
{
   return _mm256_mul_ps( a.value, b.value );
}
#elif BLAZE_SSE_MODE
{
   return _mm_mul_ps( a.value, b.value );
}
#else
= delete;
#endif
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Multiplication of two vectors of single precision complex SIMD values.
// \ingroup simd
//
// \param a The left-hand side SIMD operand.
// \param b The right-hand side SIMD operand.
// \return The result of the multiplication.
//
// This operation is only available for SSE3 and AVX.
*/
BLAZE_ALWAYS_INLINE const SIMDcfloat
   operator*( const SIMDcfloat& a, const SIMDcfloat& b ) noexcept
#if BLAZE_AVX_MODE
{
   __m256 x, y, z;

   x = _mm256_shuffle_ps( a.value, a.value, 0xA0 );
   z = _mm256_mul_ps( x, b.value );
   x = _mm256_shuffle_ps( a.value, a.value, 0xF5 );
   y = _mm256_shuffle_ps( b.value, b.value, 0xB1 );
   y = _mm256_mul_ps( x, y );
   return _mm256_addsub_ps( z, y );
}
#elif BLAZE_SSE3_MODE
{
   __m128 x, y, z;

   x = _mm_shuffle_ps( a.value, a.value, 0xA0 );
   z = _mm_mul_ps( x, b.value );
   x = _mm_shuffle_ps( a.value, a.value, 0xF5 );
   y = _mm_shuffle_ps( b.value, b.value, 0xB1 );
   y = _mm_mul_ps( x, y );
   return _mm_addsub_ps( z, y );
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
/*!\brief Expression object for 64-bit floating point multiplication operations.
// \ingroup simd
//
// The SIMDf64MultExpr class represents the compile time expression for 64-bit floating point
// multiplication operations.
*/
template< typename T1    // Type of the left-hand side operand
        , typename T2 >  // Type of the right-hand side operand
struct SIMDf64MultExpr : public SIMDf64< SIMDf64MultExpr<T1,T2> >
{
   //**Type definitions****************************************************************************
   using This     = SIMDf64MultExpr<T1,T2>;  //!< Type of this SIMDf64MultExpr instance.
   using BaseType = SIMDf64<This>;           //!< Base type of this SIMDf64MultExpr instance.
   //**********************************************************************************************

   //**Constructor*********************************************************************************
   /*!\brief Constructor for the SIMDf64MultExpr class.
   //
   // \param a The left-hand side operand for the multiplication.
   // \param b The right-hand side operand for the multiplication.
   */
   explicit BLAZE_ALWAYS_INLINE SIMDf64MultExpr( const T1& a, const T2& b )
      : a_( a )  // The left-hand side operand for the multiplication
      , b_( b )  // The right-hand side operand for the multiplication
   {}
   //**********************************************************************************************

   //**Evaluation function*************************************************************************
   /*!\brief Evaluation of the expression object.
   //
   // \return The resulting packed 64-bit floating point value.
   */
   BLAZE_ALWAYS_INLINE const SIMDdouble eval() const noexcept
#if BLAZE_MIC_MODE
   {
      return _mm512_mul_pd( a_.eval().value, b_.eval().value );
   }
#elif BLAZE_AVX_MODE
   {
      return _mm256_mul_pd( a_.eval().value, b_.eval().value );
   }
#elif BLAZE_SSE2_MODE
   {
      return _mm_mul_pd( a_.eval().value, b_.eval().value );
   }
#else
   = delete;
#endif
   //**********************************************************************************************

   //**Member variables****************************************************************************
   const T1 a_;  //!< The left-hand side operand for the multiplication.
   const T2 b_;  //!< The right-hand side operand for the multiplication.
   //**********************************************************************************************
};
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Multiplication of two vectors of double precision floating point SIMD values.
// \ingroup simd
//
// \param a The left-hand side SIMD operand.
// \param b The right-hand side SIMD operand.
// \return The result of the multiplication.
//
// This operation is only available for SSE2, AVX, and AVX-512.
*/
template< typename T1    // Type of the left-hand side operand
        , typename T2 >  // Type of the right-hand side operand
BLAZE_ALWAYS_INLINE const SIMDf64MultExpr<T1,T2>
   operator*( const SIMDf64<T1>& a, const SIMDf64<T2>& b ) noexcept
{
   return SIMDf64MultExpr<T1,T2>( ~a, ~b );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Scaling of a vector of double precision complex SIMD values.
// \ingroup simd
//
// \param a The left-hand side complex values to be scaled.
// \param b The right-hand side scalars.
// \return The result of the scaling operation.
//
// This operation is only available for SSE2, AVX, and AVX-512.
*/
BLAZE_ALWAYS_INLINE const SIMDcdouble
   operator*( const SIMDcdouble& a, const SIMDdouble& b ) noexcept
#if BLAZE_MIC_MODE
{
   return _mm512_mul_pd( a.value, b.value );
}
#elif BLAZE_AVX_MODE
{
   return _mm256_mul_pd( a.value, b.value );
}
#elif BLAZE_SSE2_MODE
{
   return _mm_mul_pd( a.value, b.value );
}
#else
= delete;
#endif
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Scaling of a vector of double precision complex SIMD values.
// \ingroup simd
//
// \param a The left-hand side scalars.
// \param b The right-hand side complex values to be scaled.
// \return The result of the scaling operation.
//
// This operation is only available for SSE2, AVX, and AVX-512.
*/
BLAZE_ALWAYS_INLINE const SIMDcdouble
   operator*( const SIMDdouble& a, const SIMDcdouble& b ) noexcept
#if BLAZE_MIC_MODE
{
   return _mm512_mul_pd( a.value, b.value );
}
#elif BLAZE_AVX_MODE
{
   return _mm256_mul_pd( a.value, b.value );
}
#elif BLAZE_SSE2_MODE
{
   return _mm_mul_pd( a.value, b.value );
}
#else
= delete;
#endif
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Multiplication of two vectors of double precision complex SIMD values.
// \ingroup simd
//
// \param a The left-hand side SIMD operand.
// \param b The right-hand side SIMD operand.
// \return The result of the multiplication.
//
// This operation is only available for SSE3 and AVX.
*/
BLAZE_ALWAYS_INLINE const SIMDcdouble
   operator*( const SIMDcdouble& a, const SIMDcdouble& b ) noexcept
#if BLAZE_AVX_MODE
{
   __m256d x, y, z;

   x = _mm256_shuffle_pd( a.value, a.value, 0 );
   z = _mm256_mul_pd( x, b.value );
   x = _mm256_shuffle_pd( a.value, a.value, 15 );
   y = _mm256_shuffle_pd( b.value, b.value, 5 );
   y = _mm256_mul_pd( x, y );
   return _mm256_addsub_pd( z, y );
}
#elif BLAZE_SSE3_MODE
{
   __m128d x, y, z;

   x = _mm_shuffle_pd( a.value, a.value, 0 );
   z = _mm_mul_pd( x, b.value );
   x = _mm_shuffle_pd( a.value, a.value, 3 );
   y = _mm_shuffle_pd( b.value, b.value, 1 );
   y = _mm_mul_pd( x, y );
   return _mm_addsub_pd( z, y );
}
#else
= delete;
#endif
//*************************************************************************************************

} // namespace blaze

#endif
