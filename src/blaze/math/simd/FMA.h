//=================================================================================================
/*!
//  \file blaze/math/simd/FMA.h
//  \brief Header file for the SIMD fused multiply-add (FMA) functionality
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

#ifndef _BLAZE_MATH_SIMD_FMA_H_
#define _BLAZE_MATH_SIMD_FMA_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/simd/Addition.h>
#include <blaze/math/simd/BasicTypes.h>
#include <blaze/math/simd/Multiplication.h>
#include <blaze/math/simd/Subtraction.h>
#include <blaze/system/Inline.h>
#include <blaze/system/Vectorization.h>


namespace blaze {

//=================================================================================================
//
//  32-BIT FLOATING POINT SIMD TYPES
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Expression object for 32-bit floating point fused multiply-add operations.
// \ingroup simd
//
// The SIMDf32FmaddExpr class represents the compile time expression for 32-bit floating point
// fused multiply-add operations.
*/
template< typename T1    // Type of the left-hand side multiplication operand
        , typename T2    // Type of the right-hand side multiplication operand
        , typename T3 >  // Type of the right-hand side addition operand
struct SIMDf32FmaddExpr : public SIMDf32< SIMDf32FmaddExpr<T1,T2,T3> >
{
   //**Type definitions****************************************************************************
   using This     = SIMDf32FmaddExpr<T1,T2,T3>;  //!< Type of this SIMDf32FMaddExpr instance.
   using BaseType = SIMDf32<This>;               //!< Base type of this SIMDf32FMaddExpr instance.
   //**********************************************************************************************

   //**Constructor*********************************************************************************
   /*!\brief Constructor for the SIMDf32FmaddExpr class.
   //
   // \param a The left-hand side operand for the multiplication.
   // \param b The right-hand side operand for the multiplication.
   // \param c The right-hand side operand for the addition.
   */
   explicit BLAZE_ALWAYS_INLINE SIMDf32FmaddExpr( const T1& a, const T2& b, const T3& c )
      : a_( a )  // The left-hand side operand for the multiplication
      , b_( b )  // The right-hand side operand for the multiplication
      , c_( c )  // The right-hand side operand for the addition
   {}
   //**********************************************************************************************

   //**Evaluation function*************************************************************************
   /*!\brief Evaluation of the expression object.
   //
   // \return The resulting packed 32-bit floating point value.
   */
   BLAZE_ALWAYS_INLINE const SIMDfloat eval() const noexcept
#if BLAZE_FMA_MODE && BLAZE_MIC_MODE
   {
      return _mm512_fmadd_ps( a_.eval().value, b_.eval().value, c_.eval().value );
   }
#elif BLAZE_FMA_MODE && BLAZE_AVX_MODE
   {
      return _mm256_fmadd_ps( a_.eval().value, b_.eval().value, c_.eval().value );
   }
#elif BLAZE_FMA_MODE && BLAZE_SSE2_MODE
   {
      return _mm_fmadd_ps( a_.eval().value, b_.eval().value, c_.eval().value );
   }
#else
   = delete;
#endif
   //**********************************************************************************************

   //**Member variables****************************************************************************
   const T1 a_;  //!< The left-hand side operand for the multiplication.
   const T2 b_;  //!< The right-hand side operand for the multiplication.
   const T3 c_;  //!< The right-hand side operand for the addition.
   //**********************************************************************************************
};
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Expression object for 32-bit floating point fused multiply-subtract operations.
// \ingroup simd
//
// The SIMDf32FmsubExpr class represents the compile time expression for 32-bit floating point
// fused multiply-subtract operations.
*/
template< typename T1    // Type of the left-hand side multiplication operand
        , typename T2    // Type of the right-hand side multiplication operand
        , typename T3 >  // Type of the right-hand side subtraction operand
struct SIMDf32FmsubExpr : public SIMDf32< SIMDf32FmsubExpr<T1,T2,T3> >
{
   //**Type definitions****************************************************************************
   using This     = SIMDf32MultExpr<T1,T2>;  //!< Type of this SIMDf32FMsubExpr instance.
   using BaseType = SIMDf32<This>;           //!< Base type of this SIMDf32FMsubExpr instance.
   //**********************************************************************************************

   //**Constructor*********************************************************************************
   /*!\brief Constructor for the SIMDf32FmsubExpr class.
   //
   // \param a The left-hand side operand for the multiplication.
   // \param b The right-hand side operand for the multiplication.
   // \param c The right-hand side operand for the subtraction.
   */
   explicit BLAZE_ALWAYS_INLINE SIMDf32FmsubExpr( const T1& a, const T2& b, const T3& c )
      : a_( a )  // The left-hand side operand for the multiplication
      , b_( b )  // The right-hand side operand for the multiplication
      , c_( c )  // The right-hand side operand for the subtraction
   {}
   //**********************************************************************************************

   //**Evaluation function*************************************************************************
   /*!\brief Evaluation of the expression object.
   //
   // \return The resulting packed 32-bit floating point value.
   */
   BLAZE_ALWAYS_INLINE const SIMDfloat eval() const noexcept
#if BLAZE_FMA_MODE && BLAZE_MIC_MODE
   {
      return _mm512_fmsub_ps( a_.eval().value, b_.eval().value, c_.eval().value );
   }
#elif BLAZE_FMA_MODE && BLAZE_AVX_MODE
   {
      return _mm256_fmsub_ps( a_.eval().value, b_.eval().value, c_.eval().value );
   }
#elif BLAZE_FMA_MODE && BLAZE_SSE2_MODE
   {
      return _mm_fmsub_ps( a_.eval().value, b_.eval().value, c_.eval().value );
   }
#else
   = delete;
#endif
   //**********************************************************************************************

   //**Member variables****************************************************************************
   const T1 a_;  //!< The left-hand side operand for the multiplication.
   const T2 b_;  //!< The right-hand side operand for the multiplication.
   const T3 c_;  //!< The right-hand side operand for the subtraction.
   //**********************************************************************************************
};
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Addition operator for fusing a 32-bit floating point multiplication and addition.
// \ingroup simd
//
// \param a The left-hand side SIMD multiplication expression.
// \param b The right-hand side SIMD addition operand.
// \return The result of the FMA operation.
//
// This operator fuses a 32-bit floating point multiplication with the addition of a 32-bit
// floating point operand. It returns an expression representing the fused multiply-add (FMA)
// operation.
*/
#if BLAZE_FMA_MODE
template< typename T1    // Type of the first multiplication operand
        , typename T2    // Type of the second multiplication operand
        , typename T3 >  // Type of the second addition operand
BLAZE_ALWAYS_INLINE const SIMDf32FmaddExpr<T1,T2,T3>
   operator+( const SIMDf32MultExpr<T1,T2>& a, const SIMDf32<T3>& b )
{
   return SIMDf32FmaddExpr<T1,T2,T3>( a.a_, a.b_, ~b );
}
#endif
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Addition operator for fusing a 32-bit floating point multiplication and addition.
// \ingroup simd
//
// \param a The left-hand side SIMD addition operand.
// \param b The right-hand side SIMD multiplication expression.
// \return The result of the FMA operation.
//
// This operator fuses a 32-bit floating point multiplication with the addition of a 32-bit
// floating point operand. It returns an expression representing the fused multiply-add (FMA)
// operation.
*/
#if BLAZE_FMA_MODE
template< typename T1    // Type of the first addition operand
        , typename T2    // Type of the first multiplication operand
        , typename T3 >  // Type of the second multiplication operand
BLAZE_ALWAYS_INLINE const SIMDf32FmaddExpr<T2,T3,T1>
   operator+( const SIMDf32<T1>& a, const SIMDf32MultExpr<T2,T3>& b )
{
   return SIMDf32FmaddExpr<T2,T3,T1>( b.a_, b.b_, ~a );
}
#endif
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Addition operator for fusing a 32-bit floating point multiplication and addition.
// \ingroup simd
//
// \param a The left-hand side SIMD multiplication expression.
// \param b The right-hand side SIMD multiplication expression.
// \return The result of the FMA operation.
//
// This operator fuses a 32-bit floating point multiplication with the addition of a 32-bit
// floating point operand. It returns an expression representing the fused multiply-add (FMA)
// operation.
*/
#if BLAZE_FMA_MODE
template< typename T1    // Type of the first operand of the left-hand side multiplication
        , typename T2    // Type of the second operand of the left-hand side multiplication
        , typename T3    // Type of the first operand of the right-hand side multiplication
        , typename T4 >  // Type of the second operand of the right-hand side multiplication
BLAZE_ALWAYS_INLINE const SIMDf32FmaddExpr< T1, T2, SIMDf32MultExpr<T3,T4> >
   operator+( const SIMDf32MultExpr<T1,T2>& a, const SIMDf32MultExpr<T3,T4>& b )
{
   return SIMDf32FmaddExpr< T1, T2, SIMDf32MultExpr<T3,T4> >( a.a_, a.b_, b );
}
#endif
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Restructuring operator for the addition of a 32-bit floating point FMA expression with
//        a 32-bit floating point operand.
// \ingroup simd
//
// \param a The left-hand side SIMD FMA expression.
// \param b The right-hand side SIMD addition operand.
// \return The restructured expression.
//
// This operator implements a performance optimized treatment of the addition of a 32-bit floating
// point FMA expression and a 32-bit floating point operand. It restructures the expression
// \f$ (a*b+c) + d \f$ to the expression \f$ (a*b) + (c+d) \f$.
*/
#if BLAZE_FMA_MODE
template< typename T1    // Type of the first FMA multiplication operand
        , typename T2    // Type of the second FMA multiplication operand
        , typename T3    // Type of the FMA addition operand
        , typename T4 >  // Type of the second addition operand
BLAZE_ALWAYS_INLINE const auto
   operator+( const SIMDf32FmaddExpr<T1,T2,T3>& a, const SIMDf32<T4>& b )
{
   return ( a.a_ * a.b_ ) + ( a.c_ + (~b) );
}
#endif
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Restructuring operator for the addition of a 32-bit floating point operand with a
//        32-bit floating point FMA expression.
// \ingroup simd
//
// \param a The left-hand side SIMD addition operand.
// \param b The right-hand side SIMD FMA expression.
// \return The restructured expression.
//
// This operator implements a performance optimized treatment of the addition of a 32-bit floating
// point operand and a 32-bit floating point FMA expression. It restructures the expression
// \f$ a + (b*c+d) \f$ to the expression \f$ (b*c) + (d+a) \f$.
*/
#if BLAZE_FMA_MODE
template< typename T1    // Type of the first addition operand
        , typename T2    // Type of the first FMA multiplication operand
        , typename T3    // Type of the second FMA multiplication operand
        , typename T4 >  // Type of the FMA addition operand
BLAZE_ALWAYS_INLINE const auto
   operator+( const SIMDf32<T1>& a, const SIMDf32FmaddExpr<T2,T3,T4>& b )
{
   return ( b.a_ * b.b_ ) + ( b.c_ + (~a) );
}
#endif
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Restructuring operator for the addition of a 32-bit floating point FMA expression with
//        a 32-bit floating point multiplication expression.
// \ingroup simd
//
// \param a The left-hand side SIMD FMA expression.
// \param b The right-hand side SIMD multiplication expression.
// \return The restructured expression.
//
// This operator implements a performance optimized treatment of the addition of a 32-bit floating
// point FMA expression and a 32-bit floating point multiplication expression. It restructures the
// expression \f$ (a*b+c) + (d*e) \f$ to the expression \f$ (a*b) + (d*e+c)\f$.
*/
#if BLAZE_FMA_MODE
template< typename T1    // Type of the first FMA multiplication operand
        , typename T2    // Type of the second FMA multiplication operand
        , typename T3    // Type of the FMA addition operand
        , typename T4    // Type of the first multiplication operand
        , typename T5 >  // Type of the second multiplication operand
BLAZE_ALWAYS_INLINE const SIMDf32FmaddExpr< T4, T5, SIMDf32FmaddExpr<T1,T2,T3> >
   operator+( const SIMDf32FmaddExpr<T1,T2,T3>& a, const SIMDf32MultExpr<T4,T5>& b )
{
   return SIMDf32FmaddExpr< T4, T5, SIMDf32FmaddExpr<T1,T2,T3> >( b.a_, b.b_, a );
}
#endif
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Restructuring operator for the addition of a 32-bit floating point multiplication
//        expression with a 32-bit floating point FMA expression.
// \ingroup simd
//
// \param a The left-hand side SIMD multiplication expression.
// \param b The right-hand side SIMD FMA expression.
// \return The restructured expression.
//
// This operator implements a performance optimized treatment of the addition of a 32-bit floating
// point multiplication expression and a 32-bit floating point FMA expression. It restructures the
// expression \f$ (a*b) + (c*d+e) \f$ to the expression \f$ (a*b) + (c*d+e)\f$.
*/
#if BLAZE_FMA_MODE
template< typename T1    // Type of the first multiplication operand
        , typename T2    // Type of the second multiplication operand
        , typename T3    // Type of the first FMA multiplication operand
        , typename T4    // Type of the second FMA multiplication operand
        , typename T5 >  // Type of the FMA addition operand
BLAZE_ALWAYS_INLINE const SIMDf32FmaddExpr< T1, T2, SIMDf32FmaddExpr<T3,T4,T5> >
   operator+( const SIMDf32MultExpr<T1,T2>& a, const SIMDf32FmaddExpr<T3,T4,T5>& b )
{
   return SIMDf32FmaddExpr< T1, T2, SIMDf32FmaddExpr<T3,T4,T5> >( a.a_, a.b_, b );
}
#endif
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Restructuring operator for the addition of two 32-bit floating point FMA expressions.
// \ingroup simd
//
// \param a The left-hand side SIMD FMA expression.
// \param b The right-hand side SIMD FMA expression.
// \return The restructured expression.
//
// This operator implements a performance optimized treatment of the addition of two 32-bit floating
// point FMA expressions. It restructures the expression \f$ (a*b+c) + (d*e+f) \f$ to the expression
// \f$ (a*b) + (d*e+c+f)\f$.
*/
#if BLAZE_FMA_MODE
template< typename T1    // Type of the first multiplication operand of the left-hand side FMA
        , typename T2    // Type of the second multiplication operand of the left-hand side FMA
        , typename T3    // Type of the addition operand of the left-hand side FMA
        , typename T4    // Type of the first multiplication operand of the right-hand side FMA
        , typename T5    // Type of the second multiplication operand of the right-hand side FMA
        , typename T6 >  // Type of the addition operand of the right-hand side FMA
BLAZE_ALWAYS_INLINE const auto
   operator+( const SIMDf32FmaddExpr<T1,T2,T3>& a, const SIMDf32FmaddExpr<T4,T5,T6>& b )
{
   return ( a.a_ * a.b_ ) + ( ( b.a_ * b.b_ ) + ( a.c_ + b.c_ ) );
}
#endif
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Restructuring operator for the addition of two 32-bit floating point FMA expressions.
// \ingroup simd
//
// \param a The left-hand side SIMD FMA expression.
// \param b The right-hand side SIMD FMA expression.
// \return The restructured expression.
//
// This operator implements a performance optimized treatment of the addition of two 32-bit floating
// point FMA expressions. It restructures the expression \f$ (a*b+c) + (d*e-f) \f$ to the expression
// \f$ (a*b) + (d*e+c-f)\f$.
*/
#if BLAZE_FMA_MODE
template< typename T1    // Type of the first multiplication operand of the left-hand side FMA
        , typename T2    // Type of the second multiplication operand of the left-hand side FMA
        , typename T3    // Type of the addition operand of the left-hand side FMA
        , typename T4    // Type of the first multiplication operand of the right-hand side FMA
        , typename T5    // Type of the second multiplication operand of the right-hand side FMA
        , typename T6 >  // Type of the subtraction operand of the right-hand side FMA
BLAZE_ALWAYS_INLINE const auto
   operator+( const SIMDf32FmaddExpr<T1,T2,T3>& a, const SIMDf32FmsubExpr<T4,T5,T6>& b )
{
   return ( a.a_ * a.b_ ) + ( ( b.a_ * b.b_ ) + ( a.c_ - b.c_ ) );
}
#endif
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Restructuring operator for the addition of two 32-bit floating point FMA expressions.
// \ingroup simd
//
// \param a The left-hand side SIMD FMA expression.
// \param b The right-hand side SIMD FMA expression.
// \return The restructured expression.
//
// This operator implements a performance optimized treatment of the addition of two 32-bit floating
// point FMA expressions. It restructures the expression \f$ (a*b-c) + (d*e+f) \f$ to the expression
// \f$ (a*b) + (d*e+f-c)\f$.
*/
#if BLAZE_FMA_MODE
template< typename T1    // Type of the first multiplication operand of the left-hand side FMA
        , typename T2    // Type of the second multiplication operand of the left-hand side FMA
        , typename T3    // Type of the subtraction operand of the left-hand side FMA
        , typename T4    // Type of the first multiplication operand of the right-hand side FMA
        , typename T5    // Type of the second multiplication operand of the right-hand side FMA
        , typename T6 >  // Type of the addition operand of the right-hand side FMA
BLAZE_ALWAYS_INLINE const auto
   operator+( const SIMDf32FmsubExpr<T1,T2,T3>& a, const SIMDf32FmaddExpr<T4,T5,T6>& b )
{
   return ( a.a_ * a.b_ ) + ( ( b.a_ * b.b_ ) + ( b.c_ - a.c_ ) );
}
#endif
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Restructuring operator for the addition of two 32-bit floating point FMA expressions.
// \ingroup simd
//
// \param a The left-hand side SIMD FMA expression.
// \param b The right-hand side SIMD FMA expression.
// \return The restructured expression.
//
// This operator implements a performance optimized treatment of the addition of two 32-bit floating
// point FMA expressions. It restructures the expression \f$ (a*b-c) + (d*e-f) \f$ to the expression
// \f$ (a*b) + (d*e-f-c)\f$.
*/
#if BLAZE_FMA_MODE
template< typename T1    // Type of the first multiplication operand of the left-hand side FMA
        , typename T2    // Type of the second multiplication operand of the left-hand side FMA
        , typename T3    // Type of the subtraction operand of the left-hand side FMA
        , typename T4    // Type of the first multiplication operand of the right-hand side FMA
        , typename T5    // Type of the second multiplication operand of the right-hand side FMA
        , typename T6 >  // Type of the subtraction operand of the right-hand side FMA
BLAZE_ALWAYS_INLINE const auto
   operator+( const SIMDf32FmsubExpr<T1,T2,T3>& a, const SIMDf32FmsubExpr<T4,T5,T6>& b )
{
   return ( a.a_ * a.b_ ) + ( ( b.a_ * b.b_ ) - ( b.c_ + a.c_ ) );
}
#endif
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Subtraction operator for fusing a 32-bit floating point multiplication and subtraction.
// \ingroup simd
//
// \param a The left-hand side SIMD multiplication expression.
// \param b The right-hand side SIMD subtraction operand.
// \return The result of the FMA operation.
//
// This operator fuses a 32-bit floating point multiplication with the subtraction of a 32-bit
// floating point operand. It returns an expression representing the fused multiply-subtract
// operation.
*/
#if BLAZE_FMA_MODE
template< typename T1    // Type of the first multiplication operand
        , typename T2    // Type of the second multiplication operand
        , typename T3 >  // Type of the second subtraction operand
BLAZE_ALWAYS_INLINE const SIMDf32FmsubExpr<T1,T2,T3>
   operator-( const SIMDf32MultExpr<T1,T2>& a, const SIMDf32<T3>& b )
{
   return SIMDf32FmsubExpr<T1,T2,T3>( a.a_, a.b_, ~b );
}
#endif
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Subtraction operator for fusing a 32-bit floating point multiplication and subtraction.
// \ingroup simd
//
// \param a The left-hand side SIMD multiplication expression.
// \param b The right-hand side SIMD multiplication expression.
// \return The result of the FMA operation.
//
// This operator fuses a 32-bit floating point multiplication with the subtraction of a 32-bit
// floating point operand. It returns an expression representing the fused multiply-subtract
// (FMA) operation.
*/
#if BLAZE_FMA_MODE
template< typename T1    // Type of the first operand of the left-hand side multiplication
        , typename T2    // Type of the second operand of the left-hand side multiplication
        , typename T3    // Type of the first operand of the right-hand side multiplication
        , typename T4 >  // Type of the second operand of the right-hand side multiplication
BLAZE_ALWAYS_INLINE const SIMDf32FmsubExpr< T1, T2, SIMDf32MultExpr<T3,T4> >
   operator-( const SIMDf32MultExpr<T1,T2>& a, const SIMDf32MultExpr<T3,T4>& b )
{
   return SIMDf32FmsubExpr< T1, T2, SIMDf32MultExpr<T3,T4> >( a.a_, a.b_, b );
}
#endif
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Restructuring operator for the subtraction of a 32-bit floating point FMA expression
//        with a 32-bit floating point operand.
// \ingroup simd
//
// \param a The left-hand side SIMD FMA expression.
// \param b The right-hand side SIMD subtraction operand.
// \return The restructured expression.
//
// This operator implements a performance optimized treatment of the subtraction of a 32-bit
// floating point FMA expression and a 32-bit floating point operand. It restructures the
// expression \f$ (a*b+c) + d \f$ to the expression \f$ (a*b) + (c+d) \f$.
*/
#if BLAZE_FMA_MODE
template< typename T1    // Type of the first FMA multiplication operand
        , typename T2    // Type of the second FMA multiplication operand
        , typename T3    // Type of the FMA subtraction operand
        , typename T4 >  // Type of the second subtraction operand
BLAZE_ALWAYS_INLINE const auto
   operator-( const SIMDf32FmsubExpr<T1,T2,T3>& a, const SIMDf32<T4>& b )
{
   return ( a.a_ * a.b_ ) - ( a.c_ + (~b) );
}
#endif
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Restructuring operator for the subtraction of a 32-bit floating point FMA expression
//        with a 32-bit floating point multiplication expression.
// \ingroup simd
//
// \param a The left-hand side SIMD FMA expression.
// \param b The right-hand side SIMD multiplication expression.
// \return The restructured expression.
//
// This operator implements a performance optimized treatment of the subtraction of a 32-bit
// floating point FMA expression and a 32-bit floating point multiplication expression. It
// restructures the expression \f$ (a*b-c) - (d*e) \f$ to the expression \f$ (a*b) - (d*e+c)\f$.
*/
#if BLAZE_FMA_MODE
template< typename T1    // Type of the first FMA multiplication operand
        , typename T2    // Type of the second FMA multiplication operand
        , typename T3    // Type of the FMA subtraction operand
        , typename T4    // Type of the first multiplication operand
        , typename T5 >  // Type of the second multiplication operand
BLAZE_ALWAYS_INLINE const auto
   operator-( const SIMDf32FmsubExpr<T1,T2,T3>& a, const SIMDf32MultExpr<T4,T5>& b )
{
   return ( a.a_ * a.b_ ) - ( b.a_ * b.b_ + a.c_ );
}
#endif
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Restructuring operator for the subtraction of a 32-bit floating point multiplication
//        expression with a 32-bit floating point FMA expression.
// \ingroup simd
//
// \param a The left-hand side SIMD multiplication expression.
// \param b The right-hand side SIMD FMA expression.
// \return The restructured expression.
//
// This operator implements a performance optimized treatment of the subtraction of a 32-bit
// floating point multiplication expression and a 32-bit floating point FMA expression. It
// restructures the expression \f$ (a*b) - (c*d+e) \f$ to the expression \f$ (a*b) - (c*d+e)\f$.
*/
#if BLAZE_FMA_MODE
template< typename T1    // Type of the first multiplication operand
        , typename T2    // Type of the second multiplication operand
        , typename T3    // Type of the first FMA multiplication operand
        , typename T4    // Type of the second FMA multiplication operand
        , typename T5 >  // Type of the FMA subtraction operand
BLAZE_ALWAYS_INLINE const SIMDf32FmsubExpr< T1, T2, SIMDf32FmsubExpr<T3,T4,T5> >
   operator-( const SIMDf32MultExpr<T1,T2>& a, const SIMDf32FmsubExpr<T3,T4,T5>& b )
{
   return SIMDf32FmsubExpr< T1, T2, SIMDf32FmsubExpr<T3,T4,T5> >( a.a_, a.b_, b );
}
#endif
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Restructuring operator for the subtraction of two 32-bit floating point FMA expressions.
// \ingroup simd
//
// \param a The left-hand side SIMD FMA expression.
// \param b The right-hand side SIMD FMA expression.
// \return The restructured expression.
//
// This operator implements a performance optimized treatment of the subtraction of two 32-bit
// floating point FMA expressions. It restructures the expression \f$ (a*b+c) - (d*e+f) \f$ to
// the expression \f$ (a*b) - (d*e+f-c)\f$.
*/
#if BLAZE_FMA_MODE
template< typename T1    // Type of the first multiplication operand of the left-hand side FMA
        , typename T2    // Type of the second multiplication operand of the left-hand side FMA
        , typename T3    // Type of the addition operand of the left-hand side FMA
        , typename T4    // Type of the first multiplication operand of the right-hand side FMA
        , typename T5    // Type of the second multiplication operand of the right-hand side FMA
        , typename T6 >  // Type of the addition operand of the right-hand side FMA
BLAZE_ALWAYS_INLINE const auto
   operator-( const SIMDf32FmaddExpr<T1,T2,T3>& a, const SIMDf32FmaddExpr<T4,T5,T6>& b )
{
   return ( a.a_ * a.b_ ) - ( ( b.a_ * b.b_ ) + ( b.c_ - a.c_ ) );
}
#endif
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Restructuring operator for the subtraction of two 32-bit floating point FMA expressions.
// \ingroup simd
//
// \param a The left-hand side SIMD FMA expression.
// \param b The right-hand side SIMD FMA expression.
// \return The restructured expression.
//
// This operator implements a performance optimized treatment of the subtraction of two 32-bit
// floating point FMA expressions. It restructures the expression \f$ (a*b+c) - (d*e-f) \f$ to
// the expression \f$ (a*b) - (d*e+f-c)\f$.
*/
#if BLAZE_FMA_MODE
template< typename T1    // Type of the first multiplication operand of the left-hand side FMA
        , typename T2    // Type of the second multiplication operand of the left-hand side FMA
        , typename T3    // Type of the addition operand of the left-hand side FMA
        , typename T4    // Type of the first multiplication operand of the right-hand side FMA
        , typename T5    // Type of the second multiplication operand of the right-hand side FMA
        , typename T6 >  // Type of the subtraction operand of the right-hand side FMA
BLAZE_ALWAYS_INLINE const auto
   operator-( const SIMDf32FmaddExpr<T1,T2,T3>& a, const SIMDf32FmsubExpr<T4,T5,T6>& b )
{
   return ( a.a_ * a.b_ ) - ( ( b.a_ * b.b_ ) - ( a.c_ + b.c_ ) );
}
#endif
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Restructuring operator for the subtraction of two 32-bit floating point FMA expressions.
// \ingroup simd
//
// \param a The left-hand side SIMD FMA expression.
// \param b The right-hand side SIMD FMA expression.
// \return The restructured expression.
//
// This operator implements a performance optimized treatment of the subtraction of two 32-bit
// floating point FMA expressions. It restructures the expression \f$ (a*b-c) - (d*e+f) \f$ to
// the expression \f$ (a*b) - (d*e+c-f)\f$.
*/
#if BLAZE_FMA_MODE
template< typename T1    // Type of the first multiplication operand of the left-hand side FMA
        , typename T2    // Type of the second multiplication operand of the left-hand side FMA
        , typename T3    // Type of the subtraction operand of the left-hand side FMA
        , typename T4    // Type of the first multiplication operand of the right-hand side FMA
        , typename T5    // Type of the second multiplication operand of the right-hand side FMA
        , typename T6 >  // Type of the addition operand of the right-hand side FMA
BLAZE_ALWAYS_INLINE const auto
   operator-( const SIMDf32FmsubExpr<T1,T2,T3>& a, const SIMDf32FmaddExpr<T4,T5,T6>& b )
{
   return ( a.a_ * a.b_ ) - ( ( b.a_ * b.b_ ) + ( a.c_ + b.c_ ) );
}
#endif
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Restructuring operator for the subtraction of two 32-bit floating point FMA expressions.
// \ingroup simd
//
// \param a The left-hand side SIMD FMA expression.
// \param b The right-hand side SIMD FMA expression.
// \return The restructured expression.
//
// This operator implements a performance optimized treatment of the subtraction of two 32-bit
// floating point FMA expressions. It restructures the expression \f$ (a*b-c) - (d*e-f) \f$ to
// the expression \f$ (a*b) - (d*e+c-f)\f$.
*/
#if BLAZE_FMA_MODE
template< typename T1    // Type of the first multiplication operand of the left-hand side FMA
        , typename T2    // Type of the second multiplication operand of the left-hand side FMA
        , typename T3    // Type of the subtraction operand of the left-hand side FMA
        , typename T4    // Type of the first multiplication operand of the right-hand side FMA
        , typename T5    // Type of the second multiplication operand of the right-hand side FMA
        , typename T6 >  // Type of the subtraction operand of the right-hand side FMA
BLAZE_ALWAYS_INLINE const auto
   operator-( const SIMDf32FmsubExpr<T1,T2,T3>& a, const SIMDf32FmsubExpr<T4,T5,T6>& b )
{
   return ( a.a_ * a.b_ ) - ( ( b.a_ * b.b_ ) + ( a.c_ - b.c_ ) );
}
#endif
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  64-BIT FLOATING POINT SIMD TYPES
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Expression object for 64-bit floating point fused multiply-add operations.
// \ingroup simd
//
// The SIMDf64FmaddExpr class represents the compile time expression for 64-bit floating point
// fused multiply-add operations.
*/
template< typename T1    // Type of the left-hand side multiplication operand
        , typename T2    // Type of the right-hand side multiplication operand
        , typename T3 >  // Type of the right-hand side addition operand
struct SIMDf64FmaddExpr : public SIMDf64< SIMDf64FmaddExpr<T1,T2,T3> >
{
   //**Type definitions****************************************************************************
   using This     = SIMDf64FmaddExpr<T1,T2,T3>;  //!< Type of this SIMDf64FMaddExpr instance.
   using BaseType = SIMDf64<This>;               //!< Base type of this SIMDf64FMaddExpr instance.
   //**********************************************************************************************

   //**Constructor*********************************************************************************
   /*!\brief Constructor for the SIMDf64FmaddExpr class.
   //
   // \param a The left-hand side operand for the multiplication.
   // \param b The right-hand side operand for the multiplication.
   // \param c The right-hand side operand for the addition.
   */
   explicit BLAZE_ALWAYS_INLINE SIMDf64FmaddExpr( const T1& a, const T2& b, const T3& c )
      : a_( a )  // The left-hand side operand for the multiplication
      , b_( b )  // The right-hand side operand for the multiplication
      , c_( c )  // The right-hand side operand for the addition
   {}
   //**********************************************************************************************

   //**Evaluation function*************************************************************************
   /*!\brief Evaluation of the expression object.
   //
   // \return The resulting packed 64-bit floating point value.
   */
   BLAZE_ALWAYS_INLINE const SIMDdouble eval() const noexcept
#if BLAZE_FMA_MODE && BLAZE_MIC_MODE
   {
      return _mm512_fmadd_pd( a_.eval().value, b_.eval().value, c_.eval().value );
   }
#elif BLAZE_FMA_MODE && BLAZE_AVX_MODE
   {
      return _mm256_fmadd_pd( a_.eval().value, b_.eval().value, c_.eval().value );
   }
#elif BLAZE_FMA_MODE && BLAZE_SSE2_MODE
   {
      return _mm_fmadd_pd( a_.eval().value, b_.eval().value, c_.eval().value );
   }
#else
   = delete;
#endif
   //**********************************************************************************************

   //**Member variables****************************************************************************
   const T1 a_;  //!< The left-hand side operand for the multiplication.
   const T2 b_;  //!< The right-hand side operand for the multiplication.
   const T3 c_;  //!< The right-hand side operand for the addition.
   //**********************************************************************************************
};
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Expression object for 64-bit floating point fused multiply-subtract operations.
// \ingroup simd
//
// The SIMDf64FmsubExpr class represents the compile time expression for 64-bit floating point
// fused multiply-subtract operations.
*/
template< typename T1    // Type of the left-hand side multiplication operand
        , typename T2    // Type of the right-hand side multiplication operand
        , typename T3 >  // Type of the right-hand side subtraction operand
struct SIMDf64FmsubExpr : public SIMDf64< SIMDf64FmsubExpr<T1,T2,T3> >
{
   //**Type definitions****************************************************************************
   using This     = SIMDf64MultExpr<T1,T2>;  //!< Type of this SIMDf64FMsubExpr instance.
   using BaseType = SIMDf64<This>;           //!< Base type of this SIMDf64FMsubExpr instance.
   //**********************************************************************************************

   //**Constructor*********************************************************************************
   /*!\brief Constructor for the SIMDf64FmsubExpr class.
   //
   // \param a The left-hand side operand for the multiplication.
   // \param b The right-hand side operand for the multiplication.
   // \param c The right-hand side operand for the subtraction.
   */
   explicit BLAZE_ALWAYS_INLINE SIMDf64FmsubExpr( const T1& a, const T2& b, const T3& c )
      : a_( a )  // The left-hand side operand for the multiplication
      , b_( b )  // The right-hand side operand for the multiplication
      , c_( c )  // The right-hand side operand for the subtraction
   {}
   //**********************************************************************************************

   //**Evaluation function*************************************************************************
   /*!\brief Evaluation of the expression object.
   //
   // \return The resulting packed 64-bit floating point value.
   */
   BLAZE_ALWAYS_INLINE const SIMDdouble eval() const noexcept
#if BLAZE_FMA_MODE && BLAZE_MIC_MODE
   {
      return _mm512_fmsub_pd( a_.eval().value, b_.eval().value, c_.eval().value );
   }
#elif BLAZE_FMA_MODE && BLAZE_AVX_MODE
   {
      return _mm256_fmsub_pd( a_.eval().value, b_.eval().value, c_.eval().value );
   }
#elif BLAZE_FMA_MODE && BLAZE_SSE2_MODE
   {
      return _mm_fmsub_pd( a_.eval().value, b_.eval().value, c_.eval().value );
   }
#else
   = delete;
#endif
   //**********************************************************************************************

   //**Member variables****************************************************************************
   const T1 a_;  //!< The left-hand side operand for the multiplication.
   const T2 b_;  //!< The right-hand side operand for the multiplication.
   const T3 c_;  //!< The right-hand side operand for the subtraction.
   //**********************************************************************************************
};
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Addition operator for fusing a 64-bit floating point multiplication and addition.
// \ingroup simd
//
// \param a The left-hand side SIMD multiplication expression.
// \param b The right-hand side SIMD addition operand.
// \return The result of the FMA operation.
//
// This operator fuses a 64-bit floating point multiplication with the addition of a 64-bit
// floating point operand. It returns an expression representing the fused multiply-add (FMA)
// operation.
*/
#if BLAZE_FMA_MODE
template< typename T1    // Type of the first multiplication operand
        , typename T2    // Type of the second multiplication operand
        , typename T3 >  // Type of the second addition operand
BLAZE_ALWAYS_INLINE const SIMDf64FmaddExpr<T1,T2,T3>
   operator+( const SIMDf64MultExpr<T1,T2>& a, const SIMDf64<T3>& b )
{
   return SIMDf64FmaddExpr<T1,T2,T3>( a.a_, a.b_, ~b );
}
#endif
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Addition operator for fusing a 64-bit floating point multiplication and addition.
// \ingroup simd
//
// \param a The left-hand side SIMD addition operand.
// \param b The right-hand side SIMD multiplication expression.
// \return The result of the FMA operation.
//
// This operator fuses a 64-bit floating point multiplication with the addition of a 64-bit
// floating point operand. It returns an expression representing the fused multiply-add (FMA)
// operation.
*/
#if BLAZE_FMA_MODE
template< typename T1    // Type of the first addition operand
        , typename T2    // Type of the first multiplication operand
        , typename T3 >  // Type of the second multiplication operand
BLAZE_ALWAYS_INLINE const SIMDf64FmaddExpr<T2,T3,T1>
   operator+( const SIMDf64<T1>& a, const SIMDf64MultExpr<T2,T3>& b )
{
   return SIMDf64FmaddExpr<T2,T3,T1>( b.a_, b.b_, ~a );
}
#endif
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Addition operator for fusing a 64-bit floating point multiplication and addition.
// \ingroup simd
//
// \param a The left-hand side SIMD multiplication expression.
// \param b The right-hand side SIMD multiplication expression.
// \return The result of the FMA operation.
//
// This operator fuses a 64-bit floating point multiplication with the addition of a 64-bit
// floating point operand. It returns an expression representing the fused multiply-add (FMA)
// operation.
*/
#if BLAZE_FMA_MODE
template< typename T1    // Type of the first operand of the left-hand side multiplication
        , typename T2    // Type of the second operand of the left-hand side multiplication
        , typename T3    // Type of the first operand of the right-hand side multiplication
        , typename T4 >  // Type of the second operand of the right-hand side multiplication
BLAZE_ALWAYS_INLINE const SIMDf64FmaddExpr< T1, T2, SIMDf64MultExpr<T3,T4> >
   operator+( const SIMDf64MultExpr<T1,T2>& a, const SIMDf64MultExpr<T3,T4>& b )
{
   return SIMDf64FmaddExpr< T1, T2, SIMDf64MultExpr<T3,T4> >( a.a_, a.b_, b );
}
#endif
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Restructuring operator for the addition of a 64-bit floating point FMA expression with
//        a 64-bit floating point operand.
// \ingroup simd
//
// \param a The left-hand side SIMD FMA expression.
// \param b The right-hand side SIMD addition operand.
// \return The restructured expression.
//
// This operator implements a performance optimized treatment of the addition of a 64-bit floating
// point FMA expression and a 64-bit floating point operand. It restructures the expression
// \f$ (a*b+c) + d \f$ to the expression \f$ (a*b) + (c+d) \f$.
*/
#if BLAZE_FMA_MODE
template< typename T1    // Type of the first FMA multiplication operand
        , typename T2    // Type of the second FMA multiplication operand
        , typename T3    // Type of the FMA addition operand
        , typename T4 >  // Type of the second addition operand
BLAZE_ALWAYS_INLINE const auto
   operator+( const SIMDf64FmaddExpr<T1,T2,T3>& a, const SIMDf64<T4>& b )
{
   return ( a.a_ * a.b_ ) + ( a.c_ + (~b) );
}
#endif
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Restructuring operator for the addition of a 64-bit floating point operand with a
//        64-bit floating point FMA expression.
// \ingroup simd
//
// \param a The left-hand side SIMD addition operand.
// \param b The right-hand side SIMD FMA expression.
// \return The restructured expression.
//
// This operator implements a performance optimized treatment of the addition of a 64-bit floating
// point operand and a 64-bit floating point FMA expression. It restructures the expression
// \f$ a + (b*c+d) \f$ to the expression \f$ (b*c) + (d+a) \f$.
*/
#if BLAZE_FMA_MODE
template< typename T1    // Type of the first addition operand
        , typename T2    // Type of the first FMA multiplication operand
        , typename T3    // Type of the second FMA multiplication operand
        , typename T4 >  // Type of the FMA addition operand
BLAZE_ALWAYS_INLINE const auto
   operator+( const SIMDf64<T1>& a, const SIMDf64FmaddExpr<T2,T3,T4>& b )
{
   return ( b.a_ * b.b_ ) + ( b.c_ + (~a) );
}
#endif
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Restructuring operator for the addition of a 64-bit floating point FMA expression with
//        a 64-bit floating point multiplication expression.
// \ingroup simd
//
// \param a The left-hand side SIMD FMA expression.
// \param b The right-hand side SIMD multiplication expression.
// \return The restructured expression.
//
// This operator implements a performance optimized treatment of the addition of a 64-bit floating
// point FMA expression and a 64-bit floating point multiplication expression. It restructures the
// expression \f$ (a*b+c) + (d*e) \f$ to the expression \f$ (a*b) + (d*e+c)\f$.
*/
#if BLAZE_FMA_MODE
template< typename T1    // Type of the first FMA multiplication operand
        , typename T2    // Type of the second FMA multiplication operand
        , typename T3    // Type of the FMA addition operand
        , typename T4    // Type of the first multiplication operand
        , typename T5 >  // Type of the second multiplication operand
BLAZE_ALWAYS_INLINE const SIMDf64FmaddExpr< T4, T5, SIMDf64FmaddExpr<T1,T2,T3> >
   operator+( const SIMDf64FmaddExpr<T1,T2,T3>& a, const SIMDf64MultExpr<T4,T5>& b )
{
   return SIMDf64FmaddExpr< T4, T5, SIMDf64FmaddExpr<T1,T2,T3> >( b.a_, b.b_, a );
}
#endif
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Restructuring operator for the addition of a 64-bit floating point multiplication
//        expression with a 64-bit floating point FMA expression.
// \ingroup simd
//
// \param a The left-hand side SIMD multiplication expression.
// \param b The right-hand side SIMD FMA expression.
// \return The restructured expression.
//
// This operator implements a performance optimized treatment of the addition of a 64-bit floating
// point multiplication expression and a 64-bit floating point FMA expression. It restructures the
// expression \f$ (a*b) + (c*d+e) \f$ to the expression \f$ (a*b) + (c*d+e)\f$.
*/
#if BLAZE_FMA_MODE
template< typename T1    // Type of the first multiplication operand
        , typename T2    // Type of the second multiplication operand
        , typename T3    // Type of the first FMA multiplication operand
        , typename T4    // Type of the second FMA multiplication operand
        , typename T5 >  // Type of the FMA addition operand
BLAZE_ALWAYS_INLINE const SIMDf64FmaddExpr< T1, T2, SIMDf64FmaddExpr<T3,T4,T5> >
   operator+( const SIMDf64MultExpr<T1,T2>& a, const SIMDf64FmaddExpr<T3,T4,T5>& b )
{
   return SIMDf64FmaddExpr< T1, T2, SIMDf64FmaddExpr<T3,T4,T5> >( a.a_, a.b_, b );
}
#endif
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Restructuring operator for the addition of two 64-bit floating point FMA expressions.
// \ingroup simd
//
// \param a The left-hand side SIMD FMA expression.
// \param b The right-hand side SIMD FMA expression.
// \return The restructured expression.
//
// This operator implements a performance optimized treatment of the addition of two 64-bit floating
// point FMA expressions. It restructures the expression \f$ (a*b+c) + (d*e+f) \f$ to the expression
// \f$ (a*b) + (d*e+c+f)\f$.
*/
#if BLAZE_FMA_MODE
template< typename T1    // Type of the first multiplication operand of the left-hand side FMA
        , typename T2    // Type of the second multiplication operand of the left-hand side FMA
        , typename T3    // Type of the addition operand of the left-hand side FMA
        , typename T4    // Type of the first multiplication operand of the right-hand side FMA
        , typename T5    // Type of the second multiplication operand of the right-hand side FMA
        , typename T6 >  // Type of the addition operand of the right-hand side FMA
BLAZE_ALWAYS_INLINE const auto
   operator+( const SIMDf64FmaddExpr<T1,T2,T3>& a, const SIMDf64FmaddExpr<T4,T5,T6>& b )
{
   return ( a.a_ * a.b_ ) + ( ( b.a_ * b.b_ ) + ( a.c_ + b.c_ ) );
}
#endif
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Restructuring operator for the addition of two 64-bit floating point FMA expressions.
// \ingroup simd
//
// \param a The left-hand side SIMD FMA expression.
// \param b The right-hand side SIMD FMA expression.
// \return The restructured expression.
//
// This operator implements a performance optimized treatment of the addition of two 64-bit floating
// point FMA expressions. It restructures the expression \f$ (a*b+c) + (d*e-f) \f$ to the expression
// \f$ (a*b) + (d*e+c-f)\f$.
*/
#if BLAZE_FMA_MODE
template< typename T1    // Type of the first multiplication operand of the left-hand side FMA
        , typename T2    // Type of the second multiplication operand of the left-hand side FMA
        , typename T3    // Type of the addition operand of the left-hand side FMA
        , typename T4    // Type of the first multiplication operand of the right-hand side FMA
        , typename T5    // Type of the second multiplication operand of the right-hand side FMA
        , typename T6 >  // Type of the subtraction operand of the right-hand side FMA
BLAZE_ALWAYS_INLINE const auto
   operator+( const SIMDf64FmaddExpr<T1,T2,T3>& a, const SIMDf64FmsubExpr<T4,T5,T6>& b )
{
   return ( a.a_ * a.b_ ) + ( ( b.a_ * b.b_ ) + ( a.c_ - b.c_ ) );
}
#endif
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Restructuring operator for the addition of two 64-bit floating point FMA expressions.
// \ingroup simd
//
// \param a The left-hand side SIMD FMA expression.
// \param b The right-hand side SIMD FMA expression.
// \return The restructured expression.
//
// This operator implements a performance optimized treatment of the addition of two 64-bit floating
// point FMA expressions. It restructures the expression \f$ (a*b-c) + (d*e+f) \f$ to the expression
// \f$ (a*b) + (d*e+f-c)\f$.
*/
#if BLAZE_FMA_MODE
template< typename T1    // Type of the first multiplication operand of the left-hand side FMA
        , typename T2    // Type of the second multiplication operand of the left-hand side FMA
        , typename T3    // Type of the subtraction operand of the left-hand side FMA
        , typename T4    // Type of the first multiplication operand of the right-hand side FMA
        , typename T5    // Type of the second multiplication operand of the right-hand side FMA
        , typename T6 >  // Type of the addition operand of the right-hand side FMA
BLAZE_ALWAYS_INLINE const auto
   operator+( const SIMDf64FmsubExpr<T1,T2,T3>& a, const SIMDf64FmaddExpr<T4,T5,T6>& b )
{
   return ( a.a_ * a.b_ ) + ( ( b.a_ * b.b_ ) + ( b.c_ - a.c_ ) );
}
#endif
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Restructuring operator for the addition of two 64-bit floating point FMA expressions.
// \ingroup simd
//
// \param a The left-hand side SIMD FMA expression.
// \param b The right-hand side SIMD FMA expression.
// \return The restructured expression.
//
// This operator implements a performance optimized treatment of the addition of two 64-bit floating
// point FMA expressions. It restructures the expression \f$ (a*b-c) + (d*e-f) \f$ to the expression
// \f$ (a*b) + (d*e-f-c)\f$.
*/
#if BLAZE_FMA_MODE
template< typename T1    // Type of the first multiplication operand of the left-hand side FMA
        , typename T2    // Type of the second multiplication operand of the left-hand side FMA
        , typename T3    // Type of the subtraction operand of the left-hand side FMA
        , typename T4    // Type of the first multiplication operand of the right-hand side FMA
        , typename T5    // Type of the second multiplication operand of the right-hand side FMA
        , typename T6 >  // Type of the subtraction operand of the right-hand side FMA
BLAZE_ALWAYS_INLINE const auto
   operator+( const SIMDf64FmsubExpr<T1,T2,T3>& a, const SIMDf64FmsubExpr<T4,T5,T6>& b )
{
   return ( a.a_ * a.b_ ) + ( ( b.a_ * b.b_ ) - ( b.c_ + a.c_ ) );
}
#endif
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Subtraction operator for fusing a 64-bit floating point multiplication and subtraction.
// \ingroup simd
//
// \param a The left-hand side SIMD multiplication expression.
// \param b The right-hand side SIMD subtraction operand.
// \return The result of the FMA operation.
//
// This operator fuses a 64-bit floating point multiplication with the subtraction of a 64-bit
// floating point operand. It returns an expression representing the fused multiply-subtract
// operation.
*/
#if BLAZE_FMA_MODE
template< typename T1    // Type of the first multiplication operand
        , typename T2    // Type of the second multiplication operand
        , typename T3 >  // Type of the second subtraction operand
BLAZE_ALWAYS_INLINE const SIMDf64FmsubExpr<T1,T2,T3>
   operator-( const SIMDf64MultExpr<T1,T2>& a, const SIMDf64<T3>& b )
{
   return SIMDf64FmsubExpr<T1,T2,T3>( a.a_, a.b_, ~b );
}
#endif
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Subtraction operator for fusing a 64-bit floating point multiplication and subtraction.
// \ingroup simd
//
// \param a The left-hand side SIMD multiplication expression.
// \param b The right-hand side SIMD multiplication expression.
// \return The result of the FMA operation.
//
// This operator fuses a 64-bit floating point multiplication with the subtraction of a 64-bit
// floating point operand. It returns an expression representing the fused multiply-subtract
// (FMA) operation.
*/
#if BLAZE_FMA_MODE
template< typename T1    // Type of the first operand of the left-hand side multiplication
        , typename T2    // Type of the second operand of the left-hand side multiplication
        , typename T3    // Type of the first operand of the right-hand side multiplication
        , typename T4 >  // Type of the second operand of the right-hand side multiplication
BLAZE_ALWAYS_INLINE const SIMDf64FmsubExpr< T1, T2, SIMDf64MultExpr<T3,T4> >
   operator-( const SIMDf64MultExpr<T1,T2>& a, const SIMDf64MultExpr<T3,T4>& b )
{
   return SIMDf64FmsubExpr< T1, T2, SIMDf64MultExpr<T3,T4> >( a.a_, a.b_, b );
}
#endif
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Restructuring operator for the subtraction of a 64-bit floating point FMA expression
//        with a 64-bit floating point operand.
// \ingroup simd
//
// \param a The left-hand side SIMD FMA expression.
// \param b The right-hand side SIMD subtraction operand.
// \return The restructured expression.
//
// This operator implements a performance optimized treatment of the subtraction of a 64-bit
// floating point FMA expression and a 64-bit floating point operand. It restructures the
// expression \f$ (a*b+c) + d \f$ to the expression \f$ (a*b) + (c+d) \f$.
*/
#if BLAZE_FMA_MODE
template< typename T1    // Type of the first FMA multiplication operand
        , typename T2    // Type of the second FMA multiplication operand
        , typename T3    // Type of the FMA subtraction operand
        , typename T4 >  // Type of the second subtraction operand
BLAZE_ALWAYS_INLINE const auto
   operator-( const SIMDf64FmsubExpr<T1,T2,T3>& a, const SIMDf64<T4>& b )
{
   return ( a.a_ * a.b_ ) - ( a.c_ + (~b) );
}
#endif
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Restructuring operator for the subtraction of a 64-bit floating point FMA expression
//        with a 64-bit floating point multiplication expression.
// \ingroup simd
//
// \param a The left-hand side SIMD FMA expression.
// \param b The right-hand side SIMD multiplication expression.
// \return The restructured expression.
//
// This operator implements a performance optimized treatment of the subtraction of a 64-bit
// floating point FMA expression and a 64-bit floating point multiplication expression. It
// restructures the expression \f$ (a*b-c) - (d*e) \f$ to the expression \f$ (a*b) - (d*e+c)\f$.
*/
#if BLAZE_FMA_MODE
template< typename T1    // Type of the first FMA multiplication operand
        , typename T2    // Type of the second FMA multiplication operand
        , typename T3    // Type of the FMA subtraction operand
        , typename T4    // Type of the first multiplication operand
        , typename T5 >  // Type of the second multiplication operand
BLAZE_ALWAYS_INLINE const auto
   operator-( const SIMDf64FmsubExpr<T1,T2,T3>& a, const SIMDf64MultExpr<T4,T5>& b )
{
   return ( a.a_ * a.b_ ) - ( b.a_ * b.b_ + a.c_ );
}
#endif
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Restructuring operator for the subtraction of a 64-bit floating point multiplication
//        expression with a 64-bit floating point FMA expression.
// \ingroup simd
//
// \param a The left-hand side SIMD multiplication expression.
// \param b The right-hand side SIMD FMA expression.
// \return The restructured expression.
//
// This operator implements a performance optimized treatment of the subtraction of a 64-bit
// floating point multiplication expression and a 64-bit floating point FMA expression. It
// restructures the expression \f$ (a*b) - (c*d+e) \f$ to the expression \f$ (a*b) - (c*d+e)\f$.
*/
#if BLAZE_FMA_MODE
template< typename T1    // Type of the first multiplication operand
        , typename T2    // Type of the second multiplication operand
        , typename T3    // Type of the first FMA multiplication operand
        , typename T4    // Type of the second FMA multiplication operand
        , typename T5 >  // Type of the FMA subtraction operand
BLAZE_ALWAYS_INLINE const SIMDf64FmsubExpr< T1, T2, SIMDf64FmsubExpr<T3,T4,T5> >
   operator-( const SIMDf64MultExpr<T1,T2>& a, const SIMDf64FmsubExpr<T3,T4,T5>& b )
{
   return SIMDf64FmsubExpr< T1, T2, SIMDf64FmsubExpr<T3,T4,T5> >( a.a_, a.b_, b );
}
#endif
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Restructuring operator for the subtraction of two 64-bit floating point FMA expressions.
// \ingroup simd
//
// \param a The left-hand side SIMD FMA expression.
// \param b The right-hand side SIMD FMA expression.
// \return The restructured expression.
//
// This operator implements a performance optimized treatment of the subtraction of two 64-bit
// floating point FMA expressions. It restructures the expression \f$ (a*b+c) - (d*e+f) \f$ to
// the expression \f$ (a*b) - (d*e+f-c)\f$.
*/
#if BLAZE_FMA_MODE
template< typename T1    // Type of the first multiplication operand of the left-hand side FMA
        , typename T2    // Type of the second multiplication operand of the left-hand side FMA
        , typename T3    // Type of the addition operand of the left-hand side FMA
        , typename T4    // Type of the first multiplication operand of the right-hand side FMA
        , typename T5    // Type of the second multiplication operand of the right-hand side FMA
        , typename T6 >  // Type of the addition operand of the right-hand side FMA
BLAZE_ALWAYS_INLINE const auto
   operator-( const SIMDf64FmaddExpr<T1,T2,T3>& a, const SIMDf64FmaddExpr<T4,T5,T6>& b )
{
   return ( a.a_ * a.b_ ) - ( ( b.a_ * b.b_ ) + ( b.c_ - a.c_ ) );
}
#endif
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Restructuring operator for the subtraction of two 64-bit floating point FMA expressions.
// \ingroup simd
//
// \param a The left-hand side SIMD FMA expression.
// \param b The right-hand side SIMD FMA expression.
// \return The restructured expression.
//
// This operator implements a performance optimized treatment of the subtraction of two 64-bit
// floating point FMA expressions. It restructures the expression \f$ (a*b+c) - (d*e-f) \f$ to
// the expression \f$ (a*b) - (d*e+f-c)\f$.
*/
#if BLAZE_FMA_MODE
template< typename T1    // Type of the first multiplication operand of the left-hand side FMA
        , typename T2    // Type of the second multiplication operand of the left-hand side FMA
        , typename T3    // Type of the addition operand of the left-hand side FMA
        , typename T4    // Type of the first multiplication operand of the right-hand side FMA
        , typename T5    // Type of the second multiplication operand of the right-hand side FMA
        , typename T6 >  // Type of the subtraction operand of the right-hand side FMA
BLAZE_ALWAYS_INLINE const auto
   operator-( const SIMDf64FmaddExpr<T1,T2,T3>& a, const SIMDf64FmsubExpr<T4,T5,T6>& b )
{
   return ( a.a_ * a.b_ ) - ( ( b.a_ * b.b_ ) - ( a.c_ + b.c_ ) );
}
#endif
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Restructuring operator for the subtraction of two 64-bit floating point FMA expressions.
// \ingroup simd
//
// \param a The left-hand side SIMD FMA expression.
// \param b The right-hand side SIMD FMA expression.
// \return The restructured expression.
//
// This operator implements a performance optimized treatment of the subtraction of two 64-bit
// floating point FMA expressions. It restructures the expression \f$ (a*b-c) - (d*e+f) \f$ to
// the expression \f$ (a*b) - (d*e+c-f)\f$.
*/
#if BLAZE_FMA_MODE
template< typename T1    // Type of the first multiplication operand of the left-hand side FMA
        , typename T2    // Type of the second multiplication operand of the left-hand side FMA
        , typename T3    // Type of the subtraction operand of the left-hand side FMA
        , typename T4    // Type of the first multiplication operand of the right-hand side FMA
        , typename T5    // Type of the second multiplication operand of the right-hand side FMA
        , typename T6 >  // Type of the addition operand of the right-hand side FMA
BLAZE_ALWAYS_INLINE const auto
   operator-( const SIMDf64FmsubExpr<T1,T2,T3>& a, const SIMDf64FmaddExpr<T4,T5,T6>& b )
{
   return ( a.a_ * a.b_ ) - ( ( b.a_ * b.b_ ) + ( a.c_ + b.c_ ) );
}
#endif
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Restructuring operator for the subtraction of two 64-bit floating point FMA expressions.
// \ingroup simd
//
// \param a The left-hand side SIMD FMA expression.
// \param b The right-hand side SIMD FMA expression.
// \return The restructured expression.
//
// This operator implements a performance optimized treatment of the subtraction of two 64-bit
// floating point FMA expressions. It restructures the expression \f$ (a*b-c) - (d*e-f) \f$ to
// the expression \f$ (a*b) - (d*e+c-f)\f$.
*/
#if BLAZE_FMA_MODE
template< typename T1    // Type of the first multiplication operand of the left-hand side FMA
        , typename T2    // Type of the second multiplication operand of the left-hand side FMA
        , typename T3    // Type of the subtraction operand of the left-hand side FMA
        , typename T4    // Type of the first multiplication operand of the right-hand side FMA
        , typename T5    // Type of the second multiplication operand of the right-hand side FMA
        , typename T6 >  // Type of the subtraction operand of the right-hand side FMA
BLAZE_ALWAYS_INLINE const auto
   operator-( const SIMDf64FmsubExpr<T1,T2,T3>& a, const SIMDf64FmsubExpr<T4,T5,T6>& b )
{
   return ( a.a_ * a.b_ ) - ( ( b.a_ * b.b_ ) + ( a.c_ - b.c_ ) );
}
#endif
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
