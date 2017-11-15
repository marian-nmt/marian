//=================================================================================================
/*!
//  \file blaze/math/expressions/TDVecDVecMultExpr.h
//  \brief Header file for the dense vector/dense vector inner product expression
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

#ifndef _BLAZE_MATH_EXPRESSIONS_TDVECDVECMULTEXPR_H_
#define _BLAZE_MATH_EXPRESSIONS_TDVECDVECMULTEXPR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/Aliases.h>
#include <blaze/math/Exception.h>
#include <blaze/math/expressions/DenseVector.h>
#include <blaze/math/SIMD.h>
#include <blaze/math/traits/MultTrait.h>
#include <blaze/math/typetraits/HasSIMDAdd.h>
#include <blaze/math/typetraits/HasSIMDMult.h>
#include <blaze/system/Optimizations.h>
#include <blaze/util/Assert.h>
#include <blaze/util/DisableIf.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/logging/FunctionTrace.h>
#include <blaze/util/Types.h>
#include <blaze/util/typetraits/IsSame.h>
#include <blaze/util/typetraits/RemoveReference.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Auxiliary helper struct for the dense vector/dense vector scalar multiplication.
// \ingroup dense_vector
*/
template< typename T1    // Type of the left-hand side dense vector
        , typename T2 >  // Type of the right-hand side dense vector
struct TDVecDVecMultExprHelper
{
   //**Type definitions****************************************************************************
   //! Composite type of the left-hand side dense vector expression.
   typedef RemoveReference_< CompositeType_<T1> >  CT1;

   //! Composite type of the right-hand side dense vector expression.
   typedef RemoveReference_< CompositeType_<T2> >  CT2;
   //**********************************************************************************************

   //**********************************************************************************************
   enum : bool { value = useOptimizedKernels &&
                         CT1::simdEnabled &&
                         CT2::simdEnabled &&
                         IsSame< ElementType_<CT1>, ElementType_<CT2> >::value &&
                         HasSIMDAdd< ElementType_<CT1>, ElementType_<CT1> >::value &&
                         HasSIMDMult< ElementType_<CT1>, ElementType_<CT1> >::value };
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL BINARY ARITHMETIC OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Default multiplication operator for the scalar product (inner product) of two dense
//        vectors (\f$ s=\vec{a}*\vec{b} \f$).
// \ingroup dense_vector
//
// \param lhs The left-hand side dense vector for the inner product.
// \param rhs The right-hand side dense vector for the inner product.
// \return The scalar product.
// \exception std::invalid_argument Vector sizes do not match.
//
// This operator represents the scalar product (inner product) of two dense vectors:

   \code
   blaze::DynamicVector<double> a, b;
   blaze::double res;
   // ... Resizing and initialization
   res = trans(a) * b;
   \endcode

// The operator returns a scalar value of the higher-order element type of the two involved
// vector element types \a T1::ElementType and \a T2::ElementType. Both vector types \a T1
// and \a T2 as well as the two element types \a T1::ElementType and \a T2::ElementType have
// to be supported by the MultTrait class template.\n
// In case the current sizes of the two given vectors don't match, a \a std::invalid_argument
// is thrown.
*/
template< typename T1    // Type of the left-hand side dense vector
        , typename T2 >  // Type of the right-hand side dense vector
inline DisableIf_< TDVecDVecMultExprHelper<T1,T2>
                 , const MultTrait_< ElementType_<T1>, ElementType_<T2> > >
   operator*( const DenseVector<T1,true>& lhs, const DenseVector<T2,false>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   if( (~lhs).size() != (~rhs).size() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Vector sizes do not match" );
   }

   typedef CompositeType_<T1>   Lhs;
   typedef CompositeType_<T2>   Rhs;
   typedef ElementType_<T1>     ET1;
   typedef ElementType_<T2>     ET2;
   typedef MultTrait_<ET1,ET2>  MultType;

   if( (~lhs).size() == 0UL ) return MultType();

   Lhs left ( ~lhs );
   Rhs right( ~rhs );

   MultType sp( left[0UL] * right[0UL] );
   size_t i( 1UL );

   for( ; (i+4UL) <= left.size(); i+=4UL ) {
      sp += left[i    ] * right[i    ] +
            left[i+1UL] * right[i+1UL] +
            left[i+2UL] * right[i+2UL] +
            left[i+3UL] * right[i+3UL];
   }
   for( ; (i+2UL) <= left.size(); i+=2UL ) {
      sp += left[i    ] * right[i    ] +
            left[i+1UL] * right[i+1UL];
   }
   for( ; i<left.size(); ++i ) {
      sp += left[i] * right[i];
   }

   return sp;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SIMD optimized multiplication operator for the scalar product (inner product) of two
//        dense vectors (\f$ s=\vec{a}*\vec{b} \f$).
// \ingroup dense_vector
//
// \param lhs The left-hand side dense vector for the inner product.
// \param rhs The right-hand side dense vector for the inner product.
// \return The scalar product.
// \exception std::invalid_argument Vector sizes do not match.
//
// This operator represents the scalar product (inner product) of two dense vectors:

   \code
   using blaze::columnVector;

   blaze::DynamicVector<double,columnVector> a, b;
   blaze::double res;
   // ... Resizing and initialization
   res = trans(a) * b;
   \endcode

// The operator returns a scalar value of the higher-order element type of the two involved
// vector element types \a T1::ElementType and \a T2::ElementType. Both vector types \a T1
// and \a T2 as well as the two element types \a T1::ElementType and \a T2::ElementType have
// to be supported by the MultTrait class template.\n
// In case the current sizes of the two given vectors don't match, a \a std::invalid_argument
// is thrown.
*/
template< typename T1    // Type of the left-hand side dense vector
        , typename T2 >  // Type of the right-hand side dense vector
inline EnableIf_< TDVecDVecMultExprHelper<T1,T2>
               , const MultTrait_< ElementType_<T1>, ElementType_<T2> > >
   operator*( const DenseVector<T1,true>& lhs, const DenseVector<T2,false>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   if( (~lhs).size() != (~rhs).size() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Vector sizes do not match" );
   }

   typedef CompositeType_<T1>   Lhs;
   typedef CompositeType_<T2>   Rhs;
   typedef ElementType_<T1>     ET1;
   typedef ElementType_<T2>     ET2;
   typedef MultTrait_<ET1,ET2>  MultType;

   enum : size_t { SIMDSIZE = SIMDTrait<MultType>::size };

   if( (~lhs).size() == 0UL ) return MultType();

   Lhs left ( ~lhs );
   Rhs right( ~rhs );

   const size_t N( left.size() );

   SIMDTrait_<MultType> xmm1, xmm2, xmm3, xmm4;
   size_t i( 0UL );

   for( ; (i+SIMDSIZE*4UL) <= N; i+=SIMDSIZE*4UL ) {
      xmm1 = xmm1 + ( left.load(i             ) * right.load(i             ) );
      xmm2 = xmm2 + ( left.load(i+SIMDSIZE    ) * right.load(i+SIMDSIZE    ) );
      xmm3 = xmm3 + ( left.load(i+SIMDSIZE*2UL) * right.load(i+SIMDSIZE*2UL) );
      xmm4 = xmm4 + ( left.load(i+SIMDSIZE*3UL) * right.load(i+SIMDSIZE*3UL) );
   }
   for( ; (i+SIMDSIZE*2UL) <= N; i+=SIMDSIZE*2UL ) {
      xmm1 = xmm1 + ( left.load(i         ) * right.load(i         ) );
      xmm2 = xmm2 + ( left.load(i+SIMDSIZE) * right.load(i+SIMDSIZE) );
   }
   for( ; (i+SIMDSIZE) <= N; i+=SIMDSIZE ) {
      xmm1 = xmm1 + ( left.load(i) * right.load(i) );
   }

   MultType sp( sum( xmm1 + xmm2 + xmm3 + xmm4 ) );

   for( ; i<N; ++i ) {
      sp += left[i] * right[i];
   }

   return sp;
}
//*************************************************************************************************

} // namespace blaze

#endif
