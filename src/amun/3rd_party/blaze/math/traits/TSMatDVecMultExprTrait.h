//=================================================================================================
/*!
//  \file blaze/math/traits/TSMatDVecMultExprTrait.h
//  \brief Header file for the TSMatDVecMultExprTrait class template
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

#ifndef _BLAZE_MATH_TRAITS_TSMATDVECMULTEXPRTRAIT_H_
#define _BLAZE_MATH_TRAITS_TSMATDVECMULTEXPRTRAIT_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/expressions/Forward.h>
#include <blaze/math/traits/TSMatTransExprTrait.h>
#include <blaze/math/typetraits/IsColumnMajorMatrix.h>
#include <blaze/math/typetraits/IsColumnVector.h>
#include <blaze/math/typetraits/IsDenseVector.h>
#include <blaze/math/typetraits/IsSparseMatrix.h>
#include <blaze/math/typetraits/IsSymmetric.h>
#include <blaze/util/InvalidType.h>
#include <blaze/util/mpl/And.h>
#include <blaze/util/mpl/If.h>
#include <blaze/util/mpl/Or.h>
#include <blaze/util/typetraits/Decay.h>
#include <blaze/util/typetraits/IsConst.h>
#include <blaze/util/typetraits/IsReference.h>
#include <blaze/util/typetraits/IsVolatile.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Evaluation of the expression type of a transpose sparse matrix/dense vector multiplication.
// \ingroup math_traits
//
// Via this type trait it is possible to evaluate the resulting expression type of a transpose
// sparse matrix/dense vector multiplication. Given the column-major sparse matrix type \a MT
// and the non-transpose dense vector type \a VT, the nested type \a Type corresponds to the
// resulting expression type. In case either \a MT is not a column-major sparse matrix type or
// \a VT is not a non-transpose dense vector type, the resulting data type \a Type is set to
// \a INVALID_TYPE.
*/
template< typename MT    // Type of the left-hand side column-major sparse matrix
        , typename VT >  // Type of the right-hand side non-transpose dense vector
struct TSMatDVecMultExprTrait
{
 private:
   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   using Tmp = If< And< IsSparseMatrix<MT>, IsColumnMajorMatrix<MT>
                      , IsDenseVector<VT> , IsColumnVector<VT> >
                 , If_< IsSymmetric<MT>
                      , SMatDVecMultExpr< TSMatTransExprTrait_<MT>, VT >
                      , TSMatDVecMultExpr<MT,VT> >
                 , INVALID_TYPE >;
   /*! \endcond */
   //**********************************************************************************************

 public:
   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   using Type = typename If_< Or< IsConst<MT>, IsVolatile<MT>, IsReference<MT>
                                , IsConst<VT>, IsVolatile<VT>, IsReference<VT> >
                            , TSMatDVecMultExprTrait< Decay_<MT>, Decay_<VT> >
                            , Tmp >::Type;
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Auxiliary alias declaration for the TSMatDVecMultExprTrait class template.
// \ingroup math_traits
//
// The TSMatDVecMultExprTrait_ alias declaration provides a convenient shortcut to access
// the nested \a Type of the TSMatDVecMultExprTrait class template. For instance, given the
// column-major sparse matrix type \a MT and the non-transpose dense vector type \a VT the
// following two type definitions are identical:

   \code
   using Type1 = typename TSMatDVecMultExprTrait<MT,VT>::Type;
   using Type2 = TSMatDVecMultExprTrait_<MT,VT>;
   \endcode
*/
template< typename MT    // Type of the left-hand side column-major sparse matrix
        , typename VT >  // Type of the right-hand side non-transpose dense vector
using TSMatDVecMultExprTrait_ = typename TSMatDVecMultExprTrait<MT,VT>::Type;
//*************************************************************************************************

} // namespace blaze

#endif
