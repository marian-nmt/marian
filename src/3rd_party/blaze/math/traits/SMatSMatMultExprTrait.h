//=================================================================================================
/*!
//  \file blaze/math/traits/SMatSMatMultExprTrait.h
//  \brief Header file for the SMatSMatMultExprTrait class template
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

#ifndef _BLAZE_MATH_TRAITS_SMATSMATMULTEXPRTRAIT_H_
#define _BLAZE_MATH_TRAITS_SMATSMATMULTEXPRTRAIT_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/expressions/Forward.h>
#include <blaze/math/typetraits/IsRowMajorMatrix.h>
#include <blaze/math/typetraits/IsSparseMatrix.h>
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
/*!\brief Evaluation of the expression type of a sparse matrix/sparse matrix multiplication.
// \ingroup math_traits
//
// Via this type trait it is possible to evaluate the resulting expression type of a sparse
// matrix/sparse matrix multiplication. Given the two row-major sparse matrix types \a MT1 and
// \a MT2, the nested type \a Type corresponds to the resulting expression type. In case either
// \a MT1 or \a MT2 is not a row-major sparse matrix, the resulting data type \a Type is set to
// \a INVALID_TYPE.
*/
template< typename MT1    // Type of the left-hand side row-major sparse matrix
        , typename MT2 >  // Type of the right-hand side row-major sparse matrix
struct SMatSMatMultExprTrait
{
 private:
   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   using Tmp = If< And< IsSparseMatrix<MT1>, IsRowMajorMatrix<MT1>
                      , IsSparseMatrix<MT2>, IsRowMajorMatrix<MT2> >
                 , SMatSMatMultExpr<MT1,MT2>
                 , INVALID_TYPE >;
   /*! \endcond */
   //**********************************************************************************************

 public:
   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   using Type = typename If_< Or< IsConst<MT1>, IsVolatile<MT1>, IsReference<MT1>
                                , IsConst<MT2>, IsVolatile<MT2>, IsReference<MT2> >
                            , SMatSMatMultExprTrait< Decay_<MT1>, Decay_<MT2> >
                            , Tmp >::Type;
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Auxiliary alias declaration for the SMatSMatMultExprTrait class template.
// \ingroup math_traits
//
// The SMatSMatMultExprTrait_ alias declaration provides a convenient shortcut to access the
// nested \a Type of the SMatSMatMultExprTrait class template. For instance, given the row-major
// sparse matrix types \a MT1 and \a MT2 the following two type definitions are identical:

   \code
   using Type1 = typename SMatSMatMultExprTrait<MT1,MT2>::Type;
   using Type2 = SMatSMatMultExprTrait_<MT1,MT2>;
   \endcode
*/
template< typename MT1    // Type of the left-hand side row-major sparse matrix
        , typename MT2 >  // Type of the right-hand side row-major sparse matrix
using SMatSMatMultExprTrait_ = typename SMatSMatMultExprTrait<MT1,MT2>::Type;
//*************************************************************************************************

} // namespace blaze

#endif
