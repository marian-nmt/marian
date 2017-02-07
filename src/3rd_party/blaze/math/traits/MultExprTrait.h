//=================================================================================================
/*!
//  \file blaze/math/traits/MultExprTrait.h
//  \brief Header file for the MultExprTrait class template
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

#ifndef _BLAZE_MATH_TRAITS_MULTEXPRTRAIT_H_
#define _BLAZE_MATH_TRAITS_MULTEXPRTRAIT_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/traits/DMatDMatMultExprTrait.h>
#include <blaze/math/traits/DMatDVecMultExprTrait.h>
#include <blaze/math/traits/DMatScalarMultExprTrait.h>
#include <blaze/math/traits/DMatSMatMultExprTrait.h>
#include <blaze/math/traits/DMatSVecMultExprTrait.h>
#include <blaze/math/traits/DMatTDMatMultExprTrait.h>
#include <blaze/math/traits/DMatTSMatMultExprTrait.h>
#include <blaze/math/traits/DVecDVecMultExprTrait.h>
#include <blaze/math/traits/DVecScalarMultExprTrait.h>
#include <blaze/math/traits/DVecSVecMultExprTrait.h>
#include <blaze/math/traits/DVecTDVecMultExprTrait.h>
#include <blaze/math/traits/DVecTSVecMultExprTrait.h>
#include <blaze/math/traits/MultTrait.h>
#include <blaze/math/traits/SMatDMatMultExprTrait.h>
#include <blaze/math/traits/SMatDVecMultExprTrait.h>
#include <blaze/math/traits/SMatScalarMultExprTrait.h>
#include <blaze/math/traits/SMatSMatMultExprTrait.h>
#include <blaze/math/traits/SMatSVecMultExprTrait.h>
#include <blaze/math/traits/SMatTDMatMultExprTrait.h>
#include <blaze/math/traits/SMatTSMatMultExprTrait.h>
#include <blaze/math/traits/SVecDVecMultExprTrait.h>
#include <blaze/math/traits/SVecScalarMultExprTrait.h>
#include <blaze/math/traits/SVecSVecMultExprTrait.h>
#include <blaze/math/traits/SVecTDVecMultExprTrait.h>
#include <blaze/math/traits/SVecTSVecMultExprTrait.h>
#include <blaze/math/traits/TDMatDMatMultExprTrait.h>
#include <blaze/math/traits/TDMatDVecMultExprTrait.h>
#include <blaze/math/traits/TDMatScalarMultExprTrait.h>
#include <blaze/math/traits/TDMatSMatMultExprTrait.h>
#include <blaze/math/traits/TDMatSVecMultExprTrait.h>
#include <blaze/math/traits/TDMatTDMatMultExprTrait.h>
#include <blaze/math/traits/TDMatTSMatMultExprTrait.h>
#include <blaze/math/traits/TDVecDMatMultExprTrait.h>
#include <blaze/math/traits/TDVecDVecMultExprTrait.h>
#include <blaze/math/traits/TDVecScalarMultExprTrait.h>
#include <blaze/math/traits/TDVecSMatMultExprTrait.h>
#include <blaze/math/traits/TDVecSVecMultExprTrait.h>
#include <blaze/math/traits/TDVecTDMatMultExprTrait.h>
#include <blaze/math/traits/TDVecTDVecMultExprTrait.h>
#include <blaze/math/traits/TDVecTSMatMultExprTrait.h>
#include <blaze/math/traits/TDVecTSVecMultExprTrait.h>
#include <blaze/math/traits/TSMatDMatMultExprTrait.h>
#include <blaze/math/traits/TSMatDVecMultExprTrait.h>
#include <blaze/math/traits/TSMatScalarMultExprTrait.h>
#include <blaze/math/traits/TSMatSMatMultExprTrait.h>
#include <blaze/math/traits/TSMatSVecMultExprTrait.h>
#include <blaze/math/traits/TSMatTDMatMultExprTrait.h>
#include <blaze/math/traits/TSMatTSMatMultExprTrait.h>
#include <blaze/math/traits/TSVecDMatMultExprTrait.h>
#include <blaze/math/traits/TSVecDVecMultExprTrait.h>
#include <blaze/math/traits/TSVecScalarMultExprTrait.h>
#include <blaze/math/traits/TSVecSMatMultExprTrait.h>
#include <blaze/math/traits/TSVecSVecMultExprTrait.h>
#include <blaze/math/traits/TSVecTDMatMultExprTrait.h>
#include <blaze/math/traits/TSVecTDVecMultExprTrait.h>
#include <blaze/math/traits/TSVecTSMatMultExprTrait.h>
#include <blaze/math/traits/TSVecTSVecMultExprTrait.h>
#include <blaze/math/typetraits/IsColumnVector.h>
#include <blaze/math/typetraits/IsDenseMatrix.h>
#include <blaze/math/typetraits/IsDenseVector.h>
#include <blaze/math/typetraits/IsMatrix.h>
#include <blaze/math/typetraits/IsRowMajorMatrix.h>
#include <blaze/math/typetraits/IsRowVector.h>
#include <blaze/math/typetraits/IsVector.h>
#include <blaze/util/InvalidType.h>
#include <blaze/util/mpl/If.h>
#include <blaze/util/mpl/Or.h>
#include <blaze/util/typetraits/Decay.h>
#include <blaze/util/typetraits/IsConst.h>
#include <blaze/util/typetraits/IsNumeric.h>
#include <blaze/util/typetraits/IsReference.h>
#include <blaze/util/typetraits/IsVolatile.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Evaluation of the resulting expression type of a multiplication.
// \ingroup math_traits
//
// Via this type trait it is possible to evaluate the return type of a multiplication expression
// between scalars, vectors, and matrices. Given the two types \a T1 and \a T2, which must be
// either scalar, vector, or matrix types, the nested type \a Type corresponds to the resulting
// return type. In case \a T1 or \a T2 don't fit or if the two types cannot be multiplied, the
// resulting data type \a Type is set to \a INVALID_TYPE.
*/
template< typename T1    // Type of the left-hand side multiplication operand
        , typename T2 >  // Type of the right-hand side multiplication operand
struct MultExprTrait
{
 private:
   //**struct Failure******************************************************************************
   /*! \cond BLAZE_INTERNAL */
   struct Failure { using Type = INVALID_TYPE; };
   /*! \endcond */
   //**********************************************************************************************

   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   using Tmp = If_< IsMatrix<T1>
                  , If_< IsDenseMatrix<T1>
                       , If_< IsRowMajorMatrix<T1>
                            , If_< IsMatrix<T2>
                                 , If_< IsDenseMatrix<T2>
                                      , If_< IsRowMajorMatrix<T2>
                                           , DMatDMatMultExprTrait<T1,T2>
                                           , DMatTDMatMultExprTrait<T1,T2> >
                                      , If_< IsRowMajorMatrix<T2>
                                           , DMatSMatMultExprTrait<T1,T2>
                                           , DMatTSMatMultExprTrait<T1,T2> > >
                                 , If_< IsVector<T2>
                                      , If_< IsDenseVector<T2>
                                           , If_< IsColumnVector<T2>
                                                , DMatDVecMultExprTrait<T1,T2>
                                                , Failure >
                                           , If_< IsColumnVector<T2>
                                                , DMatSVecMultExprTrait<T1,T2>
                                                , Failure > >
                                      , If_< IsNumeric<T2>
                                           , DMatScalarMultExprTrait<T1,T2>
                                           , Failure > > >
                            , If_< IsMatrix<T2>
                                 , If_< IsDenseMatrix<T2>
                                      , If_< IsRowMajorMatrix<T2>
                                           , TDMatDMatMultExprTrait<T1,T2>
                                           , TDMatTDMatMultExprTrait<T1,T2> >
                                      , If_< IsRowMajorMatrix<T2>
                                           , TDMatSMatMultExprTrait<T1,T2>
                                           , TDMatTSMatMultExprTrait<T1,T2> > >
                                 , If_< IsVector<T2>
                                      , If_< IsDenseVector<T2>
                                           , If_< IsColumnVector<T2>
                                                , TDMatDVecMultExprTrait<T1,T2>
                                                , Failure >
                                           , If_< IsColumnVector<T2>
                                                , TDMatSVecMultExprTrait<T1,T2>
                                                , Failure > >
                                      , If_< IsNumeric<T2>
                                           , TDMatScalarMultExprTrait<T1,T2>
                                           , Failure > > > >
                       , If_< IsRowMajorMatrix<T1>
                            , If_< IsMatrix<T2>
                                 , If_< IsDenseMatrix<T2>
                                      , If_< IsRowMajorMatrix<T2>
                                           , SMatDMatMultExprTrait<T1,T2>
                                           , SMatTDMatMultExprTrait<T1,T2> >
                                      , If_< IsRowMajorMatrix<T2>
                                           , SMatSMatMultExprTrait<T1,T2>
                                           , SMatTSMatMultExprTrait<T1,T2> > >
                                 , If_< IsVector<T2>
                                      , If_< IsDenseVector<T2>
                                           , If_< IsColumnVector<T2>
                                                , SMatDVecMultExprTrait<T1,T2>
                                                , Failure >
                                           , If_< IsColumnVector<T2>
                                                , SMatSVecMultExprTrait<T1,T2>
                                                , Failure > >
                                      , If_< IsNumeric<T2>
                                           , SMatScalarMultExprTrait<T1,T2>
                                           , Failure > > >
                            , If_< IsMatrix<T2>
                                 , If_< IsDenseMatrix<T2>
                                      , If_< IsRowMajorMatrix<T2>
                                           , TSMatDMatMultExprTrait<T1,T2>
                                           , TSMatTDMatMultExprTrait<T1,T2> >
                                      , If_< IsRowMajorMatrix<T2>
                                           , TSMatSMatMultExprTrait<T1,T2>
                                           , TSMatTSMatMultExprTrait<T1,T2> > >
                                 , If_< IsVector<T2>
                                      , If_< IsDenseVector<T2>
                                           , If_< IsColumnVector<T2>
                                                , TSMatDVecMultExprTrait<T1,T2>
                                                , Failure >
                                           , If_< IsColumnVector<T2>
                                                , TSMatSVecMultExprTrait<T1,T2>
                                                , Failure > >
                                      , If_< IsNumeric<T2>
                                           , TSMatScalarMultExprTrait<T1,T2>
                                           , Failure > > > > >
                  , If_< IsVector<T1>
                       , If_< IsDenseVector<T1>
                            , If_< IsRowVector<T1>
                                 , If_< IsMatrix<T2>
                                      , If_< IsDenseMatrix<T2>
                                           , If_< IsRowMajorMatrix<T2>
                                                , TDVecDMatMultExprTrait<T1,T2>
                                                , TDVecTDMatMultExprTrait<T1,T2> >
                                           , If_< IsRowMajorMatrix<T2>
                                                , TDVecSMatMultExprTrait<T1,T2>
                                                , TDVecTSMatMultExprTrait<T1,T2> > >
                                      , If_< IsVector<T2>
                                           , If_< IsDenseVector<T2>
                                                , If_< IsRowVector<T2>
                                                     , TDVecTDVecMultExprTrait<T1,T2>
                                                     , TDVecDVecMultExprTrait<T1,T2> >
                                                , If_< IsRowVector<T2>
                                                     , TDVecTSVecMultExprTrait<T1,T2>
                                                     , TDVecSVecMultExprTrait<T1,T2> > >
                                           , If_< IsNumeric<T2>
                                                , TDVecScalarMultExprTrait<T1,T2>
                                                , Failure > > >
                                 , If_< IsVector<T2>
                                      , If_< IsDenseVector<T2>
                                           , If_< IsRowVector<T2>
                                                , DVecTDVecMultExprTrait<T1,T2>
                                                , DVecDVecMultExprTrait<T1,T2> >
                                           , If_< IsRowVector<T2>
                                                , DVecTSVecMultExprTrait<T1,T2>
                                                , DVecSVecMultExprTrait<T1,T2> > >
                                      , If_< IsNumeric<T2>
                                           , DVecScalarMultExprTrait<T1,T2>
                                           , Failure > > >
                            , If_< IsRowVector<T1>
                                 , If_< IsMatrix<T2>
                                      , If_< IsDenseMatrix<T2>
                                           , If_< IsRowMajorMatrix<T2>
                                                , TSVecDMatMultExprTrait<T1,T2>
                                                , TSVecTDMatMultExprTrait<T1,T2> >
                                           , If_< IsRowMajorMatrix<T2>
                                                , TSVecSMatMultExprTrait<T1,T2>
                                                , TSVecTSMatMultExprTrait<T1,T2> > >
                                      , If_< IsVector<T2>
                                           , If_< IsDenseVector<T2>
                                                , If_< IsRowVector<T2>
                                                     , TSVecTDVecMultExprTrait<T1,T2>
                                                     , TSVecDVecMultExprTrait<T1,T2> >
                                                , If_< IsRowVector<T2>
                                                     , TSVecTSVecMultExprTrait<T1,T2>
                                                     , TSVecSVecMultExprTrait<T1,T2> > >
                                           , If_< IsNumeric<T2>
                                                , TSVecScalarMultExprTrait<T1,T2>
                                                , Failure > > >
                                 , If_< IsVector<T2>
                                      , If_< IsDenseVector<T2>
                                           , If_< IsRowVector<T2>
                                                , SVecTDVecMultExprTrait<T1,T2>
                                                , SVecDVecMultExprTrait<T1,T2> >
                                           , If_< IsRowVector<T2>
                                                , SVecTSVecMultExprTrait<T1,T2>
                                                , SVecSVecMultExprTrait<T1,T2> > >
                                      , If_< IsNumeric<T2>
                                           , SVecScalarMultExprTrait<T1,T2>
                                           , Failure > > > >
                       , If_< IsNumeric<T1>
                            , If_< IsMatrix<T2>
                                 , If_< IsDenseMatrix<T2>
                                      , If_< IsRowMajorMatrix<T2>
                                           , DMatScalarMultExprTrait<T2,T1>
                                           , TDMatScalarMultExprTrait<T2,T1> >
                                      , If_< IsRowMajorMatrix<T2>
                                           , SMatScalarMultExprTrait<T2,T1>
                                           , TSMatScalarMultExprTrait<T2,T1> > >
                                 , If_< IsVector<T2>
                                      , If_< IsDenseVector<T2>
                                           , If_< IsRowVector<T2>
                                                , TDVecScalarMultExprTrait<T2,T1>
                                                , DVecScalarMultExprTrait<T2,T1> >
                                           , If_< IsRowVector<T2>
                                                , TSVecScalarMultExprTrait<T2,T1>
                                                , SVecScalarMultExprTrait<T2,T1> > >
                                      , If_< IsNumeric<T2>
                                           , MultTrait<T1,T2>
                                           , Failure > > >
                            , Failure > > >;
   /*! \endcond */
   //**********************************************************************************************

 public:
   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   using Type = typename If_< Or< IsConst<T1>, IsVolatile<T1>, IsReference<T1>
                                , IsConst<T2>, IsVolatile<T2>, IsReference<T2> >
                            , MultExprTrait< Decay_<T1>, Decay_<T2> >
                            , Tmp >::Type;
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Auxiliary alias declaration for the MultExprTrait class template.
// \ingroup math_traits
//
// The MultExprTrait_ alias declaration provides a convenient shortcut to access the nested \a Type
// of the MultExprTrait class template. For instance, given the types \a T1 and \a T2 the following
// two type definitions are identical:

   \code
   using Type1 = typename MultExprTrait<T1,T2>::Type;
   using Type2 = MultExprTrait_<T1,T2>;
   \endcode
*/
template< typename T1    // Type of the left-hand side multiplication operand
        , typename T2 >  // Type of the right-hand side multiplication operand
using MultExprTrait_ = typename MultExprTrait<T1,T2>::Type;
//*************************************************************************************************

} // namespace blaze

#endif
