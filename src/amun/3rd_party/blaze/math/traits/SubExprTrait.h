//=================================================================================================
/*!
//  \file blaze/math/traits/SubExprTrait.h
//  \brief Header file for the SubExprTrait class template
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

#ifndef _BLAZE_MATH_TRAITS_SUBEXPRTRAIT_H_
#define _BLAZE_MATH_TRAITS_SUBEXPRTRAIT_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/traits/DMatDMatSubExprTrait.h>
#include <blaze/math/traits/DMatSMatSubExprTrait.h>
#include <blaze/math/traits/DMatTDMatSubExprTrait.h>
#include <blaze/math/traits/DMatTSMatSubExprTrait.h>
#include <blaze/math/traits/DVecDVecSubExprTrait.h>
#include <blaze/math/traits/DVecSVecSubExprTrait.h>
#include <blaze/math/traits/SMatDMatSubExprTrait.h>
#include <blaze/math/traits/SMatSMatSubExprTrait.h>
#include <blaze/math/traits/SMatTDMatSubExprTrait.h>
#include <blaze/math/traits/SMatTSMatSubExprTrait.h>
#include <blaze/math/traits/SubTrait.h>
#include <blaze/math/traits/SVecDVecSubExprTrait.h>
#include <blaze/math/traits/SVecSVecSubExprTrait.h>
#include <blaze/math/traits/TDMatDMatSubExprTrait.h>
#include <blaze/math/traits/TDMatSMatSubExprTrait.h>
#include <blaze/math/traits/TDMatTDMatSubExprTrait.h>
#include <blaze/math/traits/TDMatTSMatSubExprTrait.h>
#include <blaze/math/traits/TDVecTDVecSubExprTrait.h>
#include <blaze/math/traits/TDVecTSVecSubExprTrait.h>
#include <blaze/math/traits/TSMatDMatSubExprTrait.h>
#include <blaze/math/traits/TSMatSMatSubExprTrait.h>
#include <blaze/math/traits/TSMatTDMatSubExprTrait.h>
#include <blaze/math/traits/TSMatTSMatSubExprTrait.h>
#include <blaze/math/traits/TSVecTDVecSubExprTrait.h>
#include <blaze/math/traits/TSVecTSVecSubExprTrait.h>
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
/*!\brief Evaluation of the return type of a subtraction expression.
// \ingroup math_traits
//
// Via this type trait it is possible to evaluate the return type of a subtraction expression
// between scalar, vectors, and matrices. Given the two types \a T1 and \a T2, which must be
// either scalar, vector, or matrix types, the nested type \a Type corresponds to the resulting
// return type. In case \a T1 or \a T2 don't fit or if the two types cannot be subtracted, the
// resulting data type \a Type is set to \a INVALID_TYPE.
*/
template< typename T1    // Type of the left-hand side subtraction operand
        , typename T2 >  // Type of the right-hand side subtraction operand
struct SubExprTrait
{
 private:
   //**struct Failure******************************************************************************
   /*! \cond BLAZE_INTERNAL */
   struct Failure { using Type = INVALID_TYPE; };
   /*! \endcond */
   //**********************************************************************************************

 public:
   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   using Tmp = If_< IsMatrix<T1>
                  , If_< IsMatrix<T2>
                       , If_< IsDenseMatrix<T1>
                            , If_< IsDenseMatrix<T2>
                                 , If_< IsRowMajorMatrix<T1>
                                      , If_< IsRowMajorMatrix<T2>
                                           , DMatDMatSubExprTrait<T1,T2>
                                           , DMatTDMatSubExprTrait<T1,T2> >
                                      , If_< IsRowMajorMatrix<T2>
                                           , TDMatDMatSubExprTrait<T1,T2>
                                           , TDMatTDMatSubExprTrait<T1,T2> > >
                                 , If_< IsRowMajorMatrix<T1>
                                      , If_< IsRowMajorMatrix<T2>
                                           , DMatSMatSubExprTrait<T1,T2>
                                           , DMatTSMatSubExprTrait<T1,T2> >
                                      , If_< IsRowMajorMatrix<T2>
                                           , TDMatSMatSubExprTrait<T1,T2>
                                           , TDMatTSMatSubExprTrait<T1,T2> > > >
                            , If_< IsDenseMatrix<T2>
                                 , If_< IsRowMajorMatrix<T1>
                                      , If_< IsRowMajorMatrix<T2>
                                           , SMatDMatSubExprTrait<T1,T2>
                                           , SMatTDMatSubExprTrait<T1,T2> >
                                      , If_< IsRowMajorMatrix<T2>
                                           , TSMatDMatSubExprTrait<T1,T2>
                                           , TSMatTDMatSubExprTrait<T1,T2> > >
                                 , If_< IsRowMajorMatrix<T1>
                                      , If_< IsRowMajorMatrix<T2>
                                           , SMatSMatSubExprTrait<T1,T2>
                                           , SMatTSMatSubExprTrait<T1,T2> >
                                      , If_< IsRowMajorMatrix<T2>
                                           , TSMatSMatSubExprTrait<T1,T2>
                                           , TSMatTSMatSubExprTrait<T1,T2> > > > >
                       , Failure >
                  , If_< IsVector<T1>
                       , If_< IsVector<T2>
                            , If_< IsDenseVector<T1>
                                 , If_< IsDenseVector<T2>
                                      , If_< IsRowVector<T1>
                                           , If_< IsRowVector<T2>
                                                , TDVecTDVecSubExprTrait<T1,T2>
                                                , Failure >
                                           , If_< IsRowVector<T2>
                                                , Failure
                                                , DVecDVecSubExprTrait<T1,T2> > >
                                      , If_< IsRowVector<T1>
                                           , If_< IsRowVector<T2>
                                                , TDVecTSVecSubExprTrait<T1,T2>
                                                , Failure >
                                           , If_< IsRowVector<T2>
                                                , Failure
                                                , DVecSVecSubExprTrait<T1,T2> > > >
                                 , If_< IsDenseVector<T2>
                                      , If_< IsRowVector<T1>
                                           , If_< IsRowVector<T2>
                                                , TSVecTDVecSubExprTrait<T1,T2>
                                                , Failure >
                                           , If_< IsRowVector<T2>
                                                , Failure
                                                , SVecDVecSubExprTrait<T1,T2> > >
                                      , If_< IsRowVector<T1>
                                           , If_< IsRowVector<T2>
                                                , TSVecTSVecSubExprTrait<T1,T2>
                                                , Failure >
                                           , If_< IsRowVector<T2>
                                                , Failure
                                                , SVecSVecSubExprTrait<T1,T2> > > > >
                            , Failure >
                       , If_< IsNumeric<T1>
                            , If_< IsNumeric<T2>
                                 , SubTrait<T1,T2>
                                 , Failure >
                            , Failure > > >;
   /*! \endcond */
   //**********************************************************************************************

 public:
   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   using Type = typename If_< Or< IsConst<T1>, IsVolatile<T1>, IsReference<T1>
                                , IsConst<T2>, IsVolatile<T2>, IsReference<T2> >
                            , SubExprTrait< Decay_<T1>, Decay_<T2> >
                            , Tmp >::Type;
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Auxiliary alias declaration for the SubExprTrait class template.
// \ingroup math_traits
//
// The SubExprTrait_ alias declaration provides a convenient shortcut to access the nested \a Type
// of the SubExprTrait class template. For instance, given the types \a T1 and \a T2 the following
// two type definitions are identical:

   \code
   using Type1 = typename SubExprTrait<T1,T2>::Type;
   using Type2 = SubExprTrait_<T1,T2>;
   \endcode
*/
template< typename T1    // Type of the left-hand side subtraction operand
        , typename T2 >  // Type of the right-hand side subtraction operand
using SubExprTrait_ = typename SubExprTrait<T1,T2>::Type;
//*************************************************************************************************

} // namespace blaze

#endif
