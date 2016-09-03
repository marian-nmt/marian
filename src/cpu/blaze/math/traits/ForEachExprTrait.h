//=================================================================================================
/*!
//  \file blaze/math/traits/ForEachExprTrait.h
//  \brief Header file for the ForEachExprTrait class template
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

#ifndef _BLAZE_MATH_TRAITS_FOREACHEXPRTRAIT_H_
#define _BLAZE_MATH_TRAITS_FOREACHEXPRTRAIT_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/traits/DMatForEachExprTrait.h>
#include <blaze/math/traits/DVecForEachExprTrait.h>
#include <blaze/math/traits/SMatForEachExprTrait.h>
#include <blaze/math/traits/SVecForEachExprTrait.h>
#include <blaze/math/traits/TDMatForEachExprTrait.h>
#include <blaze/math/traits/TDVecForEachExprTrait.h>
#include <blaze/math/traits/TSMatForEachExprTrait.h>
#include <blaze/math/traits/TSVecForEachExprTrait.h>
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
#include <blaze/util/typetraits/IsReference.h>
#include <blaze/util/typetraits/IsVolatile.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Evaluation of the return type of a \c forEach expression.
// \ingroup math_traits
//
// Via this type trait it is possible to evaluate the return type of a \c forEach expression.
// Given the type \a T, which must either be a scalar, vector, or matrix type, the nested type
// \a Type corresponds to the resulting return type. In case the type of \a T doesn't fit or
// if no \c forEach operation exists for the type, the resulting data type \a Type is set to
// \a INVALID_TYPE.
*/
template< typename T     // Type of the forEach operand
        , typename OP >  // Type of the custom operation
struct ForEachExprTrait
{
 private:
   //**struct Failure******************************************************************************
   /*! \cond BLAZE_INTERNAL */
   struct Failure { using Type = INVALID_TYPE; };
   /*! \endcond */
   //**********************************************************************************************

   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   using Tmp = If_< IsMatrix<T>
                  , If_< IsDenseMatrix<T>
                       , If_< IsRowMajorMatrix<T>
                            , DMatForEachExprTrait<T,OP>
                            , TDMatForEachExprTrait<T,OP> >
                       , If_< IsRowMajorMatrix<T>
                            , SMatForEachExprTrait<T,OP>
                            , TSMatForEachExprTrait<T,OP> > >
                  , If_< IsVector<T>
                       , If_< IsDenseVector<T>
                            , If_< IsRowVector<T>
                                 , TDVecForEachExprTrait<T,OP>
                                 , DVecForEachExprTrait<T,OP> >
                            , If_< IsRowVector<T>
                                 , TSVecForEachExprTrait<T,OP>
                                 , SVecForEachExprTrait<T,OP> > >
                       , Failure > >;
   /*! \endcond */
   //**********************************************************************************************

 public:
   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   using Type = typename If_< Or< IsConst<T>, IsVolatile<T>, IsReference<T> >
                            , ForEachExprTrait< Decay_<T>, OP >
                            , Tmp >::Type;
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Auxiliary alias declaration for the ForEachExprTrait class template.
// \ingroup math_traits
//
// The ForEachExprTrait_ alias declaration provides a convenient shortcut to access the nested
// \a Type of the ForEachExprTrait class template. For instance, given the data type \a T and
// custom operation type \a OP the following two type definitions are identical:

   \code
   using Type1 = typename ForEachExprTrait<T,OP>::Type;
   using Type2 = ForEachExprTrait_<T,OP>;
   \endcode
*/
template< typename T     // Type of the forEach operand
        , typename OP >  // Type of the custom operation
using ForEachExprTrait_ = typename ForEachExprTrait<T,OP>::Type;
//*************************************************************************************************

} // namespace blaze

#endif
