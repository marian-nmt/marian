//=================================================================================================
/*!
//  \file blaze/math/traits/InvExprTrait.h
//  \brief Header file for the InvExprTrait class template
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

#ifndef _BLAZE_MATH_TRAITS_INVEXPRTRAIT_H_
#define _BLAZE_MATH_TRAITS_INVEXPRTRAIT_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/traits/DMatInvExprTrait.h>
#include <blaze/math/traits/TDMatInvExprTrait.h>
#include <blaze/math/typetraits/IsBLASCompatible.h>
#include <blaze/math/typetraits/IsDenseMatrix.h>
#include <blaze/math/typetraits/IsRowMajorMatrix.h>
#include <blaze/math/typetraits/UnderlyingElement.h>
#include <blaze/util/InvalidType.h>
#include <blaze/util/mpl/And.h>
#include <blaze/util/mpl/If.h>
#include <blaze/util/mpl/Or.h>
#include <blaze/util/typetraits/Decay.h>
#include <blaze/util/typetraits/IsComplex.h>
#include <blaze/util/typetraits/IsConst.h>
#include <blaze/util/typetraits/IsFloatingPoint.h>
#include <blaze/util/typetraits/IsReference.h>
#include <blaze/util/typetraits/IsVolatile.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Evaluation of the return type of an inversion expression.
// \ingroup math_traits
//
// Via this type trait it is possible to evaluate the return type of an inversion expression.
// Given the type \a T, which must either a (complex) floating point type or a dense matrix type,
// the nested type \a Type corresponds to the resulting return type. In case the type of \a T
// doesn't fit or if no inversion operation exists for the type, the resulting data type \a Type
// is set to \a INVALID_TYPE.
*/
template< typename T >  // Type of the inversion operand
struct InvExprTrait
{
 private:
   //**struct Scalar*******************************************************************************
   /*! \cond BLAZE_INTERNAL */
   struct Scalar { using Type = T; };
   /*! \endcond */
   //**********************************************************************************************

   //**struct Failure******************************************************************************
   /*! \cond BLAZE_INTERNAL */
   struct Failure { using Type = INVALID_TYPE; };
   /*! \endcond */
   //**********************************************************************************************

   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   using Tmp = If_< And< IsDenseMatrix<T>, IsBLASCompatible< UnderlyingElement_<T> > >
                  , If_< IsRowMajorMatrix<T>
                       , DMatInvExprTrait<T>
                       , TDMatInvExprTrait<T> >
                  , If_< Or< IsFloatingPoint<T>
                           , And< IsComplex<T>, IsFloatingPoint< UnderlyingElement_<T> > > >
                       , Scalar
                       , Failure > >;
   /*! \endcond */
   //**********************************************************************************************

 public:
   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   using Type = typename If_< Or< IsConst<T>, IsVolatile<T>, IsReference<T> >
                            , InvExprTrait< Decay_<T> >
                            , Tmp >::Type;
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Auxiliary alias declaration for the InvExprTrait class template.
// \ingroup math_traits
//
// The InvExprTrait_ alias declaration provides a convenient shortcut to access the nested \a Type
// of the InvExprTrait class template. For instance, given the type \a T the following two type
// definitions are identical:

   \code
   using Type1 = typename InvExprTrait<T>::Type;
   using Type2 = InvExprTrait_<T>;
   \endcode
*/
template< typename T >  // Type of the inversion operand
using InvExprTrait_ = typename InvExprTrait<T>::Type;
//*************************************************************************************************

} // namespace blaze

#endif
