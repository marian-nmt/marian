//=================================================================================================
/*!
//  \file blaze/math/traits/DMatCTransExprTrait.h
//  \brief Header file for the DMatCTransExprTrait class template
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

#ifndef _BLAZE_MATH_TRAITS_DMATCTRANSEXPRTRAIT_H_
#define _BLAZE_MATH_TRAITS_DMATCTRANSEXPRTRAIT_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/expressions/Forward.h>
#include <blaze/math/functors/Forward.h>
#include <blaze/math/traits/DMatForEachExprTrait.h>
#include <blaze/math/traits/DMatTransExprTrait.h>
#include <blaze/math/typetraits/IsDenseMatrix.h>
#include <blaze/math/typetraits/IsRowMajorMatrix.h>
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
/*!\brief Evaluation of the expression type of a dense matrix conjugate transpose operation.
// \ingroup math_traits
//
// Via this type trait it is possible to evaluate the resulting expression type of a dense
// matrix conjugate transpose operation. Given the row-major dense matrix type \a MT, the
// nested type \a Type corresponds to the resulting expression type. In case \a MT is not
// a row-major dense matrix type, the resulting \a Type is set to \a INVALID_TYPE.
*/
template< typename MT >  // Type of the dense matrix
struct DMatCTransExprTrait
{
 private:
   //**struct Failure******************************************************************************
   /*! \cond BLAZE_INTERNAL */
   struct Failure { typedef INVALID_TYPE  Type; };
   /*! \endcond */
   //**********************************************************************************************

   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   typedef If_< And< IsDenseMatrix<MT>, IsRowMajorMatrix<MT> >
              , DMatTransExprTrait< DMatForEachExprTrait_<MT,Conj> >
              , Failure >  Tmp;
   /*! \endcond */
   //**********************************************************************************************

 public:
   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   typedef typename If_< Or< IsConst<MT>, IsVolatile<MT>, IsReference<MT> >
                       , DMatCTransExprTrait< Decay_<MT> >
                       , Tmp >::Type  Type;
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Auxiliary alias declaration for the DMatCTransExprTrait class template.
// \ingroup math_traits
//
// The DMatCTransExprTrait_ alias declaration provides a convenient shortcut to access the nested
// \a Type of the DMatCTransExprTrait class template. For instance, given the row-major dense
// matrix type \a MT the following two type definitions are identical:

   \code
   using Type1 = typename DMatCTransExprTrait<MT>::Type;
   using Type2 = DMatCTransExprTrait_<MT>;
   \endcode
*/
template< typename MT >  // Type of the dense matrix
using DMatCTransExprTrait_ = typename DMatCTransExprTrait<MT>::Type;
//*************************************************************************************************

} // namespace blaze

#endif
