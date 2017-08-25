//=================================================================================================
/*!
//  \file blaze/math/traits/SVecSVecAddExprTrait.h
//  \brief Header file for the SVecSVecAddExprTrait class template
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

#ifndef _BLAZE_MATH_TRAITS_SVECSVECADDEXPRTRAIT_H_
#define _BLAZE_MATH_TRAITS_SVECSVECADDEXPRTRAIT_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/expressions/Forward.h>
#include <blaze/math/typetraits/IsColumnVector.h>
#include <blaze/math/typetraits/IsSparseVector.h>
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
/*!\brief Evaluation of the expression type of a sparse vector/sparse vector addition.
// \ingroup math_traits
//
// Via this type trait it is possible to evaluate the resulting expression type of a sparse
// vector/sparse vector addition. Given the two non-transpose sparse vector types \a VT1 and
// \a VT2, the nested type \a Type corresponds to the resulting expression type. In case
// either \a VT1 or \a VT2 is not a non-transpose sparse vector type, the resulting \a Type
// is set to \a INVALID_TYPE.
*/
template< typename VT1    // Type of the left-hand side non-transpose sparse vector
        , typename VT2 >  // Type of the right-hand side non-transpose sparse vector
struct SVecSVecAddExprTrait
{
 private:
   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   using Tmp = If< And< IsSparseVector<VT1>, IsColumnVector<VT1>
                      , IsSparseVector<VT2>, IsColumnVector<VT2> >
                 , SVecSVecAddExpr<VT1,VT2,false>
                 , INVALID_TYPE >;
   /*! \endcond */
   //**********************************************************************************************

 public:
   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   using Type = typename If_< Or< IsConst<VT1>, IsVolatile<VT1>, IsReference<VT1>
                                , IsConst<VT2>, IsVolatile<VT2>, IsReference<VT2> >
                            , SVecSVecAddExprTrait< Decay_<VT1>, Decay_<VT2> >
                            , Tmp >::Type;
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Auxiliary alias declaration for the SVecSVecAddExprTrait class template.
// \ingroup math_traits
//
// The SVecSVecAddExprTrait_ alias declaration provides a convenient shortcut to access
// the nested \a Type of the SVecSVecAddExprTrait class template. For instance, given the
// non-transpose sparse vector types \a VT1 and \a VT2 the following two type definitions
// are identical:

   \code
   using Type1 = typename SVecSVecAddExprTrait<VT1,VT2>::Type;
   using Type2 = SVecSVecAddExprTrait_<VT1,VT2>;
   \endcode
*/
template< typename VT1    // Type of the left-hand side non-transpose sparse vector
        , typename VT2 >  // Type of the right-hand side non-transpose sparse vector
using SVecSVecAddExprTrait_ = typename SVecSVecAddExprTrait<VT1,VT2>::Type;
//*************************************************************************************************

} // namespace blaze

#endif
