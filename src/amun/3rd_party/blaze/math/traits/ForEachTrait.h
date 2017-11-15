//=================================================================================================
/*!
//  \file blaze/math/traits/ForEachTrait.h
//  \brief Header file for the for-each trait
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

#ifndef _BLAZE_MATH_TRAITS_FOREACHTRAIT_H_
#define _BLAZE_MATH_TRAITS_FOREACHTRAIT_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/Aliases.h>
#include <blaze/math/typetraits/IsMatrix.h>
#include <blaze/math/typetraits/IsVector.h>
#include <blaze/math/typetraits/RemoveAdaptor.h>
#include <blaze/util/DisableIf.h>
#include <blaze/util/EnableIf.h>
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
/*!\brief Base template for the ForEachTrait class.
// \ingroup math_traits
//
// The ForEachTrait class template offers the possibility to select the resulting data type of a
// generic \a forEach operation on the given type \a T. Given the type \a T, which must either be
// a vector or matrix type, the nested type \a Type corresponds to the resulting data type of the
// operation. In case the type of \a T doesn't fit or if no \a forEach operation exists for the
// type, the resulting data type \a Type is set to \a INVALID_TYPE. Note that \a const and
// \a volatile qualifiers and reference modifiers are generally ignored.
*/
template< typename T     // Type of the operand
        , typename OP >  // Type of the custom operation
struct ForEachTrait
{
 private:
   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   struct MatrixOrVector
   {
      template< typename U >
      static DisableIf_< Or< IsVector<U>, IsMatrix<U> >, U > test( U&& );

      template< typename U >
      static EnableIf_< Or< IsVector<U>, IsMatrix<U> >, ResultType_<U> > test( U&& );

      using RT = RemoveAdaptor_< ResultType_<T> >;
      using RN = ReturnType_<T>;

      using OT = decltype( std::declval<OP>()( std::declval<RN>() ) );
      using ET = decltype( test( std::declval<OT>() ) );

      using Type = typename RT::template Rebind<ET>::Other;
   };
   /*! \endcond */
   //**********************************************************************************************

   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   struct Failure {
      using Type = INVALID_TYPE;
   };
   /*! \endcond */
   //**********************************************************************************************

   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   using Tmp = If_< Or< IsMatrix<T>, IsVector<T> >
                  , MatrixOrVector
                  , Failure >;
   /*! \endcond */
   //**********************************************************************************************

 public:
   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   using Type = typename If_< Or< IsConst<T>, IsVolatile<T>, IsReference<T> >
                            , ForEachTrait< Decay_<T>, OP >
                            , Tmp >::Type;
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Auxiliary alias declaration for the ForEachTrait class template.
// \ingroup math_traits
//
// The ForEachTrait_ alias declaration provides a convenient shortcut to access the nested \a Type
// of the ForEachTrait class template. For instance, given the type \a T and the custom operation
// type \a OP the following two type definitions are identical:

   \code
   using Type1 = typename ForEachTrait<T,OP>::Type;
   using Type2 = ForEachTrait_<T,OP>;
   \endcode
*/
template< typename T     // Type of the operand
        , typename OP >  // Type of the custom operation
using ForEachTrait_ = typename ForEachTrait<T,OP>::Type;
//*************************************************************************************************

} // namespace blaze

#endif
