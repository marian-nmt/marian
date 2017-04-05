//=================================================================================================
/*!
//  \file blaze/math/traits/SubvectorTrait.h
//  \brief Header file for the subvector trait
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

#ifndef _BLAZE_MATH_TRAITS_SUBVECTORTRAIT_H_
#define _BLAZE_MATH_TRAITS_SUBVECTORTRAIT_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

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
/*!\brief Base template for the SubvectorTrait class.
// \ingroup math_traits
//
// \section subvectortrait_general General
//
// The SubvectorTrait class template offers the possibility to select the resulting data type
// when creating a subvector of a dense or sparse vector. SubvectorTrait defines the nested
// type \a Type, which represents the resulting data type of the subvector operation. In case
// the given data type is not a dense or sparse vector type, the resulting data type \a Type
// is set to \a INVALID_TYPE. Note that \a const and \a volatile qualifiers and reference
// modifiers are generally ignored.
//
// Per default, the SubvectorTrait template only supports the following vector types:
//
// <ul>
//    <li>blaze::StaticVector</li>
//    <li>blaze::HybridVector</li>
//    <li>blaze::DynamicVector</li>
//    <li>blaze::CustomVector</li>
//    <li>blaze::CompressedVector</li>
//    <li>blaze::Subvector</li>
//    <li>blaze::Row</li>
//    <li>blaze::Column</li>
// </ul>
//
//
// \section subvectortrait_specializations Creating custom specializations
//
// It is possible to specialize the SubvectorTrait template for additional user-defined vector
// types. The following example shows the according specialization for the DynamicVector class
// template:

   \code
   template< typename T1, bool TF >
   struct SubvectorTrait< DynamicVector<T1,TF> >
   {
      typedef DynamicVector<T1,TF>  Type;
   };
   \endcode

// \n \section subvectortrait_examples Examples
//
// The following example demonstrates the use of the SubvectorTrait template, where depending
// on the given vector type the according result type is selected:

   \code
   using blaze::columnVector;
   using blaze::rowVector;

   // Definition of the result type of a dynamic column vector
   typedef blaze::DynamicVector<int,columnVector>      VectorType1;
   typedef typename SubvectorTrait<VectorType1>::Type  ResultType1;

   // Definition of the result type of the static row vector
   typedef blaze::StaticVector<int,3UL,rowVector>      VectorType2;
   typedef typename SubvectorTrait<VectorType2>::Type  ResultType2;
   \endcode
*/
template< typename VT >  // Type of the vector
struct SubvectorTrait
{
 private:
   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   struct Failure { using Type = INVALID_TYPE; };
   /*! \endcond */
   //**********************************************************************************************

 public:
   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   using Type = typename If_< Or< IsConst<VT>, IsVolatile<VT>, IsReference<VT> >
                            , SubvectorTrait< Decay_<VT> >
                            , Failure >::Type;
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Auxiliary alias declaration for the SubvectorTrait type trait.
// \ingroup math_traits
//
// The SubvectorTrait_ alias declaration provides a convenient shortcut to access the nested
// \a Type of the SubvectorTrait class template. For instance, given the vector type \a VT the
// following two type definitions are identical:

   \code
   using Type1 = typename SubvectorTrait<VT>::Type;
   using Type2 = SubvectorTrait_<VT>;
   \endcode
*/
template< typename VT >  // Type of the vector
using SubvectorTrait_ = typename SubvectorTrait<VT>::Type;
//*************************************************************************************************

} // namespace blaze

#endif
