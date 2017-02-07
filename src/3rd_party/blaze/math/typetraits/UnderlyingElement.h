//=================================================================================================
/*!
//  \file blaze/math/typetraits/UnderlyingElement.h
//  \brief Header file for the UnderlyingElement type trait
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

#ifndef _BLAZE_MATH_TYPETRAITS_UNDERLYINGELEMENT_H_
#define _BLAZE_MATH_TYPETRAITS_UNDERLYINGELEMENT_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/util/mpl/If.h>
#include <blaze/util/mpl/Or.h>
#include <blaze/util/typetraits/IsBuiltin.h>
#include <blaze/util/typetraits/IsComplex.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Evaluation of the element type of a given data type.
// \ingroup math_type_traits
//
// Via this type trait it is possible to evaluate the element type of a given data type. Examples:

   \code
   typedef double                                    Type1;  // Built-in data type
   typedef complex<float>                            Type2;  // Complex data type
   typedef StaticVector<int,3UL>                     Type3;  // Vector with built-in element type
   typedef CompressedMatrix< DynamicVector<float> >  Type4;  // Matrix with vector element type

   blaze::UnderlyingElement< Type1 >::Type  // corresponds to double
   blaze::UnderlyingElement< Type2 >::Type  // corresponds to float
   blaze::UnderlyingElement< Type3 >::Type  // corresponds to int
   blaze::UnderlyingElement< Type4 >::Type  // corresponds to DynamicVector<float>
   \endcode

// Note that per default UnderlyingElement only supports fundamental/built-in data types, complex,
// and data types with the nested type definition \a ElementType. Support for other data types can
// be added by specializing the UnderlyingElement class template.
*/
template< typename T >
struct UnderlyingElement
{
 private:
   //**struct Builtin******************************************************************************
   /*! \cond BLAZE_INTERNAL */
   template< typename T2 >
   struct Builtin { typedef T2  Type; };
   /*! \endcond */
   //**********************************************************************************************

   //**struct Complex******************************************************************************
   /*! \cond BLAZE_INTERNAL */
   template< typename T2 >
   struct Complex { typedef typename T2::value_type  Type; };
   /*! \endcond */
   //**********************************************************************************************

   //**struct Other********************************************************************************
   /*! \cond BLAZE_INTERNAL */
   template< typename T2 >
   struct Other { typedef typename T2::ElementType  Type; };
   /*! \endcond */
   //**********************************************************************************************

 public:
   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   typedef typename If_< IsBuiltin<T>
                       , Builtin<T>
                       , If_< IsComplex<T>
                            , Complex<T>
                            , Other<T> >
                       >::Type  Type;
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Auxiliary alias declaration for the UnderlyingElement type trait.
// \ingroup type_traits
//
// The UnderlyingElement_ alias declaration provides a convenient shortcut to access the
// nested \a Type of the UnderlyingElement class template. For instance, given the type \a T
// the following two type definitions are identical:

   \code
   using Type1 = typename UnderlyingElement<T>::Type;
   using Type2 = UnderlyingElement_<T>;
   \endcode
*/
template< typename T >
using UnderlyingElement_ = typename UnderlyingElement<T>::Type;
//*************************************************************************************************

} // namespace blaze

#endif
