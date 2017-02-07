//=================================================================================================
/*!
//  \file blaze/math/typetraits/UnderlyingBuiltin.h
//  \brief Header file for the UnderlyingBuiltin type trait
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

#ifndef _BLAZE_MATH_TYPETRAITS_UNDERLYINGBUILTIN_H_
#define _BLAZE_MATH_TYPETRAITS_UNDERLYINGBUILTIN_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/util/mpl/If.h>
#include <blaze/util/typetraits/IsBuiltin.h>
#include <blaze/util/typetraits/IsComplex.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Evaluation of the underlying builtin element type of a given data type.
// \ingroup math_type_traits
//
// Via this type trait it is possible to evaluate the underlying fundamental element type at the
// heart of a given data type. Examples:

   \code
   typedef double                                    Type1;  // Built-in data type
   typedef complex<float>                            Type2;  // Complex data type
   typedef StaticVector<int,3UL>                     Type3;  // Vector with built-in element type
   typedef CompressedVector< DynamicVector<float> >  Type4;  // Vector with vector element type

   blaze::UnderlyingBuiltin< Type1 >::Type  // corresponds to double
   blaze::UnderlyingBuiltin< Type2 >::Type  // corresponds to float
   blaze::UnderlyingBuiltin< Type3 >::Type  // corresponds to int
   blaze::UnderlyingBuiltin< Type4 >::Type  // corresponds to float
   \endcode

// Note that per default UnderlyingBuiltin only supports fundamental/built-in data types, complex,
// and data types with the nested type definition \a ElementType. Support for other data types can
// be added by specializing the UnderlyingBuiltin class template.
*/
template< typename T >
struct UnderlyingBuiltin
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
   struct Complex { typedef typename UnderlyingBuiltin<typename T2::value_type>::Type  Type; };
   /*! \endcond */
   //**********************************************************************************************

   //**struct Other********************************************************************************
   /*! \cond BLAZE_INTERNAL */
   template< typename T2 >
   struct Other { typedef typename UnderlyingBuiltin<typename T2::ElementType>::Type  Type; };
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
/*!\brief Auxiliary alias declaration for the UnderlyingBuiltin type trait.
// \ingroup type_traits
//
// The UnderlyingBuiltin_ alias declaration provides a convenient shortcut to access the
// nested \a Type of the UnderlyingBuiltin class template. For instance, given the type \a T
// the following two type definitions are identical:

   \code
   using Type1 = typename UnderlyingBuiltin<T>::Type;
   using Type2 = UnderlyingBuiltin_<T>;
   \endcode
*/
template< typename T >
using UnderlyingBuiltin_ = typename UnderlyingBuiltin<T>::Type;
//*************************************************************************************************

} // namespace blaze

#endif
