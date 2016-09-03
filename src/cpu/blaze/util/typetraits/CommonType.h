//=================================================================================================
/*!
//  \file blaze/util/typetraits/CommonType.h
//  \brief Header file for the CommonType type trait
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

#ifndef _BLAZE_UTIL_TYPETRAITS_COMMONTYPE_H_
#define _BLAZE_UTIL_TYPETRAITS_COMMONTYPE_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <type_traits>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Deduction of a type common to several types.
// \ingroup type_traits
//
// The CommonType type trait deduces the result type of a mixed-mode arithmetic expression between
// all types T..., that is the type all T... can be implicitly converted to. Note that cv and
// reference qualifiers are generally ignored.

   \code
   blaze::CommonType<short,int>::Type                         // Results in 'int'
   blaze::CommonType<const double,int&>::Type                 // Results in 'double'
   blaze::CommonType<char&, volatile int, const float>::Type  // Results in 'float'
   \endcode
*/
template< typename... T >
struct CommonType
{
 public:
   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   typedef typename std::common_type<T...>::type  Type;
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Auxiliary alias declaration for the CommonType type trait.
// \ingroup type_traits
//
// The CommonType_ alias declaration provides a convenient shortcut to access the nested \a Type
// of the CommonType class template. For instance, given the types \a T1 and \a T2 the following
// two type definitions are identical:

   \code
   using Type1 = typename CommonType<T1,T2>::Type;
   using Type2 = CommonType_<T1,T2>;
   \endcode
*/
template< typename... T >
using CommonType_ = typename CommonType<T...>::Type;
//*************************************************************************************************

} // namespace blaze

#endif
