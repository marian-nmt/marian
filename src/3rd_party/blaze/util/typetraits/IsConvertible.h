//=================================================================================================
/*!
//  \file blaze/util/typetraits/IsConvertible.h
//  \brief Header file for the IsConvertible type trait
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

#ifndef _BLAZE_UTIL_TYPETRAITS_ISCONVERTIBLE_H_
#define _BLAZE_UTIL_TYPETRAITS_ISCONVERTIBLE_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <type_traits>
#include <blaze/util/IntegralConstant.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Compile time pointer relationship constraint.
// \ingroup type_traits
//
// This type traits tests whether the first given template argument can be converted to the
// second template argument via copy construction. If the first argument can be converted
// to the second argument, the \a value member constnt is set to \a true, the nested type
// definition \a type is \a TrueType, and the class derives from \a TrueType. Otherwise
// \a value is set to \a false, \a type is \a FalseType, and the class derives from
// \a FalseType.

   \code
   struct A {};
   struct B : public A {};

   struct C {};
   struct D {
      D( const C& c ) {}
   };

   blaze::IsConvertible<int,unsigned int>::value    // Evaluates to 'true'
   blaze::IsConvertible<float,const double>::value  // Evaluates to 'true'
   blaze::IsConvertible<B,A>::Type                  // Results in TrueType
   blaze::IsConvertible<B*,A*>::Type                // Results in TrueType
   blaze::IsConvertible<C,D>                        // Is derived from TrueType
   blaze::IsConvertible<char*,std::string>          // Is derived from TrueType
   blaze::IsConvertible<std::string,char*>::value   // Evaluates to 'false'
   blaze::IsConvertible<A,B>::Type                  // Results in FalseType
   blaze::IsConvertible<A*,B*>                      // Is derived from FalseType
   \endcode
*/
template< typename From, typename To >
struct IsConvertible : public BoolConstant< std::is_convertible<From,To>::value >
{};
//*************************************************************************************************

} // namespace blaze

#endif
