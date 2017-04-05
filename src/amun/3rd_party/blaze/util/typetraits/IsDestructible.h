//=================================================================================================
/*!
//  \file blaze/util/typetraits/IsDestructible.h
//  \brief Header file for the IsDestructible type trait
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

#ifndef _BLAZE_UTIL_TYPETRAITS_ISDESTRUCTIBLE_H_
#define _BLAZE_UTIL_TYPETRAITS_ISDESTRUCTIBLE_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <type_traits>
#include <blaze/util/IntegralConstant.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Compile time type check.
// \ingroup type_traits
//
// The IsDestructible type trait tests whether the expression

   \code
   std::declval<U&>().~U();
   \endcode

// is well formed, where \a U represents the type \a T stripped of all extents. If an object of
// type \a T can be destroyed in this way, the \a value member constant is set to \a true, the
// nested type definition \a Type is set to \a TrueType, and the class derives from \a TrueType.
// Otherwise \a value is set to \a false, \a Type is \a FalseType and the class derives from
// \a FalseType.
*/
template< typename T >
struct IsDestructible
   : public BoolConstant< std::is_destructible<T>::value >
{};
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Compile time type check.
// \ingroup type_traits
//
// The IsDestructible type trait tests whether the expression

   \code
   std::declval<U&>().~U();
   \endcode

// is well formed and guaranteed to not throw an exception (i.e. noexcept), where \a U represents
// the type \a T stripped of all extents. If an object of type \a T can be destroyed in this way,
// the \a value member constant is set to \a true, the nested type definition \a Type is set to
// \a TrueType, and the class derives from \a TrueType. Otherwise \a value is set to \a false,
// \a Type is \a FalseType and the class derives from \a FalseType.
*/
template< typename T >
struct IsNothrowDestructible
   : public BoolConstant< std::is_nothrow_destructible<T>::value >
{};
//*************************************************************************************************

} // namespace blaze

#endif
