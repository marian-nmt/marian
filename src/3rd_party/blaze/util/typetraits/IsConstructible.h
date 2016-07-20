//=================================================================================================
/*!
//  \file blaze/util/typetraits/IsConstructible.h
//  \brief Header file for the IsConstructible type trait
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

#ifndef _BLAZE_UTIL_TYPETRAITS_ISCONSTRUCTIBLE_H_
#define _BLAZE_UTIL_TYPETRAITS_ISCONSTRUCTIBLE_H_


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
// The IsConstructible type trait tests whether the expression

   \code
   T obj( std::declval<Args>()... );
   \endcode

// is well formed. If an object of type \a T can be created in this way, the \a value member
// constant is set to \a true, the nested type definition \a Type is set to \a TrueType, and
// the class derives from \a TrueType. Otherwise \a value is set to \a false, \a Type is
// \a FalseType and the class derives from \a FalseType.
*/
template< typename T, typename... Args >
struct IsConstructible
   : public BoolConstant< std::is_constructible<T,Args...>::value >
{};
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Compile time type check.
// \ingroup type_traits
//
// The IsNothrowConstructible type trait tests whether the expression

   \code
   T obj( std::declval<Args>()... );
   \endcode

// is well formed and guaranteed to not throw an exception (i.e. noexcept). If an object of type
// \a T can be created in this way, the \a value member constant is set to \a true, the nested
// type definition \a Type is set to \a TrueType, and the class derives from \a TrueType. Otherwise
// \a value is set to \a false, \a Type is \a FalseType and the class derives from \a FalseType.
*/
template< typename T, typename... Args >
struct IsNothrowConstructible
   : public BoolConstant< std::is_nothrow_constructible<T,Args...>::value >
{};
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Compile time type check.
// \ingroup type_traits
//
// The IsDefaultConstructible type trait tests whether the expression

   \code
   T obj;
   \endcode

// is well formed. If an object of type \a T can be default constructed, the \a value member
// constant is set to \a true, the nested type definition \a Type is set to \a TrueType, and
// the class derives from \a TrueType. Otherwise \a value is set to \a false, \a Type is
// \a FalseType and the class derives from \a FalseType.
*/
template< typename T >
struct IsDefaultConstructible
   : public BoolConstant< std::is_default_constructible<T>::value >
{};
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Compile time type check.
// \ingroup type_traits
//
// The IsDefaultConstructible type trait tests whether the expression

   \code
   T obj;
   \endcode

// is well formed and guaranteed to not throw an exception (i.e. noexcept). If an object of type
// \a T can be default constructed, the \a value member constant is set to \a true, the nested
// type definition \a Type is set to \a TrueType, and the class derives from \a TrueType. Otherwise
// \a value is set to \a false, \a Type is \a FalseType and the class derives from \a FalseType.
*/
template< typename T >
struct IsNothrowDefaultConstructible
   : public BoolConstant< std::is_nothrow_default_constructible<T>::value >
{};
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Compile time type check.
// \ingroup type_traits
//
// The IsCopyConstructible type trait tests whether the expression

   \code
   T obj( std::declval<T>() );
   \endcode

// is well formed. If an object of type \a T can be copy constructed, the \a value member
// constant is set to \a true, the nested type definition \a Type is set to \a TrueType, and
// the class derives from \a TrueType. Otherwise \a value is set to \a false, \a Type is
// \a FalseType and the class derives from \a FalseType.
*/
template< typename T >
struct IsCopyConstructible
   : public BoolConstant< std::is_copy_constructible<T>::value >
{};
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Compile time type check.
// \ingroup type_traits
//
// The IsNothrowCopyConstructible type trait tests whether the expression

   \code
   T obj( std::declval<T>() );
   \endcode

// is well formed and guaranteed to not throw an exception (i.e. noexcept). If an object of type
// \a T can be copy constructed, the \a value member constant is set to \a true, the nested type
// definition \a Type is set to \a TrueType, and the class derives from \a TrueType. Otherwise
// \a value is set to \a false, \a Type is \a FalseType and the class derives from \a FalseType.
*/
template< typename T >
struct IsNothrowCopyConstructible
   : public BoolConstant< std::is_copy_constructible<T>::value >
{};
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Compile time type check.
// \ingroup type_traits
//
// The IsMoveConstructible type trait tests whether the expression

   \code
   T obj( std::move( std::declval<T>() ) );
   \endcode

// is well formed. If an object of type \a T can be move constructed, the \a value member
// constant is set to \a true, the nested type definition \a Type is set to \a TrueType, and
// the class derives from \a TrueType. Otherwise \a value is set to \a false, \a Type is
// \a FalseType and the class derives from \a FalseType.
*/
template< typename T >
struct IsMoveConstructible
   : public BoolConstant< std::is_move_constructible<T>::value >
{};
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Compile time type check.
// \ingroup type_traits
//
// The IsNothrowMoveConstructible type trait tests whether the expression

   \code
   T obj( std::move( std::declval<T>() ) );
   \endcode

// is well formed and guaranteed to not throw an exception (i.e. noexcept). If an object of type
// \a T can be move constructed, the \a value member constant is set to \a true, the nested type
// definition \a Type is set to \a TrueType, and the class derives from \a TrueType. Otherwise
// \a value is set to \a false, \a Type is \a FalseType and the class derives from \a FalseType.
*/
template< typename T >
struct IsNothrowMoveConstructible
   : public BoolConstant< std::is_nothrow_move_constructible<T>::value >
{};
//*************************************************************************************************

} // namespace blaze

#endif
