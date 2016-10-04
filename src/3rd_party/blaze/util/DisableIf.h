//=================================================================================================
/*!
//  \file blaze/util/DisableIf.h
//  \brief Header file for the DisableIf class template
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

#ifndef _BLAZE_UTIL_DISABLEIF_H_
#define _BLAZE_UTIL_DISABLEIF_H_


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Substitution Failure Is Not An Error (SFINAE) class.
// \ingroup util
//
// The DisableIfTrue class template is an auxiliary tool for an intentional application of the
// Substitution Failure Is Not An Error (SFINAE) principle. It allows a function template or a
// class template specialization to include or exclude itself from a set of matching functions
// or specializations based on properties of its template arguments. For instance, it can be
// used to restrict the selection of a function template to specific data types. The following
// example illustrates this in more detail.

   \code
   template< typename Type >
   void process( Type t ) { ... }
   \endcode

// Due to the general formulation of this function, it will always be a possible candidate for
// every possible argument. However, with the DisableIfTrue class it is for example possible
// to prohibit built-in, numeric data types as argument types:

   \code
   template< typename Type >
   typename DisableIfTrue< IsNumeric<Type>::value >::Type process( Type t ) { ... }
   \endcode

// In case the given data type is a built-in, numeric data type, the access to the nested type
// definition \a Type of the DisableIfTrue class template will fail. However, due to the SFINAE
// principle, this will only result in a compilation error in case the compiler cannot find
// another valid function.\n
// Note that in this application of the DisableIfTrue template the default for the nested type
// definition \a Type is used, which corresponds to \a void. Via the second template argument
// it is possible to explicitly specify the type of \a Type:

   \code
   // Explicity specifying the default
   typename DisableIfTrue< IsNumeric<Type>::value, void >::Type

   // In case the given data type is not a boolean data type, the nested type definition
   // 'Type' is set to float
   typename DisableIfTrue< IsBoolean<Type>::value, float >::Type
   \endcode

// For more information on the DisableIfTrue/DisableIf functionality, see the Boost library
// documentation of the enable_if family at:
//
//           \a http://www.boost.org/doc/libs/1_60_0/libs/utility/enable_if.html.
*/
template< bool Condition     // Compile time condition
        , typename T=void >  // The type to be instantiated
struct DisableIfTrue
{
   //**********************************************************************************************
   typedef T  Type;  //!< The instantiated type.
   //**********************************************************************************************
};
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief DisableIfTrue specialization for failed constraints.
// \ingroup util
//
// This specialization of the DisableIfTrue template is selected if the first template parameter
// (the compile time condition) evaluates to \a true. This specialization does not contains a
// nested type definition \a Type and therefore always results in a compilation error in case
// \a Type is accessed. However, due to the SFINAE principle the compilation process is not
// necessarily stopped if another, valid instantiation is found by the compiler.
*/
template< typename T >  // The type to be instantiated
struct DisableIfTrue<true,T>
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Auxiliary type for the DisableIfTrue class template.
// \ingroup util
//
// The DisableIfTrue_ alias declaration provides a convenient shortcut to access the nested \a Type
// of the DisableIfTrue class template. For instance, given the type \a T the following two type
// definitions are identical:

   \code
   using Type1 = typename DisableIfTrue< IsBuiltin<T>::value >::Type;
   using Type2 = DisableIfTrue_< IsBuiltin<T>::value >;
   \endcode
*/
template< bool Condition     // Compile time condition
        , typename T=void >  // The type to be instantiated
using DisableIfTrue_ = typename DisableIfTrue<Condition,T>::Type;
//*************************************************************************************************




//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Substitution Failure Is Not An Error (SFINAE) class.
// \ingroup util
//
// The DisableIf class template is an auxiliary tool for an intentional application of the
// Substitution Failure Is Not An Error (SFINAE) principle. It allows a function template
// or a class template specialization to include or exclude itself from a set of matching
// functions or specializations based on properties of its template arguments. For instance,
// it can be used to restrict the selection of a function template to specific data types.
// The following example illustrates this in more detail.

   \code
   template< typename Type >
   void process( Type t ) { ... }
   \endcode

// Due to the general formulation of this function, it will always be a possible candidate
// for every possible argument. However, with the DisableIf class it is for example possible
// to prohibit built-in, numeric data types as argument types:

   \code
   template< typename Type >
   typename DisableIf< IsNumeric<Type> >::Type process( Type t ) { ... }
   \endcode

// In case the given data type is a built-in, numeric data type, the access to the nested
// type definition \a Type of the DisableIf class template will fail. However, due to the
// SFINAE principle, this will only result in a compilation error in case the compiler cannot
// find another valid function.\n
// Note that in this application of the DisableIf template the default for the nested type
// definition \a Type is used, which corresponds to \a void. Via the second template argument
// it is possible to explicitly specify the type of \a Type:

   \code
   // Explicity specifying the default
   typename DisableIf< IsNumeric<Type>, void >::Type

   // In case the given data type is not a boolean data type, the nested type definition
   // 'Type' is set to float
   typename DisableIf< IsBoolean<Type>, float >::Type
   \endcode

// Note that in contrast to the DisableIfTrue template, the DisableIf template expects a
// type as first template argument that has a nested type definition \a value. Therefore
// the DisableIf template is the more convenient choice for all kinds of type traits.
//
// For more information on the DisableIfTrue/DisableIf functionality, see the Boost library
// documentation of the enable_if family at:
//
//           \a http://www.boost.org/doc/libs/1_60_0/libs/utility/enable_if.html.
*/
template< typename Condition  // Compile time condition
        , typename T=void >   // The type to be instantiated
struct DisableIf : public DisableIfTrue<Condition::value,T>
{};
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Auxiliary type for the DisableIf class template.
// \ingroup util
//
// The DisableIf_ alias declaration provides a convenient shortcut to access the nested \a Type
// of the DisableIf class template. For instance, given the type \a T the following two type
// definitions are identical:

   \code
   using Type1 = typename DisableIf< IsBuiltin<T> >::Type;
   using Type2 = DisableIf_< IsBuiltin<T> >;
   \endcode
*/
template< typename Condition  // Compile time condition
        , typename T=void >   // The type to be instantiated
using DisableIf_ = typename DisableIf<Condition,T>::Type;
//*************************************************************************************************

} // namespace blaze

#endif
